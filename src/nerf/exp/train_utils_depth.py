import torch

from ..nerf_helpers import get_minibatches, ndc_rays
from .volume_rendering_utils_depth import volume_render_radiance_field


def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn):
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    preds = [ network_fn(batch) for batch in batches ]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field


def get_ln_samples(near, far, num_rays, options, mode, type, device, total):
    t_vals = torch.linspace(0.0, 1.0, total, dtype = type, device = device)

    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    z_vals = z_vals.expand([num_rays, total])

    return z_vals


def get_random_samples(near, far, num_rays, options, mode, type, device, total):
    vv1, vv2 = near[0], far[0]

    r_vals = torch.rand((num_rays, total), dtype = type, device = device).sort(-1).values
    z_vals = vv1 + r_vals * (vv2 - vv1)

    return z_vals


def get_info_samples(depth_values, near, far, num_rays, options, mode, dtype, device, total, threshold = 0.5):
    mask_zero = depth_values == options.dataset.empty

    far_t = far.clone()
    far_t[~mask_zero] = depth_values[~mask_zero] + threshold

    z_vals = get_ln_samples(near, far_t, num_rays, options, mode, dtype, device, total)

    mask_samples = torch.rand((mask_zero.sum(), total), device = device) * options.dataset.far + options.dataset.near
    z_vals[mask_zero.squeeze(-1)] = mask_samples

    return torch.sort(z_vals.to(device), dim = -1).values


def get_ln_samples_sm(depth_values, near, far, num_rays, options, mode, dtype, device, total, fc1 = 10., fc2 = 2., off = 0.5):
    mask_space = (depth_values != options.dataset.empty).squeeze(-1)

    t_vals = torch.linspace(0.0, 1.0, total, dtype = dtype, device = device) - off
    t_vals = ((torch.rand_like(t_vals) - 0.5) / fc1 + t_vals).sort(-1).values / fc2
    z_vals_space = t_vals.expand([ num_rays, total ])
    z_vals_space = z_vals_space + depth_values

    z_vals = get_ln_samples(near, far, num_rays, options, mode, dtype, device, total)
    z_vals[mask_space.squeeze(-1)] = z_vals_space[mask_space]

    return z_vals.to(device)


def get_ln_samples_prox(depth_values, near, far, num_rays, options, mode, dtype, device, total, off = 0.4):
    mask_space = (depth_values != options.dataset.empty).squeeze(-1)

    depth_values_off = depth_values - off
    t_vals = torch.linspace(0.0, 1.0, total, dtype = dtype, device = device)
    z_vals_space = t_vals.expand([ num_rays, total ])
    z_vals_space = z_vals_space * (far - depth_values_off) + depth_values_off

    z_vals = get_ln_samples(near, far, num_rays, options, mode, dtype, device, total)
    z_vals[mask_space.squeeze(-1)] = z_vals_space[mask_space]

    return z_vals.to(device)


def sample_sm(ro, rd, near, far, num_rays, depth_values, options, mode, z_vals_ex = None):
    total = getattr(options.nerf, mode).num_coarse
    if depth_values is not None and mode is "train":
        z_vals = get_ln_samples(near, far, num_rays, options, mode, ro.dtype, ro.device, total)
        # z_vals = get_random_samples(near, far, num_rays, options, mode, ro.dtype, ro.device, total)
        # z_vals = get_ln_samples_prox(depth_values, near, far, num_rays, options, mode, ro.dtype, ro.device, total)
        # z_vals = get_ln_samples_sm(depth_values, near, far, num_rays, options, mode, ro.dtype, ro.device, total)
        # z_vals = get_info_samples(depth_values, near, far, num_rays, options, mode, ro.dtype, ro.device, total)
    else:
        if depth_values is None or mode is not "train":
            z_vals = get_ln_samples(near, far, num_rays, options, mode, ro.dtype, ro.device, total)
        else:
            z_vals = get_info_samples(depth_values, near, far, num_rays, options, mode, ro.dtype, ro.device, total)

    if z_vals_ex is not None:
        z_vals = torch.cat((z_vals, z_vals_ex), dim = -1).sort(-1).values

    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    return pts, z_vals


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    it = -1,
    tree=None,
):
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    depth_values = None
    if ray_batch.shape[-1] == 9 or ray_batch.shape[-1] == 12:
        depth_values = ray_batch[..., 8:9]

    # pts -> (num_rays, N_samples, 3)
    total = getattr(options.nerf, mode).num_coarse
    pts, z_vals = sample_sm(ro, rd, near, far, num_rays, depth_values, options, mode)
    if tree is not None:
        z_vals_t, indices, intersections, ray_mask = tree.batch_ray_voxel_intersect(tree.voxels, ro, rd, samples = total, verbose = mode != "train")
        
        pts[ray_mask] = ro[ray_mask][..., None, :] + rd[ray_mask][..., None, :] * z_vals_t[..., None]
        z_vals[ray_mask] = z_vals_t

    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
    )

    (
        rgb_coarse,
        depth_coarse,
        weights,
        weights_mask
    ) = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
    )

    if tree is not None and mode is "train":
        tree.ray_batch_integration(it, indices, weights[ray_mask].detach(), weights_mask[ray_mask].detach())

    rgb_fine, depth_fine, depth_std_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        pass

    return rgb_coarse, depth_coarse, rgb_fine, depth_fine, weights, z_vals


def run_one_iter_of_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    depth_data=None,
    it = -1,
    tree=None,
    device='cpu'
):
    if mode == "train":
        ray_origins = ray_origins.to(device)
        ray_directions = ray_directions.to(device)

    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))

    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
    ]

    restore_shapes += restore_shapes
    restore_shapes += [
        torch.Size([ 800, 800, 64 ]),
        torch.Size([ 800, 800, 64 ])
    ]

    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))

    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)

    if depth_data is not None:
        rays = torch.cat((rays, depth_data.unsqueeze(1)), dim=-1)

    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    pred = []
    for index, batch in enumerate(batches):
        output_tp = predict_and_render_radiance(
            batch.to(device),
            model_coarse,
            model_fine,
            options,
            mode = mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            it = it,
            tree=tree
        )

        if mode == "validation":
            output_tp = tuple([ val.cpu() if val is not None else None for val in output_tp ])

        pred.append(output_tp)

    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]

    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else (None)
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

    return tuple(synthesized_images)

