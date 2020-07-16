import torch

from ..nerf_helpers import get_minibatches, ndc_rays
from ..nerf_helpers import sample_pdf_2 as sample_pdf
from .volume_rendering_utils_depth import volume_render_radiance_field
from torch.distributions.multivariate_normal import MultivariateNormal


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


def get_ln_samples(near, far, num_rays, options, mode, type, device):
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype = type,
        device = device,
    )

    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    return z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])


def get_info_samples(depth_values, num_rays, options, mode, device, variance = 0.0001, use_stacking = True, use_truncated = True):
    total = getattr(options.nerf, mode).num_coarse

    z_vals = MultivariateNormal(torch.zeros((total - 1,)).to(device), torch.eye(total - 1).to(device) * variance).sample((num_rays,))
    if use_stacking and use_truncated:
        # stack 2-halfs, stronger sampling at peak
        shift = -torch.tensor(variance).sqrt()
        mask = z_vals < shift

        z_vals[mask] = abs(z_vals[mask]) + shift

    if use_truncated:
        z_vals = depth_values[:, None] - z_vals

    z_vals = torch.cat((z_vals, depth_values[:, None]), dim = -1)

    mask_zero = depth_values == 0
    mask_size = mask_zero.sum()
    mask_samples = torch.rand(mask_size, total, device = device) * 4.0 + 2.0
    z_vals[mask_zero] = mask_samples

    return torch.sort(z_vals.to(device), dim = -1).values


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    it = -1,
):
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    depth_values = None
    if ray_batch.shape[-1] == 9 or ray_batch.shape[-1] == 12:
        depth_values = ray_batch[..., 8]

    if depth_values is not None and mode is "train":
        z_vals_1 = get_ln_samples(near, far, num_rays, options, mode, ro.dtype, ro.device)
        z_vals_2 = get_info_samples(depth_values, num_rays, options, mode, ro.device)
        z_vals = torch.cat((z_vals_1, z_vals_2), -1)
        z_vals = torch.sort(z_vals.to(ro.device), dim = 1).values
    else:
        if depth_values is None or mode is not "train":
            z_vals = get_ln_samples(near, far, num_rays, options, mode, ro.dtype, ro.device)
        else:
            z_vals = get_info_samples(depth_values, num_rays, options, mode, ro.device)

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand

    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

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
        disp_coarse,
        acc_coarse,
        weights,
        depth_coarse,
    ) = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        depth_data = None
    )

    # if (it + 1) % 2000 == 0:
    #     mask_zero = depth_values == 0
    #     print('---------------')
    #     print(weights[mask_zero, ...].shape)
    #     print(weights[mask_zero, ...].sum())
    #     print(weights[mask_zero, ...][0])
    #     print(depth_values)
    #     print(depth_coarse[~mask_zero])
    #     print(depth_coarse[mask_zero])
    #
    #     exit(-1)

    # Change volume_render_radiance_field_depth for volume_render_radiance_field
    weights_1 = weights

    rgb_fine, disp_fine, acc_fine, depth_fine = None, None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # t_vals = torch.linspace(
        #     0.0,
        #     1.0,
        #     getattr(options.nerf, mode).num_coarse,
        #     dtype = ro.type,
        #     device = ro.device,
        # )
        #
        # z_samples = (t_vals - 0.5) / 10 + depth_coarse[:, None]
        #
        # z_samples_all = z_samples

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()
        z_samples_all = torch.cat((z_vals, z_samples), dim=-1)
        z_vals, _ = torch.sort(z_samples_all, dim=-1)

        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
        )

        rgb_fine, disp_fine, acc_fine, weights_2, depth_fine = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(
                options.nerf, mode
            ).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
        )


    return rgb_coarse, disp_coarse, acc_coarse, depth_coarse, rgb_fine, disp_fine, acc_fine, depth_fine
    # return rgb_coarse, disp_coarse, acc_coarse, depth_coarse, rgb_fine, disp_fine, acc_fine, depth_fine, z_vals, weights, radiance_field
    # data1, data2 = z_vals, torch.cat((weights_1, weights_2), dim=-1)
    # data1 = data1.detach().cpu()
    # data2 = data2.detach().cpu()
    # return rgb_coarse, disp_coarse, acc_coarse, depth_coarse, rgb_fine, disp_fine, acc_fine, depth_fine, data1, data2


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
):
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
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]

    if model_fine:
        restore_shapes += restore_shapes

    # restore_shapes += [
    #     (800, 800, 64),
    #     (800, 800, 64),
    #     (800, 800, 64, 4),
    # ]

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
    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            options,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            it = it,
        )
        for index, batch in enumerate(batches)
    ]

    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]

    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None, None])

    return tuple(synthesized_images)
