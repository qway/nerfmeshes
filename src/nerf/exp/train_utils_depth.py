import torch

from ..nerf_helpers import get_minibatches, ndc_rays
from .volume_rendering_utils_depth import volume_render_radiance_field



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
        z_vals_t, indices, intersections, ray_mask = tree.batch_ray_voxel_intersect(tree.voxels, ro, rd, samples_count = total, verbose = mode != "train")

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



