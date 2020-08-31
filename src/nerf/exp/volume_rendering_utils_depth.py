import torch
from ..nerf_helpers import cumprod_exclusive


def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std = 0.0,
):
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )

    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim = -1,
    ) * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb, psdf = torch.sigmoid(radiance_field[..., :3]), radiance_field[..., 3]

    if radiance_field_noise_std > 0.0:
        psdf += torch.randn_like(psdf) * radiance_field_noise_std

    psdf_scale = torch.relu(psdf)
    psdf_scale = 1.0 - torch.exp(-psdf_scale * dists)

    psdf_resid = 1.0 - psdf_scale + 1e-10
    attenuation = cumprod_exclusive(psdf_resid)
    weights = psdf_scale * attenuation

    mask_weights = (attenuation > 1e-3).float()

    rgb_map = (weights[..., None] * rgb).sum(dim = -2)
    depth_map = (weights * depth_values).sum(dim = -1)

    return rgb_map, depth_map, weights, mask_weights
