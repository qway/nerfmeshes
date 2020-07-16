import torch

from ..nerf_helpers import cumprod_exclusive
from torch.autograd import Variable

def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    depth_data = None
):
    # TESTED
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )

    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :3])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )

    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim = -2)

    # norm = torch.distributions.normal.Normal(torch.tensor([2.4]), torch.tensor([0.01]))
    # cdf = norm.cdf(torch.linspace(2, 6, weights.shape[-1])).to(weights.device)
    # cdf = Variable(cdf, requires_grad = True)
    #
    # weights1 = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    # weights = weights * cdf

    # depth_map = (weights1 * depth_values).sum(dim = -1)

    weights1 = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim = -1)
    depth_map = (weights1 * depth_values).sum(dim = -1)
    # if depth_data is not None:
    #     print('-----------------')
    #     mask_zero = depth_data != 0
    #     print(weights[mask_zero].shape)
    #     index = (depth_values[0] < depth_data[mask_zero][0]).sum()
    #     print(depth_values[0][index-10:index+10])
    #     print(weights[mask_zero][0][index-10:index+10])
    #     print(depth_map[mask_zero][0])

    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map









import torch

from ..nerf_helpers import cumprod_exclusive
from torch.autograd import Variable

def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    depth_data = None
):
    # TESTED
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )

    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :3])

    sigma_a = torch.nn.functional.relu(radiance_field[..., 3])
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim = -2)

    # norm = torch.distributions.normal.Normal(torch.tensor([2.4]), torch.tensor([0.01]))
    # cdf = norm.cdf(torch.linspace(2, 6, weights.shape[-1])).to(weights.device)
    # cdf = Variable(cdf, requires_grad = True)
    #
    # weights1 = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)
    # weights = weights * cdf

    # depth_map = (weights1 * depth_values).sum(dim = -1)

    weights1 = alpha * torch.cumprod(1.0 - alpha + 1e-10, dim = -1)
    depth_map = (weights1 * depth_values).sum(dim = -1)
    # if depth_data is not None:
    #     print('-----------------')
    #     mask_zero = depth_data != 0
    #     print(weights[mask_zero].shape)
    #     index = (depth_values[0] < depth_data[mask_zero][0]).sum()
    #     print(depth_values[0][index-10:index+10])
    #     print(weights[mask_zero][0][index-10:index+10])
    #     print(depth_map[mask_zero][0])

    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    return rgb_map, disp_map, acc_map, weights, depth_map
