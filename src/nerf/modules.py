import torch

### TODO: Add model
from nerf import cumprod_exclusive


class PositionalEncoding(torch.nn.Module):
    """Apply positional encoding to the input.
    """
    def __init__(
        self,
        num_encoding_functions: int = 6,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0,
                num_encoding_functions - 1,
                num_encoding_functions
            )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0,
                2.0 ** (num_encoding_functions - 1),
                num_encoding_functions
            )
        self.register_buffer('frequency_bands', frequency_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input = [x] if self.include_input else []
        xshape = list(x.shape)
        x = x[..., None].expand(*xshape, self.num_encoding_functions)
        x = self.frequency_bands*x
        x = x.view(*xshape[:-1], -1)
        encoding = torch.cat(input + [torch.sin(x), torch.cos(x)], dim=-1)
        return encoding


class VolumeRenderer(torch.nn.Module):
    def __init__(self, train_radiance_field_noise_std=0.0,
            train_white_background=False, val_radiance_field_noise_std=0.0,
            val_white_background=False):
        super(VolumeRenderer, self).__init__()
        self.train_radiance_field_noise_std = train_radiance_field_noise_std
        self.train_white_background = train_white_background
        self.val_radiance_field_noise_std = val_radiance_field_noise_std
        self.val_white_background = val_white_background
        one_e_10 = torch.tensor([1e10])
        self.register_buffer('one_e_10', one_e_10)


    def forward( self,
            radiance_field,
            depth_values,
            ray_directions):
        if self.training:
            radiance_field_noise_std = self.train_radiance_field_noise_std
            white_background = self.train_white_background
        else:
            radiance_field_noise_std = self.val_radiance_field_noise_std
            white_background = self.val_white_background


        # TESTED
        dists = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                self.one_e_10.expand(depth_values[..., :1].shape),
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
            # noise = noise.to(radiance_field)
        sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
        alpha = 1.0 - torch.exp(-sigma_a * dists)
        weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

        rgb_map = weights[..., None] * rgb
        rgb_map = rgb_map.sum(dim=-2)
        depth_map = weights * depth_values
        depth_map = depth_map.sum(dim=-1)
        # depth_map = (weights * depth_values).sum(dim=-1)
        acc_map = weights.sum(dim=-1)
        disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map),
                                   depth_map / acc_map)

        if white_background:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, disp_map, acc_map, weights, depth_map

class RaySampleInterval(torch.nn.Module):
    def __init__(self, point_amount=64, lindisp=True, perturb=False):
        super(RaySampleInterval, self).__init__()
        self.lindisp = lindisp
        self.perturb = perturb

        point_intervals = torch.linspace(
            0.0, 1.0, point_amount
        )[None, :]

        self.register_buffer('point_intervals', point_intervals)

    def forward(self, bounds):
        near, far = bounds[..., 0, None], bounds[..., 1, None]

        # Sample in disparity space, as opposed to in depth space. Sampling in disparity is
        # nonlinear when viewed as depth sampling! (The closer to the camera the more samples)
        if not self.lindisp:
            point_intervals = near * (1.0 - self.point_intervals) + far * self.point_intervals
        else:
            point_intervals = 1.0 / (
                    1.0 / near * (1.0 - self.point_intervals) + 1.0 / far * self.point_intervals
            )

        if self.perturb:
            # Get intervals between samples.
            mids = 0.5 * (point_intervals[..., 1:] + point_intervals[..., :-1])
            upper = torch.cat((mids, point_intervals[..., -1:]), dim=-1)
            lower = torch.cat((point_intervals[..., :1], mids), dim=-1)
            # Stratified samples in those intervals.
            t_rand = torch.rand(
                point_intervals.shape, dtype=self.point_intervals.dtype, device=self.point_intervals.device
            )
            point_intervals = lower + (upper - lower) * t_rand
        return point_intervals

