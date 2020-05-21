import torch

### TODO: Add model
from nerf import cumprod_exclusive
import torchsearchsorted

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
        one_e_10.requires_grad = False
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
        point_intervals.requires_grad = False
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


class SamplePDF(torch.nn.Module):
    def __init__(self, num_samples):
        super(SamplePDF, self).__init__()
        self.num_samples = num_samples
        u = torch.linspace(0.0, 1.0, steps=self.num_samples)
        u.requires_grad = False
        self.register_buffer("u", u)

    def forward(self, point_interval, weights, perturb):
        points_on_rays_mid = 0.5 * (point_interval[..., 1:] + point_interval[..., :-1])
        interval_samples = self.sample_pdf(
            points_on_rays_mid,
            weights[..., 1:-1],
            self.u,
            det=(perturb == 0.0)).detach()

        point_interval, _ = torch.sort(
            torch.cat((point_interval, interval_samples), dim=-1), dim=-1
        )
        return point_interval

    def sample_pdf(self, bins, weights, u, det=False):
        r"""sample_pdf function from another concurrent pytorch implementation
        by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
        """

        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat(
            [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
        )  # (batchsize, len(bins))

        # Take uniform samples
        if det:
            u = u.expand(list(cdf.shape[:-1]) + [self.num_samples])
        else:
            u = torch.rand(
                list(cdf.shape[:-1]) + [self.num_samples],
                dtype=weights.dtype,
                device=weights.device,
            )

        # Invert CDF
        u = u.contiguous()
        cdf = cdf.contiguous()
        inds = torchsearchsorted.searchsorted(cdf, u, side="right")
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack((below, above), dim=-1)  # (batchsize, num_samples, 2)

        matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples


