import math
import torch

from nerf import cumprod_exclusive
from dataclasses import dataclass


class PositionalEncoding(torch.nn.Module):
    """Apply positional encoding to the input.
    """

    def __init__(self, num_encoding_functions: int = 6, include_input: bool = True, log_sampling: bool = True):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        if log_sampling:
            frequency_bands = 2.0 ** torch.linspace(
                0.0, num_encoding_functions - 1, num_encoding_functions
            )
        else:
            frequency_bands = torch.linspace(
                2.0 ** 0.0, 2.0 ** (num_encoding_functions - 1), num_encoding_functions
            )
        self.register_buffer("frequency_bands", frequency_bands)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input = [x] if self.include_input else []
        xshape = list(x.shape)
        x = x[..., None].expand(*xshape, self.num_encoding_functions)
        x = self.frequency_bands * x
        x = x.view(*xshape[:-1], -1)
        encoding = torch.cat(input + [torch.sin(x), torch.cos(x)], dim = -1)
        return encoding

    def output_size(self):
        return 2 * 3 * self.num_encoding_functions + (3 if self.include_input else 0)


@dataclass
class OutputBundle:
    rgb_map: torch.Tensor = None
    depth_map: torch.Tensor = None
    weights: torch.Tensor = None
    mask_weights: torch.Tensor = None
    acc_map: torch.Tensor = None
    disp_map: torch.Tensor = None


class VolumeRenderer(torch.nn.Module):
    def __init__(
            self,
            train_radiance_field_noise_std = 0.0,
            val_radiance_field_noise_std = 0.0,
            white_background = False,
            attenuation_threshold=1e-3
    ):
        super(VolumeRenderer, self).__init__()
        self.train_radiance_field_noise_std = train_radiance_field_noise_std
        self.val_radiance_field_noise_std = val_radiance_field_noise_std
        self.attenuation_threshold = attenuation_threshold
        self.white_background = white_background
        one_e_10 = torch.tensor([1e10])
        one_e_10.requires_grad = False
        self.register_buffer("one_e_10", one_e_10)

    def forward(self, radiance_field, depth_values, ray_directions):
        if self.training:
            radiance_field_noise_std = self.train_radiance_field_noise_std
        else:
            radiance_field_noise_std = self.val_radiance_field_noise_std

        dists = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                self.one_e_10.expand(depth_values[..., :1].shape),
            ),
            dim = -1,
        ) * ray_directions[..., None, :].norm(p = 2, dim = -1)

        rgb = radiance_field[..., :3]
        noise = 0.0
        if radiance_field_noise_std > 0.0:
            noise = (
                    torch.randn(
                        radiance_field[..., 3].shape,
                        dtype = radiance_field.dtype,
                        device = radiance_field.device,
                    )
                    * radiance_field_noise_std
            )

        sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
        alpha = 1.0 - torch.exp(-sigma_a * dists)

        weight_attenuation = cumprod_exclusive(1.0 - alpha + 1e-10)
        mask_weights = (weight_attenuation > self.attenuation_threshold).float()
        weights = alpha * weight_attenuation

        rgb_map = weights[..., None] * rgb
        rgb_map = rgb_map.sum(dim = -2)

        acc_map = weights.sum(dim = -1)

        depth_map = (weights * depth_values).sum(dim = -1)
        disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)
        disp_map[torch.isnan(disp_map)] = 0
        if not self.training:
            depth_map[acc_map < 1.0] = 0

        if self.white_background:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return OutputBundle(
            rgb_map = rgb_map,
            depth_map = depth_map,
            weights = weights,
            mask_weights = mask_weights,
            acc_map = acc_map,
            disp_map = disp_map
        )


class DensityExtractor(torch.nn.Module):
    def __init__(self):
        super(DensityExtractor, self).__init__()
        one_e_10 = torch.tensor([1e10])
        one_e_10.requires_grad = False
        self.register_buffer("one_e_10", one_e_10)

    def forward(self, radiance_field, depth_values, ray_directions):
        dists = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                self.one_e_10.expand(depth_values[..., :1].shape),
            ),
            dim = -1,
        )
        dists = dists * ray_directions[..., None, :].norm(p = 2, dim = -1)

        sigma_a = torch.nn.functional.relu(radiance_field[..., 3])
        alpha = 1.0 - torch.exp(-sigma_a * dists)
        weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

        return weights


class RaySampleInterval(torch.nn.Module):
    def __init__(self, count):
        super(RaySampleInterval, self).__init__()
        self.count = count

        # Ray sample count
        point_intervals = torch.linspace(0.0, 1.0, self.count, requires_grad = False)[None, :]
        self.register_buffer("point_intervals", point_intervals, persistent = False)

    def forward(self, cfg, ray_count, near, far):
        if len(near.shape) > 0 and near.shape[0] == ray_count:
            near, far = near[:, None], far[:, None]

        # Sample in disparity space, as opposed to in depth space. Sampling in disparity is
        # nonlinear when viewed as depth sampling! (The closer to the camera the more samples)
        if not cfg.lindisp:
            point_intervals = near * (1.0 - self.point_intervals) + far * self.point_intervals
        else:
            point_intervals = 1.0 / (1.0 / near * (1.0 - self.point_intervals) + 1.0 / far * self.point_intervals)

        if len(near.shape) == 0 or near.shape[0] != ray_count:
            point_intervals = point_intervals.expand([ ray_count, self.count ])

        if cfg.perturb:
            # Get intervals between samples.
            mids = 0.5 * (point_intervals[..., 1:] + point_intervals[..., :-1])
            upper = torch.cat((mids, point_intervals[..., -1:]), dim = -1)
            lower = torch.cat((point_intervals[..., :1], mids), dim = -1)

            # Stratified samples in those intervals.
            t_rand = torch.rand(
                point_intervals.shape,
                dtype = point_intervals.dtype,
                device = point_intervals.device,
            )

            point_intervals = lower + (upper - lower) * t_rand

        return point_intervals


class SamplePDF(torch.nn.Module):
    def __init__(self, num_samples):
        super(SamplePDF, self).__init__()
        self.num_samples = num_samples
        u = torch.linspace(0.0, 1.0, steps = self.num_samples)
        u.requires_grad = False
        self.register_buffer("u", u)

    def forward(self, point_interval, weights, perturb):
        points_on_rays_mid = 0.5 * (point_interval[..., 1:] + point_interval[..., :-1])
        interval_samples = self.sample_pdf(
            points_on_rays_mid, weights[..., 1:-1], self.u, det = (perturb == 0.0)
        ).detach()

        point_interval, _ = torch.sort(
            torch.cat((point_interval, interval_samples), dim = -1), dim = -1
        )
        return point_interval

    def sample_pdf(self, bins, weights, u, det = False):
        r"""sample_pdf function from another concurrent pytorch implementation
        by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
        """

        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, dim = -1, keepdim = True)
        cdf = torch.cumsum(pdf, dim = -1)
        cdf = torch.cat(
            [torch.zeros_like(cdf[..., :1]), cdf], dim = -1
        )  # (batchsize, len(bins))

        # Take uniform samples
        if det:
            u = u.expand(list(cdf.shape[:-1]) + [self.num_samples])
        else:
            u = torch.rand(
                list(cdf.shape[:-1]) + [self.num_samples],
                dtype = weights.dtype,
                device = weights.device,
            )

        # Invert CDF
        u = u.contiguous().detach()
        cdf = cdf.contiguous().detach()

        inds = torch.searchsorted(cdf, u, right = True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack((below, above), dim = -1)  # (batchsize, num_samples, 2)

        matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples


class SimpleModule(torch.nn.Module):
    def __init__(self, in_features, out_features, activation = torch.nn.ReLU()):
        super(SimpleModule, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.linear(x))


class SkipModule(torch.nn.Module):
    def __init__(self, in_features, out_features, activation = torch.nn.ReLU()):
        super(SkipModule, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features, activation)
        self.linear2 = torch.nn.Linear(out_features, out_features, activation)
        self.linear3 = torch.nn.Linear(in_features + out_features, out_features, activation)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.linear2(x1)
        x = torch.cat((x, x1), dim = -1)
        return self.linear3(x)


class MultiSkipModule(torch.nn.Module):
    def __init__(self, hidden_size, skip_size, layer_count, skip_step = 1, basic_module = SimpleModule):
        super(MultiSkipModule, self).__init__()
        self.num_layers = torch.nn.ModuleList(
            [basic_module(hidden_size + skip_size, hidden_size) for
             _ in range(layer_count)]
        )

        self.skip_layers = torch.nn.ModuleList([torch.nn.ModuleList(
            [basic_module(hidden_size, hidden_size) for _ in range(skip_step)]
        ) for _ in range(layer_count)])

    def forward(self, x, skip_value):
        value = x
        for hlayer, skiplayer in zip(self.num_layers, self.skip_layers):
            value = torch.cat([value, skip_value], dim = -1)
            value = hlayer(value)
            for layer in skiplayer:
                value = layer(value)
        return value


class SirenModule(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super(SirenModule, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        init_bounds = math.sqrt(6 / in_features) * weight_multiplier
        torch.nn.init.uniform_(self.linear.weight, a = -init_bounds, b = init_bounds)

    def forward(self, x):
        return torch.sin(self.linear(x))


class SirenModuleNormal(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super(SirenModuleNormal, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        torch.nn.init.normal_(self.linear.weight, mean = 0, std = weight_multiplier)

    def forward(self, x):
        return torch.sin(self.linear(x))


class SirenModuleExp(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super(SirenModuleExp, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        torch.nn.init.uniform_(self.linear.weight, a = -weight_multiplier, b = weight_multiplier)
        self.linear.weight.data = 2 ** self.linear.weight.data

    def forward(self, x):
        return torch.sin(self.linear(x))


class PotCoSirenModule(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super(PotCoSirenModule, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features // 2)
        torch.nn.init.uniform_(self.linear.weight, a = -weight_multiplier,
                               b = weight_multiplier)
        self.linear.weight.data = 2 ** self.linear.weight.data

    def forward(self, x):
        x = self.linear(x)
        return torch.cat([torch.sin(x), torch.cos(x)], dim = -1)


class CoSirenModule(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super(CoSirenModule, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features // 2)
        init_bounds = math.sqrt(24 / in_features) * weight_multiplier
        torch.nn.init.uniform_(self.linear.weight, a = -init_bounds, b = init_bounds)

    def forward(self, x):
        x = self.linear(x)
        return torch.cat([torch.sin(x), torch.cos(x)], dim = -1) - (math.pi / 4)


class GaussianNTK(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super(GaussianNTK, self).__init__()
        self.b = 2. ** torch.linspace(0, max_posenc_log_scale, out_features // in_fea) - 1
        self.osize = out_features
        self.a = torch.nn.Parameter(torch.ones((out_features)))

    def forward(self, x):
        x = self.linear(x)
        return torch.cat([self.a * torch.sin(x), self.a * torch.cos(x)], dim = -1)

    def output_size(self):
        return 2 * self.osize


class Embbed2(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super(Embbed2, self).__init__()
        self.b = 2. ** torch.linspace(0, weight_multiplier, out_features // in_features) - 1
        self.b = torch.nn.Parameter(
            torch.reshape(torch.eye(in_features) * self.b[:, None, None], [out_features, in_features]))
        self.osize = out_features
        self.a = torch.nn.Parameter(torch.ones((out_features)))

    def forward(self, x):
        x = torch.matmul(x, self.b.T)
        return torch.cat([self.a * torch.sin(x), self.a * torch.cos(x)], dim = -1)

    def output_size(self):
        return 2 * self.osize


class SpatialEmbedding(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super(SpatialEmbedding, self).__init__()
        self.b = torch.zeros((in_features, out_features))
        self.b.normal_(0, weight_multiplier)
        self.b = torch.nn.Parameter(2. ** self.b - 1)
        self.osize = out_features
        self.a = torch.nn.Parameter(torch.ones((out_features)))

    def forward(self, x):
        x = torch.matmul(x, self.b)
        return torch.cat([self.a * torch.sin(x), self.a * torch.cos(x)], dim = -1)

    def output_size(self):
        return 2 * self.osize


class SimpleSpatialEmbedding(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super(SimpleSpatialEmbedding, self).__init__()
        self.b = torch.zeros((in_features, out_features))
        self.b.normal_(0, weight_multiplier)
        self.b = torch.nn.Parameter(2. ** self.b - 1)
        self.osize = out_features

    def forward(self, x):
        x = torch.matmul(x, self.b)
        return torch.cat([torch.sin(x), torch.cos(x)], dim = -1)

    def output_size(self):
        return 2 * self.osize


class SimpleLuminance(torch.nn.Module):
    def __init__(self):
        super(SimpleLuminance, self).__init__()

    def forward(self, color, luminance):
        return color + luminance


class MultiplyLuminance(torch.nn.Module):
    def __init__(self):
        super(MultiplyLuminance, self).__init__()

    def forward(self, color, luminance):
        return color * (1 + luminance)


class NoLuminance(torch.nn.Module):
    def __init__(self):
        super(NoLuminance, self).__init__()

    def forward(self, color, luminance):
        return color


class FillUpLuminance(torch.nn.Module):
    def __init__(self):
        super(FillUpLuminance, self).__init__()

    def forward(self, color, luminance):
        return color + (1 - color) * luminance


class BoundedLuminance(torch.nn.Module):
    def __init__(self):
        super(BoundedLuminance, self).__init__()
        self.register_buffer("one", torch.tensor([1.0]))

    def forward(self, color, luminance):
        return torch.min(color + luminance, self.one)


def get_luminance_function(func):
    if func == "simple":
        return SimpleLuminance()
    elif func == "disabled":
        return NoLuminance()
    elif func == "multiply":
        return MultiplyLuminance()
    elif func == "fillup":
        return FillUpLuminance()
    elif func == "min1":
        return BoundedLuminance()


class ResBlock(torch.nn.Module):
    def __init__(self, hidden, hidden_mid = None):
        super(ResBlock, self).__init__()
        self.l0 = torch.nn.Sequential(
            SimpleModule(hidden, hidden_mid),
            SimpleModule(hidden_mid, hidden))

    def forward(self, x):
        return self.l0(x) + x


class FastRotPos(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super(FastRotPos, self).__init__()
        b = torch.zeros((in_features, out_features))
        b.normal_()
        b /= b.norm(dim = 0)
        multiplier = 2.0 ** (torch.rand((1, out_features)) * weight_multiplier) - 1
        self.register_buffer("b", (b * multiplier).detach())

    def forward(self, x):
        x = torch.matmul(x, self.b)
        return torch.cat([torch.sin(x), torch.cos(x)], dim = -1)

    def output_size(self):
        return 2 * self.b.shape[1]


class FlexiblePositionalEncoding(torch.nn.Module):
    """Apply positional encoding to the input.
    """

    def __init__(self, in_features, out_features, weight_multiplier = 1.0):
        super().__init__()
        self.num_encoding_functions = out_features
        frequency_bands = 2.0 ** torch.linspace(
            0.0, weight_multiplier, out_features
        )
        frequency_bands = (torch.eye(in_features)[..., None] * frequency_bands).view(in_features, -1)
        self.register_buffer("frequency_bands", frequency_bands)
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.frequency_bands)
        encoding = torch.cat([x, torch.sin(out), torch.cos(out)], dim = -1)
        return encoding

    def output_size(self):
        return 2 * self.frequency_bands.shape[-1] + self.in_features


def get_encoding(encoding):
    return {
        "fastrot": FastRotPos,
        "spatial": SpatialEmbedding,
        "positional": FlexiblePositionalEncoding,
    }[encoding]
