import torch

from nerf import (
    PositionalEncoding,
    SimpleModule,
    SimpleLuminance,
    get_luminance_function, SkipModule,
)


class FlexibleNeRFModel(torch.nn.Module):
    def __init__(
        self,
        num_layers=4,
        hidden_size=128,
        skip_connect_every=4,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        log_sampling_xyz=True,
        log_sampling_dir=True,
        use_viewdirs=True,
    ):
        super(FlexibleNeRFModel, self).__init__()
        self.encode_xyz = PositionalEncoding(
            num_encoding_fn_xyz, include_input_xyz, log_sampling_xyz
        )
        self.encode_dir = PositionalEncoding(
            num_encoding_fn_dir, include_input_dir, log_sampling_dir
        )

        include_input_xyz = 3 if include_input_xyz else 0
        include_input_dir = 3 if include_input_dir else 0
        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir
        self.skip_connect_every = skip_connect_every
        self.num_layers = num_layers
        if not use_viewdirs:
            self.dim_dir = 0

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layers_xyz = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            if i % self.skip_connect_every == 0 and i > 0 and i != num_layers - 1:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + hidden_size, hidden_size)
                )
            else:
                self.layers_xyz.append(torch.nn.Linear(hidden_size, hidden_size))

        self.use_viewdirs = use_viewdirs
        if self.use_viewdirs:
            self.layers_dir = torch.nn.ModuleList()
            # This deviates from the original paper, and follows the code release instead.
            self.layers_dir.append(
                torch.nn.Linear(self.dim_dir + hidden_size, hidden_size // 2)
            )

            self.fc_alpha = torch.nn.Linear(hidden_size, 1)
            self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
            self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        else:
            self.fc_out = torch.nn.Linear(hidden_size, 4)

        self.relu = torch.nn.functional.relu

    def forward(self, ray_points, ray_directions=None):
        xyz = self.encode_xyz(ray_points)
        x = self.layer1(xyz)
        for i, layer in enumerate(self.layers_xyz):
            if (
                i % self.skip_connect_every == 0
                and i > 0 and i != self.num_layers - 1
            ):
                x = torch.cat((x, xyz), dim=-1)
            x = self.relu(layer(x))
        if self.use_viewdirs:
            view = self.encode_dir(ray_directions)
            feat = self.relu(self.fc_feat(x))
            alpha = self.fc_alpha(x)
            x = torch.cat((feat, view), dim=-1)
            for l in self.layers_dir:
                x = self.relu(l(x))
            rgb = self.fc_rgb(x)
            return torch.cat((rgb, alpha), dim=-1)
        else:
            return self.fc_out(x)


class SimpleModel(torch.nn.Module):
    def __init__(
        self,
        hidden_layers=4,
        hidden_layers_view=2,
        hidden_size=128,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        log_sampling_xyz=True,
        log_sampling_dir=True,
    ):
        super(SimpleModel, self).__init__()
        self.encode_xyz = PositionalEncoding(
            num_encoding_fn_xyz, include_input_xyz, log_sampling_xyz
        )
        self.encode_dir = PositionalEncoding(
            num_encoding_fn_dir, include_input_dir, log_sampling_dir
        )
        self.hidden_layers = torch.nn.Sequential(
            SimpleModule(self.encode_xyz.output_size(), hidden_size),
            *[SimpleModule(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )

        self.hidden_layers_view_amount = hidden_layers_view
        if hidden_layers_view >= 0:
            self.density_out = torch.nn.Linear(hidden_size, 1)
            self.color_out = torch.nn.Sequential(
                SimpleModule(hidden_size + self.encode_dir.output_size(), hidden_size),
                *[
                    SimpleModule(hidden_size, hidden_size)
                    for _ in range(hidden_layers_view - 1)
                ],
                torch.nn.Linear(hidden_size, 3)
            )
        else:
            self.out = torch.nn.Linear(hidden_size, 4)

    def forward(self, ray_points, ray_directions=None):
        x = self.encode_xyz(ray_points)
        x = self.hidden_layers(x)
        if self.hidden_layers_view_amount >= 0 and ray_directions:
            color = self.encode_dir(ray_directions)
            color = torch.cat((x, color), dim=-1)
            color = self.color_out(color)
            x = self.out(x)
            return torch.cat((color, x), dim=-1)
        else:
            return self.out(x)


class ViewLuminanceModel(torch.nn.Module):
    def __init__(
        self,
        hidden_layers=4,
        hidden_layers_view=2,
        hidden_size=128,
        num_encoding_fn_xyz=6,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
        log_sampling_xyz=True,
        log_sampling_dir=True,
        luminance_function="simple",
    ):
        super(ViewLuminanceModel, self).__init__()
        self.encode_xyz = PositionalEncoding(
            num_encoding_fn_xyz, include_input_xyz, log_sampling_xyz
        )
        self.encode_dir = PositionalEncoding(
            num_encoding_fn_dir, include_input_dir, log_sampling_dir
        )
        self.hidden_layers = torch.nn.Sequential(
            SkipModule(self.encode_xyz.output_size(), hidden_size),
            *[SimpleModule(hidden_size, hidden_size) for _ in range(hidden_layers)]
        )

        self.out = torch.nn.Linear(hidden_size, 4)
        self.luminance_out = torch.nn.Sequential(
            SimpleModule(hidden_size + self.encode_dir.output_size() + self.encode_xyz.output_size(), hidden_size),
            *[
                SimpleModule(hidden_size, hidden_size)
                for _ in range(hidden_layers_view - 1)
            ],
            torch.nn.Linear(hidden_size, 1)
        )
        self.luminance_function = get_luminance_function(luminance_function)


    def forward(self, ray_points, ray_directions=None, only_luminance=True):
        xyz = self.encode_xyz(ray_points)
        x = self.hidden_layers(xyz)
        if ray_directions is not None:
            luminance = self.encode_dir(ray_directions)
            luminance = torch.cat((x, luminance, xyz), dim=-1)
            luminance = self.luminance_out(luminance)
            x = self.out(x)
            if only_luminance:
                color = luminance.expand(x[..., :3].shape)
            else:
                color = self.luminance_function(x[..., :3], luminance)
            return torch.cat((color, x[..., 2:3]), dim=-1)
        else:
            return self.out(x)
