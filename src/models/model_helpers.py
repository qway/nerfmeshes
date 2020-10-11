import collections
import numpy as np
import torch


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _nest_dict_rec(k, v, out, sep="_"):
    k, *rest = k.split(sep, 1)
    if rest:
        _nest_dict_rec(rest[0], v, out.setdefault(k, {}), sep)
    else:
        out[k] = v


def nest_dict(flat, sep="_"):
    result = {}
    for k, v in flat.items():
        _nest_dict_rec(k, v, result, sep)
    return result


def intervals_to_ray_points(point_intervals, ray_directions, ray_origin):
    ray_points = ray_origin[..., None, :] + ray_directions[..., None, :] * point_intervals[..., :, None]

    return ray_points


def get_ln_samples(near, far, num_rays, options, mode, type, device, total):
    t_vals = torch.linspace(0.0, 1.0, total, dtype=type, device=device)

    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    z_vals = z_vals.expand([num_rays, total])

    return z_vals


def get_random_samples(near, far, num_rays, options, mode, type, device, total):
    vv1, vv2 = near[0], far[0]

    r_vals = torch.rand((num_rays, total), dtype=type, device=device).sort(-1).values
    z_vals = vv1 + r_vals * (vv2 - vv1)

    return z_vals


def get_info_samples(depth_values, near, far, num_rays, options, mode, dtype, device, total, threshold=0.5):
    mask_zero = depth_values == options.dataset.empty

    far_t = far.clone()
    far_t[~mask_zero] = depth_values[~mask_zero] + threshold

    z_vals = get_ln_samples(near, far_t, num_rays, options, mode, dtype, device, total)

    mask_samples = torch.rand((mask_zero.sum(), total),
                              device=device) * options.dataset.far + options.dataset.near
    z_vals[mask_zero.squeeze(-1)] = mask_samples

    return torch.sort(z_vals.to(device), dim=-1).values


def get_ln_samples_sm(depth_values, near, far, num_rays, options, mode, dtype, device, total, fc1=10., fc2=2.,
                      off=0.5):
    mask_space = (depth_values != options.dataset.empty).squeeze(-1)

    t_vals = torch.linspace(0.0, 1.0, total, dtype=dtype, device=device) - off
    t_vals = ((torch.rand_like(t_vals) - 0.5) / fc1 + t_vals).sort(-1).values / fc2
    z_vals_space = t_vals.expand([num_rays, total])

    # z_vals_space = ((torch.rand((num_rays, total), device = device) - 0.5)).sort(-1).values
    # z_vals_space = z_vals_space + depth_values

    z_vals = get_ln_samples(near, far, num_rays, options, mode, dtype, device, total)
    z_vals[mask_space.squeeze(-1)] = z_vals_space[mask_space]

    return z_vals.to(device)


def get_ln_samples_prox(depth_values, near, far, num_rays, options, mode, dtype, device, total, off=0.4):
    mask_space = (depth_values != options.dataset.empty).squeeze(-1)

    depth_values_off = depth_values - off
    t_vals = torch.linspace(0.0, 1.0, total, dtype=dtype, device=device)
    z_vals_space = t_vals.expand([num_rays, total])
    z_vals_space = z_vals_space * (far - depth_values_off) + depth_values_off

    z_vals = get_ln_samples(near, far, num_rays, options, mode, dtype, device, total)
    z_vals[mask_space.squeeze(-1)] = z_vals_space[mask_space]

    return z_vals.to(device)


def sample_sm(ro, rd, near, far, num_rays, depth_values, options, mode, z_vals_ex=None):
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
            z_vals = get_info_samples(depth_values, near, far, num_rays, options, mode, ro.dtype, ro.device,
                                      total)

    if z_vals_ex is not None:
        z_vals = torch.cat((z_vals, z_vals_ex), dim=-1).sort(-1).values

    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    return pts, z_vals
