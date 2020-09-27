import torch


def batch_random_sampling(cfg, coords, ray_bundle):
    # Unpack ray bundle
    ray_directions, ray_targets = ray_bundle

    # Random 2D samples
    select_inds = torch.randperm(coords.shape[0])[:cfg.nerf.train.num_random_rays]
    select_inds = coords[select_inds]

    # Select random sub-samples
    ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
    ray_targets = ray_targets[select_inds[:, 0], select_inds[:, 1], :]

    return ray_directions, ray_targets, select_inds
