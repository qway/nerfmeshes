import argparse
import os

import numpy as np
import torch
from tqdm import trange

from data import load_blender_data, load_llff_data, batch_random_sampling
from nerf import get_ray_bundle, meshgrid_xy


def cache_dataset(hwf, ray_origins, ray_directions, target, img_idx, cfg, type, batch_idx = -1):
    """
        Script to run and cache a dataset for faster train-eval loops.
    """
    batch_rays = torch.stack([ray_origins, ray_directions], dim = 0)

    cache_dict = {
        "hwf": hwf,
        "ray_bundle": batch_rays.detach().cpu(),
        "target": target.detach().cpu(),
    }

    if batch_idx != -1:
        path = str(img_idx).zfill(4) + str(batch_idx).zfill(4) + ".data"
    else:
        path = str(img_idx).zfill(4) + ".data"

    save_path = os.path.join(cfg.dataset.caching.cache_dir, type, path)

    torch.save(cache_dict, save_path)


def cache_nerf_dataset(cfg):
    images, poses, hwf = None, None, None

    if cfg.dataset.type == "blender":
        # testskip = args.blender_stride TODO(0)
        images, poses, hwf = load_blender_data(
            cfg.dataset.basedir, reduced_resolution = cfg.dataset.half_res
        )
    elif cfg.dataset.type == "llff":
        pass
        # images, poses, bds, render_poses, i_test = load_llff_data(
        #     args.data_path, factor = args.llff_downsample_factor
        # )
        # hwf = poses[0, :3, -1]
        # poses = poses[:, :3, :4]
        # if not isinstance(i_test, list):
        #     i_test = [i_test]
        # if args.llffhold > 0:
        #     i_test = np.arange(images.shape[0])[:: args.llffhold]
        # i_val = i_test
        # i_train = np.array(
        #     [
        #         i
        #         for i in np.arange(images.shape[0])
        #         if (i not in i_test and i not in i_val)
        #     ]
        # )
        # H, W, focal = hwf
        # hwf = [int(H), int(W), focal]
        # images = torch.from_numpy(images)
        # poses = torch.from_numpy(poses)
    else:
        raise Exception(f"Illegal dataset parameter - ${cfg.dataset.type}")

    # Device on which to run.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataset directories
    os.makedirs(os.path.join(cfg.dataset.caching.cache_dir, "train"), exist_ok = True)
    os.makedirs(os.path.join(cfg.dataset.caching.cache_dir, "val"), exist_ok = True)
    os.makedirs(os.path.join(cfg.dataset.caching.cache_dir, "test"), exist_ok = True)

    # Coordinates to sample from
    H, W, _ = hwf
    coords = torch.stack(
        meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
        dim = -1,
    ).reshape((-1, 2))

    for img_idx in trange(images.shape[0]):
        ray_targets = images[img_idx].to(device)
        pose_target = poses[img_idx, :3, :4].to(device)
        ray_origins, ray_directions = get_ray_bundle(*hwf, pose_target)

        if cfg.dataset.caching.sample_all:
            cache_dataset(hwf, ray_origins, ray_directions, ray_targets, img_idx, cfg, "train")
        else:
            for batch_idx in range(cfg.dataset.caching.num_variations):
                batch_rays = batch_random_sampling(cfg, coords, (ray_origins, ray_directions, ray_targets))

                cache_dataset(hwf, *batch_rays, img_idx, batch_idx, cfg, "train")

    # for img_idx in tqdm(i_val):
    #     img_target = images[img_idx].to(device)
    #     pose_target = poses[img_idx, :3, :4].to(device)
    #     ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
    #
    #     cache_dict = {
    #         "height": H,
    #         "width": W,
    #         "focal_length": focal,
    #         "ray_origins": ray_origins.detach().cpu(),
    #         "ray_directions": ray_directions.detach().cpu(),
    #         "target": img_target.detach().cpu(),
    #     }
    #
    #     save_path = os.path.join(args.save_dir, "val", str(img_idx).zfill(4) + ".data")
    #     torch.save(cache_dict, save_path)

