import json
import os
import cv2
import imageio
import torch
import numpy as np

from pathlib import Path
from data.data_helpers import read_depth_from_exr


def load_blender_data(cfg, data_config):
    """
    Args:
        cfg: Experiment configuration
        data_config: Path to the config of the dataset.

    Returns:
        imgs: The images.
        depth_maps: The depth maps associated with the images.
        normal_maps: The normal maps associated with the images.
        poses: Camera poses associated with the images.
        [H, W, focal]: The camera parameters.
    """
    # reduced_resolution = None, empty = 0.
    json_path = Path(data_config)
    basedir = json_path.parent

    print(f"Reading from ${json_path}")
    with json_path.open("r") as fp:
        metadata = json.load(fp)

    imgs = []
    poses = []
    depth_maps = []
    normal_maps = []
    for frame in metadata["frames"]:
        bundle_path = basedir / frame["file_path"]

        # Load rgb image
        img_path = bundle_path.with_suffix(".png")
        imgs.append(imageio.imread(img_path))

        # Load depth map
        depth_map_path = bundle_path.with_suffix("_depth.exr")
        if os.path.exists(depth_map_path):
            depth_map = read_depth_from_exr(depth_map_path)
            depth_map[depth_map == depth_map.max(initial = 0)] = cfg.dataset.empty

            depth_maps.append(depth_map[..., 0])

        # Load normal map
        normal_map_path = bundle_path.with_suffix("_normal.png")
        if os.path.exists(normal_map_path):
            normal_maps.append(imageio.imread(normal_map_path))

        poses.append(np.array(frame["transform_matrix"]))

    imgs = (np.array(imgs) / 255.0).astype(np.float32)[..., :3]
    depth_maps = np.array(depth_maps).astype(np.float32)
    normal_maps = (np.array(normal_maps) / 255.0).astype(np.float32)[..., :3]
    normal_maps = normal_maps / np.linalg.norm(normal_maps, axis = -1)[..., None]
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metadata["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # Customize data resolution
    reduced_resolution = cfg.dataset.reduced_resolution
    if reduced_resolution is not None:
        # TODO: resolution for depth/normals change too
        H = H // reduced_resolution
        W = W // reduced_resolution
        focal = focal / reduced_resolution
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize = (H, W), interpolation = cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
    else:
        imgs = torch.from_numpy(imgs)

    poses = torch.from_numpy(poses)

    return imgs, depth_maps, normal_maps, poses, [H, W, focal]
