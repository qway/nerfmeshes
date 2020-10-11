import json
import os
import cv2
import imageio
import torch
import numpy as np

from pathlib import Path
from data.data_helpers import DataBundle, read_depth_from_exr


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

    print(f"Reading from {json_path}...")
    with json_path.open("r") as fp:
        metadata = json.load(fp)

    imgs = []
    poses = []
    depth = []
    normals = []
    for frame in metadata["frames"]:
        bundle_path = basedir / frame["file_path"]

        # Load rgb image
        img_path = bundle_path.with_suffix(".png")
        img = imageio.imread(img_path)[..., :3]

        imgs.append(img)

        # Load depth map
        depth_map_path = Path(f"{str(bundle_path)}_depth.exr")
        if os.path.exists(depth_map_path):
            depth_map = read_depth_from_exr(str(depth_map_path))
            depth_map[depth_map == depth_map.max(initial=0)] = cfg.dataset.empty

            depth.append(depth_map[..., 0])

        # Load normal map
        normal_map_path = Path(f"{str(bundle_path)}_normal.png")
        if os.path.exists(normal_map_path):
            try:
                normal_map = imageio.imread(normal_map_path)
                normals.append(normal_map)
            except:
                pass

        # Extract poses
        poses.append(np.array(frame["transform_matrix"])[:3, :4])

    size = len(imgs)
    print(f"Finished reading from {json_path} with {size} assets.")

    imgs = (np.array(imgs) / 255.0).astype(np.float32)
    if len(depth) != size:
        depth = None
    else:
        depth = torch.from_numpy(np.array(depth).astype(np.float32))

    if len(normals) != size:
        normals = None
    else:
        normals = (np.array(normals) / 255.0).astype(np.float32)[..., :3]
        normals = torch.from_numpy(normals / np.linalg.norm(normals, axis=-1)[..., None])

    # Pre-process poses
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metadata["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    # Customize data resolution
    reduced_resolution = cfg.dataset.reduced_resolution
    if reduced_resolution is not None and reduced_resolution > 1:
        H = H // reduced_resolution
        W = W // reduced_resolution
        focal = focal / reduced_resolution

        # TODO: resolution for depth/normals change too
        print(f"Using reduced resolution: {reduced_resolution} of size {W}x{H}")
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
    else:
        imgs = torch.from_numpy(imgs)

    if cfg.dataset.white_background:
        imgs = imgs * imgs[..., -1:] + (1.0 - imgs[..., -1:])

    poses = torch.from_numpy(poses)

    # Blender data bundle
    return DataBundle(
        ray_targets=imgs,
        target_depth=depth,
        target_normals=normals,
        poses=poses,
        hwf=(H, W, focal),
        size=size
    )
