import json
from pathlib import Path

import cv2
import imageio
import torch
import numpy as np


def load_blender_data(data_config, reduced_resolution = None):
    """
    Args:
        data_config: Path to the config of the dataset.
        reduced_resolution: Provides an option to divide the resolution by a number.

    Returns:
        imgs: The images.
        poses: Camera poses associated with the images.
        [H, W, focal]: The camera parameters.
    """
    json_path = Path(data_config)
    basedir = json_path.parent

    print(f"Reading from ${json_path}")
    with json_path.open("r") as fp:
        metadata = json.load(fp)

    imgs = []
    poses = []

    print(len(metadata["frames"]))
    for frame in metadata["frames"]:
        fname = basedir / (frame["file_path"] + ".png")
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame["transform_matrix"]))

    imgs = (np.array(imgs) / 255.0).astype(np.float32)[..., :3]
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metadata["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    if reduced_resolution:
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

    poses = torch.from_numpy(poses)

    return imgs, poses, [H, W, focal]
