import cv2
import imageio
from pathlib import Path
import numpy as np
import json
import torch
from torch.utils.data import Dataset

from nerf import get_ray_bundle


def load_blender_data2(data_config, reduced_resolution=None, start=0, stop=-1):
    """

    Args:
        data_config: Path to the config of the dataset.
        reduced_resolution: Provides an option to divide the resolution by a number.
        skip: If frames in dataset should be skipped.

    Returns:

    """
    json_path = Path(data_config)
    basedir = json_path.parent
    with json_path.open("r") as fp:
        metadata = json.load(fp)

    imgs = []
    poses = []

    for frame in metadata["frames"][start:stop]:
        fname = basedir / (frame["file_path"] + ".png")
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame["transform_matrix"]))
    imgs = (np.array(imgs) / 255.0).astype(np.float32)
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
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)

    poses = torch.from_numpy(poses)

    return imgs, poses, [H, W, focal]


def convert_poses_to_rays(poses, H, W, focal):
    all_ray_origins = []
    all_ray_directions = []
    for pose in poses:
        pose_target = pose[:3, :4]
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
        all_ray_origins.append(ray_origins)
        all_ray_directions.append(ray_directions)
    all_ray_origins = torch.stack(all_ray_origins, 0)
    all_ray_directions = torch.stack(all_ray_directions, 0)
    return all_ray_directions, all_ray_origins


class BlenderRayDataset(Dataset):
    """
    A basic pytorch dataset for NeRF based on blender data. Note that it loads data
    already in rays and as such the original pictures are not recoverable. Do not use if
    you want to reconstruct images!
    """

    def __init__(
        self,
        config_path,
        num_random_rays,
        near,
        far,
        shuffle=False,
        start=0,
        stop=-1,
        white_background=False,
    ):
        self.num_random_rays = num_random_rays
        images, poses, hwf = load_blender_data2(
            config_path, reduced_resolution=None, start=start, stop=stop,
        )
        H, W, self.focal = hwf
        self.H, self.W = int(H), int(W)
        if white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])

        all_ray_directions, all_ray_origins = convert_poses_to_rays(
            poses, self.H, self.W, self.focal
        )

        # Linearize images, ray_origins, ray_directions
        self.ray_targets = torch.from_numpy(images).view(-1, 4)
        self.all_ray_origins = all_ray_origins.view(-1, 3)
        self.all_ray_directions = all_ray_directions.view(-1, 3)
        self.ray_bounds = (
            torch.tensor([near, far], dtype=self.ray_targets.dtype)
                .view(1, 2)
                .expand(self.num_random_rays, 2)
        )

        if shuffle:
            shuffled_indices = np.arange(self.ray_targets.shape[0])
            np.random.shuffle(shuffled_indices)
            self.ray_targets = self.ray_targets[shuffled_indices]
            self.all_ray_directions = self.all_ray_directions[shuffled_indices]
            self.all_ray_origins = self.all_ray_origins[shuffled_indices]

    def __len__(self):
        return int(self.ray_targets.shape[0] / self.num_random_rays)

    def __getitem__(self, idx):
        start = idx * self.num_random_rays
        indices = slice(start, start + self.num_random_rays)
        # Select pose and image
        ray_origin = self.all_ray_origins[indices]
        ray_direction = self.all_ray_directions[indices]
        ray_target = self.ray_targets[indices]
        ray_bounds = self.ray_bounds
        return (ray_origin, ray_direction, ray_bounds, ray_target)


class BlenderImageDataset(Dataset):
    """
    A basic pytorch dataset for NeRF based on blender data. A single datapoint is equal
    to a full picture, so some additonal preprocessing might be needed.
    """

    def __init__(
        self,
        config_path,
        near,
        far,
        shuffle=False,
        start=0,
        stop=-1,
        white_background=False,
    ):
        images, poses, hwf = load_blender_data2(
            config_path, reduced_resolution=None, start=start, stop=stop,
        )
        H, W, self.focal = hwf
        self.H, self.W = int(H), int(W)
        if white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])

        all_ray_directions, all_ray_origins = convert_poses_to_rays(
            poses, self.H, self.W, self.focal
        )

        # Linearize images, ray_origins, ray_directions
        total_images = images.shape[0]
        self.ray_targets = torch.from_numpy(images).view(total_images, -1, 4)
        self.all_ray_origins = all_ray_origins.view(total_images, -1, 3)
        self.all_ray_directions = all_ray_directions.view(total_images, -1, 3)
        self.ray_bounds = (
            torch.tensor([near, far], dtype=self.ray_targets.dtype)
            .view(1, 2)
            .expand(self.ray_targets.shape[1], 2)
        )

        if shuffle:
            shuffled_indices = np.arange(self.ray_targets.shape[0])
            np.random.shuffle(shuffled_indices)
            self.ray_targets = self.ray_targets[shuffled_indices]
            self.all_ray_directions = self.all_ray_directions[shuffled_indices]
            self.all_ray_origins = self.all_ray_origins[shuffled_indices]

    def __len__(self):
        return int(self.ray_targets.shape[0])

    def __getitem__(self, idx):
        ray_origin = self.all_ray_origins[idx]
        ray_direction = self.all_ray_directions[idx]
        ray_target = self.ray_targets[idx]
        ray_bounds = self.ray_bounds
        return (ray_origin, ray_direction, ray_bounds, ray_target)
