import math

import cv2
import imageio
from pathlib import Path
import numpy as np
import json
import torch
from torch.utils.data import Dataset

from data.load_blender import load_blender_data
from data.load_colmap import read_model
from data.load_llff import load_llff_data
from data.load_scannet import SensorData
from nerf import get_ray_bundle, meshgrid_xy


def dummy_rays_simple_radial(height: int, width: int, camera, resolution):
    f, cx, cy, k = camera[0], camera[1], camera[2], camera[0]
    f *= resolution
    cx *= resolution
    cy *= resolution
    ii, jj = meshgrid_xy(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
    )
    directions = torch.stack(
        [(ii - cx) / f, (jj - cy) / f, torch.ones_like(ii),], dim=-1,
    )
    #directions /= torch.norm(directions, dim=-1)[...,None]
    return directions


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


def get_rays(H, W, camera, poses, camera_model="SIMPLE_RADIAL", resolution=1.0):
    if camera_model == "SIMPLE_RADIAL":
        dummies = dummy_rays_simple_radial(H, W, camera, resolution)
    else:
        raise NotImplementedError(f"Camera model {camera_model} not implemented!")
    all_ray_origins = []
    all_ray_directions = []
    for pose in poses:
        ray_directions = torch.sum(dummies[..., None, :] * pose[:3, :3],
                                   dim=-1).float()
        ray_origins = (pose[:3, -1]).expand(ray_directions.shape).float()

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
        images, poses, hwf = load_blender_data(
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
        images, poses, hwf = load_blender_data(
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


class ScanNetDataset(Dataset):
    def __init__(
        self,
        data,
        num_random_rays=None,
        near=2,
        far=6,
        start=0,
        stop=-1,
        skip=None,
        skip_every=None,
        resolution=1.0,
        scale=1.0
    ):
        """

        :param data: A loaded .sens file
        :param num_random_rays: Number of rays that should be in a batch. If None, a batch consists of the full image.
        :param near: Near plane for bounds.
        :param far: Far plane for bounds.
        :param start: Start index for images.
        :param stop: Stop index for images.
        :param skip: If any images should be skipped.
        :param skip_every: The inverse of skip.
        """
        self.scale = scale
        self.num_random_rays = num_random_rays
        self.data = data
        self.skip = skip
        self.skip_every = skip_every
        self.resolution = resolution
        if stop == -1:
            self.stop = len(self.data.frames)
        else:
            self.stop = stop
        if skip:
            self.data_len = math.ceil(self.stop / skip)
            skip_every = None
        elif skip_every:
            self.data_len = self.stop - math.ceil(self.stop / skip_every)
        else:
            self.data_len = len(data.frames)
        self.H, self.W = int(self.data.color_height*resolution), int(self.data.color_width*resolution)
        self.dummy_rays = dummy_rays_simple_radial( # TODO: Implement other camera models
            self.H, self.W, self.data.intrinsic_color, self.resolution
        )
        if self.num_random_rays:
            self.pixels = torch.stack(
                meshgrid_xy(torch.arange(self.H), torch.arange(self.W)), dim=-1,
            ).reshape((-1, 2))
            self.ray_bounds = (
                torch.tensor([near, far], dtype=torch.float32)
                .view(1, 2)
                .expand(self.num_random_rays, 2)
            )
        else:
            self.ray_bounds = (
                torch.tensor([near, far], dtype=torch.float32)
                .view(1, 2)
                .expand(self.dummy_rays.shape[0] * self.dummy_rays.shape[1], 2)
            )

    def __len__(self):
        return self.data_len

    def __getitem__(self, item):
        # Get actual datapoint
        if self.skip:
            item *= self.skip
        if self.skip_every:
            item += (item // self.skip_every-1) + 1
        data_frame = self.data.frames[item]
        # Format image
        image = data_frame.decompress_color(self.data.color_compression_type)
        if self.resolution != 1.0:
            image = cv2.resize(image, dsize=(self.H, self.W), interpolation=cv2.INTER_AREA)
            image = np.transpose(image,(1,0,2))
        image = torch.from_numpy(image / 255.0).float()
        image = torch.cat((image, torch.ones(self.H, self.W, 1, dtype=image.dtype)), dim=-1)

        # Resolve ray directions and positions
        pose = torch.from_numpy(data_frame.camera_to_world)
        ray_directions = torch.sum(self.dummy_rays[..., None, :] * pose[:3, :3], dim=-1).float()
        ray_positions = (pose[:3, -1]*self.scale).expand(ray_directions.shape).float()

        if self.num_random_rays:

            # Choose subset to sample
            pixel_idx = np.random.choice(
                self.pixels.shape[0], size=(self.num_random_rays), replace=False
            )
            ray_idx = self.pixels[pixel_idx]
            return (
                ray_positions[ray_idx[:, 0], ray_idx[:, 1]],
                ray_directions[ray_idx[:, 0], ray_idx[:, 1]],
                self.ray_bounds,
                image[ray_idx[:, 0], ray_idx[:, 1]],
            )
        else:
            ray_positions = ray_positions.view(-1, 3)
            ray_directions = ray_directions.view(-1, 3)
            image = image.view(-1, 4)
            return ray_positions, ray_directions, self.ray_bounds, image


class ColmapDataset(Dataset):
    """
    A basic pytorch dataset for NeRF based on colmap data.
    """

    def __init__(
        self,
        config_folder_path,
        num_random_rays,
        near,
        far,
        shuffle=False,
        start=None,
        stop=None,
        downscale_factor=1.0
    ):
        super(ColmapDataset, self).__init__()
        resolution = 1/downscale_factor
        self.num_random_rays = num_random_rays
        if config_folder_path is not Path:
            config_folder_path = Path(config_folder_path)
        cameras, images, points3D = read_model(path=config_folder_path / "sparse" / "0", ext=".bin")

        list_of_keys = list(cameras.keys())
        cam = cameras[list_of_keys[0]]
        print('Cameras', len(cam))
        self.H, self.W, self.focal = int(cam.height), int(cam.width), cam.params[0]
        # Ignore Camera distortion for now

        w2c = []
        imgs = []
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        for image in list(images.values())[start:stop]:
            fname = config_folder_path / "images" / image.name
            imgs.append(imageio.imread(fname))

            R = image.qvec2rotmat()
            t = image.tvec.reshape([3, 1])
            m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
            w2c.append(m)

        w2c = np.stack(w2c, 0)
        c2w = np.linalg.inv(w2c)

        poses = torch.from_numpy(c2w[:, :3, :4])

        imgs = (np.array(imgs) / 255.0).astype(np.float32)

        if resolution!=1:
            self.H = int(self.H * resolution)
            self.W = int(self.W * resolution)
            self.focal = self.focal * resolution
            imgs = [
                torch.from_numpy(
                    cv2.resize(img, dsize=(self.W, self.H), interpolation=cv2.INTER_AREA)
                )
                for img in imgs
            ]
            imgs = torch.stack(imgs, 0)
        else:
            imgs = torch.from_numpy(imgs)

        if imgs.shape[-1] == 3:
            imgs = torch.cat((imgs, torch.ones(*imgs.shape[:-1],1)), dim=-1)

        all_ray_directions, all_ray_origins = get_rays(
            self.H, self.W, cam.params, poses, cam.model, resolution
        )


        if num_random_rays >= 0:
            # Linearize images, ray_origins, ray_directions
            self.ray_targets = imgs.view(-1, 4).float()
            self.all_ray_origins = all_ray_origins.view(-1, 3).float()
            self.all_ray_directions = all_ray_directions.view(-1, 3).float()
            self.ray_bounds = (
                torch.tensor([near, far], dtype=self.ray_targets.dtype)
                .view(1, 2)
                .expand(self.num_random_rays, 2).float()
            )
        else:
            total_images = imgs.shape[0]
            self.ray_targets = imgs.view(total_images, -1, 4).float()
            self.all_ray_origins = all_ray_origins.view(total_images, -1, 3).float()
            self.all_ray_directions = all_ray_directions.view(total_images, -1, 3).float()
            self.ray_bounds = (
                torch.tensor([near, far], dtype=self.ray_targets.dtype)
                    .view(1, 2)
                    .expand(self.ray_targets.shape[1], 2)
            ).float()

        if shuffle:
            shuffled_indices = np.arange(self.ray_targets.shape[0])
            np.random.shuffle(shuffled_indices)
            self.ray_targets = self.ray_targets[shuffled_indices]
            self.all_ray_directions = self.all_ray_directions[shuffled_indices]
            self.all_ray_origins = self.all_ray_origins[shuffled_indices]

    def __len__(self):
        if self.num_random_rays >= 0:
            return int(self.ray_targets.shape[0] / self.num_random_rays)
        else:
            return self.ray_targets.shape[0]

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Not a valid batch number!")
        if self.num_random_rays >= 0:
            start = idx * self.num_random_rays
            indices = slice(start, start + self.num_random_rays)
        else:
            indices = slice(idx, idx + 1)
        # Select pose and image
        ray_origin = self.all_ray_origins[indices]
        ray_direction = self.all_ray_directions[indices]
        ray_target = self.ray_targets[indices]
        ray_bounds = self.ray_bounds
        return (ray_origin, ray_direction, ray_bounds, ray_target)


class LLFFColmapDataset(Dataset):
    def __init__(
        self,
        config_folder_path,
        num_random_rays,
        shuffle=True,
        start=None,
        stop=None,
        downscale_factor=1,
        spherify=True,
    ):
        super(LLFFColmapDataset, self).__init__()
        self.num_random_rays = num_random_rays

        images, poses, bds, render_poses, i_test = load_llff_data(
            config_folder_path, factor=downscale_factor, spherify=spherify
        )
        hwf = poses[0, :3, -1]
        images = images[start:stop]
        poses = poses[start:stop, :3, :4]
        bds = bds[start:stop]


        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)

        H, W, self.focal = hwf
        self.H, self.W = int(H), int(W)

        all_ray_directions, all_ray_origins = convert_poses_to_rays(
            poses, self.H, self.W, self.focal
        )

        if images.shape[-1] == 3:
            images = torch.cat((images, torch.ones(*images.shape[:-1],1)), dim=-1)

        #bds[..., 0] -= 0.5
        #bds[..., 1] += 0.5
        ray_bounds = torch.from_numpy(bds)[:, None, None, :].expand(*images.shape[:-1],2)

        # Linearize images, ray_origins, ray_directions
        if num_random_rays >= 0:
            # Linearize images, ray_origins, ray_directions
            self.ray_targets = images.view(-1, 4).float()
            self.all_ray_origins = all_ray_origins.view(-1, 3).float()
            self.all_ray_directions = all_ray_directions.view(-1, 3).float()
            self.ray_bounds = ray_bounds.reshape(-1, 2).float()
        else:
            total_images = images.shape[0]
            self.ray_targets = images.view(total_images, -1, 4).float()
            self.all_ray_origins = all_ray_origins.view(total_images, -1, 3).float()
            self.all_ray_directions = all_ray_directions.view(total_images, -1,
                                                              3).float()
            self.ray_bounds = ray_bounds.view(total_images, -1, 2).float()

        if shuffle:
            shuffled_indices = np.arange(self.ray_targets.shape[0])
            np.random.shuffle(shuffled_indices)
            self.ray_targets = self.ray_targets[shuffled_indices]
            self.all_ray_directions = self.all_ray_directions[shuffled_indices]
            self.all_ray_origins = self.all_ray_origins[shuffled_indices]
            self.ray_bounds = self.ray_bounds[shuffled_indices]

    def __len__(self):
        if self.num_random_rays >= 0:
            return int(self.ray_targets.shape[0] / self.num_random_rays)
        else:
            return self.ray_targets.shape[0]

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Not a valid batch number!")
        if self.num_random_rays >= 0:
            start = idx * self.num_random_rays
            indices = slice(start, start + self.num_random_rays)
        else:
            indices = slice(idx, idx + 1)
        # Select pose and image
        ray_origin = self.all_ray_origins[indices]
        ray_direction = self.all_ray_directions[indices]
        ray_target = self.ray_targets[indices]
        ray_bounds = self.ray_bounds[indices]
        return (ray_origin, ray_direction, ray_bounds, ray_target)


def tensor_to_pcloud_point(x, c):
    return f"{x[0]};{x[1]};{x[2]};{c[0]};{c[1]};{c[2]}"

def rays_to_pcloud_string(ray_origin, ray_direction, ray_bounds, ray_target):
    if ray_target.shape[0] == 1:
        ray_origin, ray_direction, ray_bounds, ray_target = ray_origin[0], ray_direction[0], ray_bounds[0], ray_target[0]
    rays = [tensor_to_pcloud_point(rorg + rdir, rtar) for rorg, rdir, rtar in zip(ray_origin, ray_direction, ray_target)]
    rays += [tensor_to_pcloud_point(rorg, (0,0,0)) for rorg in ray_origin]
    return """
""".join(rays)


def data_set_to_pcloud_string(dataset):
    return "\n".join([rays_to_pcloud_string(*data) for data in dataset])



if __name__ == '__main__':
    # Tests for the ScanNet dataloader
    dataset = LLFFColmapDataset(
        "../../data/mountainbike/",
        num_random_rays=1000,
        #stop=10,  # Debug by loading only small part of the dataset
        downscale_factor=32
    )
    pcloud = data_set_to_pcloud_string(dataset)

    if dataset:
        pass
