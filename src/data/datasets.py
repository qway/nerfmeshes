import os
import math
import cv2
import imageio
import numpy as np
import glob
import torch
import time

from abc import abstractmethod
from tqdm import trange
from enum import Enum
from pathlib import Path
from torch.utils.data import Dataset
from data.loaders.load_blender import load_blender_data
from data.loaders.load_colmap import read_model
from data.loaders.load_llff import load_llff_data
from nerf import get_ray_bundle, meshgrid_xy
from data import batch_random_sampling, pose_spherical
from data.data_helpers import DataBundle


class DatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"


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
        [(ii - cx) / f, (jj - cy) / f, torch.ones_like(ii), ], dim=-1,
    )

    # directions /= torch.norm(directions, dim=-1)[...,None]
    return directions


def convert_poses_to_rays(poses, H, W, focal):
    ray_origins = []
    ray_directions = []
    for pose in poses:
        chunk_ray_origins, chunk_ray_directions = get_ray_bundle(H, W, focal, pose)

        ray_origins.append(chunk_ray_origins)
        ray_directions.append(chunk_ray_directions)

    ray_origins = torch.stack(ray_origins, 0)
    ray_directions = torch.stack(ray_directions, 0)

    return ray_origins, ray_directions


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


class SynthesizableDataset(Dataset):

    STEP_SIZE = 3

    def __init__(self):
        super(SynthesizableDataset, self).__init__()

        # # Synthetic mesh target
        # self.target_mesh = None
        #
        # # Mesh path
        # model_path = Path(self.cfg.dataset.basedir) / f"model.obj"
        #
        # if os.path.exists(model_path):
        #     print("Loading 3D mesh...")
        #     # Loading target 3D mesh
        #     verts, faces, _ = load_obj(model_path)
        #
        #     # Find the faces and vertices
        #     faces_idx = faces.verts_idx
        #
        #     # Target mesh
        #     self.target_mesh = create_mesh(verts, faces_idx)

    def synthesis(self):
        print("Synthesizing dataset...")

        # Rotation lin samples around y-axis
        rot_samples = np.linspace(-270, 90, (360 // SynthesizableDataset.STEP_SIZE), endpoint=False)

        # Synthesizable 3D poses, 360° around the scene
        poses = torch.stack(
            [
                torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
                for angle in rot_samples
            ], 0,
        )

        # Synthetic data bundle
        self.synthetic_bundle = DataBundle(
            poses=poses,
            ray_bounds=self.ray_bounds,
            hwf=self.data_bundle.hwf,
            size=len(poses)
        )

        # Get synthetic ray origins and directions
        self.synthetic_bundle.ray_origins, self.synthetic_bundle.ray_directions = convert_poses_to_rays(
            poses, *self.data_bundle.hwf
        )


class CachingDataset(SynthesizableDataset, Dataset):

    def __init__(self, cfg, type):
        super(CachingDataset, self).__init__()
        assert type in DatasetType.__members__.values(), f"Invalid dataset type {type} expected {list(DatasetType.__members__.keys())}"

        self.cfg, self.type = cfg, type
        self.shuffle = True

        # Empty data bundle
        self.data_bundle = None

        # Synthetic data bundle
        self.synthetic_bundle = None

        # Dataset filters
        self.filters = ["ray_origins", "ray_directions", "ray_targets", "ray_bounds", "target_depth", "size", "hwf"]

        # Default experiment ray bounds
        self.ray_bounds = torch.tensor([self.cfg.dataset.near, self.cfg.dataset.far]).float()
        self.num_random_rays = self.cfg.nerf.train.num_random_rays

        # Cached dataset path
        self.path = os.path.join(self.cfg.dataset.caching.cache_dir, self.type.value)

        # Device on which to run.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        start_time = time.time()
        if self.cfg.dataset.caching.use_caching:
            # Dataset path
            cache_dir_exists = os.path.exists(self.path)
            if not cache_dir_exists:
                print(f"The path ${self.path} does not exist, creating one...")

                # Create dataset directory
                os.makedirs(self.path, exist_ok=True)

            if self.cfg.dataset.caching.override_caching or not cache_dir_exists:
                if cache_dir_exists:
                    print(f"Overriding the cached dataset to {self.path}...")
                else:
                    print(f"Creating the cached dataset to {self.path}...")

                self.cache_dataset()
            else:
                print(f"Using existent cached dataset from {self.path}...")

            self.paths = glob.glob(os.path.join(self.path, "*.data"))
            if len(self.paths) == 0:
                if cache_dir_exists:
                    print(f"The previous cached dataset is corrupted in {self.path}, overriding it...")
                    self.cache_dataset()

            self.paths = glob.glob(os.path.join(self.path, "*.data"))
            assert len(self.paths) > 0, f"There is a critical issue when caching the dataset"

            self.init_sampling(torch.load(self.paths[0])['hwf'])
            size = len(self.paths)
        else:
            self.data_bundle = self.load_dataset()

            self.init_sampling(self.data_bundle.hwf)
            self.data_bundle.ray_origins, self.data_bundle.ray_directions = convert_poses_to_rays(
                self.data_bundle.poses, *self.data_bundle.hwf
            )

            if self.cfg.dataset.use_ndc:
                # Use normalized device coordinates
                self.data_bundle.ndc()

            size = self.data_bundle.size

        time_last = time.time() - start_time
        if self.cfg.dataset.caching.use_caching:
            print(f"Using cached dataset in {time_last}s seconds with {size} assets...")
        else:
            print(f"Load whole dataset into the memory {time_last}s seconds...")

    def __len__(self):
        if self.synthetic_bundle is not None:
            return self.synthetic_bundle.size

        return len(self.paths) if self.cfg.dataset.caching.use_caching else self.data_bundle.size

    def __getitem__(self, idx):
        # Retrieve bundle sample
        if self.cfg.dataset.caching.use_caching:
            bundle = DataBundle.deserialize(torch.load(self.paths[idx]))
        else:
            if self.synthetic_bundle is not None:
                bundle = self.synthetic_bundle[idx]
            else:
                bundle = self.data_bundle[idx]

        # Random sampling if training
        if self.type == DatasetType.TRAIN:
            fn = lambda x: batch_random_sampling(self.cfg, self.coords, x)
            if self.cfg.dataset.use_ndc:
                # Use normalized device coordinates
                bundle = bundle.apply(fn, ["ray_origins", "ray_directions", "ray_targets", "target_depth", "target_normals"])
            else:
                bundle = bundle.apply(fn, ["ray_directions", "ray_targets", "target_depth", "target_normals"])

        return bundle.serialize(self.filters)

    def init_sampling(self, hwf):
        # Unpack data props
        H, W, _ = hwf

        # Coordinates to sample from, list of H * W indices in form of (width, height), H * W * 2
        self.coords = torch.stack(
            meshgrid_xy(torch.arange(H), torch.arange(W)),
            dim=-1,
        ).reshape((-1, 2))

    def save_dataset(self, bundle: DataBundle, img_idx, batch_idx=-1):
        """
            Script to run and cache a dataset for faster train-eval loops.
        """
        if batch_idx != -1:
            # Small dataset chunks (random sub-samples)
            raise NotImplementedError
        else:
            path = str(img_idx).zfill(4) + ".data"

        # location for the cached data
        save_path = os.path.join(self.cfg.dataset.caching.cache_dir, self.type.value, path)

        # serialize and save
        torch.save(bundle.to("cpu").serialize(self.filters), save_path)

    def cache_dataset(self):
        # TODO(0) testskip = args.blender_stride, offset for a small dataset
        # Unpacking data
        bundle = self.load_dataset().to(self.device)

        # Coordinates to sample from
        self.init_sampling(bundle.hwf)

        for img_idx in trange(bundle.size):
            # Create data chunk bundle
            sample = bundle[img_idx]
            sample.ray_origins, sample.ray_directions = get_ray_bundle(*sample.hwf, sample.poses)
            if self.cfg.dataset.use_ndc:
                # Use normalized device coordinates
                sample.ndc()

            if self.cfg.dataset.caching.sample_all or self.type == DatasetType.VALIDATION:
                self.save_dataset(sample, img_idx)
            else:
                raise NotImplementedError

    @property
    def dataset_path(self):
        return Path(self.cfg.dataset.basedir)

    @abstractmethod
    def load_dataset(self) -> DataBundle:
        pass


class BlenderDataset(CachingDataset):
    """
        A basic PyTorch dataset for NeRF based on blender data. A single sample is equal
        to a full picture, so some additional pre-processing might be needed.
    """

    def __init__(self, cfg, type=DatasetType.TRAIN):
        super(BlenderDataset, self).__init__(cfg, type)
        print("Loading Blender Data...")

    @property
    def dataset_path(self):
        return Path(self.cfg.dataset.basedir) / f"transforms_{self.type.value}.json"

    def load_dataset(self):
        # Blender data bundle
        bundle = load_blender_data(self.cfg, self.dataset_path)

        if bundle.ray_bounds is None:
            bundle.ray_bounds = self.ray_bounds

        return bundle


class ColmapDataset(CachingDataset):
    def __init__(self, cfg, spherify=True, type=DatasetType.TRAIN):
        self.downscale_factor = cfg.dataset.llff_downsample_factor
        self.spherify = spherify

        super(ColmapDataset, self).__init__(cfg, type)
        print("Loading Colmap Data...")

    def load_dataset(self):
        images, pose_mats, bounds, render_poses, i_test = load_llff_data(
            self.dataset_path, factor=self.downscale_factor, spherify=self.spherify
        )

        # Find train & validation partition
        samples_hold_count = self.cfg.dataset.llff_hold_step
        if samples_hold_count > 0:
            val_indices = np.arange(images.shape[0])[::samples_hold_count]
        else:
            val_indices = np.array([i_test])

        train_indices = np.array([i for i in np.arange(images.shape[0]) if i not in val_indices])

        # Select based on the dataset type
        target_indices = train_indices if self.type == DatasetType.TRAIN else val_indices

        # Split manually into train and validation
        pose_mats = torch.from_numpy(pose_mats[target_indices, ...])
        bounds = torch.from_numpy(bounds[target_indices, ...])
        images = torch.from_numpy(images[target_indices, ...])

        # HWF is constant always
        poses, hwf = pose_mats[:, :3, :4], tuple(pose_mats[0, :3, -1].long().tolist())

        # Colmap data bundle
        return DataBundle(
            ray_targets=images,
            ray_bounds=bounds,
            poses=poses,
            hwf=hwf,
            size=images.shape[0]
        )


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
        self.H, self.W = int(self.data.color_height * resolution), int(self.data.color_width * resolution)
        self.dummy_rays = dummy_rays_simple_radial(  # TODO: Implement other camera models
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
            item += (item // self.skip_every - 1) + 1
        data_frame = self.data.frames[item]
        # Format image
        image = data_frame.decompress_color(self.data.color_compression_type)
        if self.resolution != 1.0:
            image = cv2.resize(image, dsize=(self.H, self.W), interpolation=cv2.INTER_AREA)
            image = np.transpose(image, (1, 0, 2))
        image = torch.from_numpy(image / 255.0).float()
        image = torch.cat((image, torch.ones(self.H, self.W, 1, dtype=image.dtype)), dim=-1)

        # Resolve ray directions and positions
        pose = torch.from_numpy(data_frame.camera_to_world)
        ray_directions = torch.sum(self.dummy_rays[..., None, :] * pose[:3, :3], dim=-1).float()
        ray_positions = (pose[:3, -1] * self.scale).expand(ray_directions.shape).float()

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


class GeneralColmapDataset(Dataset):
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
        super(GeneralColmapDataset, self).__init__()
        resolution = 1 / downscale_factor
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

        if resolution != 1:
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
            imgs = torch.cat((imgs, torch.ones(*imgs.shape[:-1], 1)), dim=-1)

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
