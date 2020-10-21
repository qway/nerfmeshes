from dataclasses import astuple, dataclass, fields
from typing import Dict
from nerf.nerf_helpers import ndc_rays

import torch
import numpy as np
import OpenEXR as exr, Imath


def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w.astype(np.float32)


def batch_random_sampling(cfg, coords, ray_bundle: tuple):
    # Random 2D samples
    select_inds = torch.randperm(coords.shape[0])[:cfg.nerf.train.num_random_rays]
    select_inds = coords[select_inds]

    # Unpack ray bundle and select random sub-samples
    ray_bundle = tuple([
        ray_batch[select_inds[:, 0], select_inds[:, 1], ...] if ray_batch is not None else None
        for ray_batch in ray_bundle
    ])

    return ray_bundle


def read_depth_from_exr(filename):
    file = exr.InputFile(filename)
    header = file.header()

    dw = header['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    # convert all channels in the image to numpy arrays
    channelData = dict()
    for channel in header['channels']:
        data = file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
        data = np.fromstring(data, dtype = np.float32)
        data = np.reshape(data, size)

        channelData[channel] = data

    if 'Z' in header['channels']:
        return channelData['Z']

    channels = ['R', 'G', 'B', 'A'] if 'A' in header['channels'] else ['R', 'G', 'B']
    img = np.concatenate([channelData[channel][..., np.newaxis] for channel in channels], axis = 2)

    return img


@dataclass
class DataBundle:
    ray_origins: torch.Tensor = None
    ray_directions: torch.Tensor = None
    ray_targets: torch.Tensor = None
    ray_bounds: torch.Tensor = None
    target_depth: torch.Tensor = None
    target_normals: torch.Tensor = None
    poses: torch.Tensor = None
    size: int = -1
    hwf: tuple = None

    def __iter__(self):
        return iter(astuple(self))

    def __getitem__(self, keys):
        if isinstance(keys, int):
            bundle = DataBundle()
            for field in fields(self):
                value = getattr(self, field.name)
                if value is not None and isinstance(value, torch.Tensor) and value.shape[0] == self.size:
                    value = value[keys]

                setattr(bundle, field.name, value)

            return bundle

        if not isinstance(keys, tuple):
            return getattr(self, keys)

        return iter([ getattr(self, k) for k in keys ])

    @staticmethod
    def deserialize(dict: Dict):
        bundle = DataBundle()
        for field in fields(bundle):
            if field.name in dict:
                setattr(bundle, field.name, dict[field.name])

        return bundle

    def apply(self, func, names):
        rel_names = [ name for name in names if getattr(self, name) is not None ]
        mapping = [ getattr(self, name) for name in rel_names ]
        values = func(mapping)

        bundle = DataBundle()
        for field in fields(self):
            if field.name in rel_names:
                setattr(bundle, field.name, values[rel_names.index(field.name)])
            else:
                setattr(bundle, field.name, getattr(self, field.name))

        return bundle

    def to_ray_batch(self):
        """ Removes all unnecessary dimensions from a ray batch. """
        self.ray_origins = self.ray_origins.view(-1, 3)
        self.ray_directions = self.ray_directions.view(-1, 3)
        self.ray_bounds = self.ray_bounds.view(2)

        if self.ray_targets is not None:
            self.ray_targets = self.ray_targets.view(-1, 3)

        if self.target_depth is not None:
            self.target_depth = self.target_depth.view(-1)

        return self

    def to(self, device):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and isinstance(value, torch.Tensor):
                value = value.to(device)

            setattr(self, field.name, value)

        return self

    def serialize(self, filters) -> Dict:
        return {
            field.name: getattr(self, field.name) for field in fields(self)
            if getattr(self, field.name) is not None and field.name in filters
        }

    def ndc(self):
        self.ray_origins, self.ray_directions = ndc_rays(*self.hwf, 1.0, self.ray_origins[None, None, :], self.ray_directions)

        return self
