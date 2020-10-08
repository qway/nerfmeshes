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
    return c2w


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
