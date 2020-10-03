import json
import os
import OpenEXR as exr, Imath
import cv2
import imageio
import numpy as np
import torch


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


def load_blender_data(basedir, categories = None, half_res = False, testskip = 1, debug = False, empty = 0.):
    splits = [ "train", "val", "test" ] if categories is None else categories

    # Load meta data files
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    imgs = []
    depth_imgs = []
    poses = []
    counts = [0]

    for meta_name_type in splits:
        meta = metas[meta_name_type]

        curr_imgs = []
        curr_depth_imgs = []
        curr_poses = []

        if meta_name_type == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            f_name = os.path.join(basedir, frame["file_path"] + ".png")
            curr_imgs.append(imageio.imread(f_name))

            if meta_name_type == "test_high_res":
                f_depth_name = os.path.join(basedir, frame["file_path"] + "_depth_0001.exr")
                depth_curr = read_depth_from_exr(f_depth_name)
                depth_curr[depth_curr == depth_curr.max(initial = 0)] = empty
                depth_curr = depth_curr[..., 0]
            else:
                depth_curr = np.ones((800, 800)) * empty

            curr_depth_imgs.append(depth_curr)
            curr_poses.append(np.array(frame["transform_matrix"]))

        curr_imgs = (np.array(curr_imgs) / 255.0).astype(np.float32)
        curr_depth_imgs = np.array(curr_depth_imgs).astype(np.float32)
        curr_poses = np.array(curr_poses).astype(np.float32)
        counts.append(counts[-1] + curr_imgs.shape[0])

        imgs.append(curr_imgs)
        depth_imgs.append(curr_depth_imgs)
        poses.append(curr_poses)

    i_split = [ np.arange(counts[i], counts[i + 1]) for i in range(len(splits)) ]

    imgs = np.concatenate(imgs, 0)
    depth_imgs = np.concatenate(depth_imgs, 0)
    poses = np.concatenate(poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    # In debug mode, return extremely tiny images
    if debug:
        H = H // 32
        W = W // 32
        focal = focal / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        poses = torch.from_numpy(poses)

        return imgs, poses, depth_imgs, render_poses, [H, W, focal], i_split

    # Use half-resolution
    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = H // 2
        W = W // 2
        focal = focal / 2.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
    else:
        imgs = torch.from_numpy(imgs)

    poses = torch.from_numpy(poses)

    return imgs, poses, torch.from_numpy(depth_imgs), render_poses, [H, W, focal], i_split
