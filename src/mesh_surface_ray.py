import os
import argparse
import numpy as np
import torch
import yaml
import models

from plyfile import PlyData, PlyElement
from tqdm import tqdm
from lightning_modules import PathParser
from nerf import get_ray_bundle
from data import pose_spherical
from nerf.nerf_helpers import batchify


def export_ply(vertices, diffuse, normals, filename):
    print(f"Total vertices {len(vertices)}")

    names = 'x, y, z, nx, ny, nz, red, green, blue'
    formats = 'f4, f4, f4, f4, f4, f4, u1, u1, u1'
    arr = np.concatenate((vertices, normals, diffuse * 255), axis = -1)
    vertices_s = np.core.records.fromarrays(arr.transpose(), names = names, formats = formats)

    # Recreate the PlyElement instance
    v = PlyElement.describe(vertices_s, 'vertex')

    # Create the PlyData instance
    p = PlyData([ v ], text = True)

    p.write(filename)


def get_grid(size):
    x = torch.arange(size)
    a, b = torch.meshgrid(x, x)

    return torch.stack([ a.flatten(), b.flatten() ], dim = -1)


def export_ray_trace(model, config_args, cfg, device):

    # Mesh Extraction
    samples_dimen_y = 8
    samples_dimen_x = 4
    plane_near = 0
    plane_far = 2.0
    img_size = 30
    step_size = 2
    dist_threshold = 0.002
    prob_threshold = 0.

    # Data
    vertices, triangles, normals, diffuse = [], [], [], []
    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angleY, angleX, plane_far)).float()
            for angleY in np.linspace(-180, 180, samples_dimen_y, endpoint = False)
            for angleX in np.linspace(-90, 90, samples_dimen_x, endpoint = True)
        ], dim = 0
    )

    hwf = [ img_size, img_size, 1111.1111 ]

    grid = get_grid(img_size)
    for i, pose in enumerate(tqdm(render_poses)):
        pose = pose[:3, :4].to(device)

        # Ray origins & directions
        ray_origins, ray_directions = get_ray_bundle(hwf[0], hwf[1], hwf[2], pose)
        batch_generator = batchify(ray_directions, batch_size = config_args.batch_size,
                                   device = device, progress = False)

        depth_map = []
        rgb_map = []
        for (ray_directions,) in batch_generator:
            data = (ray_origins[None, :], ray_directions.view(-1, 3), (torch.tensor(0), torch.tensor(4)))

            # Query fine rgb and depth
            output_bundle = model.query(data)

            # Accumulate queried rgb and depth
            rgb_map.append(output_bundle.rgb_map)
            depth_map.append(output_bundle.depth_map)

        rgb_map = torch.cat(rgb_map, dim = 0)
        depth_map = torch.cat(depth_map, dim = 0).view(img_size, img_size)

        # # Apply nn mask
        # surface_points = ray_origins + ray_directions * bundle.depth_map[..., None]
        #
        # acc = []
        # initial_values = surface_points[grid.T[0], grid.T[1]].view(img_size, img_size, -1)
        # size = step_size * 2 + 1
        # size_samples = size ** 2 - 1
        # for a in range(-step_size, step_size + 1):
        #     for b in range(-step_size, step_size + 1):
        #         offset = torch.tensor([ a, b ])
        #
        #         new_grid = grid + offset
        #         new_grid = new_grid.clamp(0, img_size - 1)
        #
        #         new_grid = surface_points[new_grid.T[0], new_grid.T[1]].view(img_size, img_size, -1)
        #         new_grid_s = ((new_grid - initial_values) ** 2).sum(-1) < dist_threshold
        #
        #         acc.append(new_grid_s)
        #
        # new_mask = torch.stack(acc, -1).sum(-1).squeeze(-1) > size_samples * prob_threshold
        #
        # dep_mask = (bundle.depth_map > 0)
        # mask = new_mask * dep_mask
        #
        # ray_directions, bundle.depth_map = ray_directions[mask], bundle.depth_map[mask]

        surface_points = ray_origins + ray_directions * depth_map[..., None]

        vertices.append(surface_points.view(-1, 3).cpu().detach())
        normals.append((-ray_directions).view(-1, 3).cpu().detach())
        if not config_args.no_color:
            # [mask]
            rgb_fine = rgb_map.view(img_size, img_size, -1)

            diffuse.append(rgb_fine.view(-1, 3).cpu().detach())

    # Query the whole diffuse map
    diffuse_fine = torch.cat(diffuse, dim = 0).numpy()
    vertices_fine = torch.cat(vertices, dim = 0).numpy()
    normals_fine = torch.cat(normals, dim = 0).numpy()

    # Export model
    export_ply(vertices_fine, diffuse_fine, normals_fine, os.path.join(config_args.save_dir, "mesh.ply"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-checkpoint", type=str, default=None,
        help="Training log path with the config and checkpoints to load existent configuration.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="model_last.ckpt",
        help="Load existent configuration from the latest checkpoint by default.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024,
        help="Higher batch size results in faster processing but needs more device memory.",
    )
    parser.add_argument(
        "--base-dir",
        type = str,
        required = False,
        help = "Override the default base dir.",
    )
    parser.add_argument(
        "--no-color", action="store_true", default=False,
        help="Disable vertex color generation, useful for debugging."
    )
    parser.add_argument(
        "--save-dir", type = str, help = "Save mesh to this directory, if specified."
    )

    parser.add_argument('--cache-mesh', dest = 'cache_mesh', action = 'store_true')
    parser.add_argument('--no-cache-mesh', dest = 'cache_mesh', action = 'store_false')
    parser.set_defaults(cache_mesh = True)

    config_args = parser.parse_args()

    # Existent log path
    path_parser = PathParser()
    cfg, _ = path_parser.parse(None, config_args.log_checkpoint, None, config_args.checkpoint)

    # Available device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    # Load model checkpoint
    print(f"Loading model from {path_parser.checkpoint_path}")
    model = getattr(models, cfg.experiment.model).load_from_checkpoint(path_parser.checkpoint_path)
    # model = model.eval().to(device)
    model = model.eval()

    export_ray_trace(model, config_args, cfg, device)


if __name__ == "__main__":
    main()
