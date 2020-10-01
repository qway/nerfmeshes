import argparse
import numpy as np
import torch
import yaml

from plyfile import PlyData, PlyElement
from tqdm import tqdm

from nerf import (
    CfgNode,
    get_ray_bundle,
    pose_spherical,
    models,
    get_embedding_function,
    run_one_iter_of_nerf,
)


def export_obj(vertices, triangles, diffuse, normals, filename):
    """
    Exports a mesh in the (.obj) format.
    """
    print('Writing to obj')

    with open(filename, 'w') as fh:

        print(f"Total vertices {len(vertices)}")
        for index, v in enumerate(vertices):
            fh.write("vn {} {} {}".format(*normals[index]))
            fh.write("\n")

            fh.write("v {} {} {}".format(*v))
            if len(diffuse) > index:
                fh.write(" {} {} {}".format(*diffuse[index]))

            fh.write("\n")

        for f in triangles:
            fh.write("f")
            for index in f:
                fh.write(" {}//{}".format(index + 1, index + 1))

            fh.write("\n")


def export_ply(vertices, diffuse, normals, filename):
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


def export_ray_trace(model_coarse, model_fine, config_args, cfg, encode_position_fn, encode_direction_fn, device):

    # Mesh Extraction
    samples_dimen_y = 8
    samples_dimen_x = 4
    plane_near = 0
    plane_far = 4.0
    img_size = 800
    step_size = 2
    dist_threshold = 0.002
    prob_threshold = 0.6

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

        # cfg.nerf['validation']['num_coarse'] = 64
        # cfg.nerf['validation']['num_fine'] = 64
        with torch.no_grad():
            _, _, _, rgb_fine, _, depth_fine = run_one_iter_of_nerf(
                hwf[0], hwf[1], hwf[2],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
            )

            # Apply nn mask
            surface_points = ray_origins + ray_directions * depth_fine[..., None]

            acc = []
            initial_values = surface_points[grid.T[0], grid.T[1]].view(img_size, img_size, -1)
            size = step_size * 2 + 1
            size_samples = size ** 2 - 1
            for a in range(-step_size, step_size + 1):
                for b in range(-step_size, step_size + 1):
                    offset = torch.tensor([ a, b ])

                    new_grid = grid + offset
                    new_grid = new_grid.clamp(0, img_size - 1)

                    new_grid = surface_points[new_grid.T[0], new_grid.T[1]].view(img_size, img_size, -1)
                    new_grid_s = ((new_grid - initial_values) ** 2).sum(-1) < dist_threshold

                    acc.append(new_grid_s)

            new_mask = torch.stack(acc, -1).sum(-1).squeeze(-1) > size_samples * prob_threshold

            dep_mask = (depth_fine > 0)
            mask = new_mask * dep_mask

            ray_origins, ray_directions, depth_fine = ray_origins[mask], ray_directions[mask], depth_fine[mask]
            rgb_fine = rgb_fine[mask]

            surface_points = ray_origins + ray_directions * depth_fine[..., None]

            vertices.append(surface_points.view(-1, 3).cpu().detach())
            normals.append((-ray_directions).view(-1, 3).cpu().detach())
            diffuse.append(rgb_fine.view(-1, 3).cpu().detach())

    # Query the whole diffuse map
    diffuse_fine = torch.cat(diffuse, dim = 0).numpy()
    vertices_fine = torch.cat(vertices, dim = 0).numpy()
    normals_fine = torch.cat(normals, dim = 0).numpy()

    # Export model
    # export_obj(vertices_fine, [], diffuse_fine, normals_fine, "lego-sampling.obj")
    export_ply(vertices_fine, diffuse_fine, normals_fine, "lego-sampling.ply")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type = str, required = True, help = "Path to (.yml) config file."
    )
    parser.add_argument(
        "--base-dir",
        type = str,
        required = False,
        help = "Override the default base dir.",
    )
    parser.add_argument(
        "--checkpoint",
        type = str,
        required = True,
        help = "Checkpoint / pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--save-dir", type = str, help = "Save mesh to this directory, if specified."
    )

    parser.add_argument(
        "--iso-level",
        type = float,
        help = "Iso-Level to be queried",
        default = 32
    )

    parser.add_argument('--cache-mesh', dest = 'cache_mesh', action = 'store_true')
    parser.add_argument('--no-cache-mesh', dest = 'cache_mesh', action = 'store_false')
    parser.set_defaults(cache_mesh = True)

    config_args = parser.parse_args()

    # Read config file.
    cfg = None
    with open(config_args.config, "r") as f:
        cfg_dict = yaml.load(f, Loader = yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Device on which to run.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    encode_position_fn = get_embedding_function(
        num_encoding_functions = cfg.models.coarse.num_encoding_fn_xyz,
        include_input = cfg.models.coarse.include_input_xyz,
        log_sampling = cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions = cfg.models.coarse.num_encoding_fn_dir,
            include_input = cfg.models.coarse.include_input_dir,
            log_sampling = cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz = cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir = cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz = cfg.models.coarse.include_input_xyz,
        include_input_dir = cfg.models.coarse.include_input_dir,
        use_viewdirs = cfg.models.coarse.use_viewdirs,
    )
    model_coarse.to(device)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz = cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir = cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz = cfg.models.fine.include_input_xyz,
            include_input_dir = cfg.models.fine.include_input_dir,
            use_viewdirs = cfg.models.fine.use_viewdirs,
        )
        model_fine.to(device)

    checkpoint = torch.load(config_args.checkpoint)
    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        try:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        except:
            print(
                "The checkpoint has a fine-level model, but it could "
                "not be loaded (possibly due to a mismatched config file."
            )

    model_coarse.eval()
    if model_fine:
        model_fine.eval()

    export_ray_trace(model_coarse, model_fine, config_args, cfg, encode_position_fn, encode_direction_fn, device)


if __name__ == "__main__":
    main()
