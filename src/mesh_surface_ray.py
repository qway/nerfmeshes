import argparse
import numpy as np
import torch
import torchvision
import yaml
import math

from nerf.train_utils import predict_and_render_radiance
from tqdm import tqdm

from nerf import (
    CfgNode,
    get_ray_bundle,
    load_blender_data,
    load_llff_data,
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

        for index, v in enumerate(vertices):
            fh.write("v {} {} {}".format(*v))
            if len(diffuse) > index:
                fh.write(" {} {} {}".format(*diffuse[index]))

            fh.write("\n")

        for n in normals:
            fh.write("vn {} {} {}\n".format(*n))

        for f in triangles:
            fh.write("f")
            for index in f:
                fh.write(" {}//{}".format(index + 1, index + 1))

            fh.write("\n")


def export_ray_trace(model_coarse, model_fine, config_args, cfg, encode_position_fn, encode_direction_fn, device):

    # Mesh Extraction
    radius = 2
    plane_near = 0
    plane_far = 4

    # Data
    vertices, triangles, normals, diffuse = [], [], [], []

    for _ in tqdm(range(256)):

        # azimuthal, polar angle
        alpha, omega = torch.rand(1024, device = device) * 2 * math.pi, torch.rand(1024, device = device) * math.pi
        x = radius * torch.sin(alpha) * torch.cos(omega)
        y = radius * torch.sin(alpha) * torch.sin(omega)
        z = radius * torch.cos(alpha)

        # Ray origins
        ray_origins = torch.stack((x, y, z), -1)

        # Ray directions
        ray_directions = -(ray_origins / ray_origins.norm(dim = -1)[:, None])

        near = plane_near * torch.ones_like(ray_directions[..., :1])
        far = plane_far * torch.ones_like(ray_directions[..., :1])

        # Generating ray batches
        rays = torch.cat((ray_origins, ray_directions, near, far), dim = -1)
        if cfg.nerf.use_viewdirs:
            # Provide ray directions as input
            view_dirs = ray_directions / ray_directions.norm(p = 2, dim = -1).unsqueeze(-1)
            rays = torch.cat((rays, view_dirs.view((-1, 3))), dim = -1)

        # move to appropriate device
        ray_batch = rays.to(device)

        _, _, _, diffuse_fine, _, depth_fine = predict_and_render_radiance(
            ray_batch, model_coarse,
            model_fine, cfg,
            mode = "validation",
            encode_position_fn = encode_position_fn,
            encode_direction_fn = encode_direction_fn,
        )

        surface_points = ray_origins + ray_directions * depth_fine[:, None]

        vertices.append(surface_points.cpu().detach())
        diffuse.append(diffuse_fine.cpu().detach())

    # Query the whole diffuse map
    diffuse_fine = torch.cat(diffuse, dim = 0).numpy()
    vertices_fine = torch.cat(vertices, dim = 0).numpy()

    print(diffuse_fine.shape)
    print(vertices_fine.shape)

    # Export model
    export_obj(vertices_fine, [], diffuse_fine, [], "lego-sampling.obj")


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
