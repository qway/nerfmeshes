import argparse
import os
import time
import numpy as np
import torch
import torchvision
import yaml

from pathlib import Path
from nerf import run_network
from skimage import measure
from scipy.spatial import KDTree
from tqdm import tqdm

from nerf import get_minibatches

from nerf import (
    CfgNode,
    load_blender_data,
    load_llff_data,
    models,
    get_embedding_function,
    predict_and_render_radiance,
    volume_render_radiance_field
)


def export_obj(vertices, triangles, diffuse, normals, filename):
    """
    Exports a mesh in the (.obj) format.
    """

    with open(filename, 'w') as fh:

        for index, v in enumerate(vertices):
            fh.write("v {} {} {} {} {} {}\n".format(*v, *diffuse[index]))

        for n in normals:
            fh.write("vn {} {} {}\n".format(*n))

        for f in triangles:
            fh.write("f")
            for index in f:
                fh.write(" {}//{}".format(index + 1, index + 1))

            fh.write("\n")

def generate(config, checkpoint, save_dir):

    cfg, model_name = None, None
    with open(config, "r") as f:
        cfg_dict = yaml.load(f, Loader = yaml.FullLoader)
        cfg, model_name = CfgNode(cfg_dict), Path(f.name).stem

    # Device on which to run.
    device = "cpu" if not torch.cuda.is_available() else "cuda"

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
    ).to(device)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz = cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir = cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz = cfg.models.fine.include_input_xyz,
            include_input_dir = cfg.models.fine.include_input_dir,
            use_viewdirs = cfg.models.fine.use_viewdirs,
        ).to(device)

    checkpoint = torch.load(checkpoint)
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

    # Mesh Extraction
    N = 128
    # iso_value = 32
    batch_size = 512
    density_samples_count = 6
    chunk = int(density_samples_count / 2)
    distance_length = 0.001
    distance_threshold = 0.001
    view_disparity = 0.2
    limit = 1.2
    t = np.linspace(-limit, limit, N)
    sampling_method = 0
    adjust_normals = False
    specific_view = False
    plane_near = 0
    plane_far = 6

    vertices, triangles, normals, diffuse = None, None, None, None
# if configargs.cache_mesh:
    query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    pts = torch.from_numpy(query_pts)
    dimension = pts.shape[-1]

    pts_flat = pts.reshape((-1, dimension))
    pts_flat_batch = pts_flat.reshape((-1, batch_size, dimension))

    density = np.zeros((pts_flat.shape[0]))
    for idx, batch in enumerate(tqdm(pts_flat_batch)):
        batch = batch.cuda()

        embedded = encode_position_fn(batch)
        if encode_direction_fn is not None:
            embedded_dirs = encode_direction_fn(batch)
            embedded = torch.cat((embedded, embedded_dirs), dim = -1)

        result_batch = model_fine(embedded)

        # Extracting the density
        density[idx * batch_size: (idx + 1) * batch_size] = result_batch[..., 3].cpu().detach().numpy()

    # Create a 3D density grid
    grid_alpha = density.reshape((N, N, N))
    iso_value = np.maximum(grid_alpha, 0).mean()

    # Extracting iso-surface triangulated
    vertices, triangles, normals, values = measure.marching_cubes(grid_alpha, iso_value)

    vertices = np.ascontiguousarray(vertices)
   
    vertices = torch.from_numpy(vertices) / N * 2 * limit - limit
    # torch.from_numpy(np.flip(x, axis=0).copy())
    triangles = torch.from_numpy(np.flip(triangles, axis=0).copy())
    return vertices, triangles