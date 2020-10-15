import argparse
import os
import numpy as np
import torch
import yaml
import models

from pytorch3d.structures import Meshes
from nerf import get_minibatches
from nerf.nerf_helpers import cumprod_exclusive

# try:
#     import marching_cubes as mcubes
# except:
#     print("Error, justus' version of pymcubes not installed!")
#     print("""
#     Run the following commands(Note that its currently not possible to install through a package manager, since it depends on eigen):
#
#     poetry shell
#     cd ../additional_dependencies/PyMarchingCubes
#     python setup.py install
# """)


from pathlib import Path
from skimage import measure
from scipy.spatial import KDTree
from tqdm import tqdm
from nerf import CfgNode
from models.model_helpers import nest_dict
from nerf.nerf_helpers import export_obj


def create_mesh(verts, faces_idx):
    # We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
    # (scale, center) will be used to bring the predicted mesh to its original center and scale
    verts = verts - verts.mean(0)
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    # We construct a Meshes structure for the target mesh
    target_mesh = Meshes(verts=[verts], faces=[faces_idx])

    return target_mesh


def adjusting_normals(config_args, density, N, chunk, density_samples_count, distance_length, limit, normals,
                      pts_flat, sampling_method, vertices, distance_threshold):
    # Re-adjust normals based on NERF's density grid
    # Create a density KDTree look-up table
    tree = KDTree(pts_flat) if sampling_method == 0 else None
    # Create some density samples
    density_samples = np.linspace(
        -distance_length, distance_length, density_samples_count
    )[:, np.newaxis]
    # Adjust normals with the assumption of having proper geometry
    print("Adjusting normals")
    for index, vertex in enumerate(tqdm(vertices)):
        vertex_norm = vertex[[1, 0, 2]] / N * 2 * limit - limit
        vertex_direction = normals[index][[1, 0, 2]]

        # Sample points across the ray direction (a.k.a normal)
        samples = (
                vertex_norm[np.newaxis, :].repeat(density_samples_count, 0)
                + vertex_direction[np.newaxis, :].repeat(density_samples_count, 0)
                * density_samples
        )

        def extract_cum_density(samples):
            inliers_indices = None
            if sampling_method == 0:
                # Sample 1th nearest neighbor
                distances, indices = tree.query(samples, 1)

                # Filter outliers
                inliers_indices = indices[distances <= distance_threshold]
            elif sampling_method == 1:
                # Sample based on grid proximity
                indices = (
                    (
                            np.around((samples + limit) / 2 / limit * N)
                            * N ** np.arange(2, -1, -1)
                    )
                        .sum(1)
                        .astype(int)
                )

                # Filtering exceeding boundaries
                inliers_indices = indices[~(indices >= N ** 3)]
            else:
                # Sample based on re-computing the radiance field
                indices = (
                    (
                            np.around((samples + limit) / 2 / limit * N)
                            * N ** np.arange(2, -1, -1)
                    )
                        .sum(1)
                        .astype(int)
                )

                # Filtering exceeding boundaries
                inliers_indices = indices[~(indices >= N ** 3)]

            return density[inliers_indices].sum()

        # Extract densities
        sample_density_1 = extract_cum_density(samples[:chunk])
        sample_density_2 = extract_cum_density(samples[chunk:])

        # Re-direct the normal
        if sample_density_1 < sample_density_2:
            normals[index] *= -1


def export_marching_cubes(model, config_args, cfg, device):
    # Mesh Extraction
    N = config_args.res
    iso_value = config_args.iso_level
    batch_size = 1024
    density_samples_count = 6
    chunk = int(density_samples_count / 2)
    gap = 1e0
    gap_samples = 128
    distance_length = 0.001
    distance_threshold = 0.001
    view_disparity = 1e-3
    limit = config_args.limit
    sampling_method = 0
    dynamic_disparity = False
    with_view_dependence = True
    adjust_normals = False

    vertices, triangles, normals, diffuse = None, None, None, None
    mesh_cache_path = os.path.join(config_args.save_dir, "mesh_cache.pt")
    if config_args.cache_mesh:
        print("Generating mesh geometry...")

        if config_args.super_sampling >= 1:
            pass
            # grid_alpha_x, pts_flat_x = sample_points((N + (N - 1) * config_args.super_sampling, N, N), batch_size, device,
            #                                          model, limit)
            # grid_alpha_y, pts_flat_y = sample_points((N, N + (N - 1) * config_args.super_sampling, N), batch_size, device,
            #                                          model, limit)
            # grid_alpha_z, pts_flat_z = sample_points((N, N, N + (N - 1) * config_args.super_sampling), batch_size, device,
            #                                          model, limit)
            # if iso_value is None:
            #     iso_value_x = np.maximum(grid_alpha_x, 0).mean()
            #     iso_value_y = np.maximum(grid_alpha_x, 0).mean()
            #     iso_value_z = np.maximum(grid_alpha_x, 0).mean()
            #     iso_value = np.mean([iso_value_x, iso_value_y, iso_value_z])
            #
            # print("Iso-Value:", iso_value)
            # vertices, triangles = mcubes.marching_cubes_super_sampling(grid_alpha_x, grid_alpha_y, grid_alpha_z,
            #                                                            iso_value)
            # vertices = np.ascontiguousarray(vertices)
            # mcubes.export_obj(vertices, triangles, os.path.join(config_args.save_dir, "mesh.obj"))
            # return
        else:
            # Extract model geometry
            vertices, triangles, normals, grid_alpha, pts_flat, density = extract_geometry(model, N, batch_size, limit,
                                                                                           iso_value)

            # targets = torch.from_numpy(vertices) / N * 2 * limit - limit
            inv_normals = -normals

            # Reorder coordinates
            inv_normals = inv_normals[:, [1, 0, 2]]

            if adjust_normals:
                adjusting_normals(config_args, density, N, chunk, density_samples_count, distance_length,
                                  limit, normals, pts_flat, sampling_method, vertices, distance_threshold)

        torch.save((vertices, triangles, normals), mesh_cache_path)
        print("Mesh geometry saved successfully")
    else:
        print("Loading mesh geometry...")
        vertices, triangles, normals = torch.load(mesh_cache_path)

    # Extracting the diffuse color
    # Ray targets
    targets = vertices

    # Swap x-axis and y-axis
    targets = targets[:, [1, 0, 2]]

    # Ray directions
    directions = normals

    # Query directly without specific-views
    if not with_view_dependence:
        diffuse = np.zeros((len(targets), 3))
        for idx in tqdm(range(0, len(targets) // batch_size + 1)):
            offset1 = batch_size * idx
            offset2 = np.minimum(batch_size * (idx + 1), len(vertices))
            pos_batch = targets[offset1:offset2].to(device)
            normal_batch = inv_normals[offset1:offset2].to(device)
            result_batch = model.sample_points(pos_batch, normal_batch)
            # result_batch = model.sample_points(pos_batch, None)

            # Current color hack since the network does not normalize colors
            result_batch = result_batch[..., :3]
            # Query the whole diffuse map
            diffuse[offset1:offset2] = result_batch[..., :3].cpu().detach().numpy()
    else:
        # Move ray origins slightly towards positive sdf
        ray_origins = targets - view_disparity * inv_normals

        if dynamic_disparity:
            # Create some density samples
            density_samples = torch.linspace(1e-3, gap, gap_samples).repeat(targets.shape[0], 1)[..., None]
            density_points = targets[:, None, :] + density_samples * directions[:, None, :]
            density_indices = ((density_points + limit) / (2 * limit) * N).long().clamp_(0, N - 1)
            indices = (density_indices.view(-1, 3)[:, [1, 0, 2]] * (N ** torch.arange(2, -1, -1))[None, :]).sum(-1)

            tn_density = torch.from_numpy(density)
            tn_samples = tn_density[indices].view(targets.shape[0], gap_samples)
            # tn_attenut = (tn_samples * cumprod_exclusive(tn_samples >= 0))
            tn_attenut = tn_samples * cumprod_exclusive(torch.relu(tn_samples))
            tn_discrip = (tn_attenut > 1e-13).sum(-1)
            tn_offset = tn_discrip[:, None].float() / gap_samples * gap
            view_disparity = torch.max(tn_offset, torch.ones((targets.shape[0], 1)) * view_disparity)

            # Careful, do not set the near plane to zero!
            ray_bounds = (view_disparity * 2).expand(view_disparity.shape[0], 2).clone()
            ray_bounds[:, 0] = 0.001
        else:
            ray_bounds = (
                torch.tensor([0.001, 2 * view_disparity], dtype=ray_origins.dtype)
                    .expand(ray_origins.shape[0], 2)
            )

        # Generating ray batches
        rays = torch.cat((ray_origins, ray_bounds), dim=-1)
        if cfg.nerf.use_viewdirs:
            # Provide ray directions as input
            rays = torch.cat((rays, inv_normals), dim=-1)

        ray_batches = get_minibatches(rays, chunksize=batch_size)

        pred = []
        print("Started ray-casting")
        for ray_batch in tqdm(ray_batches):
            # move to appropriate device
            ray_batch = ray_batch.to(device)
            bray_origins, bray_bounds, bray_directions = ray_batch[..., :3], ray_batch[..., 3:5], ray_batch[..., 5:8]

            diffuse, _ = model.forward((bray_origins, bray_directions, bray_bounds.transpose(0, 1)))
            pred.append(diffuse.cpu().detach())

        # Query the whole diffuse map
        diffuse = torch.cat(pred, dim=0).numpy()

    # Export model
    print("Saving final model...", end="")
    export_obj(vertices, triangles, diffuse, normals, os.path.join(config_args.save_dir, "mesh.obj"))
    print("Saved!")


# model, N = 400, batch_size = 4096, limit = 1.2, iso_value = 32
def sample_points(N, batch_size, device, model, limit, color=False):
    if isinstance(N, tuple):
        x, y, z = N
        t_x = np.linspace(-limit, limit, x)
        t_y = np.linspace(-limit, limit, y)
        t_z = np.linspace(-limit, limit, z)
        query_pts = np.stack(np.meshgrid(t_y, t_x, t_z), -1).astype(np.float32)
    else:
        x, y, z = N, N, N
        t = np.linspace(-limit, limit, N)
        query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)

    pts = torch.from_numpy(query_pts)
    dimension = pts.shape[-1]
    pts_flat = pts.reshape((-1, dimension))
    pts_flat_batch = pts_flat.reshape((-1, batch_size, dimension))

    density = np.zeros((pts_flat.shape[0]))
    if color:
        colors = np.zeros((pts_flat.shape[0], 3))
    for idx, batch in enumerate(tqdm(pts_flat_batch)):
        batch = batch.to(device)
        result_batch = model.sample_points(
            batch, batch
        )  # Reuse positions as fake rays

        # Extracting the density
        density[idx * batch_size: (idx + 1) * batch_size] = (
            result_batch[..., 3].cpu().detach().numpy()
        )

        if color:
            colors[idx * batch_size: (idx + 1) * batch_size] = (
                result_batch[..., :3].cpu().detach().numpy()
            )

    # Create a 3D density grid
    grid_alpha = density.reshape((x, y, z))
    if color:
        grid_color = colors.reshape((x, y, z, 3))
        return grid_alpha, grid_color, pts_flat

    return grid_alpha, pts_flat, density


def extract_geometry(model, N=400, batch_size=4096, limit=1.2, iso_value=32):
    # Sample points based on the grid
    grid_alpha, pts_flat, density = sample_points(N, batch_size, device, model, limit)

    # Adaptive iso level
    min_a, max_a, std_a = grid_alpha.min(), grid_alpha.max(), grid_alpha.std()
    iso_value = min(max(iso_value, min_a + std_a), max_a - std_a)
    print(f"Min density {min_a}, Max density: {max_a}, Mean density {grid_alpha.mean()}")
    print(f"Querying based on iso level: {iso_value}")

    # Extracting iso-surface triangulated
    vertices, triangles, normals, values = measure.marching_cubes(grid_alpha, iso_value)

    normals = torch.from_numpy(np.ascontiguousarray(normals))
    vertices = torch.from_numpy(np.ascontiguousarray(vertices)) / N * 2 * limit - limit
    triangles = torch.from_numpy(np.ascontiguousarray(triangles))

    return vertices, triangles, normals, grid_alpha, pts_flat, density


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint / pre-trained model to evaluate.",
    )
    parser.add_argument(
        "--save-dir", type=str, help="Save mesh to this directory, if specified.", default="."
    )
    parser.add_argument(
        "--iso-level",
        type=float,
        help="Iso-Level to be queried",
        default=32
    )
    parser.add_argument(
        "--limit",
        type=float,
        help="Limits in -xyz to xyz for mcubes.",
        default=1.2
    )
    parser.add_argument(
        "--res",
        type=int,
        help="Sampling resolution for mcubes.",
        default=128
    )
    parser.add_argument(
        "--super-sampling",
        type=int,
        help="Add super sampling along the edges.",
        default=0,
    )
    parser.add_argument('--no-cache-mesh', dest='cache_mesh', action='store_false')
    parser.set_defaults(no_cache_mesh=True)

    config_args = parser.parse_args()
    log_folder = Path(config_args.config)
    with log_folder.open() as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(nest_dict(hparams, sep="."))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = config_args.checkpoint
    model = getattr(models, cfg.experiment.model).load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    export_marching_cubes(model, config_args, cfg, device)
