import os
import imageio
import argparse
import torch
import models
import torch.nn.functional as F

from lightning_modules import PathParser
from nerf import (
    mse2psnr,
    export_point_cloud,
    cast_to_pil_image,
    cast_to_disparity_image,
    batchify
)
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from data.datasets import BlenderDataset, DatasetType
from data.data_helpers import DataBundle


def eval_nerf(model, config_args, cfg, device):

    # Create dataset loader
    dataset = BlenderDataset(cfg, type = DatasetType.TEST)
    if config_args.synthesis_images:
        dataset.synthesis()

    data_loader = DataLoader(dataset, batch_size = 1)

    # Create directory to save images to.
    root_dir = Path(config_args.save_dir) / cfg.experiment.id
    if config_args.save_images:
        # Output
        images_dir = root_dir / "images"
        os.makedirs(images_dir, exist_ok=True)

        # Targets
        targets_dir = root_dir / "targets"
        os.makedirs(targets_dir, exist_ok=True)

    # Create directory to save disparity assets to.
    if config_args.save_disparity:
        disparity_dir = root_dir / "disparity"
        os.makedirs(disparity_dir, exist_ok=True)

    # Evaluation loop
    losses = []
    for img_nr, ray_batch in enumerate(tqdm(data_loader)):

        # Unpacking bundle
        bundle = DataBundle.deserialize(ray_batch).to_ray_batch()

        # Manual batching, since images are expensive to be kept on GPU
        batch_size = cfg.nerf.validation.chunksize
        batch_count = bundle.ray_directions.shape[0] / batch_size

        # Current metrics
        loss = 0
        rgb_map, disp_map = [], []
        batch_generator = batchify(bundle.ray_directions, bundle.ray_targets, batch_size = batch_size, device = device, progress=False)
        for (ray_directions, ray_targets) in batch_generator:
            # Query fine rgb and depth
            output_bundle = model.query((bundle.ray_origins.to(device), ray_directions, bundle.ray_bounds))

            # Accumulate queried rgb and depth
            rgb_map.append(output_bundle.rgb_map)
            disp_map.append(output_bundle.disp_map)

            if not config_args.synthesis_images:
                # Compute mean squared error
                loss += F.mse_loss(output_bundle.rgb_map, ray_targets)

        if not config_args.synthesis_images:
            loss /= batch_count
            losses.append(loss)

        # RGB and depth map output
        rgb_map, disp_map = torch.cat(rgb_map, 0), torch.cat(disp_map, 0)

        # Save images to newly created folder.
        if config_args.save_images:
            # Save image outputs
            file_name = os.path.join(images_dir, f"{img_nr:04d}.png")
            imageio.imwrite(file_name, cast_to_pil_image(rgb_map.view(bundle.hwf[0], bundle.hwf[1], 3)))

            if not config_args.synthesis_images:
                # Save image targets
                file_name = os.path.join(targets_dir, f"{img_nr:04d}.png")
                imageio.imwrite(file_name, cast_to_pil_image(bundle.ray_targets.view(bundle.hwf[0], bundle.hwf[1], 3)))

        # Save disparity assets to.
        if config_args.save_disparity:
            file_name = os.path.join(disparity_dir, f"{img_nr:04d}.png")
            imageio.imwrite(file_name, cast_to_disparity_image(
                disp_map.view(bundle.hwf[0], bundle.hwf[1]), white_background = True
            ))

        if not config_args.synthesis_images:
            print(f"[EVAL] Iter: {img_nr} Loss MSE {loss} / PSNR: {mse2psnr(loss)}")

    if not config_args.synthesis_images:
        total_loss = torch.stack(losses).mean()
        print(f"Dataset loss MSE: {total_loss} / PSNR: {mse2psnr(total_loss)}")


if __name__ == "__main__":
    torch.set_printoptions(threshold = 100, edgeitems = 50, precision = 8, sci_mode = False)

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
        "--save-dir", type=str, default=".",
        help="Save assets to this directory, if specified.",
    )
    parser.add_argument(
        "--save-images", action="store_true", default=False,
        help="Save view images.",
    )
    parser.add_argument(
        "--save-disparity", action="store_true", default=False,
        help="Save disparity images.",
    )
    parser.add_argument(
        "--synthesis-images", action="store_true", default=False,
        help="Synthesis new views 360° around the neural scene.",
    )
    config_args = parser.parse_args()

    # Existent log path
    path_parser = PathParser()
    cfg, _ = path_parser.parse(None, config_args.log_checkpoint, None, config_args.checkpoint)

    # Available device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model checkpoint
    print(f"Loading model from {path_parser.checkpoint_path}")
    model = getattr(models, cfg.experiment.model).load_from_checkpoint(path_parser.checkpoint_path)
    model = model.eval().to(device)

    with torch.no_grad():
        # Evaluate model
        eval_nerf(model, config_args, cfg, device)
