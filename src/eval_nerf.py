import argparse
import os
import imageio
import numpy as np
import torch
import torchvision
import yaml
import time
import models

from nerf import (
    mse2psnr,
    export_point_cloud,
)

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from data.datasets import BlenderDataset, DatasetType
from nerf import CfgNode, RaySampleInterval, SamplePDF
from models import nest_dict, get_ray_batch


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)

    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))

    return img


def cast_to_disparity_image(tensor):
    img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    img = img.clamp(0, 1) * 255

    return img.detach().cpu().numpy().astype(np.uint8)


def main():
    torch.set_printoptions(threshold = 100, edgeitems = 50, precision = 8, sci_mode = False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "folder", type=str, help="Path to Log Folder"
    )
    parser.add_argument(
        "--save", action="store_true", help="Should images be saved?"
    )
    parser.set_defaults(save = False)
    parser.add_argument(
        "--save-dir", type=str, help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--save-disparity", action="store_true"
    )
    parser.add_argument(
        "--high-freq-sampling", action="store_true"
    )
    parser.add_argument('--synthesis-images', dest = 'synthesis_images', action = 'store_true')
    parser.set_defaults(synthesis_images = False)

    config_args = parser.parse_args()

    # Read config file.
    log_folder = Path(config_args.folder)
    with (log_folder / "hparams.yaml").open() as f:
        hparams = yaml.load(f, Loader = yaml.FullLoader)
        cfg = CfgNode(nest_dict(hparams, sep = "."))

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"

    try:
        checkpoint_path = next(log_folder.glob('*.ckpt'))
    except:
        raise FileNotFoundError("Could not find a .ckpt file in folder ", config_args.checkpoint)

    model = getattr(models, "NeRFModel").load_from_checkpoint(checkpoint_path, cfg = cfg)
    if config_args.high_freq_sampling:
        model.sample_interval = RaySampleInterval(cfg.nerf.train.num_coarse)
        model.sample_pdf = SamplePDF(cfg.nerf.train.num_fine * 2)

    model.eval()
    model.to(device)

    # if config_args.synthesis_images:
    #     eval_poses = render_poses.float()
    # else:
    #     eval_poses = poses.float()

    test_dataset = BlenderDataset(cfg, type = DatasetType.TEST)
    test_dataloader = DataLoader(test_dataset, batch_size = 1)

    # Create directory to save images to.
    if config_args.save and config_args.save_dir:
        save_dir = log_folder.name + "-eval-images"
        os.makedirs(save_dir, exist_ok=True)
        if config_args.save_disparity:
            os.makedirs(os.path.join(save_dir, "disparity"), exist_ok=True)

    # Evaluation loop
    dataset_coarse_loss = []
    dataset_fine_loss = []
    times_per_image = []
    for img_nr, image_ray_batch in enumerate(tqdm(test_dataloader)):
        start = time.time()
        # rgb = None, None
        # disp = None, None

        ray_origins, ray_directions, bounds, ray_targets, dep_target = get_ray_batch(
            image_ray_batch
        )
        # Manual batching, since images can be bigger than memory allows
        batchsize = cfg.nerf.validation.chunksize//2
        batchcount = ray_origins.shape[0] / batchsize

        coarse_loss = 0
        fine_loss = 0
        all_rgb_coarse = []
        all_disp_coarse = []
        all_rgb_fine = []
        all_disp_fine = []
        for i in trange(0, ray_origins.shape[0], batchsize, position = 0):
            rgb_coarse, disp_coarse, _, depth_coarse, rgb_fine, disp_fine, _, depth_fine = model.forward(
                (
                    ray_origins[i: i + batchsize].to(device),
                    ray_directions[i: i + batchsize].to(device),
                    bounds[i: i + batchsize].to(device),
                )
            )
            all_rgb_coarse.append(rgb_coarse)
            all_disp_coarse.append(disp_coarse)

            batch_targets = ray_targets[i: i + batchsize, :3].to(device)
            coarse_loss += torch.nn.functional.mse_loss(
                rgb_coarse[..., :3], batch_targets
            )

            if rgb_fine is not None:
                fine_loss += torch.nn.functional.mse_loss(
                    rgb_fine[..., :3], batch_targets
                )
                all_rgb_fine.append(rgb_fine)
                all_disp_fine.append(disp_fine)

        coarse_loss /= batchcount
        if rgb_fine is not None:
            rgb = torch.cat(all_rgb_fine, 0)
            disp = torch.cat(all_disp_fine, 0)
            fine_loss /= batchcount
        else:
            rgb = torch.cat(all_rgb_coarse, 0)
            disp = torch.cat(all_disp_coarse, 0)

        dataset_coarse_loss.append(coarse_loss.item())
        if rgb_fine is not None:
            dataset_fine_loss.append(fine_loss.item())

        if config_args.save:
            savefile = os.path.join(save_dir, f"{img_nr:04d}.png")
            imageio.imwrite(
                savefile, cast_to_image(rgb[..., :3].view(test_dataset.H, test_dataset.W, 3), cfg.dataset.type.lower())
            )

            savefile = os.path.join(config_args.save_dir, f"{i:04d} target.png")
            imageio.imwrite(
                savefile, cast_to_image(ray_targets[..., :3], cfg.dataset.type.lower())
            )

            if config_args.save_disparity:
                savefile = os.path.join(config_args.save_dir, "disparity", f"{i:04d}.png")
                # disp.view(test_dataset.H, test_dataset.W, 1
                imageio.imwrite(savefile, cast_to_disparity_image(disp))


        depth_target = depth_coarse if depth_fine is None else depth_fine
        export_point_cloud(i, ray_origins, ray_directions, depth_target, dep_target)

        times_per_image.append(time.time() - start)
        if config_args.save_dir:
            save_file = os.path.join(config_args.save_dir, f"{i:04d} depth_target.png")
            imageio.imwrite(save_file, cast_to_disparity_image(dep_target))

            save_file = os.path.join(config_args.save_dir, f"{i:04d} depth_output.png")
            imageio.imwrite(save_file, cast_to_disparity_image(depth_target))


        coarse_depth_loss = torch.nn.functional.mse_loss(depth_coarse, dep_target)
        fine_depth_loss = torch.tensor(0.)
        if depth_fine is not None:
            fine_depth_loss = torch.nn.functional.mse_loss(depth_fine, dep_target)

        print(f"Loss MSE image {i}: Coarse Loss: {coarse_loss} / Fine Loss: {fine_loss}")
        print(f"Loss PSNR image {i}: Coarse PSNR: {mse2psnr(coarse_loss.item())} / Fine PSNR: {mse2psnr(fine_loss.item())}")
        print(f"Loss Depth image {i}: Coarse Depth: {coarse_depth_loss} / Fine Depth: {fine_depth_loss}")

        tqdm.write(f"Avg time per image: {sum(times_per_image) / (img_nr + 1)}")

    mse_coarse = np.mean(dataset_coarse_loss)
    print("Coarse MSE:", mse_coarse, "\tCoarse PSNR:", mse2psnr(mse_coarse))
    if rgb_fine is not None:
        mse_fine = np.mean(dataset_fine_loss)
        print("Fine MSE:", mse_fine, "\tFine PSNR:", mse2psnr(mse_fine))


if __name__ == "__main__":
    with torch.no_grad():
        main()
