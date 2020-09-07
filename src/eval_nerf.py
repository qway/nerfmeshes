import argparse
import os
import imageio
import numpy as np
import torch
import torchvision
import yaml

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from data.datasets import BlenderDataset
from nerf import CfgNode, RaySampleInterval, SamplePDF
from model_nerf import NeRFModel, nest_dict, get_ray_batch


def mse2psnr(mse):
    # For numerical stability, avoid a zero mse loss.
    if mse == 0:
        mse = 1e-5

    return -10.0 * np.log10(mse)


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
    parser.add_argument(
        "--save-disparity", action="store_true"
    )
    parser.add_argument(
        "--high-freq-sampling", action="store_true"
    )
    args = parser.parse_args()
    log_folder = Path(args.folder)
    with (log_folder / "hparams.yaml").open() as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(nest_dict(hparams, sep="."))

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = "cpu"
    try:
        checkpoint_path = next(log_folder.glob('*.ckpt'))
    except:
        raise FileNotFoundError("Could not find a .ckpt file in folder ",
                                args.checkpoint)
    model = NeRFModel.load_from_checkpoint(checkpoint_path, cfg=cfg)
    if args.high_freq_sampling:
        model.sample_interval = RaySampleInterval(cfg.nerf.train.num_coarse)
        model.sample_pdf = SamplePDF(cfg.nerf.train.num_fine*2)
    model.eval()
    model.to(device)

    test_dataset = BlenderDataset(
        Path(cfg.dataset.basedir) / "transforms_test.json",
        cfg.dataset.near,
        cfg.dataset.far,
        shuffle=True,
        white_background=cfg.nerf.train.white_background,
        #stop=5,  # Debug by only loading a small part of the dataset
    )

    test_dataloader = DataLoader(test_dataset, batch_size=None)

    # Create directory to save images to.
    if args.save:
        savedir = log_folder.name + "-eval-images"
        os.makedirs(savedir, exist_ok=True)
        if args.save_disparity:
            os.makedirs(os.path.join(savedir, "disparity"), exist_ok=True)

    # Evaluation loop
    dataset_coarse_loss = []
    dataset_fine_loss = []
    for img_nr, image_ray_batch in enumerate(tqdm(test_dataloader, position=0)):
        with torch.no_grad():
            ray_origins, ray_directions, bounds, ray_targets = get_ray_batch(
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
            for i in trange(0, ray_origins.shape[0], batchsize, position=1):
                rgb_coarse, disp_coarse, _, rgb_fine, disp_fine, _ = model.forward(
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


        if args.save:
            savefile = os.path.join(savedir, f"{img_nr:04d}.png")
            imageio.imwrite(
                savefile, cast_to_image(rgb[..., :3].view(test_dataset.H, test_dataset.W, 3), cfg.dataset.type.lower())
            )
            if args.save_disparity:
                savefile = os.path.join(savedir, "disparity", f"{img_nr:04d}.png")
                imageio.imwrite(savefile, cast_to_disparity_image(disp.view(test_dataset.H, test_dataset.W, 1)))

    mse_coarse = np.mean(dataset_coarse_loss)
    print("Coarse MSE:", mse_coarse, "\tCoarse PSNR:", mse2psnr(mse_coarse))
    if rgb_fine is not None:
        mse_fine = np.mean(dataset_fine_loss)
        print("Fine MSE:", mse_fine, "\tFine PSNR:", mse2psnr(mse_fine))

if __name__ == "__main__":
    main()
