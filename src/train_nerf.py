import os

import argparse
import torch
import torchvision
import yaml
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from typing import Tuple

from nerf import models, CfgNode, mse2psnr, sample_pdf_2, BlenderRayDataset, \
    BlenderImageDataset, VolumeRenderer, RaySampleInterval, sample_pdf
import numpy as np


def fix_seed(seed):
    # Seed experiment for repeatability
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(configargs):
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    return cfg


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


def create_models(cfg) -> Tuple[torch.nn.Module, torch.nn.Module]:
    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
    )
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
        )
    return model_coarse, model_fine


def intervals_to_raypoints(point_intervals, ray_direction, ray_origin):
    raypoints = (
        ray_origin[..., None, :]
        + ray_direction[..., None, :] * point_intervals[..., :, None]
    )
    return raypoints


def create_intervals(
    ray_origin, ray_direction, bounds, point_amount=64, lindisp=True, perturb=False
):
    num_rays = ray_origin.shape[0]
    near, far = bounds[..., 0, None], bounds[..., 1, None]

    point_intervals = torch.linspace(
        0.0, 1.0, point_amount, dtype=ray_origin.dtype, device=ray_origin.device,
    )
    point_intervals = point_intervals[None, :]
    # Sample in disparity space, as opposed to in depth space. Sampling in disparity is
    # nonlinear when viewed as depth sampling! (The closer to the camera the more samples)
    if not lindisp:
        point_intervals = near * (1.0 - point_intervals) + far * point_intervals
    else:
        point_intervals = 1.0 / (
            1.0 / near * (1.0 - point_intervals) + 1.0 / far * point_intervals
        )

    if perturb:
        # Get intervals between samples.
        mids = 0.5 * (point_intervals[..., 1:] + point_intervals[..., :-1])
        upper = torch.cat((mids, point_intervals[..., -1:]), dim=-1)
        lower = torch.cat((point_intervals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(
            point_intervals.shape, dtype=ray_origin.dtype, device=ray_origin.device
        )
        point_intervals = lower + (upper - lower) * t_rand
    return point_intervals


def sample_from_weighted_interval(point_interval, weights, num_fine, perturb, device):
    points_on_rays_mid = 0.5 * (point_interval[..., 1:] + point_interval[..., :-1])
    interval_samples = sample_pdf_2(
        points_on_rays_mid,
        weights[..., 1:-1],
        num_fine,
        det=(perturb == 0.0)).detach()

    point_interval, _ = torch.sort(
        torch.cat((point_interval, interval_samples), dim=-1), dim=-1
    )
    return point_interval


class NeRFModel(LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.dataset_basepath = Path(cfg.dataset.basedir)

        self.loss = torch.nn.MSELoss()

        self.model_coarse, self.model_fine = create_models(cfg)
        self.volume_renderer = VolumeRenderer(
            cfg.nerf.train.radiance_field_noise_std,
            cfg.nerf.train.white_background,
            cfg.nerf.validation.radiance_field_noise_std,
            cfg.nerf.validation.white_background,
        )
        self.sample_interval = RaySampleInterval(cfg.nerf.train.num_coarse)

    def forward(self, x):
        """ Does a prediction for a batch of rays.

        Args:
            x: Tensor of camera rays containing position, direction and bounds.

        Returns: Tensor with the calculated pixel value for each ray.

        """
        ray_origins, ray_directions, bounds = x

        if self.model_coarse.training:
            nerf_cfg = self.cfg.nerf.train
        else:
            nerf_cfg = self.cfg.nerf.validation

        ray_intervals = self.sample_interval(bounds)
        raypoints = intervals_to_raypoints(ray_intervals, ray_directions, ray_origins)
        # Expand rays to match batchsize
        expanded_ray_directions = ray_directions[..., None, :].expand_as(raypoints)

        coarse_radiance_field = self.model_coarse(raypoints, expanded_ray_directions)
        (
            rgb_coarse,
            disp_coarse,
            acc_coarse,
            weights,
            depth_coarse,
        ) = self.volume_renderer(
            coarse_radiance_field,
            ray_intervals,
            ray_directions,
        )
        rgb_fine, disp_fine, acc_fine = None, None, None
        if nerf_cfg.num_fine > 0 and self.model_fine:
            ray_intervals = sample_from_weighted_interval(
                ray_intervals, weights, nerf_cfg.num_fine, nerf_cfg.perturb, self.device
            )
            raypoints = intervals_to_raypoints(
                ray_intervals, ray_directions, ray_origins
            )
            # Expand rays to match batchsize
            expanded_ray_directions = ray_directions[..., None, :].expand_as(raypoints)
            fine_radiance_field = self.model_fine(raypoints, expanded_ray_directions)
            rgb_fine, disp_fine, acc_fine, _, _ = self.volume_renderer(
                fine_radiance_field,
                ray_intervals,
                ray_directions
            )

        return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine

    def training_step(self, ray_batch, batch_idx):
        ray_origins, ray_directions, bounds, ray_targets = ray_batch

        rgb_coarse, _, _, rgb_fine, _, _ = self.forward(
            (ray_origins, ray_directions, bounds)
        )

        coarse_loss = self.loss(rgb_coarse[..., :3], ray_targets[..., :3])

        if rgb_fine is not None:
            fine_loss = self.loss(rgb_fine[..., :3], ray_targets[..., :3])
            loss = coarse_loss + fine_loss
        else:
            fine_loss = None
            loss = coarse_loss

        psnr = mse2psnr(loss.item())

        log_vals = {
            "train/loss": loss.item(),
            "train/coarse_loss": coarse_loss.item(),
            "train/psnr": psnr,
        }
        if rgb_fine is not None:
            log_vals["train/fine_loss"] = fine_loss.item()

        output = {
            "loss": loss,
            "progress_bar": {"training_loss": loss},
            "log": log_vals,
        }
        return output

    def validation_step(self, image_ray_batch, batch_idx):
        ray_origins, ray_directions, bounds, ray_targets = image_ray_batch

        # Manual batching, since images can be bigger than memory allows
        batchsize = self.cfg.nerf.validation.chunksize
        batchcount = ray_origins.shape[0] / batchsize

        coarse_loss = 0
        fine_loss = 0
        all_rgb_coarse = []
        all_rgb_fine = []
        for i in range(0, ray_origins.shape[0], batchsize):
            rgb_coarse, _, _, rgb_fine, _, _ = self.forward(
                (
                    ray_origins[i : i + batchsize],
                    ray_directions[i : i + batchsize],
                    bounds[i : i + batchsize],
                )
            )
            all_rgb_coarse.append(rgb_coarse)

            coarse_loss += self.loss(
                rgb_coarse[..., :3], ray_targets[i : i + batchsize, :3]
            )

            if rgb_fine is not None:
                fine_loss += self.loss(
                    rgb_fine[..., :3], ray_targets[i : i + batchsize, :3]
                )
                all_rgb_fine.append(rgb_fine)

        rgb_coarse = torch.cat(all_rgb_coarse, 0)

        coarse_loss /= batchcount
        if rgb_fine is not None:
            rgb_fine = torch.cat(all_rgb_fine, 0)
            fine_loss /= batchcount
            loss = fine_loss + coarse_loss
        else:
            rgb_fine = None
            loss = coarse_loss

        psnr = mse2psnr(loss.item())

        log_vals = {
            "validation/loss": loss.item(),
            "validation/coarse_loss": coarse_loss.item(),
            "validation/psnr": psnr,
        }
        if rgb_fine is not None:
            log_vals["validation/fine_loss"] = fine_loss.item()

        self.logger.experiment.add_image(
            "validation/rgb_coarse",
            cast_to_image(
                rgb_coarse[..., :3].view(self.val_dataset.H, self.val_dataset.W, 3)
            ),
            self.global_step,
        )
        if rgb_fine is not None:
            self.logger.experiment.add_image(
                "validation/rgb_fine",
                cast_to_image(
                    rgb_fine[..., :3].view(self.val_dataset.H, self.val_dataset.W, 3)
                ),
                self.global_step,
            )
        self.logger.experiment.add_image(
            "validation/img_target",
            cast_to_image(
                ray_targets[..., :3].view(self.val_dataset.H, self.val_dataset.W, 3)
            ),
            self.global_step,
        )
        output = {
            "val_loss": loss,
            "log": log_vals,
        }
        return output

    def train_dataloader(self):
        self.train_dataset = BlenderRayDataset(
            self.dataset_basepath / "transforms_train.json",
            self.cfg.nerf.train.num_random_rays,
            cfg.dataset.near,
            cfg.dataset.far,
            shuffle=True,
            white_background=self.cfg.nerf.train.white_background
            # stop=5, # Debug by loading only small part of the dataset
        )

        train_dataloader = DataLoader(self.train_dataset, batch_size=None)
        return train_dataloader

    def val_dataloader(self):
        self.val_dataset = BlenderImageDataset(
            self.dataset_basepath / "transforms_val.json",
            self.cfg.dataset.near,
            self.cfg.dataset.far,
            shuffle=True,
            white_background=self.cfg.nerf.train.white_background,
            stop=1,  #  Debug by only loading a small part of the dataset
        )
        val_dataloader = DataLoader(self.val_dataset, batch_size=None)
        return val_dataloader

    def configure_optimizers(self):
        return getattr(torch.optim, self.cfg.optimizer.type)(
            self.parameters(), lr=self.cfg.optimizer.lr
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--gpus", type=int, nargs="*", default=None, help="Gpus that should be used(Use instead of CUDA_VISIBLE_DEVICES)"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = load_config(configargs)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)
    fix_seed(cfg.experiment.randomseed)

    trainer = Trainer(
        gpus=list(configargs.gpus),
        val_check_interval=0.25,
        default_root_dir="../../logs",
        profiler=True,
        max_epochs=20,
    )
    model = NeRFModel(cfg)

    trainer.fit(model)
