import collections
import os

import argparse
import torch
import torchvision
import yaml
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from typing import Tuple

from nerf import models, CfgNode, mse2psnr, sample_pdf_2, BlenderRayDataset, \
    BlenderImageDataset, VolumeRenderer, RaySampleInterval, sample_pdf, SamplePDF
import numpy as np

def load_config(configargs):
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    return cfg

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


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

def get_ray_batch(ray_batch):
    """ Removes all unecessary dimensions from a ray batch. """
    ray_origins, ray_directions, bounds, ray_targets = ray_batch
    ray_origins = ray_origins.view(-1, 3)
    ray_directions = ray_directions.view(-1, 3)
    bounds = bounds.view(-1, 2)
    ray_targets = ray_targets.view(-1, 4)
    return ray_origins, ray_directions, bounds, ray_targets


class NeRFModel(LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.hparams = flatten_dict(cfg, sep=".")
        self.logger.experiment.add_text(cfg.dump(), 0)
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
        self.sample_pdf = SamplePDF(cfg.nerf.train.num_fine)

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
            ray_intervals = self.sample_pdf(
                ray_intervals, weights, nerf_cfg.perturb)
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
        ray_origins, ray_directions, bounds, ray_targets = get_ray_batch(ray_batch)


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

        psnr = mse2psnr(loss)

        log_vals = {
            "train/loss": loss.item(),
            "train/coarse_loss": coarse_loss.item(),
            "train/psnr": psnr.item(),
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
        ray_origins, ray_directions, bounds, ray_targets = get_ray_batch(image_ray_batch)

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

        psnr = mse2psnr(loss)




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
            "validation/loss": loss,
            "validation/coarse_loss": coarse_loss,
            "validation/psnr": psnr,
        }
        if rgb_fine is not None:
            output["validation/fine_loss"] = fine_loss
        return output

    def validation_epoch_end(self, outputs):
        log_vals = {}
        for k in outputs[0].keys():
            log_vals[k] = torch.stack([x[k] for x in outputs]).mean()
        val_loss_mean = log_vals['val_loss']
        del log_vals['val_loss']
        for k,v in log_vals.items():
            log_vals[k] = v.item()

        return {'val_loss': val_loss_mean, "log": log_vals}

    def train_dataloader(self):
        self.train_dataset = BlenderRayDataset(
            self.dataset_basepath / "transforms_train.json",
            self.cfg.nerf.train.num_random_rays,
            cfg.dataset.near,
            cfg.dataset.far,
            shuffle=True,
            white_background=self.cfg.nerf.train.white_background,
            #stop=5, # Debug by loading only small part of the dataset
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
        "--gpus", type=int, default=1, help="Amount of Gpus that should be used(In most cases leave at 1)"
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
    seed_everything(cfg.experiment.randomseed)

    trainer = Trainer(
        gpus=configargs.gpus,
        val_check_interval=cfg.experiment.validate_every,
        default_root_dir="../logs",
        #profiler=True, # Activate for very simple profiling
        #fast_dev_run=True, # Activate when debugging
        max_steps=cfg.experiment.train_iters,
        log_gpu_memory=True,
        deterministic=True,
        accumulate_grad_batches=2
    )
    model = NeRFModel(cfg)

    trainer.fit(model)
