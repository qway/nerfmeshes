import collections
import os
import numpy as np
import argparse
import time
import torch
import torchvision
import yaml
import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from typing import Tuple
from data.datasets import BlenderDataset, ScanNetDataset, ColmapDataset, LLFFColmapDataset, DatasetType
from data.load_scannet import SensorData
from nerf import models, CfgNode, mse2psnr, VolumeRenderer, RaySampleInterval, \
    SamplePDF, DensityExtractor



def flatten_dict(d, parent_key = "", sep = "_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep = sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def nest_dict(flat, sep = "_"):
    result = {}
    for k, v in flat.items():
        _nest_dict_rec(k, v, result, sep)
    return result


def _nest_dict_rec(k, v, out, sep = "_"):
    k, *rest = k.split(sep, 1)
    if rest:
        _nest_dict_rec(rest[0], v, out.setdefault(k, {}), sep)
    else:
        out[k] = v


def intervals_to_ray_points(point_intervals, ray_directions, ray_origin):
    ray_points = ray_origin[..., None, :] + ray_directions[..., None, :] * point_intervals[..., :, None]

    return ray_points


def create_models(cfg) -> Tuple[torch.nn.Module, torch.nn.Module]:
    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse_type)(**cfg.models.coarse)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine") and cfg.models.use_fine:
        model_fine = getattr(models, cfg.models.fine_type)(**cfg.models.fine)

    return model_coarse, model_fine


def get_ray_batch(ray_batch):
    """ Removes all unnecessary dimensions from a ray batch. """
    ray_origins, ray_directions, ray_targets = ray_batch
    ray_origins = ray_origins.view(-1, 3)
    ray_directions = ray_directions.view(-1, 3)
    ray_targets = ray_targets.view(-1, 3)

    return ray_origins, ray_directions, ray_targets


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu().float()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


class NeRFModel(LightningModule):
    def __init__(self, cfg, hparams = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hparams:
            self.cfg = CfgNode(nest_dict(hparams, sep = "."))
            self.hparams = hparams
        else:
            self.cfg = cfg
            self.hparams = flatten_dict(cfg, sep = ".")

        self.dataset_basepath = Path(cfg.dataset.basedir)
        self.loss = torch.nn.MSELoss()

        self.train_dataset, self.val_dataset = None, None
        self.model_coarse, self.model_fine = create_models(cfg)
        self.volume_renderer = VolumeRenderer(
            cfg.nerf.train.radiance_field_noise_std,
            cfg.nerf.train.white_background,
            cfg.nerf.validation.radiance_field_noise_std,
            cfg.nerf.validation.white_background,
        )

        self.sample_interval = RaySampleInterval(cfg, cfg.nerf.train.num_coarse, perturb = cfg.nerf.train.perturb)
        self.sample_pdf = SamplePDF(cfg.nerf.train.num_fine)

        self.load_train_dataset()
        self.load_val_dataset()
        if self.cfg.dataset.type == "scannet":
            self.sensor_data = None

    def forward(self, x):
        """ Does a prediction for a batch of rays.

        Args:
            x: Tensor of camera rays containing position, direction and bounds.

        Returns: Tensor with the calculated pixel value for each ray.

        """
        ray_origins, ray_directions = x

        if self.model_coarse.training:
            nerf_cfg = self.cfg.nerf.train
        else:
            nerf_cfg = self.cfg.nerf.validation

        ray_intervals = self.sample_interval(self.cfg.dataset.near, self.cfg.dataset.far)
        ray_intervals = ray_intervals[:ray_directions.shape[0]]
        ray_points = intervals_to_ray_points(ray_intervals, ray_directions, ray_origins)

        # Expand rays to match batchsize
        expanded_ray_directions = ray_directions[..., None, :].expand_as(ray_points)

        coarse_radiance_field = self.model_coarse(ray_points, expanded_ray_directions)
        (
            rgb_coarse,
            disp_coarse,
            acc_coarse,
            weights,
            depth_coarse,
        ) = self.volume_renderer(coarse_radiance_field, ray_intervals, ray_directions, )

        rgb_fine, disp_fine, acc_fine = None, None, None
        if nerf_cfg.num_fine > 0 and self.model_fine is not None:
            fine_ray_intervals = self.sample_pdf(ray_intervals, weights, nerf_cfg.perturb)
            ray_points = intervals_to_ray_points(
                fine_ray_intervals, ray_directions, ray_origins
            )

            # Expand rays to match batch_size
            expanded_ray_directions = ray_directions[..., None, :].expand_as(ray_points)
            fine_radiance_field = self.model_fine(ray_points, expanded_ray_directions)

            rgb_fine, disp_fine, acc_fine, _, _ = self.volume_renderer(
                fine_radiance_field, fine_ray_intervals, ray_directions
            )
            return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine

        return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine

    def sample_points(self, points, rays = None, **kwargs):
        model = self.model_coarse
        if self.model_fine:
            model = self.model_fine

        results = model(points, rays, **kwargs)
        if isinstance(results, tuple):
            return results[0]

        return results

    def training_step(self, ray_batch, batch_idx):
        ray_origins, ray_directions, ray_targets = get_ray_batch(ray_batch)
        rgb_coarse, _, _, rgb_fine, _, _ = self.forward(
            (ray_origins, ray_directions)
        )

        coarse_loss = self.loss(rgb_coarse, ray_targets)
        coarse_psnr = mse2psnr(coarse_loss)

        loss = coarse_loss
        if rgb_fine is not None:
            fine_loss = self.loss(rgb_fine, ray_targets)
            fine_psnr = mse2psnr(fine_loss)

            loss += fine_loss

        log_vals = {
            "train/loss": loss,
            "train/coarse_loss": coarse_loss,
            "train/coarse_psnr": coarse_psnr,
            "train/lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }

        if rgb_fine is not None:
            log_vals = {
                **log_vals,
                "train/fine_loss": fine_loss,
                "train/fine_psnr": fine_psnr,
            }

        return {
            "loss": loss,
            "log": log_vals
        }

    def validation_step(self, image_ray_batch, batch_idx):
        ray_origins, ray_directions, ray_targets = get_ray_batch(image_ray_batch)

        # Manual batching, since images are very expensive in terms of GPU memory
        batch_size = self.cfg.nerf.validation.chunksize
        batch_count = ray_targets.shape[0] / batch_size

        coarse_loss = 0
        fine_loss = 0
        all_rgb_coarse = []
        all_rgb_fine = []
        for i in range(0, ray_targets.shape[0], batch_size):
            rgb_coarse, _, _, rgb_fine, _, _ = self.forward(
                (ray_origins, ray_directions[i: i + batch_size])
            )

            all_rgb_coarse.append(rgb_coarse)

            coarse_loss += self.loss(
                rgb_coarse[..., :3], ray_targets[i: i + batch_size, :3]
            )

            if self.model_fine is not None:
                fine_loss += self.loss(
                    rgb_fine[..., :3], ray_targets[i: i + batch_size, :3]
                )

                all_rgb_fine.append(rgb_fine)

        rgb_coarse = torch.cat(all_rgb_coarse, 0)

        coarse_loss /= batch_count
        if self.model_fine is not None:
            rgb_fine = torch.cat(all_rgb_fine, 0)
            fine_loss /= batch_count
            loss = fine_loss + coarse_loss
        else:
            rgb_fine = None
            loss = coarse_loss

        psnr = mse2psnr(loss)

        self.logger.experiment.add_image(
            "validation/rgb_coarse/" + str(batch_idx),
            cast_to_image(
                rgb_coarse[..., :3].view(self.val_dataset.H, self.val_dataset.W, 3)
            ),
            self.global_step,
        )

        if rgb_fine is not None:
            self.logger.experiment.add_image(
                "validation/rgb_fine/" + str(batch_idx),
                cast_to_image(
                    rgb_fine[..., :3].view(self.val_dataset.H, self.val_dataset.W, 3)
                ),
                self.global_step,
            )

        self.logger.experiment.add_image(
            "validation/img_target/" + str(batch_idx),
            cast_to_image(
                ray_targets[..., :3].view(self.val_dataset.H, self.val_dataset.W, 3)
            ),
            self.global_step,
        )

        output = {
            "val_loss": loss,
            "log": {
                "validation/loss": loss,
                "validation/coarse_loss": coarse_loss,
                "validation/psnr": psnr
            }
        }

        # if rgb_fine is not None:
        #     output["validation/fine_loss"] = fine_loss

        return output

    def validation_epoch_end(self, outputs):
        log_mean = { "log": {} }
        for k in outputs[0]["log"].keys():
            log_mean["log"][k] = torch.stack([ x["log"][k] for x in outputs ]).mean()

        log_mean['val_loss'] = torch.stack([x["val_loss"] for x in outputs]).mean()

        return log_mean

    def load_train_dataset(self):
        if self.cfg.dataset.type == "blender":
            self.train_dataset = BlenderDataset(self.cfg, type = DatasetType.TRAIN)
        elif self.cfg.dataset.type == "scannet":
            if not self.sensor_data:
                self.sensor_data = SensorData(
                    self.dataset_basepath / (self.dataset_basepath.name + ".sens")
                )
            self.train_dataset = ScanNetDataset(
                self.sensor_data,
                num_random_rays = self.cfg.nerf.train.num_random_rays,
                near = self.cfg.dataset.near,
                far = self.cfg.dataset.far,
                stop = 1000,  # Debug by loading only small part of the dataset
                skip_every = 10000,
                scale = self.cfg.dataset.scale_factor,
                resolution = self.cfg.dataset.resolution
            )
        elif self.cfg.dataset.type == "colmap":
            self.train_dataset = LLFFColmapDataset(
                self.cfg.dataset.basedir,
                num_random_rays = self.cfg.nerf.train.num_random_rays,
                start = 1,  # Debug by loading only small part of the dataset
                downscale_factor = self.cfg.dataset.downscale_factor
            )

    def load_val_dataset(self):
        if self.cfg.dataset.type == "blender":
            self.val_dataset = BlenderDataset(self.cfg, type = DatasetType.VALIDATION)
        elif self.cfg.dataset.type == "scannet":
            if not self.sensor_data:
                self.sensor_data = SensorData(
                    self.dataset_basepath / (self.dataset_basepath.name + ".sens")
                )
            self.val_dataset = ScanNetDataset(
                self.sensor_data,
                near = self.cfg.dataset.near,
                far = self.cfg.dataset.far,
                stop = 10,  # Debug by loading only small part of the dataset
                skip = 100,
                scale = self.cfg.dataset.scale_factor,
                resolution = self.cfg.dataset.resolution
            )
        elif self.cfg.dataset.type == "colmap":
            self.val_dataset = LLFFColmapDataset(
                self.cfg.dataset.basedir,
                num_random_rays = -1,
                stop = 2,  # Debug by loading only small part of the dataset
                downscale_factor = self.cfg.dataset.downscale_factor
            )

        self.val_num_samples = self.cfg.nerf.validation.num_samples
        if self.val_num_samples != -1:
            self.val_num_samples = max(min(len(self.val_dataset), self.val_num_samples), 1)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size = 1, shuffle = False, num_workers = self.cfg.dataset.num_workers, pin_memory = False)

        return train_dataloader

    def val_dataloader(self):
        # Use a smaller pool of random batch samples
        sampler = None
        if self.val_num_samples != -1:
            # TODO(0) https://github.com/pytorch/pytorch/pull/39214
            sampler = torch.utils.data.RandomSampler(self.val_dataset, replacement = True, num_samples = self.val_num_samples)

        # Create data loader
        val_dataloader = DataLoader(self.val_dataset, shuffle = False, batch_size = 1, sampler = sampler, num_workers = self.cfg.dataset.num_workers, pin_memory = False)

        return val_dataloader

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg.optimizer.type)(
            self.parameters(), lr = self.cfg.optimizer.lr
        )

        scheduler = getattr(torch.optim.lr_scheduler, self.cfg.scheduler.type)(
            optimizer, **self.cfg.scheduler.options
        )

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler_dict]
