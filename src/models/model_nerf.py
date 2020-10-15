import torch
import time

from models import BaseModel
from models.model_helpers import intervals_to_ray_points
from typing import Tuple
from nerf import models, RaySampleInterval, cast_to_image
from data.data_helpers import DataBundle


def create_models(cfg) -> Tuple[torch.nn.Module, torch.nn.Module]:
    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse_type)(**cfg.models.coarse)

    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine") and cfg.models.use_fine:
        model_fine = getattr(models, cfg.models.fine_type)(**cfg.models.fine)

    return model_coarse, model_fine


class NeRFModel(BaseModel):

    def __init__(self, cfg, *args, **kwargs):
        super(NeRFModel, self).__init__(cfg, *args, **kwargs)

        self.model_coarse, self.model_fine = create_models(self.cfg)
        # samples_total =  self.cfg.nerf.train.num_coarse
        # if self.cfg.models.use_fine:
        #     samples_total = max(samples_total, samples_total + self.cfg.nerf.train.num_fine)
        self.sample_interval = RaySampleInterval()

    def get_model(self):
        return self.model_fine if self.model_fine is not None else self.model_coarse

    def forward(self, x):
        """ Does a prediction for a batch of rays.

        Args:
            x: Tensor of camera rays containing position, direction and bounds.

        Returns: Tensor with the calculated pixel value for each ray.
        """
        ray_origins, ray_directions, (near, far) = x

        # Get current configuration
        nerf_cfg = self.cfg.nerf.train if self.model_coarse.training else self.cfg.nerf.validation

        # Generating depth samples
        ray_count = ray_directions.shape[0]
        ray_intervals = self.sample_interval(nerf_cfg, ray_count, nerf_cfg.num_coarse, near, far)

        # Samples across each ray (num_rays, samples_count, 3)
        ray_points = intervals_to_ray_points(ray_intervals, ray_directions, ray_origins)

        # Expand rays to match batch size
        expanded_ray_directions = ray_directions[..., None, :].expand_as(ray_points)

        coarse_radiance_field = self.model_coarse(ray_points, expanded_ray_directions)
        rgb_coarse, _, weights, _, = self.volume_renderer(coarse_radiance_field, ray_intervals, ray_directions)

        rgb_fine, disp_fine, acc_fine, depth_fine = None, None, None, None
        if nerf_cfg.num_fine > 0 and self.model_fine is not None:
            fine_ray_intervals = self.sample_pdf(ray_intervals, weights, nerf_cfg.perturb)
            ray_points = intervals_to_ray_points(
                fine_ray_intervals, ray_directions, ray_origins
            )

            # Expand rays to match batch_size
            expanded_ray_directions = ray_directions[..., None, :].expand_as(ray_points)
            fine_radiance_field = self.model_fine(ray_points, expanded_ray_directions)

            rgb_fine, depth_fine, _, _, = self.volume_renderer(
                fine_radiance_field, fine_ray_intervals, ray_directions
            )

        return rgb_coarse, rgb_fine

    def training_step(self, ray_batch, batch_idx):
        # Unpacking bundle
        bundle = DataBundle.deserialize(ray_batch).to("cpu").to_ray_batch()

        # Manual batching, since images are expensive to be kept on GPU
        batch_size = self.cfg.nerf.train.chunksize
        batch_count = bundle.ray_targets.shape[0] / batch_size

        coarse_loss, fine_loss = 0, 0
        for i in range(0, bundle.ray_targets.shape[0], batch_size):
            # re-usable slice
            tn_slice = slice(i, i + batch_size)

            # Forward pass
            rgb_coarse, rgb_fine = self.forward((bundle.ray_origins.to(self.device), bundle.ray_directions[tn_slice].to(self.device), bundle.ray_bounds.to(self.device)))
            rgb_target = bundle.ray_targets[tn_slice].to(self.device)
            coarse_loss += self.loss(rgb_coarse, rgb_target)

            if self.model_fine is not None:
                fine_loss += self.loss(rgb_fine, rgb_target)

        #  Compute loss
        coarse_loss /= batch_count
        coarse_psnr = self.criterion_psnr(coarse_loss)

        log_vals = {
            "train/coarse_loss": coarse_loss,
            "train/coarse_psnr": coarse_psnr,
        }

        loss = coarse_loss
        if self.cfg.models.use_fine:
            fine_loss /= batch_count
            fine_psnr = self.criterion_psnr(fine_loss)

            loss += fine_loss
            log_vals = {
                **log_vals,
                "train/fine_loss": fine_loss,
                "train/fine_psnr": fine_psnr,
            }

        return {
            "loss": loss,
            "log": {
                "train/loss": loss,
                **log_vals,
                "train/lr": self.trainer.optimizers[0].param_groups[0]['lr']
            }
        }

    def validation_step(self, image_ray_batch, batch_idx):
        bundle = DataBundle.deserialize(image_ray_batch).to_ray_batch()

        # Manual batching, since images are expensive to be kept on GPU
        batch_size = self.cfg.nerf.validation.chunksize
        batch_count = bundle.ray_targets.shape[0] / batch_size

        coarse_loss = 0
        fine_loss = 0
        all_rgb_coarse = []
        all_rgb_fine = []
        for i in range(0, bundle.ray_targets.shape[0], batch_size):
            # re-usable slice
            tn_slice = slice(i, i + batch_size)

            rgb_coarse, rgb_fine = self.forward((bundle.ray_origins, bundle.ray_directions[tn_slice], bundle.ray_bounds))
            rgb_target = bundle.ray_targets[tn_slice]
            coarse_loss += self.loss(rgb_coarse, rgb_target)

            all_rgb_coarse.append(rgb_coarse)
            if self.model_fine is not None:
                fine_loss += self.loss(rgb_fine, rgb_target)

                all_rgb_fine.append(rgb_fine)

        coarse_loss /= batch_count
        loss = coarse_loss

        rgb_coarse = torch.cat(all_rgb_coarse, 0)
        self.logger.experiment.add_image(
            "validation/rgb_coarse/" + str(batch_idx),
            cast_to_image(rgb_coarse.view(bundle.hwf[0], bundle.hwf[1], 3)),
            self.global_step,
        )

        coarse_psnr = self.criterion_psnr(coarse_loss)
        log_vals = {
            "validation/coarse_loss": coarse_loss,
            "validation/coarse_psnr": coarse_psnr
        }

        if self.model_fine is not None:
            rgb_fine = torch.cat(all_rgb_fine, 0)
            self.logger.experiment.add_image(
                "validation/rgb_fine/" + str(batch_idx),
                cast_to_image(rgb_fine.view(bundle.hwf[0], bundle.hwf[1], 3)),
                self.global_step,
            )

            fine_loss /= batch_count
            loss += fine_loss

            fine_psnr = self.criterion_psnr(fine_loss)
            log_vals = {
                **log_vals,
                "validation/fine_loss": fine_loss,
                "validation/fine_psnr": fine_psnr
            }

        self.logger.experiment.add_image(
            "validation/img_target/" + str(batch_idx),
            cast_to_image(bundle.ray_targets.view(bundle.hwf[0], bundle.hwf[1], 3)),
            self.global_step,
        )

        output = {
            "val_loss": loss,
            "log": {
                **log_vals,
                "validation/loss": loss,
            }
        }

        return output
