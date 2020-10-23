import torch

from models import BaseModel
from models.model_helpers import intervals_to_ray_points
from typing import Tuple
from nerf import models, cast_to_image, SamplePDF, RaySampleInterval
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

        # Create NeRF models
        self.model_coarse, self.model_fine = create_models(self.cfg)

        # Custom modules
        self.sample_pdf = SamplePDF(self.cfg.nerf.train.num_fine)
        self.sampler = RaySampleInterval(self.cfg.nerf.train.num_coarse)

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
        ray_intervals = self.sampler(nerf_cfg, ray_count, near, far)

        # Samples across each ray (num_rays, samples_count, 3)
        ray_points = intervals_to_ray_points(ray_intervals, ray_directions, ray_origins)

        # Expand rays to match batch size
        expanded_ray_directions = ray_directions[..., None, :].expand_as(ray_points)

        # Coarse inference
        coarse_radiance_field = self.model_coarse(ray_points, expanded_ray_directions)
        coarse_bundle = self.volume_renderer(coarse_radiance_field, ray_intervals, ray_directions)

        fine_bundle = None
        if self.model_fine is not None:
            fine_ray_intervals = self.sample_pdf(ray_intervals, coarse_bundle.weights, nerf_cfg.perturb)
            ray_points = intervals_to_ray_points(
                fine_ray_intervals, ray_directions, ray_origins
            )

            # Expand rays to match batch_size
            expanded_ray_directions = ray_directions[..., None, :].expand_as(ray_points)

            # Fine inference
            fine_radiance_field = self.model_fine(ray_points, expanded_ray_directions)
            fine_bundle = self.volume_renderer(fine_radiance_field, fine_ray_intervals, ray_directions)

        return coarse_bundle, fine_bundle

    def query(self, ray_batch):
        # Fine query
        coarse_bundle, fine_bundle = self.forward(ray_batch)
        if fine_bundle is not None:
            return fine_bundle

        return coarse_bundle

    def training_step(self, ray_batch, batch_idx):
        # Unpacking bundle
        bundle = DataBundle.deserialize(ray_batch).to("cpu").to_ray_batch()

        # Manual batching, since images are expensive to be kept on GPU
        batch_size = self.cfg.nerf.train.chunksize
        batch_count = bundle.ray_targets.shape[0] / batch_size

        coarse_loss, fine_loss = 0, 0
        for i in range(0, bundle.ray_targets.shape[0], batch_size):
            # Re-usable slice, maybe generator to use instead
            tn_slice = slice(i, i + batch_size)

            if self.cfg.dataset.use_ndc:
                ray_origins = bundle.ray_origins[tn_slice].to(self.device)
            else:
                ray_origins = bundle.ray_origins.to(self.device)

            rgb_target = bundle.ray_targets[tn_slice].to(self.device)

            # Ray batch
            ray_batch = (ray_origins, bundle.ray_directions[tn_slice].to(self.device), bundle.ray_bounds.to(self.device))

            # Forward pass
            coarse_bundle, fine_bundle = self.forward(ray_batch)
            coarse_loss += self.loss(coarse_bundle.rgb_map, rgb_target)

            # Early stopping if the scene data is too sparse
            self.check_early_stopping(coarse_bundle.rgb_map)
            if self.model_fine is not None:
                fine_loss += self.loss(fine_bundle.rgb_map, rgb_target)

                # Early stopping if the scene data is too sparse
                self.check_early_stopping(fine_bundle.rgb_map)

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
            rgb_target = bundle.ray_targets[tn_slice]
            if self.cfg.dataset.use_ndc:
                ray_origins = bundle.ray_origins[tn_slice]
            else:
                ray_origins = bundle.ray_origins

            coarse_bundle, fine_bundle = self.forward((ray_origins, bundle.ray_directions[tn_slice], bundle.ray_bounds))
            coarse_loss += self.loss(coarse_bundle.rgb_map, rgb_target)

            all_rgb_coarse.append(coarse_bundle.rgb_map)
            if self.model_fine is not None:
                fine_loss += self.loss(fine_bundle.rgb_map, rgb_target)

                all_rgb_fine.append(fine_bundle.rgb_map)

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
