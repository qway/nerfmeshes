import torch

from nerf import TreeSampling
from models import BaseModel
from models.model_helpers import intervals_to_ray_points
from typing import Dict, Any
from nerf import models, cast_to_image, RaySampleInterval
from data.data_helpers import DataBundle
from nerf.loggers import LoggerDepthProjection, LoggerTreeWeights, LoggerTree, LoggerDepthLoss


class BuFFModel(BaseModel):
    def __init__(self, cfg, *args, **kwargs):
        super(BuFFModel, self).__init__(cfg, *args, **kwargs)

        # Primary model
        self.model = getattr(models, self.cfg.models.coarse_type)(**self.cfg.models.coarse)

        # Create a tree for sampling
        self.tree = TreeSampling(self.cfg, "cuda" if torch.cuda.is_available() else "cpu")

        # Modules
        self.sampler = RaySampleInterval(self.cfg.nerf.train.num_coarse)

        # Loggers
        self.logger_depth_projection = LoggerDepthProjection(self.cfg.logging.projection_step_size, 'Point Cloud')
        self.logger_tree_weights = LoggerTreeWeights(self.tree, "Tree Memm")
        self.logger_tree = LoggerTree(self.tree, "Tree")
        self.logger_depth_loss = LoggerDepthLoss("train", self.cfg.dataset.empty)

    def get_model(self):
        return self.model

    def forward(self, x):
        """ Does a prediction for a batch of rays.

        Args:
            x: Tensor of camera rays containing position, direction and bounds.

        Returns: Tensor with the calculated pixel value for each ray.
        """
        ray_origins, ray_directions, (near, far) = x

        # Get current configuration
        nerf_cfg = self.cfg.nerf.train if self.model.training else self.cfg.nerf.validation

        # Generating depth samples
        ray_count = ray_directions.shape[0]
        ray_samples = self.sampler(nerf_cfg, ray_count, near, far)

        # Update the samples
        ray_intervals, indices, mask = self.tree.batch_ray_voxel_intersect(ray_origins, ray_directions, near, far, samples_count=nerf_cfg.num_coarse)
        ray_intervals[~mask] = ray_samples[~mask]

        # Samples across each ray (num_rays, samples_count, 3)
        ray_samples = intervals_to_ray_points(ray_intervals, ray_directions, ray_origins)

        # Expand rays to match batch size
        expanded_ray_directions = ray_directions[..., None, :].expand_as(ray_samples)

        # Model inference
        radiance_field = self.model(ray_samples, expanded_ray_directions)
        bundle = self.volume_renderer(radiance_field, ray_intervals, ray_directions)

        if self.training:
            # Perform ray batch integration into the tree
            self.tree.ray_batch_integration(self.global_step, indices[mask], bundle.weights[mask].detach(), bundle.mask_weights[mask].detach())

        return bundle

    def query(self, ray_batch):
        # Fine query
        return self.forward(ray_batch)

    def training_step(self, ray_batch, batch_idx):
        # Unpacking bundle
        bundle = DataBundle.deserialize(ray_batch).to_ray_batch()
        logger = self.logger.experiment

        # Forward pass
        output_bundle = self.forward((bundle.ray_origins, bundle.ray_directions, bundle.ray_bounds))

        # Early stopping if the scene data is too sparse
        self.check_early_stopping(output_bundle.rgb_map)

        # Compute losses
        loss = self.loss(output_bundle.rgb_map, bundle.ray_targets)
        psnr = self.criterion_psnr(loss)

        log_vals = {
            "train/loss": loss,
            "train/psnr": psnr,
        }

        # Depth consideration
        log_vals = self.logger_depth_loss.tick(log_vals, output_bundle.rgb_map, bundle.ray_targets, output_bundle.depth_map, bundle.target_depth)

        # Loggers
        self.logger_depth_projection.tick(logger, self.global_step, bundle.ray_origins, bundle.ray_directions, output_bundle.depth_map, bundle.target_depth)
        self.logger_tree_weights.tick(logger, self.global_step)

        # Tree consolidation
        if self.tree.ticked(self.global_step):
            self.tree.consolidate()

        # Log tree structure after consolidation
        self.logger_tree.tick(logger, self.global_step)

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

        loss = 0.
        rgb_chunks = []
        for i in range(0, bundle.ray_targets.shape[0], batch_size):
            # re-usable slice
            tn_slice = slice(i, i + batch_size)

            output_bundle = self.forward((bundle.ray_origins, bundle.ray_directions[tn_slice], bundle.ray_bounds))
            loss += self.loss(output_bundle.rgb_map, bundle.ray_targets[tn_slice])
            rgb_chunks.append(output_bundle.rgb_map)

        # Mean loss
        loss /= batch_count

        rgb_map = torch.cat(rgb_chunks, 0)
        self.logger.experiment.add_image(
            "validation/rgb_coarse/" + str(batch_idx),
            cast_to_image(rgb_map.view(bundle.hwf[0], bundle.hwf[1], 3)),
            self.global_step,
        )

        psnr = self.criterion_psnr(loss)
        log_vals = {
            "validation/loss": loss,
            "validation/psnr": psnr
        }

        self.logger.experiment.add_image(
            "validation/img_target/" + str(batch_idx),
            cast_to_image(
                bundle.ray_targets.view(bundle.hwf[0], bundle.hwf[1], 3)
            ),
            self.global_step,
        )

        output = {
            "val_loss": loss,
            "log": log_vals
        }

        return output

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['tree'] = self.tree.serialize()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.tree.deserialize(checkpoint['tree'])
