import torch
import pytorch_lightning as pl

# git+https://github.com/facebookresearch/pytorch3d.git@stable
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from torch.utils.tensorboard import SummaryWriter
from mesh_nerf import extract_geometry, create_mesh

from abc import abstractmethod
from pathlib import Path
from torch.utils.data import DataLoader
from data.datasets import BlenderDataset, ScanNetDataset, LLFFDataset, DatasetType, SynthesizableDataset
from data.loaders.load_scannet import SensorData
from nerf import CfgNode, mse2psnr, VolumeRenderer, SamplePDF
from models.model_helpers import nest_dict, flatten_dict


class BaseModel(pl.LightningModule):
    def __init__(self, cfg, hparams=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hparams:
            self.cfg = CfgNode(nest_dict(hparams, sep="."))
            self.hparams = hparams
        else:
            self.cfg = cfg
            self.hparams = flatten_dict(cfg, sep=".")

        self.dataset_basepath = Path(cfg.dataset.basedir)
        self.loss = torch.nn.MSELoss()
        self.criterion_psnr = mse2psnr

        self.train_dataset, self.val_dataset = None, None
        self.volume_renderer = VolumeRenderer(
            cfg.nerf.train.radiance_field_noise_std,
            cfg.nerf.validation.radiance_field_noise_std,
            cfg.dataset.white_background,
        )

        self.sample_pdf = SamplePDF(cfg.nerf.train.num_fine)

        self.load_train_dataset()
        self.load_val_dataset()
        if self.cfg.dataset.type == "scannet":
            self.sensor_data = None

    @abstractmethod
    def get_model(self):
        pass

    def sample_points(self, points, rays=None, **kwargs):
        model = self.get_model()
        results = model(points, rays, **kwargs)
        if isinstance(results, tuple):
            return results[0]

        return results

    def validation_epoch_end(self, outputs):
        log_mean = {"log": {}}
        for k in outputs[0]["log"].keys():
            log_mean["log"][k] = torch.stack([x["log"][k] for x in outputs]).mean()

        log_mean['val_loss'] = torch.stack([x["val_loss"] for x in outputs]).mean()

        # Compute chamfer Loss
        if self.cfg.experiment.chamfer_loss and isinstance(self.val_dataset, SynthesizableDataset):
            assert self.val_dataset.target_mesh is not None, "To compute the " \
                                                             "chamfer loss, a target mesh .obj must be provided in the dataset folder"

            # Target model to query based on the grid
            model = self.get_model()

            # Read the input 3D model
            vertices, faces, _, _, _, _ = extract_geometry(model)

            # We construct a Meshes structure for the target mesh
            input_mesh = create_mesh(vertices, faces)

            # Sparse sampling
            target_samples = sample_points_from_meshes(self.val_dataset.target_mesh,
                                                       self.cfg.experiment.chamfer_sampling_size)
            input_samples = sample_points_from_meshes(input_mesh, self.cfg.experiment.chamfer_sampling_size)

            chamfer_loss, _ = chamfer_distance(target_samples, input_samples)
            log_mean["log"]["validation/chamfer_loss"] = chamfer_loss

        return log_mean

    def load_dataset(self, dataset_type):
        dataset = None
        if self.cfg.dataset.type == "blender":
            dataset = BlenderDataset(self.cfg, type=dataset_type)
        elif self.cfg.dataset.type == "scannet":
            if not self.sensor_data:
                self.sensor_data = SensorData(
                    self.dataset_basepath / (self.dataset_basepath.name + ".sens")
                )
            dataset = ScanNetDataset(
                self.sensor_data,
                num_random_rays=self.cfg.nerf.train.num_random_rays,
                near=self.cfg.dataset.near,
                far=self.cfg.dataset.far,
                stop=1000,  # Debug by loading only small part of the dataset
                skip_every=10000,
                scale=self.cfg.dataset.scale_factor,
                resolution=self.cfg.dataset.resolution
            )
        elif self.cfg.dataset.type == "colmap":
            dataset = LLFFDataset(self.cfg, type=dataset_type)

        return dataset

    def load_train_dataset(self):
        # Create dataset
        self.train_dataset = self.load_dataset(DatasetType.TRAIN)

    def train_dataloader(self):
        # Create data loader
        train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=False,
                                      num_workers=self.cfg.dataset.num_workers, pin_memory=False)

        return train_dataloader

    def load_val_dataset(self):
        # Create dataset
        self.val_dataset = self.load_dataset(DatasetType.VALIDATION)

        self.val_num_samples = self.cfg.nerf.validation.num_samples
        if self.val_num_samples != -1:
            self.val_num_samples = max(min(len(self.val_dataset), self.val_num_samples), 1)

    def val_dataloader(self):
        # Use a smaller pool of random batch samples
        sampler = None
        if self.val_num_samples != -1:
            # TODO(0) https://github.com/pytorch/pytorch/pull/39214
            sampler = torch.utils.data.RandomSampler(self.val_dataset, replacement=True,
                                                     num_samples=self.val_num_samples)

        # Create data loader
        val_dataloader = DataLoader(self.val_dataset, shuffle=False, batch_size=1, sampler=sampler,
                                    num_workers=self.cfg.dataset.num_workers, pin_memory=False)

        return val_dataloader

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg.optimizer.type)(
            self.parameters(), lr=self.cfg.optimizer.lr
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
