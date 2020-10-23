import torch
import pytorch_lightning as pl

# git+https://github.com/facebookresearch/pytorch3d.git@stable
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from torch.utils.tensorboard import SummaryWriter
from mesh_nerf import extract_geometry, create_mesh
from torch.optim.lr_scheduler import LambdaLR
from abc import abstractmethod
from torch.utils.data import DataLoader
from data.datasets import BlenderDataset, ColmapDataset, DatasetType, SynthesizableDataset
from nerf import CfgNode, mse2psnr, VolumeRenderer
from models.model_helpers import nest_dict, flatten_dict


class BaseModel(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = CfgNode(nest_dict(cfg, sep="."))
        self.hparams = flatten_dict(cfg, sep=".")

        # Criterions
        self.loss = torch.nn.MSELoss()
        self.criterion_psnr = mse2psnr

        # Custom modules
        self.volume_renderer = VolumeRenderer(
            self.cfg.nerf.train.radiance_field_noise_std,
            self.cfg.nerf.validation.radiance_field_noise_std,
            self.cfg.dataset.white_background,
            attenuation_threshold=1e-5
        )

        # Dataset types
        self.train_dataset, self.val_dataset = None, None

    @abstractmethod
    def get_model(self):
        pass

    def setup(self, stage):
        # Load datasets
        self.load_train_dataset()
        self.load_val_dataset()

        # Set up trainer parameters
        # Min, max train steps
        steps_train = self.cfg.experiment.train_iters
        self.trainer.min_steps = steps_train
        self.trainer.max_steps = steps_train

        # Min, max train epochs
        epochs_train = steps_train // len(self.train_dataset)
        self.trainer.max_epochs = epochs_train
        self.trainer.min_epochs = epochs_train

        # Val relative to train step, as opposed to epoch
        self.trainer.check_val_every_n_epoch = self.cfg.experiment.validate_every // len(self.train_dataset)

    @abstractmethod
    def query(self, ray_batch):
        pass

    def sample_points(self, points, rays=None, **kwargs):
        # Get finest model
        model = self.get_model()

        results = model.forward(points, rays, **kwargs)
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
            vertices, faces, _, _, _, _ = extract_geometry(model, self.device, None)

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
            raise NotImplementedError
        elif self.cfg.dataset.type == "colmap":
            dataset = ColmapDataset(self.cfg, type=dataset_type)

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

    def get_scheduler(self, optimizer):
        gamma = self.cfg.scheduler.options.gamma
        step_size = self.cfg.scheduler.options.step_size

        return LambdaLR(
            optimizer,
            lr_lambda=lambda step: gamma ** (step / step_size)
        )

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.cfg.optimizer.type)(
            self.parameters(), lr=self.cfg.optimizer.lr
        )

        if hasattr(torch.optim.lr_scheduler, self.cfg.scheduler.type):
            scheduler = getattr(torch.optim.lr_scheduler, self.cfg.scheduler.type)(
                optimizer, **self.cfg.scheduler.options
            )
        else:
            scheduler = self.get_scheduler(optimizer)

        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler_dict]

    def check_early_stopping(self, rgb):
        # Current experiment
        experiment = self.cfg.experiment
        if experiment.use_early_stopping and self.global_step == experiment.early_stopping_step:
            rgb_sum = rgb.sum()
            if rgb_sum < 1e-12:
                print(f"Model is stuck in local minima, model collapsing to {rgb_sum}")
                print("Restart the training again, exiting now...")
                exit(-1)
