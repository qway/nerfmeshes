import os
import argparse
import time
import torch
import yaml

from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from nerf import CfgNode
from model_nerf import NeRFModel, nest_dict
from lightning_modules import LoggerCallback


class PathParser:

    def __init__(self, root_path: str):
        # root path
        self.root_path = Path(root_path)

        # logger root log path
        self.log_root_dir = None

        # logger log path
        self.log_dir = None

        # logger log version
        self.exp_name, self.log_name, self.log_version = None, None, None

        # checkpoints dir
        self.checkpoint_dir = None

        # latest best checkpoint dir
        self.checkpoint_path = None

        # config path
        self.config_path = None

    def parse(self, config_args):
        path = config_args.load_checkpoint
        if path is not None:
            # path segments
            self.exp_name, self.log_name, self.log_version = os.path.normpath(path).split(os.path.sep)[-3:]

            self.config_path = str(self.root_path / path / "hparams.yaml")
        else:
            self.config_path = config_args.config

        # Read config file.
        with open(self.config_path, "r") as file:
            cfg_dict = yaml.load(file, Loader = yaml.FullLoader)
            cfg = CfgNode(nest_dict(cfg_dict, sep = "."))

        if path is None:
            self.exp_name = cfg.experiment.id
            self.log_name = str(time.strftime("%y-%m-%d-%H:%M-")) + config_args.run_name

        # Log root experiment path
        self.log_root_dir = str(self.root_path / self.exp_name)

        # Train logger
        os.makedirs(Path(self.log_root_dir) / self.log_name, exist_ok = True)
        logger = TensorBoardLogger(self.log_root_dir, self.log_name, version=self.log_version,
                                   default_hp_metric=False)

        # Logger log dir
        self.log_dir = Path(logger.log_dir)

        # Checkpoint dir
        self.checkpoint_dir = self.log_dir / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok = True)

        if path is not None:
            # latest best checkpoint path
            self.checkpoint_path = str(self.checkpoint_dir / "model-last.ckpt")

        self.checkpoint_dir = self.log_dir / "checkpoints"

        return cfg, logger


if __name__ == "__main__":
    torch.set_printoptions(threshold = 100, edgeitems = 50, precision = 8, sci_mode = False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type = str, required = True, help = "Path to (.yml) config file."
    )
    parser.add_argument(
        "--gpus",
        type = int,
        default = 1,
        help = "Amount of Gpus that should be used(In most cases leave at 1)",
    )
    parser.add_argument(
        "--load-checkpoint",
        type = str,
        default = None,
        help = "Path to the training log to resume training.",
    )
    parser.add_argument(
        "--run-name",
        type = str,
        default = "default",
        help = "Name of the run should be appended to the date"
    )

    config_args = parser.parse_args()

    # root path
    parser = PathParser("../logs")
    cfg, logger = parser.parse(config_args)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)
    seed_everything(cfg.experiment.randomseed)

    # Create model
    model = NeRFModel(cfg)

    # Model checkpoint generator
    checkpoint_callback = ModelCheckpoint(filepath = parser.checkpoint_dir, save_top_k = 3, save_last = True, verbose = True,
        monitor = "val_loss", mode = "min", prefix = "model")

    # Trainer callbacks
    loggerCallback = LoggerCallback(cfg)

    check_val_every_n_epoch = cfg.experiment.validate_every // len(model.train_dataset)
    steps_train = cfg.experiment.train_iters
    epochs_train = steps_train // len(model.train_dataset)

    trainer = Trainer(
        weights_summary = None,
        resume_from_checkpoint = parser.checkpoint_path,
        gpus = config_args.gpus,
        check_val_every_n_epoch = check_val_every_n_epoch,
        default_root_dir = parser.log_dir,
        logger = logger,
        num_sanity_val_steps = 0,
        checkpoint_callback = checkpoint_callback,
        row_log_interval = 1,
        log_gpu_memory = None,
        profiler = False,
        fast_dev_run = False,
        max_epochs = epochs_train,
        min_epochs = epochs_train,
        min_steps = steps_train,
        max_steps = steps_train,
        deterministic = True,
        progress_bar_refresh_rate = 0,
        accumulate_grad_batches = 1,
        callbacks = [ loggerCallback ]
    )

    logger.experiment.add_text("config", f"\t{cfg.dump()}".replace("\n", "\n\t"), 0)
    logger.experiment.add_text("params", f"\t{ModelSummary(model, mode = 'full')}".replace("\n", "\n\t"), 0)

    trainer.fit(model)
