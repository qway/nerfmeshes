import os
import argparse
import time
import torch
import yaml
import models

from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler

from nerf import CfgNode
from models import nest_dict
from lightning_modules import LoggerCallback


class PathParser:

    def __init__(self):
        # root path
        self.root_path = None

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
            # relative path segments
            segments = os.path.normpath(path).split(os.path.sep)

            # path segments
            self.exp_name, self.log_name, self.log_version = segments[-3:]

            # self.config_path = str(Path(path) / "hparams.yaml")
            self.config_path = config_args.config
        else:
            self.config_path = config_args.config

        # Read config file.
        with open(self.config_path, "r") as file:
            cfg_dict = yaml.load(file, Loader=yaml.FullLoader)
            cfg = CfgNode(nest_dict(cfg_dict, sep="."))

        self.root_path = Path(cfg.experiment.logdir)
        if path is None:
            self.exp_name = cfg.experiment.id
            self.log_name = config_args.run_name

        # Log root experiment path
        self.log_root_dir = str(self.root_path / self.exp_name)

        # Train logger
        os.makedirs(Path(self.log_root_dir) / self.log_name, exist_ok=True)
        # TODO: default_hp_metric -> False
        logger = TensorBoardLogger(self.log_root_dir, self.log_name, version=self.log_version)

        # Logger log dir
        self.log_dir = Path(logger.log_dir)

        # Checkpoint dir
        self.checkpoint_dir = self.log_dir / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if path is not None:
            # latest best checkpoint path
            self.checkpoint_path = str(self.checkpoint_dir / "model_last.ckpt")

        return cfg, logger


def main():
    torch.set_printoptions(threshold=100, edgeitems=50, precision=8, sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="Amount of Gpus that should be used(In most cases leave at 1)",
    )
    parser.add_argument(
        "--load-checkpoint", type=str, default=None,
        help="Path to the training log to resume training.",
    )
    parser.add_argument(
        "--run-name", type=str, default="default",
        help="Name of the training log run"
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=False,
        help="Run deterministic training, useful for experimenting"
    )
    parser.add_argument(
        "--use-profiler", action="store_true", default=False,
        help="Run profiler for the training set"
    )
    config_args = parser.parse_args()

    # Log path
    parser = PathParser()
    cfg, logger = parser.parse(config_args)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)
    if config_args.deterministic:
        seed_everything(cfg.experiment.randomseed)

    # Create model
    model = getattr(models, cfg.experiment.model)(cfg)

    # Model checkpoint generator
    checkpoint_callback = ModelCheckpoint(filepath=parser.checkpoint_dir, save_top_k=3, save_last=True, verbose=True,
                                          monitor="val_loss", mode="min", prefix="model_")

    # Trainer callbacks
    logger_callback = LoggerCallback(cfg)

    # Optional profiler
    profiler = None
    if config_args.use_profiler:
        profiler = AdvancedProfiler(output_filename="report.txt", line_count_restriction=.4)

    trainer = Trainer(
        weights_summary=None,
        resume_from_checkpoint=parser.checkpoint_path,
        gpus=config_args.gpus,
        default_root_dir=parser.log_dir,
        logger=logger,
        num_sanity_val_steps=0,
        checkpoint_callback=checkpoint_callback,
        row_log_interval=1,
        log_gpu_memory=None,
        precision=32,
        profiler=profiler,
        fast_dev_run=False,
        deterministic=config_args.deterministic,
        progress_bar_refresh_rate=0,
        accumulate_grad_batches=1,
        callbacks=[logger_callback]
    )

    # Add log props
    logger.experiment.add_text("description", cfg.experiment.description, 0)
    logger.experiment.add_text("config", f"\t{cfg.dump()}".replace("\n", "\n\t"), 0)
    logger.experiment.add_text("params", f"\t{ModelSummary(model, mode='full')}".replace("\n", "\n\t"), 0)

    trainer.fit(model)

    print("Done!")


if __name__ == "__main__":
    print(os.getcwd())
    main()
