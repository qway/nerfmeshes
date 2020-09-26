import os
import argparse
import time
import torch
import torchvision
import re
import yaml
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from nerf import models, CfgNode, mse2psnr, VolumeRenderer, RaySampleInterval, SamplePDF, DensityExtractor
from lightning_modules import LoggerCallback
from model_nerf import NeRFModel, nest_dict, get_ray_batch


def load_config(config_args):
    cfg = None

    with open(config_args.config, "r") as f:
        cfg_dict = yaml.load(f, Loader = yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    return cfg


def create_intervals(
        ray_origin, ray_direction, bounds, point_amount = 64, lindisp = True, perturb = False
):
    near, far = bounds[..., 0, None], bounds[..., 1, None]

    point_intervals = torch.linspace(
        0.0, 1.0, point_amount, dtype = ray_origin.dtype, device = ray_origin.device,
    )
    point_intervals = point_intervals[None, :]

    # Sample in disparity space, as opposed to in depth space. Sampling in disparity is
    # nonlinear when viewed as depth sampling! (The closer to the camera the more samples)
    if not lindisp:
        point_intervals = near * (1.0 - point_intervals) + far * point_intervals
    else:
        point_intervals = 1.0 / (1.0 / near * (1.0 - point_intervals) + 1.0 / far * point_intervals)

    if perturb:
        # Get intervals between samples.
        mids = 0.5 * (point_intervals[..., 1:] + point_intervals[..., :-1])
        upper = torch.cat((mids, point_intervals[..., -1:]), dim = -1)
        lower = torch.cat((point_intervals[..., :1], mids), dim = -1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(
            point_intervals.shape, dtype = ray_origin.dtype, device = ray_origin.device
        )
        point_intervals = lower + (upper - lower) * t_rand

    return point_intervals


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
        default = "",
        help = "Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--run-name",
        type = str,
        default = "default",
        help = "Name of the run should be appended to the date"
    )

    config_args = parser.parse_args()

    # Read config file.
    cfg = load_config(config_args)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)
    seed_everything(cfg.experiment.randomseed)

    # Create model
    model = NeRFModel(cfg)

    log_dir = Path("../logs") / cfg.experiment.id
    os.makedirs(log_dir, exist_ok = True)

    log_name = str(time.strftime("%y-%m-%d-%H:%M-")) + config_args.run_name
    logger = TensorBoardLogger(log_dir, log_name, default_hp_metric = False)

    checkpoint_dir = Path(logger.log_dir) / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok = True)

    # Model checkpoint generator
    checkpoint_callback = ModelCheckpoint(filepath = str(checkpoint_dir), save_top_k = 3, save_last = True, verbose = True,
        monitor = "val_loss", mode = "min", prefix = "model")

    # Trainer callbacks
    loggerCallback = LoggerCallback(cfg)

    check_val_every_n_epoch = cfg.experiment.validate_every // len(model.train_dataset)
    trainer = Trainer(
        weights_summary = None,
        gpus = config_args.gpus,
        check_val_every_n_epoch = check_val_every_n_epoch,
        default_root_dir = str(log_dir),
        logger = logger,
        num_sanity_val_steps = 0,
        checkpoint_callback = checkpoint_callback,
        row_log_interval = 1,
        log_gpu_memory = None,
        # profiler=True, # Activate for very simple profiling
        # fast_dev_run=True, # Activate when debugging
        max_steps = cfg.experiment.train_iters,
        min_steps = cfg.experiment.train_iters,
        deterministic = True,
        progress_bar_refresh_rate = 0,
        accumulate_grad_batches = 1,
        callbacks = [loggerCallback]
    )

    logger.experiment.add_text("config", f"\t{cfg.dump()}".replace("\n", "\n\t"), 0)
    logger.experiment.add_text("params", f"\t{ModelSummary(model, mode = 'full')}".replace("\n", "\n\t"), 0)

    trainer.fit(model)
