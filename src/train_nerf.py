import os
import argparse
import time
import torch
import torchvision
import re
import yaml
import numpy as np

from pathlib import Path
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from nerf import models, CfgNode, mse2psnr, VolumeRenderer, RaySampleInterval, SamplePDF, DensityExtractor
from lightning_modules import LoggerCallback
from model_nerf import NeRFModel, nest_dict, get_ray_batch


def load_config(configargs):
    cfg = None

    with open(configargs.config, "r") as f:
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
        "--runname",
        type = str,
        default = None,
        help = "Name of the run should be appended to the date"
    )

    configargs = parser.parse_args()

    # Read config file.
    cfg = load_config(configargs)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)
    seed_everything(cfg.experiment.randomseed)

    model = NeRFModel(cfg)

    logdir = Path("../logs") / (cfg.experiment.id)
    os.makedirs(logdir, exist_ok = True)
    runname = str(time.strftime("%y-%m-%d-%H:%M-"))
    if configargs.runname:
        logger = TensorBoardLogger(logdir, "", runname + configargs.runname)
    else:
        logger = TensorBoardLogger(logdir, "", runname)

    # with open(os.path.join(logdir, "config.yml"), "w") as f:
    #   f.write(cfg.dump())  # cfg, f, default_flow_style=False)
    checkpoint_callback = ModelCheckpoint(
        filepath = str(logdir),
        save_top_k = True,
        verbose = True,
        monitor = "val_loss",
        mode = "min",
        prefix = "",
    )

    loggerCallback = LoggerCallback(cfg)
    trainer = Trainer(
        weights_summary = None,
        gpus = configargs.gpus,
        # val_check_interval = int(cfg.experiment.train_iters / cfg.experiment.validate_every),
        check_val_every_n_epoch = cfg.experiment.validate_every / 100,
        default_root_dir = str(logdir),
        logger = logger,
        nb_sanity_val_steps = 0,
        # checkpoint_callback=checkpoint_callback,
        # profiler=True, # Activate for very simple profiling
        # fast_dev_run=True, # Activate when debugging
        max_epochs = int(cfg.experiment.train_iters / cfg.experiment.print_every),
        min_epochs = int(cfg.experiment.train_iters / cfg.experiment.print_every),
        log_gpu_memory = 'min_max',
        deterministic = True,
        progress_bar_refresh_rate = 0,
        accumulate_grad_batches = cfg.nerf.train.chunksize // cfg.nerf.train.num_random_rays,
        callbacks = [ loggerCallback ]
    )

    logger.experiment.add_text("config", f"\t{cfg.dump()}".replace("\n", "\n\t"), 0)
    # logger.experiment.add_text("params", f"\t{ModelSummary(model, trainer.weights_summary).__str__()}".replace("\n", "\n\t"), 0)

    trainer.fit(model)
