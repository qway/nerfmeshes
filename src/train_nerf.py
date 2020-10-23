import os
import argparse
import torch
import models

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.profiler import AdvancedProfiler

from lightning_modules import LoggerCallback, PathParser


def main():
    torch.set_printoptions(threshold=100, edgeitems=50, precision=8, sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to (.yml) config file if running new experiment."
    )
    parser.add_argument(
        "--log-checkpoint", type=str, default=None,
        help="Training log path with the config and checkpoints to resume the experiment.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="model_last.ckpt",
        help="Resume training from the latest checkpoint by default.",
    )
    parser.add_argument(
        "--run-name", type=str, default="default",
        help="Name of the training log run"
    )
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="Amount of Gpus that should be used(In most cases leave at 1)",
    )
    parser.add_argument(
        "--precision", type=int, default=32,
        help="Full precision (32) default, half precision (16) on newer devices to speed up the training.",
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
    path_parser = PathParser()
    cfg, logger = path_parser.parse(config_args.config, config_args.log_checkpoint, config_args.run_name, config_args.checkpoint, create_logger = True)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)
    if config_args.deterministic:
        seed_everything(cfg.experiment.randomseed)

    # Create model
    model = getattr(models, cfg.experiment.model)(cfg)

    # Model checkpoint generator
    checkpoint_callback = ModelCheckpoint(filepath=path_parser.checkpoint_dir, save_top_k=3, save_last=True, verbose=True,
                                          monitor="val_loss", mode="min", prefix="model_")

    # Trainer callbacks
    logger_callback = LoggerCallback(cfg)

    # Optional profiler
    profiler = None
    if config_args.use_profiler:
        profiler = AdvancedProfiler(output_filename="report.txt", line_count_restriction=.4)

    trainer = Trainer(
        weights_summary=None,
        resume_from_checkpoint=path_parser.checkpoint_path,
        gpus=config_args.gpus,
        default_root_dir=path_parser.log_dir,
        logger=logger,
        num_sanity_val_steps=0,
        checkpoint_callback=checkpoint_callback,
        row_log_interval=1,
        log_gpu_memory=None,
        precision=config_args.precision,
        profiler=profiler,
        fast_dev_run=False,
        deterministic=config_args.deterministic,
        progress_bar_refresh_rate=0,
        accumulate_grad_batches=1,
        callbacks=[logger_callback]
    )

    if config_args.log_checkpoint is not None:
        # Add log props
        logger.experiment.add_text("description", cfg.experiment.description, 0)
        logger.experiment.add_text("config", f"\t{cfg.dump()}".replace("\n", "\n\t"), 0)
        logger.experiment.add_text("params", f"\t{ModelSummary(model, mode='full')}".replace("\n", "\n\t"), 0)

    trainer.fit(model)

    print("Done!")


if __name__ == "__main__":
    print(os.getcwd())
    main()
