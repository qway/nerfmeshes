import os
import re
import yaml

from pathlib import Path
from nerf import CfgNode
from models.model_helpers import nest_dict
from tqdm import tqdm

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger


class LoggerCallback(Callback):
    """
        Global progress bar, manages two progress bars for both train & validation steps.
    """

    def __init__(self, cfg, global_progress: bool = True, leave_global_progress: bool = True):
        super().__init__()

        self.cfg = cfg
        self.global_progress = global_progress
        self.global_desc = "{step}/{total_steps}"
        self.val_desc = "{val_step}/{val_total_steps}"
        self.leave_global_progress = leave_global_progress
        self.global_pb, self.val_pb = None, None

    def get_global_step(self, trainer):
        # Batch count total
        train_batch_count = len(trainer.train_dataloader)

        # Global step
        global_step = trainer.batch_idx + 1 + trainer.current_epoch * train_batch_count

        return global_step

    def format(self, value):
        return '{:.6f}'.format(value)

    def extract_acronym(self, value):
        tokens = re.split(r'[\s_/]', value)

        if len(tokens) > 2:
            value = [token[0] for token in tokens[1:]]
        else:
            value = tokens[-1]

        return "".join(value)

    def extract_metrics(self, trainer, step=-1, type="train"):
        # Gather trainer metrics
        metrics = trainer.callback_metrics

        # Filter metrics
        metrics = {name: value for name, value in metrics.items() if type in name}

        if step != -1:
            # Print to the logger
            trainer.logger.log_metrics(metrics, step=step)

        # Use acronyms instead for compact logging
        if self.cfg.logging.use_acronyms:
            metrics = {self.extract_acronym(name): value for name, value in metrics.items()}

        # Format trainer metrics
        metrics = " ".join([f"{name.upper()}: {self.format(value)}" for name, value in metrics.items()])

        return metrics

    def init_trackers(self, trainer, pl_module):
        if self.global_pb is None:
            # Initialize trainer tracker
            self.global_pb = tqdm(
                desc="TRAIN",
                total=trainer.max_steps,
                position=0,
                initial=trainer.global_step,
                leave=self.leave_global_progress,
                disable=not self.global_progress,
            )

        if self.val_pb is None:
            # Initialize valid tracker
            self.val_pb = tqdm(
                desc="VALID",
                total=pl_module.val_num_samples,
                position=1,
                initial=0,
                leave=self.leave_global_progress,
                disable=not self.global_progress,
            )

    def on_sanity_check_start(self, trainer, pl_module):
        self.init_trackers(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        """Called when the trainer initialization end."""
        self.global_pb.close()
        self.val_pb.close()

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch begins."""
        self.init_trackers(trainer, pl_module)
        self.global_pb.unpause()

    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the training batch ends."""
        # Check if print individual step logs
        global_step = self.get_global_step(trainer)
        if global_step % self.cfg.experiment.print_every == 0:
            # Gather trainer metrics
            metrics = self.extract_metrics(trainer, global_step, "train")

            self.global_pb.write(f"[TRAIN] Iter: {global_step} {metrics}")

        # Update progress
        self.global_pb.update(1)

    def on_validation_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        global_step = self.get_global_step(trainer)

        self.val_pb.reset()
        self.val_pb.write("  [VAL] =======> Iter: " + str(global_step))

    def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        # Update progress
        self.val_pb.update(1)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the epoch ends."""

        # Gather trainer metrics
        metrics = self.extract_metrics(trainer, -1, "validation")

        self.val_pb.write(metrics)

    def on_validation_end(self, trainer, pl_module):
        """Called when the validation loop ends."""
        self.val_pb.clear()


class PathParser:
    """
        Path parser that returns experiment configuration props and manage paths for model's persistence.
    """

    LOG_RUN_NAME = "default"
    CHECKPOINT_NAME_LAST = "model_last.ckpt"

    def __init__(self):
        # root path
        self.root_path = None

        # config path
        self.config_path = None

        # logger root log path
        self.log_root_dir, self.log_dir = None, None

        # logger log version
        self.exp_name, self.log_name, self.log_version = None, None, None

        # checkpoints dir, latest best checkpoint path
        self.checkpoint_dir, self.checkpoint_path = None, None

    def parse(self, config_path = None, log_path = None, run_name = LOG_RUN_NAME, checkpoint_name = CHECKPOINT_NAME_LAST, create_logger = False):
        assert ((config_path is not None) != (log_path is not None)), \
            "Either config or log with checkpoints must be provided, append option --help for more information."

        if log_path is not None:
            # relative path segments
            segments = os.path.normpath(log_path).split(os.path.sep)

            # path segments
            self.exp_name, self.log_name, self.log_version = segments[-3:]

            # Logger log dir
            self.log_dir = Path(log_path)

            # Config path
            self.config_path = str(self.log_dir / TensorBoardLogger.NAME_HPARAMS_FILE)
        else:
            self.config_path = config_path

        # Read config file.
        with open(self.config_path, "r") as file:
            cfg_dict = yaml.load(file, Loader=yaml.FullLoader)
            cfg = CfgNode(nest_dict(cfg_dict, sep="."))

        self.root_path = Path(cfg.experiment.logdir)
        if log_path is None:
            self.exp_name = cfg.experiment.id
            self.log_name = run_name

        # Log root experiment path
        self.log_root_dir = str(self.root_path / self.exp_name)

        # Train logger
        logger = None
        if create_logger:
            os.makedirs(Path(self.log_root_dir) / self.log_name, exist_ok=True)
            # Create logger instance
            logger = TensorBoardLogger(self.log_root_dir, self.log_name, version=self.log_version)
            print("Logger initiated...")

            # Logger log dir, if conflicting log_version
            self.log_dir = Path(logger.log_dir)

        print(f"Current log dir {self.log_dir}")
        # Checkpoint dir
        self.checkpoint_dir = self.log_dir / "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if log_path is not None:
            # latest best checkpoint path
            self.checkpoint_path = str(self.checkpoint_dir / checkpoint_name)

        return cfg, logger
