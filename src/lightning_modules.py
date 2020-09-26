import re

from tqdm import tqdm
from pytorch_lightning.callbacks import Callback


class LoggerCallback(Callback):
    """
        Global progress bar.
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

    def extract_metrics(self, trainer, step = -1, type = "train"):
        # Gather trainer metrics
        metrics = trainer.callback_metrics

        # Filter metrics
        metrics = {name: value for name, value in metrics.items() if type in name}

        if step != -1:
            # Print to the logger
            trainer.logger.log_metrics(metrics, step = step)

        # Use acronyms instead for compact logging
        if self.cfg.logging.use_acronyms:
            metrics = {self.extract_acronym(name): value for name, value in metrics.items()}

        # Format trainer metrics
        metrics = " ".join([f"{name.upper()}: {self.format(value)}" for name, value in metrics.items()])

        return metrics

    def on_fit_start(self, trainer, pl_module):
        """Called when fit begins"""
        self.global_pb = tqdm(
            desc = "TRAIN",
            total = trainer.max_steps,
            position = 0,
            initial = trainer.batch_idx,
            leave = self.leave_global_progress,
            disable = not self.global_progress,
        )

        self.val_pb = tqdm(
            desc = "VALID",
            total = pl_module.val_num_samples,
            position = 1,
            initial = 0,
            leave = self.leave_global_progress,
            disable = not self.global_progress,
        )

    def on_fit_end(self, trainer, pl_module):
        """Called when the trainer initialization end."""
        self.global_pb.close()
        self.val_pb.close()

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch begins."""
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
