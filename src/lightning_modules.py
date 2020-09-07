import re
import torch

from tqdm.auto import tqdm
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import ProgressBar


class LoggerCallback(Callback):
    """
        Global progress bar.
    """

    def __init__(self, cfg, global_progress: bool = True, leave_global_progress: bool = True):
        super().__init__()

        self.cfg = cfg
        self.global_progress = global_progress
        self.global_desc = "{step}/{total_steps}"
        self.leave_global_progress = leave_global_progress
        self.global_pb = None

    def get_desc(self, trainer):
        step = trainer.batch_idx + 1 + trainer.current_epoch * 100
        return self.global_desc.format(step = step, total_steps = trainer.max_epochs * 100)

    def on_train_start(self, trainer, pl_module):
        self.global_pb = tqdm(
            desc = self.get_desc(trainer),
            total = trainer.max_epochs * 100,
            initial = trainer.batch_idx,
            leave = self.leave_global_progress,
            disable = not self.global_progress,
        )

    def on_fit_end(self, trainer, pl_module):
        self.global_pb.close()
        self.global_pb = None

    def format(self, value):
        return '{:.6f}'.format(value)

    def extract_acronym(self, value):
        tokens = re.split(r'[\s_/]', value)

        if len(tokens) > 2:
            value = [token[0] for token in tokens[1:]]
        else:
            value = tokens[-1]

        return "".join(value)

    def on_batch_end(self, trainer, pl_module):
        if self.global_pb is None:
            self.on_train_start(trainer, pl_module)

        # Set description
        self.global_pb.set_description(self.get_desc(trainer))

        global_step = trainer.batch_idx + 1 + trainer.current_epoch * 100
        if global_step % self.cfg.experiment.print_every == 0:
            # Gather trainer metrics
            metrics = trainer.callback_metrics

            # Filter metrics
            metrics = {name: value for name, value in metrics.items() if "train" in name}

            # Use acronyms instead for compact logging
            if self.cfg.logging.use_acronyms:
                metrics = {self.extract_acronym(name): value for name, value in metrics.items()}

            # Format trainer metrics
            metrics = " ".join([f"{name.upper()}: {self.format(value)}" for name, value in metrics.items()])

            self.global_pb.write(f"[TRAIN] Iter: {global_step} {metrics}")

        # Update progress
        self.global_pb.update(1)