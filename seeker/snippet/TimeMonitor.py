#date: 2024-08-16T16:48:18Z
#url: https://api.github.com/gists/a1e2432a51b34a106001db92e332a94b
#owner: https://api.github.com/users/kabouzeid

# Copyright (c) Karim Abou Zeid

from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override


class TimeMonitor(Callback):
    def __init__(self, prog_bar=True) -> None:
        super().__init__()
        self.prog_bar = prog_bar

    @override
    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str,
    ) -> None:
        if stage != "fit":
            return

        if not isinstance(trainer.profiler, pl.profilers.SimpleProfiler):
            raise MisconfigurationException(
                "Cannot use `TimeMonitor` callback without `Trainer(profiler='simple')`."
            )

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pl_module.log_dict(
            {
                "batch_time": trainer.profiler.recorded_durations["run_training_batch"][
                    -1
                ],
                "data_time": trainer.profiler.recorded_durations[
                    "[_TrainingEpochLoop].train_dataloader_next"
                ][-1],
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
