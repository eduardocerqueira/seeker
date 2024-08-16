#date: 2024-08-16T16:48:57Z
#url: https://api.github.com/gists/c394b2300e6bd57e0ca1a22478072941
#owner: https://api.github.com/users/kabouzeid

# Copyright (c) Karim Abou Zeid

from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.types import STEP_OUTPUT
from typing_extensions import override


class CudaMemoryMonitor(Callback):
    def __init__(self, name="cuda_memory", prog_bar=True) -> None:
        super().__init__()
        self.name = name
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

        if not trainer.loggers:
            raise MisconfigurationException(
                "Cannot use `GpuMemoryMonitor` callback with `Trainer(logger=False)`."
            )

        device = trainer.strategy.root_device
        if not device.type == "cuda":
            raise MisconfigurationException(
                f"Cannot use `GpuMemoryMonitor` callback with {device.type} device."
            )

    def _get_and_log_device_stats(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", key: str
    ) -> None:
        max_mem_gibi = torch.cuda.max_memory_allocated(trainer.strategy.root_device) / (
            2**30
        )
        pl_module.log(
            f"{key}_{self.name}",
            max_mem_gibi,
            on_step=True,
            on_epoch=False,
            reduce_fx=torch.max,
            prog_bar=self.prog_bar,
        )

    @override
    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        torch.cuda.reset_peak_memory_stats()

    @override
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._get_and_log_device_stats(trainer, pl_module, "train")

    @override
    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        torch.cuda.reset_peak_memory_stats()

    @override
    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._get_and_log_device_stats(trainer, pl_module, "validation")

    @override
    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        torch.cuda.reset_peak_memory_stats()

    @override
    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._get_and_log_device_stats(trainer, pl_module, "test")
