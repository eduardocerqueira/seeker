#date: 2024-11-05T17:08:32Z
#url: https://api.github.com/gists/69f73a763ec89d4609a657e2f2691b8d
#owner: https://api.github.com/users/AlessandroW

"""Cross-validation Lightning CLI with MLFlow logging
Tested with
- mlflow = "2.16.2"
- lightning = "2.4.0"
License MIT

Core Idea:
- DataModule expects a `val_filename: str` to create the validation data loader.
- If you provide a multiple val_filenames (`val_filename: Union[list, str]`) train a model with each val_filename
- Use MLFlow to combine all runs under a parent run (nested runs)
- Log cross-validation metrics, e.g., mean and std for each metric, in the parent run
- The core idea works without MLFlow, just remove the respective code.
"""
from pathlib import Path
import tempfile
import logging
from time import time

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning import Trainer, LightningModule
import numpy as np

LOG = logging.getLogger(__name__)


class MLFlowSaveConfigCallback(SaveConfigCallback):
    def __init__(self, parser, config, config_filename='config.yaml', overwrite=False, multifile=False):
        super().__init__(parser, config, config_filename,
                         overwrite, multifile, save_to_log_dir=False)

    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.is_global_zero:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / 'config.yaml'
                self.parser.save(
                    self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
                )
                trainer.logger.experiment.log_artifact(local_path=config_path,
                                                       run_id=trainer.logger.run_id)


class MyLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        self.parent_run_id = None
        self.child_runs = []
        self.val_filenames = None
        super().__init__(*args, **kwargs)

    def before_fit(self):
        def fit():
            # HACK cross-validation
            # copy-paste from cli.py
            subcommand = "fit"  # `before_{subcommand}` was called
            default = getattr(self.trainer, subcommand)
            fn = getattr(self, subcommand, default)
            fn_kwargs = self._prepare_subcommand_kwargs(subcommand)
            fn(**fn_kwargs)

        # only called if val_filename is a list (dm.setup() expects a str)
        if isinstance(self.config.fit.data.init_args.val_filename, list):
            # Create run, if necessary
            self.trainer.logger.experiment
            self.parent_run_id = self.trainer.logger._run_id
            self.val_filenames = self.config.fit.data.init_args.val_filename

            # HACK set subcommand to skip actual fit call, see
            # https://github.com/Lightning-AI/pytorch-lightning/blob/3627c5bfac704d44c0d055a2cdf6f3f9e3f9e8c1/src/lightning/pytorch/cli.py#L698
            self.subcommand = None
            status = "FINISHED"
            try:
                self.config.fit.data.init_args.val_filename = self.val_filenames[0]
                assert self.parent_run_id is not None
                for i, val_filename in enumerate(self.val_filenames, start=1):
                    LOG.info(
                        f"Running fold {i}/{len(self.val_filenames)}, {val_filename}")
                    self.datamodule.val_filename = val_filename
                    # HACK re-create a run
                    self.trainer.logger._initialized = False
                    self.trainer.logger._run_id = None
                    self.trainer.logger.experiment
                    # ...and make it a nested run
                    self.trainer.logger.experiment.set_tag(
                        self.trainer.logger._run_id, "mlflow.parentRunId", self.parent_run_id)
                    # ...keep track of them
                    self.child_runs.append(self.trainer.logger._run_id)
                    # reset global step counter, otherwise it won't start the training again
                    # taken from training_epoch_loop.py -> global_step
                    lightning_module = self.trainer.lightning_module
                    if lightning_module is None or lightning_module.automatic_optimization:
                        self.trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.reset()
                    else:
                        self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.reset()
                    # reset epoch counter
                    self.trainer.fit_loop.epoch_progress.reset()
                    # ...call fit for each fold
                    fit()
            # catch exception to stop the parent run
            except Exception as e:
                LOG.exception(e)
                status = "FAILED"
            child_runs = [self.trainer.logger._mlflow_client.get_run(
                run) for run in self.child_runs]
            # process the metrics, e.g., calculate mean and std.
            if child_runs:
                from mlflow.entities import Metric
                metrics_list: list[Metric] = []
                timestamp_ms = int(time() * 1000)
                for metric in child_runs[0].data.metrics:
                    values = [run.data.metrics[metric]
                              for run in child_runs if metric in run.data.metrics]
                    metrics_list.append(
                        Metric(key=f"{metric}_mean", value=np.mean(values), timestamp=timestamp_ms, step=0))
                    metrics_list.append(
                        Metric(key=f"{metric}_std", value=np.std(values), timestamp=timestamp_ms, step=0))
                self.trainer.logger._mlflow_client.log_batch(
                    run_id=self.parent_run_id, metrics=metrics_list, **self.trainer.logger._log_batch_kwargs)
            self.trainer.logger._mlflow_client.set_terminated(
                self.parent_run_id, status)


def main():
    MyLightningCLI(subclass_mode_model=True,
                   # Save actual config for reproducability
                   save_config_callback=MLFlowSaveConfigCallback)


if __name__ == "__main__":
    main()
