"""Custom Lightning callbacks for the BoccaccioAI training loop."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import lightning as L
import yaml

log = logging.getLogger(__name__)


class TrainingStatsCallback(L.Callback):
    """Log throughput and estimated model-FLOPs utilisation (MFU).

    Parameters
    ----------
    num_params:
        Total number of trainable model parameters.  Used when computing
        the theoretical FLOPs per token (``6 * num_params``).
    peak_flops:
        Peak FLOPs of the accelerator (e.g. 312e12 for an A100-80 GB in
        BF16).  Set to ``None`` to skip MFU logging.
    sequence_length:
        Sequence length used during training.
    gradient_accumulation_steps:
        Number of gradient-accumulation steps.
    """

    def __init__(
        self,
        num_params: int,
        sequence_length: int,
        gradient_accumulation_steps: int = 1,
        peak_flops: float | None = None,
    ) -> None:
        super().__init__()
        self.num_params = num_params
        self.sequence_length = sequence_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.peak_flops = peak_flops
        self._step_start: float | None = None

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._step_start = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self._step_start is None:
            return

        elapsed = time.perf_counter() - self._step_start
        if elapsed <= 0:
            return

        batch_size = batch["input_ids"].shape[0]
        tokens_processed = (
            batch_size
            * self.sequence_length
            * self.gradient_accumulation_steps
        )
        tokens_per_sec = tokens_processed / elapsed

        pl_module.log(
            "perf/tokens_per_sec",
            tokens_per_sec,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        if self.peak_flops is not None and self.peak_flops > 0:
            # Approximate FLOPs per token: 6 * N (forward + backward).
            model_flops = 6 * self.num_params * tokens_processed / elapsed
            mfu = model_flops / self.peak_flops
            pl_module.log(
                "perf/mfu",
                mfu,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )


class SaveConfigCallback(L.Callback):
    """Persist model and training configuration at the start of training.

    Both configs are written as YAML files inside the trainer's log
    directory so that every run is fully reproducible.

    Parameters
    ----------
    model_config:
        A :class:`BoccaccioConfig` dataclass (or any object whose
        ``__dict__`` is serialisable).
    training_config:
        A plain dictionary with training hyper-parameters.
    """

    def __init__(
        self,
        model_config: Any,
        training_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config

    def on_train_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        save_dir = Path(trainer.log_dir) if trainer.log_dir else None
        if save_dir is None:
            log.warning("No log directory available; skipping config save.")
            return

        save_dir.mkdir(parents=True, exist_ok=True)

        model_cfg_path = save_dir / "model_config.yaml"
        training_cfg_path = save_dir / "training_config.yaml"

        # Serialise model config (works for dataclasses and plain dicts).
        model_dict = (
            self.model_config.__dict__
            if hasattr(self.model_config, "__dict__")
            else dict(self.model_config)
        )
        model_cfg_path.write_text(
            yaml.dump(model_dict, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        log.info("Saved model config to %s", model_cfg_path)

        training_cfg_path.write_text(
            yaml.dump(
                dict(self.training_config),
                default_flow_style=False,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        log.info("Saved training config to %s", training_cfg_path)
