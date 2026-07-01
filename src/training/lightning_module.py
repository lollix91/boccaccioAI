"""PyTorch Lightning module for training the BoccaccioAI model."""

from __future__ import annotations

import logging
from typing import Any

import torch
import lightning as L

from src.model.config import BoccaccioConfig
from src.model.transformer import BoccaccioForCausalLM
from src.training.scheduler import get_cosine_schedule_with_warmup

log = logging.getLogger(__name__)


class BoccaccioLightningModule(L.LightningModule):
    """Lightning wrapper around :class:`BoccaccioForCausalLM`.

    Parameters
    ----------
    config:
        Model architecture configuration.
    training_config:
        Dictionary with training hyper-parameters.  Expected keys include
        ``learning_rate``, ``weight_decay``, ``beta1``, ``beta2``,
        ``warmup_steps``, ``num_tokens``, ``micro_batch_size``,
        ``gradient_accumulation_steps``, ``sequence_length``, and
        optionally ``compile_model``.
    """

    def __init__(self, config: BoccaccioConfig, training_config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.training_config = training_config

        self.model = BoccaccioForCausalLM(config)

        if training_config.get("compile_model", False):
            self.model = torch.compile(self.model)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info("Trainable parameters: %s (%.2fM)", num_params, num_params / 1e6)

    # ------------------------------------------------------------------
    # Training / validation steps
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/ppl", torch.exp(loss), prog_bar=True, on_step=True, on_epoch=False)
        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val/ppl", torch.exp(loss), prog_bar=True, on_step=True, on_epoch=False)
        return loss

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Biases, embedding weights, and RMSNorm weights do not get
            # weight-decay; only Linear *weight* matrices are decayed.
            if name.endswith(".bias"):
                no_decay_params.append(param)
            elif "embed_tokens" in name:
                no_decay_params.append(param)
            elif "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.training_config["weight_decay"]},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.training_config["learning_rate"],
            betas=(
                self.training_config["beta1"],
                self.training_config["beta2"],
            ),
        )

        # Calculate total steps for cosine schedule.
        # Pre-training uses num_tokens for 1 epoch.
        # Fine-tuning uses num_tokens * num_epochs.
        batch_tokens = (
            self.training_config["micro_batch_size"]
            * self.training_config["gradient_accumulation_steps"]
            * self.training_config["sequence_length"]
        )

        num_tokens = self.training_config.get("num_tokens", 0)
        num_epochs = self.training_config.get("num_epochs", 1)

        if num_tokens > 0:
            total_steps = (num_tokens * num_epochs) // batch_tokens
        else:
            total_steps = 5000  # fallback

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=self.training_config["warmup_steps"],
            total_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
