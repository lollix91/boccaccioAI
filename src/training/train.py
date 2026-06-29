"""BoccaccioAI - Main training entrypoint.

Invoked as ``python -m src.training.train`` from the project root.
Supports both pre-training (on binary memmap data) and instruction
fine-tuning (on JSONL data).

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import yaml
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from src.model.config import BoccaccioConfig
from src.training.lightning_module import BoccaccioLightningModule
from src.training.callbacks import TrainingStatsCallback, SaveConfigCallback
from src.data.dataset import PreTokenizedDataset, InstructionDataset

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BoccaccioAI training entrypoint")

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pretrain", "finetune"],
        help="Training mode: 'pretrain' or 'finetune'.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model.yaml",
        help="Path to model architecture YAML.",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="model",
        choices=["model", "nano"],
        help="Model variant to load from the config.",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training.yaml",
        help="Path to training hyper-parameters YAML.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the tokenized data directory.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer/boccaccio-32k.json",
        help="Path to the tokenizer JSON file.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a Lightning checkpoint to resume training from.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="boccaccio-ai",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run Weights & Biases in offline mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Dataset / DataLoader helpers
# ------------------------------------------------------------------

def _build_dataloaders(
    mode: str,
    data_path: str,
    tokenizer_path: str,
    train_config: dict,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and (optional) validation DataLoaders."""

    sequence_length: int = train_config["sequence_length"]
    batch_size: int = train_config["micro_batch_size"]
    num_workers: int = train_config.get("num_workers", 4)

    if mode == "pretrain":
        train_file = str(Path(data_path) / "train.bin")
        val_file = str(Path(data_path) / "val.bin")

        train_dataset = PreTokenizedDataset(train_file, sequence_length)
        val_dataset = PreTokenizedDataset(val_file, sequence_length)

        log.info(
            "Pre-training datasets -- train: %d sequences, val: %d sequences",
            len(train_dataset),
            len(val_dataset),
        )
    else:
        train_file = str(Path(data_path) / "train.jsonl")
        val_file = Path(data_path) / "val.jsonl"

        train_dataset = InstructionDataset(train_file, tokenizer_path, sequence_length)

        if val_file.exists():
            val_dataset = InstructionDataset(str(val_file), tokenizer_path, sequence_length)
            log.info(
                "Fine-tuning datasets -- train: %d examples, val: %d examples",
                len(train_dataset),
                len(val_dataset),
            )
        else:
            val_dataset = None
            log.info(
                "Fine-tuning dataset -- train: %d examples (no validation set)",
                len(train_dataset),
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

def _build_callbacks(
    train_config: dict,
    model_config: BoccaccioConfig,
    num_params: int,
) -> list[L.Callback]:
    """Assemble the list of Lightning callbacks."""

    checkpoint_dir = train_config.get("checkpoint_dir", "checkpoints")
    checkpoint_every = train_config.get("checkpoint_every_n_steps", 1000)
    save_top_k = train_config.get("save_top_k", 3)

    callbacks: list[L.Callback] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            every_n_train_steps=checkpoint_every,
            save_top_k=save_top_k,
            monitor="val/loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        TrainingStatsCallback(
            num_params=num_params,
            sequence_length=train_config["sequence_length"],
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        ),
        SaveConfigCallback(model_config, train_config),
    ]

    return callbacks


# ------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------

def _build_logger(
    wandb_project: str,
    wandb_offline: bool,
    mode: str,
) -> WandbLogger | TensorBoardLogger:
    """Try to create a WandB logger; fall back to TensorBoard on failure."""

    try:
        logger = WandbLogger(
            project=wandb_project,
            name=mode,
            offline=wandb_offline,
        )
        log.info("Using Weights & Biases logger (project=%s)", wandb_project)
        return logger
    except Exception:
        log.warning(
            "Failed to initialise WandB logger; falling back to TensorBoard."
        )
        return TensorBoardLogger(save_dir="logs", name=mode)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()

    # Seed everything for reproducibility.
    L.seed_everything(args.seed)

    # ── Load configs ──────────────────────────────────────────────
    model_config = BoccaccioConfig.from_yaml(args.model_config, args.model_variant)
    log.info("Model config loaded: %s (variant=%s)", args.model_config, args.model_variant)

    with open(args.training_config, "r", encoding="utf-8") as f:
        full_training_config = yaml.safe_load(f)
    train_config: dict = full_training_config[args.mode]
    log.info("Training config loaded: %s [%s]", args.training_config, args.mode)

    # ── Datasets & DataLoaders ────────────────────────────────────
    train_loader, val_loader = _build_dataloaders(
        mode=args.mode,
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        train_config=train_config,
    )

    # ── Lightning module ──────────────────────────────────────────
    model = BoccaccioLightningModule(model_config, train_config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Callbacks ─────────────────────────────────────────────────
    callbacks = _build_callbacks(train_config, model_config, num_params)

    # ── Logger ────────────────────────────────────────────────────
    logger = _build_logger(args.wandb_project, args.wandb_offline, args.mode)

    # ── Compute max_steps (pretrain) or max_epochs (finetune) ─────
    trainer_kwargs: dict = {}
    if args.mode == "pretrain":
        effective_batch_tokens = (
            train_config["micro_batch_size"]
            * train_config["gradient_accumulation_steps"]
            * train_config["sequence_length"]
        )
        max_steps = int(train_config["num_tokens"]) // effective_batch_tokens
        trainer_kwargs["max_steps"] = max_steps
        log.info(
            "Pre-training for %d steps (~%d tokens, effective batch = %d tokens/step)",
            max_steps,
            int(train_config["num_tokens"]),
            effective_batch_tokens,
        )
    else:
        max_epochs = train_config.get("num_epochs", 3)
        trainer_kwargs["max_epochs"] = max_epochs
        log.info("Fine-tuning for %d epoch(s)", max_epochs)

    # ── Trainer ───────────────────────────────────────────────────
    trainer = L.Trainer(
        accelerator=train_config.get("accelerator", "auto"),
        devices=train_config.get("devices", "auto"),
        strategy=train_config.get("strategy", "auto"),
        precision=train_config.get("precision", "bf16-mixed"),
        accumulate_grad_batches=train_config.get("gradient_accumulation_steps", 1),
        gradient_clip_val=train_config.get("gradient_clip_val", 1.0),
        val_check_interval=train_config.get("val_check_interval", None),
        log_every_n_steps=train_config.get("log_every_n_steps", 50),
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        **trainer_kwargs,
    )

    # ── Train ─────────────────────────────────────────────────────
    ckpt_path = args.resume_from if args.resume_from else None
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    # ── Save final model weights ──────────────────────────────────
    checkpoint_dir = Path(train_config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    final_path = checkpoint_dir / "model.pt"
    state_dict = model.model.state_dict()
    torch.save(state_dict, final_path)
    log.info("Final model weights saved to %s", final_path)

    log.info("Training complete.")


if __name__ == "__main__":
    main()
