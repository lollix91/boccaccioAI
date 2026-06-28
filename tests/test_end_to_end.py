"""BoccaccioAI - End-to-end smoke test with nano model.

Verifies that the full pipeline works: config loading, model creation,
forward pass, loss computation, backward pass, and generation.
Runs entirely on CPU with synthetic data -- no GPU or real datasets needed.

De Lauretis Tech
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.model.config import BoccaccioConfig
from src.model.transformer import BoccaccioForCausalLM
from src.training.scheduler import get_cosine_schedule_with_warmup


def test_config():
    """Test config creation and YAML loading."""
    print("[1/7] Testing config...", end=" ")

    config = BoccaccioConfig.nano()
    assert config.hidden_size == 256
    assert config.num_layers == 4
    assert config.head_dim == 32
    assert config.vocab_size == 32000

    config_from_yaml = BoccaccioConfig.from_yaml("configs/model.yaml", "nano")
    assert config_from_yaml.hidden_size == 256

    print("OK")
    return config


def test_model_creation(config: BoccaccioConfig):
    """Test model instantiation and parameter count."""
    print("[2/7] Testing model creation...", end=" ")

    model = BoccaccioForCausalLM(config)
    num_params = model.count_parameters()

    assert num_params > 0
    print(f"OK ({num_params:,} params, {num_params / 1e6:.2f}M)")
    return model


def test_forward_pass(model: BoccaccioForCausalLM, config: BoccaccioConfig):
    """Test forward pass with random input."""
    print("[3/7] Testing forward pass...", end=" ")

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids=input_ids)

    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)
    assert outputs["loss"] is None

    print("OK")


def test_loss_computation(model: BoccaccioForCausalLM, config: BoccaccioConfig):
    """Test loss computation with labels."""
    print("[4/7] Testing loss computation...", end=" ")

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)

    assert outputs["loss"] is not None
    assert outputs["loss"].ndim == 0
    assert outputs["loss"].item() > 0

    print(f"OK (loss={outputs['loss'].item():.4f})")


def test_backward_pass(model: BoccaccioForCausalLM, config: BoccaccioConfig):
    """Test backward pass and gradient computation."""
    print("[5/7] Testing backward pass...", end=" ")

    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    model.zero_grad()
    outputs = model(input_ids=input_ids, labels=labels)
    outputs["loss"].backward()

    has_grads = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grads = True
            break

    assert has_grads, "No gradients computed"
    print("OK")


def test_scheduler():
    """Test cosine schedule with warmup."""
    print("[6/7] Testing LR scheduler...", end=" ")

    param = torch.nn.Parameter(torch.randn(10))
    optimizer = torch.optim.AdamW([param], lr=6e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=10, total_steps=100)

    lrs = []
    for step in range(100):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[0] < lrs[9], "LR should increase during warmup"
    assert lrs[50] < lrs[10], "LR should decrease after warmup"
    assert lrs[-1] < lrs[10], "Final LR should be lower than peak"

    print("OK")


def test_dataset_and_generation(model: BoccaccioForCausalLM, config: BoccaccioConfig):
    """Test dataset loading from synthetic data and model generation."""
    print("[7/7] Testing dataset + generation...", end=" ")

    tmpdir = Path(tempfile.mkdtemp())

    try:
        # Create synthetic binary data for PreTokenizedDataset.
        num_tokens = 2048 * 10  # 10 sequences
        tokens = np.random.randint(0, config.vocab_size, size=num_tokens, dtype=np.uint16)
        data_path = tmpdir / "train.bin"
        mm = np.memmap(data_path, dtype=np.uint16, mode="w+", shape=(num_tokens,))
        mm[:] = tokens
        mm.flush()
        del mm

        from src.data.dataset import PreTokenizedDataset
        ds = PreTokenizedDataset(str(data_path), sequence_length=2048)
        assert len(ds) == 10
        sample = ds[0]
        assert sample["input_ids"].shape == (2048,)

        # Close the memmap before cleanup.
        del ds

        # Test generation (just a few tokens to verify it runs).
        model.eval()
        with torch.no_grad():
            input_ids = torch.randint(0, config.vocab_size, (1, 16))
            generated = input_ids
            for _ in range(5):
                outputs = model(generated)
                logits = outputs["logits"][:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=-1)

        assert generated.shape == (1, 21)  # 16 prompt + 5 generated
    finally:
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    print("OK")


def main():
    print("=" * 60)
    print("BoccaccioAI - End-to-End Smoke Test (nano model)")
    print("=" * 60)
    print()

    config = test_config()
    model = test_model_creation(config)
    test_forward_pass(model, config)
    test_loss_computation(model, config)
    test_backward_pass(model, config)
    test_scheduler()
    test_dataset_and_generation(model, config)

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
