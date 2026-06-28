# AGENTS.md -- BoccaccioAI Project Rules

De Lauretis Tech -- BoccaccioAI 1B Italian LLM

---

## Build and Run Commands

### Full pipeline (sequential)

```bash
bash scripts/01_train_tokenizer.sh
bash scripts/02_preprocess_data.sh
bash scripts/02_5_smoke_test.sh
bash scripts/03_pretrain.sh
bash scripts/04_finetune.sh
bash scripts/05_evaluate.sh
```

### Individual phases

Train tokenizer:
```bash
python -m src.tokenizer.train_tokenizer \
    --config configs/tokenizer.yaml \
    --output-dir tokenizer/ \
    --vocab-size 32000 \
    --corpus-size-gb 5
```

Data pipeline (download, filter, tokenize):
```bash
python -m src.data.download --output-dir data/raw --max-size-gb 30
python -m src.data.filter --input-dir data/raw --output-dir data/filtered --jaccard-threshold 0.85 --min-doc-length 200 --num-workers 4
python -m src.data.tokenize_corpus --input-dir data/filtered --output-dir data/tokenized/pretrain --tokenizer-path tokenizer/boccaccio-32k.json --sequence-length 2048 --val-split 0.005
```

Smoke test on local GPU (nano model, 200 steps, ~10 min):
```bash
bash scripts/02_5_smoke_test.sh
```

Pre-training:
```bash
python -m src.training.train \
    --mode pretrain \
    --model-config configs/model.yaml \
    --model-variant model \
    --training-config configs/training.yaml \
    --data-path data/tokenized/pretrain \
    --tokenizer-path tokenizer/boccaccio-32k.json
```

Fine-tuning:
```bash
python -m src.training.train \
    --mode finetune \
    --model-config configs/model.yaml \
    --model-variant model \
    --training-config configs/training.yaml \
    --data-path data/tokenized/finetune \
    --tokenizer-path tokenizer/boccaccio-32k.json \
    --resume-from checkpoints/pretrain/last.ckpt
```

Inference:
```bash
python -m src.inference.generate --model-dir checkpoints/finetune --mode generate --prompt "Testo di esempio"
python -m src.inference.generate --model-dir checkpoints/finetune --mode qa --context "Contesto." --question "Domanda?"
```

### Test command

Quick smoke test using the `nano` model variant (4 layers, 256 hidden, runs on CPU):

```bash
python -m src.training.train --mode pretrain --model-variant nano
```

This uses `configs/model.yaml` section `nano` (256 hidden, 4 layers, 8 heads) and is suitable for verifying that the training loop, data loading, and model forward pass work correctly without GPU hardware.

### Dependencies

Two requirements files exist:

- **`requirements.txt`**: Full dependencies for GPU training (Fase 3+). Includes `torch`, `lightning`, `flash-attn`. Requires CUDA.
- **`requirements-vm.txt`**: Minimal dependencies for Fasi 1-2 on CPU-only VM. Excludes GPU libraries. Used by `scripts/vm_setup.sh`.

---

## Code Conventions

- **Python version**: 3.10+ required. Use `from __future__ import annotations` in every module for PEP 604 union types.
- **Type hints**: All function signatures must have complete type annotations. Use `dict`, `list`, `tuple` (lowercase) for built-in generics. Use `X | None` instead of `Optional[X]`.
- **No emojis**: Do not use emojis anywhere in code, comments, docstrings, or documentation.
- **Logging**: Use the `logging` module (`log = logging.getLogger(__name__)`). Do not use `print()` for status messages in library code. `print()` is acceptable only in CLI entry points for final user-facing output.
- **CLI**: Use `argparse` for all command-line interfaces. Define a `parse_args()` function and a `main()` function. Guard entry points with `if __name__ == "__main__": main()`.
- **Docstrings**: Use NumPy-style docstrings with `Parameters`, `Returns`, and `Raises` sections.
- **Imports**: Group imports in this order: (1) standard library, (2) third-party, (3) local project. Separate each group with a blank line. Use absolute imports (`from src.model.config import ...`), not relative.
- **Constants**: UPPER_SNAKE_CASE for module-level constants.
- **Formatting**: Follow PEP 8. Maximum line length: 100 characters.
- **Error handling**: Raise `ValueError` for invalid configuration or arguments. Do not silently ignore errors.

---

## Directory Structure Conventions

```
configs/          YAML configuration only. One file per concern (model, tokenizer, training).
scripts/          Bash scripts numbered sequentially (01_, 02_, ...) matching pipeline phases.
src/              All Python source code. Each subdirectory is a package with __init__.py.
  src/data/       Data download, filtering, deduplication, tokenization, dataset classes.
  src/inference/  Generation and inference scripts.
  src/model/      Model architecture: config dataclass, attention, layers, full transformer.
  src/tokenizer/  Tokenizer training script.
  src/training/   Lightning module, LR scheduler, callbacks, training entry point.
data/             Runtime data directory (not committed). Subdirs: raw/, filtered/, tokenized/.
tokenizer/        Trained tokenizer artifacts (boccaccio-32k.json).
checkpoints/      Model checkpoints (not committed). Subdirs: pretrain/, finetune/.
```

- Every Python package under `src/` must have an `__init__.py`.
- Scripts are executed from the project root directory.
- All module entry points use `python -m src.<package>.<module>` invocation style.
- Configuration paths default to `configs/` and `tokenizer/` relative to the project root.

---

## Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Decoder-only transformer | Standard autoregressive LM | Simplest architecture for text generation; proven at scale by GPT, LLaMA, Mistral. |
| ~1B parameters | 2048 hidden, 24 layers | Largest model trainable on a single H100 within a reasonable time/cost budget (~15-20h, ~30-35 EUR). |
| GQA 4:1 | 16 query heads, 4 KV heads | Reduces KV cache memory by 4x during inference with minimal quality loss vs. full MHA. Follows LLaMA 2/3 approach. |
| SwiGLU activation | Intermediate size 5504 | Better training efficiency than ReLU/GELU. Size follows ~(8/3)*hidden rounded to nearest multiple of 256. |
| RoPE | theta=10000 | Relative positional encoding that generalizes better than learned absolute embeddings. Standard choice for modern LLMs. |
| RMSNorm | eps=1e-5 | Faster than LayerNorm (no mean computation), equally effective for transformers. |
| BPE tokenizer 32K | Trained on Italian text | Custom Italian vocabulary avoids the token efficiency penalty of multilingual tokenizers. 32K is a good balance between vocabulary coverage and embedding table size. |
| No dropout | 0.0 everywhere | Standard practice for LLM pre-training. Regularization comes from the large dataset and training dynamics. |
| Tied embeddings | Input = LM head | Reduces parameter count by ~65M (32K * 2048) with no quality penalty for models at this scale. |
| PyTorch Lightning | Training framework | Clean separation of model logic from training boilerplate. Built-in support for mixed precision, gradient accumulation, checkpointing, logging. |
| BF16 mixed precision | Training and inference | Better numerical range than FP16 (no loss scaling needed). Native H100 support. |
| CulturaX dataset | Italian text source | Large-scale, curated multilingual corpus with good Italian coverage. Filtered with MinHash deduplication (Jaccard 0.85) and minimum document length (200 chars). |
| Cosine LR schedule | With linear warmup | Standard practice. 750 warmup steps (~8% of total). Min LR = 10% of peak LR. |
| Single GPU training | 1x H100 | Budget-constrained design. No FSDP/DDP complexity. Model fits in memory with BF16. |
