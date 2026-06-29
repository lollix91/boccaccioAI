"""BoccaccioAI - Test inferenza in locale con checkpoint Lightning.

Carica un checkpoint .ckpt di PyTorch Lightning (non model.pt) e
esegue inferenza su GPU locale (RTX 3060 12GB o simile).

Lo script estrae il model state dict dal checkpoint Lightning,
lo carica nel modello BoccaccioForCausalLM, e genera testo.

Uso:
    # Generazione libera
    python scripts/test_inference.py --checkpoint checkpoints/pretrain/last.ckpt --prompt "Il gatto"

    # Generazione con parametri custom
    python scripts/test_inference.py --checkpoint checkpoints/pretrain/last.ckpt --prompt "Roma e'" --temperature 0.5 --max-new-tokens 128

    # Modalita' Q&A (richiede fine-tuning, non funziona con pre-train)
    python scripts/test_inference.py --checkpoint checkpoints/pretrain/last.ckpt --mode qa --context "Roma e' la capitale d'Italia." --question "Qual e' la capitale?"

    # Specifica GPU
    python scripts/test_inference.py --checkpoint checkpoints/pretrain/last.ckpt --prompt "Test" --device cuda:0

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.model.config import BoccaccioConfig
from src.model.transformer import BoccaccioForCausalLM


# ─── Generation ───────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: BoccaccioForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Autoregressive generation with top-k and top-p sampling."""
    generated = input_ids
    prompt_len = input_ids.shape[1]

    for i in range(max_new_tokens):
        # Truncate to last 2048 tokens if context exceeds
        if generated.shape[1] > 2048:
            generated = generated[:, -2048:]

        outputs = model(generated)
        logits = outputs["logits"][:, -1, :]

        if temperature > 0:
            logits = logits / temperature

        # Top-k
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            threshold = top_k_values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-p
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        if temperature > 0:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated


# ─── Load checkpoint ──────────────────────────────────────────

def load_lightning_checkpoint(
    checkpoint_path: str,
    config: BoccaccioConfig,
    device: torch.device,
) -> BoccaccioForCausalLM:
    """Load a Lightning .ckpt file into BoccaccioForCausalLM."""
    print(f"Caricamento checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Lightning stores model weights under 'state_dict' with 'model.' prefix
    state_dict = ckpt.get("state_dict", ckpt)

    # Remove 'model.' prefix if present (Lightning module wraps the model)
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("model."):
            new_key = key[len("model."):]
        cleaned[new_key] = value

    model = BoccaccioForCausalLM(config)
    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    if missing:
        print(f"  Chiavi mancanti: {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  Chiavi inattese: {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Modello caricato: {param_count / 1e6:.1f}M parametri")
    print(f"  Device: {device}")

    return model


# ─── CLI ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BoccaccioAI - Test inferenza in locale con checkpoint Lightning",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path al file .ckpt di Lightning.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer/boccaccio-32k.json",
        help="Path al tokenizer JSON.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/model.yaml",
        help="Path al config YAML del modello.",
    )
    parser.add_argument(
        "--config-variant",
        type=str,
        default="model",
        choices=["model", "nano"],
        help="Variante del config.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "qa"],
        help="Modalita': generate (libera) o qa (domanda-risposta).",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt per generate mode.")
    parser.add_argument("--context", type=str, default=None, help="Contesto per qa mode.")
    parser.add_argument("--question", type=str, default=None, help="Domanda per qa mode.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Token da generare.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura sampling.")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda, cuda:0, cpu, auto).",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 60)
    print("  BoccaccioAI - Test Inferenza Locale")
    print("=" * 60)
    print()

    # Config
    config = BoccaccioConfig.from_yaml(args.config_path, variant=args.config_variant)

    # Model
    model = load_lightning_checkpoint(args.checkpoint, config, device)

    # Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    print(f"  Tokenizer: {args.tokenizer_path}")
    print()

    # Build prompt
    if args.mode == "qa":
        if not args.context or not args.question:
            print("ERRORE: modalita' qa richiede --context e --question")
            sys.exit(1)
        prompt = (
            f"### Contesto:\n{args.context}\n"
            f"### Domanda:\n{args.question}\n"
            f"### Risposta:\n"
        )
    else:
        if not args.prompt:
            print("ERRORE: modalita' generate richiede --prompt")
            sys.exit(1)
        prompt = args.prompt

    print(f"Prompt: {prompt}")
    print()

    # Tokenize
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
    print(f"Token prompt: {input_ids.shape[1]}")

    # Generate
    eos_token_id = tokenizer.token_to_id("</s>")
    print(f"Generazione ({args.max_new_tokens} token, T={args.temperature})...")
    print("-" * 60)

    t0 = time.time()
    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )
    elapsed = time.time() - t0

    # Decode only the generated part
    prompt_len = input_ids.shape[1]
    generated_ids = output_ids[0, prompt_len:].tolist()
    generated_text = tokenizer.decode(generated_ids)

    print(generated_text)
    print("-" * 60)
    print(f"Generati {len(generated_ids)} token in {elapsed:.1f}s ({len(generated_ids) / elapsed:.1f} token/s)")
    print()

    # VRAM
    if device.type == "cuda":
        vram = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"VRAM peak: {vram:.1f} GB")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
