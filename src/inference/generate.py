"""BoccaccioAI inference script -- free generation and Q&A modes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.model.config import BoccaccioConfig
from src.model.transformer import BoccaccioForCausalLM


# ------------------------------------------------------------------ #
#  Core generation function                                          #
# ------------------------------------------------------------------ #

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
    """Autoregressive token generation with temperature, top-k and top-p sampling.

    Parameters
    ----------
    model : BoccaccioForCausalLM
        The causal language model.
    input_ids : torch.Tensor
        Prompt token ids, shape ``(1, seq_len)``.
    max_new_tokens : int
        Maximum number of tokens to generate.
    temperature : float
        Softmax temperature (lower = more deterministic).
    top_k : int
        Keep only the *top_k* highest-probability tokens.
    top_p : float
        Nucleus sampling threshold (cumulative probability).
    eos_token_id : int | None
        If provided, stop generation when this token is sampled.

    Returns
    -------
    torch.Tensor
        Full sequence (prompt + generated tokens), shape ``(1, seq_len + generated)``.
    """
    generated = input_ids

    for _ in range(max_new_tokens):
        # Forward pass -- we only need the logits at the last position.
        outputs = model(generated)
        logits = outputs["logits"][:, -1, :]  # (1, vocab_size)

        # Temperature scaling.
        logits = logits / temperature

        # Top-k filtering.
        if top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            threshold = top_k_values[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Top-p (nucleus) filtering.
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Mask tokens whose cumulative probability exceeds top_p.
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

            # Scatter back to original ordering.
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        # Sample from the filtered distribution.
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

        generated = torch.cat([generated, next_token], dim=-1)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated


# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BoccaccioAI text generation (free generation / Q&A)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the saved model checkpoint directory.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer/boccaccio-32k.json",
        help="Path to the tokenizer JSON file.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/model.yaml",
        help="Path to the model config YAML file.",
    )
    parser.add_argument(
        "--config-variant",
        type=str,
        default="model",
        choices=["model", "nano"],
        help="Config variant to load ('model' or 'nano').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "qa"],
        help="Inference mode: 'generate' for free generation, 'qa' for Q&A.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt for generation mode.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Context text for Q&A mode.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question text for Q&A mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k filtering value.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Device to run inference on.",
    )

    return parser.parse_args()


# ------------------------------------------------------------------ #
#  Main                                                              #
# ------------------------------------------------------------------ #

def main() -> None:
    args = parse_args()

    # ---- device -------------------------------------------------- #
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ---- config -------------------------------------------------- #
    config = BoccaccioConfig.from_yaml(args.config_path, variant=args.config_variant)

    # ---- model --------------------------------------------------- #
    model = BoccaccioForCausalLM(config)

    checkpoint_dir = Path(args.model_dir)
    checkpoint_path = checkpoint_dir / "model.pt"
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    # ---- tokenizer ----------------------------------------------- #
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # ---- build prompt -------------------------------------------- #
    if args.mode == "qa":
        if args.context is None or args.question is None:
            raise ValueError("Q&A mode requires both --context and --question.")
        prompt = (
            f"### Contesto:\n{args.context}\n"
            f"### Domanda:\n{args.question}\n"
            f"### Risposta:\n"
        )
    else:
        if args.prompt is None:
            raise ValueError("Generation mode requires --prompt.")
        prompt = args.prompt

    # ---- tokenize ------------------------------------------------ #
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    # ---- generate ------------------------------------------------ #
    eos_token_id = tokenizer.token_to_id("</s>")

    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
    )

    # ---- decode & print ------------------------------------------ #
    output_text = tokenizer.decode(output_ids.squeeze(0).tolist())
    print(output_text)


if __name__ == "__main__":
    main()
