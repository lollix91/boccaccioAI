"""
BoccaccioAI - BPE Tokenizer Training Script

Trains a byte-level BPE tokenizer on a subset of CulturaX Italian data
using the HuggingFace tokenizers library.

De Lauretis Tech
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from tokenizers import Tokenizer, models, trainers
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("boccaccio.tokenizer")

BYTES_PER_GB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer for BoccaccioAI on CulturaX Italian data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tokenizer.yaml",
        help="Path to the tokenizer YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory from config.",
    )
    parser.add_argument(
        "--corpus-size-gb",
        type=float,
        default=None,
        help="Override the corpus size (in GB) used for training.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override the vocabulary size from config.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config from %s", config_path)
    return config


def corpus_iterator(corpus_size_gb: float, source: str, lang: str):
    """Stream text samples from CulturaX until the target byte size is reached."""
    from datasets import load_dataset

    target_bytes = int(corpus_size_gb * BYTES_PER_GB)
    logger.info(
        "Streaming %s (lang=%s), target size: %.2f GB (%s bytes)",
        source, lang, corpus_size_gb, f"{target_bytes:,}",
    )

    dataset = load_dataset(source, lang, split="train", streaming=True)

    collected_bytes = 0
    pbar = tqdm(
        total=target_bytes,
        unit="B",
        unit_scale=True,
        desc="Collecting corpus",
    )

    try:
        for example in dataset:
            text = example["text"]
            text_bytes = len(text.encode("utf-8"))
            collected_bytes += text_bytes
            pbar.update(text_bytes)
            yield text

            if collected_bytes >= target_bytes:
                break
    finally:
        pbar.close()

    logger.info(
        "Corpus collection complete: %.2f GB collected",
        collected_bytes / BYTES_PER_GB,
    )


def build_tokenizer(
    vocab_size: int,
    special_tokens: list[str],
    min_frequency: int,
    show_progress: bool,
) -> tuple[Tokenizer, trainers.BpeTrainer]:
    """Construct a byte-level BPE tokenizer and its trainer."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=show_progress,
    )

    return tokenizer, trainer


def validate_tokenizer(tokenizer: Tokenizer, special_tokens: list[str]) -> None:
    """Print validation stats and fertility metrics for the trained tokenizer."""
    vocab_size = tokenizer.get_vocab_size()
    logger.info("Vocabulary size: %d", vocab_size)

    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        logger.info("  Special token: %-8s -> id %s", token, token_id)

    test_sentences = [
        "L'intelligenza artificiale sta trasformando il mondo della tecnologia.",
        "Il gatto nero dorme tranquillamente sul divano di casa.",
        "La Repubblica Italiana e' una democrazia parlamentare fondata sul lavoro.",
    ]

    logger.info("--- Fertility analysis ---")
    for sentence in test_sentences:
        encoding = tokenizer.encode(sentence)
        num_tokens = len(encoding.ids)
        num_words = len(sentence.split())
        fertility = num_tokens / num_words if num_words > 0 else 0.0

        logger.info("  Text:      %s", sentence)
        logger.info("  Tokens:    %d (words: %d, fertility: %.2f tok/word)", num_tokens, num_words, fertility)
        logger.info("  Token ids: %s", encoding.ids)
        logger.info("  Decoded:   %s", tokenizer.decode(encoding.ids))
        logger.info("")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    tok_cfg = config["tokenizer"]

    # Resolve parameters (CLI overrides take precedence over config).
    vocab_size = args.vocab_size or tok_cfg["vocab_size"]
    corpus_size_gb = args.corpus_size_gb if args.corpus_size_gb is not None else tok_cfg["training_corpus_size_gb"]
    output_dir = args.output_dir or tok_cfg["output_dir"]
    output_name = tok_cfg["output_name"]
    source = tok_cfg["training_corpus_source"]
    lang = tok_cfg["training_corpus_lang"]
    min_frequency = tok_cfg.get("min_frequency", 2)
    show_progress = tok_cfg.get("show_progress", True)

    special_tokens_cfg = tok_cfg["special_tokens"]
    special_tokens = [
        special_tokens_cfg["pad"],
        special_tokens_cfg["bos"],
        special_tokens_cfg["eos"],
        special_tokens_cfg["unk"],
    ]

    logger.info("=== BoccaccioAI Tokenizer Training ===")
    logger.info("Vocab size:   %d", vocab_size)
    logger.info("Corpus size:  %.2f GB", corpus_size_gb)
    logger.info("Output dir:   %s", output_dir)
    logger.info("Special tkns: %s", special_tokens)

    # Build tokenizer and trainer.
    tokenizer, trainer = build_tokenizer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=show_progress,
    )

    # Stream corpus and train.
    logger.info("Starting tokenizer training...")
    text_iterator = corpus_iterator(corpus_size_gb, source, lang)
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    logger.info("Training complete.")

    # Save tokenizer.
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / f"{output_name}.json"
    tokenizer.save(str(json_path))
    logger.info("Saved raw tokenizer to %s", json_path)

    # Save as HuggingFace-compatible PreTrainedTokenizerFast.
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=special_tokens_cfg["bos"],
        eos_token=special_tokens_cfg["eos"],
        unk_token=special_tokens_cfg["unk"],
        pad_token=special_tokens_cfg["pad"],
    )
    hf_dir = output_path / f"{output_name}-hf"
    hf_tokenizer.save_pretrained(str(hf_dir))
    logger.info("Saved HuggingFace tokenizer to %s", hf_dir)

    # Validate.
    validate_tokenizer(tokenizer, special_tokens)
    logger.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        sys.exit(130)
