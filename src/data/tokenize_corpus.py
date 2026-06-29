"""
BoccaccioAI - Corpus Pre-Tokenization Script (Streaming)

Pre-tokenizes filtered JSONL text shards into binary memmap format
for efficient training data loading.

Streaming architecture: reads and tokenizes one shard at a time,
writing token IDs directly to a flat binary file. Memory usage is
O(1 shard) ~500MB instead of O(full corpus) ~30GB+.

De Lauretis Tech
"""

import argparse
import json
import logging
import struct
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("boccaccio.tokenize_corpus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize filtered corpus into binary format for BoccaccioAI training."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/filtered",
        help="Directory containing filtered JSONL shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tokenized/pretrain",
        help="Directory for tokenized binary output.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="tokenizer/boccaccio-32k.json",
        help="Path to trained tokenizer JSON file.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Context window length for packed sequences.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.005,
        help="Fraction of data to reserve for validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    tokenizer_path = Path(args.tokenizer_path)
    sequence_length: int = args.sequence_length
    val_split: float = args.val_split

    logger.info("=== BoccaccioAI Corpus Pre-Tokenization ===")
    logger.info("Input dir:       %s", input_dir)
    logger.info("Output dir:      %s", output_dir)
    logger.info("Tokenizer:       %s", tokenizer_path)
    logger.info("Sequence length: %d", sequence_length)
    logger.info("Val split:       %.4f", val_split)

    # Load tokenizer.
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    logger.info("Loaded tokenizer with vocab size %d", vocab_size)

    eos_token_id = tokenizer.token_to_id("</s>")
    if eos_token_id is None:
        raise ValueError("EOS token '</s>' not found in tokenizer vocabulary")
    logger.info("EOS token id: %d", eos_token_id)

    # Verify token IDs fit in uint16.
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(
            f"Vocab size {vocab_size} exceeds uint16 max ({np.iinfo(np.uint16).max + 1}). "
            "Cannot store token IDs as uint16."
        )

    # Find shards.
    shard_files = sorted(input_dir.glob("*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")
    logger.info("Found %d JSONL shard(s) in %s", len(shard_files), input_dir)

    # ─── Streaming: tokenize shard-by-shard, write to flat binary ──
    output_dir.mkdir(parents=True, exist_ok=True)
    all_tokens_path = output_dir / "all_tokens.bin"

    logger.info("Streaming tokenization to %s ...", all_tokens_path)
    total_tokens = 0
    total_docs = 0

    # Write token IDs as uint16 directly to binary file
    with open(all_tokens_path, "wb") as bin_file:
        # Use a buffer for efficient writes
        buffer = np.empty(100_000, dtype=np.uint16)
        buf_idx = 0

        for shard_path in tqdm(shard_files, desc="Tokenizing shards"):
            with open(shard_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    text = record.get("text", "")
                    if not text:
                        continue

                    encoding = tokenizer.encode(text)
                    ids = encoding.ids
                    total_docs += 1

                    # Write token IDs + EOS
                    for tid in ids:
                        buffer[buf_idx] = tid
                        buf_idx += 1
                        if buf_idx >= len(buffer):
                            bin_file.write(buffer[:buf_idx].tobytes())
                            total_tokens += buf_idx
                            buf_idx = 0

                    # EOS token
                    buffer[buf_idx] = eos_token_id
                    buf_idx += 1
                    if buf_idx >= len(buffer):
                        bin_file.write(buffer[:buf_idx].tobytes())
                        total_tokens += buf_idx
                        buf_idx = 0

        # Flush remaining buffer
        if buf_idx > 0:
            bin_file.write(buffer[:buf_idx].tobytes())
            total_tokens += buf_idx

    logger.info("Tokenized %s documents into %s tokens", f"{total_docs:,}", f"{total_tokens:,}")

    # ─── Pack into sequences and split train/val ──────────
    num_sequences = total_tokens // sequence_length
    usable_tokens = num_sequences * sequence_length
    if usable_tokens < total_tokens:
        logger.info("Discarding last %d tokens (partial chunk)", total_tokens - usable_tokens)

    logger.info("Packing into %d sequences of length %d", num_sequences, sequence_length)

    # Memory-map the flat file for efficient splitting
    memmap = np.memmap(all_tokens_path, dtype=np.uint16, mode="r", shape=(total_tokens,))

    num_val = max(1, int(num_sequences * val_split))
    num_train = num_sequences - num_val

    train_tokens = num_train * sequence_length
    val_tokens = num_val * sequence_length

    logger.info("Train: %d sequences (%s tokens)", num_train, f"{train_tokens:,}")
    logger.info("Val:   %d sequences (%s tokens)", num_val, f"{val_tokens:,}")

    # Write train split
    train_path = output_dir / "train.bin"
    logger.info("Saving train split to %s", train_path)
    train_memmap = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(train_tokens,))
    train_memmap[:] = memmap[:train_tokens]
    train_memmap.flush()
    del train_memmap

    # Write val split
    val_path = output_dir / "val.bin"
    logger.info("Saving val split to %s", val_path)
    val_memmap = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(val_tokens,))
    val_memmap[:] = memmap[train_tokens:train_tokens + val_tokens]
    val_memmap.flush()
    del val_memmap

    del memmap

    # Save metadata
    meta_path = output_dir / "meta.json"
    meta = {
        "vocab_size": vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "sequence_length": sequence_length,
        "total_documents": total_docs,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

    # Clean up intermediate file
    all_tokens_path.unlink()
    logger.info("Cleaned up intermediate file: %s", all_tokens_path)

    # Summary.
    logger.info("=== Pre-tokenization complete ===")
    logger.info("Total tokens:      %s", f"{total_tokens:,}")
    logger.info("Train tokens:      %s", f"{train_tokens:,}")
    logger.info("Val tokens:        %s", f"{val_tokens:,}")
    logger.info("Train sequences:   %d", num_train)
    logger.info("Val sequences:     %d", num_val)
    logger.info("Sequence length:   %d", sequence_length)


if __name__ == "__main__":
    main()
