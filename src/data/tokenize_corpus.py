"""
BoccaccioAI - Corpus Pre-Tokenization Script

Pre-tokenizes filtered JSONL text shards into binary memmap format
for efficient training data loading.

De Lauretis Tech
"""

import argparse
import json
import logging
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


def read_jsonl_shards(input_dir: Path) -> list[str]:
    """Read all JSONL shards and return a list of text documents."""
    shard_files = sorted(input_dir.glob("*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No JSONL files found in {input_dir}")

    logger.info("Found %d JSONL shard(s) in %s", len(shard_files), input_dir)
    documents: list[str] = []

    for shard_path in tqdm(shard_files, desc="Reading shards"):
        with open(shard_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if text:
                    documents.append(text)

    logger.info("Loaded %d documents from %d shard(s)", len(documents), len(shard_files))
    return documents


def tokenize_documents(
    documents: list[str],
    tokenizer: Tokenizer,
    eos_token_id: int,
) -> list[int]:
    """Tokenize all documents and concatenate with EOS separators."""
    all_token_ids: list[int] = []

    for doc in tqdm(documents, desc="Tokenizing documents"):
        encoding = tokenizer.encode(doc)
        all_token_ids.extend(encoding.ids)
        all_token_ids.append(eos_token_id)

    return all_token_ids


def pack_sequences(
    token_ids: list[int],
    sequence_length: int,
) -> np.ndarray:
    """Pack token IDs into fixed-length chunks, discarding the final partial chunk."""
    total_tokens = len(token_ids)
    num_sequences = total_tokens // sequence_length
    usable_tokens = num_sequences * sequence_length

    if usable_tokens < total_tokens:
        logger.info(
            "Discarding last %d tokens (partial chunk)",
            total_tokens - usable_tokens,
        )

    arr = np.array(token_ids[:usable_tokens], dtype=np.uint16)
    return arr.reshape(num_sequences, sequence_length)


def save_split(
    data: np.ndarray,
    output_path: Path,
) -> None:
    """Save a packed array as a flat uint16 numpy memmap file."""
    flat = data.reshape(-1)
    memmap = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=flat.shape)
    memmap[:] = flat[:]
    memmap.flush()
    del memmap


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

    # Read documents.
    documents = read_jsonl_shards(input_dir)

    # Tokenize and concatenate.
    all_token_ids = tokenize_documents(documents, tokenizer, eos_token_id)
    total_tokens = len(all_token_ids)
    logger.info("Total tokens after concatenation: %s", f"{total_tokens:,}")

    # Verify token IDs fit in uint16.
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(
            f"Vocab size {vocab_size} exceeds uint16 max ({np.iinfo(np.uint16).max + 1}). "
            "Cannot store token IDs as uint16."
        )

    # Pack into fixed-length sequences.
    packed = pack_sequences(all_token_ids, sequence_length)
    num_sequences = packed.shape[0]
    logger.info("Packed into %d sequences of length %d", num_sequences, sequence_length)

    # Split into train and val.
    num_val = max(1, int(num_sequences * val_split))
    num_train = num_sequences - num_val

    train_data = packed[:num_train]
    val_data = packed[num_train:]

    train_tokens = num_train * sequence_length
    val_tokens = num_val * sequence_length

    logger.info("Train: %d sequences (%s tokens)", num_train, f"{train_tokens:,}")
    logger.info("Val:   %d sequences (%s tokens)", num_val, f"{val_tokens:,}")

    # Save to disk.
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    meta_path = output_dir / "meta.json"

    logger.info("Saving train split to %s", train_path)
    save_split(train_data, train_path)

    logger.info("Saving val split to %s", val_path)
    save_split(val_data, val_path)

    meta = {
        "vocab_size": vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "sequence_length": sequence_length,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved metadata to %s", meta_path)

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
