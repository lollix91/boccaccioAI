"""
BoccaccioAI - Data Filtering / Cleaning Pipeline

Takes raw Italian text from CulturaX JSONL shards and applies three sequential
quality-filtering stages to produce a clean dataset:

  1. Heuristic filtering  (length, alpha ratio, punctuation, duplicates)
  2. MinHash LSH deduplication  (near-duplicate removal via datasketch)
  3. Perplexity filtering  (optional, requires a KenLM model)

De Lauretis Tech
"""

from __future__ import annotations

import argparse
import json
import logging
import string
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("boccaccio.data.filter")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and clean raw CulturaX Italian JSONL shards for BoccaccioAI.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw JSONL shards (default: data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/filtered",
        help="Directory for filtered output shards (default: data/filtered).",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=0.85,
        help="MinHash LSH Jaccard similarity threshold (default: 0.85).",
    )
    parser.add_argument(
        "--min-doc-length",
        type=int,
        default=200,
        help="Minimum document length in characters (default: 200).",
    )
    parser.add_argument(
        "--max-doc-length",
        type=int,
        default=100_000,
        help="Maximum document length in characters (default: 100000).",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of permutations for MinHash signatures (default: 128).",
    )
    parser.add_argument(
        "--kenlm-model",
        type=str,
        default=None,
        help="Path to a KenLM .arpa/.binary model file. If omitted the "
             "perplexity filter stage is skipped.",
    )
    parser.add_argument(
        "--max-perplexity",
        type=float,
        default=1000.0,
        help="Maximum perplexity threshold (default: 1000).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for heuristic filtering (default: 4).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Stage 1 -- Heuristic Filtering
# ---------------------------------------------------------------------------

_PUNCTUATION_SET = set(string.punctuation)


def _passes_heuristic(
    doc: dict[str, Any],
    min_doc_length: int,
    max_doc_length: int,
) -> bool:
    """Return True if *doc* passes every heuristic quality check."""
    text: str = doc.get("text", "")
    length = len(text)

    # Length bounds.
    if length < min_doc_length or length > max_doc_length:
        return False

    # Alphabetic character ratio >= 0.70.
    alpha_count = sum(1 for ch in text if ch.isalpha())
    if alpha_count / length < 0.70:
        return False

    # Punctuation density <= 0.15.
    punct_count = sum(1 for ch in text if ch in _PUNCTUATION_SET)
    if punct_count / length > 0.15:
        return False

    # Repeated-line ratio <= 0.30.
    lines = text.splitlines()
    if lines:
        unique_lines = set(lines)
        duplicate_ratio = 1.0 - len(unique_lines) / len(lines)
        if duplicate_ratio > 0.30:
            return False

    # Average line length >= 20 characters.
    if lines:
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if avg_line_length < 20:
            return False

    return True


def heuristic_filter(
    docs: list[dict[str, Any]],
    min_doc_length: int,
    max_doc_length: int,
    num_workers: int,
) -> list[dict[str, Any]]:
    """Apply heuristic quality filters in parallel and return kept documents."""
    logger.info(
        "Stage 1 - Heuristic filtering on %s documents (workers=%d) ...",
        f"{len(docs):,}",
        num_workers,
    )

    check_fn = partial(
        _passes_heuristic,
        min_doc_length=min_doc_length,
        max_doc_length=max_doc_length,
    )

    with Pool(processes=num_workers) as pool:
        verdicts = list(
            tqdm(
                pool.imap(check_fn, docs, chunksize=512),
                total=len(docs),
                desc="Heuristic filter",
            )
        )

    kept = [doc for doc, ok in zip(docs, verdicts) if ok]
    removed = len(docs) - len(kept)
    logger.info(
        "Stage 1 complete: kept %s, removed %s",
        f"{len(kept):,}",
        f"{removed:,}",
    )
    return kept


# ---------------------------------------------------------------------------
# Stage 2 -- MinHash LSH Deduplication
# ---------------------------------------------------------------------------


def _word_ngrams(text: str, n: int = 5) -> list[str]:
    """Extract word-level n-grams from *text*."""
    words = text.split()
    if len(words) < n:
        return [" ".join(words)]
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _build_minhash(text: str, num_perm: int) -> MinHash:
    """Create a MinHash signature for *text* using word 5-grams."""
    mh = MinHash(num_perm=num_perm)
    for gram in _word_ngrams(text, n=5):
        mh.update(gram.encode("utf-8"))
    return mh


def deduplicate_minhash(
    docs: list[dict[str, Any]],
    jaccard_threshold: float,
    num_perm: int,
) -> list[dict[str, Any]]:
    """Remove near-duplicate documents using MinHash LSH."""
    logger.info(
        "Stage 2 - MinHash LSH deduplication on %s documents "
        "(threshold=%.2f, num_perm=%d) ...",
        f"{len(docs):,}",
        jaccard_threshold,
        num_perm,
    )

    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
    kept: list[dict[str, Any]] = []
    duplicates = 0

    for idx, doc in enumerate(
        tqdm(docs, desc="MinHash dedup")
    ):
        text: str = doc.get("text", "")
        mh = _build_minhash(text, num_perm)

        # Query for existing near-duplicates.
        result = lsh.query(mh)
        if result:
            duplicates += 1
            continue

        # No duplicate found -- keep this document.
        key = f"doc_{idx}"
        lsh.insert(key, mh)
        kept.append(doc)

    logger.info(
        "Stage 2 complete: kept %s, removed %s duplicates",
        f"{len(kept):,}",
        f"{duplicates:,}",
    )
    return kept


# ---------------------------------------------------------------------------
# Stage 3 -- Perplexity Filtering (optional)
# ---------------------------------------------------------------------------


def perplexity_filter(
    docs: list[dict[str, Any]],
    kenlm_model_path: str,
    max_perplexity: float,
) -> list[dict[str, Any]]:
    """Remove documents whose KenLM perplexity exceeds *max_perplexity*.

    If the ``kenlm`` package is not installed the stage is skipped with a
    warning and the original list is returned unchanged.
    """
    try:
        import kenlm  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "kenlm is not installed -- skipping perplexity filtering. "
            "Install it with: pip install https://github.com/kpu/kenlm/archive/master.zip"
        )
        return docs

    if not Path(kenlm_model_path).is_file():
        logger.warning(
            "KenLM model file not found at %s -- skipping perplexity filtering.",
            kenlm_model_path,
        )
        return docs

    logger.info(
        "Stage 3 - Perplexity filtering on %s documents "
        "(model=%s, max_perplexity=%.1f) ...",
        f"{len(docs):,}",
        kenlm_model_path,
        max_perplexity,
    )

    model = kenlm.Model(kenlm_model_path)

    kept: list[dict[str, Any]] = []
    removed = 0

    for doc in tqdm(docs, desc="Perplexity filter"):
        text: str = doc.get("text", "")
        # KenLM scores in log10; convert to perplexity.
        log_score = model.score(text, bos=True, eos=True)
        num_words = len(text.split()) or 1
        # perplexity = 10 ^ (-log10_prob / num_words)
        ppl = 10.0 ** (-log_score / num_words)

        if ppl <= max_perplexity:
            kept.append(doc)
        else:
            removed += 1

    logger.info(
        "Stage 3 complete: kept %s, removed %s",
        f"{len(kept):,}",
        f"{removed:,}",
    )
    return kept


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_jsonl_shards(input_dir: str) -> list[dict[str, Any]]:
    """Read all ``*.jsonl`` files from *input_dir* and return a flat list."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    shard_files = sorted(input_path.glob("*.jsonl"))
    if not shard_files:
        logger.error("No .jsonl files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d shard(s) in %s", len(shard_files), input_dir)

    docs: list[dict[str, Any]] = []
    for shard in tqdm(shard_files, desc="Reading shards"):
        with open(shard, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))

    logger.info("Loaded %s documents from disk.", f"{len(docs):,}")
    return docs


def write_jsonl_shards(
    docs: list[dict[str, Any]],
    output_dir: str,
    docs_per_shard: int = 50_000,
) -> None:
    """Write *docs* as JSONL shards into *output_dir*."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_shards = (len(docs) + docs_per_shard - 1) // docs_per_shard
    logger.info(
        "Writing %s documents to %d shard(s) in %s",
        f"{len(docs):,}",
        total_shards,
        output_dir,
    )

    for shard_idx in range(total_shards):
        start = shard_idx * docs_per_shard
        end = min(start + docs_per_shard, len(docs))
        shard_name = f"filtered_{shard_idx:05d}.jsonl"
        shard_path = output_path / shard_name

        with open(shard_path, "w", encoding="utf-8") as fh:
            for doc in docs[start:end]:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info("Finished writing shards.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    logger.info("=== BoccaccioAI Data Filtering Pipeline ===")
    logger.info("Input dir:          %s", args.input_dir)
    logger.info("Output dir:         %s", args.output_dir)
    logger.info("Min doc length:     %d", args.min_doc_length)
    logger.info("Max doc length:     %d", args.max_doc_length)
    logger.info("Jaccard threshold:  %.2f", args.jaccard_threshold)
    logger.info("Num permutations:   %d", args.num_perm)
    logger.info("KenLM model:        %s", args.kenlm_model or "(none)")
    logger.info("Max perplexity:     %.1f", args.max_perplexity)
    logger.info("Workers:            %d", args.num_workers)

    # Load raw documents.
    docs = read_jsonl_shards(args.input_dir)
    original_count = len(docs)

    # Stage 1: Heuristic filtering.
    docs = heuristic_filter(
        docs,
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
        num_workers=args.num_workers,
    )

    # Stage 2: MinHash LSH deduplication.
    docs = deduplicate_minhash(
        docs,
        jaccard_threshold=args.jaccard_threshold,
        num_perm=args.num_perm,
    )

    # Stage 3: Perplexity filtering (optional).
    if args.kenlm_model is not None:
        docs = perplexity_filter(
            docs,
            kenlm_model_path=args.kenlm_model,
            max_perplexity=args.max_perplexity,
        )
    else:
        logger.info("Stage 3 - Skipped (no --kenlm-model provided).")

    # Save filtered output.
    write_jsonl_shards(docs, args.output_dir)

    # Summary.
    final_count = len(docs)
    reduction = (1.0 - final_count / original_count) * 100.0 if original_count else 0.0
    logger.info("=== Filtering Summary ===")
    logger.info("Original documents:  %s", f"{original_count:,}")
    logger.info("Final documents:     %s", f"{final_count:,}")
    logger.info("Reduction:           %.2f%%", reduction)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Filtering interrupted by user.")
        sys.exit(130)
