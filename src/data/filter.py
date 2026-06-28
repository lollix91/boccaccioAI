"""
BoccaccioAI - Data Filtering / Cleaning Pipeline (Streaming)

Takes raw Italian text from CulturaX JSONL shards and applies sequential
quality-filtering stages to produce a clean dataset:

  Pass 1 - Heuristic filtering (streaming, shard per shard):
    - Length, alpha ratio, punctuation, repeated lines
    - Writes surviving docs to data/heuristic/

  Pass 2 - MinHash LSH deduplication (signature-only in RAM):
    - Reads heuristic-passed docs, builds MinHash signatures
    - Only signatures (not full docs) are held in RAM
    - Writes deduplicated docs to data/filtered/

  Pass 3 - Perplexity filtering (optional, requires KenLM model):
    - Streaming, shard per shard

Memory usage: ~500MB per shard during Pass 1,
              ~10GB for MinHash signatures of ~10M docs during Pass 2.

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
from typing import Any, Iterator

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
# I/O helpers (streaming)
# ---------------------------------------------------------------------------


def iter_jsonl_shards(input_dir: str) -> tuple[list[Path], Iterator[dict[str, Any]]]:
    """Return (shard_files, generator) that yields docs one at a time.

    The generator reads shard files sequentially, yielding parsed JSON dicts.
    Memory usage is O(1) -- only one line in memory at a time.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    shard_files = sorted(input_path.glob("*.jsonl"))
    if not shard_files:
        logger.error("No .jsonl files found in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d shard(s) in %s", len(shard_files), input_dir)

    def _gen() -> Iterator[dict[str, Any]]:
        for shard in shard_files:
            with open(shard, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    return shard_files, _gen()


def iter_jsonl_shard_file(path: Path) -> Iterator[dict[str, Any]]:
    """Yield docs from a single JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_shards(
    docs: list[dict[str, Any]],
    output_dir: str,
    docs_per_shard: int = 50_000,
    prefix: str = "filtered",
) -> int:
    """Write *docs* as JSONL shards into *output_dir*. Returns shard count."""
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
        shard_name = f"{prefix}_{shard_idx:05d}.jsonl"
        shard_path = output_path / shard_name

        with open(shard_path, "w", encoding="utf-8") as fh:
            for doc in docs[start:end]:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info("Finished writing %d shard(s).", total_shards)
    return total_shards


# ---------------------------------------------------------------------------
# Stage 1 -- Heuristic Filtering (streaming, shard per shard)
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
        avg_line_length = sum(len(line) for line in lines) / len(lines)
        if avg_line_length < 20:
            return False

    return True


def heuristic_filter_streaming(
    input_dir: str,
    output_dir: str,
    min_doc_length: int,
    max_doc_length: int,
    num_workers: int,
) -> tuple[int, int]:
    """Apply heuristic filters shard-by-shard (streaming).

    Reads each shard, filters in parallel, writes surviving docs to output_dir.
    Returns (total_input, total_kept).
    """
    logger.info(
        "Stage 1 - Heuristic filtering (streaming, workers=%d) ...",
        num_workers,
    )

    input_path = Path(input_dir)
    shard_files = sorted(input_path.glob("*.jsonl"))
    if not shard_files:
        logger.error("No .jsonl files found in %s", input_dir)
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    check_fn = partial(
        _passes_heuristic,
        min_doc_length=min_doc_length,
        max_doc_length=max_doc_length,
    )

    total_input = 0
    total_kept = 0
    shard_out_idx = 0
    docs_per_shard = 50_000
    buffer: list[dict[str, Any]] = []

    def flush_buffer() -> None:
        nonlocal shard_out_idx, buffer
        if not buffer:
            return
        shard_name = f"heuristic_{shard_out_idx:05d}.jsonl"
        shard_path = output_path / shard_name
        with open(shard_path, "w", encoding="utf-8") as fh:
            for doc in buffer:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        shard_out_idx += 1
        buffer = []

    for shard in tqdm(shard_files, desc="Heuristic filter"):
        # Read shard into memory (one shard = ~500MB, safe for 16GB RAM)
        shard_docs = list(iter_jsonl_shard_file(shard))
        total_input += len(shard_docs)

        # Filter in parallel
        with Pool(processes=num_workers) as pool:
            verdicts = list(
                pool.imap(check_fn, shard_docs, chunksize=512)
            )

        for doc, ok in zip(shard_docs, verdicts):
            if ok:
                buffer.append(doc)
                total_kept += 1
                if len(buffer) >= docs_per_shard:
                    flush_buffer()

        # Free shard memory
        del shard_docs

    flush_buffer()

    removed = total_input - total_kept
    logger.info(
        "Stage 1 complete: input %s, kept %s, removed %s (%.1f%%)",
        f"{total_input:,}",
        f"{total_kept:,}",
        f"{removed:,}",
        (removed / total_input * 100) if total_input else 0,
    )
    return total_input, total_kept


# ---------------------------------------------------------------------------
# Stage 2 -- MinHash LSH Deduplication (signature-only in RAM)
# ---------------------------------------------------------------------------


def _word_ngrams(text: str, n: int = 5) -> list[str]:
    """Extract word-level n-grams from *text*."""
    words = text.split()
    if len(words) < n:
        return [" ".join(words)]
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def _build_minhash_bytes(text: str, num_perm: int) -> bytes:
    """Create a MinHash signature for *text* and return as raw bytes.

    Each MinHash with 128 perms uses ~1KB. For 10M docs = ~10GB.
    We store only the hashset bytes, not the full MinHash object.
    """
    mh = MinHash(num_perm=num_perm)
    for gram in _word_ngrams(text, n=5):
        mh.update(gram.encode("utf-8"))
    return mh.hashset.tobytes()  # raw bytes, ~num_perm * 4 bytes


def deduplicate_minhash_streaming(
    input_dir: str,
    output_dir: str,
    jaccard_threshold: float,
    num_perm: int,
) -> tuple[int, int]:
    """Remove near-duplicate documents using MinHash LSH (streaming).

    Pass 2a: Read heuristic-passed docs, build MinHash signatures,
             store signatures + doc references in LSH index.
    Pass 2b: Iterate again, keep only docs that are NOT duplicates,
             write them to output_dir.

    Memory: ~num_perm * 4 bytes per doc for signatures + LSH index overhead.
    For 10M docs with 128 perms: ~10GB (fits in 16GB RAM).

    Returns (total_input, total_kept).
    """
    import numpy as np

    input_path = Path(input_dir)
    shard_files = sorted(input_path.glob("*.jsonl"))
    if not shard_files:
        logger.error("No .jsonl files found in %s", input_dir)
        sys.exit(1)

    logger.info(
        "Stage 2 - MinHash LSH deduplication (streaming, threshold=%.2f, num_perm=%d) ...",
        jaccard_threshold,
        num_perm,
    )

    lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)

    # Pass 2a: Build LSH index with all signatures
    total_docs = 0
    doc_keys: list[str] = []  # key like "shard_00003:line_00124"

    logger.info("Pass 2a: Building MinHash LSH index ...")
    for shard in tqdm(shard_files, desc="Building LSH index"):
        for line_idx, line in enumerate(open(shard, "r", encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            text: str = doc.get("text", "")

            mh = MinHash(num_perm=num_perm)
            for gram in _word_ngrams(text, n=5):
                mh.update(gram.encode("utf-8"))

            key = f"{shard.stem}:{line_idx}"
            lsh.insert(key, mh)
            doc_keys.append(key)
            total_docs += 1

    logger.info("LSH index built with %s documents.", f"{total_docs:,}")

    # Pass 2b: Stream through docs again, keep only non-duplicates
    logger.info("Pass 2b: Filtering duplicates ...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    duplicates = 0
    kept = 0
    shard_out_idx = 0
    docs_per_shard = 50_000
    buffer: list[dict[str, Any]] = []

    def flush_buffer() -> None:
        nonlocal shard_out_idx, buffer
        if not buffer:
            return
        shard_name = f"filtered_{shard_out_idx:05d}.jsonl"
        shard_path = output_path / shard_name
        with open(shard_path, "w", encoding="utf-8") as fh:
            for doc in buffer:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        shard_out_idx += 1
        buffer = []

    # Track which keys we've already seen as duplicates
    seen_duplicates: set[str] = set()

    for shard in tqdm(shard_files, desc="Dedup filtering"):
        for line_idx, line in enumerate(open(shard, "r", encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            text: str = doc.get("text", "")

            mh = MinHash(num_perm=num_perm)
            for gram in _word_ngrams(text, n=5):
                mh.update(gram.encode("utf-8"))

            key = f"{shard.stem}:{line_idx}"
            result = lsh.query(mh)

            # A doc is a duplicate if the query returns keys other than itself
            # AND it's not the first occurrence (first one wins)
            other_keys = [k for k in result if k != key]
            if other_keys:
                # This is a near-duplicate of an earlier doc
                # Check if any of the others were already kept (first occurrence)
                is_first = all(k in seen_duplicates for k in other_keys)
                if not is_first:
                    duplicates += 1
                    seen_duplicates.add(key)
                    continue

            kept += 1
            buffer.append(doc)
            if len(buffer) >= docs_per_shard:
                flush_buffer()

    flush_buffer()

    logger.info(
        "Stage 2 complete: input %s, kept %s, removed %s duplicates",
        f"{total_docs:,}",
        f"{kept:,}",
        f"{duplicates:,}",
    )
    return total_docs, kept


# ---------------------------------------------------------------------------
# Stage 3 -- Perplexity Filtering (optional, streaming)
# ---------------------------------------------------------------------------


def perplexity_filter_streaming(
    input_dir: str,
    output_dir: str,
    kenlm_model_path: str,
    max_perplexity: float,
) -> tuple[int, int]:
    """Remove documents whose KenLM perplexity exceeds *max_perplexity*.

    Streaming: reads shard-by-shard, writes passing docs to output_dir.
    """
    try:
        import kenlm  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "kenlm is not installed -- skipping perplexity filtering. "
            "Install it with: pip install https://github.com/kpu/kenlm/archive/master.zip"
        )
        # Just copy input to output
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.copytree(input_dir, output_dir)
        return 0, 0

    if not Path(kenlm_model_path).is_file():
        logger.warning(
            "KenLM model file not found at %s -- skipping perplexity filtering.",
            kenlm_model_path,
        )
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.copytree(input_dir, output_dir)
        return 0, 0

    logger.info(
        "Stage 3 - Perplexity filtering (streaming, model=%s, max_perplexity=%.1f) ...",
        kenlm_model_path,
        max_perplexity,
    )

    model = kenlm.Model(kenlm_model_path)

    input_path = Path(input_dir)
    shard_files = sorted(input_path.glob("*.jsonl"))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    shard_out_idx = 0
    docs_per_shard = 50_000
    buffer: list[dict[str, Any]] = []

    def flush_buffer() -> None:
        nonlocal shard_out_idx, buffer
        if not buffer:
            return
        shard_name = f"ppl_filtered_{shard_out_idx:05d}.jsonl"
        shard_path = output_path / shard_name
        with open(shard_path, "w", encoding="utf-8") as fh:
            for doc in buffer:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        shard_out_idx += 1
        buffer = []

    for shard in tqdm(shard_files, desc="Perplexity filter"):
        for line in open(shard, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            text: str = doc.get("text", "")
            total += 1

            log_score = model.score(text, bos=True, eos=True)
            num_words = len(text.split()) or 1
            ppl = 10.0 ** (-log_score / num_words)

            if ppl <= max_perplexity:
                buffer.append(doc)
                kept += 1
                if len(buffer) >= docs_per_shard:
                    flush_buffer()

    flush_buffer()

    removed = total - kept
    logger.info(
        "Stage 3 complete: input %s, kept %s, removed %s",
        f"{total:,}",
        f"{kept:,}",
        f"{removed:,}",
    )
    return total, kept


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

    # Intermediate directory for heuristic-passed docs
    heuristic_dir = str(Path(args.output_dir).parent / "heuristic")

    # ─── Pass 1: Heuristic filtering (streaming) ──────────
    total_input, heuristic_kept = heuristic_filter_streaming(
        input_dir=args.input_dir,
        output_dir=heuristic_dir,
        min_doc_length=args.min_doc_length,
        max_doc_length=args.max_doc_length,
        num_workers=args.num_workers,
    )

    # ─── Pass 2: MinHash LSH deduplication (streaming) ────
    dedup_input, dedup_kept = deduplicate_minhash_streaming(
        input_dir=heuristic_dir,
        output_dir=args.output_dir,
        jaccard_threshold=args.jaccard_threshold,
        num_perm=args.num_perm,
    )

    # ─── Pass 3: Perplexity filtering (optional) ──────────
    if args.kenlm_model is not None:
        ppl_dir = str(Path(args.output_dir).parent / "ppl_filtered")
        ppl_input, ppl_kept = perplexity_filter_streaming(
            input_dir=args.output_dir,
            output_dir=ppl_dir,
            kenlm_model_path=args.kenlm_model,
            max_perplexity=args.max_perplexity,
        )
        final_count = ppl_kept
    else:
        logger.info("Stage 3 - Skipped (no --kenlm-model provided).")
        final_count = dedup_kept

    # ─── Cleanup intermediate heuristic dir ───────────────
    import shutil
    heuristic_path = Path(heuristic_dir)
    if heuristic_path.exists():
        shutil.rmtree(heuristic_path)
        logger.info("Cleaned up intermediate directory: %s", heuristic_dir)

    # ─── Summary ──────────────────────────────────────────
    reduction = (1.0 - final_count / total_input) * 100.0 if total_input else 0.0
    logger.info("=== Filtering Summary ===")
    logger.info("Original documents:  %s", f"{total_input:,}")
    logger.info("After heuristic:     %s", f"{heuristic_kept:,}")
    logger.info("After dedup:         %s", f"{dedup_kept:,}")
    logger.info("Final documents:     %s", f"{final_count:,}")
    logger.info("Reduction:           %.2f%%", reduction)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Filtering interrupted by user.")
        sys.exit(130)
