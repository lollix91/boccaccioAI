"""
BoccaccioAI - CulturaX Italian Data Download Script

Downloads the Italian subset of the CulturaX dataset from HuggingFace
and saves it as sharded JSONL files for downstream processing.

De Lauretis Tech
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("boccaccio.data.download")

BYTES_PER_GB = 1024 ** 3
BYTES_PER_MB = 1024 ** 2
SHARD_SIZE_BYTES = 500 * BYTES_PER_MB  # 500 MB per shard

DATASET_NAME = "uonlp/CulturaX"
DATASET_LANG = "it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Italian subset of CulturaX for BoccaccioAI."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory where raw JSONL shards will be saved (default: data/raw).",
    )
    parser.add_argument(
        "--max-size-gb",
        type=float,
        default=30.0,
        help="Maximum amount of data to download in GB (default: 30).",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=4,
        help="Number of parallel download processes (default: 4).",
    )
    return parser.parse_args()


def open_new_shard(output_dir: Path, shard_index: int):
    """Open a new shard file for writing and return the file handle."""
    shard_path = output_dir / f"culturax_it_{shard_index:05d}.jsonl"
    logger.info("Opening new shard: %s", shard_path.name)
    return open(shard_path, "w", encoding="utf-8")


def download_culturax(
    output_dir: Path,
    max_size_gb: float,
    num_proc: int,
) -> dict:
    """
    Stream the Italian split of CulturaX and write documents to sharded
    JSONL files.  Returns a statistics dictionary.
    """
    target_bytes = int(max_size_gb * BYTES_PER_GB)

    logger.info("Dataset:      %s (lang=%s)", DATASET_NAME, DATASET_LANG)
    logger.info("Target size:  %.2f GB (%s bytes)", max_size_gb, f"{target_bytes:,}")
    logger.info("Shard size:   %d MB", SHARD_SIZE_BYTES // BYTES_PER_MB)
    logger.info("Output dir:   %s", output_dir)
    logger.info("Num procs:    %d", num_proc)

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_LANG,
        split="train",
        streaming=True,
        num_proc=num_proc,
    )

    total_bytes = 0
    total_docs = 0
    shard_index = 0
    shard_bytes = 0
    shard_file = open_new_shard(output_dir, shard_index)

    pbar = tqdm(
        total=target_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Downloading CulturaX IT",
    )

    try:
        for example in dataset:
            text = example.get("text", "")
            url = example.get("url", "")
            timestamp = example.get("timestamp", "")

            record = {
                "text": text,
                "url": url,
                "timestamp": timestamp,
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode("utf-8"))

            shard_file.write(line)
            shard_bytes += line_bytes
            total_bytes += line_bytes
            total_docs += 1
            pbar.update(line_bytes)

            # Rotate shard when the current one exceeds the size limit.
            if shard_bytes >= SHARD_SIZE_BYTES:
                shard_file.close()
                logger.info(
                    "Shard %05d complete: %d docs, %.2f MB",
                    shard_index,
                    total_docs,
                    shard_bytes / BYTES_PER_MB,
                )
                shard_index += 1
                shard_bytes = 0
                shard_file = open_new_shard(output_dir, shard_index)

            # Stop once we have reached the target download size.
            if total_bytes >= target_bytes:
                logger.info("Reached target size (%.2f GB). Stopping.", max_size_gb)
                break

    except KeyboardInterrupt:
        logger.warning("Download interrupted by user. Saving progress...")

    finally:
        shard_file.close()
        pbar.close()

    stats = {
        "total_documents": total_docs,
        "total_bytes": total_bytes,
        "total_shards": shard_index + 1,
        "avg_doc_length_bytes": total_bytes / total_docs if total_docs > 0 else 0,
    }
    return stats


def log_statistics(stats: dict) -> None:
    """Log a summary of the download run."""
    logger.info("=== Download Statistics ===")
    logger.info("Total documents:  %s", f"{stats['total_documents']:,}")
    logger.info(
        "Total size:       %.2f GB (%s bytes)",
        stats["total_bytes"] / BYTES_PER_GB,
        f"{stats['total_bytes']:,}",
    )
    logger.info("Total shards:     %d", stats["total_shards"])
    logger.info(
        "Avg doc length:   %.0f bytes (%.1f KB)",
        stats["avg_doc_length_bytes"],
        stats["avg_doc_length_bytes"] / 1024,
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== BoccaccioAI - CulturaX Italian Download ===")

    stats = download_culturax(
        output_dir=output_dir,
        max_size_gb=args.max_size_gb,
        num_proc=args.num_proc,
    )

    log_statistics(stats)
    logger.info("Done. Output written to %s", output_dir.resolve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
