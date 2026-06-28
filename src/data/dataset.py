"""
BoccaccioAI - PyTorch Datasets for Pre-Training and Instruction Fine-Tuning

Provides Dataset classes for loading pre-tokenized binary data (pre-training)
and JSONL instruction data (fine-tuning).

De Lauretis Tech
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset


class PreTokenizedDataset(Dataset):
    """Dataset for loading pre-tokenized binary data during pre-training.

    Reads a flat uint16 numpy memmap file produced by ``tokenize_corpus.py``
    and serves fixed-length token sequences.
    """

    def __init__(self, data_path: str, sequence_length: int = 2048) -> None:
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        self.data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        self.num_sequences = len(self.data) // self.sequence_length

        if self.num_sequences == 0:
            raise ValueError(
                f"Data file {self.data_path} contains {len(self.data)} tokens, "
                f"which is fewer than one full sequence of length {self.sequence_length}."
            )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.num_sequences:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.num_sequences} sequences"
            )

        start = idx * self.sequence_length
        end = start + self.sequence_length
        tokens = np.array(self.data[start:end], dtype=np.int64)
        tokens = torch.from_numpy(tokens)

        return {"input_ids": tokens, "labels": tokens}


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning.

    Reads JSONL files with ``{"context", "question", "answer"}`` fields and
    formats them using a chat-style template. Loss is computed only on the
    answer portion by masking the prompt tokens with -100 in the labels.
    """

    PROMPT_TEMPLATE = "### Contesto:\n{context}\n### Domanda:\n{question}\n### Risposta:\n"
    ANSWER_TEMPLATE = "{answer}</s>"

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        sequence_length: int = 2048,
    ) -> None:
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load examples from JSONL.
        self.examples: list[dict[str, str]] = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.examples.append(json.loads(line))

        if not self.examples:
            raise ValueError(f"No examples found in {self.data_path}")

        # Load tokenizer.
        tokenizer_file = Path(tokenizer_path)
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_file}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))

        # Resolve special token IDs.
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_token_id is None:
            self.pad_token_id = 0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self.examples):
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self.examples)} examples"
            )

        example = self.examples[idx]
        context = example.get("context", "")
        question = example.get("question", "")
        answer = example.get("answer", "")

        # Build prompt and full text.
        prompt_text = self.PROMPT_TEMPLATE.format(context=context, question=question)
        answer_text = self.ANSWER_TEMPLATE.format(answer=answer)
        full_text = prompt_text + answer_text

        # Tokenize prompt and full text separately to find the boundary.
        prompt_ids = self.tokenizer.encode(prompt_text).ids
        full_ids = self.tokenizer.encode(full_text).ids

        prompt_length = len(prompt_ids)

        # Truncate or pad to sequence_length.
        if len(full_ids) > self.sequence_length:
            full_ids = full_ids[: self.sequence_length]
            prompt_length = min(prompt_length, self.sequence_length)

        actual_length = len(full_ids)
        padding_length = self.sequence_length - actual_length

        # Build input_ids with padding.
        input_ids = full_ids + [self.pad_token_id] * padding_length

        # Build labels: -100 for prompt tokens and padding tokens.
        labels = [-100] * prompt_length + full_ids[prompt_length:] + [-100] * padding_length

        # Build attention mask: 1 for real tokens, 0 for padding.
        attention_mask = [1] * actual_length + [0] * padding_length

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
            "attention_mask": attention_mask_tensor,
        }


if __name__ == "__main__":
    import sys

    print("BoccaccioAI Dataset classes")
    print(f"  PreTokenizedDataset: for pre-training on binary memmap files")
    print(f"  InstructionDataset:  for instruction fine-tuning on JSONL data")

    if len(sys.argv) > 1 and sys.argv[1] == "--test-pretokenized":
        if len(sys.argv) < 3:
            print("Usage: python dataset.py --test-pretokenized <data_path>")
            sys.exit(1)
        ds = PreTokenizedDataset(sys.argv[2])
        print(f"  Loaded {len(ds)} sequences from {sys.argv[2]}")
        sample = ds[0]
        print(f"  Sample input_ids shape: {sample['input_ids'].shape}")
        print(f"  Sample input_ids dtype: {sample['input_ids'].dtype}")
        print(f"  First 20 token IDs: {sample['input_ids'][:20].tolist()}")

    if len(sys.argv) > 1 and sys.argv[1] == "--test-instruction":
        if len(sys.argv) < 4:
            print("Usage: python dataset.py --test-instruction <data_path> <tokenizer_path>")
            sys.exit(1)
        ds = InstructionDataset(sys.argv[2], sys.argv[3])
        print(f"  Loaded {len(ds)} examples from {sys.argv[2]}")
        sample = ds[0]
        print(f"  Sample input_ids shape: {sample['input_ids'].shape}")
        print(f"  Sample labels shape:    {sample['labels'].shape}")
        print(f"  Sample attn_mask shape: {sample['attention_mask'].shape}")
        num_masked = (sample["labels"] == -100).sum().item()
        print(f"  Masked label positions: {num_masked}")
