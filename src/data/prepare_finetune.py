"""Prepara il dataset di fine-tuning unendo 3 fonti:

1. anakin87/fine-instructions-ita-70k (70k esempi, formato conversations)
2. raicrits/Orca_ITA_200k (200k esempi, formato system/question/response)
3. Refusal examples (1.1k esempi, formato conversations)

Formatta tutto nel formato chat:
  user domanda  assistant risposta <|end|>

Poi tokenizza in formato binario (train.bin / val.bin).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer

random.seed(42)

# ─── Config ───────────────────────────────────────────────────

TOKENIZER_PATH = "tokenizer/boccaccio-32k.json"
OUTPUT_DIR = Path("data/tokenized/finetune")
REFUSAL_PATH = Path("data/finetune/refusal_examples.json")
VAL_SPLIT = 0.01
MAX_SEQ_LEN = 2048

USER_PREFIX = "user"
ASSISTANT_PREFIX = "assistant"
END_TOKEN = "<|end|>"

# ─── Load tokenizer ───────────────────────────────────────────

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
print(f"Tokenizer loaded: {TOKENIZER_PATH} (vocab={tokenizer.get_vocab_size()})")

# ─── Format helpers ───────────────────────────────────────────

def format_conversation(conversations: list[dict]) -> str:
    """Formatta conversazioni nel formato chat (da fine-instructions)."""
    text = ""
    for turn in conversations:
        role = turn["role"]
        content = turn["content"].strip()
        if role == "user":
            text += f"{USER_PREFIX} {content} {ASSISTANT_PREFIX} "
        elif role == "assistant":
            text += f"{content} {END_TOKEN}"
    return text


def format_orca(system_prompt: str, question: str, response: str) -> str | None:
    """Formatta esempi Orca nel formato chat. Ritorna None se dati mancanti."""
    if not question or not response:
        return None
    q = question.strip()
    sp = (system_prompt or "").strip()
    if sp:
        q = f"{sp}\n\n{q}"
    r = response.strip()
    return f"{USER_PREFIX} {q} {ASSISTANT_PREFIX} {r} {END_TOKEN}"


# ─── Load datasets ────────────────────────────────────────────

print("\n=== Caricamento dataset ===")

# 1. fine-instructions-ita-70k
print("Caricamento fine-instructions-ita-70k...")
ds1 = load_dataset("anakin87/fine-instructions-ita-70k")
instructions = [ex for ex in ds1["train"] if ex["quality"] >= 3]
print(f"  Instruction: {len(instructions)} esempi")

# 2. Orca_ITA_200k
print("Caricamento Orca_ITA_200k...")
ds2 = load_dataset("raicrits/Orca_ITA_200k")
orca = list(ds2["train"])
print(f"  Orca: {len(orca)} esempi")

# 3. Refusal examples
print("Caricamento refusal examples...")
with open(REFUSAL_PATH, "r", encoding="utf-8") as f:
    refusals = json.load(f)
print(f"  Refusal: {len(refusals)} esempi")

# ─── Tokenize all ─────────────────────────────────────────────

print("\n=== Preparazione esempi (formattazione) ===")

# Build a unified list of formatted texts, then shuffle for good mixing.
all_texts = []

for ex in instructions:
    text = format_conversation(ex["conversations"])
    all_texts.append(("instruction", text))

for ex in orca:
    text = format_orca(
        ex.get("system_prompt_it", ""),
        ex.get("question_it", ""),
        ex.get("response_it", ""),
    )
    if text is not None:
        all_texts.append(("orca", text))

for ex in refusals:
    text = format_conversation(ex["conversations"])
    all_texts.append(("refusal", text))

print(f"  Totale esempi: {len(all_texts)}")
random.shuffle(all_texts)

print("\n=== Tokenizzazione ===")

all_ids = []
stats = {"instructions": 0, "orca": 0, "refusals": 0, "skipped": 0}

for i, (source, text) in enumerate(all_texts):
    ids = tokenizer.encode(text).ids
    if len(ids) > MAX_SEQ_LEN:
        ids = ids[:MAX_SEQ_LEN]
    if len(ids) < 10:
        stats["skipped"] += 1
        continue
    all_ids.extend(ids)
    if source == "instruction":
        stats["instructions"] += 1
    elif source == "orca":
        stats["orca"] += 1
    elif source == "refusal":
        stats["refusals"] += 1
    if (i + 1) % 50000 == 0:
        print(f"  Processati {i+1}/{len(all_texts)}...")

print(f"\n  Statistiche: {stats}")
print(f"  Totale token: {len(all_ids):,}")

# ─── Split train/val ──────────────────────────────────────────
# NOTE: Do NOT shuffle all_ids - that would destroy the linguistic structure.
# The tokens must remain in their original order (contiguous sequences).
# Shuffling of sequences happens at the DataLoader level during training.

val_size = int(len(all_ids) * VAL_SPLIT)
val_size = (val_size // MAX_SEQ_LEN) * MAX_SEQ_LEN

val_ids = all_ids[:val_size]
train_ids = all_ids[val_size:]

print(f"\n=== Split ===")
print(f"  Train: {len(train_ids):,} token ({len(train_ids) // MAX_SEQ_LEN} sequenze)")
print(f"  Val:   {len(val_ids):,} token ({len(val_ids) // MAX_SEQ_LEN} sequenze)")

# ─── Save ─────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_array = np.array(train_ids, dtype=np.uint16)
val_array = np.array(val_ids, dtype=np.uint16)

train_path = OUTPUT_DIR / "train.bin"
val_path = OUTPUT_DIR / "val.bin"

train_array.tofile(train_path)
val_array.tofile(val_path)

print(f"\n=== Salvataggio ===")
print(f"  {train_path}: {train_path.stat().st_size / 1e9:.2f} GB")
print(f"  {val_path}: {val_path.stat().st_size / 1e6:.2f} MB")

# Meta
meta = {
    "vocab_size": tokenizer.get_vocab_size(),
    "sequence_length": MAX_SEQ_LEN,
    "train_tokens": len(train_ids),
    "val_tokens": len(val_ids),
    "train_sequences": len(train_ids) // MAX_SEQ_LEN,
    "val_sequences": len(val_ids) // MAX_SEQ_LEN,
    "format": "chat",
    "special_tokens": {
        "user_prefix": USER_PREFIX,
        "assistant_prefix": ASSISTANT_PREFIX,
        "end_token": END_TOKEN,
    },
    "sources": stats,
}

meta_path = OUTPUT_DIR / "meta.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"  {meta_path}")
print(f"\n=== Fatto ===")
print(f"  Token totali: {len(all_ids):,} (~{len(all_ids)/1e6:.1f}M)")
