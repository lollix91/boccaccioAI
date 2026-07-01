"""Upload essential files to freshly recreated HF Hub repo.
Uploads from local PC: tokenizer, configs, finetune data, pretrain meta.
Uploads from Vast.ai: pretrain checkpoint (too large for local).
"""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "lollix91/boccaccio-data"
repo_type = "dataset"

def upload_local(local_path: str, repo_path: str) -> None:
    if not os.path.exists(local_path):
        print(f"  SKIP (not found locally): {local_path}")
        return
    print(f"  Uploading {local_path} -> {repo_path}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("    OK")

# 1. Pretrain checkpoint (the final one, 7.9GB)
upload_local("checkpoints/last.ckpt", "checkpoints/pretrain/last.ckpt")

# 2. Tokenizer
upload_local("tokenizer/boccaccio-32k.json", "tokenizer/boccaccio-32k.json")

# 3. Configs
for cfg in ["configs/model.yaml", "configs/tokenizer.yaml", "configs/training.yaml"]:
    upload_local(cfg, cfg)

# 4. Finetune data (small, ~204MB)
print("  Uploading data/tokenized/finetune/ folder...")
api.upload_folder(
    folder_path="data/tokenized/finetune",
    path_in_repo="data/tokenized/finetune",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("    OK")

# 5. Pretrain meta (small)
upload_local("data/tokenized/pretrain/meta.json", "data/tokenized/pretrain/meta.json")

# 6. Pretrain train.bin and val.bin (13.3GB - we have these locally)
upload_local("data/tokenized/pretrain/train.bin", "data/tokenized/pretrain/train.bin")
upload_local("data/tokenized/pretrain/val.bin", "data/tokenized/pretrain/val.bin")

print("\nDone! Files uploaded.")
print("\nNOTE: checkpoints/pretrain/last.ckpt must be uploaded from Vast.ai")
print("      (not available locally). The daemon will handle finetune checkpoints.")

print("\nRemaining files:")
files = api.list_repo_files(repo_id, repo_type=repo_type)
for f in files:
    print(f"  {f}")
