"""Delete stale finetune checkpoints from HF Hub (from the broken run with token shuffle)."""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "lollix91/boccaccio-data"
repo_type = "dataset"

files_to_delete = [
    "checkpoints/finetune/epoch=0-step=10500.ckpt",
    "checkpoints/finetune/epoch=0-step=500.ckpt",
    "checkpoints/finetune/last-v1.ckpt",
    "checkpoints/finetune/last.ckpt",
]

for fpath in files_to_delete:
    try:
        api.delete_file(fpath, repo_id=repo_id, repo_type=repo_type)
        print(f"  Deleted: {fpath}")
    except Exception as e:
        print(f"  Error deleting {fpath}: {e}")

print("\nDone. Remaining files:")
files = api.list_repo_files(repo_id, repo_type=repo_type)
for f in files:
    print(f"  {f}")
