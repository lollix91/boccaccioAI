"""Squash HF Hub repo history to reclaim storage from deleted files."""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))
repo_id = "lollix91/boccaccio-data"
repo_type = "dataset"

print(f"Squashing history for {repo_id} ({repo_type})...")
print("This will collapse all commits into one, deleting old file versions from LFS storage.")
print()

try:
    api.super_squash_history(repo_id=repo_id, repo_type=repo_type)
    print("Squash completed successfully!")
except Exception as e:
    print(f"Error: {e}")

print("\nRemaining files:")
files = api.list_repo_files(repo_id, repo_type=repo_type)
for f in files:
    print(f"  {f}")
