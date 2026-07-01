import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN", "")
api = HfApi(token=HF_TOKEN)
repo = "lollix91/boccaccio-data"

files = [
    "data/tokenized/finetune/train.bin",
    "data/tokenized/finetune/val.bin",
    "data/tokenized/finetune/meta.json",
]

for f in files:
    print(f"Uploading {f}...")
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id=repo,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print(f"  OK")

print("Upload completato.")
