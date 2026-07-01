import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN", ""))
files = api.list_repo_files("lollix91/boccaccio-data", repo_type="dataset")
total = 0
for f in sorted(files):
    info = api.get_paths_info("lollix91/boccaccio-data", [f], repo_type="dataset")
    if info and info[0].size:
        total += info[0].size
        print(f"{f}: {info[0].size/1e9:.2f} GB")
print(f"TOTAL: {total/1e9:.2f} GB")
