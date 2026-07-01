import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN", "")
api = HfApi(token=HF_TOKEN)
repo = "lollix91/boccaccio-data"

# Lista file attuali
files = api.list_repo_files(repo, repo_type="dataset")
print("File attuali su HF:")
for f in sorted(files):
    print(f"  {f}")

# Cancella tutti i checkpoint nominati, tieni solo last.ckpt
to_delete = [f for f in files if f.startswith("checkpoints/pretrain/epoch=0-step=")]
print(f"\nCheckpoint da cancellare: {len(to_delete)}")
for f in to_delete:
    print(f"  Cancello {f}...")
    api.delete_file(path_in_repo=f, repo_id=repo, repo_type="dataset", token=HF_TOKEN)
    print(f"    OK")

# Verifica finale
print("\nFile rimanenti:")
files = api.list_repo_files(repo, repo_type="dataset")
total = 0
for f in sorted(files):
    info = api.get_paths_info(repo, [f], repo_type="dataset")
    if info and info[0].size:
        total += info[0].size
        print(f"  {f}: {info[0].size/1e9:.2f} GB")
print(f"\nTOTAL: {total/1e9:.2f} GB")
