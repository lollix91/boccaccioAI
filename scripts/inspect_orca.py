import json
from datasets import load_dataset

print("=== Orca_ITA_200k ===")
ds = load_dataset("raicrits/Orca_ITA_200k")
print(f"Total: {len(ds['train'])}")
print(f"Features: {ds['train'].features}")
print()
for i in range(3):
    ex = ds['train'][i]
    print(f"--- Example {i} ---")
    for k, v in ex.items():
        s = str(v)[:300]
        print(f"  {k}: {s}")
    print()
