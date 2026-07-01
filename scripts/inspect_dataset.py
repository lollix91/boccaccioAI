import json
from datasets import load_dataset

ds = load_dataset('anakin87/fine-instructions-ita-70k')
print(f"Total examples: {len(ds['train'])}")
print(f"Features: {ds['train'].features}")
print()

for i in range(3):
    ex = ds['train'][i]
    print(f"=== Example {i} ===")
    print(f"ID: {ex['id']}")
    print(f"Quality: {ex['quality']}")
    print(f"Conversations: {json.dumps(ex['conversations'], ensure_ascii=False, indent=2)[:500]}")
    print()
