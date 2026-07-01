import sys
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Find the latest version
versions = sorted(glob.glob("/workspace/boccaccioAI/logs/finetune/version_*"))
if not versions:
    print("No versions found")
    sys.exit(1)
path = versions[-1]
print(f"Reading: {path}")

ea = EventAccumulator(path)
ea.Reload()
tags = ea.Tags()
print(f"Tags: {tags['scalars']}")

if 'train/loss' in tags['scalars']:
    events = ea.Scalars('train/loss')
    print(f"\ntrain/loss: {len(events)} events (first 10)")
    for e in events[:10]:
        print(f"  step={e.step}, value={e.value:.4f}")
    print("...")
    for e in events[-5:]:
        print(f"  step={e.step}, value={e.value:.4f}")

if 'val/loss' in tags['scalars']:
    events = ea.Scalars('val/loss')
    print(f"\nval/loss: {len(events)} events (last 5)")
    for e in events[-5:]:
        print(f"  step={e.step}, value={e.value:.4f}")

if 'perf/tokens_per_sec' in tags['scalars']:
    events = ea.Scalars('perf/tokens_per_sec')
    print(f"\nperf/tokens_per_sec: {len(events)} events (last 3)")
    for e in events[-3:]:
        print(f"  step={e.step}, value={e.value:.0f}")
