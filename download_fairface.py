"""Download 300 FairFace validation images per race group into data/fairface/<race>/.

Run from a terminal:  python download_fairface.py
"""
from pathlib import Path

from datasets import load_dataset

PER_GROUP = 300

ds = load_dataset("HuggingFaceM4/FairFace", "0.25", split="validation")
races = ds.features["race"].names

counts = {}
for i, row in enumerate(ds):
    r = races[row["race"]]
    if counts.get(r, 0) >= PER_GROUP:
        continue
    d = Path("data/fairface") / r
    d.mkdir(parents=True, exist_ok=True)
    row["image"].convert("RGB").save(d / f"{i}.jpg")
    counts[r] = counts.get(r, 0) + 1
    if all(counts.get(x, 0) >= PER_GROUP for x in races):
        break

print(counts)
