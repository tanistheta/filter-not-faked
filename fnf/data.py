"""Image sources. Each loader yields dicts with keys: id, image (RGB uint8),
group (dataset demographic label, e.g. FairFace race), plus any extra labels.

All images here are REAL photos, so every 'fake' prediction is a false positive.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def fairface(n_per_group: int = 300, split: str = "validation", seed: int = 0,
             size: int = 224, padding: str = "0.25"):
    """Stratified sample from FairFace on the Hugging Face hub
    (HuggingFaceM4/FairFace). Group names come from the dataset's own
    ClassLabel metadata, so nothing about label order is hard-coded."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/FairFace", padding, split=split)
    race_names = ds.features["race"].names
    gender_names = ds.features["gender"].names
    age_names = ds.features["age"].names

    rng = np.random.default_rng(seed)
    races = np.array(ds["race"])
    for r in range(len(race_names)):
        idx = np.flatnonzero(races == r)
        take = rng.choice(idx, size=min(n_per_group, len(idx)), replace=False)
        for i in take:
            row = ds[int(i)]
            img = np.array(row["image"].convert("RGB").resize((size, size)))
            yield {
                "id": f"fairface_{split}_{int(i)}",
                "image": img,
                "group": race_names[r],
                "gender": gender_names[row["gender"]],
                "age": age_names[row["age"]],
            }


def folder(path: str, size: int = 224, n_per_group: int | None = None, seed: int = 0):
    """Images from a local folder. Sub-folder name is used as the group,
    e.g. data/indian/*.jpg, data/white/*.jpg. Files at the top level get
    group 'all'. If n_per_group is set, a seeded random sample of that many
    images is taken from each group (empty sub-folders are skipped)."""
    from PIL import Image

    root = Path(path)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    groups: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        rel = p.relative_to(root)
        group = rel.parts[0] if len(rel.parts) > 1 else "all"
        groups.setdefault(group, []).append(p)

    rng = np.random.default_rng(seed)
    for group, paths in groups.items():
        if n_per_group and len(paths) > n_per_group:
            idx = sorted(rng.choice(len(paths), size=n_per_group, replace=False))
            paths = [paths[i] for i in idx]
        print(f"[data] {group}: {len(paths)} images")
        for p in paths:
            img = np.array(Image.open(p).convert("RGB").resize((size, size)))
            yield {"id": str(p.relative_to(root)), "image": img, "group": group}


def synthetic(n: int = 120, size: int = 224, seed: int = 0):
    """Fake 'faces' for testing only: skin-coloured ellipses with texture,
    spread across light to dark tones. Never use for results."""
    import cv2

    rng = np.random.default_rng(seed)
    # RGB skin tones from light to dark.
    tones = np.array([[241, 194, 167], [224, 172, 138], [198, 134, 99],
                      [161, 102, 70], [120, 74, 50], [80, 50, 35]], float)
    for k in range(n):
        t = k % len(tones)
        img = np.full((size, size, 3), 40, np.uint8)
        base = tones[t] + rng.normal(0, 6, 3)
        cv2.ellipse(img, (size // 2, size // 2), (size // 3, int(size / 2.4)), 0, 0, 360,
                    tuple(int(c) for c in np.clip(base, 0, 255)), -1)
        noise = rng.normal(0, 9, (size, size, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        yield {"id": f"synth_{k}", "image": img, "group": f"tone{t}"}


SOURCES = {"fairface": fairface, "folder": folder, "synthetic": synthetic}
