"""Run the filter dose-response experiment on real face images.

For every image: measure baseline skin tone (ITA), apply each filter at each
strength, re-measure skin tone, and score every variant with every detector.
Writes one long-format CSV row per (image, filter, strength, detector).

Examples
  python run_experiment.py --source synthetic --detectors highfreq --out results/test
  python run_experiment.py --source fairface --n-per-group 300 \
      --detectors prithivMLmods/Deep-Fake-Detector-v2-Model --out results/fairface
  python run_experiment.py --source folder --path data/my_faces --detectors highfreq
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fnf import data, detectors, filters, skintone

# Fast default: control + two doses (3 dose points per filter). The original
# 5-level sweep is still available with --strengths 0 0.25 0.5 0.75 1.0.
STRENGTHS = [0.0, 0.5, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(data.SOURCES), default="synthetic")
    ap.add_argument("--path", help="folder for --source folder")
    ap.add_argument("--n-per-group", type=int, default=75,
                    help="images per group (FairFace race, or sub-folder for --source folder)")
    ap.add_argument("--max-images", type=int, default=0,
                    help="stop after this many images (0 = no limit); use 20 for a timing test")
    ap.add_argument("--n", type=int, default=120, help="synthetic image count")
    ap.add_argument("--detectors", nargs="+", default=["highfreq"],
                    help="'highfreq' or Hugging Face model ids (optionally 'id::FakeLabel')")
    ap.add_argument("--filters", nargs="+", default=list(filters.FILTERS))
    ap.add_argument("--strengths", nargs="+", type=float, default=STRENGTHS)
    ap.add_argument("--jpeg-quality", type=int, default=85, help="0 disables the JPEG save")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--save-examples", type=int, default=6, help="images to save as filter grids")
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)

    if args.source == "fairface":
        items = data.fairface(n_per_group=args.n_per_group, seed=args.seed)
    elif args.source == "folder":
        items = data.folder(args.path, n_per_group=args.n_per_group, seed=args.seed)
    else:
        items = data.synthetic(n=args.n, seed=args.seed)

    dets = [detectors.load(d) for d in args.detectors]
    strengths = sorted(set(args.strengths) | {0.0})
    jq = args.jpeg_quality or None

    rows, buf_imgs, buf_meta = [], [], []

    def flush():
        if not buf_imgs:
            return
        for det in dets:
            scores = det.score(buf_imgs)
            for meta, s in zip(buf_meta, scores):
                for filt in meta["filters"]:
                    rows.append({**{k: v for k, v in meta.items() if k != "filters"},
                                 "filter": filt, "detector": det.name, "score": float(s)})
        buf_imgs.clear()
        buf_meta.clear()
        # Save partial results so a stopped run still leaves usable data.
        pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)

    n_versions = 1 + len(args.filters) * (len(strengths) - 1)
    print(f"{n_versions} versions per image x {len(dets)} detector(s)")

    t0, n_img = time.time(), 0
    for item in items:
        if args.max_images and n_img >= args.max_images:
            break
        img = item.pop("image")
        base_ita = skintone.ita(img)
        meta0 = {**item, "ita_base": base_ita, "band_base": skintone.band(base_ita)}
        examples = [] if n_img < args.save_examples else None

        for strength in strengths:
            # Strength 0 is the shared control: score once, attach to every filter.
            filt_list = args.filters if strength == 0 else None
            targets = [(None, filt_list)] if strength == 0 else [(f, [f]) for f in args.filters]
            for fname, attach in targets:
                v = filters.apply(img, fname or args.filters[0], 0.0 if fname is None else strength, jq)
                a = skintone.ita(v)
                buf_imgs.append(v)
                buf_meta.append({**meta0, "strength": strength, "ita_after": a,
                                 "band_after": skintone.band(a), "filters": attach})
                if examples is not None:
                    examples.append((fname or "control", strength, v))
        if examples:
            save_grid(examples, out / "examples" / f"{n_img:03d}_{item['group']}.jpg")

        n_img += 1
        if len(buf_imgs) >= args.batch:
            flush()
        if n_img % 10 == 0:
            el = time.time() - t0
            print(f"{n_img} images, {el:.0f}s elapsed, {el / n_img:.2f}s per image", flush=True)
    flush()

    df = pd.DataFrame(rows)
    df.to_csv(out / "scores.csv", index=False)
    el = time.time() - t0
    print(f"Done: {n_img} images, {len(df)} rows in {el / 60:.1f} min -> {out / 'scores.csv'}")
    if n_img:
        print(f"Per image: {el / n_img:.2f}s. Time for N images ~= N x {el / n_img:.2f}s")


def save_grid(examples, path):
    """Rows = filters, columns = strengths; control in the first column."""
    import cv2

    control = next(v for f, s, v in examples if f == "control")
    by_filter = {}
    for f, s, v in examples:
        if f != "control":
            by_filter.setdefault(f, []).append((s, v))
    rows = []
    for f, lst in by_filter.items():
        lst.sort(key=lambda t: t[0])
        rows.append(np.hstack([control] + [v for _, v in lst]))
    grid = np.vstack(rows)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
