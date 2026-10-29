"""Analyse scores.csv from run_experiment.py.

Outputs (in the same folder unless --out is given):
  fpr_by_band.csv        FPR with 95% bootstrap CI per detector/filter/strength/band
  fpr_by_group.csv       same, by dataset group (e.g. FairFace race)
  label_shift.csv        share of images whose skin-tone band changes after filtering
  measurement_gap.csv    FPR computed with post-filter band vs true (pre-filter) band
  regression.csv         logistic model: flagged ~ strength * ITA, image-clustered SEs
  fig_dose_response_<detector>.png
  fig_label_shift.png
  summary.md             plain-language headline numbers

All images are real, so a positive prediction is a false positive. The decision
threshold is fixed per detector on the unfiltered (strength 0) images so that
the pooled FPR equals --target-fpr; it is then held constant across filters,
which is how a deployed detector behaves.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fnf.skintone import BAND_ORDER


def thresholds(df, target):
    base = df[df.strength == 0].drop_duplicates(["id", "detector"])
    return base.groupby("detector").score.quantile(1 - target).to_dict()


def boot_ci(flags, n_boot=1000, seed=0):
    flags = np.asarray(flags, float)
    if len(flags) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(flags, size=(n_boot, len(flags))).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def rate_table(df, by, n_boot):
    out = []
    for keys, g in df.groupby(["detector", "filter", "strength", by]):
        lo, hi = boot_ci(g.flagged, n_boot)
        out.append(dict(zip(["detector", "filter", "strength", by], keys),
                        n=len(g), fpr=g.flagged.mean(), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(out)


def regression(df):
    """flagged ~ strength + ita_z + strength:ita_z per detector x filter.
    A negative interaction means filtering raises FPR more for darker skin
    (lower ITA). SEs are clustered by image because each image appears at
    every strength."""
    import statsmodels.formula.api as smf

    rows = []
    for (det, filt), g in df.groupby(["detector", "filter"]):
        g = g.dropna(subset=["ita_base"]).copy()
        g["ita_z"] = (g.ita_base - g.ita_base.mean()) / g.ita_base.std()
        if g.flagged.nunique() < 2:
            continue
        try:
            m = smf.logit("flagged ~ strength * ita_z", data=g).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(g["id"])[0]})
        except Exception as e:  # perfect separation etc.
            print(f"regression failed for {det}/{filt}: {e}")
            continue
        for term in m.params.index:
            rows.append(dict(detector=det, filter=filt, term=term, coef=m.params[term],
                             se=m.bse[term], p=m.pvalues[term],
                             odds_ratio=np.exp(m.params[term])))
    return pd.DataFrame(rows)


def band_colors(bands):
    """Sequential single-hue ramp, light to dark, matching the order of the
    skin-tone bands (an ordered magnitude, so one hue, not categorical)."""
    import matplotlib

    cmap = matplotlib.colormaps["copper_r"]
    steps = np.linspace(0.15, 0.95, len(BAND_ORDER))
    lut = {b: cmap(s) for b, s in zip(BAND_ORDER, steps)}
    return [lut[b] for b in bands]


def plot_dose(tab, det, path, min_n):
    import matplotlib.pyplot as plt

    t = tab[(tab.detector == det) & (tab.n >= min_n)]
    filts = list(t["filter"].unique())
    fig, axes = plt.subplots(1, len(filts), figsize=(4.2 * len(filts), 3.8), sharey=True,
                             squeeze=False)
    bands = [b for b in BAND_ORDER if b in set(t.band_base)]
    cols = dict(zip(bands, band_colors(bands)))
    for ax, f in zip(axes[0], filts):
        tf = t[t["filter"] == f]
        for b in bands:
            tb = tf[tf.band_base == b].sort_values("strength")
            if tb.empty:
                continue
            ax.fill_between(tb.strength, tb.ci_lo, tb.ci_hi, color=cols[b], alpha=0.12, lw=0)
            ax.plot(tb.strength, tb.fpr, color=cols[b], lw=2, marker="o", ms=5, label=b,
                    markeredgecolor="white", markeredgewidth=1)
        ax.set_title(f.replace("_", " + "), fontsize=10)
        ax.set_xlabel("Filter strength")
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0][0].set_ylabel("False-positive rate (real flagged as fake)")
    axes[0][-1].legend(title="Skin tone (ITA band)", fontsize=7, title_fontsize=8,
                       frameon=False, loc="upper left", bbox_to_anchor=(1.12, 1))
    fig.suptitle(f"{det}: false positives vs filter strength", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_shift(shift, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    filts = list(shift["filter"].unique())
    greys = ["#1f4e79", "#7f7f7f", "#c55a11"][:len(filts)]
    for f, c in zip(filts, greys):
        s = shift[shift["filter"] == f].sort_values("strength")
        ax.plot(s.strength, 100 * s.share_changed, lw=2, marker="o", ms=5, color=c, label=f)
    ax.set_xlabel("Filter strength")
    ax.set_ylabel("Images whose skin-tone band changes (%)")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Filters move faces between the groups being audited", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-n", type=int, default=20, help="hide cells with fewer images")
    args = ap.parse_args()
    run = Path(args.run_dir)

    df = pd.read_csv(run / "scores.csv")
    thr = thresholds(df, args.target_fpr)
    df["threshold"] = df.detector.map(thr)
    df["flagged"] = (df.score >= df.threshold).astype(int)

    by_band = rate_table(df, "band_base", args.n_boot)
    by_band.to_csv(run / "fpr_by_band.csv", index=False)
    rate_table(df, "group", args.n_boot).to_csv(run / "fpr_by_group.csv", index=False)

    one_det = df[df.detector == df.detector.iloc[0]]
    shift = (one_det.assign(changed=one_det.band_base != one_det.band_after)
             .groupby(["filter", "strength"]).changed.mean()
             .rename("share_changed").reset_index())
    shift.to_csv(run / "label_shift.csv", index=False)

    # The measurement paradox: an auditor who labels skin tone from the image
    # they are scoring sees the post-filter band, not the person's real one.
    true_tab = rate_table(df, "band_base", 200).rename(columns={"band_base": "band", "fpr": "fpr_true_band"})
    seen_tab = rate_table(df, "band_after", 200).rename(columns={"band_after": "band", "fpr": "fpr_seen_band"})
    gap = true_tab[["detector", "filter", "strength", "band", "n", "fpr_true_band"]].merge(
        seen_tab[["detector", "filter", "strength", "band", "fpr_seen_band"]],
        on=["detector", "filter", "strength", "band"], how="outer")
    gap["difference"] = gap.fpr_seen_band - gap.fpr_true_band
    gap.to_csv(run / "measurement_gap.csv", index=False)

    reg = regression(df)
    reg.to_csv(run / "regression.csv", index=False)

    for det in df.detector.unique():
        plot_dose(by_band, det, run / f"fig_dose_response_{det}.png", args.min_n)
    plot_shift(shift, run / "fig_label_shift.png")

    write_summary(run, df, by_band, shift, reg, thr, args)
    print((run / "summary.md").read_text())


def write_summary(run, df, by_band, shift, reg, thr, args):
    L = [f"# Results summary: {run.name}", "",
         f"Images: {df.id.nunique()} | detectors: {', '.join(df.detector.unique())} | "
         f"threshold set for pooled {args.target_fpr:.0%} FPR on unfiltered images", ""]
    for det in df.detector.unique():
        L.append(f"## {det} (threshold {thr[det]:.3f})")
        t = by_band[(by_band.detector == det) & (by_band.n >= args.min_n)]
        smax = t.strength.max()
        for f in t["filter"].unique():
            a = t[(t["filter"] == f) & (t.strength == 0)].set_index("band_base").fpr
            b = t[(t["filter"] == f) & (t.strength == smax)].set_index("band_base").fpr
            parts = [f"{band}: {a[band]:.1%} -> {b[band]:.1%}"
                     for band in BAND_ORDER if band in a.index and band in b.index]
            L.append(f"- **{f}** (strength 0 -> {smax}): " + "; ".join(parts))
        r = reg[(reg.detector == det) & (reg.term == "strength:ita_z")]
        for _, row in r.iterrows():
            direction = ("hits darker skin harder" if row.coef < 0
                         else "hits lighter skin harder")
            L.append(f"- Interaction ({row['filter']}): OR={row.odds_ratio:.2f} per SD of ITA, "
                     f"p={row.p:.3g} -> filter effect {direction}")
        L.append("")
    s = shift[shift.strength == shift.strength.max()]
    L.append("## Label shift at max strength")
    for _, row in s.iterrows():
        L.append(f"- {row['filter']}: {row.share_changed:.1%} of faces change skin-tone band")
    L.append("")
    L.append("Bands with fewer than "
             f"{args.min_n} images are hidden. Check CIs in fpr_by_band.csv before claiming a gap.")
    (run / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
