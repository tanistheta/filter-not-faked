"""Secondary analysis (added after the main run): skin tone as relative ITA
tertiles instead of absolute Chardon bands, plus absolute-scale (percentage
point) effects. ITA on in-the-wild photos is shifted dark in absolute terms,
so tertiles of the sample's own ITA are the safer grouping.

Usage: python tertile_analysis.py results/fairface
Writes tertile_fpr.csv, tertile_effects.csv, fig_tertile_dose.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

run = Path(sys.argv[1])
s = pd.read_csv(run / "scores.csv")
rows_fpr, rows_eff = [], []
TERTS = ["darkest third", "middle third", "lightest third"]
for det, d0 in s.groupby("detector"):
    thr = d0[d0.strength == 0].drop_duplicates("id").score.quantile(0.95)
    d0 = d0.assign(flagged=(d0.score >= thr).astype(int))
    base = d0[d0.strength == 0].drop_duplicates("id").dropna(subset=["ita_base"])
    q = base.ita_base.quantile([1 / 3, 2 / 3]).values
    d0["tert"] = pd.cut(d0.ita_base, [-np.inf, q[0], q[1], np.inf], labels=TERTS)
    rng = np.random.default_rng(0)
    for (f, t, st), g in d0.dropna(subset=["tert"]).groupby(["filter", "tert", "strength"], observed=True):
        bs = rng.choice(g.flagged.values, (1000, len(g))).mean(1)
        rows_fpr.append(dict(detector=det, filter=f, tertile=t, strength=st, n=len(g),
                             fpr=g.flagged.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    for f, d in d0[d0.tert.notna()].groupby("filter"):
        piv = d.pivot_table(index="id", columns="strength", values="flagged")
        tert = d.drop_duplicates("id").set_index("id").tert.reindex(piv.index)
        delta = piv[piv.columns.max()] - piv[0.0]
        D, L = delta[tert == TERTS[0]].values, delta[tert == TERTS[2]].values
        bs = [rng.choice(D, len(D)).mean() - rng.choice(L, len(L)).mean() for _ in range(2000)]
        g = d.dropna(subset=["ita_base"]).copy()
        g["ita_z"] = (g.ita_base - g.ita_base.mean()) / g.ita_base.std()
        m = smf.ols("flagged ~ strength * ita_z", data=g).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(g.id)[0]})
        rows_eff.append(dict(detector=det, filter=f, increase_darkest_pp=100 * D.mean(),
                             increase_lightest_pp=100 * L.mean(), diff_pp=100 * (D.mean() - L.mean()),
                             diff_ci_lo=np.percentile(bs, 2.5) * 100, diff_ci_hi=np.percentile(bs, 97.5) * 100,
                             lpm_interaction=m.params["strength:ita_z"], lpm_p=m.pvalues["strength:ita_z"]))
fpr, eff = pd.DataFrame(rows_fpr), pd.DataFrame(rows_eff)
fpr.to_csv(run / "tertile_fpr.csv", index=False)
eff.to_csv(run / "tertile_effects.csv", index=False)
print(eff.round(3).to_string())

import matplotlib.pyplot as plt
det = fpr.detector.iloc[0]
cols = {"darkest third": "#5a3410", "middle third": "#a0652e", "lightest third": "#e0b48a"}
filts = ["smooth", "lighten", "smooth_lighten"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, f in zip(axes, filts):
    t = fpr[(fpr.detector == det) & (fpr["filter"] == f)]
    for k in TERTS:
        tk = t[t.tertile == k].sort_values("strength")
        ax.fill_between(tk.strength, 100 * tk.ci_lo, 100 * tk.ci_hi, color=cols[k], alpha=0.15, lw=0)
        ax.plot(tk.strength, 100 * tk.fpr, color=cols[k], lw=2.2, marker="o", label=k)
    ax.set_title(f.replace("_", " + "))
    ax.set_xlabel("Filter strength")
    ax.set_xticks([0, 0.5, 1])
    ax.grid(axis="y", color="#e5e5e5")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Real faces flagged as fake (%)")
axes[-1].legend(title="Skin tone (ITA)", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(run / "fig_tertile_dose.png", dpi=180, bbox_inches="tight")
