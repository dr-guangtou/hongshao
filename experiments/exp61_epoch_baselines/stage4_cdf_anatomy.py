"""exp61 Stage 4 — WHY the per-epoch ceiling is WORSE on the mass CDFs.

The user's question, after reading the Stage 3 battery: the per-epoch fits
improve the average curves of growth in places, but tier 2e (the population
distributions of the aperture and annulus masses) shows them WORSE than the
shared law in several cells. Why?

A CDF distance can grow for two quite different reasons, and tier 2e's single
number does not separate them:

  LOCATION  the model's whole distribution is shifted — it puts systematically
            too much or too little mass in that bin. W1 is the mean horizontal
            gap in dex, so a pure shift of d dex gives W1 = |d| exactly.
  SHAPE     the model's distribution has the wrong WIDTH or wrong tails even
            after the shift is removed. A conditional-mean model is
            under-dispersed by construction (it predicts the mean, so it
            cannot reproduce the scatter around it), and a fit that tracks the
            conditional mean harder is MORE under-dispersed, not less.

This stage measures both, for every tier 2e quantity, epoch and model:

  * `dmed`   median(log10 model) − median(log10 truth) [dex] — the shift.
  * `spread` the model's robust population width divided by the truth's,
             (P84−P16)/2 in log10 mass. 1.0 = right width; below 1 =
             under-dispersed, which is the conditional-mean signature.
  * `w1`     the tier 2e number, as a ratio to the truth's own split-half
             floor, exactly as `qa.cdf_distances` computes it.
  * `w1_c`   the same after subtracting each sample's OWN median, so the
             location part is gone and only the shape mismatch remains.

Then w1 - w1_c is the part of the tier 2e distance that is a shift, and w1_c
is the part that is a width or tail error. A model can be worse for either
reason and the fix is different in each case.

GATE. The recomputed `w1` must reproduce the tier 2e table `qa.evaluate`
printed in `outputs/stage3_qa.log`, or this script is measuring something
else and stops.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp61_epoch_baselines/stage4_cdf_anatomy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", HERE):
    sys.path.insert(0, str(p))

import matplotlib.pyplot as plt                          # noqa: E402

import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from hongshao import qa                                  # noqa: E402
from hongshao.plotting import save_fig, set_style        # noqa: E402
from stage3_qa import MODELS, PRETTY, COLORS, build_cogs  # noqa: E402

ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
FIGDIR = HERE / "figures"
OUT = HERE / "outputs" / "stage4_cdf_anatomy.npz"
#: the quantities the figure shows: the inner aperture the user asked about,
#: the two outer annuli where the ceiling is worst, and the Re-relative
#: annulus where its z=2 cell moves furthest
SHOW = ("kpc:M(<10)", "kpc:M(30-50)", "kpc:M(50-100)", "Re:M(2-4Re)")
ZLE_SHORT = r"$z \leq 1.0$"
#: the incumbent's tier 2e W1 ratios from outputs/stage3_qa.log, read first
LOG_W1_INCUMBENT = {
    "kpc:M(<10)": (1.8, 1.5, 1.6, 2.7, 5.8),
    "kpc:M(<30)": (1.8, 1.6, 1.8, 1.7, 4.4),
    "kpc:M(30-50)": (5.1, 5.8, 5.2, 7.7, 6.0),
    "kpc:M(50-100)": (5.9, 6.0, 5.6, 8.2, 6.2),
    "Re:M(<2Re)": (2.0, 1.2, 1.5, 4.2, 6.9),
    "Re:M(2-4Re)": (4.1, 3.8, 4.5, 3.9, 3.2),
}
RULE = "=" * 96


def spread(v):
    """Robust population width of a 1-D sample, in the sample's own units."""
    p16, p84 = np.nanpercentile(v, [16.0, 84.0])
    return float(p84 - p16) / 2.0


def anatomy(truth, model, key, j):
    """The four numbers, for one quantity at one epoch."""
    tv, mv = truth[key][:, j], model[key][:, j]
    ok_t = np.isfinite(tv) & (tv > 1.0)
    ok_m = np.isfinite(mv) & (mv > 1.0)
    t = np.log10(np.clip(tv[ok_t], 1.0, None))
    m = np.log10(np.clip(mv[ok_m], 1.0, None))
    d = qa.cdf_distances(t, m, seed=j)
    dc = qa.cdf_distances(t - np.median(t), m - np.median(m), seed=j)
    return dict(dmed=float(np.median(m) - np.median(t)),
                spread=spread(m) / max(spread(t), 1e-12),
                w1=d["w1_ratio"], w1_c=dc["w1_ratio"],
                ks=d["ks_ratio"], w1_dex=d["w1"])


def main():
    print(f"{RULE}\nexp61 STAGE 4 — the anatomy of the tier 2e CDF distances"
          f"\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    R = F.R_GRID
    cogs = build_cogs(recs, data, mask, base, theta)
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    print(f"  {int(good.sum())} galaxies, the Stage 3 sample exactly")

    res = {}
    for name in MODELS:
        truth, model, _, keys = qa.measure_all(cogs[name][good], data[good], R)
        for key in qa.CDF_KEYS:
            for j in range(5):
                res[(name, key, j)] = anatomy(truth, model, key, j)

    print("\n  GATE — recomputed W1 ratios against the tier 2e table in "
          "outputs/stage3_qa.log")
    worst = 0.0
    for key, ref in LOG_W1_INCUMBENT.items():
        got = [res[("incumbent", key, j)]["w1"] for j in range(5)]
        worst = max(worst, max(abs(g - r) for g, r in zip(got, ref)))
        print(f"    {key:<16}" + "".join(f"{g:>7.1f}" for g in got)
              + "   log " + "".join(f"{r:>7.1f}" for r in ref))
    print(f"    worst difference {worst:.2f} (the log prints one decimal)")
    if worst > 0.06:
        raise SystemExit("REFUSING TO CONTINUE: the recomputed tier 2e "
                         "distances do not reproduce the battery's own table.")

    for key in qa.CDF_KEYS:
        print(f"\n{RULE}\n  {key}\n{RULE}")
        print(f"  {'model':<16}{'measure':<28}"
              + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for name in MODELS:
            r = [res[(name, key, j)] for j in range(5)]
            print(f"  {name:<16}{'shift of the median [dex]':<28}"
                  + "".join(f"{v['dmed']:>+10.3f}" for v in r))
            print(f"  {'':<16}{'width / truth width':<28}"
                  + "".join(f"{v['spread']:>10.2f}" for v in r))
            print(f"  {'':<16}{'W1 / floor  (tier 2e)':<28}"
                  + "".join(f"{v['w1']:>10.1f}" for v in r))
            print(f"  {'':<16}{'W1 / floor, median removed':<28}"
                  + "".join(f"{v['w1_c']:>10.1f}" for v in r))

    figure(res)
    np.savez(OUT, keys=np.array(list(qa.CDF_KEYS)), names=np.array(MODELS),
             **{f"{q}_{name}_{key}": np.array(
                 [res[(name, key, j)][q] for j in range(5)])
                for q in ("dmed", "spread", "w1", "w1_c")
                for name in MODELS for key in qa.CDF_KEYS})
    print(f"\n  saved {OUT.relative_to(ROOT)}")


def figure(res, name="exp61_cdf_anatomy"):
    z = np.array(ANCHOR_Z)
    set_style()
    fig, axes = plt.subplots(2, len(SHOW), figsize=(16.0, 8.2))
    for c, key in enumerate(SHOW):
        top, bot = axes[0, c], axes[1, c]
        for mdl in ("incumbent", "per-epoch-best"):
            w1 = [res[(mdl, key, j)]["w1"] for j in range(5)]
            w1c = [res[(mdl, key, j)]["w1_c"] for j in range(5)]
            top.plot(z, w1, "-o", lw=1.8, ms=6, color=COLORS[mdl], zorder=3,
                     label=(PRETTY[mdl] if c == 0 else None))
            top.plot(z, w1c, "--o", lw=1.2, ms=5, color=COLORS[mdl],
                     mfc="white", mew=1.3, zorder=3,
                     label=(f"{mdl}, median removed" if c == 0 else None))
        top.axhline(1.0, color="0.55", lw=1.0, ls=":", zorder=1)
        top.set_title(qa._tex(key) + "\nhow far the predicted population is "
                      "from the true one", fontsize=9)
        top.set_xticks(z)
        top.set_xticklabels([f"{v:.1f}" for v in z], fontsize=7.5)
        top.set_ylim(bottom=0)
        if c == 0:
            top.set_ylabel("distance between the two distributions,\n"
                           "in units of the truth's own split-half noise")
            top.legend(loc="upper left", fontsize=7)

        for mdl in MODELS:
            sp = [res[(mdl, key, j)]["spread"] for j in range(5)]
            bot.plot(z, sp, "-o", lw=1.8, ms=6, color=COLORS[mdl], zorder=3,
                     label=(PRETTY[mdl] if c == 0 else None))
        bot.axhline(1.0, color="0.55", lw=1.2, ls="--", zorder=1)
        bot.set_title("how WIDE the predicted population is", fontsize=9)
        bot.set_xlabel("redshift of the observed profile")
        bot.set_xticks(z)
        bot.set_xticklabels([f"{v:.1f}" for v in z], fontsize=7.5)
        if c == 0:
            bot.set_ylabel("model population width / true width\n"
                           "(1.0 = right width; below 1 = too narrow)")
            bot.legend(loc="lower left", fontsize=7)
        if key == "kpc:M(50-100)":
            w = [res[(m, key, 2)]["spread"] for m in MODELS]
            lo, hi = bot.get_ylim()
            bot.set_ylim(lo, hi + 0.62 * (hi - lo))
            bot.annotate(
                "At z=1.0 every one of the three fits includes this epoch, "
                "and the\nwidth falls with how large a share of the fit the "
                f"epoch got:\n{w[0]:.2f} sharing five epochs, {w[2]:.2f} "
                f"sharing three, {w[1]:.2f} with the\nepoch to itself.\n\n"
                "At z $\\geq$ 1.5 the " + ZLE_SHORT + " refit was not fitted "
                "here at all,\nand its width goes back to the shared law's. "
                "That is the control\nthat makes the narrowing the FIT's "
                "doing and not the epoch's.",
                xy=(0.02, 0.97), xycoords="axes fraction", fontsize=7.5,
                color="0.25", va="top")

    fig.suptitle("exp61 — why the per-epoch ceiling scores WORSE on the mass "
                 "distributions, measured.\nTOP: the solid line is the tier "
                 "2e distance; the hollow dashed line is the same after each "
                 "distribution is centred on its own median, so the gap "
                 "between them is\nthe part that is a systematic shift and "
                 "what is left is a wrong WIDTH.  BOTTOM: the width itself. "
                 "Every model is too narrow — a law that predicts the "
                 "conditional\nmean cannot reproduce the scatter around it "
                 "— and in the outer annuli, which the loss barely weighs, "
                 "the narrowing grows with how hard that epoch was fitted.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for p in save_fig(fig, FIGDIR / name):
        print(f"  wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
