"""exp84 — the figures that show what the layer does, in plain terms.

  1. `exp84_size_scatter`   at each epoch and for R20 / R50 / R80, the spread
     of galaxy sizes at fixed stellar mass: TNG (grey), the mean model (blue,
     visibly too narrow), and the layer's draws (red) — the quantity the
     width sub-gate measures.
  2. `exp84_width_summary`  the same as one number per epoch — the model's
     size spread over TNG's, pass band 0.8–1.2 — for the mean, the old v1
     layer on its own mean, and the new layer (held out); at fixed stellar
     mass and at fixed halo mass; plus how much a galaxy's size rank
     persists from one epoch to the next (TNG vs the layer and its controls).
  3. `exp84_examples`       four galaxies: the measured profile at five
     epochs, the mean model, and eight draws — what "a draw is a
     realization, not a prediction" looks like.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp84_layer_rebaseline/figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layer_common as LC                                # noqa: E402
import predictor as P                                    # noqa: E402
import stage3_adopt as S3                                # noqa: E402
from hongshao import qa                                  # noqa: E402

FIGDIR = HERE / "figures/qa"                             # the experiment's own figures, never the repo-level figures/qa
EPOCH_COLORS = qa._EPOCH_COLORS
C_TRUTH, C_MEAN, C_LAYER, C_V1 = "#777777", "#0072B2", "#D55E00", "#999999"
ROUND1 = P.OUTDIR / "stage2_judge.npz"
ROUND2 = P.OUTDIR / "stage2_judge_gauss-centred-gauss-2scale-gauss-2sc-cent.npz"
N_DRAW, SEED = 8, 7


def gate_from_npz(z, name, which):
    keys = [str(k) for k in z[f"gate_ms_{name}_keys"]]
    vals = np.asarray(z[f"gate_{which}_{name}_vals"], float)
    return {(k.split("|")[0], int(k.split("|")[1])): vals[i] for i, k in enumerate(keys)}


def size_scatter_figure(d, mean, draws, R):
    lm_t = np.log10(np.clip(d[:, :, -1], 1.0, None))
    lm_m = np.log10(np.clip(mean[:, :, -1], 1.0, None))
    ls_t, ls_m = LC.log_sizes(d, R), LC.log_sizes(mean, R)
    fig, axes = plt.subplots(3, 5, figsize=(16, 8.4), sharey="row")
    for i, key in enumerate(LC.SIZE_KEYS):
        rt = LC.size_residuals(ls_t[key], lm_t)
        rm = LC.size_residuals(ls_m[key], lm_m)
        rl = np.concatenate([LC.size_residuals(LC.log_sizes(dr, R, (key,))[key],
                                               np.log10(np.clip(dr[:, :, -1], 1.0, None))) for dr in draws])
        for j in range(5):
            ax = axes[i, j]
            lim = 3.2 * np.nanstd(rt[:, j])
            bins = np.linspace(-lim, lim, 41)
            ax.hist(rt[:, j], bins, density=True, color=C_TRUTH, alpha=0.45, label="TNG (truth)")
            ax.hist(rm[:, j], bins, density=True, histtype="step", lw=1.8, color=C_MEAN, label="mean model")
            ax.hist(rl[:, j], bins, density=True, histtype="step", lw=1.8, color=C_LAYER, label="mean + layer")
            st, sm, sl = np.nanstd(rt[:, j]), np.nanstd(rm[:, j]), np.nanstd(rl[:, j])
            ax.text(0.03, 0.95, f"spread / TNG's\nmean {sm / st:.2f}\nlayer {sl / st:.2f}", transform=ax.transAxes,
                    va="top", fontsize=8.5, color="k")
            if i == 0:
                ax.set_title(f"z = {P.ANCHOR_Z[j]}", color=EPOCH_COLORS[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{key}: galaxies per dex", fontsize=10)
            if i == 2:
                ax.set_xlabel(qa._tex(f"log {key} minus the mass-size relation [dex]"), fontsize=9)
            ax.set_xlim(-lim, lim)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("exp84 — how spread out are galaxy sizes at a fixed stellar mass? TNG vs the mean model vs the mean with the "
                 "stochastic layer (8 draws)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"exp84_size_scatter.{ext}", dpi=150)
    plt.close(fig)


def width_summary_figure():
    z1, z2 = np.load(ROUND1, allow_pickle=True), np.load(ROUND2, allow_pickle=True)
    t0 = np.load(P.OUTDIR / "stage0_targets.npz")
    mean_ms = {tuple(k.split("|")) if False else (k.split("|")[0], int(k.split("|")[1])): v
               for k, v in zip([str(s) for s in t0["gate_ms_keys"]], np.asarray(t0["gate_ms_vals"], float))}
    mean_mh = {(k.split("|")[0], int(k.split("|")[1])): v
               for k, v in zip([str(s) for s in t0["gate_mh_keys"]], np.asarray(t0["gate_mh_vals"], float))}
    layer_ms, layer_mh = gate_from_npz(z2, "gauss-2scale", "ms"), gate_from_npz(z2, "gauss-2scale", "mh")
    first_ms, first_mh = gate_from_npz(z1, "gauss", "ms"), gate_from_npz(z1, "gauss", "mh")
    v1 = {}
    e73 = P.ROOT / "experiments/exp73_size_relative/outputs/size_gate_layer.npz"
    if e73.exists():
        zz = np.load(e73, allow_pickle=True)
        for key, (o, w, so, sw) in zip(zz["keys"], zz["vals"]):
            label, k, j, src = str(key).split("|")
            if src == "layer":
                v1[("ms" if label.startswith("STELLAR") else "mh", k, int(j))] = w
    zs = np.array(P.ANCHOR_Z)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5))
    for i, key in enumerate(LC.SIZE_KEYS):
        for r, (which, mean_g, lay_g, first_g, label) in enumerate(
                (("ms", mean_ms, layer_ms, first_ms, "at fixed stellar mass"),
                 ("mh", mean_mh, layer_mh, first_mh, "at fixed halo mass"))):
            ax = axes[r, i]
            ax.axhspan(0.8, 1.2, color="#009E73", alpha=0.12, lw=0)
            ax.axhline(1.0, color="k", lw=0.8)
            ax.plot(zs, [mean_g[(key, j)][1] for j in range(5)], "o-", color=C_MEAN, lw=2, label="mean model (no layer)")
            if v1:
                ax.plot(zs, [v1.get((which, key, j), np.nan) for j in range(5)], "s--", color=C_V1, lw=1.5,
                        label="old v1 layer (on its own, older mean)")
            ax.plot(zs, [first_g[(key, j)][1] for j in range(5)], "^:", color="#CC79A7", lw=1.5,
                    label="new layer, first try (both size draws)")
            ax.errorbar(zs, [lay_g[(key, j)][1] for j in range(5)], yerr=[lay_g[(key, j)][2] for j in range(5)],
                        fmt="D-", color=C_LAYER, lw=2.2, ms=6, label="new layer, final (held out)")
            ax.set_ylim(0.1, 2.4)
            ax.set_xticks(zs)
            ax.tick_params(labelsize=9)
            ax.set_title(f"{key} {label}", fontsize=11)
            if i == 0:
                ax.set_ylabel("model's size spread / TNG's\n(green band = pass)", fontsize=10)
            ax.set_xlabel("redshift", fontsize=10)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="upper right", frameon=False)
    # row 3: the persistence of a galaxy's size rank from one epoch to the next
    pt = {k: LC.nearest_epoch(t0[f"persistence_truth_{k}"]) for k in LC.SIZE_KEYS}
    pm = {k: LC.nearest_epoch(t0[f"persistence_mean_{k}"]) for k in LC.SIZE_KEYS}
    rows = [("TNG (truth)", pt, C_TRUTH), ("mean model", pm, C_MEAN)]
    for name, src, col in (("new layer (final)", z2, C_LAYER), ("control: epochs drawn independently", z1, "#CC79A7"),
                           ("control: one size for life", z1, "#E69F00")):
        vn = {"new layer (final)": "gauss-2scale", "control: epochs drawn independently": "independent",
              "control: one size for life": "persistent"}[name]
        s = np.asarray(src[f"summary_{vn}"], float)          # [.., persistence R20, R50, R80 at 9:12]
        rows.append((name, {k: s[9 + a] for a, k in enumerate(LC.SIZE_KEYS)}, col))
    for i, key in enumerate(LC.SIZE_KEYS):
        ax = axes[2, i]
        ax.bar(range(len(rows)), [r[1][key] for r in rows], color=[r[2] for r in rows], alpha=0.85)
        ax.axhline(pt[key], color=C_TRUTH, ls="--", lw=1)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(f"{key}: does a galaxy's size rank persist to the next epoch?", fontsize=10)
        if i == 0:
            ax.set_ylabel("rank correlation, neighbouring epochs", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("exp84 — the size-spread gate per epoch (rows 1-2) and the persistence of size (row 3)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"exp84_width_summary.{ext}", dpi=150)
    plt.close(fig)


def examples_figure(d, mean, draws, R, lmh, rng):
    n = len(d)
    # four galaxies: a light, a typical, a heavy halo, and one whose centre declines
    with np.errstate(invalid="ignore", divide="ignore"):
        dc = np.log10(d[:, 0, LC.I5] / d[:, 4, LC.I5])
    order = np.argsort(lmh[:, 0])
    picks = [order[int(0.1 * n)], order[int(0.5 * n)], order[int(0.9 * n)],
             int(np.nanargmin(np.where(np.isfinite(dc), dc, np.inf)))]
    labels = ["light halo", "typical halo", "heavy halo", "strongest central decline"]
    fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharex=True)
    for r, (i, lab) in enumerate(zip(picks, labels)):
        for j in range(5):
            ax = axes[r, j]
            for s in range(draws.shape[0]):
                ax.plot(R, np.log10(np.clip(draws[s, i, j], 1.0, None)), color=C_LAYER, alpha=0.35, lw=1)
            ax.plot(R, np.log10(np.clip(mean[i, j], 1.0, None)), color=C_MEAN, lw=2.2, label="mean model")
            ax.plot(R, np.log10(np.clip(d[i, j], 1.0, None)), color="k", lw=2.2, ls="--", label="TNG (truth)")
            ax.set_xscale("log")
            lo = np.nanmin(np.log10(np.clip(d[i, j], 1.0, None))) - 0.3
            hi = np.nanmax(np.log10(np.clip(d[i, j], 1.0, None))) + 0.3
            ax.set_ylim(lo, hi)
            if r == 0:
                ax.set_title(f"z = {P.ANCHOR_Z[j]}", color=EPOCH_COLORS[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(qa._tex(f"{lab}\nlog M*(<R) [Msun]"), fontsize=9)
            if r == 3:
                ax.set_xlabel("R [kpc]", fontsize=10)
            ax.grid(alpha=0.3)
    axes[0, 0].plot([], [], color=C_LAYER, alpha=0.6, lw=1, label="8 draws (mean + layer)")
    axes[0, 0].legend(fontsize=8, loc="lower right", frameon=False)
    fig.suptitle("exp84 — four galaxies: the measured profile, the mean model, and eight draws of the layer "
                 "(a draw is a possible galaxy with this halo, not a prediction of this one)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"exp84_examples.{ext}", dpi=150)
    plt.close(fig)


def main(summary_only=False):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("exp84 figures")
    width_summary_figure()
    print(f"  wrote {FIGDIR / 'exp84_width_summary.png'}")
    if summary_only:
        return
    recs, data, keep, lmh, pred = P.build(False, verbose=False)
    rows = np.where(keep)[0]
    d, lmh_r = data[rows], lmh[rows]
    mean = pred.predict(rows=rows)
    layer = S3.load_layer()
    draws = S3.draw_cogs(pred, layer, rows, np.random.default_rng(SEED), n_draw=N_DRAW)
    size_scatter_figure(d, mean, draws, pred.R)
    print(f"  wrote {FIGDIR / 'exp84_size_scatter.png'}")
    examples_figure(d, mean, draws, pred.R, lmh_r, np.random.default_rng(SEED))
    print(f"  wrote {FIGDIR / 'exp84_examples.png'}")


if __name__ == "__main__":
    main(summary_only="--summary-only" in sys.argv[1:])
