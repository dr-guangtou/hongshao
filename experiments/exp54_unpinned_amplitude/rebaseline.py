"""exp54 Stage 2 — re-baseline the record: what every absolute QA number in
exp38 / exp40 / exp47 / exp48 / exp53 becomes once the amplitude is PREDICTED.

Those numbers were all measured with `M_model(<500) == M_TNG(<500)` pinned per
galaxy per epoch, so they are **oracle-amplitude scores**: they say how well a
radial prescription distributes a mass it was handed. This stage measures the
same tiers under a realistic, correlated amplitude error and publishes the
translation.

The degradation is decomposed into three separable pieces, so a reader can see
which part of it is bookkeeping and which is real:

  oracle-500   the exp53 incumbent, pinned to the truth at 500 kpc
  oracle-100   the same, pinned to the truth at the MEASURED 100 kpc anchor
               -> isolates the change of reference radius alone
  hybrid-mean  pinned to an OUT-OF-FOLD PREDICTED amplitude (conditional mean)
  hybrid-draw  ... plus a draw from the residual distribution, preserving the
               CROSS-EPOCH correlation

Why the draw matters: a conditional mean is under-dispersed by construction, so
`hybrid-mean` understates the population scatter and flatters every
distribution tier. And the draw must be correlated ACROSS EPOCHS -- a galaxy
over-predicted at z=0.4 is over-predicted at z=0.7 -- or the mass GROWTH
becomes noise and tier 2c is scored against a defect we invented.

**No refit anywhere.** Under the split objective the shape fit is independent
of the amplitude provider (plan §4.3.2-3), so the kernel parameters are the
adopted ones throughout and every difference below is amplitude alone.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
     experiments/exp54_unpinned_amplitude/rebaseline.py [--figures]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp53_deposition_only"):
    sys.path.insert(0, str(p))

import kernel as K                                      # noqa: E402
import scoreboard as SB                                 # noqa: E402
import stage2_multiepoch as s2                          # noqa: E402
from hongshao import qa                                 # noqa: E402

ANCHOR_Z = [0.4, 0.7, 1.0, 1.5, 2.0]
KS = [0, 1, 2, 3, 4]
R_ANCHOR = 100.0
SEED = 0
RULE = "=" * 92


def amplitude_predictions(d, y):
    """Out-of-fold predicted ``log10 M*(<100)`` (n, 5), its residuals, and the
    cross-epoch residual correlation.

    Predictor = the best halo-only row on the Stage 1 scoreboard: official
    DiffMAH 4 params + concentration excess + fz2.
    """
    n = len(y)
    mu = np.full_like(y, np.nan)
    for k in range(5):
        feats = ([d["dm4"][:, j] for j in range(4)]
                 + [d["c_exc_k"][:, k], d["fz2"]])
        res, _ = SB.score_linear(y[:, k], feats)
        mu[:, k] = y[:, k] - res
    resid = y - mu
    corr = np.corrcoef(resid.T)
    return mu, resid, corr


def draw_amplitudes(mu, resid, corr, rng):
    """mu + a cross-epoch-correlated Gaussian draw with the measured per-epoch
    spread. Reproduces both the scatter and its coherence in redshift."""
    L = np.linalg.cholesky(corr + 1e-10 * np.eye(len(corr)))
    z = rng.standard_normal((len(mu), corr.shape[0]))
    return mu + (z @ L.T) * resid.std(axis=0)


def build_products(gals, spec, theta, amp_mean, amp_draw, R):
    """The four CoG products, (n, 5, 24) in Msun. Shape identical in all four;
    only the number they are scaled to differs."""
    e = s2._W["e"]
    i100 = int(np.argmin(np.abs(R - R_ANCHOR)))
    out = {k: np.full((len(gals), 5, len(R)), np.nan)
           for k in ("oracle-500", "oracle-100", "hybrid-mean", "hybrid-draw")}
    truth = np.stack([np.array([g["data"][k] for k in KS]) for g in gals])
    for i, g in enumerate(gals):
        for k in KS:
            m = SB_forward(spec, theta, g, k, e.R_EXT)
            if m is None or not np.isfinite(m).all() or m[-1] <= 0 or m[i100] <= 0:
                continue
            shape = m[:-1]
            out["oracle-500"][i, k] = shape * (g["m500"][k] / m[-1])
            out["oracle-100"][i, k] = shape * (truth[i, k, i100] / shape[i100])
            out["hybrid-mean"][i, k] = shape * (10.0 ** amp_mean[i, k] / shape[i100])
            out["hybrid-draw"][i, k] = shape * (10.0 ** amp_draw[i, k] / shape[i100])
    return out, truth


def SB_forward(spec, theta, g, k, R):
    """The kernel's unpinned dimensionless profile (contract.forward_shape)."""
    e = s2._W["e"]
    dM = K.deposit_weights(spec, theta, g)
    if dM is None:
        return None
    mah = g["mah"]
    mask = mah["snap"] <= e.ANCHOR_SNAP[k]
    B = K.basis(spec, theta, g["cond"], mah["t"], mah["t_obs"], e.pe.AT[k], R)
    return B @ (dM * mask)


# --------------------------------------------------------------------------- #
def tier_table(evals, R):
    """The translation table, tier by tier."""
    names = list(evals)
    base = names[0]

    print(f"\n{RULE}\nTIER 1+2 — aperture masses: dex scatter (bias %) at z=0.4"
          f" and z=2.0\n{RULE}")
    keys = [k for k in evals[base]["keys"]
            if np.isfinite(evals[base]["model"][k]).any()]      # skip off-grid
    print(f"  {'quantity':>16}  " + "  ".join(f"{n:>21}" for n in names))
    for key in keys:
        cells = []
        for nm in names:
            t, m = evals[nm]["truth"][key], evals[nm]["model"][key]
            c = []
            for j in (0, 4):
                b = 100 * np.nanmedian(qa.relerr(m[:, j], t[:, j]))
                s = qa.dex_scatter(m[:, j], t[:, j])
                c.append(f"{s:.3f}({b:+.0f}%)")
            cells.append(" ".join(c))
        print(f"  {key:>16}  " + "  ".join(f"{c:>21}" for c in cells))

    print(f"\n{RULE}\nTIER 2b — observational planes: energy ratio vs the "
          f"truth's split-half floor\n(1.0 = indistinguishable; LOWER IS "
          f"BETTER)\n{RULE}")
    print(f"  {'plane':>26}{'epoch':>7}  " + "  ".join(f"{n:>12}" for n in names))
    for pl in evals[base]["planes"]:
        for j in (0, 4):
            row = [evals[nm]["planes"][pl][j][1]["energy_ratio"] for nm in names]
            lab = f"{pl[0]} vs {pl[1]}"
            print(f"  {lab:>26}{ANCHOR_Z[j]:>7}  "
                  + "  ".join(f"{v:>12.2f}" for v in row))

    print(f"\n{RULE}\nTIER 2c — cross-epoch GROWTH planes (the tier the pin "
          f"made meaningless)\n{RULE}")
    print(f"  {'quantity':>14}{'epochs':>10}  " + "  ".join(f"{n:>12}" for n in names))
    for gk in evals[base]["growth"]:
        q, ref, j = gk
        row = [evals[nm]["growth"][gk]["energy_ratio"] for nm in names]
        print(f"  {q:>14}{f'{ANCHOR_Z[ref]}->{ANCHOR_Z[j]}':>10}  "
              + "  ".join(f"{v:>12.2f}" for v in row))

    print(f"\n{RULE}\nTIER 2d — MASS-SIZE planes\n{RULE}")
    print(f"  {'fraction':>12}{'epoch':>7}  " + "  ".join(f"{n:>12}" for n in names))
    for sk in evals[base]["sizes"]:
        key, j = sk
        if j not in (0, 4):
            continue
        row = [evals[nm]["sizes"][sk][1]["energy_ratio"] for nm in names]
        print(f"  {key:>12}{ANCHOR_Z[j]:>7}  "
              + "  ".join(f"{v:>12.2f}" for v in row))

    print(f"\n{RULE}\nTIER 3 — profile max|rel| (median over galaxies, R>5 kpc)"
          f"\n{RULE}")
    print(f"  {'epoch':>7}  " + "  ".join(f"{n:>12}" for n in names))
    for j in (0, 2, 4):
        row = [np.nanmedian(evals[nm]["mr_out"][:, j]) for nm in names]
        print(f"  {ANCHOR_Z[j]:>7}  " + "  ".join(f"{v:>12.4f}" for v in row))


def main(figures=False):
    s2._w_init(None)
    gals, e = s2._W["gals"], s2._W["e"]
    R = e.R
    spec = K.Spec("moffat", q_free=True)
    theta = K.from_exp38_theta(K.adopted_theta())

    d = SB.load()
    y = d["y"]
    amp_mean, resid, corr = amplitude_predictions(d, y)
    rng = np.random.default_rng(SEED)
    amp_draw = draw_amplitudes(amp_mean, resid, corr, rng)

    print(f"exp54 STAGE 2 — re-baselining the record   (n={len(gals)})")
    print(f"\namplitude provider: official DiffMAH 4 params + concentration "
          f"excess + fz2,\nout-of-fold, per epoch. Residual scatter (dex): "
          + " ".join(f"z={z}:{s:.4f}" for z, s in zip(ANCHOR_Z, resid.std(0))))
    print("\ncross-epoch residual correlation (why the draw must be "
          "correlated):")
    for j, z in enumerate(ANCHOR_Z):
        print(f"   z={z:<5}" + " ".join(f"{v:+.3f}" for v in corr[j]))

    prods, truth = build_products(gals, spec, theta, amp_mean, amp_draw, R)
    figdir = HERE / "figures" if figures else None
    evals = {}
    for nm, m in prods.items():
        evals[nm] = qa.evaluate(m, truth, R, ANCHOR_Z, name=nm, figdir=figdir,
                                verbose=False, figures=figures,
                                bin_by=d["lmh_k"][:, 0],
                                bin_label=r"logM$_h$(z=0.4)")
    tier_table(evals, R)

    print(f"\n{RULE}\nHEADLINE TRANSLATION\n{RULE}")
    o5, o1 = evals["oracle-500"], evals["oracle-100"]
    hm, hd = evals["hybrid-mean"], evals["hybrid-draw"]
    k100 = "kpc:M(<100)"
    print("  paired metrics (read hybrid-mean) — these DO degrade:")
    print(f"    tier 3 max|rel| at z=0.4      {np.nanmedian(o5['mr_out'][:, 0]):.4f}"
          f"  ->  {np.nanmedian(hm['mr_out'][:, 0]):.4f}   "
          f"({100 * (np.nanmedian(hm['mr_out'][:, 0]) / np.nanmedian(o5['mr_out'][:, 0]) - 1):+.0f}%)")
    print(f"    M(<100) dex scatter at z=0.4  "
          f"{qa.dex_scatter(o5['model'][k100][:, 0], o5['truth'][k100][:, 0]):.3f}"
          f"  ->  {qa.dex_scatter(hm['model'][k100][:, 0], hm['truth'][k100][:, 0]):.3f}"
          f"   (was pinned, now predicted)")
    pl = list(evals["oracle-500"]["planes"])[0]
    print("\n  distribution metrics (read hybrid-draw) — these BARELY degrade:")
    print(f"    tier 2b plane at z=0.4        "
          f"{o5['planes'][pl][0][1]['energy_ratio']:.2f}  ->  "
          f"{hd['planes'][pl][0][1]['energy_ratio']:.2f}   "
          f"(mean-only would say {hm['planes'][pl][0][1]['energy_ratio']:.2f})")
    print("\n  THE PLAN PREDICTED tier 2b z=0.4 would move 2.1 -> 2.9 under a")
    print("  0.137 dex amplitude error. Measured: it moves to 3.38 if the error")
    print("  is injected into the MEAN, but only to 2.20 with a properly")
    print("  cross-epoch-correlated DRAW. The predicted degradation is largely")
    print("  an artifact of the wrong operation for a pairing-blind metric.")
    gk = [k for k in o5["growth"] if k[0] == "Mtot"]
    print("\n  the tier the pin actually hid — cross-epoch MASS GROWTH:")
    for k in gk:
        print(f"    Mtot {ANCHOR_Z[k[1]]}->{ANCHOR_Z[k[2]]:<5}  "
              f"{o5['growth'][k]['energy_ratio']:.2f} (pinned, meaningless)  ->  "
              f"{hd['growth'][k]['energy_ratio']:.2f} (real)")
    print("  An energy ratio BELOW 1 means indistinguishable from the truth —")
    print("  which the pinned model achieves by construction, not by merit.")

    print(f"\n{RULE}\nHOW TO USE THIS TABLE\n{RULE}")
    print("  Every absolute number in exp38/40/47/48/53 is an `oracle-500`")
    print("  number. `oracle-100` differs from it ONLY by the reference radius;")
    print("  `hybrid-mean` adds a realistic amplitude error; `hybrid-draw` adds")
    print("  its scatter. The shape is IDENTICAL in all four columns and no")
    print("  parameter was refitted, so every difference is amplitude alone.")
    print("\n  WHICH COLUMN TO READ DEPENDS ON THE METRIC, and getting this")
    print("  backwards inverts the conclusion:")
    print("    * PAIRED metrics (tier 1+2 scatter, tier 3 max|rel|) compare a")
    print("      model galaxy to ITS OWN truth. Read `hybrid-mean`. A draw is a")
    print("      realization, not a prediction of that galaxy, so `hybrid-draw`")
    print("      adds an extra var(resid) and roughly sqrt(2)-inflates it.")
    print("    * DISTRIBUTION metrics (tiers 2b/2c/2d/2e, all energy ratios)")
    print("      compare POPULATIONS and are pairing-blind. Read `hybrid-draw`.")
    print("      `hybrid-mean` is under-dispersed by construction and is")
    print("      PENALIZED for it, which is why it scores WORSE than the draw.")


if __name__ == "__main__":
    main("--figures" in sys.argv)
