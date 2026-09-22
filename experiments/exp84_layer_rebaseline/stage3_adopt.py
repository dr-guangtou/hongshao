"""exp84 Stage 3 — package the layer (exp60 stage3d's role): the chosen variant
calibrated on the FULL fitting sample, frozen to `outputs/hongshao_v2_layer.npz`,
with `draw_cogs` as the entry point every later script uses, and the standard
QA battery with the drawn populations overlaid (`figures/qa/*exp84_v2_layer*`).

Stage 2's held-out tables are the layer's validation record; the full-sample
tables printed here are the shipped artifact's own numbers (in-sample for the
size widths, so read Stage 2 for the verdict). Adoption is the user's call.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp84_layer_rebaseline/stage3_adopt.py [--variant gauss|gauss-scaled|rows|independent|persistent]
     [--scale S] [--tables-only]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layer_common as LC                                # noqa: E402
import predictor as P                                    # noqa: E402
import stage2_judge as S2                                # noqa: E402
from sampler import Sampler                              # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE = P.RULE
OUT = P.OUTDIR / "hongshao_v2_layer.npz"
FIGDIR = P.ROOT / "figures/qa"
SEED, N_DRAW = 2084, 8


def load_layer(path=OUT):
    return dict(np.load(path, allow_pickle=True))


def draw_cogs(pred, layer, rows, rng, n_draw=1):
    """(n_draw, len(rows), 5, nr) drawn CoGs [Msun] around the adopted mean for
    the galaxies `rows` of `pred` — the layer's public entry point."""
    n = len(rows)
    sigma, corr, scale = layer["sigma"], layer["corr"], float(layer["scale"])
    L = np.linalg.cholesky(corr + 1e-9 * np.eye(10))
    L_amp = np.linalg.cholesky(layer["amp_corr"] + 1e-9 * np.eye(5))
    out = []
    for _ in range(n_draw):
        v = (rng.standard_normal((n, 10)) @ L.T) * sigma[None, :] * scale
        dev = {"c": S2._expand(v[:, :5], rows, pred.n), "e": S2._expand(v[:, 5:], rows, pred.n)}
        m = pred.predict(size_dev=dev, rows=rows)
        eps = (rng.standard_normal((n, 5)) @ L_amp.T) * layer["sig_add"][None, :]
        out.append(m * 10.0 ** eps[:, :, None])
    return np.stack(out)


def main(variant="gauss", scale=None, tables_only=False):
    kw = dict(S2.VARIANTS)[variant]
    print(f"{RULE}\nexp84 STAGE 3 — the layer packaged: variant {variant}, full-sample calibration\n{RULE}\n")
    recs, data, keep, lmh, pred = P.build(False)
    rows = np.where(keep)[0]
    n = len(rows)
    R = pred.R
    d, lmh_r = data[rows], lmh[rows]
    mean = pred.predict(rows=rows)
    a1 = np.load(P.OUTDIR / "stage1_anatomy.npz")
    t0 = np.load(P.OUTDIR / "stage0_targets.npz")
    assert np.array_equal(a1["rows"], rows)
    rng = np.random.default_rng(SEED)
    pf = lambda dev: pred.predict(size_dev={a: S2._expand(dev[a], rows, pred.n) for a in P.AXES}, rows=rows)  # noqa: E731
    smp = Sampler(a1["delta_c"], a1["delta_e"], t0["amp_sigma"], t0["amp_corr"], np.arange(n),
                  form=kw["form"], corr=kw["corr"], scale=1.0 if kw["scale"] is None else kw["scale"])
    if scale is not None:
        smp.scale = float(scale)
    elif kw["scale"] is None:
        S2.calibrate_scale(smp, pf, d, R, n, rng)
    induced = smp.calibrate_amplitude(pf, mean, n, rng)
    print(f"  {smp.describe()}")
    print(f"  amplitude: induced by the size draws " + " ".join(f"{v:.3f}" for v in induced)
          + "; sig_add " + " ".join(f"{v:.3f}" for v in smp.sig_add) + "; target " + " ".join(f"{v:.3f}" for v in smp.amp_sigma))
    if not tables_only:
        np.savez(OUT, variant=variant, form=kw["form"], corr_kind=kw["corr"], scale=smp.scale, sigma=smp.sigma,
                 corr=smp.corr, corr_fitted=smp.corr_fitted, anatomy_medians=smp.medians, sig_add=smp.sig_add,
                 amp_sigma=smp.amp_sigma, amp_corr=smp.amp_corr, rows=rows, seed=SEED,
                 mean_file=str(pred.adopted["file"].name), theta_full=pred.adopted["theta_full"],
                 note="draws centred: the anatomy's medians are NOT applied (S3-strict)")
        print(f"  frozen -> {OUT.relative_to(P.ROOT)}")
        layer = load_layer()
    else:
        layer = dict(sigma=smp.sigma, corr=smp.corr, scale=smp.scale, amp_corr=smp.amp_corr, sig_add=smp.sig_add)

    draws = draw_cogs(pred, layer, rows, np.random.default_rng(SEED + 1), n_draw=N_DRAW)
    print(f"\n  the standard battery on the MEAN with {N_DRAW} drawn populations overlaid -> {FIGDIR.relative_to(P.ROOT)}/qa_*_exp84_v2_layer.*")
    ev_mean = qa.evaluate(mean, d, R, list(P.ANCHOR_Z), name="exp84_v2_layer", figdir=str(FIGDIR),
                          verbose=False, figures=not tables_only, bin_by=lmh_r[:, 0], halo_mass_epochs=lmh_r,
                          draw_cogs=draws)
    ev = qa.evaluate_draws(draws, d, R, list(P.ANCHOR_Z), halo_mass_epochs=lmh_r)
    dg = S2.direct_gates(draws, mean, d, R, t0)
    for which, label in (("ms", "STELLAR mass"), ("mh", "HALO mass")):
        print(f"\n  tier 2d at fixed {label}, FULL SAMPLE (in-sample for the layer; Stage 2 is the held-out record)")
        qa.print_draw_gate([("mean", ev_mean[f"size_gate_{which}"][0]), (variant, ev[f"gate_{which}"])], P.ANCHOR_Z, label)
    print(f"\n  S3 {dg['s3_lowz']:.3f} (z<=1) / {dg['s3_all']:.3f} dex; S5 sigma ratio "
          + " ".join(f"{v:.2f}" for v in dg["s5_sigma"] / smp.amp_sigma) + f", nearest-epoch {dg['s5_near']:+.3f} "
          f"(target {LC.nearest_epoch(smp.amp_corr):+.3f}); S4 {dg['s4']:+.3f}; persistence "
          + "/".join(f"{dg['persistence'][k][0]:+.2f}" for k in LC.SIZE_KEYS)
          + f" (truth " + "/".join(f"{LC.nearest_epoch(t0[f'persistence_truth_{k}']):+.2f}" for k in LC.SIZE_KEYS) + ")"
          f"; S1 declining {100 * dg['s1'][0]:.1f}% (truth {100 * float(t0['s1_frac']):.1f}%)")
    print("\n  tier 2e, full sample")
    qa.print_draw_cdfs([("mean", ev_mean["cdfs"]), (variant, ev["cdfs"])], P.ANCHOR_Z)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(variant=a[a.index("--variant") + 1] if "--variant" in a else "gauss",
         scale=float(a[a.index("--scale") + 1]) if "--scale" in a else None,
         tables_only="--tables-only" in a)
