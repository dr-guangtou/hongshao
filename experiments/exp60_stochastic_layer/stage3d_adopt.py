"""exp60 Stage 3d — freeze the ADOPTED layer (Option A) and run the standard
battery with drawn populations.

THE DECISION (user, 2026-08-27): **Option A adopted** — the population layer
in its S3-strict form: the amplitude draw (exact by calibration), the size-
axis draw, and the recentred, formation-tilted core mixture, around the
UNCHANGED incumbent. Option B (the decline-realistic variant, S1-scaled) is
ON RECORD, not adopted: its calibration and its price are documented in the
README and stage3c's log, to be revisited if 3-D / multi-axis profiles
shrink the orientation component of the decline targets, or if the
mean-model question is reopened.

WHAT THIS SCRIPT DOES.
  1. Calibrates the adopted variant on the FULL sample (the held-out
     judgment in stage3c stands as the validation record; the shipped
     artifact uses all the data, standard practice) and freezes every
     constant to `outputs/hongshao_v1_layer.npz`.
  2. Draws 8 realizations and echoes the layer's headline statistics
     (in-sample echo — the honest numbers are stage3c's held-out ones).
  3. Runs `hongshao.qa.evaluate` — the battery every promoted model is shown
     on — for the incumbent mean model WITH the drawn populations passed as
     `draw_cogs`, so the standard figures show the generative layer overlaid
     the way exp41-era reports did.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp60_stochastic_layer/stage3d_adopt.py [--tables-only]
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

import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from stage3c_judge import I5, Sampler                    # noqa: E402
from hongshao import qa                                  # noqa: E402

ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
N_DRAW, SEED = 8, 42
SPEC_OUT = HERE / "outputs" / "hongshao_v1_layer.npz"
QADIR = HERE / "figures" / "qa"
RULE = "=" * 96


def main(tables_only=False):
    print(f"{RULE}\nexp60 STAGE 3d — hongshao v1 layer: freeze (Option A) "
          f"and the standard battery\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    n = len(recs)
    cz = np.load(HERE / "outputs" / "stage3a_cube.npz")
    a1 = np.load(HERE / "outputs" / "stage1_core_anatomy.npz")
    a2 = {"delta_c": np.load(HERE / "outputs" / "stage2_size_anatomy.npz"
                             )["delta_c"]}
    t0 = dict(np.load(HERE / "outputs" / "stage0_targets.npz"))
    fform = a1["fform"]
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    gal = np.where(good)[0]

    # --- full-sample calibration of the ADOPTED (S3-strict, tilted) form -- #
    shift = -float(np.median(a1["a_i"][np.isfinite(a1["a_i"])]))
    smp = Sampler(cz, a1, a2, t0, np.arange(n), fform, tilt=True, shift=shift)
    smp.sig_add = None
    smp.calibrate(gal, np.random.default_rng(SEED + 1))
    print(f"  adopted: S3-strict tilted; pool shift {shift:+.4f}; "
          f"sig_add per epoch "
          + " ".join(f"{v:.4f}" for v in smp.sig_add))

    np.savez(SPEC_OUT, variant="A_population_S3strict_tilted",
             shift=shift, scale=1.0, sig_add=smp.sig_add,
             amp_corr=t0["amp_corr"], q_edges=t0["q_edges"],
             a_pools=np.array(smp.a_pools, dtype=object),
             a_pool_all=smp.a_pool_all, dc_pool=smp.dc_pool,
             log_rc=cz["log_rc"], dc_grid=cz["dc_grid"],
             a_grid=cz["a_grid"], cube_file="stage3a_cube.npz",
             option_b_on_record="S1-scaled tilted: scale 0.20, shift "
                                "calibrated to the declining fraction "
                                "(~+0.37); see stage3c_judge.log and the "
                                "README — NOT adopted")
    print(f"  frozen -> {SPEC_OUT.relative_to(ROOT)}")

    # --- draws and the in-sample echo ------------------------------------- #
    draws = np.full((N_DRAW, n, 5, 24), np.nan, np.float32)
    for r in range(N_DRAW):
        rng = np.random.default_rng(SEED + 10 + r)
        draws[r, gal] = smp.profiles(gal, rng)
    dch = draws[:, gal, 0, I5] - draws[:, gal, 4, I5]
    fin = np.isfinite(dch)
    print(f"\n  in-sample echo over {N_DRAW} realizations (the VALIDATION "
          f"numbers are stage3c's, held out):")
    print(f"    declining {100 * np.mean(dch[fin] < 0):.1f}% "
          f"(held-out record 26.7%; target 41.8% is Option B territory)")
    dev = draws[:, gal][:, :, :, F.I100] \
        - np.log10(np.clip(slow.predict(theta)[gal][None, :, :, F.I100],
                           1.0, None))
    print(f"    amplitude sigma per epoch "
          + " ".join(f"{np.nanstd(dev[:, :, k]):.4f}" for k in range(5))
          + "  (targets " + " ".join(f"{v:.4f}" for v in t0["amp_sigma"])
          + ")")

    # --- the standard battery, drawn populations overlaid ------------------ #
    mean_cogs = slow.predict(theta)
    logms = np.log10(np.clip(data[good][:, 0, -1], 1.0, None))
    QADIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{RULE}\nSTANDARD QA — the incumbent mean with the adopted "
          f"layer's drawn populations\n{RULE}")
    qa.evaluate(mean_cogs[good], data[good], F.R_GRID, list(ANCHOR_Z),
                name="exp60_v1_layer",
                figdir=(None if tables_only else QADIR),
                figures=not tables_only, verbose=True,
                draw_cogs=10.0 ** draws[:, good].astype(np.float64),
                bin_by=lmh[good][:, 0], bin_label=r"logM$_h$(z=0.4)",
                bin_by_ms=logms, ms_label=r"logM$_*$ (total)")
    print(f"\n  figures -> {QADIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main("--tables-only" in sys.argv[1:])
