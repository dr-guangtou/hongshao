"""exp61 Stage 2a — the numbers the two tables never saved, for the figures.

`stage0_fits.npz` holds each fit's own theta, its own loss and its own scores;
`stage1_mx2_fit.npz` holds the curvature fit's theta and starts. Three things
the figures need are in neither file, and all three are cheap re-evaluations
of thetas that are already frozen — no fitting happens here:

  1. THE CROSS-EPOCH LOSS MATRIX. Every one of the seven parameter sets
     (the shared incumbent, the five single-epoch fits, the z <= 1.0 fit)
     evaluated at every one of the five epochs. The figure reports each cell
     as the share of that parameter set's loss at that epoch which a fit
     devoted to the epoch alone removes,

         recoverable[i, j] = 100 * (1 - loss_j(theta_j*) / loss_j(theta_i)) ,

     so the diagonal is 0 by construction and the top row is exactly the
     "+4.48 / +2.33 / +1.84 / +5.20 / +15.86 percent" line of the Stage 0 log.
     A cell says: hold this parameter set, look at this epoch, and this much
     of what it is losing there is reachable without any new physics.

  2. THE BIAS CURRENCIES, per parameter set. score_A and score_F divide by
     the per-epoch benchmark scatter and cannot tell a bias from scatter
     (exp58 P-B), so the loss table alone cannot say whether the single-epoch
     freedom actually removes the z=2 systematic offset the user sees. The
     frozen exp59 B battery answers that directly: B1, the median of
     log10(model / measured) at the 103 kpc anchor; B3, the median relative
     error of the mass inside 10.25 kpc. Measured here for all seven sets and
     for the M2 curvature fit, alongside the null exactly as the gates require.

  3. THE PARAMETER BOUNDS, so the drift panels can show that no fitted value
     sits at one (the Stage 0 log's "AT BOUND" column was empty).

SELF-CHECK BEFORE ANYTHING IS WRITTEN. Every re-evaluation is compared with
the value the fits themselves recorded: the diagonal of the loss matrix
against each fit's saved loss, the diagonal of the score matrices against
each fit's saved scores, the top row against the saved shared-incumbent
scores, and the null's B-gate values against the numbers in the Stage 1 log.
A mismatch stops the script — a figure drawn from quantities that do not
reproduce the table it illustrates is worse than no figure.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp61_epoch_baselines/stage2_figure_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp59_conditioned_efficiency"):
    sys.path.insert(0, str(p))

import b_gates as BG                                     # noqa: E402
import conditioned as C                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage4_target as S4                               # noqa: E402

ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
#: row order of every matrix written here
ROW_LABELS = ["shared", "z=0.4", "z=0.7", "z=1.0", "z=1.5", "z=2.0", "z<=1.0"]
#: the keys those rows come from inside stage0_fits.npz (None = theta_shared)
ROW_KEYS = [None, "fit_z=0.4", "fit_z=0.7", "fit_z=1.0", "fit_z=1.5",
            "fit_z=2.0", "fit_zle1.0"]
#: the null's B-gate values recorded in outputs/stage1_mx2_fit.log. Re-measured
#: here and required to agree, so a figure cannot silently drift from the log.
NULL_B1_LOG = (-0.0087, +0.0065, +0.0110, +0.0016, -0.0461)
NULL_B2_LOG, NULL_B2B_LOG = -0.1216, +0.0816
NULL_B3_LOG = (-2.2, +0.5, +2.3, -1.5, -12.7)
STAGE0 = HERE / "outputs" / "stage0_fits.npz"
STAGE1 = HERE / "outputs" / "stage1_mx2_fit.npz"
OUT = HERE / "outputs" / "stage2_figure_data.npz"
RULE = "=" * 96


def main():
    print(f"{RULE}\nexp61 STAGE 2a — re-evaluating the frozen thetas for the "
          f"figures (no fitting)\n{RULE}\n")
    s0 = np.load(STAGE0, allow_pickle=True)
    s1 = np.load(STAGE1, allow_pickle=True)
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    print(f"  sample: {len(recs)} galaxies; shared incumbent loss "
          f"{slow.loss(theta):.6f}")

    thetas = [np.asarray(theta, float) if k is None
              else np.asarray(s0[f"{k}_theta"], float) for k in ROW_KEYS]
    per_epoch = [S35.StackedProblem(base, recs, data, mask, epochs=[j])
                 for j in range(5)]

    n = len(thetas)
    loss = np.empty((n, 5))
    sa = np.empty((n, 5))
    sf = np.empty((n, 5))
    for i, th in enumerate(thetas):
        for j, pr in enumerate(per_epoch):
            loss[i, j] = pr.loss(th)
            a, f = pr.per_epoch(th)
            sa[i, j], sf[i, j] = a[0], f[0]

    best = np.array([loss[1 + j, j] for j in range(5)])      # the diagonal
    recoverable = 100.0 * (1.0 - best[None, :] / loss)

    # --- self-check against what the fits themselves recorded -------------- #
    print("\n  SELF-CHECK — every re-evaluation against the saved fits")
    for j in range(5):
        key = ROW_KEYS[1 + j]
        d_l = abs(loss[1 + j, j] - float(s0[f"{key}_loss"]))
        d_a = abs(sa[1 + j, j] - float(s0[f"{key}_sa"][0]))
        d_f = abs(sf[1 + j, j] - float(s0[f"{key}_sf"][0]))
        print(f"    {ROW_LABELS[1 + j]:<8} loss {d_l:.2e}  score_A {d_a:.2e}"
              f"  score_F {d_f:.2e}")
        if max(d_l, d_a, d_f) > 1e-9:
            raise SystemExit("REFUSING TO WRITE: a re-evaluated single-epoch "
                             "fit does not reproduce its own saved numbers.")
    d_sa = float(np.max(np.abs(sa[0] - s0["sa_shared"])))
    d_sf = float(np.max(np.abs(sf[0] - s0["sf_shared"])))
    d_ra = float(np.max(np.abs(sa[6] - s0["sa_scope"])))
    print(f"    shared   score_A {d_sa:.2e}  score_F {d_sf:.2e}; "
          f"z<=1.0 all five epochs score_A {d_ra:.2e}")
    if max(d_sa, d_sf, d_ra) > 1e-9:
        raise SystemExit("REFUSING TO WRITE: the shared or z<=1.0 rows do not "
                         "reproduce the saved baseline table.")

    # --- the bias currencies, null first ---------------------------------- #
    fell, ok = S4.decline_flag(data)
    gates = [BG.measure(slow.predict(th), data, mask, fell, ok)
             for th in thetas]
    sp_m2 = C.CSpec(base, "M2")
    pr_m2 = C.ConditionedProblem(sp_m2, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    th_m2 = np.asarray(s1["theta"], float)
    gates.append(BG.measure(pr_m2.predict(th_m2), data, mask, fell, ok))

    b1 = np.array([g["b1"] for g in gates])
    b2 = np.array([g["b2"] for g in gates])
    b2b = np.array([g["b2b"] for g in gates])
    b3 = np.array([g["b3"] for g in gates])

    print("\n  THE NULL'S GATE VALUES vs the Stage 1 log (the zero point "
          "read first)")
    print("    B1  " + " ".join(f"{v:+.4f}" for v in b1[0])
          + "   log " + " ".join(f"{v:+.4f}" for v in NULL_B1_LOG))
    print("    B3  " + " ".join(f"{v:+6.1f}" for v in b3[0])
          + "   log " + " ".join(f"{v:+6.1f}" for v in NULL_B3_LOG))
    print(f"    B2  {b2[0]:+.4f} (log {NULL_B2_LOG:+.4f})    "
          f"B2b {b2b[0]:+.4f} (log {NULL_B2B_LOG:+.4f})")
    if (np.max(np.abs(b1[0] - NULL_B1_LOG)) > 5e-4
            or np.max(np.abs(b3[0] - NULL_B3_LOG)) > 0.05
            or abs(b2[0] - NULL_B2_LOG) > 5e-4
            or abs(b2b[0] - NULL_B2B_LOG) > 5e-4):
        raise SystemExit("REFUSING TO WRITE: the re-measured null does not "
                         "reproduce the gate values in the Stage 1 log.")

    print("\n  WHAT THE SINGLE-EPOCH FREEDOM DOES TO THE TWO BIASES "
          "(each fit at its own epoch)")
    print(f"  {'':<26}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
    print(f"  {'B1 shared [dex]':<26}"
          + "".join(f"{v:>+10.4f}" for v in b1[0]))
    print(f"  {'B1 single-epoch [dex]':<26}"
          + "".join(f"{b1[1 + j, j]:>+10.4f}" for j in range(5)))
    print(f"  {'B3 shared [%]':<26}" + "".join(f"{v:>+9.1f}%" for v in b3[0]))
    print(f"  {'B3 single-epoch [%]':<26}"
          + "".join(f"{b3[1 + j, j]:>+9.1f}%" for j in range(5)))

    lo, hi = np.array(base.bounds()).T
    np.savez(OUT, row_labels=np.array(ROW_LABELS), anchor_z=np.array(ANCHOR_Z),
             names=np.array(base.theta_names), thetas=np.array(thetas),
             bound_lo=lo, bound_hi=hi, loss=loss, best_loss=best,
             recoverable=recoverable, sa=sa, sf=sf,
             b1=b1, b2=b2, b2b=b2b, b3=b3,
             gate_labels=np.array(ROW_LABELS + ["M2"]))
    print(f"\n  saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
