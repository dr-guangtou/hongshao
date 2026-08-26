"""exp59 Stage 4a — can a formation-conditioned expansion STRENGTH ever close
the decliner-specific central residual? Measured by evaluation, before the fit.

Stage 2's pinned-size `X3` leaves exactly the residual the plan's Stage 4
names as its trigger: the declining galaxies' central-error span is 0.239
(their z=2 centre −0.205 dex) while the non-declining group sits at 0.018.
The pre-registered Stage 4 candidate is `X4` — the expansion strength
conditioned on the deposit's formation state, `A -> A0 + A_f (f_form_j − 0.6)`
— with the core radius FROZEN, because exp57's X4 failure is attributed to
the free `Rc` drifting to a 95 kpc near-homologous core.

This sweep applies the exp59 discipline (Stage 0b): before the ~2 h fit,
sweep `A_f` with everything else held at the Stage 2 optimum and read the
gate quantities directly. THE QUALIFICATION RULE, declared before the numbers
are read: some grid `A_f` must bring the declining group's span (a) to
**0.20 or below** (a real improvement on 0.239) while the non-declining
group's span (b) stays within its **0.060** limit. The pre-registered sign,
carried over from exp57's diagnostic: `A_f < 0` (deposits in earlier-formed
haloes expand more). exp57's free-`Rc` fit contradicted that sign in a regime
this sweep does not enter (`Rc` here is frozen at 1.98 kpc); if the sweep
shows the leverage lies at `A_f > 0` instead, that is a falsification to
report, not to paper over.

The same caveat as Stage 0b, stated: a conditional slice DISQUALIFIES
honestly; a qualifying sweep still requires the fit and the gates.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp59_conditioned_efficiency/stage4_leverage.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP57 = ROOT / "experiments/exp57_expansion_term"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude", EXP57):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage4_target as S4                               # noqa: E402

PINNED = (EXP57 / "outputs" / "stage3_fits" /
          "gompertz_log-E2-S2+X3+frozen_log_f0_b#pinned_size.npz")
GRID = (-10.0, -6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0, 10.0)
#: the rule, frozen before the numbers: (a) <= A_LIM and (b) <= B_LIM at
#: some grid point, else Stage 4's fit is not run.
A_LIM, B_LIM = 0.20, 0.060
I2 = 0                       # R_GRID[0] = 2 kpc, stage4_target's aperture
RULE = "=" * 96


def spans(pred, data, mask, fell, ok):
    """G5-style central spans: median log10(model/measured) M*(<2 kpc) per
    epoch per group; span = max - min over the five epochs."""
    finite = np.isfinite(pred).all(axis=2)
    use = np.asarray(mask, bool) & finite & (data[:, :, I2] > 0) \
        & (pred[:, :, I2] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.log10(pred[:, :, I2] / data[:, :, I2])
    e = np.where(use, e, np.nan)
    out = {}
    for name, g in (("dec", fell & ok), ("non", (~fell) & ok)):
        med = np.array([np.nanmedian(e[g, k]) for k in range(5)])
        out[name] = (float(med.max() - med.min()), med)
    return out


def main(smoke=False):
    print(f"{RULE}\nexp59 STAGE 4a — leverage of the formation-conditioned "
          f"expansion strength\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    fell, ok = S4.decline_flag(data)
    z = np.load(PINNED, allow_pickle=True)
    th_fit = np.asarray(z["theta"], float)
    a0, lrc = th_fit[-2], th_fit[-1]
    sp = X.XSpec(base, "X4")
    pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
    print(f"  base: the Stage 2 pinned-size optimum (loss "
          f"{float(z['loss']):.6f}), A0 = {a0:+.4f}, log_Rc = {lrc:+.4f} "
          f"(Rc = {10 ** lrc:.2f} kpc), both HELD")
    print(f"  rule: some A_f must give declining span <= {A_LIM} with "
          f"non-declining span <= {B_LIM}")
    print(f"  pre-registered sign: A_f < 0\n")
    print(f"  {'A_f':>7} {'declining span':>15} {'non-decl span':>14} "
          f"{'decl z=2 median':>16}   qualifies")
    best = None
    for af in GRID:
        th = np.concatenate([th_fit[:-2], [a0, af, lrc]])
        s = spans(pr.predict(th), data, mask, fell, ok)
        q = s["dec"][0] <= A_LIM and s["non"][0] <= B_LIM
        print(f"  {af:>+7.1f} {s['dec'][0]:>15.4f} {s['non'][0]:>14.4f} "
              f"{s['dec'][1][4]:>+16.4f}   {'YES' if q else '-'}")
        if q and (best is None or s["dec"][0] < best[1]):
            best = (af, s["dec"][0])
    if best:
        print(f"\n  VERDICT: QUALIFIES at A_f = {best[0]:+.1f} (declining "
              f"span {best[1]:.3f}) — the Stage 4 fit is warranted.")
    else:
        print(f"\n  VERDICT: no grid point qualifies — the Stage 4 fit is "
              f"NOT run, and the plan is complete.")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
