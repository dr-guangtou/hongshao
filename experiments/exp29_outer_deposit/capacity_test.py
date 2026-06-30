"""exp29 Phase 1 — capacity ceiling: CAN the cumulative-deposit model fit all 5
CoGs jointly? (Is the multi-epoch fit fundamentally possible?)

Key lever: the CoG is LINEAR in the deposit masses, so for any fixed set of widths
the jointly-optimal non-negative masses are a convex NNLS solve on the stacked
5x24 CoG vector, with the cumulative structure (deposit i feeds epoch k iff
t_i <= t_k) in the design matrix. Only the low-D width law is a nonlinear outer
loop. Centred-Gaussian kernel; phenomenological free fraction per MAH step.

Three rungs of flexibility (decision gate 1):
  1a  free masses (NNLS) + power-law width sigma(t)=s0 (t/t_obs)^g          [2 width par]
  1b  free masses + flexible width: piecewise-linear log-sigma(log t), 5 knots
  1c  free masses + free per-knot widths on a coarse deposit grid (near-ceiling)

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/capacity_test.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls, minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z                                  # noqa: E402
from cog_extrapolate import measured_cog                                           # noqa: E402
from profile_fit import pick_galaxies                                              # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                          # noqa: E402

R = COG_RAD_KPC                                  # 24 radii, 2..148 kpc
EPOCH_SNAP = np.array(ANCHOR_SNAP)               # [72,59,50,40,33]
OBS_SNAP = 72                                    # we observe at z=0.4


def _deposits(gi):
    """Per-galaxy deposit times/snaps up to the z=0.4 observation (dip-free MAH)."""
    mah = dipfree_mah(gi)
    keep = mah["snap"] <= OBS_SNAP
    return mah["snap"][keep], mah["t"][keep], mah["t_obs"]


def _design(snaps, widths):
    """Stacked design matrix A (5*24, N): cumulative Gaussian-CoG kernel, gated."""
    kappa = 1.0 - np.exp(-R[:, None] ** 2 / (2.0 * widths[None, :] ** 2))     # (24, N)
    return np.vstack([kappa * (snaps <= sk)[None, :] for sk in EPOCH_SNAP])   # (120, N)


def _nnls_rms(snaps, widths, logC):
    """NNLS masses for given widths; return (per-epoch log RMS, masses)."""
    b = (10.0 ** logC).reshape(-1)                       # 120 linear CoG values
    A = _design(snaps, widths)
    w = 1.0 / (np.log(10.0) * b)                         # row weights -> log-space RMS
    m, _ = nnls(A * w[:, None], b * w)
    pred = (A @ m).reshape(5, -1)
    rms = np.array([np.sqrt(np.mean((np.log10(np.clip(pred[k], 1.0, None)) - logC[k]) ** 2))
                    for k in range(5)])
    return rms, m


def fit_powerlaw(gi, logC):                              # rung 1a
    snaps, t, t_obs = _deposits(gi)
    def loss(q):
        widths = (10.0 ** q[0]) * (t / t_obs) ** q[1]
        return float(np.mean(_nnls_rms(snaps, np.clip(widths, 0.3, 500), logC)[0]))
    best = min((minimize(loss, [np.log10(s0), g], method="Nelder-Mead",
                         options=dict(maxiter=3000, xatol=1e-3, fatol=1e-7))
                for s0 in (20, 60) for g in (0.5, 1.5)), key=lambda r: r.fun)
    widths = np.clip((10.0 ** best.x[0]) * (t / t_obs) ** best.x[1], 0.3, 500)
    return _nnls_rms(snaps, widths, logC)[0]


def leave_one_epoch_out(gi, logC):
    """Fit width(power-law)+NNLS masses on 4 epochs, PREDICT the held-out 5th.

    Tests whether free-fraction is a genuine predictor or just interpolates."""
    snaps, t, t_obs = _deposits(gi)
    held = np.full(5, np.nan)
    for h in range(1, 5):                                # z=0.4 always observed; hold out a higher-z epoch
        train = [k for k in range(5) if k != h]

        def loss(q):
            widths = np.clip((10.0 ** q[0]) * (t / t_obs) ** q[1], 0.3, 500)
            A = _design(snaps, widths)
            rows = np.concatenate([np.arange(24) + 24 * k for k in train])
            b = (10.0 ** logC).reshape(-1)
            w = 1.0 / (np.log(10.0) * b)
            m, _ = nnls((A * w[:, None])[rows], (b * w)[rows])
            pred_h = A[24 * h:24 * h + 24] @ m
            return np.sqrt(np.mean((np.log10(np.clip(pred_h, 1.0, None)) - logC[h]) ** 2)), m, widths

        best = min((minimize(lambda q: loss(q)[0], [np.log10(s0), g], method="Nelder-Mead",
                             options=dict(maxiter=2000, xatol=1e-3, fatol=1e-7))
                    for s0 in (20, 60) for g in (0.5, 1.5)), key=lambda r: r.fun)
        held[h] = loss(best.x)[0]
    return held


def fit_spline_width(gi, logC, n_knots=5):               # rung 1b
    snaps, t, t_obs = _deposits(gi)
    lt = np.log10(t); knots = np.linspace(lt.min(), lt.max(), n_knots)
    def widths_of(q):
        return np.clip(10.0 ** np.interp(lt, knots, q), 0.3, 500)
    def loss(q):
        return float(np.mean(_nnls_rms(snaps, widths_of(q), logC)[0]))
    q0 = np.full(n_knots, np.log10(40.0))
    best = min((minimize(loss, q0 + d, method="Nelder-Mead",
                         options=dict(maxiter=8000, xatol=1e-3, fatol=1e-8))
                for d in (0.0, np.linspace(-0.8, 0.8, n_knots))), key=lambda r: r.fun)
    return _nnls_rms(snaps, widths_of(best.x), logC)[0]


def main():
    gxs = pick_galaxies(20)
    print("exp29 Phase 1 — capacity ceiling: joint 5-epoch CoG RMS (dex), per galaxy\n")
    print(f"  {'idx':>5s} {'logM*':>6s} | {'1a power-law':>26s} | {'1b spline-width':>26s}")
    print(f"  {'':>5s} {'':>6s} | in-sample 1a (z.4-z2) | in-sample 1b (z.4-z2) "
          "| hold-out (z.7,1,1.5,2)")
    rows_a, rows_b, rows_h = [], [], []
    for logm, gi in gxs:
        logC = measured_cog(gi)
        if logC is None:
            continue
        ra = fit_powerlaw(gi, logC); rb = fit_spline_width(gi, logC)
        ho = leave_one_epoch_out(gi, logC)
        rows_a.append(ra); rows_b.append(rb); rows_h.append(ho)
        print(f"  {gi:5d} {logm:6.2f} | " + " ".join(f"{x:5.3f}" for x in ra) +
              " | " + " ".join(f"{x:5.3f}" for x in rb) +
              " | " + " ".join(f"{x:5.3f}" for x in ho[1:]))
    ma, mb, mh = np.median(rows_a, 0), np.median(rows_b, 0), np.median(rows_h, 0)
    print(f"\n  MEDIAN 1a power-law (in-sample): " + " ".join(f"{x:5.3f}" for x in ma) +
          f"   (mean {ma.mean():.3f})")
    print(f"  MEDIAN 1b spline-width (in-smpl): " + " ".join(f"{x:5.3f}" for x in mb) +
          f"   (mean {mb.mean():.3f})")
    print(f"  MEDIAN hold-out (predict 1 epoch):       " +
          " ".join(f"{x:5.3f}" for x in mh[1:]) + f"   (mean {np.nanmean(mh):.3f})")
    print("\n[gate 1] PASSED if in-sample << 0.05 dex (structure capable) AND hold-out "
          "stays low (free-fraction generalizes, not just interpolates).")


if __name__ == "__main__":
    sys.exit(main())
