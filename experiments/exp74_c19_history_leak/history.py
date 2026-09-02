"""exp74 Stage 0a — the PRE-EPOCH halo history: a DiffMAH curve per galaxy per
epoch fitted only to what the halo had done by that epoch.

WHY. The engine reads each halo through the official DiffMAH curve, four numbers
fitted to the whole peak-mass history (2 Gyr to z = 0, anchored at z = 0). The
curve the model integrates before z = 2 is therefore a function of the halo's
z = 0 mass -- and before 2.15 Gyr it is the fit's extrapolation from later,
mostly future, data. exp71 measured what that does (C19): future growth at
fixed M200c(z_k) is predicted with R^2 0.50-0.69 from the DiffMAH curve read
before z_k, 0.00 from the measured history. Here the curve is refitted per
epoch to the catalog's peak-mass history at snapshots <= that epoch's anchor,
ANCHORED AT THAT EPOCH with the anchor mass fixed to the measured value, so
by construction it cannot know the future.

WHAT IS FITTED. `hs["M200c"]` (log10, `selection.HS_NPZ`) at the catalog's 16
snapshots, `GroupFlag == 1`, turned into a peak-mass history by a running
maximum in time (the official fit is to log Mpeak); snapshots earlier than
T_MIN_PAST are dropped (the official cut is 2 Gyr, which would leave ONE point
before z = 2; 1.0 Gyr keeps snapshots 17/21/25 at 93/99/100 per cent coverage).
`logmp` is FIXED at the anchor's measured peak mass; `logtc`, `early`, `late`
are fitted by bounded least squares (as `hongshao.diffmah.fit_mah`). Galaxies
with too few points get a lower TIER (fewer free shape parameters), never
dropped: dropping on history coverage would select on assembly history, the
thing under test.

  tier 3   >= 4 points   logtc, early, late free
  tier 2   3 points      early, late free; logtc at the mid log-time
  tier 1   2 points      one power law (early = late)
  tier 0   < 2 points    the OFFICIAL shape (logtc, early, late) with only the
                         anchor mass replaced -- counted and reported; it leaks
                         the future's shape and is the price of not dropping

GATES (run here, before any model sees the curves):
  G1  at z = 0.4 the pre-epoch curve agrees with the official one at the
      shared snapshots to within the official fit's own rms on those points
  G2  THE gate: cross-validated R^2 of future growth at fixed M200c(z_k) from
      the pre-epoch curve read at the pre-anchor snapshots (exp71's
      `channel_origin.cv_r2`, mass stripped from both sides) is ~ 0, like the
      measured history, while the official curve scores 0.50-0.69

Writes `outputs/history_curves.npz`; `load_curves(curves_official, k)` rebuilds
the epoch's `HaloCurve` list from it for the model scripts.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u experiments/exp74_c19_history_leak/history.py [--smoke]
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp71_c18_selection_leakage", HERE):
    sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402
import selection as SEL                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
from channel_origin import cv_r2, MIN_COVER, N_HIST      # noqa: E402

RULE, THIN = "=" * 100, "-" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
T_MIN_PAST = 1.0                 # Gyr; the official fit's 2.0 leaves one point before z = 2
MAH_K = E.MAH_K
BOUNDS = dict(early=(0.1, 12.0), late=(0.05, 12.0))


def peak_history(hs):
    """(n, n_snap) log10 peak mass along the catalog snapshots, NaN where the
    catalog has no valid central-halo entry; the running maximum is taken over
    the VALID entries in time order (the official fit is to log Mpeak)."""
    m = np.asarray(hs["M200c"], float)
    ok = (np.asarray(hs["GroupFlag"]) == 1) & np.isfinite(m) & (m > 0)
    out = np.full_like(m, np.nan)
    for g in range(m.shape[0]):
        run = -np.inf
        for i in range(m.shape[1]):
            if ok[g, i]:
                run = max(run, m[g, i])
                out[g, i] = run
    return out


def _curve(lt, logmp, logtc, early, late, logt0):
    s = expit(MAH_K * (lt - logtc))
    return logmp + (early + (late - early) * s) * (lt - logt0)


def fit_anchored(lt, y, logt0, logmp, official):
    """Fit (logtc, early, late) with logmp FIXED at the anchor; returns the
    three numbers, the tier, and the rms. `official` = (logtc, early, late)
    of the official curve, used only for tier 0."""
    n = len(lt)
    if n >= 4:
        def resid(p):
            return _curve(lt, logmp, p[0], p[1], p[2], logt0) - y
        lo = [lt[0], BOUNDS["early"][0], BOUNDS["late"][0]]
        hi = [lt[-1], BOUNDS["early"][1], BOUNDS["late"][1]]
        i_lo = min(3, n - 1)
        e0 = float(np.clip((y[i_lo] - y[0]) / max(lt[i_lo] - lt[0], 1e-3), *BOUNDS["early"]))
        l0 = float(np.clip((y[-1] - y[max(0, n - 4)]) / max(lt[-1] - lt[max(0, n - 4)], 1e-3), *BOUNDS["late"]))
        x0 = np.clip([float(np.median(lt)), e0, l0], lo, hi)
        sol = least_squares(resid, x0, bounds=(lo, hi), method="trf", max_nfev=2000)
        return float(sol.x[0]), float(sol.x[1]), float(sol.x[2]), 3, float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    if n == 3:
        logtc = float(0.5 * (lt[0] + lt[-1]))

        def resid(p):
            return _curve(lt, logmp, logtc, p[0], p[1], logt0) - y
        lo = [BOUNDS["early"][0], BOUNDS["late"][0]]
        hi = [BOUNDS["early"][1], BOUNDS["late"][1]]
        sl = float(np.clip((y[-1] - y[0]) / max(lt[-1] - lt[0], 1e-3), 0.1, 12.0))
        sol = least_squares(resid, np.clip([sl, sl], lo, hi), bounds=(lo, hi), method="trf", max_nfev=1000)
        return logtc, float(sol.x[0]), float(sol.x[1]), 2, float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    if n == 2:
        sl = float(np.clip((y[-1] - y[0]) / max(lt[-1] - lt[0], 1e-3), 0.1, 12.0))
        return float(0.5 * (lt[0] + lt[-1])), sl, sl, 1, 0.0
    return official[0], official[1], official[2], 0, np.nan


def build_history(recs, curves, hs, verbose=True):
    """Per epoch, the anchored pre-epoch fit for every galaxy of `recs`.
    Returns dict with (n, 5) arrays logmp, logtc, early, late, logt0, tier, rms,
    n_pts, and the snapshot bookkeeping."""
    rows = np.array([h.row for h in recs])
    snaps = np.asarray(hs["snaps"], int)
    t_snap = np.array([E.T_SNAP[s] for s in snaps])
    lt_snap = np.log10(t_snap)
    peak = peak_history(hs)[rows]
    n = len(recs)
    out = {k: np.full((n, 5), np.nan) for k in ("logmp", "logtc", "early", "late", "logt0", "rms")}
    out["tier"] = np.full((n, 5), -1, int)
    out["n_pts"] = np.zeros((n, 5), int)
    out["anchor_from_official"] = np.zeros((n, 5), bool)
    for k in EPOCHS:
        anchor = E.ANCHOR_SNAP[k]
        cols = np.where((snaps <= anchor) & (t_snap >= T_MIN_PAST))[0]
        ia = int(np.where(snaps == anchor)[0][0])
        logt0 = float(lt_snap[ia])
        for g, hc in enumerate(curves):
            y_all = peak[g, cols]
            ok = np.isfinite(y_all)
            if np.isfinite(peak[g, ia]):
                logmp = float(peak[g, ia])
            else:
                logmp = float(E.log_mah(logt0, hc))       # rare; counted
                out["anchor_from_official"][g, k] = True
            lt, y = lt_snap[cols][ok], y_all[ok]
            logtc, early, late, tier, rms = fit_anchored(
                lt, y, logt0, logmp, (hc.logtc, hc.early, hc.late))
            out["logmp"][g, k], out["logtc"][g, k] = logmp, logtc
            out["early"][g, k], out["late"][g, k] = early, late
            out["logt0"][g, k], out["tier"][g, k], out["rms"][g, k] = logt0, tier, rms
            out["n_pts"][g, k] = int(ok.sum())
        if verbose:
            tiers = [int((out["tier"][:, k] == t).sum()) for t in (3, 2, 1, 0)]
            print(f"  z={ANCHOR_Z[k]}: {len(cols)} usable snapshots "
                  f"({t_snap[cols].min():.2f}-{t_snap[cols].max():.2f} Gyr); tiers 3/2/1/0 = "
                  f"{tiers[0]}/{tiers[1]}/{tiers[2]}/{tiers[3]}; anchor mass from the official "
                  f"curve for {int(out['anchor_from_official'][:, k].sum())}; median rms "
                  f"{np.nanmedian(out['rms'][:, k]):.4f} dex")
    out["rows"], out["snaps"] = rows, snaps
    return out


def curves_for_epoch(curves_official, hist, k):
    """The epoch's `HaloCurve` list: the official record with the four DiffMAH
    numbers and the anchor replaced by the pre-epoch fit."""
    return [dataclasses.replace(hc, logmp=float(hist["logmp"][g, k]), logtc=float(hist["logtc"][g, k]),
                                early=float(hist["early"][g, k]), late=float(hist["late"][g, k]),
                                logt0=float(hist["logt0"][g, k]))
            for g, hc in enumerate(curves_official)]


def load_curves(curves_official, k, path=None):
    hist = dict(np.load(path or (OUTDIR / "history_curves.npz"), allow_pickle=True))
    assert np.array_equal(hist["rows"], np.array([hc.row for hc in curves_official])), \
        "history_curves.npz was built for a different record list"
    return curves_for_epoch(curves_official, hist, k)


def gate_g1(curves, hist, hs, peak, rows):
    """At z = 0.4: pre-epoch vs official at the shared snapshots, against the
    official's own rms on the peak data there."""
    snaps = np.asarray(hs["snaps"], int)
    t_snap = np.array([E.T_SNAP[s] for s in snaps])
    cols = np.where((snaps <= E.ANCHOR_SNAP[0]) & (t_snap >= T_MIN_PAST))[0]
    lt = np.log10(t_snap[cols])
    pre = curves_for_epoch(curves, hist, 0)
    d_curve, d_off, d_pre = [], [], []
    for g, (hc, pc) in enumerate(zip(curves, pre)):
        y = peak[g, cols]
        ok = np.isfinite(y)
        if ok.sum() < 4:
            continue
        a, b = E.log_mah(lt[ok], hc), E.log_mah(lt[ok], pc)
        d_curve.append(np.sqrt(np.mean((a - b) ** 2)))
        d_off.append(np.sqrt(np.mean((a - y[ok]) ** 2)))
        d_pre.append(np.sqrt(np.mean((b - y[ok]) ** 2)))
    d_curve, d_off, d_pre = map(np.array, (d_curve, d_off, d_pre))
    print(f"\n  G1  z=0.4, {len(d_curve)} galaxies with >= 4 shared points ({t_snap[cols].min():.2f}-"
          f"{t_snap[cols].max():.2f} Gyr):\n      rms(pre-epoch - official) median {np.median(d_curve):.4f} "
          f"dex, 90th pct {np.percentile(d_curve, 90):.4f};\n      official vs the peak data on the same "
          f"points: median {np.median(d_off):.4f} dex; pre-epoch vs the data: {np.median(d_pre):.4f} dex")
    ok = np.median(d_curve) <= np.median(d_off)
    print(f"      {'OK' if ok else 'FAILED'}: the two curves differ by "
          f"{'less' if ok else 'MORE'} than the official fit's own misfit on these points"
          f"\n      (the official also fits snapshots AFTER z=0.4 up to z=0 and anchors there, "
          f"so a residual difference of this size is the z=0.4 -> 0 growth's influence)")
    return ok, d_curve, d_off, d_pre


def gate_g2(curves, hist, hs, rows, lmh_cat, growth, keep):
    """THE gate: cv R^2 of future growth at fixed M200c(z_k) from the curve read
    at the pre-anchor snapshots -- official vs pre-epoch (exp71 part 2)."""
    snaps = list(np.asarray(hs["snaps"], int))
    m_all = np.asarray(hs["M200c"], float)
    ok_all = (np.asarray(hs["GroupFlag"]) == 1) & np.isfinite(m_all) & (m_all > 0)
    cover = ok_all.mean(axis=0)
    print(f"\n  G2  cross-validated R^2 of FUTURE growth (to z=0.4) at fixed M200c(z_k), from the curve "
          f"read at the {N_HIST} pre-anchor snapshots\n      (mass stripped from both sides; exp71 "
          f"channel_origin section 2). The measured history scores 0.00 there.")
    print(f"      {'epoch':>6}{'snapshots read':>22}{'official':>12}{'pre-epoch':>12}{'n':>8}")
    r2 = np.full((5, 2), np.nan)
    for k in range(1, 5):
        anchor = E.ANCHOR_SNAP[k]
        usable = [s for i, s in enumerate(snaps) if s < anchor and cover[i] >= MIN_COVER]
        take = usable[-N_HIST:]
        lt = np.log10(np.array([E.T_SNAP[s] for s in take]))
        pre = curves_for_epoch(curves, hist, k)
        X_off = np.array([E.log_mah(lt, hc) for hc in curves])
        X_pre = np.array([E.log_mah(lt, hc) for hc in pre])
        ok = keep[:, k] & np.isfinite(growth[:, k]) & np.isfinite(X_pre).all(1)
        D0 = np.column_stack([np.ones(int(ok.sum())), lmh_cat[ok, k]])
        g_res = growth[ok, k] - D0 @ np.linalg.lstsq(D0, growth[ok, k], rcond=None)[0]

        def strip(X):
            return X - D0 @ np.linalg.lstsq(D0, X, rcond=None)[0]
        r2[k] = [cv_r2(g_res, strip(X_off[ok])), cv_r2(g_res, strip(X_pre[ok]))]
        print(f"      {ANCHOR_Z[k]:>6}" + f"{' '.join(f'z={np.asarray(hs['z'])[snaps.index(s)]:.1f}' for s in take):>22}"
              + f"{r2[k, 0]:>12.3f}{r2[k, 1]:>12.3f}{int(ok.sum()):>8}")
    worst = np.nanmax(r2[1:, 1])
    ok = worst < 0.05
    print(f"      {'OK' if ok else 'FAILED'}: the pre-epoch curve's worst R^2 is {worst:.3f} "
          f"(threshold 0.05); the official's best is {np.nanmax(r2[1:, 0]):.3f}")
    return ok, r2


def main(smoke=False):
    print(f"{RULE}\nexp74 Stage 0a — the pre-epoch halo history, fitted and gated\n{RULE}\n")
    recs, data, mask_legacy, lmh_dm, _, _, _ = S0.build(smoke)
    curves = E.build_curves(recs, verbose=True)
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    rows = np.array([h.row for h in recs])
    print(f"  {len(recs)} galaxies; catalog snapshots "
          + " ".join(str(s) for s in np.asarray(hs['snaps'], int))
          + f"; pre-epoch fits use t >= {T_MIN_PAST} Gyr (official cut 2.0 Gyr)\n")
    hist = build_history(recs, curves, hs, verbose=True)
    peak = peak_history(hs)[rows]

    lmh_cat = SEL.sample_masses(hs)[rows]
    growth = lmh_cat[:, 0][:, None] - lmh_cat
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    keep = good[:, None] & np.isfinite(lmh_cat)

    g1, d_curve, d_off, d_pre = gate_g1(curves, hist, hs, peak, rows)
    g2, r2 = gate_g2(curves, hist, hs, rows, lmh_cat, growth, keep)

    # what the anchoring changes at the epochs themselves: the curve's mass at
    # the anchor is now the MEASURED peak mass, not DiffMAH's smoothed value
    print(f"\n  the anchor mass, pre-epoch (measured peak) minus official DiffMAH at the same time "
          f"[dex]: median / 16-84")
    for k in EPOCHS:
        d = hist["logmp"][:, k] - np.array([E.log_mah(hist["logt0"][0, k], hc) for hc in curves])
        print(f"      z={ANCHOR_Z[k]}: {np.nanmedian(d):+.4f}  "
              f"{np.nanpercentile(d, 16):+.4f} / {np.nanpercentile(d, 84):+.4f}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if smoke else ""
    np.savez(OUTDIR / f"history_curves{tag}.npz", **hist, t_min_past=T_MIN_PAST,
             g1_ok=g1, g2_ok=g2, g2_r2=r2, g1_rms_curve=d_curve, g1_rms_official=d_off, g1_rms_pre=d_pre)
    print(f"\n  gates: G1 {'OK' if g1 else 'FAILED'}, G2 {'OK' if g2 else 'FAILED'}")
    print(f"wrote {OUTDIR / f'history_curves{tag}.npz'}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
