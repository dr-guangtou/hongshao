"""exp74 variant B — the MEASURED halo history as the engine's input.

WHAT IT IS. Every fit so far read each halo through a four-number DiffMAH
curve. Here the engine reads the halo's mass history straight from the TNG300
merger tree (`selection.HS_NPZ`: M200c at sixteen snapshots, z = 12 to 0.4),
turned into a peak-mass history by a running maximum in time and interpolated
between the measured points. The full history, future included, is the input;
the deposition integral is truncated at each epoch as it always was, so the
future is available but never consulted. The final halo mass is there for any
part of the model that wants it (the user's rule, 2026-09-04: the halo's full
history is legitimate input; the galaxy's own stellar mass never is).

HOW IT IS SMOOTHED — locally. Between measured points the curve is a monotone
cubic (PCHIP) in (log t, log M): each segment's shape is set by the two points
it joins and the slopes at those points, and each slope by the points on
either side. A later snapshot helps steady the curve just before it and no
further; nothing pins the early history to the mass at z = 0. Before the
first usable point (T_MIN_PAST = 1.0 Gyr, snapshot 17 at z = 5, 93 per cent
coverage) the curve is a power law in time continuing the first segment's
slope — the same extrapolation the pre-epoch fit needed there, and the
official curve's own regime before its 2 Gyr data cut.

Galaxies with fewer than two usable points (none expected) fall back to the
pre-epoch DiffMAH curve and are counted.

GATES (`main`): G1' the curve passes through the measured peak masses (exact
by construction; the official curve's rms there is 0.105 dex); G2 the
no-future test at the pre-anchor SNAPSHOT times (trivially the measured
values) and, the version that matters for an interpolant, at the geometric
MIDPOINTS between those snapshots, where the PCHIP slopes use the neighbours
— all three representations side by side.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u experiments/exp74_c19_history_leak/measured.py [--smoke]
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

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
import history as HI                                     # noqa: E402
from channel_origin import cv_r2, MIN_COVER, N_HIST      # noqa: E402

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
OUTDIR = HERE / "outputs"
T_MIN_PAST = HI.T_MIN_PAST
SLOPE_CLIP = (0.3, 12.0)         # dlogM/dlogt of the early power law, as DiffMAH's `early` bounds


@dataclasses.dataclass
class MeasuredCurve(E.HaloCurve):
    """A `HaloCurve` whose mass history is an interpolated TABLE. The four
    DiffMAH numbers are kept (the pre-epoch z = 0.4 fit) only so that code
    reading them for bookkeeping still works; the engine reads the table."""
    lt_knots: np.ndarray = None
    y_knots: np.ndarray = None
    slope0: float = np.nan

    def __post_init__(self):
        self._pchip = PchipInterpolator(self.lt_knots, self.y_knots, extrapolate=False)
        self._dpchip = self._pchip.derivative()

    def log_mah_table(self, lt):
        lt = np.asarray(lt, float)
        y = self._pchip(lt)
        early = lt < self.lt_knots[0]
        y = np.where(early, self.y_knots[0] + self.slope0 * (lt - self.lt_knots[0]), y)
        late = lt > self.lt_knots[-1]
        if np.any(late):                         # never consulted by a truncated integral
            y = np.where(late, self.y_knots[-1], y)
        return y

    def dm_dlnt_table(self, lt):
        lt = np.asarray(lt, float)
        s = self._dpchip(lt)
        s = np.where(lt < self.lt_knots[0], self.slope0, s)
        s = np.where(lt > self.lt_knots[-1], 0.0, s)
        return 10.0 ** self.log_mah_table(lt) * s


def build_measured(recs, curves, hs, hist=None, verbose=True):
    """One `MeasuredCurve` per galaxy (the SAME curve serves every epoch).
    Returns (curves_measured, info)."""
    rows = np.array([h.row for h in recs])
    snaps = np.asarray(hs["snaps"], int)
    t_snap = np.array([E.T_SNAP[s] for s in snaps])
    lt_snap = np.log10(t_snap)
    peak = HI.peak_history(hs)[rows]
    usable = t_snap >= T_MIN_PAST
    if hist is None:
        hist = dict(np.load(OUTDIR / "history_curves.npz", allow_pickle=True))
    out, n_knots, fallback, slopes = [], np.zeros(len(recs), int), 0, np.full(len(recs), np.nan)
    for g, hc in enumerate(curves):
        ok = usable & np.isfinite(peak[g])
        lt, y = lt_snap[ok], peak[g, ok]
        # the z=0.4 pre-epoch fit's four numbers ride along as bookkeeping
        base = dict(logmp=float(hist["logmp"][g, 0]), logtc=float(hist["logtc"][g, 0]),
                    early=float(hist["early"][g, 0]), late=float(hist["late"][g, 0]),
                    logt0=float(hist["logt0"][g, 0]))
        if len(lt) < 2:
            fallback += 1
            out.append(dataclasses.replace(hc, **base))
            continue
        # PCHIP needs strictly increasing y? No -- x strictly increasing; y may
        # be flat (peak mass holds), which PCHIP handles with zero slope
        first = PchipInterpolator(lt, y).derivative()(lt[0])
        s0 = float(np.clip(first if np.isfinite(first) and first > 0 else (y[1] - y[0]) / (lt[1] - lt[0]),
                           *SLOPE_CLIP))
        mc = MeasuredCurve(**{f.name: getattr(hc, f.name) for f in dataclasses.fields(E.HaloCurve)},
                           lt_knots=lt, y_knots=y, slope0=s0)
        for k, v in base.items():
            setattr(mc, k, v)
        out.append(mc)
        n_knots[g], slopes[g] = len(lt), s0
    if verbose:
        print(f"  measured curves: {len(out) - fallback} of {len(out)} galaxies with >= 2 usable points "
              f"(t >= {T_MIN_PAST} Gyr), {fallback} fell back to the pre-epoch DiffMAH curve; knots per galaxy "
              f"median {int(np.median(n_knots[n_knots > 0]))} (min {n_knots[n_knots > 0].min()}); early power-law "
              f"slope median {np.nanmedian(slopes):.2f} (16-84: {np.nanpercentile(slopes, 16):.2f}-"
              f"{np.nanpercentile(slopes, 84):.2f})")
    return out, dict(n_knots=n_knots, slope0=slopes, fallback=fallback)


INPUT_KINDS = ("official", "pre-epoch", "measured")


def build_input(recs, kind, hs=None, hist_path=None, verbose=False):
    """THE ONE ENTRY POINT for the halo-history input (the user, 2026-09-05:
    keep the model able to take DiffMAH parameters and the measured MAH alike).

    kind = "official"   the official DiffMAH curve, one per galaxy for every
                        epoch (every experiment before exp74);
           "pre-epoch"  a DiffMAH curve per galaxy PER EPOCH fitted to the
                        history before that epoch (`history.py`);
           "measured"   the measured running-peak history interpolated, one
                        per galaxy for every epoch (this module).
    Returns {epoch: curve list} for every kind, so a caller can always write
    `curves[k]`; for "official" and "measured" the five lists are the same
    object and `predict2` may be called once for all epochs."""
    if kind not in INPUT_KINDS:
        raise ValueError(f"kind must be one of {INPUT_KINDS}, got {kind!r}")
    curves = E.build_curves(recs, verbose=verbose)
    if kind == "official":
        return {k: curves for k in EPOCHS}
    hist = dict(np.load(hist_path or (OUTDIR / "history_curves.npz"), allow_pickle=True))
    assert np.array_equal(hist["rows"], np.array([hc.row for hc in curves])), \
        "history_curves.npz was built for a different record list"
    if kind == "pre-epoch":
        return {k: HI.curves_for_epoch(curves, hist, k) for k in EPOCHS}
    hs = np.load(SEL.HS_NPZ, allow_pickle=True) if hs is None else hs
    meas, _ = build_measured(recs, curves, hs, hist, verbose=verbose)
    return {k: meas for k in EPOCHS}


def gate_g2_all(curves_by_name, hs, rows, lmh_cat, growth, keep, at="snapshots"):
    """cv R^2 of future growth at fixed M200c(z_k) from each representation read
    at the pre-anchor snapshot times, or at the midpoints between them."""
    snaps = list(np.asarray(hs["snaps"], int))
    m_all = np.asarray(hs["M200c"], float)
    ok_all = (np.asarray(hs["GroupFlag"]) == 1) & np.isfinite(m_all) & (m_all > 0)
    cover = ok_all.mean(axis=0)
    names = list(curves_by_name)
    print(f"\n  G2 ({at}) — cross-validated R^2 of FUTURE growth at fixed M200c(z_k); the measured history "
          f"scores 0.00 at the snapshots")
    print(f"      {'epoch':>6}" + "".join(f"{n:>14}" for n in names) + f"{'n':>8}")
    r2 = np.full((5, len(names)), np.nan)
    for k in range(1, 5):
        anchor = E.ANCHOR_SNAP[k]
        usable = [s for i, s in enumerate(snaps) if s < anchor and cover[i] >= MIN_COVER]
        take = usable[-N_HIST:]
        lt = np.log10(np.array([E.T_SNAP[s] for s in take]))
        if at == "midpoints":
            lt = 0.5 * (lt[:-1] + lt[1:])
            lt = np.append(lt, 0.5 * (lt[-1] + np.log10(E.T_SNAP[anchor])))
        ok = keep[:, k] & np.isfinite(growth[:, k])
        D0 = np.column_stack([np.ones(int(ok.sum())), lmh_cat[ok, k]])
        g_res = growth[ok, k] - D0 @ np.linalg.lstsq(D0, growth[ok, k], rcond=None)[0]

        def strip(X):
            return X - D0 @ np.linalg.lstsq(D0, X, rcond=None)[0]
        for j, n in enumerate(names):
            cv = curves_by_name[n](k)
            X = np.array([E.log_mah(lt, hc) for hc in cv])
            fin = np.isfinite(X[ok]).all(1)
            r2[k, j] = cv_r2(g_res[fin], strip(X[ok])[fin])
        print(f"      {ANCHOR_Z[k]:>6}" + "".join(f"{v:>14.3f}" for v in r2[k]) + f"{int(ok.sum()):>8}")
    return r2


def main(smoke=False):
    print(f"{RULE}\nexp74 variant B — the measured halo history as the engine's input, built and gated\n{RULE}\n")
    recs, data, mask_legacy, lmh_dm, _, _, _ = S0.build(smoke)
    curves = E.build_curves(recs, verbose=True)
    hs = np.load(SEL.HS_NPZ, allow_pickle=True)
    rows = np.array([h.row for h in recs])
    tag = "_smoke" if smoke else ""
    hist = dict(np.load(OUTDIR / f"history_curves{tag}.npz", allow_pickle=True))
    meas, info = build_measured(recs, curves, hs, hist)
    pre = {k: HI.curves_for_epoch(curves, hist, k) for k in EPOCHS}

    # G1': the table passes through the measured peaks; the official curve's misfit there
    peak = HI.peak_history(hs)[rows]
    snaps = np.asarray(hs["snaps"], int)
    t_snap = np.array([E.T_SNAP[s] for s in snaps]); lt_snap = np.log10(t_snap)
    cols = np.where(t_snap >= T_MIN_PAST)[0]
    d_meas, d_off = [], []
    for g, (hc, mc) in enumerate(zip(curves, meas)):
        y = peak[g, cols]; ok = np.isfinite(y)
        if ok.sum() < 2:
            continue
        d_meas.append(np.sqrt(np.mean((E.log_mah(lt_snap[cols][ok], mc) - y[ok]) ** 2)))
        d_off.append(np.sqrt(np.mean((E.log_mah(lt_snap[cols][ok], hc) - y[ok]) ** 2)))
    print(f"\n  G1' rms against the measured peak masses at the usable snapshots: measured curve "
          f"{np.median(d_meas):.1e} dex (median), official DiffMAH {np.median(d_off):.4f} dex  OK")

    lmh_cat = SEL.sample_masses(hs)[rows]
    growth = lmh_cat[:, 0][:, None] - lmh_cat
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    keep = good[:, None] & np.isfinite(lmh_cat)
    reps = {"official": lambda k: curves, "pre-epoch": lambda k: pre[k], "measured": lambda k: meas}
    r2_snap = gate_g2_all(reps, hs, rows, lmh_cat, growth, keep, at="snapshots")
    r2_mid = gate_g2_all(reps, hs, rows, lmh_cat, growth, keep, at="midpoints")
    worst = np.nanmax(r2_mid[1:, 2])
    ok = worst < 0.05
    print(f"\n  G2 verdict for the measured curve: worst R^2 at the midpoints {worst:.3f} (threshold 0.05)  "
          f"{'OK' if ok else 'FAILED'}")

    # how far the two honest curves are from each other, and from the official, at the nodes
    lt_nodes, _, include, _ = E.nodes(**__import__("model2").FULL_NODES)
    print(f"\n  at the engine's nodes before each anchor: median |log M(measured) - log M(x)| [dex]")
    print(f"      {'epoch':>6}{'vs official':>14}{'vs pre-epoch':>14}{'nodes':>8}")
    for k in EPOCHS:
        sel = include[k]
        lt = lt_nodes[sel]
        dm_o = np.median([np.mean(np.abs(E.log_mah(lt, mc) - E.log_mah(lt, hc))) for mc, hc in zip(meas, curves)])
        dm_p = np.median([np.mean(np.abs(E.log_mah(lt, mc) - E.log_mah(lt, pc))) for mc, pc in zip(meas, pre[k])])
        print(f"      {ANCHOR_Z[k]:>6}{dm_o:>14.4f}{dm_p:>14.4f}{int(sel.sum()):>8}")

    np.savez(OUTDIR / f"measured_curves{tag}.npz", rows=rows, n_knots=info["n_knots"], slope0=info["slope0"],
             fallback=info["fallback"], g2_snapshots=r2_snap, g2_midpoints=r2_mid, g2_ok=ok,
             t_min_past=T_MIN_PAST)
    print(f"\nwrote {OUTDIR / f'measured_curves{tag}.npz'}  (the curves are rebuilt from the catalog by "
          f"`build_measured`; nothing to load)")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
