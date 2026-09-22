"""exp84 Stage 1 — the anatomy: what per-galaxy, PER-EPOCH deviations of the
two deposit sizes does the data ask for around the adopted mean?

exp60 Stage 2's trick, per epoch: galaxies are independent in the forward
model and a deviation at epoch k acts only on epoch k's evaluation, so ONE
call with the same deviation at every galaxy-epoch gives every
galaxy-epoch's own loss at that deviation. Fifteen calls per axis replace
11,780 one-dimensional refits; a 9x9 grid replaces the joint ones. The
per-galaxy-epoch deviation is the SMALLEST one within 1% of its best loss
(`TOL`, so a flat valley resolves toward zero), refined parabolically when
interior.
The sweeps go through `size_dev` itself, the code path the draws use.

Deliverables (printed, saved to `outputs/stage1_anatomy.npz`):
  (a) the control — every other parameter swept one at a time (shared,
      per-galaxy-epoch loss curves): which coordinates carry the
      individuality. The amplitude a0 is expected to lead (the amplitude
      draw carries it); the two sizes must lead the rest.
  (b) delta_c, delta_e (n, 5): the 1-D minima and the JOINT minima, their
      per-epoch percentiles, the loss gain, the fraction at the grid edge
      (the anatomy's own ceiling, stated before the draw is built).
  (c) the 5x5 cross-epoch correlation of each axis (what decision 2's
      correlation is fitted to), the c x e cross-correlation per epoch, and
      the anatomy's PERSISTENCE next to the truth's from Stage 0.
  (d) what would belong to conditioning: Spearman of the deltas with the
      halo mass at the epoch, f_form(z=0.4), and the early-mass fraction.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp84_layer_rebaseline/stage1_anatomy.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layer_common as LC                                # noqa: E402
import predictor as P                                    # noqa: E402

E = P.SL.E
RULE, THIN = P.RULE, P.THIN
OUT = P.OUTDIR / "stage1_anatomy.npz"
#: the 1-D deviation grid per size axis [dex]: dense near zero, out to a
#: factor ten each way (a deposit ten times larger or smaller than the mean's)
DELTAS_1D = np.array([-1.0, -0.7, -0.5, -0.35, -0.25, -0.15, -0.08, 0.0,
                      0.08, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0])
#: the joint grid, both axes
DELTAS_2D = np.array([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
#: the control sweep of the other parameters, as fractions of each one's box
BOX_FRACTIONS = np.array([-0.35, -0.2, -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.2, 0.35])
#: the parameters the control does not sweep: the two axes, and the held tau_d
NOT_SWEPT = ("log_f_c", "log_f_e", "tau_d")


#: THE SMALLEST-SUFFICIENT rule: a galaxy-epoch's deviation is the SMALLEST
#: |delta| whose loss is within TOL of that galaxy-epoch's best over the grid
#: (TOL as a fraction of its loss at the mean). Without it, a compact deposit
#: pushed below the 2 kpc grid sits on a flat valley the loss cannot see into,
#: and argmin picks the grid edge by noise (the smoke run put 37% of the
#: compact axis on the -1 dex edge). With it, a plateau resolves toward zero.
TOL = 0.01


def parabola(xl, xc, xr, yl, yc, yr):
    """(x_min, y_min) of the parabola through three points, clipped to
    [xl, xr]; where the curvature is not positive the centre point is kept."""
    denom = yl - 2 * yc + yr
    convex = denom > 1e-30
    safe = np.where(convex, denom, 1.0)
    shift = np.where(convex, 0.5 * (yl - yr) / safe * (xr - xl) / 2, 0.0)
    shift = np.clip(shift, xl - xc, xr - xc)
    y_par = np.where(convex, yc - (yr - yl) ** 2 / (8.0 * safe), yc)
    return xc + shift, np.minimum(y_par, yc)


def _smallest_sufficient(flat_grid_abs, ys_flat, l0):
    """ys_flat (G, ...) over a flattened grid with |delta| sizes
    `flat_grid_abs` (G,): the index of the smallest |delta| within TOL * l0
    of the minimum (ties broken by the smaller loss)."""
    finite = np.isfinite(ys_flat)
    y_min = np.min(np.where(finite, ys_flat, np.inf), axis=0)
    ok = finite & (ys_flat <= y_min + TOL * np.abs(l0))
    # rank: |delta| first, loss second
    order = np.argsort(flat_grid_abs, kind="stable")
    ys_o, ok_o = ys_flat[order], ok[order]
    first = np.argmax(ok_o, axis=0)                      # the first sufficient point in |delta| order
    return order[first], y_min


def argmin_refined_1d(grid, ys, l0):
    """ys (G, ...) -> (x, y, at_edge) per trailing index by the
    smallest-sufficient rule, refined parabolically when interior."""
    k, y_grid_min = _smallest_sufficient(np.abs(grid), ys, l0)
    idx = np.indices(k.shape)
    x = grid[k].astype(float)
    y = ys[(k, *idx)]
    inner = (k > 0) & (k < len(grid) - 1)
    ki = np.clip(k, 1, len(grid) - 2)
    xp, yp = parabola(grid[ki - 1], grid[ki], grid[ki + 1], ys[(ki - 1, *idx)], ys[(ki, *idx)], ys[(ki + 1, *idx)])
    x = np.where(inner, xp, x)
    y = np.where(inner, yp, y)
    return x, y, ~inner


def argmin_refined_2d(gc, ge, ys, l0):
    """ys (Gc, Ge, n, 5) -> (dc, de, y, edge) per galaxy-epoch by the
    smallest-sufficient rule (|delta_c| + |delta_e|), refined separably."""
    Gc, Ge = len(gc), len(ge)
    size = (np.abs(gc)[:, None] + np.abs(ge)[None, :]).ravel()
    k, _ = _smallest_sufficient(size, ys.reshape(Gc * Ge, *ys.shape[2:]), l0)
    ic, ie = k // Ge, k % Ge
    idx = np.indices(k.shape)
    y = ys[(ic, ie, *idx)]
    edge = (ic == 0) | (ic == Gc - 1) | (ie == 0) | (ie == Ge - 1)
    icc, iec = np.clip(ic, 1, Gc - 2), np.clip(ie, 1, Ge - 2)
    dc, yc_ = parabola(gc[icc - 1], gc[icc], gc[icc + 1], ys[(icc - 1, iec, *idx)], ys[(icc, iec, *idx)], ys[(icc + 1, iec, *idx)])
    de, ye_ = parabola(ge[iec - 1], ge[iec], ge[iec + 1], ys[(icc, iec - 1, *idx)], ys[(icc, iec, *idx)], ys[(icc, iec + 1, *idx)])
    c_edge, e_edge = (ic == 0) | (ic == Gc - 1), (ie == 0) | (ie == Ge - 1)
    dc = np.where(c_edge, gc[ic], dc)
    de = np.where(e_edge, ge[ie], de)
    y = np.where(c_edge | e_edge, y, np.minimum(yc_, ye_))
    return dc, de, y, edge


def corr_pairwise(x):
    """(n, m) -> (m, m) Pearson, pairwise-complete."""
    m = x.shape[1]
    out = np.full((m, m), np.nan)
    for a in range(m):
        for b in range(m):
            g = np.isfinite(x[:, a]) & np.isfinite(x[:, b])
            out[a, b] = np.corrcoef(x[g, a], x[g, b])[0, 1] if g.sum() > 10 else np.nan
    return out


def percentiles_table(delta, label):
    print(f"    {label:<28}" + "".join(f"{e:>26}" for e in LC.EPOCH_LABELS))
    print(f"    {'percentiles 5/25/50/75/95':<28}"
          + "".join(("  " + " ".join(f"{v:+.2f}" for v in np.nanpercentile(delta[:, j], [5, 25, 50, 75, 95]))).rjust(26)
                    for j in range(5)))


def main(smoke=False):
    print(f"{RULE}\nexp84 STAGE 1 — the anatomy: per-galaxy, per-epoch deviations of the two deposit sizes\n{RULE}\n")
    recs, data, keep, lmh, pred = P.build(smoke)
    rows = np.where(keep)[0]
    n = len(rows)
    d = data[rows]
    use = np.ones((n, 5), bool)
    t0 = np.load(P.OUTDIR / "stage0_targets.npz") if not smoke else None
    zeros = np.zeros((pred.n, 5))

    def losses(dev_c=0.0, dev_e=0.0, theta13=None):
        sd = dict(c=zeros + dev_c, e=zeros + dev_e)
        return LC.gal_epoch_losses(pred.predict(size_dev=sd, rows=rows, theta13=theta13), d, use)

    t_start = time.time()
    l0 = losses()
    print(f"  per-galaxy-epoch loss at the mean: median {np.nanmedian(l0):.3f} "
          f"(per epoch " + " ".join(f"{np.nanmedian(l0[:, j]):.3f}" for j in range(5)) + ")")

    # --- (b) the 1-D sweeps of the two axes ---------------------------------------
    one_d = {}
    for axis in P.AXES:
        ys = np.full((len(DELTAS_1D), n, 5), np.nan)
        for g, v in enumerate(DELTAS_1D):
            ys[g] = losses(**{f"dev_{axis}": v})
        x, y, edge = argmin_refined_1d(DELTAS_1D, ys, l0)
        one_d[axis] = dict(delta=x, y=y, edge=edge, ys=ys)
        gain = 100 * (1 - y / l0)
        print(f"\n  1-D axis {axis} ({P.AXIS_PARAMETER[axis]}): median gain {np.nanmedian(gain):.1f}% "
              f"(p90 {np.nanpercentile(gain, 90):.1f}%), at the grid edge {100 * np.nanmean(edge):.1f}% "
              f"({time.time() - t_start:.0f} s so far)")
        percentiles_table(x, f"delta_{axis} [dex]")

    # --- (b) the joint grid ---------------------------------------------------
    ys2 = np.full((len(DELTAS_2D), len(DELTAS_2D), n, 5), np.nan)
    for i, vc in enumerate(DELTAS_2D):
        for j, ve in enumerate(DELTAS_2D):
            ys2[i, j] = losses(dev_c=vc, dev_e=ve)
    dc, de, y2, edge2 = argmin_refined_2d(DELTAS_2D, DELTAS_2D, ys2, l0)
    gain2 = 100 * (1 - y2 / l0)
    print(f"\n  JOINT (delta_c, delta_e) on the {len(DELTAS_2D)}x{len(DELTAS_2D)} grid: median gain {np.nanmedian(gain2):.1f}% "
          f"(p90 {np.nanpercentile(gain2, 90):.1f}%; the 1-D axes alone {np.nanmedian(100 * (1 - one_d['c']['y'] / l0)):.1f}% / "
          f"{np.nanmedian(100 * (1 - one_d['e']['y'] / l0)):.1f}%), at an edge {100 * np.nanmean(edge2):.1f}% "
          f"({time.time() - t_start:.0f} s)")
    percentiles_table(dc, "delta_c (joint) [dex]")
    percentiles_table(de, "delta_e (joint) [dex]")
    print(f"    per-epoch gain median: " + " ".join(f"{np.nanmedian(gain2[:, j]):5.1f}%" for j in range(5)))

    # --- (c) the correlations ---------------------------------------------------
    print(f"\n{THIN}\n  (c) THE CORRELATION STRUCTURE of the joint deltas (Pearson, pairwise complete)\n{THIN}")
    both = np.column_stack([dc, de])                       # (n, 10): c at 5 epochs, then e
    corr10 = corr_pairwise(both)
    LC.print_matrix(corr10[:5, :5], f"delta_c across epochs (nearest-epoch mean {LC.nearest_epoch(corr10[:5, :5]):+.3f})")
    LC.print_matrix(corr10[5:, 5:], f"delta_e across epochs (nearest-epoch mean {LC.nearest_epoch(corr10[5:, 5:]):+.3f})")
    print(f"    c x e at the same epoch: " + " ".join(f"{corr10[j, 5 + j]:+.3f}" for j in range(5)))
    pers_c, pers_e = LC.persistence(dc), LC.persistence(de)
    print(f"\n  PERSISTENCE (Spearman, nearest-epoch mean): the anatomy's delta_c {LC.nearest_epoch(pers_c):+.3f}, "
          f"delta_e {LC.nearest_epoch(pers_e):+.3f}"
          + (f"; the TRUTH's size residual R20 {LC.nearest_epoch(t0['persistence_truth_R20']):+.3f}, "
             f"R50 {LC.nearest_epoch(t0['persistence_truth_R50']):+.3f}, R80 {LC.nearest_epoch(t0['persistence_truth_R80']):+.3f}"
             if t0 is not None else ""))

    # --- (d) what would belong to conditioning -------------------------------------
    print(f"\n{THIN}\n  (d) what a conditioning would see — Spearman of the joint deltas with halo features\n{THIN}")
    fform = np.array([h.f_form[-1] for h in recs])[rows]
    lm2 = np.array([E.log_mah(np.array([np.log10(2.0)]), pred.curves[i])[0] for i in rows])
    feats = {"log Mh at the epoch": lmh[rows], "f_form(z=0.4)": np.repeat(fform[:, None], 5, axis=1),
             "early fraction log M(2 Gyr) - log Mh(z_k)": lm2[:, None] - lmh[rows]}
    print(f"    {'feature':<44}{'axis':<6}" + "".join(f"{e:>9}" for e in LC.EPOCH_LABELS))
    feat_rho = {}
    for name, x in feats.items():
        for axis, delta in (("c", dc), ("e", de)):
            r = []
            for j in range(5):
                g = np.isfinite(delta[:, j]) & np.isfinite(x[:, j])
                r.append(spearmanr(delta[g, j], x[g, j]).statistic if g.sum() > 10 else np.nan)
            feat_rho[(name, axis)] = np.array(r)
            print(f"    {name:<44}{axis:<6}" + "".join(f"{v:>+9.3f}" for v in r))

    # --- (a) the control: the other parameters, one at a time ----------------------
    print(f"\n{THIN}\n  (a) THE CONTROL — every other parameter freed per galaxy-epoch, one at a time "
          f"(median loss gain; the two sizes must lead the rest)\n{THIN}")
    spec = pred.spec
    lo, hi = np.array(spec.bounds()).T
    control = {}
    names = [nm for nm in spec.theta_names if nm not in NOT_SWEPT]
    print(f"    {'parameter':<12}{'median gain':>13}{'p90 gain':>10}{'edge hits':>11}")
    for nm in names:
        j = spec.index(nm)
        grid = np.clip(pred.theta13[j] + BOX_FRACTIONS * (hi[j] - lo[j]), lo[j], hi[j])
        ys = np.full((len(grid), n, 5), np.nan)
        for g, v in enumerate(grid):
            ys[g] = losses(theta13=pred.theta_with(**{nm: v}))
        x, y, edge = argmin_refined_1d(grid, ys, l0)
        gain = 100 * (1 - y / l0)
        control[nm] = dict(gain_median=float(np.nanmedian(gain)), gain_p90=float(np.nanpercentile(gain, 90)),
                           edge=float(np.nanmean(edge)))
        print(f"    {nm:<12}{control[nm]['gain_median']:>12.1f}%{control[nm]['gain_p90']:>9.1f}%{100 * control[nm]['edge']:>10.1f}%")
    for axis in P.AXES:
        gain = 100 * (1 - one_d[axis]["y"] / l0)
        print(f"    {P.AXIS_PARAMETER[axis]:<12}{np.nanmedian(gain):>12.1f}%{np.nanpercentile(gain, 90):>9.1f}%"
              f"{100 * np.nanmean(one_d[axis]['edge']):>10.1f}%   <- the layer's axis {axis}")
    print(f"  ({pred.n_call} model calls, {(time.time() - t_start) / 60:.1f} min)")

    if smoke:
        print("\n  (smoke: nothing saved)")
        return
    np.savez(OUT, rows=rows, l0=l0, deltas_1d=DELTAS_1D, deltas_2d=DELTAS_2D,
             delta_c_1d=one_d["c"]["delta"], delta_e_1d=one_d["e"]["delta"],
             edge_c_1d=one_d["c"]["edge"], edge_e_1d=one_d["e"]["edge"],
             delta_c=dc, delta_e=de, y_joint=y2, gain_joint=gain2, edge_joint=edge2,
             corr10=corr10, persistence_c=pers_c, persistence_e=pers_e,
             control_names=np.array(list(control)),
             control_gain=np.array([[v["gain_median"], v["gain_p90"], v["edge"]] for v in control.values()]),
             feat_keys=np.array([f"{a}|{b}" for (a, b) in feat_rho]), feat_rho=np.array(list(feat_rho.values())),
             fform=fform, lmh=lmh[rows])
    print(f"\n  saved -> {OUT.relative_to(P.ROOT)}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
