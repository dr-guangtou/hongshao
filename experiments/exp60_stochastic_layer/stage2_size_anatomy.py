"""exp60 Stage 2 — the size/shape anatomy on the incumbent: which single
parameter carries the per-galaxy individuality, and what its deviation
distribution looks like.

THE QUESTION (exp41's Stage 0, redone for the deposition-only model). The
layer's second component perturbs ONE base parameter per galaxy, drawn from
an empirical distribution. Before fitting any distribution: free each of the
seven parameters one at a time, per galaxy, and measure (a) how much
per-galaxy loss each buys (the individuality axis), (b) the winning axis's
per-galaxy deviation distribution, (c) how much of it is predictable from
halo features (belongs to conditioning — expected ~none, per exp41's finding
4 on the old kernel, but measured not assumed), and (d) whether it is the
SAME individuality Stage 1's core component already carries — if the two
correlate strongly, drawing them independently in Stage 3 would double-count
diversity, and the plan's "couple only if a gate demands it" becomes "couple,
the anatomy demands it".

HOW IT IS CHEAP. As in Stage 1: galaxies are independent in the forward
model, so sweeping a parameter SHARED across the population and reading each
galaxy's own loss curve gives every per-galaxy 1-D refit at once; the
per-galaxy minimum is refined by parabolic interpolation on its grid
neighbourhood. ~15 evaluations per parameter replace 2397 optimizations.

THE PER-GALAXY LOSS mirrors the population objective's structure (amplitude
term over SIGMA_A plus the 103 kpc-normalised shape term over L_F_REF,
masked epochs only), so "loss improvement" means the same thing it means in
every fit — but it is a per-galaxy diagnostic, defined here, and no
population number below is comparable with `fit.Problem` losses.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp60_stochastic_layer/stage2_size_anatomy.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term"):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import scoreboard as SB                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402

#: symmetric displacement grid per parameter, in the parameter's own units,
#: clipped to the box. Coarse on purpose — this is anatomy, not measurement;
#: the per-galaxy minimum is refined parabolically between neighbours.
DELTAS = np.array([-2.5, -1.6, -1.0, -0.6, -0.35, -0.2, -0.1, 0.0,
                   0.1, 0.2, 0.35, 0.6, 1.0, 1.6, 2.5])
OUT = HERE / "outputs" / "stage2_size_anatomy.npz"
RULE = "=" * 96


def gal_losses(pred, data, mask):
    """Per-galaxy loss, (n,), mirroring the population objective's structure:
    mean over the galaxy's masked epochs of (dA/SIGMA_A)^2 plus the mean
    squared relative error of the 103 kpc-normalised profile over L_F_REF^2.
    NaN where the galaxy has no admitted epoch or a broken prediction."""
    finite = np.isfinite(pred).all(axis=2) & (pred[:, :, F.I100] > 0)
    use = np.asarray(mask, bool) & finite & (data[:, :, F.I100] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dA = ((np.log10(pred[:, :, F.I100]) - np.log10(data[:, :, F.I100]))
              / F.SIGMA_A[None, :]) ** 2
        Fm = pred / pred[:, :, F.I100][:, :, None]
        Fd = data / data[:, :, F.I100][:, :, None]
        dF = np.mean(((Fm - Fd) / Fd) ** 2, axis=2) / F.L_F_REF ** 2
    tot = np.where(use, dA + dF, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(tot, axis=1)


def para_min(grid, ys):
    """Parabolic refinement of the minimum of ys(grid) per galaxy."""
    n = ys.shape[1]
    k = np.nanargmin(ys, axis=0)
    x0 = grid[k]
    y_min = ys[k, np.arange(n)]
    inner = (k > 0) & (k < len(grid) - 1)
    i = np.where(inner)[0]
    xl, xc, xr = grid[k[i] - 1], grid[k[i]], grid[k[i] + 1]
    yl = ys[k[i] - 1, i]
    yc = ys[k[i], i]
    yr = ys[k[i] + 1, i]
    denom = (yl - 2 * yc + yr)
    okd = np.abs(denom) > 1e-30
    shift = np.zeros(len(i))
    shift[okd] = 0.5 * (yl - yr)[okd] / denom[okd] * (xr - xl)[okd] / 2
    x0[i] = xc + np.clip(shift, (xl - xc), (xr - xc))
    return x0, y_min, k


def main(smoke=False):
    print(f"{RULE}\nexp60 STAGE 2 — the per-galaxy individuality axis of the "
          f"incumbent\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    n = len(recs)
    lo, hi = np.array(base.bounds()).T
    l0 = gal_losses(slow.predict(theta), data, mask)
    print(f"  sample {n}; per-galaxy loss at the incumbent: median "
          f"{np.nanmedian(l0):.3f}")

    results = {}
    print(f"\n  (a) WHICH PARAMETER CARRIES THE INDIVIDUALITY — free one per "
          f"galaxy at a time:")
    print(f"  {'parameter':<12}{'median gain':>13}{'p90 gain':>10}"
          f"{'grid edge hits':>16}")
    for j, name in enumerate(base.theta_names):
        grid = np.clip(theta[j] + DELTAS, lo[j], hi[j])
        ys = np.full((len(grid), n), np.nan)
        for g, v in enumerate(grid):
            th = theta.copy()
            th[j] = v
            ys[g] = gal_losses(slow.predict(th), data, mask)
        dbest, y_min, k = para_min(grid, ys)
        gain = 100 * (1 - y_min / l0)
        edge = float(np.mean((k == 0) | (k == len(grid) - 1)))
        results[name] = dict(delta=dbest - theta[j], gain=gain, edge=edge)
        print(f"  {name:<12}{np.nanmedian(gain):>12.1f}%"
              f"{np.nanpercentile(gain, 90):>9.1f}%"
              f"{100 * edge:>15.1f}%")

    names = list(base.theta_names)
    med = {nm: np.nanmedian(results[nm]["gain"]) for nm in names}
    top = max(names, key=lambda nm: med[nm])
    d_top = results[top]["delta"]
    print(f"\n  (b) THE WINNING AXIS: {top} (median gain {med[top]:.1f}%). "
          f"Per-galaxy deviation distribution:")
    q = np.nanpercentile(d_top, [5, 25, 50, 75, 95])
    print(f"    percentiles 5/25/50/75/95: "
          + "  ".join(f"{v:+.3f}" for v in q))

    print(f"\n  (c) mutual rank correlation of the per-galaxy deltas (do the "
          f"axes see one thing?):")
    from scipy.stats import spearmanr
    fin = np.isfinite(d_top)
    for nm in names:
        if nm == top:
            continue
        m = fin & np.isfinite(results[nm]["delta"])
        r = spearmanr(d_top[m], results[nm]["delta"][m]).statistic
        print(f"    {top} x {nm:<10} rho = {r:+.3f}")

    print(f"\n  (d) is it predictable from halo features (conditioning), and "
          f"is it Stage 1's core axis?")
    d = SB.load()
    idx_of = {int(r): i for i, r in enumerate(d["rows"])}
    sel = np.array([idx_of.get(int(h.row), -1) for h in recs])
    okd = sel >= 0
    feats = {"logMh(z=0.4)": lmh[:, 0],
             "f_form(z=0.4)": np.array([h.f_form[-1] for h in recs]),
             "fz2": np.where(okd, d["fz2"][np.clip(sel, 0, None)], np.nan),
             "c_exc(z=0.4)": np.where(okd, d["c_exc_k"][np.clip(sel, 0, None), 0], np.nan)}
    a1 = np.load(HERE / "outputs" / "stage1_core_anatomy.npz")
    a_i = a1["a_i"]
    if len(a_i) == n:
        feats["Stage-1 A_i (core axis)"] = a_i
    else:
        print(f"    (Stage-1 A_i is full-sample, this run is not — the "
              f"independence question needs the full run)")
    for nm, x in feats.items():
        m = fin & np.isfinite(x)
        r = spearmanr(d_top[m], x[m]).statistic
        print(f"    {top} vs {nm:<24} rho = {r:+.3f}"
              + ("   <- the independence question" if "A_i" in nm else ""))

    if not smoke:
        OUT.parent.mkdir(exist_ok=True)
        np.savez(OUT, top=top, names=np.array(names), l0=l0,
                 **{f"delta_{nm}": results[nm]["delta"] for nm in names},
                 **{f"gain_{nm}": results[nm]["gain"] for nm in names})
        print(f"\n  saved {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
