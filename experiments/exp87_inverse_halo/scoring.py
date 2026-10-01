"""exp87 — predictive distributions on a truncated support and their proper scores.

Every method in the experiment hands back a PREDICTIVE DISTRIBUTION for each
galaxy, not a point. Two representations share one interface:

  TruncNormal(m, s, cut)   a normal with LATENT mean m and scale s, restricted
                           to [cut, inf) and renormalised. `cut=None` is the
                           plain normal. Closed forms throughout.
  Gridded(grid, pdf)       any density tabulated on a uniform grid (the
                           generative posterior, the climatology); the CDF is
                           piecewise linear and every integral is exact for it.

Scores (lower is better; the units are the target's, dex for a log mass):
  crps        the continuous ranked probability score — the PRIMARY metric
  logscore    minus the log predictive density at the truth [nats]
  twcrps      the threshold-weighted CRPS, weight 1[t >= t0] (the massive end,
              read without selecting galaxies by their true mass)
  pit         F(y): uniform on (0, 1) when the predictive is calibrated

Run `python scoring.py` for the self-test: the closed-form truncated-normal
scores against the gridded quadrature, and against Monte Carlo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
_SQRT_PI = np.sqrt(np.pi)
#: the latent mean is not allowed further than this many scales below the cut:
#: beyond it the truncated normal is an exponential tail and the survival
#: function underflows
MAX_BELOW_CUT = 6.0
PDF_FLOOR = 1e-12


def _phi(z):
    return np.exp(-0.5 * z * z - _LOG_SQRT_2PI)


class TruncNormal:
    """Normal(m, s) restricted to y >= cut. Arrays of shape (n,)."""

    def __init__(self, m, s, cut=None):
        self.s = np.asarray(s, float)
        m = np.asarray(m, float)
        self.cut = None if cut is None or not np.isfinite(cut) else float(cut)
        if self.cut is not None:
            m = np.maximum(m, self.cut - MAX_BELOW_CUT * self.s)
        self.m = m

    def _ell(self):
        return None if self.cut is None else (self.cut - self.m) / self.s

    def _survival(self):
        """A = P(latent >= cut), and its log."""
        if self.cut is None:
            return np.ones_like(self.m), np.zeros_like(self.m)
        ell = self._ell()
        return ndtr(-ell), log_ndtr(-ell)

    def logscore(self, y):
        z = (np.asarray(y, float) - self.m) / self.s
        return 0.5 * z * z + np.log(self.s) + _LOG_SQRT_2PI + self._survival()[1]

    def crps(self, y):
        z = (np.asarray(y, float) - self.m) / self.s
        if self.cut is None:
            return self.s * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * _phi(z) - 1.0 / _SQRT_PI)
        ell = self._ell()
        A, _ = self._survival()
        z = np.maximum(z, ell)                            # the truth cannot lie below the cut
        return self.s * (z * (2.0 * ndtr(z) - 1.0 - ndtr(ell)) / A + 2.0 * _phi(z) / A
                         - ndtr(-ell * np.sqrt(2.0)) / (_SQRT_PI * A * A))

    def pit(self, y):
        z = (np.asarray(y, float) - self.m) / self.s
        if self.cut is None:
            return ndtr(z)
        A, _ = self._survival()
        return np.clip((ndtr(z) - ndtr(self._ell())) / A, 0.0, 1.0)

    def quantile(self, p):
        p = np.asarray(p, float)
        if self.cut is None:
            return self.m + self.s * ndtri(p)
        ell = self._ell()
        A, _ = self._survival()
        return self.m + self.s * ndtri(np.clip(ndtr(ell) + p * A, 1e-300, 1.0 - 1e-16))

    def mean(self):
        if self.cut is None:
            return self.m
        ell = self._ell()
        return self.m + self.s * np.exp(-0.5 * ell * ell - _LOG_SQRT_2PI - self._survival()[1])

    def median(self):
        return self.quantile(0.5)

    def to_grid(self, grid):
        grid = np.asarray(grid, float)
        z = (grid[None, :] - self.m[:, None]) / self.s[:, None]
        pdf = _phi(z) / self.s[:, None]
        if self.cut is not None:
            pdf = np.where(grid[None, :] >= self.cut - 1e-12, pdf, 0.0)
        return Gridded(grid, pdf)

    def twcrps(self, y, t0, grid):
        return self.to_grid(grid).twcrps(y, t0)


class Gridded:
    """A density on a uniform grid, one row per galaxy (or one row broadcast to all)."""

    def __init__(self, grid, pdf):
        self.grid = np.asarray(grid, float)
        self.h = float(self.grid[1] - self.grid[0])
        pdf = np.atleast_2d(np.asarray(pdf, float))
        cells = 0.5 * (pdf[:, 1:] + pdf[:, :-1]) * self.h
        total = cells.sum(1, keepdims=True)
        self.pdf = pdf / total
        self.cdf = np.concatenate([np.zeros((len(pdf), 1)), np.cumsum(cells / total, axis=1)], axis=1)
        F0, F1 = self.cdf[:, :-1], self.cdf[:, 1:]
        cell_f2 = self.h * (F0 * F0 + F0 * F1 + F1 * F1) / 3.0
        G0, G1 = 1.0 - F0, 1.0 - F1
        cell_g2 = self.h * (G0 * G0 + G0 * G1 + G1 * G1) / 3.0
        zero = np.zeros((len(pdf), 1))
        self._cum_f2 = np.concatenate([zero, np.cumsum(cell_f2, axis=1)], axis=1)       # integral of F^2 from the grid's start
        self._cum_g2 = np.concatenate([zero, np.cumsum(cell_g2, axis=1)], axis=1)       # integral of (1-F)^2 from the grid's start

    def _rows(self, n):
        return np.zeros(n, int) if len(self.pdf) == 1 else np.arange(n)

    def _locate(self, x):
        x = np.clip(np.asarray(x, float), self.grid[0], self.grid[-1])
        j = np.minimum(((x - self.grid[0]) / self.h).astype(int), len(self.grid) - 2)
        return j, (x - self.grid[j]) / self.h

    def _integrals_to(self, x):
        """(int F^2, int (1-F)^2) from the grid's start up to x, exact for the piecewise-linear CDF."""
        j, u = self._locate(x)
        r = self._rows(len(j))
        F0 = self.cdf[r, j]
        Fx = F0 + u * (self.cdf[r, j + 1] - F0)
        f2 = self._cum_f2[r, j] + u * self.h * (F0 * F0 + F0 * Fx + Fx * Fx) / 3.0
        G0, Gx = 1.0 - F0, 1.0 - Fx
        g2 = self._cum_g2[r, j] + u * self.h * (G0 * G0 + G0 * Gx + Gx * Gx) / 3.0
        return f2, g2

    def crps(self, y):
        f2, g2 = self._integrals_to(y)
        r = self._rows(len(np.atleast_1d(y)))
        return f2 + (self._cum_g2[r, -1] - g2)

    def twcrps(self, y, t0):
        y = np.asarray(y, float)
        f2_y, g2_y = self._integrals_to(y)
        f2_t, g2_t = self._integrals_to(np.full(len(y), float(t0)))
        r = self._rows(len(y))
        total_g2 = self._cum_g2[r, -1]
        below = y <= t0                                   # the truth lies below the threshold: only (1-F)^2 from t0 on
        return np.where(below, total_g2 - g2_t, (f2_y - f2_t) + (total_g2 - g2_y))

    def pit(self, y):
        j, u = self._locate(y)
        r = self._rows(len(j))
        return self.cdf[r, j] + u * (self.cdf[r, j + 1] - self.cdf[r, j])

    def logscore(self, y):
        j, u = self._locate(y)
        r = self._rows(len(j))
        return -np.log(np.maximum(self.pdf[r, j] + u * (self.pdf[r, j + 1] - self.pdf[r, j]), PDF_FLOOR))

    def mean(self):
        return np.trapezoid(self.pdf * self.grid[None, :], dx=self.h, axis=1)

    def quantile(self, p):
        out = np.empty(len(self.cdf))
        for i, row in enumerate(self.cdf):
            out[i] = np.interp(p, row, self.grid)
        return out

    def median(self):
        return self.quantile(0.5)

    def to_grid(self, grid=None):
        return self


def target_grid(target, cut=None):
    lo = C.GRID_LO[target] if cut is None else float(cut)
    return np.arange(lo, C.GRID_HI[target] + 0.5 * C.GRID_STEP, C.GRID_STEP)


def climatology(y_train, grid, bin_width=0.05):
    """The no-information predictive: the training targets' own distribution,
    as a histogram density on the grid (one row, broadcast to every galaxy). A
    histogram keeps a hard selection edge sharp; a kernel estimate would round it."""
    y_train = np.asarray(y_train, float)
    edges = np.arange(grid[0], grid[-1] + bin_width, bin_width)
    hist = np.histogram(y_train[np.isfinite(y_train)], edges)[0].astype(float) + 1e-3
    centres = 0.5 * (edges[:-1] + edges[1:])
    pdf = np.interp(grid, centres, hist, left=hist[0], right=hist[-1])
    return Gridded(grid, pdf)


def summarize(pred, y, t0=None, grid=None, n=None):
    """Per-galaxy score arrays and their summary for one predictive against the truth."""
    y = np.asarray(y, float)
    crps, logs, pit = pred.crps(y), pred.logscore(y), pred.pit(y)
    mean, median = pred.mean(), pred.median()
    if len(np.atleast_1d(mean)) == 1:
        mean, median = np.full(len(y), mean[0]), np.full(len(y), median[0])
    err = mean - y
    per = dict(crps=crps, logscore=logs, pit=pit, err_mean=err, err_median=median - y)
    out = dict(n=int(len(y)), crps=float(np.mean(crps)), logscore=float(np.mean(logs)),
               rmse=float(np.sqrt(np.mean(err ** 2))), mae_median=float(np.mean(np.abs(median - y))),
               bias=float(np.mean(err)), catastrophic=float(np.mean(np.abs(median - y) > C.CATASTROPHIC_DEX)),
               cover68=float(np.mean((pit > 0.16) & (pit < 0.84))), cover90=float(np.mean((pit > 0.05) & (pit < 0.95))),
               pit_mean=float(np.mean(pit)), pit_var=float(np.var(pit)),
               width_ratio=float(np.std(mean) / np.std(y)))
    if t0 is not None:
        tw = pred.twcrps(y, t0) if isinstance(pred, Gridded) else pred.twcrps(y, t0, grid)
        per["twcrps"] = tw
        out["twcrps"] = float(np.mean(tw))
    return out, per


def paired_bootstrap(a, b, n_boot=None, seed=None):
    """Mean of (a - b) over galaxies with a bootstrap 95% interval, and the same
    relative to mean(b). `a`, `b`: per-galaxy scores of two predictives on the
    SAME galaxies. Negative = `a` is better (scores are costs)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    rng = np.random.default_rng(C.SEED if seed is None else seed)
    idx = rng.integers(0, len(a), size=(n_boot or C.N_BOOTSTRAP, len(a)))
    d = (a - b)[idx].mean(1)
    rel = d / b[idx].mean(1)
    return dict(diff=float(np.mean(a - b)), lo=float(np.quantile(d, 0.025)), hi=float(np.quantile(d, 0.975)),
                rel=float(np.mean(a - b) / np.mean(b)), rel_lo=float(np.quantile(rel, 0.025)), rel_hi=float(np.quantile(rel, 0.975)))


def significant_gain(candidate, reference, delta):
    """Is `candidate` better than `reference` by the plan's rule? The paired
    interval of the CRPS difference must exclude zero and the relative gain
    must reach `delta`. Returns (bool, the bootstrap summary)."""
    boot = paired_bootstrap(candidate, reference)
    return bool(boot["hi"] < 0.0 and -boot["rel"] >= delta), boot


def reliability(pred_mean, y, n_bins=8):
    """Calibration read in bins of the PREDICTION (never of the truth): per bin
    the median prediction, the median truth and the median residual; and the
    slope of the truth on the prediction (1 for a calibrated conditional mean)."""
    pred_mean, y = np.asarray(pred_mean, float), np.asarray(y, float)
    q = np.quantile(pred_mean, np.linspace(0, 1, n_bins + 1))
    rows = []
    for a, b in zip(q[:-1], q[1:]):
        sel = (pred_mean >= a) & (pred_mean <= b)
        if sel.sum() >= 10:
            rows.append((float(np.median(pred_mean[sel])), float(np.median(y[sel])), float(np.median(y[sel] - pred_mean[sel])),
                         float(np.std(y[sel] - pred_mean[sel])), int(sel.sum())))
    slope = float(np.polyfit(pred_mean, y, 1)[0])
    return dict(slope=slope, bins=rows, max_offset=float(max(abs(r[2]) for r in rows)))


def self_test():
    rng = np.random.default_rng(0)
    n = 4000
    m = rng.normal(13.2, 0.3, n)
    s = rng.uniform(0.12, 0.3, n)
    cut = 13.0
    tn = TruncNormal(m, s, cut)
    y = tn.quantile(rng.uniform(1e-6, 1 - 1e-6, n))
    grid = np.arange(cut, 17.0, 0.0025)
    gr = tn.to_grid(grid)
    for name in ("crps", "logscore", "pit"):
        a, b = getattr(tn, name)(y), getattr(gr, name)(y)
        tol = 2e-4 if name == "crps" else 2e-3            # the gridded CDF is a trapezoid sum: 1e-3 in the PIT
        assert np.max(np.abs(a - b)) < tol, (name, float(np.max(np.abs(a - b))))
        print(f"    {name:<9} closed form vs gridded: max |difference| {np.max(np.abs(a - b)):.1e}")
    assert np.max(np.abs(tn.mean() - gr.mean())) < 2e-4 and np.max(np.abs(tn.median() - gr.median())) < 3e-3
    # CRPS against its definition E|X - y| - 0.5 E|X - X'| by Monte Carlo on a few galaxies
    for i in range(3):
        draws = tn.quantile(rng.uniform(size=(200000, 1)))[:, 0] if False else \
            TruncNormal(np.full(200000, m[i]), np.full(200000, s[i]), cut).quantile(rng.uniform(1e-9, 1 - 1e-9, 200000))
        mc = np.mean(np.abs(draws - y[i])) - 0.5 * np.mean(np.abs(draws[:100000] - draws[100000:]))
        assert abs(mc - tn.crps(y)[i]) < 3e-3, (mc, tn.crps(y)[i])
    pit = tn.pit(y)
    assert abs(pit.mean() - 0.5) < 0.02 and abs(pit.var() - 1 / 12) < 0.01, (pit.mean(), pit.var())
    # the threshold-weighted score equals the CRPS when the threshold is the grid's start
    assert np.max(np.abs(gr.twcrps(y, grid[0]) - gr.crps(y))) < 1e-9
    assert np.all(gr.twcrps(y, 14.0) <= gr.crps(y) + 1e-12)
    # the plain normal nests: no cut
    pn = TruncNormal(m, s, None)
    from hongshao.metrics import crps_gaussian, gaussian_logscore
    assert np.allclose(pn.crps(y), crps_gaussian(y, m, s)) and np.allclose(pn.logscore(y), gaussian_logscore(y, m, s))
    # a true predictive scores better than a shifted or a double-counted one
    naive = TruncNormal(tn.mean(), s, None)
    assert tn.crps(y).mean() < naive.crps(y).mean() and tn.crps(y).mean() < TruncNormal(m + 0.1, s, cut).crps(y).mean()
    print(f"    truncated CRPS {tn.crps(y).mean():.4f} < plain normal at the truncated mean {naive.crps(y).mean():.4f}; "
          f"PIT mean {pit.mean():.3f}, variance {pit.var():.4f} (uniform: 0.500, 0.0833)")
    print("  scoring self-test OK")


if __name__ == "__main__":
    self_test()
