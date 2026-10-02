"""exp87 — fitting a truncated-normal predictive by its own likelihood.

A sample cut at y >= c shows, at fixed input x, only the upper part of the
population's distribution of y. The honest predictive for such a sample is a
normal with LATENT mean m(x) and scale s(x), restricted to [c, inf):

    -log p(y | x, y >= c) = z^2/2 + log s + log sqrt(2 pi) + log(1 - Phi(l)),
    z = (y - m)/s,  l = (c - m)/s.

Fitting an ordinary regression and renormalising its normal above the cut
counts the truncation twice (the ordinary fit has already bent its mean toward
the truncated one). Three pieces:

  TruncLinear     m = Dm @ alpha, log s = Ds @ beta by maximum truncated
                  likelihood (analytic gradient, ridge on the mean's slopes).
                  `cut=None` is the ordinary heteroscedastic normal.
  CommonHead      the same family built on a POINT prediction f of any learner:
                  m = a0 + a1 f + a2 (f - mean f)^2, log s = b0 + b1 f. Every
                  flexible method is scored through it, so no method is
                  favoured by its own noise model.
  mills_targets   the pseudo-latent response y - s lambda((c - m)/s), whose
                  conditional mean is the latent mean: a learner refitted on
                  it learns m(x) instead of the truncated mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import scoring as S                                      # noqa: E402

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
LOG_S_BOUNDS = (np.log(0.01), np.log(2.0))


def inverse_mills(ell):
    """lambda(l) = phi(l) / (1 - Phi(l)), stable in both tails."""
    ell = np.asarray(ell, float)
    return np.exp(-0.5 * ell * ell - _LOG_SQRT_2PI - log_ndtr(-ell))


def truncnorm_nll(y, m, s, cut):
    z = (y - m) / s
    out = 0.5 * z * z + np.log(s) + _LOG_SQRT_2PI
    if cut is not None:
        out = out + log_ndtr(-(cut - m) / s)
    return out


class TruncLinear:
    """Maximum truncated-normal likelihood with a linear latent mean and a
    log-linear scale. Designs are standardised internally; `ridge` penalises
    the mean's slopes (per galaxy, in units of the standardised design)."""

    def __init__(self, cut=None, ridge=0.0, scale_ridge=1e-3):
        self.cut = None if cut is None or not np.isfinite(cut) else float(cut)
        self.ridge, self.scale_ridge = float(ridge), float(scale_ridge)

    @staticmethod
    def _standardise(D, mu=None, sd=None):
        D = np.atleast_2d(np.asarray(D, float))
        if mu is None:
            mu, sd = D.mean(0), D.std(0)
            sd = np.where(sd > 0, sd, 1.0)
        return np.column_stack([np.ones(len(D)), (D - mu) / sd]), mu, sd

    def fit(self, Dm, y, Ds=None):
        y = np.asarray(y, float)
        Xm, self.mu_m, self.sd_m = self._standardise(Dm)
        Ds = np.zeros((len(y), 0)) if Ds is None else Ds
        Xs, self.mu_s, self.sd_s = self._standardise(Ds) if np.size(Ds) else (np.ones((len(y), 1)), None, None)
        p, q = Xm.shape[1], Xs.shape[1]
        n = len(y)
        penalty_m = np.r_[0.0, np.full(p - 1, self.ridge)]
        penalty_s = np.r_[0.0, np.full(q - 1, self.scale_ridge)]
        alpha0 = np.linalg.solve(Xm.T @ Xm + n * np.diag(penalty_m) + 1e-10 * np.eye(p), Xm.T @ y)
        beta0 = np.zeros(q)
        beta0[0] = np.log(max(np.std(y - Xm @ alpha0), 1e-3))
        cut = self.cut

        def objective(theta):
            alpha, beta = theta[:p], theta[p:]
            m = Xm @ alpha
            ls = np.clip(Xs @ beta, *LOG_S_BOUNDS)
            s = np.exp(ls)
            z = (y - m) / s
            nll = 0.5 * z * z + ls + _LOG_SQRT_2PI
            dm = -z / s
            dls = 1.0 - z * z
            if cut is not None:
                ell = np.minimum((cut - m) / s, S.MAX_BELOW_CUT)
                lam = inverse_mills(ell)
                nll = nll + log_ndtr(-ell)
                dm = dm + lam / s
                dls = dls + lam * ell
            value = nll.mean() + 0.5 * np.sum(penalty_m * alpha * alpha) + 0.5 * np.sum(penalty_s * beta * beta)
            grad = np.r_[Xm.T @ dm / n + penalty_m * alpha, Xs.T @ dls / n + penalty_s * beta]
            return value, grad

        res = minimize(objective, np.r_[alpha0, beta0], jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        self.alpha, self.beta, self.nll = res.x[:p], res.x[p:], float(res.fun)
        self.converged = bool(res.success)
        return self

    def latent(self, Dm, Ds=None):
        Xm, _, _ = self._standardise(Dm, self.mu_m, self.sd_m)
        if self.mu_s is None:
            Xs = np.ones((len(Xm), 1))
        else:
            Xs, _, _ = self._standardise(Ds, self.mu_s, self.sd_s)
        return Xm @ self.alpha, np.exp(np.clip(Xs @ self.beta, *LOG_S_BOUNDS))

    def predict(self, Dm, Ds=None):
        m, s = self.latent(Dm, Ds)
        return S.TruncNormal(m, s, self.cut)

    def raw_coefficients(self):
        """(intercept, slopes) of the latent mean in the ORIGINAL units of the design."""
        slopes = self.alpha[1:] / self.sd_m
        return float(self.alpha[0] - np.sum(slopes * self.mu_m)), slopes


class CommonHead:
    """The predictive every flexible learner is scored through, fitted by
    truncated likelihood on out-of-fold point predictions of the training set."""

    def __init__(self, cut=None, curvature=True):
        self.cut, self.curvature = cut, curvature

    def _designs(self, f):
        f = np.asarray(f, float)
        Dm = np.column_stack([f, (f - self.f_mean) ** 2]) if self.curvature else f[:, None]
        return Dm, f[:, None]

    def fit(self, f_oof, y):
        self.f_mean = float(np.mean(f_oof))
        Dm, Ds = self._designs(f_oof)
        self.model = TruncLinear(self.cut).fit(Dm, y, Ds)
        return self

    def predict(self, f):
        Dm, Ds = self._designs(f)
        return self.model.predict(Dm, Ds)


def mills_targets(y, m, s, cut):
    """The pseudo-latent response: E[y - s lambda((c - m)/s) | x, y >= c] = m(x)."""
    if cut is None:
        return np.asarray(y, float)
    ell = np.minimum((cut - np.asarray(m, float)) / np.asarray(s, float), S.MAX_BELOW_CUT)
    return np.asarray(y, float) - np.asarray(s, float) * inverse_mills(ell)


def self_test():
    rng = np.random.default_rng(1)
    n = 60000
    x = rng.normal(0.0, 1.0, (n, 2))
    m = 13.1 + 0.25 * x[:, 0] - 0.1 * x[:, 1]
    s = np.exp(np.log(0.2) + 0.15 * x[:, 0])
    y = m + s * rng.standard_normal(n)
    keep = y >= 13.0
    x, y, m, s = x[keep], y[keep], m[keep], s[keep]
    fit = TruncLinear(13.0).fit(x, y, x[:, [0]])
    intercept, slopes = fit.raw_coefficients()
    m_hat, s_hat = fit.latent(x, x[:, [0]])
    print(f"    truncated ML on {keep.sum()} of {n} draws: intercept {intercept:.3f} (13.100), slopes {slopes[0]:+.3f} / {slopes[1]:+.3f} "
          f"(+0.250 / -0.100), median scale {np.median(s_hat):.3f} ({np.median(s):.3f})")
    assert abs(intercept - 13.1) < 0.02 and np.allclose(slopes, [0.25, -0.1], atol=0.02) and abs(np.median(s_hat) / np.median(s) - 1) < 0.05
    naive = TruncLinear(None).fit(x, y, x[:, [0]])
    print(f"    the ordinary fit on the same draws: slopes {naive.raw_coefficients()[1][0]:+.3f} / {naive.raw_coefficients()[1][1]:+.3f}, "
          f"median scale {np.median(naive.latent(x, x[:, [0]])[1]):.3f} (biased toward zero and narrow, as expected)")
    assert abs(naive.raw_coefficients()[1][0]) < 0.22
    # the pseudo-latent response has the latent mean
    pseudo = mills_targets(y, m, s, 13.0)
    slope = np.linalg.lstsq(np.column_stack([np.ones(len(y)), x]), pseudo, rcond=None)[0]
    assert abs(slope[1] - 0.25) < 0.02 and abs(slope[0] - 13.1) < 0.02, slope
    # the head recovers the latent family from the truncated conditional mean
    f = S.TruncNormal(m, s, 13.0).mean()
    head = CommonHead(13.0).fit(f, y)
    gain = S.TruncNormal(m, s, 13.0).crps(y).mean() / head.predict(f).crps(y).mean()
    print(f"    the common head on the true truncated mean: CRPS within {100 * (1 - gain):.2f}% of the oracle's")
    assert gain > 0.985
    print("  heads self-test OK")


if __name__ == "__main__":
    self_test()
