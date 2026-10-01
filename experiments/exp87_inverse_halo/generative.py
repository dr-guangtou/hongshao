"""exp87 — L5, the GENERATIVE INVERSE: p(y | x) is proportional to p(x | y) pi(y).

Why it is on the ladder. A sample selected on the target y (a halo-mass cut)
leaves the FORWARD conditional p(x | y) unbiased, as long as the selection
does not depend on x at fixed y. So the forward relation can be fitted by
ordinary regression of the curve of growth on the halo mass, on all the
galaxies, and the inverse follows from Bayes' rule with an EXPLICIT prior
pi(y). The prior is the only place the selection enters:

  prior="train"  the training targets' own distribution (a histogram: a hard
                 selection edge stays sharp). The right prior for scoring on a
                 sample drawn like the training set.
  prior="box"    the TNG300 box's halo-mass function above the cut
                 (`data.log_prior_density`): the complete population. On the
                 parent sample the two must agree; elsewhere the difference is
                 the price of the selection.

The forward model: the curve of growth is standardised and compressed to `k`
principal components inside the training fold; each component's mean is a
polynomial of degree `degree` in y; the residuals share one full covariance.
`student=True` replaces the normal residual by a multivariate Student-t whose
degrees of freedom are chosen by the training likelihood (heavy tails).

`resolution` (in `info`) is the prior-free forward resolution
(b^T Sigma^-1 b)^(-1/2) at the training median of y: the width the posterior
would have under a flat prior — the one number comparable across populations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import data as D                                         # noqa: E402
import methods as M                                      # noqa: E402
import scoring as S                                      # noqa: E402

NU_GRID = (3.0, 5.0, 8.0, 15.0, 40.0)


class GenerativeInverse(M.Method):
    rung = "L5"

    def __init__(self, name, k=4, degree=1, prior="train", student=False, epoch=0, target="mh"):
        super().__init__(name)
        self.k, self.degree, self.prior, self.student, self.epoch, self.target = k, degree, prior, student, epoch, target

    def _basis(self, y):
        t = np.asarray(y, float) - self.y0
        return np.column_stack([t ** d for d in range(self.degree + 1)])

    def _log_like(self, z, grid):
        """log p(z_i | y_g) for every test galaxy i and grid value g: (n, G)."""
        mean = self._basis(grid) @ self.coef                                  # (G, k)
        diff = z[:, None, :] - mean[None, :, :]                               # (n, G, k)
        maha = np.einsum("ngk,kl,ngl->ng", diff, self.prec, diff)
        if self.nu is None:
            return -0.5 * maha
        return -0.5 * (self.nu + self.k_eff) * np.log1p(maha / self.nu)

    def fit_predict(self, X_train, y_train, X_test, cut):
        X_train, X_test, y_train = np.asarray(X_train, float), np.asarray(X_test, float), np.asarray(y_train, float)
        self.k_eff = min(self.k, X_train.shape[1])
        scaler = StandardScaler().fit(X_train)
        pca = PCA(self.k_eff, random_state=C.SEED).fit(scaler.transform(X_train))
        z_train, z_test = pca.transform(scaler.transform(X_train)), pca.transform(scaler.transform(X_test))
        self.y0 = float(np.median(y_train))
        B = self._basis(y_train)
        self.coef = np.linalg.lstsq(B, z_train, rcond=None)[0]                # (degree+1, k)
        resid = z_train - B @ self.coef
        cov = np.cov(resid.T).reshape(self.k_eff, self.k_eff)
        self.nu = None
        if self.student:
            maha = np.einsum("nk,kl,nl->n", resid, np.linalg.inv(cov), resid)
            best = (-np.inf, None)
            for nu in NU_GRID:
                scale = (nu - 2.0) / nu                                        # scale matrix = cov * (nu-2)/nu
                ll = np.sum(gammaln(0.5 * (nu + self.k_eff)) - gammaln(0.5 * nu) - 0.5 * self.k_eff * np.log(nu * np.pi * scale)
                            - 0.5 * (nu + self.k_eff) * np.log1p(maha / scale / nu))
                if ll > best[0]:
                    best = (ll, nu)
            self.nu = best[1]
            cov = cov * (self.nu - 2.0) / self.nu
        self.prec = np.linalg.inv(cov)
        slope = self.coef[1] if self.degree >= 1 else np.zeros(self.k_eff)
        resolution = float(1.0 / np.sqrt(max(slope @ self.prec @ slope, 1e-12)))
        lo = cut if cut is not None else min(C.GRID_LO[self.target], float(y_train.min()) - 0.5)
        hi = max(C.GRID_HI[self.target], float(y_train.max()) + 0.5)
        grid = np.arange(lo, hi + C.GRID_STEP, C.GRID_STEP)
        if self.prior == "box":
            assert self.target == "mh", "the box prior is a halo-mass function"
            log_prior = D.log_prior_density(grid, self.epoch)
        else:
            log_prior = np.log(S.climatology(y_train, grid).pdf[0])
        log_post = self._log_like(z_test, grid) + log_prior[None, :]
        log_post -= log_post.max(1, keepdims=True)
        self.info = dict(k=self.k_eff, degree=self.degree, prior=self.prior, nu=self.nu, resolution=resolution)
        return S.Gridded(grid, np.exp(log_post))


def generative_methods(epoch, target="mh", box=True):
    out = [GenerativeInverse("gen-k3", k=3, epoch=epoch, target=target),
           GenerativeInverse("gen-k4", k=4, epoch=epoch, target=target),
           GenerativeInverse("gen-k6", k=6, epoch=epoch, target=target),
           GenerativeInverse("gen-k12", k=12, epoch=epoch, target=target),
           GenerativeInverse("gen-k24", k=24, epoch=epoch, target=target),
           GenerativeInverse("gen-k4-quad", k=4, degree=2, epoch=epoch, target=target),
           GenerativeInverse("gen-k4-t", k=4, student=True, epoch=epoch, target=target)]
    if box and target == "mh":
        out.append(GenerativeInverse("gen-k4-box", k=4, prior="box", epoch=epoch, target=target))
    return out


def self_test():
    """On the synthetic world the generative inverse with the true prior family
    must score like the oracle, and its resolution must equal the latent scale."""
    import synthetic as SY
    rng = np.random.default_rng(3)
    x, y, m, s, fold = SY.draw_sample(rng)
    used = fold >= 0
    import harness as HN
    gen, infos = HN.oof_scores(GenerativeInverse("gen", k=4), x, y, fold, SY.CUT)
    oracle = S.TruncNormal(m, s, SY.CUT).crps(y)[used].mean()
    res = float(np.mean([i["resolution"] for i in infos]))
    print(f"    generative inverse on the synthetic sample: CRPS {gen['crps'][used].mean():.4f} vs the oracle's {oracle:.4f}; "
          f"forward resolution {res:.4f} vs the latent scale {SY.S_LATENT:.4f}")
    assert gen["crps"][used].mean() / oracle - 1.0 < 0.02 and abs(res / SY.S_LATENT - 1.0) < 0.05
    print("  generative self-test OK")


if __name__ == "__main__":
    self_test()
