"""exp87 — the method ladder L0-L3 behind one interface.

Every method takes a training design and target, a test design, and the
population's lower cut on the target (None = no cut), and returns a predictive
distribution for the test galaxies (`scoring.TruncNormal` or `Gridded`):

    predictive = method.fit_predict(X_train, y_train, X_test, cut)

Two routes to the truncated-normal predictive, which must agree on a linear
truth (the mechanics gate checks it):

  direct   the latent mean is a linear function of a (possibly expanded)
           design, fitted by the truncated likelihood itself (`heads.TruncLinear`)
  head     any point learner; its out-of-fold predictions on the training set
           feed the common head (`heads.CommonHead`). `mills=True` refits the
           learner on the pseudo-latent response so it learns the latent mean.

Everything that looks at the data (scalers, PCA, PLS, hyper-parameters) is
fitted inside the training set handed in; the caller owns the outer folds.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import heads as H                                        # noqa: E402
import scoring as S                                      # noqa: E402

RIDGE_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)


def inner_folds(n, seed=C.SEED):
    return list(KFold(C.N_INNER_FOLDS, shuffle=True, random_state=seed).split(np.arange(n)))


class Method:
    """Base: a name, the ladder rung, and `fit_predict`."""
    rung = "L?"

    def __init__(self, name):
        self.name = name
        self.info = {}

    def fit_predict(self, X_train, y_train, X_test, cut):
        raise NotImplementedError


class Climatology(Method):
    """L0: the training targets' own distribution, the same for every galaxy."""
    rung = "L0"

    def __init__(self, target="mh"):
        super().__init__("climatology")
        self.target = target

    def fit_predict(self, X_train, y_train, X_test, cut):
        lo = cut if cut is not None else min(C.GRID_LO[self.target], float(np.min(y_train)) - 0.3)
        grid = np.arange(lo, max(C.GRID_HI[self.target], float(np.max(y_train)) + 0.3) + C.GRID_STEP, C.GRID_STEP)
        clim = S.climatology(y_train, grid)
        return S.Gridded(grid, np.repeat(clim.pdf, len(X_test), axis=0))


class DirectLinear(Method):
    """L1/L2 direct route: transform -> truncated-likelihood linear latent mean.

    `transform`: None (the standardised design itself) or a factory returning an
    sklearn transformer fitted on the training design (PCA, PCA + polynomial,
    PCA + splines). The scale's log is linear in the ordinary-least-squares
    prediction. `ridge_grid`: candidates for the mean's ridge, chosen by inner
    cross-validated truncated CRPS (one value = no search).
    """

    def __init__(self, name, rung, transform=None, ridge_grid=(0.0,), heteroscedastic=True):
        super().__init__(name)
        self.rung, self.transform, self.ridge_grid, self.het = rung, transform, tuple(ridge_grid), heteroscedastic

    def _design(self, X_train, X_test, y_train):
        if self.transform is None:
            return np.asarray(X_train, float), np.asarray(X_test, float)
        tf = self.transform()
        try:
            A = tf.fit_transform(X_train, y_train)
        except TypeError:
            A = tf.fit_transform(X_train)
        A = A[0] if isinstance(A, tuple) else A
        return np.asarray(A, float), np.asarray(tf.transform(X_test), float)

    @staticmethod
    def _scale_design(D_train, D_other, y_train):
        coef = np.linalg.lstsq(np.column_stack([np.ones(len(D_train)), D_train]), y_train, rcond=None)[0]
        return (D_train @ coef[1:] + coef[0])[:, None], (D_other @ coef[1:] + coef[0])[:, None]

    def _fit(self, D_train, y_train, cut, ridge):
        Ds = self._scale_design(D_train, D_train, y_train)[0] if self.het else None
        return H.TruncLinear(cut, ridge=ridge).fit(D_train, y_train, Ds)

    def fit_predict(self, X_train, y_train, X_test, cut):
        D_train, D_test = self._design(X_train, X_test, y_train)
        ridge = self.ridge_grid[0]
        if len(self.ridge_grid) > 1:
            cost = []
            for r in self.ridge_grid:
                c = 0.0
                for tr, te in inner_folds(len(y_train)):
                    Ds_tr, Ds_te = self._scale_design(D_train[tr], D_train[te], y_train[tr])
                    model = H.TruncLinear(cut, ridge=r).fit(D_train[tr], y_train[tr], Ds_tr if self.het else None)
                    c += model.predict(D_train[te], Ds_te if self.het else None).crps(y_train[te]).sum()
                cost.append(c)
            ridge = self.ridge_grid[int(np.argmin(cost))]
        Ds_tr, Ds_te = self._scale_design(D_train, D_test, y_train)
        model = H.TruncLinear(cut, ridge=ridge).fit(D_train, y_train, Ds_tr if self.het else None)
        self.info = dict(ridge=float(ridge), n_features=int(D_train.shape[1]), nll=model.nll, converged=model.converged)
        self.model = model
        return model.predict(D_test, Ds_te if self.het else None)


class LearnerHead(Method):
    """L3 (and cross-checks): a point learner scored through the common head."""

    def __init__(self, name, rung, factory, mills=False, max_train=None):
        super().__init__(name)
        self.rung, self.factory, self.mills, self.max_train = rung, factory, mills, max_train

    def _oof(self, X, y, rng):
        f = np.empty(len(y))
        for tr, te in inner_folds(len(y)):
            tr = self._thin(tr, rng)
            f[te] = self.factory().fit(X[tr], y[tr]).predict(X[te])
        return f

    def _thin(self, rows, rng):
        if self.max_train is None or len(rows) <= self.max_train:
            return rows
        return rng.choice(rows, self.max_train, replace=False)

    def fit_predict(self, X_train, y_train, X_test, cut):
        rng = np.random.default_rng(C.SEED)
        X_train, X_test, y_train = np.asarray(X_train, float), np.asarray(X_test, float), np.asarray(y_train, float)
        response = y_train
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for iteration in range(3 if (self.mills and cut is not None) else 1):
                f_oof = self._oof(X_train, response, rng)
                head = H.CommonHead(cut, curvature=(iteration == 0)).fit(f_oof, y_train)
                if not (self.mills and cut is not None) or iteration == 2:
                    break
                m, s = head.model.latent(*head._designs(f_oof))
                response = H.mills_targets(y_train, m, s, cut)
            rows = self._thin(np.arange(len(y_train)), rng)
            f_test = self.factory().fit(X_train[rows], response[rows]).predict(X_test)
        self.info = dict(mills=bool(self.mills and cut is not None), head_nll=head.model.nll)
        return head.predict(f_test)


# --------------------------------------------------------------------------- #
# the registry                                                                  #
# --------------------------------------------------------------------------- #
def _pca(k):
    return lambda: make_pipeline(StandardScaler(), PCA(k, random_state=C.SEED))


def _pca_poly(k):
    return lambda: make_pipeline(StandardScaler(), PCA(k, random_state=C.SEED), StandardScaler(),
                                 PolynomialFeatures(2, include_bias=False))


def _pca_spline(k):
    return lambda: make_pipeline(StandardScaler(), PCA(k, random_state=C.SEED),
                                 SplineTransformer(n_knots=5, degree=3, include_bias=False))


def _poly_only():
    """A cubic in the (few) input columns: the flexible baseline for a one-column design."""
    return make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False))


class _PLSScores:
    def __init__(self, k):
        self.k = k

    def fit_transform(self, X, y):
        self.scaler = StandardScaler().fit(X)
        self.pls = PLSRegression(self.k, scale=False).fit(self.scaler.transform(X), y)
        return self.pls.transform(self.scaler.transform(X))

    def transform(self, X):
        return self.pls.transform(self.scaler.transform(X))


def _scaled(model):
    return lambda: make_pipeline(StandardScaler(), model())


def _pca_scaled(k, model):
    return lambda: make_pipeline(StandardScaler(), PCA(k, random_state=C.SEED), StandardScaler(), model())


def _gp():
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(4)) + WhiteKernel(0.05)
    return GaussianProcessRegressor(kernel, normalize_y=True, random_state=C.SEED, n_restarts_optimizer=0)


def _gbm():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=C.SEED)


def linear_methods(n_features):
    """L1 for a multi-column design: the direct truncated-likelihood family."""
    out = [DirectLinear("linear", "L1"), DirectLinear("ridge", "L1", ridge_grid=RIDGE_GRID)]
    for k in (2, 3, 4, 6):
        if k < n_features:
            out.append(DirectLinear(f"pca{k}", "L1", transform=_pca(k)))
            out.append(DirectLinear(f"pls{k}", "L1", transform=(lambda k=k: _PLSScores(k))))
    return out


def nonlinear_direct_methods(n_features):
    """L2: curvature in the latent mean, still by the truncated likelihood."""
    out = []
    for k in (3, 4):
        if k <= n_features:
            out.append(DirectLinear(f"pca{k}-poly2", "L2", transform=_pca_poly(k), ridge_grid=RIDGE_GRID))
            out.append(DirectLinear(f"pca{k}-spline", "L2", transform=_pca_spline(k), ridge_grid=RIDGE_GRID))
    return out


def flexible_methods(n_features, mills=True):
    """L3: nonparametric point learners through the common head."""
    k = min(4, n_features)
    out = [LearnerHead("ridge+head", "L1", _scaled(lambda: Ridge(alpha=1.0))),
           LearnerHead("knn", "L3", _pca_scaled(k, lambda: KNeighborsRegressor(30, weights="distance"))),
           LearnerHead("gbm", "L3", _gbm),
           LearnerHead("forest", "L3", lambda: RandomForestRegressor(300, min_samples_leaf=10, n_jobs=1, random_state=C.SEED)),
           LearnerHead("mlp", "L3", _scaled(lambda: MLPRegressor((64, 32), alpha=1e-2, max_iter=600, early_stopping=True,
                                                               random_state=C.SEED))),
           LearnerHead("gp", "L3", _pca_scaled(k, _gp), max_train=C.GP_MAX_POINTS)]
    if mills:
        out.append(LearnerHead("gbm-mills", "L3", _gbm, mills=True))
    return out


def scalar_method(name):
    """L0: one input column, the direct truncated-likelihood line."""
    return DirectLinear(name, "L0")
