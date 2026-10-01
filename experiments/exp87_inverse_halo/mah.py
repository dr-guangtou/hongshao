"""exp87 Stage 5 — vector to vector: the halo's mass accretion history from the
stellar curve of growth.

THE TARGET is the history's SHAPE before the observed epoch, with the mass at
the epoch divided out (the mass itself is Stages 3-4's target):

    D(tau) = log10 Mpeak(t_k - tau) - log10 M(t_k),   tau = 0.5, 1, 2, 3, 4, 6 Gyr
             (only lookbacks reaching no earlier than 1 Gyr after the Big Bang)

from the dense M200c history of the main progenitor (running maximum).
Everything used is PRE-EPOCH: nothing after t_k enters a target, so the
official DiffMAH fit (anchored at z = 0, it sees the future: C19) is not a
target; the leak-free per-epoch DiffMAH parameters of exp74 are, scored in
CURVE space (the predicted parameters' curve against the measured D).

Methods (linear rung first; the rule decides whether to go on):
  ridge        one ridge regression per lookback on the 24 points
  pls3         partial least squares, three components
  rrr2         reduced-rank regression, rank 2
  gbm-modes    gradient boosting on each of the history's first three
               principal components (fitted in the training fold)
  diffmah      ridge on the three DiffMAH shape parameters, read in curve space
  multi-epoch  ridge on the curves of growth at ALL epochs up to the observed
               one (a labelled extension: the stellar history as the input)

Baselines every method is read against:
  climatology  the training mean history
  mediated     what the curve of growth says through its halo-mass estimate
               alone: the history regressed on the CoG-predicted mass
  oracle-mh    the history regressed on the TRUE mass at the epoch
  cog+oracle   ridge on the 24 points plus the true mass

Scores, out of fold: the RMS error per lookback [dex]; the per-galaxy error of
the history's FIRST principal component and of the whole curve, compared
between methods by the paired bootstrap (the plan's rule, delta from the gate).

Run:  PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/mah.py [--epochs 0 1 2 3 4] [--smoke]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import data as D                                         # noqa: E402
import methods as M                                      # noqa: E402
import scoring as S                                      # noqa: E402

RULE = "=" * 100
SO_NPZ = C.ROOT / "experiments/exp46_highz_ridge/outputs/so_history.npz"
PRE_NPZ = C.ROOT / "experiments/exp74_c19_history_leak/outputs/history_curves.npz"
TIME_TXT = C.ROOT / "data/external/tng_cosmic_time.txt"
LOOKBACKS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
T_MIN_GYR = 1.0
H_LITTLE = 0.6774
ALPHAS = np.logspace(-3, 3, 13)
MAH_BOARD = C.OUTDIR / "mah_scoreboard.jsonl"


def history_targets(sample):
    """(Y (n, J), lookbacks used, t_k) for a curated sample at its epoch: the
    log peak-mass ratio at each lookback before the epoch."""
    so = np.load(SO_NPZ, allow_pickle=True)
    t_snap = np.loadtxt(TIME_TXT)
    snap = C.ANCHOR_SNAP[sample.epoch]
    t_k = float(t_snap[snap])
    taus = [tau for tau in LOOKBACKS if t_k - tau >= T_MIN_GYR]
    mass = np.asarray(so["Group_M_Crit200"], float)[sample.flags["population_row"], :snap + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.log10(mass * 1e10 / H_LITTLE)
    Y = np.full((len(sample), len(taus)), np.nan)
    times = t_snap[:snap + 1]
    for i, row in enumerate(logm):
        ok = np.isfinite(row)
        if ok.sum() < 5 or not ok[-1]:
            continue
        peak = np.maximum.accumulate(row[ok])
        Y[i] = np.interp([t_k - tau for tau in taus], times[ok], peak) - peak[-1]
    return Y, taus, t_k


def diffmah_shape(sample):
    """exp74's pre-epoch DiffMAH shape parameters (logtc, early, late) and the anchor log t0, (n, 3), (n,)."""
    pre = np.load(PRE_NPZ, allow_pickle=True)
    rows, k = sample.flags["population_row"], sample.epoch
    return np.column_stack([pre[key][rows, k] for key in ("logtc", "early", "late")]), np.asarray(pre["logt0"][rows, k], float)


def diffmah_curve(params, logt0, taus, t_k):
    """D(tau) of the DiffMAH curve with shape `params` (n, 3), anchored at the epoch."""
    out = np.empty((len(params), len(taus)))
    for j, tau in enumerate(taus):
        lt = np.log10(t_k - tau)
        index = params[:, 1] + (params[:, 2] - params[:, 1]) * expit(3.5 * (lt - params[:, 0]))
        out[:, j] = index * (lt - logt0)                  # log M(t) - log M(t0): the anchor mass cancels
    return out


# --------------------------------------------------------------------------- #
# vector methods: fit_predict(X_train, Y_train, X_test) -> Y_test               #
# --------------------------------------------------------------------------- #
def ridge_vec(X_train, Y_train, X_test):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)).fit(X_train, Y_train)
    return model.predict(X_test)


def pls_vec(k):
    def run(X_train, Y_train, X_test):
        scaler = StandardScaler().fit(X_train)
        return PLSRegression(min(k, X_train.shape[1]), scale=False).fit(scaler.transform(X_train), Y_train).predict(scaler.transform(X_test))
    return run


def rrr_vec(rank):
    def run(X_train, Y_train, X_test):
        scaler = StandardScaler().fit(X_train)
        A = scaler.transform(X_train)
        mean = Y_train.mean(0)
        beta = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ (Y_train - mean))
        fitted = A @ beta
        _, _, vt = np.linalg.svd(fitted, full_matrices=False)
        proj = vt[:rank].T @ vt[:rank]
        return scaler.transform(X_test) @ beta @ proj + mean
    return run


def gbm_modes(X_train, Y_train, X_test, n_modes=3):
    pca = PCA(min(n_modes, Y_train.shape[1])).fit(Y_train)
    scores = pca.transform(Y_train)
    pred = np.column_stack([M._gbm().fit(X_train, scores[:, j]).predict(X_test) for j in range(scores.shape[1])])
    return pca.inverse_transform(pred)


def poly_vec(X_train, Y_train, X_test):
    model = make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), RidgeCV(alphas=ALPHAS)).fit(X_train, Y_train)
    return model.predict(X_test)


def mean_vec(X_train, Y_train, X_test):
    return np.tile(Y_train.mean(0), (len(X_test), 1))


def mediated_vec(Xa_train, Y_train, Xa_test):
    """The history through the curve of growth's own mass estimate ONLY. The
    design's last column is the true mass (used for training the mass estimate,
    never read for the test galaxies); the history is regressed on the mass
    estimate, out of fold inside the training set."""
    cog, mass = Xa_train[:, :-1], Xa_train[:, -1]
    estimate = np.empty(len(mass))
    for a, b in M.inner_folds(len(mass)):
        estimate[b] = ridge_vec(cog[a], mass[a], cog[b])
    estimate_test = ridge_vec(cog, mass, Xa_test[:, :-1])
    return poly_vec(estimate[:, None], Y_train, estimate_test[:, None])


def oof_vector(fit_predict, X, Y, fold):
    out = np.full(Y.shape, np.nan)
    for f in range(C.N_FOLDS):
        tr, te = (fold >= 0) & (fold != f), fold == f
        out[te] = fit_predict(X[tr], Y[tr], X[te])
    return out


def mode1_scores(Y, pred, fold):
    """The first principal component of the history (fitted in each training fold): truth and prediction, (n,), (n,)."""
    truth, guess = np.full(len(Y), np.nan), np.full(len(Y), np.nan)
    for f in range(C.N_FOLDS):
        tr, te = (fold >= 0) & (fold != f), fold == f
        pca = PCA(1).fit(Y[tr])
        sign = np.sign(pca.components_[0].sum()) or 1.0
        truth[te], guess[te] = sign * pca.transform(Y[te])[:, 0], sign * pca.transform(pred[te])[:, 0]
    return truth, guess


def run_epoch(epoch, smoke=False):
    s = D.load_curated(epoch, verbose=False)
    Y, taus, t_k = history_targets(s)
    mh = s.targets["mh"]
    ok = np.isfinite(Y).all(1) & np.isfinite(mh)
    fold = np.where(ok & ~s.lockbox, s.fold, -1)
    if smoke:
        keep = np.flatnonzero(fold >= 0)[::6]
        thin = np.full(len(fold), -1)
        thin[keep] = fold[keep]
        fold = thin
    used = fold >= 0
    X = s.logcog
    delta = float(json.loads((C.OUTDIR / "gate_mechanics.json").read_text())["delta"])
    print(f"\n--- z = {C.ANCHOR_Z[epoch]} (t = {t_k:.2f} Gyr): {used.sum()} development galaxies; lookbacks {taus} Gyr; "
          f"median history D = " + " / ".join(f"{v:+.3f}" for v in np.median(Y[used], axis=0))
          + "; its scatter " + " / ".join(f"{v:.3f}" for v in np.std(Y[used], axis=0)) + " dex ---", flush=True)

    preds = {}
    t0 = time.time()
    preds["climatology"] = oof_vector(mean_vec, X, Y, fold)
    preds["mediated"] = oof_vector(mediated_vec, np.column_stack([X, mh]), Y, fold)
    preds["oracle-mh"] = oof_vector(poly_vec, mh[:, None], Y, fold)
    preds["ridge"] = oof_vector(ridge_vec, X, Y, fold)
    preds["pls3"] = oof_vector(pls_vec(3), X, Y, fold)
    preds["rrr2"] = oof_vector(rrr_vec(2), X, Y, fold)
    preds["gbm-modes"] = oof_vector(gbm_modes, X, Y, fold)
    preds["cog+oracle"] = oof_vector(ridge_vec, np.column_stack([X, mh]), Y, fold)
    preds["cog+oracle-gbm"] = oof_vector(gbm_modes, np.column_stack([X, mh]), Y, fold)
    # the DiffMAH route: predict the three shape parameters, read the curve
    params, logt0 = diffmah_shape(s)
    okp = np.isfinite(params).all(1)
    fold_p = np.where(okp, fold, -1)
    p_hat = oof_vector(ridge_vec, X, np.where(okp[:, None], params, 0.0), fold_p)
    preds["diffmah"] = np.where(okp[:, None], diffmah_curve(p_hat, logt0, taus, t_k), np.nan)
    preds["diffmah-truth"] = np.where(okp[:, None], diffmah_curve(params, logt0, taus, t_k), np.nan)
    # the stellar history as the input (curves of growth at this and every earlier epoch)
    if epoch < 4:
        stack = [X]
        for later in range(epoch + 1, 5):
            other = D.load_curated(later, verbose=False)
            assert np.array_equal(other.index, s.index)
            stack.append(other.logcog)
        preds["multi-epoch"] = oof_vector(ridge_vec, np.column_stack(stack), Y, fold)
    print(f"    fitted {len(preds)} predictors in {time.time() - t0:.0f} s", flush=True)

    rows_out = []
    err1, errc = {}, {}
    for name, pred in preds.items():
        good = used & np.isfinite(pred).all(1)
        rms = np.sqrt(np.mean((pred[good] - Y[good]) ** 2, axis=0))
        t1, g1 = mode1_scores(Y, np.where(np.isfinite(pred), pred, 0.0), fold)
        err1[name] = np.where(good, np.abs(g1 - t1), np.nan)
        errc[name] = np.where(good, np.sqrt(np.mean((pred - Y) ** 2, axis=1)), np.nan)
        rows_out.append(dict(epoch=epoch, method=name, n=int(good.sum()), rms=rms.tolist(), mode1_mae=float(np.nanmean(err1[name][used])),
                             curve_rms=float(np.sqrt(np.nanmean(errc[name][used] ** 2))), lookbacks=taus, config=C.config_hash()))
    clim = next(r for r in rows_out if r["method"] == "climatology")
    print(f"    {'method':<16}{'n':>5}" + "".join(f"{f'{tau:g} Gyr':>9}" for tau in taus) + f"{'curve':>9}{'skill':>8}{'mode-1 MAE':>12}{'skill':>8}")
    for r in rows_out:
        print(f"    {r['method']:<16}{r['n']:>5}" + "".join(f"{v:>9.4f}" for v in r["rms"]) + f"{r['curve_rms']:>9.4f}"
              f"{1 - (r['curve_rms'] / clim['curve_rms']) ** 2:>8.3f}{r['mode1_mae']:>12.4f}{1 - r['mode1_mae'] / clim['mode1_mae']:>8.3f}")

    def paired(a, b, label):
        good = used & np.isfinite(err1[a]) & np.isfinite(err1[b])
        ok1, boot1 = S.significant_gain(err1[a][good], err1[b][good], delta)
        okc, bootc = S.significant_gain(errc[a][good], errc[b][good], delta)
        print(f"    {label}: first component {100 * -boot1['rel']:+.1f}% [{100 * -boot1['rel_hi']:+.1f}, {100 * -boot1['rel_lo']:+.1f}] "
              f"{'SIGNIFICANT' if ok1 else 'not significant'}; whole curve {100 * -bootc['rel']:+.1f}% "
              f"[{100 * -bootc['rel_hi']:+.1f}, {100 * -bootc['rel_lo']:+.1f}] {'SIGNIFICANT' if okc else 'not significant'}")
        return dict(a=a, b=b, mode1_gain=-boot1["rel"], mode1_significant=ok1, curve_gain=-bootc["rel"], curve_significant=okc)
    comps = [paired("mediated", "climatology", "the mass estimate alone over nothing"),
             paired("ridge", "mediated", "the PROFILE over its own mass estimate (ridge)"),
             paired("gbm-modes", "mediated", "the PROFILE over its own mass estimate (boosting)"),
             paired("gbm-modes", "ridge", "boosting over ridge"),
             paired("oracle-mh", "climatology", "the true mass over nothing"),
             paired("cog+oracle", "oracle-mh", "ORACLE: profile + true mass over the true mass"),
             paired("cog+oracle-gbm", "cog+oracle", "ORACLE: boosting over ridge"),
             paired("diffmah", "ridge", "the DiffMAH-parameter route against the direct route")]
    if "multi-epoch" in preds:
        comps.append(paired("multi-epoch", "ridge", "the stellar HISTORY over the single-epoch profile"))
    stop_linear = not (comps[1]["mode1_significant"] or comps[2]["mode1_significant"])
    print(f"    RULE: the profile's gain over its own mass estimate on the first component is "
          f"{'not significant for ridge or boosting -> STOP at the linear rung, report the null' if stop_linear else 'significant -> the history carries profile information beyond the mass'}")
    if not smoke:
        with open(MAH_BOARD, "a") as fh:
            for r in rows_out:
                fh.write(json.dumps(r) + "\n")
        np.savez(C.OUTDIR / f"mah_epoch{epoch}.npz", Y=Y, fold=fold, index=s.index, lookbacks=np.array(taus), mh=mh,
                 comparisons=json.dumps(comps), **{f"pred_{k.replace('-', '_').replace('+', '_')}": v for k, v in preds.items()})
    return dict(epoch=epoch, stop_linear=stop_linear, comparisons=comps)


def main(epochs=(0,), smoke=False):
    print(f"{RULE}\nexp87 STAGE 5 — the accretion history from the curve of growth{' (SMOKE)' if smoke else ''} (config {C.config_hash()})\n{RULE}")
    out = [run_epoch(k, smoke=smoke) for k in epochs]
    if not smoke:
        (C.OUTDIR / "stage5_mah.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    a = sys.argv
    eps = tuple(int(v) for v in a[a.index("--epochs") + 1:] if v.isdigit()) if "--epochs" in a else (0,)
    sys.exit(main(epochs=eps, smoke="--smoke" in a))
