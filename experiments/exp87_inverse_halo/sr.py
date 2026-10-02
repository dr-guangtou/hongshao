"""exp87 Stage 4 — symbolic regression (PySR) on the inverse problem, nested in
the development folds and judged in LATENT space.

Why latent space. On a sample cut at y >= c the conditional mean of y bends
like a hockey stick near the cut. A symbolic search fitted to y itself would
spend its complexity rediscovering that bend, which is the selection function,
not physics. So inside each outer fold the baseline (the truncated-likelihood
ridge on the 24 points) is fitted first, its latent (m, s) give the
pseudo-latent response y* = y - s lambda((c - m)/s) whose conditional mean is
the latent mean, and the search runs on that:

  residual  the target is y* - m_baseline(x): what the linear model missed
  direct    the target is y* itself: can a formula replace the linear model?

The complexity is chosen inside the fold (the search sees 75% of the training
galaxies; the Pareto front's member with the lowest error on the other 25% is
kept). The kept formula then enters a truncated-likelihood head with the
baseline (residual) or alone (direct), and the held-out fold is scored like
every other method. One checkpoint per (configuration, fold).

Operator stages (`config.SR_STAGES`): S1 + - * square; S2 adds / sqrt log exp;
S3 adds pow, tanh, max, min. Inputs: `summaries` (seven positive, readable
numbers: four log masses and three radii in kpc), `pca4` (fold-internal
principal components) and `raw24` (the 24 log masses).

Acceptance (the plan): the out-of-fold CRPS beats the baseline's by delta with
a paired interval excluding zero, AND the formula recurs — the same skeleton
in at least 4 of 5 folds, or every pair of fold formulas correlating above
0.98 on the development galaxies.

Run one configuration (detached):
    PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/sr.py --stage S1 --mode residual --inputs summaries \\
        [--target mh] [--sample parent --epoch 0 --population parent] [--smoke]
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import data as D                                         # noqa: E402
import harness as HN                                     # noqa: E402
import heads as H                                        # noqa: E402
import methods as M                                      # noqa: E402
import scoring as S                                      # noqa: E402

SR_DIR = C.OUTDIR / "sr"
NESTING = {"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0}, "square": {"square": 0}, "sqrt": {"sqrt": 0},
           "tanh": {"tanh": 0, "exp": 0}}


def sr_inputs(sample, name):
    """(design, variable names) of one input set, before any fold-internal step."""
    x, radii = sample.logcog, sample.radii
    if name == "summaries":
        cols = [x[:, -1], D.aperture_logmass(x, radii, 0.0, 10.0), D.aperture_logmass(x, radii, 0.0, 30.0),
                D.aperture_logmass(x, radii, 50.0, 100.0), *(10.0 ** D.sizes(x, radii)).T]
        return np.column_stack(cols), ["m148", "m10", "m30", "m50_100", "r20", "r50", "r80"]
    if name == "shell_summaries":                       # the differential profile in seven readable numbers
        edges = [(0.0, 10.0), (10.0, 30.0), (30.0, 50.0), (50.0, 100.0), (100.0, float(radii[-2])), (float(radii[-2]), None)]
        cols = [x[:, -1]] + [D.aperture_logmass(x, radii, lo, hi) for lo, hi in edges]
        return np.column_stack(cols), ["m148", "s0_10", "s10_30", "s30_50", "s50_100", "s100_132", "s132_148"]
    if name == "raw24":
        return x.copy(), [f"m{int(round(r))}" if i not in (0, 1, 2) else f"m{r:.1f}".replace(".", "p") for i, r in enumerate(radii)]
    if name == "pca4":
        return x.copy(), ["p1", "p2", "p3", "p4"]
    raise KeyError(name)


def new_regressor(stage, seed, timeout):
    from pysr import PySRRegressor
    cfg = C.SR_STAGES[stage]
    nesting = {k: {kk: vv for kk, vv in v.items() if kk in cfg["unary"]} for k, v in NESTING.items() if k in cfg["unary"]}
    kwargs = dict(niterations=10 ** 6, timeout_in_seconds=timeout, maxsize=cfg["maxsize"], binary_operators=cfg["binary"],
                  unary_operators=cfg["unary"], nested_constraints=nesting, model_selection="best", elementwise_loss="L2DistLoss()",
                  progress=False, verbosity=0, deterministic=True, parallelism="serial", random_state=seed,
                  output_directory=tempfile.mkdtemp(prefix="pysr_exp87_"))
    if "pow" in cfg["binary"]:
        kwargs["constraints"] = {"pow": (-1, 1)}
    return PySRRegressor(**kwargs)


def skeleton(expression):
    """The formula's shape: every numerical constant replaced by one symbol."""
    import sympy as sp
    const = sp.Symbol("c")
    return sp.srepr(expression.xreplace({a: const for a in expression.atoms(sp.Float, sp.Integer, sp.Rational)}))


def fold_search(stage, timeout, X_search, t_search, X_valid, t_valid, names, seed):
    """Run the search on the 75% part; keep the Pareto member with the lowest
    error on the 25% part. Returns (model, index of the kept equation, table)."""
    model = new_regressor(stage, seed, timeout)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_search, t_search, variable_names=names)
        eqs = model.equations_
        valid = []
        for i in range(len(eqs)):
            pred = np.asarray(model.predict(X_valid, index=i), float)
            valid.append(float(np.mean((pred - t_valid) ** 2)) if np.isfinite(pred).all() else np.inf)
    keep = int(np.argmin(valid))
    table = [dict(complexity=int(eqs["complexity"].iloc[i]), loss=float(eqs["loss"].iloc[i]), valid=valid[i],
                  equation=str(eqs["equation"].iloc[i])) for i in range(len(eqs))]
    return model, keep, table


def run(stage, mode, inputs, spec, smoke=False, force=False, baseline="raw24"):
    """`baseline`: the feature set of the linear model the search is read against
    (and whose residual it fits in residual mode): "raw24" (the plan's) or
    "shells24" (the differential profile, the better linear model)."""
    timeout = 12 if smoke else C.SR_STAGES[stage]["timeout"]
    base_spec = dict(spec, feature=baseline, extra=None)
    s, rows, cut, X24, _, y, fold = HN.cell_setup(base_spec)
    Xin, names = sr_inputs(s, inputs)
    if smoke:
        keep_rows = np.flatnonzero(fold >= 0)[::max(1, int((fold >= 0).sum() // 500))]
        thin = np.full(len(fold), -1)
        thin[keep_rows] = fold[keep_rows]
        fold = thin
    used = fold >= 0
    tag = f"{spec['sample']}__{spec['epoch']}__{spec['population']}__{spec['target']}__{inputs}__sr-{stage}-{mode}" \
        + ("" if baseline == "raw24" else f"-vs-{baseline}") + ("__smoke" if smoke else "")
    SR_DIR.mkdir(parents=True, exist_ok=True)
    per = {k: np.full(len(y), np.nan) for k in HN.PER_KEYS}
    base_crps = np.full(len(y), np.nan)
    g_all = np.full((C.N_FOLDS, len(y)), np.nan)
    infos, t_start = [], time.time()
    tw = C.TW_THRESHOLD_LOGMH if spec["target"] == "mh" else None
    grid = S.target_grid("mh", cut) if tw is not None else None
    rng = np.random.default_rng(C.SEED)
    for f in range(C.N_FOLDS):
        tr, te = used & (fold != f), fold == f
        ckpt = SR_DIR / f"{tag}__fold{f}.npz"
        base_model = M.DirectLinear("ridge", "L1", ridge_grid=M.RIDGE_GRID)
        base_pred = base_model.fit_predict(X24[tr], y[tr], X24[te], cut)
        base_crps[te] = base_pred.crps(y[te])
        D_tr, _ = base_model._design(X24[tr], X24[tr], y[tr])
        Ds_tr = base_model._scale_design(D_tr, D_tr, y[tr])[0]
        m_tr, s_tr = base_model.model.latent(D_tr, Ds_tr)
        m_te = base_pred.m
        pseudo = H.mills_targets(y[tr], m_tr, s_tr, cut)
        if inputs == "pca4":
            scaler = StandardScaler().fit(Xin[tr])
            pca = PCA(4, random_state=C.SEED).fit(scaler.transform(Xin[tr]))
            Z = pca.transform(scaler.transform(Xin))
        else:
            Z = Xin
        offset = 0.0 if mode == "residual" else float(np.mean(pseudo))
        target = (pseudo - m_tr) if mode == "residual" else (pseudo - offset)
        if ckpt.exists() and not force:
            z = np.load(ckpt, allow_pickle=True)
            g_dev, info = z["g_dev"], json.loads(str(z["info"]))
        else:
            order = rng.permutation(int(tr.sum()))
            n_valid = len(order) // 4
            search, valid = order[n_valid:], order[:n_valid]
            Ztr = Z[tr]
            t0 = time.time()
            model, keep, table = fold_search(stage, timeout, Ztr[search], target[search], Ztr[valid], target[valid], names, C.SEED + f)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g_dev = np.asarray(model.predict(Z, index=keep), float)
                expr = model.sympy(index=keep)
            info = dict(fold=f, equation=table[keep]["equation"], complexity=table[keep]["complexity"], valid_mse=table[keep]["valid"],
                        null_mse=float(np.mean((target[valid] - np.mean(target[search])) ** 2)), skeleton=skeleton(expr),
                        seconds=round(time.time() - t0, 1), pareto=table)
            np.savez(ckpt, g_dev=g_dev, info=json.dumps(info))
        g_dev = np.where(np.isfinite(g_dev), g_dev, 0.0)
        g_all[f] = g_dev
        if mode == "residual":
            head = H.TruncLinear(cut).fit(np.column_stack([m_tr, g_dev[tr]]), y[tr], m_tr[:, None])
            pred = head.predict(np.column_stack([m_te, g_dev[te]]), m_te[:, None])
        else:
            head = H.CommonHead(cut).fit(g_dev[tr] + offset, y[tr])
            pred = head.predict(g_dev[te] + offset)
        per["crps"][te], per["logscore"][te], per["pit"][te] = pred.crps(y[te]), pred.logscore(y[te]), pred.pit(y[te])
        mean, median = pred.mean(), pred.median()
        per["pred_mean"][te], per["pred_median"][te] = mean, median
        per["err_mean"][te], per["err_median"][te] = mean - y[te], median - y[te]
        per["latent_m"][te], per["latent_s"][te] = pred.m, pred.s
        if tw is not None:
            per["twcrps"][te] = pred.twcrps(y[te], tw, grid)
        infos.append(info)
        print(f"    fold {f}: complexity {info['complexity']:>2}, validation MSE {info['valid_mse']:.5f} (constant {info['null_mse']:.5f}); "
              f"{info['equation'][:110]}  ({info['seconds']:.0f} s)", flush=True)
    skeletons = [i["skeleton"] for i in infos]
    recurrence = max(skeletons.count(k) for k in set(skeletons))
    corr = np.corrcoef(g_all[:, used])
    min_corr = float(np.nanmin(corr)) if np.isfinite(corr).all() else float("nan")
    delta = float(json.loads((C.OUTDIR / "gate_mechanics.json").read_text())["delta"])
    gain_ok, boot = S.significant_gain(per["crps"][used], base_crps[used], delta)
    recurs = bool(recurrence >= C.SR_MIN_FOLD_RECURRENCE or (np.isfinite(min_corr) and min_corr > 0.98))
    verdict = dict(stage=stage, mode=mode, inputs=inputs, baseline=baseline, gain=-boot["rel"], gain_lo=-boot["rel_hi"], gain_hi=-boot["rel_lo"],
                   significant=gain_ok, recurrence=int(recurrence), min_fold_correlation=min_corr, recurs=recurs,
                   accepted=bool(gain_ok and recurs), baseline_crps=float(np.mean(base_crps[used])),
                   equations=[i["equation"] for i in infos], complexities=[i["complexity"] for i in infos])
    cell_spec = dict(spec, feature=inputs, extra="none", method=f"sr-{stage}-{mode}" + ("" if baseline == "raw24" else f"-vs-{baseline}"))
    row = HN.write_cell(cell_spec, tag, "L4", per, y, fold, s.index, cut, Z.shape[1], [verdict], time.time() - t_start, smoke=smoke)
    verdict["cell"], verdict["crps"] = tag, row["crps"]
    (SR_DIR / f"{tag}__verdict.json").write_text(json.dumps(dict(verdict, row=row), indent=1))
    print(f"  SR {stage} {mode} on {inputs}: CRPS {row['crps']:.4f} vs the baseline's {verdict['baseline_crps']:.4f}: gain {100 * verdict['gain']:+.2f}% "
          f"[{100 * verdict['gain_lo']:+.2f}, {100 * verdict['gain_hi']:+.2f}] {'significant' if gain_ok else 'not significant'}; the same skeleton in "
          f"{recurrence} of {C.N_FOLDS} folds, min fold correlation {min_corr:.3f} -> {'ACCEPTED' if verdict['accepted'] else 'not accepted'}", flush=True)
    return verdict


if __name__ == "__main__":
    a = sys.argv

    def arg(name, default):
        return a[a.index(name) + 1] if name in a else default
    spec = dict(sample=arg("--sample", "parent"), epoch=int(arg("--epoch", 0)), population=arg("--population", "parent"),
                target=arg("--target", "mh"))
    run(arg("--stage", "S1"), arg("--mode", "residual"), arg("--inputs", "summaries"), spec, smoke="--smoke" in a, force="--force" in a,
        baseline=arg("--baseline", "raw24"))
