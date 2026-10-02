"""exp87 Stage 8 (added 2026-10-02 at the user's request) — EXPLORATORY symbolic
regression: which formulas turn the z = 0.4 stellar mass distribution into the
halo's mass at z = 0.4, 0.7, 1.0, 1.5 and 2.0?

The robustness rule of Stage 4 (a formula must recur across folds) is DROPPED
on purpose: the aim here is to explore functional forms, with longer searches
and a richer operator set. What is kept is an honest score.

Two ways of writing the stellar mass distribution (always the z = 0.4 one):

  approach 1  the PROFILE: the stellar mass inside a central aperture and in a
              series of annuli (`annuli6`: six broad bins; `annuli9`: nine bins
              on the grid's own radii)
  approach 2  a PARAMETRISED curve of growth (`cogparams.py`): a Sersic fit, a
              logistic in log R (`hill`), an inner Sersic + outer exponential
              (`double`), a cubic in log R (`logpoly`), and mass + three
              enclosed-mass radii (`sizes`)

Units: every stellar mass is log10(M / 10^10 Msun), every radius log10(R / kpc),
the target is h = log10(M200c / 10^13 Msun).

Targets: z = 0.4 is the complete parent (truncated at 10^13: the search runs on
the pseudo-latent response, as in Stage 4, and the score is the truncated
CRPS); z >= 0.7 is the main progenitor's mass on the curated sample (no cut,
plain normal), exactly Stage 7's populations.

One search per (input set, epoch), on the development folds only:
  folds 0-2  the search (PySR, multithreaded, `EXPLORE` below)
  fold 3     ranks the Pareto front (the formulas as found)
  fold 4     scores them as found: a number no choice was made on
and, to compare with Stage 7's tables, the "refit" score: the formula's
SHAPE is frozen, its constants are refitted on four folds and scored on the
fifth, over all five folds (the shape itself was chosen on folds 0-3, so this
number is mildly optimistic; the fold-4 number is the strict one).
The lockbox is not touched.

Run:
    PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/stage8_explore.py bench
    PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/stage8_explore.py search --part 0/2 [--smoke]
    PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/stage8_explore.py annulus
    PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/stage8_explore.py report
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cogparams as CP                                   # noqa: E402
import config as C                                       # noqa: E402
import data as D                                         # noqa: E402
import harness as HN                                     # noqa: E402
import heads as H                                        # noqa: E402
import methods as M                                      # noqa: E402

OUT = C.OUTDIR / "stage8"
RULE = "=" * 100
SEARCH_FOLDS, SELECT_FOLD, TEST_FOLD = (0, 1, 2), 3, 4
MASS_UNIT, HALO_UNIT = 10.0, 13.0
THREADS = 4
EXPLORE = dict(binary=["+", "-", "*", "/", "pow", "max", "min"], unary=["square", "cube", "sqrt", "log", "exp", "tanh"],
               maxsize=45, timeout=900)
NESTING = {"exp": {"exp": 0, "log": 0}, "log": {"log": 0, "exp": 1}, "square": {"square": 0, "cube": 0},
           "cube": {"square": 0, "cube": 0}, "sqrt": {"sqrt": 0}, "tanh": {"tanh": 0, "exp": 0}}
APPROACH = {"annuli6": 1, "annuli9": 1, "sersic": 2, "hill": 2, "double": 2, "logpoly": 2, "sizes": 2}
#: two formulas are "the same" when their predictions differ by less than this fraction of the residual scatter
DISTINCT_FRACTION = 0.25
N_BEST = 3
#: a formula whose error exceeds this many times that of a straight line in M*(<148 kpc) has a pole inside the data and is not ranked
UNSTABLE_FACTOR = 1.1
#: "near-best": within this factor (rms on the ranking fold) of the best formula
NEAR_BEST = 1.02
#: the convergence check: the same search run four times longer (`search --deep`)
DEEP_TIMEOUT = 3600
#: a numerical literal of a PySR equation string (not the digits inside a variable name)
NUMBER = re.compile(r"(?<![\w.])(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
FUNCTIONS = dict(square=np.square, cube=lambda v: v ** 3, sqrt=np.sqrt, log=np.log, exp=np.exp, tanh=np.tanh,
                 max=np.maximum, min=np.minimum)


# --------------------------------------------------------------------------- #
# the inputs and the targets                                                    #
# --------------------------------------------------------------------------- #
def input_set(sample, name):
    """(n, p) inputs read from the z = 0.4 curve of growth, and the variable names."""
    x, radii = sample.logcog, sample.radii
    if name in ("annuli6", "annuli9"):
        edges = [0.0, 10.0, 30.0, 50.0, 100.0, radii[-2], radii[-1]] if name == "annuli6" else [0.0, *radii[[3, 6, 9, 12, 15, 18, 20, 22, 23]]]
        cols = [D.aperture_logmass(x, radii, float(lo), float(hi)) - MASS_UNIT for lo, hi in zip(edges[:-1], edges[1:])]
        names = [f"c{hi:.0f}" if lo == 0.0 else f"a{lo:.0f}_{hi:.0f}" for lo, hi in zip(edges[:-1], edges[1:])]
        return np.column_stack(cols), names
    table, _ = CP.table_for(sample.index, name)
    table = table.copy()
    table[:, 0] -= MASS_UNIT
    names = {"sersic": ["m_tot", "lr_e", "n_ser"], "hill": ["m_inf", "lr_h", "a_hill"], "double": ["m_tot", "f_out", "lr_in", "n_in", "lr_out"],
             "logpoly": ["m20", "a1", "a2", "a3"], "sizes": ["m148", "lr20", "lr50", "lr80"]}[name]
    return table, names


def load_epoch(k):
    """(z = 0.4 sample, target halo mass at epoch k, the cut on it, folds)."""
    if k == 0:
        s = D.load_parent(verbose=False)
        y, cut = s.targets["mh"], C.PARENT_CUT
        ok = np.isfinite(y)
    else:
        s = D.load_curated(0, verbose=False)
        sk = D.load_curated(k, verbose=False)
        assert np.array_equal(sk.index, s.index)
        y, cut = sk.targets["mh"], None
        ok = np.isfinite(y) & np.isfinite(s.targets["mh"])
    return s, y, cut, np.where(ok & ~s.lockbox, s.fold, -1)


def latent_targets(X24, y, train, other, cut):
    """The pseudo-latent response (Stage 4's device) from the truncated ridge on
    the 24 shell masses fitted on `train`: for the `train` rows and the `other`
    rows. Without a cut it is the target itself."""
    if cut is None:
        return y[train], y[other]
    base = M.DirectLinear("ridge", "L1", ridge_grid=M.RIDGE_GRID)
    pred = base.fit_predict(X24[train], y[train], X24[other], cut)
    D_tr, _ = base._design(X24[train], X24[train], y[train])
    m_tr, s_tr = base.model.latent(D_tr, base._scale_design(D_tr, D_tr, y[train])[0])
    return H.mills_targets(y[train], m_tr, s_tr, cut), H.mills_targets(y[other], pred.m, pred.s, cut)


# --------------------------------------------------------------------------- #
# a formula as a function with free constants                                   #
# --------------------------------------------------------------------------- #
def parametrise(equation, names):
    """A PySR equation string as (template with {i} slots, callable(columns...,
    constants...), the constants as found). Every numerical literal is a free
    constant (PySR optimises each of them)."""
    constants = []

    def slot(match):
        constants.append(float(match.group(0)))
        return f"{{{len(constants) - 1}}}"
    template = NUMBER.sub(slot, equation)
    counter = iter(range(len(constants)))
    code = compile(NUMBER.sub(lambda _match: f"k_[{next(counter)}]", equation).replace("^", "**"), "<formula>", "eval")
    scope = dict(FUNCTIONS, __builtins__={})

    def fn(*args):
        return eval(code, scope, dict(zip(names, args[:len(names)]), k_=args[len(names):]))
    return template, fn, np.array(constants)


def evaluate(fn, X, constants):
    with np.errstate(all="ignore"):
        out = np.asarray(fn(*X.T, *constants), float)
    return np.broadcast_to(out, (len(X),)).copy()


def refit(fn, constants, X, target):
    """Least-squares constants of a frozen formula, started from those found."""
    if len(constants) == 0:
        return constants

    def residual(c):
        r = evaluate(fn, X, c) - target
        return np.where(np.isfinite(r), r, 10.0)
    start_cost = float(np.sum(residual(constants) ** 2))
    try:
        res = least_squares(residual, constants, x_scale=np.maximum(np.abs(constants), 1e-2), max_nfev=300)
    except ValueError:
        return constants
    return res.x if np.isfinite(res.cost) and 2.0 * res.cost <= start_cost else constants


def formula_text(template, constants, digits=4):
    return template.format(*(f"{float(v):.{digits}g}" for v in constants))


def head_scores(g_train, y_train, g_test, y_test, cut):
    """Per-galaxy (CRPS, error of the predictive mean) of a point prediction g
    read through the common head fitted on the training rows."""
    fill = float(np.nanmean(g_train)) if np.isfinite(g_train).any() else 0.0
    g_train, g_test = np.where(np.isfinite(g_train), g_train, fill), np.where(np.isfinite(g_test), g_test, fill)
    pred = H.CommonHead(cut).fit(g_train, y_train).predict(g_test)
    return pred.crps(y_test), pred.mean() - y_test


# --------------------------------------------------------------------------- #
# benchmarks: the linear models on the same galaxies                            #
# --------------------------------------------------------------------------- #
def quad(k):
    return M.DirectLinear("linear+quad", "L2", transform=(lambda: M._LinearPlusQuadratic(k)), ridge_grid=M.RIDGE_GRID)


def ridge():
    return M.DirectLinear("ridge", "L1", ridge_grid=M.RIDGE_GRID)


def benchmarks(force=False):
    """Out-of-fold CRPS and error per galaxy of the reference models, per epoch:
    nothing, M*(<148 kpc), the 24 shell masses (linear, and + quadratic), and
    the linear and linear + quadratic model on each of Stage 8's input sets."""
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{RULE}\nexp87 STAGE 8 — benchmarks on the search's own galaxies (out of fold, CRPS / RMSE in dex)\n{RULE}")
    for k in range(5):
        path = OUT / f"benchmarks__{C.EPOCH_TAG[k]}.npz"
        if path.exists() and not force:
            continue
        s, y, cut, fold = load_epoch(k)
        used = fold >= 0
        shells, _ = D.feature_set(s, "shells24")
        designs = {"nothing": (shells[:, :1], M.Climatology("mh")), "M*(<148)": (s.logcog[:, [-1]], M.scalar_method("line")),
                   "24 shells": (shells, ridge()), "24 shells + quad": (shells, quad(6))}
        for name in APPROACH:
            X, _ = input_set(s, name)
            designs[f"{name} linear"] = (X, ridge())
            designs[f"{name} + quad"] = (X, quad(min(6, X.shape[1])))
        out = dict(index=s.index, fold=fold, y=y)
        print(f"\n--- halo mass at z = {C.ANCHOR_Z[k]} ({'parent, truncated at 13.0' if k == 0 else 'main progenitor, curated'}; {used.sum()} development galaxies) ---")
        print(f"    {'model':<22}{'CRPS':>8}{'RMSE':>8}{'fold-4 CRPS':>13}{'fold-4 RMSE':>13}")
        for name, (X, method) in designs.items():
            per, _ = HN.oof_scores(method, X, y, fold, cut)
            out[f"crps::{name}"], out[f"err::{name}"] = per["crps"], per["err_mean"]
            t = fold == TEST_FOLD
            print(f"    {name:<22}{per['crps'][used].mean():>8.4f}{np.sqrt(np.mean(per['err_mean'][used] ** 2)):>8.4f}"
                  f"{per['crps'][t].mean():>13.4f}{np.sqrt(np.mean(per['err_mean'][t] ** 2)):>13.4f}", flush=True)
        np.savez(path, **out)


# --------------------------------------------------------------------------- #
# one search                                                                    #
# --------------------------------------------------------------------------- #
def read_hall_of_fame(directory):
    """The Pareto front from PySR's own `hall_of_fame.csv` (complexity, loss, equation)."""
    import csv
    path = max(Path(directory).rglob("hall_of_fame.csv"), key=lambda q: q.stat().st_mtime)
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return dict(equations=[r["Equation"] for r in rows], complexity=[int(r["Complexity"]) for r in rows], loss=[float(r["Loss"]) for r in rows],
                pysr_select=[None] * len(rows))


class ExportSkipped(Exception):
    """Raised in place of PySR's sympy export, which runs after the search has saved its hall of fame."""


def _skip_export(*args, **kwargs):
    raise ExportSkipped


def new_regressor(timeout, directory):
    """PySR's export of the formulas to sympy overflows the recursion limit, or
    hangs, on deep `pow` / `max` / `min` forms. The search is finished and saved
    by then, so the export is skipped and the hall of fame is read from its file
    (the formulas are parsed by `parametrise`, checked against PySR's own
    predictions on the first 24 searches)."""
    import pysr.export
    from pysr import PySRRegressor
    pysr.export.pysr2sympy = _skip_export
    return PySRRegressor(niterations=10 ** 7, timeout_in_seconds=timeout, maxsize=EXPLORE["maxsize"], binary_operators=EXPLORE["binary"],
                         unary_operators=EXPLORE["unary"], nested_constraints=NESTING, constraints={"pow": (-1, 1)},
                         model_selection="best", elementwise_loss="L2DistLoss()", progress=False, verbosity=0,
                         parallelism="multithreading", output_directory=directory)


def search(name, k, smoke=False, force=False, deep=False):
    """One search. `deep` is the convergence check: the same search run `DEEP_TIMEOUT` seconds, kept apart."""
    tag = f"{name}__{C.EPOCH_TAG[k]}" + ("__smoke" if smoke else "") + ("__deep" if deep else "")
    path = OUT / f"{tag}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    s, y, cut, fold = load_epoch(k)
    X, names = input_set(s, name)
    X24, _ = D.feature_set(s, "shells24")
    used = fold >= 0
    in_search, in_select, in_test = np.isin(fold, SEARCH_FOLDS), fold == SELECT_FOLD, fold == TEST_FOLD
    t_search, t_select = latent_targets(X24, y, in_search, in_select, cut)
    raw_path = OUT / f"{tag}__pareto.json"
    OUT.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not force:
        raw = json.loads(raw_path.read_text())
    else:
        t0 = time.time()
        directory = tempfile.mkdtemp(prefix="pysr_exp87_s8_")
        model = new_regressor(30 if smoke else DEEP_TIMEOUT if deep else EXPLORE["timeout"], directory)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X[in_search], t_search - HALO_UNIT, variable_names=names)
        except ExportSkipped:
            pass
        raw = read_hall_of_fame(directory)
        raw["seconds"] = round(time.time() - t0, 1)
        raw_path.write_text(json.dumps(raw))
    seconds = raw["seconds"]
    # per-fold training targets for the refit score, and the all-development targets for the final constants
    fold_targets = {f: latent_targets(X24, y, used & (fold != f), fold == f, cut)[0] for f in range(C.N_FOLDS)}
    final_target = latent_targets(X24, y, used, used, cut)[0]
    train_as_found = in_search | in_select
    members = []
    for equation, complexity, loss, theirs in zip(raw["equations"], raw["complexity"], raw["loss"], raw["pysr_select"]):
        template, fn, constants = parametrise(equation, names)
        g = evaluate(fn, X, constants) + HALO_UNIT
        theirs = g[in_select] if theirs is None else np.asarray(theirs, float) + HALO_UNIT
        both = np.isfinite(theirs) & np.isfinite(g[in_select])
        if not (both.any() and np.allclose(theirs[both], g[in_select][both], atol=1e-5)):
            print(f"    WARNING: the parsed formula disagrees with PySR's own prediction and is dropped: {equation}", flush=True)
            continue
        select_mse = float(np.mean((g[in_select] - t_select) ** 2)) if np.isfinite(g[in_select]).all() else float("inf")
        crps_t, err_t = head_scores(g[train_as_found], y[train_as_found], g[in_test], y[in_test], cut)
        crps_cv, err_cv = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for f in range(C.N_FOLDS):
            tr, te = used & (fold != f), fold == f
            c_f = refit(fn, constants, X[tr], fold_targets[f] - HALO_UNIT)
            crps_cv[te], err_cv[te] = head_scores(evaluate(fn, X[tr], c_f) + HALO_UNIT, y[tr], evaluate(fn, X[te], c_f) + HALO_UNIT, y[te], cut)
        c_final = refit(fn, constants, X[used], final_target - HALO_UNIT)
        members.append(dict(
            complexity=complexity, search_mse=loss, select_mse=select_mse,
            test_crps=float(crps_t.mean()), test_rmse=float(np.sqrt(np.mean(err_t ** 2))), test_nan=int((~np.isfinite(g[in_test])).sum()),
            cv_crps=float(crps_cv[used].mean()), cv_rmse=float(np.sqrt(np.mean(err_cv[used] ** 2))), n_constants=len(constants),
            as_found=formula_text(template, constants), refitted=formula_text(template, c_final), skeleton=template,
            prediction=np.round(evaluate(fn, X[used], c_final) + HALO_UNIT, 4).tolist()))
    out = dict(input=name, approach=APPROACH[name], epoch=k, z=C.ANCHOR_Z[k], variables=names, n_search=int(in_search.sum()), n_select=int(in_select.sum()),
               n_test=int(in_test.sum()), n_development=int(used.sum()), cut=cut, seconds=round(seconds, 1), settings=EXPLORE, threads=THREADS,
               config=C.config_hash(), members=members)
    path.write_text(json.dumps(out))
    best = min(members, key=lambda m: m["select_mse"])
    print(f"  [{name:<8} z = {C.ANCHOR_Z[k]}] {len(members)} formulas in {seconds:.0f} s; best on the ranking fold: complexity {best['complexity']}, "
          f"fold-4 CRPS {best['test_crps']:.4f} RMSE {best['test_rmse']:.4f}, refit CRPS {best['cv_crps']:.4f} RMSE {best['cv_rmse']:.4f}\n"
          f"      h = {best['refitted'][:200]}", flush=True)
    return out


def run_list():
    """Every (input set, epoch), ordered so that two jobs each get every input set and epoch."""
    return [(name, k) for k in range(5) for name in APPROACH]


# --------------------------------------------------------------------------- #
# the report                                                                    #
# --------------------------------------------------------------------------- #
def stable(members, reference):
    """The members without a pole inside the data. A formula that divides by
    something that crosses zero can rank first on one fold and fail on another,
    so a member is ranked only if it is finite on every galaxy and BOTH its
    refit error and its as-found fold-4 error stay below `UNSTABLE_FACTOR`
    times those of the plainest model, a straight line in M*(<148 kpc)
    (`reference`). The filter removes blow-ups only: every competitive formula
    is 5 to 20 per cent BELOW that line."""
    return [m for m in members if np.isfinite(m["select_mse"]) and m["test_nan"] == 0 and np.isfinite(m["cv_rmse"])
            and m["cv_rmse"] <= UNSTABLE_FACTOR * reference["cv_rmse"] and m["test_rmse"] <= UNSTABLE_FACTOR * reference["test_rmse"]]


def pick(members, reference, n_best=N_BEST):
    """The `n_best` best stable members on the ranking fold that are different
    functions: a different shape, and predictions that differ from every formula
    already kept by more than `DISTINCT_FRACTION` of the residual scatter."""
    kept = []
    for m in sorted(stable(members, reference), key=lambda m: m["select_mse"]):
        p = np.asarray(m["prediction"])
        same = any(m["skeleton"] == q["skeleton"] and m["input"] == q["input"] for q in kept) or any(
            m["input"] == q["input"] and np.sqrt(np.nanmean((p - np.asarray(q["prediction"])) ** 2)) < DISTINCT_FRACTION * m["cv_rmse"] for q in kept)
        if not same:
            kept.append(m)
        if len(kept) == n_best:
            break
    return kept


def near_best(members, reference, tolerance=NEAR_BEST):
    """The stable formulas whose ranking-fold error is within `tolerance` (rms) of the best."""
    good = stable(members, reference)
    best = min(m["select_mse"] for m in good)
    return [m for m in good if np.sqrt(m["select_mse"]) <= tolerance * np.sqrt(best)]


def compact(members, reference):
    """The simplest formula whose ranking-fold error is within 2 per cent (rms) of the best."""
    return min(near_best(members, reference), key=lambda m: m["complexity"])


def variable_use(members, names, reference):
    """The fraction of the near-best formulas of one search that use each variable."""
    good = near_best(members, reference)
    return {n: float(np.mean([re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])", m["skeleton"]) is not None for m in good])) for n in names}, len(good)


# --------------------------------------------------------------------------- #
# the plain reading the formulas are compared with: one annulus, two annuli     #
# --------------------------------------------------------------------------- #
def single_annulus(force=False):
    """Out-of-fold CRPS of each of `annuli9`'s nine masses ALONE (a straight
    line), of every pair of them (linear), and of the centre + the outermost
    annulus with curvature, per epoch. The best pair is chosen on the same
    out-of-fold scores it is reported with (36 pairs: mildly optimistic)."""
    path = C.OUTDIR / "stage8_single_annulus.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    out = {}
    for k in range(5):
        s, y, cut, fold = load_epoch(k)
        used = fold >= 0
        X, names = input_set(s, "annuli9")

        def score(columns, method):
            per, _ = HN.oof_scores(method, X[:, columns], y, fold, cut)
            return float(per["crps"][used].mean())
        alone = {n: score([j], M.scalar_method("line")) for j, n in enumerate(names)}
        pairs = {f"{names[i]} + {names[j]}": score([i, j], ridge()) for i in range(len(names)) for j in range(i + 1, len(names))}
        ends = [0, len(names) - 1]
        out[str(C.ANCHOR_Z[k])] = dict(alone=alone, pairs=pairs, ends_quad=score(ends, quad(2)))
        best = min(pairs, key=pairs.get)
        print(f"  z = {C.ANCHOR_Z[k]}: best single {min(alone, key=alone.get)} {min(alone.values()):.4f}; best pair {best} {pairs[best]:.4f}; "
              f"{names[0]} + {names[-1]} {pairs[names[0] + ' + ' + names[-1]]:.4f}, with curvature {out[str(C.ANCHOR_Z[k])]['ends_quad']:.4f}", flush=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def report():
    summary = {}
    print(f"{RULE}\nexp87 STAGE 8 — exploratory symbolic regression: the z = 0.4 stellar mass distribution -> halo mass at five epochs\n"
          f"  h = log10(M200c / 10^13 Msun); stellar masses log10(M / 10^10 Msun); radii log10(R / kpc); CRPS and RMSE in dex\n{RULE}")
    for k in range(5):
        bench = np.load(OUT / f"benchmarks__{C.EPOCH_TAG[k]}.npz")
        fold = bench["fold"]
        used, test = fold >= 0, fold == TEST_FOLD

        def bench_row(name):
            return dict(cv_crps=float(bench[f"crps::{name}"][used].mean()), cv_rmse=float(np.sqrt(np.mean(bench[f"err::{name}"][used] ** 2))),
                        test_crps=float(bench[f"crps::{name}"][test].mean()), test_rmse=float(np.sqrt(np.mean(bench[f"err::{name}"][test] ** 2))))
        refs = {n: bench_row(n) for n in ("nothing", "M*(<148)", "24 shells", "24 shells + quad")}
        print(f"\n{RULE}\nHALO MASS AT z = {C.ANCHOR_Z[k]}  ({int(used.sum())} development galaxies, {int(test.sum())} in fold 4)\n{RULE}")
        print(f"    {'reference':<34}{'refit CRPS':>11}{'RMSE':>8}{'fold-4 CRPS':>13}{'RMSE':>8}")
        for n, r in refs.items():
            print(f"    {n:<34}{r['cv_crps']:>11.4f}{r['cv_rmse']:>8.4f}{r['test_crps']:>13.4f}{r['test_rmse']:>8.4f}")
        summary[str(C.ANCHOR_Z[k])] = dict(references=refs, approaches={})
        plain = refs["M*(<148)"]
        for approach in (1, 2):
            members, linear, per_set, usage = [], {}, {}, {}
            for name in (n for n, a in APPROACH.items() if a == approach):
                path = OUT / f"{name}__{C.EPOCH_TAG[k]}.json"
                if not path.exists():
                    continue
                run = json.loads(path.read_text())
                mine = [dict(m, input=name, variables=run["variables"]) for m in run["members"]]
                members += mine
                linear[name] = dict(linear=bench_row(f"{name} linear"), quad=bench_row(f"{name} + quad"))
                per_set[name] = min(stable(mine, plain), key=lambda m: m["select_mse"])
                usage[name] = variable_use(mine, run["variables"], plain)
            if not members:
                continue
            best, small = pick(members, plain), compact(members, plain)
            dropped = len(members) - len(stable(members, plain))
            print(f"\n  APPROACH {approach} ({'the profile: a central aperture and annuli' if approach == 1 else 'a parametrised curve of growth'}); "
                  f"{len(members)} formulas, {dropped} not ranked (a pole in the data, or an error above {UNSTABLE_FACTOR} x the stellar-mass line's)")
            print(f"    {'per input set':<34}{'refit CRPS':>11}{'RMSE':>8}{'fold-4 CRPS':>13}{'RMSE':>8}   / linear + quadratic   | best formula of the set (size): refit, fold-4 CRPS")
            for name, r in linear.items():
                m = per_set[name]
                print(f"    {name + ' linear':<34}{r['linear']['cv_crps']:>11.4f}{r['linear']['cv_rmse']:>8.4f}{r['linear']['test_crps']:>13.4f}"
                      f"{r['linear']['test_rmse']:>8.4f}   / {r['quad']['cv_crps']:.4f} {r['quad']['test_crps']:.4f}        | ({m['complexity']:>2}) {m['cv_crps']:.4f} {m['test_crps']:.4f}")
            for label, m in [*((f"best {i + 1}", m) for i, m in enumerate(best)), ("compact", small)]:
                print(f"    {label + ' [' + m['input'] + ', size ' + str(m['complexity']) + ']':<34}{m['cv_crps']:>11.4f}{m['cv_rmse']:>8.4f}{m['test_crps']:>13.4f}{m['test_rmse']:>8.4f}")
                print(f"        h = {m['refitted']}")
            print("    variables used by the formulas within 2 per cent of the best (fraction of them):")
            for name, (use, count) in usage.items():
                print(f"      {name:<8} ({count:>2} formulas)  " + "  ".join(f"{n} {v:.2f}" for n, v in use.items()))
            strip = lambda m: {key: v for key, v in m.items() if key != "prediction"}          # noqa: E731
            summary[str(C.ANCHOR_Z[k])]["approaches"][str(approach)] = dict(
                linear=linear, best=[strip(m) for m in best], compact=strip(small), per_set={n: strip(m) for n, m in per_set.items()},
                usage={n: dict(use=u, n_formulas=c) for n, (u, c) in usage.items()}, n_formulas=len(members), n_not_ranked=dropped)
    annulus_path = C.OUTDIR / "stage8_single_annulus.json"
    if annulus_path.exists():
        table = json.loads(annulus_path.read_text())
        names = list(next(iter(table.values()))["alone"])
        print(f"\n{RULE}\nTHE PLAIN READING: one annulus alone (a straight line), CRPS in dex, out of fold\n{RULE}")
        print(f"    {'z':<6}" + "".join(f"{n:>10}" for n in names) + "   | best pair (linear)           | centre + outermost, + curvature")
        for z, row in table.items():
            pair = min(row["pairs"], key=row["pairs"].get)
            ends = row["pairs"][f"{names[0]} + {names[-1]}"]
            print(f"    {z:<6}" + "".join(f"{row['alone'][n]:>10.4f}" for n in names) + f"   | {pair:<20} {row['pairs'][pair]:.4f} | {ends:.4f}, {row['ends_quad']:.4f}")
        summary["single_annulus"] = table
    deep_runs = sorted(OUT.glob("*__deep.json"))
    if deep_runs:
        print(f"\n{RULE}\nDOES A LONGER SEARCH HELP? the same search run {DEEP_TIMEOUT} s instead of {EXPLORE['timeout']} s (best stable formula on the ranking fold)\n{RULE}")
        print(f"    {'search':<20}{'seconds':>9}{'size':>6}{'search rms':>12}{'ranking rms':>13}{'refit CRPS':>12}{'fold-4 CRPS':>13}")
        summary["deep"] = {}
        for path in deep_runs:
            long_run, short_run = json.loads(path.read_text()), json.loads((OUT / path.name.replace("__deep", "")).read_text())
            rows = {}
            for run in (short_run, long_run):
                m = min(stable(run["members"], summary[str(run["z"])]["references"]["M*(<148)"]), key=lambda m: m["select_mse"])
                rows[str(int(run["seconds"]))] = {key: m[key] for key in ("complexity", "search_mse", "select_mse", "cv_crps", "test_crps", "refitted")}
                print(f"    {run['input'] + ', z = ' + str(run['z']):<20}{run['seconds']:>9.0f}{m['complexity']:>6}{np.sqrt(m['search_mse']):>12.4f}"
                      f"{np.sqrt(m['select_mse']):>13.4f}{m['cv_crps']:>12.4f}{m['test_crps']:>13.4f}")
            print(f"        the longer search's formula: h = {m['refitted']}")
            summary["deep"][f"{long_run['input']}__z{long_run['z']}"] = rows
    (C.OUTDIR / "stage8_explore.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  wrote {C.OUTDIR / 'stage8_explore.json'}")


if __name__ == "__main__":
    a = sys.argv
    if "bench" in a:
        benchmarks(force="--force" in a)
    elif "annulus" in a:
        single_annulus(force="--force" in a)
    elif "report" in a:
        report()
    elif "search" in a:
        os.environ.setdefault("PYTHON_JULIACALL_THREADS", str(THREADS))
        for i, word in enumerate(a):                     # a search stopped after its hall of fame was saved: `--recover name:k:directory`
            if word == "--recover":
                name, k, directory = a[i + 1].split(":", 2)
                raw = dict(read_hall_of_fame(directory), seconds=float(EXPLORE["timeout"]))
                OUT.mkdir(parents=True, exist_ok=True)
                (OUT / f"{name}__{C.EPOCH_TAG[int(k)]}__pareto.json").write_text(json.dumps(raw))
        part, n_parts = (int(v) for v in a[a.index("--part") + 1].split("/")) if "--part" in a else (0, 1)
        only = a[a.index("--only") + 1].split(",") if "--only" in a else None
        t_start = time.time()
        for j, (name, k) in enumerate(run_list()):
            if j % n_parts == part and (only is None or f"{name}:{k}" in only):
                search(name, k, smoke="--smoke" in a, force="--force" in a, deep="--deep" in a)
        print(f"STAGE8 SEARCH part {part}/{n_parts} DONE in {time.time() - t_start:.0f} s")
