"""exp81 — the evidence-based rethink, measurement 1: the baseline's ERROR
BUDGET and the HALO-PREDICTABILITY CEILING of what is left.

For the adopted baseline (and, for comparison, Stage 0 C's frozen q_e point
and the loss's q_e basin from exp80) on the fitting sample, per epoch and
radius:

  1. the residual r = log10(model / truth) of M*(<R): its median (the
     SYSTEMATIC part), its 16-84 half-width (the per-galaxy part), the rms,
     and the share of the mean-square residual that is systematic;
     the same for the amplitude-pinned residual (the model rescaled to the
     truth's M*(<100 kpc): the SHAPE part).
  2. THE CEILING: how much of the per-galaxy residual ANY function of the
     halo history could recover. A gradient-boosted regressor (5-fold,
     out-of-fold predictions only) from halo-history features to r, with two
     feature sets: the history BEFORE the epoch only (the honest input), and
     the FULL history including the future and the final mass (the user's
     rule allows it). The out-of-fold R^2 of r, and the rms of r after the
     correction, is the floor no deterministic mean model of the halo can
     go below at that epoch and radius. The same regressor on the TRUTH
     itself gives the halo-only ceiling of the whole problem, next to the
     baseline's own R^2.
  3. THE CENTRE, split by decline: the median residual at 4.9 kpc for
     galaxies whose true M*(<4.9) fell between z = 2 and 0.4 and the rest
     (C8), for the three models — does a post-deposition expansion (q_e)
     emulate the decline?

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=4 PYTHONPATH=. uv run python -u \\
     experiments/exp81_evidence_rethink/residual_budget.py [--smoke]
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp73_size_relative",
          ROOT / "experiments/exp74_c19_history_leak",
          ROOT / "experiments/exp78_size_aware_objective",
          ROOT / "experiments/exp80_deposit_size_law", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import size_law as SL                                    # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S0T = _by_path("exp78_stage0_terms", ROOT / "experiments/exp78_size_aware_objective/stage0_terms.py")
RB = S0T.RB
E80 = ROOT / "experiments/exp80_deposit_size_law/outputs"
ANCHOR_Z = list(E.ANCHOR_Z)
EPOCHS = (0, 1, 2, 3, 4)
R_SHOW = (2.0, 4.92, 10.25, 32.58, 103.45, 148.22)
OUTDIR = HERE / "outputs"
RULE = "=" * 110
N_FOLD = 5


def history_features(curves, k, full):
    """(n, p) features of the measured halo history: log M at 12 log-spaced
    times from 1 Gyr to the epoch (full=False) or to z = 0.4 (full=True),
    plus the growth rate at the epoch and the peak/final mass."""
    t_end = E.T_ANCHOR[0] if full else E.T_ANCHOR[k]
    lt = np.log10(np.geomspace(1.0, t_end, 12))
    lt_k = np.log10(E.T_ANCHOR[k])
    X = []
    for hc in curves:
        lm = E.log_mah(lt, hc)
        dlm = E.dm_dlnt(np.array([lt_k]), hc)[0] / 10.0 ** E.log_mah(np.array([lt_k]), hc)[0]
        X.append(np.r_[lm, np.log10(max(dlm, 1e-3)), E.log_mah(np.array([lt_k]), hc)[0]])
    return np.array(X)


def oof_r2(y, X, seed=0):
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    pred = np.empty(len(y))
    for tr, te in KFold(N_FOLD, shuffle=True, random_state=seed).split(X):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    res = y - pred
    return 1.0 - res.var() / y.var(), float(np.std(res)), float(np.std(y))


def main(smoke=False):
    t0 = time.time()
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, pr, Rm, truth_m, n_inner, rows_m = S0T.build(smoke)
    spec_b, th_b = RB.adopted_baseline()
    rows = pr.all_rows
    cv = [meas[i] for i in rows]
    d0 = np.load(E80 / "stage0_cand_a_expand_free.npz", allow_pickle=True)
    vals = dict(zip([str(n) for n in d0["free"]], np.asarray(d0["x_tuned"], float)))
    th_f = th_b.copy()
    for n in ("log_f_e", "b_e"):
        th_f[spec2.index(n)] = vals[n]
    fq = np.load(E80 / "stage1_fit_start_cont_tuned.npz", allow_pickle=True)
    thq = np.asarray(fq["theta"], float)
    models = {"baseline": (th_b, SL.LAW_DEFAULT),
              "frozen q_e point": (th_f, SL.with_law(q_e=float(vals["q_e"]))),
              "loss's q_e basin": (thq[:12], SL.with_law(q_e=float(thq[12])))}
    pred = {lab: SL.predict_law(spec2, th, law, cv, F.R_GRID, epochs=EPOCHS) for lab, (th, law) in models.items()}
    truth = data[rows]
    ir = [int(np.argmin(np.abs(F.R_GRID - r))) for r in R_SHOW]
    good = np.isfinite(truth).all(axis=(1, 2)) & (truth > 0).all(axis=(1, 2))
    print(f"{RULE}\nexp81 measurement 1 — the baseline's error budget and the halo-predictability ceiling ({int(good.sum())} galaxies)\n{RULE}")

    # ---- 1. the budget ------------------------------------------------------ #
    print(f"\n1. ERROR BUDGET of log10(model/truth) per epoch and radius: median (systematic) | 16-84 half-width (per galaxy) | "
          f"rms | systematic share of the mean square;  then the AMPLITUDE-PINNED (shape) residual's median | half-width")
    budget = {}
    for lab in models:
        m = pred[lab][good]; t = truth[good]
        r = np.log10(np.clip(m, 1, None)) - np.log10(t)
        pin = np.log10(np.clip(m / m[:, :, F.I100][:, :, None] * t[:, :, F.I100][:, :, None], 1, None)) - np.log10(t)
        print(f"\n  --- {lab} ---")
        print(f"  {'epoch':>6}" + "".join(f"{f'R={F.R_GRID[i]:.0f}':>30}" for i in ir))
        for k in EPOCHS:
            cells = []
            for i in ir:
                x, y = r[:, k, i], pin[:, k, i]
                med, hw = np.median(x), 0.5 * (np.percentile(x, 84) - np.percentile(x, 16))
                rms = np.sqrt(np.mean(x ** 2)); share = med ** 2 / rms ** 2
                cells.append(f"{med:+.3f} |{hw:.3f} |{rms:.3f} |{100 * share:3.0f}%  {np.median(y):+.3f} |{0.5 * (np.percentile(y, 84) - np.percentile(y, 16)):.3f}")
                budget[(lab, k, i)] = (med, hw, rms, share, np.median(y))
            print(f"  {ANCHOR_Z[k]:>6}" + "".join(f"{c:>30}" for c in cells))

    # ---- 2. the ceiling ----------------------------------------------------- #
    print(f"\n2. THE HALO-PREDICTABILITY CEILING (5-fold out-of-fold gradient boosting; features = the measured history's "
          f"log M at 12 times + growth rate + mass at the epoch): out-of-fold R^2 of the baseline's residual r from the "
          f"PAST-only history | the FULL history (incl. the future); rms of r before -> after the full-history correction; "
          f"then the truth's own log M*(<R): R^2 from the full history vs the baseline's R^2")
    r_b = np.log10(np.clip(pred["baseline"][good], 1, None)) - np.log10(truth[good])
    lt_truth = np.log10(truth[good])
    cvg = [c for c, g in zip(cv, good) if g]
    ceil = {}
    for k in EPOCHS:
        Xp, Xf = history_features(cvg, k, False), history_features(cvg, k, True)
        cells = []
        for i in ir:
            r2p, _, _ = oof_r2(r_b[:, k, i], Xp)
            r2f, s_after, s_before = oof_r2(r_b[:, k, i], Xf)
            r2t, st_after, st_before = oof_r2(lt_truth[:, k, i], Xf)
            r2_model = 1.0 - np.var(r_b[:, k, i]) / np.var(lt_truth[:, k, i])
            ceil[(k, i)] = (r2p, r2f, s_before, s_after, r2t, r2_model)
            cells.append(f"{r2p:+.2f} |{r2f:+.2f}  {s_before:.3f}->{s_after:.3f}   truth {r2t:.2f} vs model {r2_model:.2f}")
        print(f"  z={ANCHOR_Z[k]:<4}" + "".join(f"{c:>46}" for c in cells))
        print(f"  {'':<6}" + "".join(f"{f'R={F.R_GRID[i]:.0f}':>46}" for i in ir) if k == 0 else "", end="\n" if k == 0 else "")

    # ---- 3. the centre by decline ------------------------------------------ #
    I3 = int(np.argmin(np.abs(F.R_GRID - 4.92)))
    t = truth[good]
    dec = t[:, 4, I3] > t[:, 0, I3]
    print(f"\n3. THE CENTRE (M*(<4.9 kpc)), median log10(model/truth) per epoch, split by whether the TRUE central mass "
          f"DECLINED between z = 2 and 0.4 ({100 * dec.mean():.0f}% did):")
    print(f"  {'model':<18}{'subset':<20}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'span':>8}")
    for lab in models:
        r = np.log10(np.clip(pred[lab][good][:, :, I3], 1, None)) - np.log10(t[:, :, I3])
        for nm, sel in (("all", np.ones(len(r), bool)), ("centre declined", dec), ("did not decline", ~dec)):
            meds = [np.median(r[sel, k]) for k in EPOCHS]
            print(f"  {lab if nm == 'all' else '':<18}{nm:<20}" + "".join(f"{v:>+9.3f}" for v in meds) + f"{np.ptp(meds):>8.3f}")
    # the true decline itself and each model's own decline
    print(f"  the true M*(<4.9) change z=2 -> 0.4 [dex], median: all {np.median(np.log10(t[:, 0, I3] / t[:, 4, I3])):+.3f}, "
          f"decliners {np.median(np.log10(t[dec, 0, I3] / t[dec, 4, I3])):+.3f}; the models': "
          + ", ".join(f"{lab} {np.median(np.log10(pred[lab][good][dec, 0, I3] / pred[lab][good][dec, 4, I3])):+.3f}" for lab in models)
          + " (decliners)")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTDIR / f"residual_budget{'_smoke' if smoke else ''}.npz",
             budget_keys=np.array([f"{a}|{k}|{i}" for (a, k, i) in budget]), budget=np.array(list(budget.values())),
             ceil_keys=np.array([f"{k}|{i}" for (k, i) in ceil]), ceil=np.array(list(ceil.values())), r_show=np.array(R_SHOW))
    print(f"\n  {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main("--smoke" in sys.argv)
