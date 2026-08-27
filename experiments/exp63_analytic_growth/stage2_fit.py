"""exp63 Stage 2 — the two-channel mean, fitted at z=0.4 ONLY, predicted at
z=0.7-2.0.

The model is `model2.py` (compact Sersic channel + extended gompertz_log
channel, the compact share a logistic in the halo mass at the deposit's own
time, twelve parameters, nested to the incumbent). This script:

  1. builds the objective (D4): at z=0.4, on the sane + mh-complete mask,
         L = score_A^2 + score_F^2 + score_S^2
     with score_A and score_F exactly exp54's (amplitude at 100 kpc over
     SIGMA_A; the production shape loss over L_F_REF) and score_S the
     shells/log objective on the amplitude-pinned profiles over its value for
     the NESTED INCUMBENT on the exact engine — so every term is ~1 at the
     null and the innermost aperture is priced (exp54's loss never saw it);
  2. fits by L-BFGS-B (`hongshao.fitting.minimize_loss`, finite differences)
     from seven starts — the nested incumbent (named), Stage 1's default,
     four seeded jitters of it, and the default with n_c = 2 — checkpointing
     after every start (`.PARTIAL.npz`);
  3. applies the gates G2a-G2f (thresholds frozen in the Stage 2 plan) to the
     best fit AND to the null, and records the verdicts (the gates judge the
     model; they do not stop the outputs);
  4. PREDICTS z=0.7-2.0 by truncating the same integral: the standard battery,
     the both-sample bias tables, the C8 decliner split, and whether the model
     reproduces Stage 1's leverage correlations;
  5. four figures.

Nothing at z > 0.4 enters the loss. The fit uses the reduced quadrature
(model2.FIT_NODES, 1.2e-5 dex from the full one); every reported number uses
the full nodes.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp63_analytic_growth/stage2_fit.py [--smoke] [--fit-only] [--eval-only]
       [--starts a-b] [--delay]

`--delay` fits the 13-parameter model with the extended channel's
accretion-to-deposition delay (Stage 2b); outputs carry the `_delay` tag.
`--tau-fixed X` holds tau_d at X (a physics-set value) and fits the other
twelve at z=0.4; outputs carry the `_tauX` tag.
`--growth` fits the 13-parameter model whose split also depends on the halo's
growth rate at the deposit (Stage 2c); outputs carry the `_growth` tag.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", HERE):
    sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402
import fit as F                                          # noqa: E402
import model2 as M2                                      # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage0_rebaseline as S0R                          # noqa: E402
from hongshao import qa                                  # noqa: E402
from hongshao.fitting import minimize_loss               # noqa: E402
from hongshao.objective import Objective                 # noqa: E402

RULE = "=" * 92
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
QADIR = FIGDIR / "qa"
ANCHOR_Z = E.ANCHOR_Z
TABLE_KEYS = ("kpc:M(<10)", "kpc:M(<100)", "kpc:M(50-100)")
#: the gates, frozen (Stage 2 plan, Step 3)
G2A_MEDIAN, G2A_TOL, G2A_WIDTH_DATA, G2A_WIDTH_FRAC = 1.00, 0.05, 0.38, 0.7
G2B_TOL = 0.05
G2C_SLOPE_TRUTH, G2C_TOL = 1.42, 0.10
G2E_LOG_F_C, G2E_TOL = np.log10(0.02), 0.5
I3 = 3                                                   # R_GRID[3] = 4.92 kpc
MAX_EVALS = {"smoke": 60, "full": 3000}


# --------------------------------------------------------------------------- #
class Problem2:
    """The z=0.4 objective on the fit mask."""

    def __init__(self, spec2, curves, data, fit_rows, R, l_s_ref=None):
        self.spec2, self.R = spec2, R
        self.curves = [curves[i] for i in fit_rows]
        d = data[fit_rows, 0, :]
        self.d = d
        self.A_d = np.log10(d[:, F.I100])
        self.F_d = (d / d[:, F.I100][:, None])[:, None, :]
        self.D_msun = d[:, None, :]                      # for the shells/log term, in Msun
        self.obj_F = Objective()
        self.obj_S = Objective(quantity="shells", residual="log")
        self.sig = F.SIGMA_A[0]
        self.l_s_ref = l_s_ref
        self.n_eval = 0

    def scores(self, theta, nodes=M2.FIT_NODES):
        m = M2.predict2(self.spec2, theta, self.curves, self.R, epochs=(0,), nodes=nodes)[:, 0, :]
        good = np.isfinite(m).all(1) & (m[:, F.I100] > 0)
        n_bad = int((~good).sum())
        if good.sum() < 10:
            return np.nan, np.nan, np.nan, n_bad
        mg = m[good]
        A_m = np.log10(np.clip(mg[:, F.I100], 1.0, None))
        score_A = float(np.sqrt(np.mean(((A_m - self.A_d[good]) / self.sig) ** 2)))
        F_m = (mg / np.clip(mg[:, F.I100], 1.0, None)[:, None])[:, None, :]
        score_F = float(self.obj_F(F_m, self.F_d[good], self.R)) / F.L_F_REF
        # the shells/log objective floors empty shells at 1 Msun, so it must see
        # profiles in SOLAR MASSES: the model pinned to the data's M(<100 kpc)
        P_m = F_m * self.d[good][:, None, F.I100][:, :, None]
        l_s = float(self.obj_S(P_m, self.D_msun[good], self.R))
        score_S = l_s / self.l_s_ref if self.l_s_ref else l_s
        return score_A, score_F, score_S, n_bad

    def loss(self, theta):
        self.n_eval += 1
        a, f, s, n_bad = self.scores(theta)
        if not np.all(np.isfinite([a, f, s])):
            return F.FAIL
        return a * a + f * f + s * s + F.FAIL * n_bad / len(self.curves)


def starts_for(spec2, th_inc, rng):
    base = M2.default_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split)
    lo, hi = np.array(spec2.bounds()).T
    out = [("nested", M2.clip_to_bounds(spec2, M2.nested_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split))),
           ("default", base)]
    for k in range(4):
        j = base + rng.normal(0, 0.05, base.shape) * np.maximum(np.abs(base), 0.3)
        out.append((f"jitter{k}", np.clip(j, lo, hi)))
    n2 = base.copy(); n2[spec2.index("n_c")] = 2.0
    out.append(("default_n2", n2))
    return out


def inner_slope(cogs, R):
    return np.log10(cogs[..., I3] / cogs[..., 0]) / np.log10(R[I3] / R[0])


def railed(spec2, theta, tol=1e-3):
    lo, hi = np.array(spec2.bounds()).T
    return [n for n, v, a, b in zip(spec2.theta_names, theta, lo, hi)
            if v - a < tol * max(abs(a), 1) or b - v < tol * max(abs(b), 1)]


def gates(name, cogs, data, lmh, good, spec2, theta, curves):
    """The G2 table for one product. Returns dict gate -> (value_str, passed)."""
    R = F.R_GRID
    out = {}
    sm, sd = inner_slope(cogs[good][:, 0], R), inner_slope(data[good][:, 0], R)
    med, wid = float(np.median(sm)), float(np.percentile(sm, 84) - np.percentile(sm, 16))
    out["G2a inner slope"] = (f"median {med:.3f} (data {np.median(sd):.3f}), width {wid:.2f} (data {np.percentile(sd, 84) - np.percentile(sd, 16):.2f})",
                              abs(med - G2A_MEDIAN) <= G2A_TOL and wid >= G2A_WIDTH_FRAC * G2A_WIDTH_DATA)
    lm = lmh[good][:, 0]; edges = np.percentile(lm, [0, 100 / 3, 200 / 3, 100])
    worst = 0.0
    for c in range(3):
        sel = (lm >= edges[c]) & (lm <= edges[c + 1])
        mm, dd = cogs[good][sel][:, 0], data[good][sel][:, 0]
        raw = np.median((mm - dd) / dd, axis=0)
        pin = mm / mm[:, F.I100][:, None] * dd[:, F.I100][:, None]
        worst = max(worst, float(np.max(np.abs(raw))), float(np.max(np.abs(np.median((pin - dd) / dd, axis=0)))))
    out["G2b tercile profiles"] = (f"worst |median residual| {100 * worst:.1f}%", worst <= G2B_TOL)
    t, m = qa.measure_all(cogs[good], data[good], R)[:2]
    x_t, y_t = np.log10(t["kpc:M(<30)"][:, 0]), np.log10(t["kpc:M(30-50)"][:, 0])
    x_m, y_m = np.log10(np.clip(m["kpc:M(<30)"][:, 0], 1, None)), np.log10(np.clip(m["kpc:M(30-50)"][:, 0], 1, None))
    st, sm_ = qa.plane_stats(x_t, y_t)["slope"], qa.plane_stats(x_m, y_m)["slope"]
    out["G2c plane slope"] = (f"M(30-50)|M(<30): model {sm_:.2f}, truth {st:.2f}", abs(sm_ - st) <= G2C_TOL)
    big = M2.predict2(spec2, theta, curves[:20], np.array([1e9]))[:, 0, 0]
    cm = M2.channel_masses(spec2, theta, curves[:20]).sum(1)
    cons = float(np.max(np.abs(big / cm - 1)))
    out["G2d conservation"] = (f"max |M(<1e9)/deposited - 1| = {cons:.1e}", cons < 1e-6)
    p = spec2.unpack(theta)
    out["G2e sizes physical"] = (f"log f_c {p['log_f_c']:+.2f} (0.02 -> {G2E_LOG_F_C:+.2f}), log f_e {p['log_f_e']:+.2f}",
                                 p["log_f_c"] < p["log_f_e"] and abs(p["log_f_c"] - G2E_LOG_F_C) <= G2E_TOL)
    share = float(M2.compact_share(p, 13.5))
    rl = railed(spec2, theta)
    out["G2f identifiable"] = (f"railed {rl or 'none'}; compact share at 10^13.5 = {share:.2f}",
                               not rl and 0.05 <= share <= 0.95)
    print(f"\n  GATES — {name}")
    for g, (v, ok) in out.items():
        print(f"    {g:<24} {'PASS' if ok else 'FAIL':<5} {v}")
    return out


def decliner_split(cogs, data, good, label):
    """C8: median log10(model/truth) of M*(<4.92 kpc) per epoch, split on whether
    the measured M*(<4.92) fell between z=2 and z=0.4."""
    d, m = data[good], cogs[good]
    dec = d[:, 4, I3] > d[:, 0, I3]
    err = np.log10(np.clip(m[:, :, I3], 1, None)) - np.log10(d[:, :, I3])
    print(f"    {label:<22}" + "".join(f"{np.median(err[:, j]):>+9.3f}" for j in range(5))
          + f"   span {np.ptp([np.median(err[:, j]) for j in range(5)]):.3f}")
    for nm, sel in (("  centre declined", dec), ("  did not decline", ~dec)):
        print(f"    {nm + f' ({100 * sel.mean():.0f}%)':<22}" + "".join(
            f"{np.median(err[sel, j]):>+9.3f}" for j in range(5))
              + f"   span {np.ptp([np.median(err[sel, j]) for j in range(5)]):.3f}")
    return err, dec


def leverage_model(spec2, theta, curves, lmh):
    from scipy.stats import spearmanr
    from stage1_deconvolve import halo_variables, partial_spearman
    cm = M2.channel_masses(spec2, theta, curves)
    share = cm[:, 0] / cm.sum(1)
    _, hv = halo_variables([None] * 0, curves) if False else (None, None)
    return share


# --------------------------------------------------------------------------- #
def _spec(delay, growth):
    if delay:
        return M2.Spec2.with_delay()
    if growth:
        return M2.Spec2.with_growth_split()
    return M2.Spec2()


def run_fit(smoke, starts_sel, tag, delay=False, tau_fixed=None, growth=False):
    recs, data, mask, lmh, spec_inc, th_inc, _ = S0.build(smoke)
    curves = E.build_curves(recs, verbose=False)
    R = F.R_GRID
    spec2 = _spec(delay, growth)
    bounds = spec2.bounds()
    if tau_fixed is not None:
        # tau_d set from physics, not fitted: an equal-bounds box freezes it
        bounds[spec2.index("tau_d")] = (tau_fixed, tau_fixed)
        print(f"  tau_d FIXED at {tau_fixed} Hubble times at accretion (not fitted)")
    fit_rows = np.where(np.asarray(mask, bool)[:, 0] & np.isfinite(data[:, 0]).all(1)
                        & (data[:, 0] > 0).all(1))[0]
    print(f"  fit sample: {len(fit_rows)} of {len(curves)} galaxies (sane + mh-complete at z=0.4)")

    # the shells/log reference: the NESTED incumbent on the exact engine (frozen)
    pr = Problem2(spec2, curves, data, fit_rows, R, l_s_ref=None)
    th_nested = M2.nested_theta(th_inc, delay=spec2.delay, growth=spec2.growth_split)
    a, f, s_raw, nb = pr.scores(th_nested)
    pr.l_s_ref = s_raw
    a, f, s, nb = pr.scores(th_nested)
    l_null = pr.loss(th_nested)
    print(f"  the null (nested incumbent, exact engine, z=0.4): score_A {a:.4f}, score_F {f:.4f}, "
          f"shells/log {s_raw:.5f} (-> score_S = 1 by definition), loss {l_null:.6f}, failures {nb}")
    a2, f2, s2, _ = pr.scores(th_nested, nodes=M2.FULL_NODES)
    print(f"  the same on the full nodes: {a2:.4f} / {f2:.4f} / {s2:.4f} (fit nodes are adequate)")

    rng = np.random.default_rng(63)
    starts = starts_for(spec2, th_inc, rng)
    if tau_fixed is not None:
        starts = [(nm, np.r_[th[:-1], tau_fixed]) for nm, th in starts]
    if starts_sel is not None:
        a_, b_ = starts_sel
        starts = starts[a_:b_ + 1]
    results = []
    partial = OUTDIR / f"stage2_fit{tag}.PARTIAL.npz"
    max_evals = MAX_EVALS["smoke" if smoke else "full"]
    for name, p0 in starts:
        t0 = time.time(); pr.n_eval = 0
        l0 = pr.loss(p0)
        r = minimize_loss(pr.loss, p0, method="lbfgsb", bounds=bounds,
                          max_evals=max_evals, fd_step=1e-5)
        dt = time.time() - t0
        rl = railed(spec2, r.x)
        print(f"  start {name:<11} loss {l0:.6f} -> {r.fun:.6f}  ({pr.n_eval} evals, {dt / 60:.1f} min)"
              f"{'  RAILED: ' + ','.join(rl) if rl else ''}", flush=True)
        results.append(dict(name=name, theta0=p0, theta=np.asarray(r.x), loss=float(r.fun),
                            loss0=float(l0), n_eval=pr.n_eval, railed=rl))
        OUTDIR.mkdir(parents=True, exist_ok=True)
        np.savez(partial, names=np.array([q["name"] for q in results]),
                 thetas=np.array([q["theta"] for q in results]),
                 losses=np.array([q["loss"] for q in results]),
                 losses0=np.array([q["loss0"] for q in results]),
                 l_s_ref=pr.l_s_ref, loss_null=l_null, theta_nested=th_nested,
                 fit_rows=fit_rows, theta_names=np.array(spec2.theta_names))
    best = min(results, key=lambda q: q["loss"])
    print(f"\n  best start: {best['name']} loss {best['loss']:.6f} (null {l_null:.6f}, "
          f"{100 * (1 - best['loss'] / l_null):+.1f}%)")
    print("  " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(spec2.theta_names, best["theta"])))
    out = OUTDIR / f"stage2_fit{tag}.npz"
    np.savez(out, theta_best=best["theta"], loss_best=best["loss"], best_name=best["name"],
             names=np.array([q["name"] for q in results]),
             thetas=np.array([q["theta"] for q in results]),
             losses=np.array([q["loss"] for q in results]),
             losses0=np.array([q["loss0"] for q in results]),
             l_s_ref=pr.l_s_ref, loss_null=l_null, theta_nested=th_nested, theta_incumbent=th_inc,
             fit_rows=fit_rows, theta_names=np.array(spec2.theta_names))
    print(f"  saved {out.relative_to(ROOT)}")


def run_eval(smoke, tag, tables_only=False, delay=False, growth=False):
    recs, data, mask, lmh, spec_inc, th_inc, _ = S0.build(smoke)
    curves = E.build_curves(recs, verbose=False)
    R = F.R_GRID
    spec2 = _spec(delay, growth)
    fz = np.load(OUTDIR / f"stage2_fit{tag}.npz", allow_pickle=True)
    theta = np.asarray(fz["theta_best"], float)
    th_nested = np.asarray(fz["theta_nested"], float)
    print(f"  best theta ({str(fz['best_name'])}, loss {float(fz['loss_best']):.6f} vs null "
          f"{float(fz['loss_null']):.6f}):")
    print("  " + "  ".join(f"{n}={v:+.4f}" for n, v in zip(spec2.theta_names, theta)))
    cogs = {"baseline": M2.predict2(spec2, th_nested, curves, R),
            "stage2": M2.predict2(spec2, theta, curves, R)}
    good = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    for m in cogs.values():
        good &= np.isfinite(m).all(axis=(1, 2)) & (m > 0).all(axis=(1, 2))
    msk = np.asarray(mask, bool)[good]
    logms = np.log10(np.clip(data[good][:, 0, -1], 1.0, None))
    print(f"  QA sample {int(good.sum())} galaxies; fit mask keeps "
          + "/".join(f"{int(msk[:, j].sum())}" for j in range(5)) + " per epoch")

    g_null = gates("baseline (nested incumbent, exact engine)", cogs["baseline"], data, lmh, good, spec2, th_nested, curves)
    g_fit = gates("stage2 two-channel fit", cogs["stage2"], data, lmh, good, spec2, theta, curves)

    print(f"\n{RULE}\nPREDICTIONS at z=0.7-2.0 (never fitted) — median relative bias [%], all / fit mask\n{RULE}")
    out = {}
    QADIR.mkdir(parents=True, exist_ok=True)
    for m in cogs:
        out[m] = qa.evaluate(cogs[m][good], data[good], R, list(ANCHOR_Z), name=f"exp63_stage2_{m}{tag}",
                             figdir=(None if (tables_only or m == "baseline") else QADIR),
                             figures=not (tables_only or m == "baseline"), verbose=(m == "stage2"),
                             bin_by=lmh[good][:, 0], bin_label=r"logM$_h$(z=0.4)",
                             bin_by_ms=logms, ms_label=r"logM$_*$ (total)")
    for key in TABLE_KEYS:
        print(f"\n  {key}")
        print(f"  {'product':<12}{'sample':<16}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z))
        for m in cogs:
            t, mm = out[m]["truth"][key], out[m]["model"][key]
            print(f"  {m:<12}{'all':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[:, j], t[:, j])):>9.1f}%" for j in range(5)))
            print(f"  {'':<12}{'fit mask':<16}" + "".join(
                f"{100 * np.nanmedian(qa.relerr(mm[msk[:, j], j], t[msk[:, j], j])):>9.1f}%" for j in range(5)))
    print("\n  tier 3, profile max|rel| beyond 5 kpc (median)")
    for m in cogs:
        print(f"  {m:<12}" + "".join(f"{np.nanmedian(out[m]['mr_out'][:, j]):>10.4f}" for j in range(5)))
    print(f"\n  C8 — median log10(model/truth) of M*(<4.92 kpc) per epoch (z=0.4 ... 2.0), decliner split")
    errs = {}
    for m in cogs:
        errs[m] = decliner_split(cogs[m], data, good, m)

    # Stage 1's leverage, reproduced?
    from stage1_deconvolve import halo_variables, partial_spearman
    lmh0, hv = halo_variables(recs, curves)
    cm = M2.channel_masses(spec2, theta, curves)
    share = cm[:, 0] / cm.sum(1)
    print(f"\n  the model's compact share at z=0.4: median {np.median(share):.2f}, 16-84% "
          f"{np.percentile(share, 16):.2f}-{np.percentile(share, 84):.2f}; corr with logMh "
          f"{np.corrcoef(lmh0, share)[0, 1]:+.2f} (Stage 1 required: extended share 0.35->0.57, r=+0.53)")
    print("  partial Spearman at fixed logMh of the model's compact share (Stage 1's inner share in brackets):")
    s1 = np.load(OUTDIR / "stage1_deconvolve.npz", allow_pickle=True) if (OUTDIR / "stage1_deconvolve.npz").exists() else None
    ref = {}
    if s1 is not None:
        vars_ = list(s1["leverage_vars"]); stats_ = list(s1["leverage_stats"])
        L = s1["leverage_gompertz_c0.80"]
        ref = {v: L[stats_.index("inner share s<5"), vars_.index(v)] for v in vars_}
    print("    " + "  ".join(f"{v} {partial_spearman(share, x, lmh0):+.2f} [{ref.get(v, np.nan):+.2f}]"
                             for v, x in hv.items()))

    # figures
    paths = [S0R.compare_figure(cogs, data, lmh, good, f"exp63_stage2_residuals{tag}",
                                models=("baseline", "stage2"), styles={"baseline": "--", "stage2": "-"},
                                pretty={"baseline": "baseline", "stage2": "stage2"},
                                title="exp63 Stage 2 — the two-channel fit (solid) against the exact-engine incumbent (dashed); fitted at z=0.4 only")]
    paths += figures_stage2(cogs, data, lmh, good, spec2, theta, th_nested, curves, share, tag)
    for p in paths:
        print(f"  wrote {p.relative_to(ROOT)}")
    np.savez(OUTDIR / f"stage2_eval{tag}.npz", theta=theta, good=good,
             inner_slope_model=inner_slope(cogs["stage2"][good][:, 0], R),
             inner_slope_baseline=inner_slope(cogs["baseline"][good][:, 0], R),
             inner_slope_data=inner_slope(data[good][:, 0], R), share=share,
             gates_fit=np.array([[g, v, str(ok)] for g, (v, ok) in g_fit.items()]),
             gates_null=np.array([[g, v, str(ok)] for g, (v, ok) in g_null.items()]),
             **{f"bias_{m}_{k}_{s}": np.array([np.nanmedian(qa.relerr(out[m]["model"][k][sel[:, j], j],
                                                                       out[m]["truth"][k][sel[:, j], j]))
                                               for j in range(5)])
                for m in cogs for k in TABLE_KEYS for s, sel in (("all", np.ones_like(msk)), ("fitmask", msk))},
             **{f"c8_{m}": errs[m][0] for m in cogs}, decliner=errs["stage2"][1])
    print(f"  saved {(OUTDIR / f'stage2_eval{tag}.npz').relative_to(ROOT)}")


def figures_stage2(cogs, data, lmh, good, spec2, theta, th_nested, curves, share, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.qa import _tex, _pct
    set_style()
    R = F.R_GRID
    paths = []
    # gates figure: inner slope + the plane
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    sd = inner_slope(data[good][:, 0], R)
    bins = np.linspace(0, 2.2, 45)
    ax[0].hist(sd, bins=bins, alpha=0.4, color="#E69F00", label=f"TNG: median {np.median(sd):.2f}")
    for m, col in (("baseline", "#999999"), ("stage2", "#0072B2")):
        sm = inner_slope(cogs[m][good][:, 0], R)
        ax[0].hist(sm, bins=bins, histtype="step", lw=2, color=col, label=f"{m}: median {np.median(sm):.2f}")
    ax[0].set_xlabel(r"CoG slope $d\log M_*/d\log R$ over 2-4.9 kpc, z=0.4"); ax[0].set_ylabel("galaxies")
    ax[0].legend(fontsize=8); ax[0].set_title("G2a: the inner slope")
    t, m2 = qa.measure_all(cogs["stage2"][good], data[good], R)[:2]
    ax[1].scatter(np.log10(t["kpc:M(<30)"][:, 0]), np.log10(t["kpc:M(30-50)"][:, 0]), s=4, alpha=0.3, color="#E69F00", label="truth")
    ax[1].scatter(np.log10(m2["kpc:M(<30)"][:, 0]), np.log10(m2["kpc:M(30-50)"][:, 0]), s=4, alpha=0.3, color="#0072B2", label="stage2 mean")
    ax[1].set_xlabel(_tex("log M*(<30 kpc)")); ax[1].set_ylabel("log M*(30-50 kpc)"); ax[1].legend(fontsize=8)
    ax[1].set_title("G2c: the plane (mean model, no layer)")
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage2_gates{tag}"); paths.append(FIGDIR / f"exp63_stage2_gates{tag}.png")

    # channels figure: compact share vs logMh, and the two size laws vs z with Stage 1's requirement
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    lm0 = lmh[:, 0]
    ax[0].scatter(lm0, share, s=5, alpha=0.4, color="#0072B2", label="model: compact share of M*(z=0.4)")
    s1p = OUTDIR / "stage1_deconvolve.npz"
    if s1p.exists():
        s1 = np.load(s1p, allow_pickle=True)
        w = s1["w_gompertz_c0.80"]; sg = s1["s_grid"]
        inner = w[:, sg < 15].sum(1) / w.sum(1)
        ax[0].scatter(s1["lmh"], inner, s=5, alpha=0.25, color="#E69F00", label=_tex("Stage 1 required: share at s<15 kpc"))
    ax[0].set_xlabel(_tex("log Mh(z=0.4)")); ax[0].set_ylabel("compact share"); ax[0].legend(fontsize=8)
    ax[0].set_title("the split against halo mass")
    p = spec2.unpack(theta)
    zz = np.linspace(0.4, 8, 100)
    ax[1].plot(zz, p["log_f_c"] + p["b_c"] * np.log10(1 + zz), "-", color="#0072B2", lw=2, label="compact channel s_c/R200c")
    ax[1].plot(zz, p["log_f_e"] + p["b_e"] * np.log10(1 + zz), "-", color="#D55E00", lw=2, label="extended channel s_e/R200c")
    pn = spec2.unpack(th_nested)
    ax[1].plot(zz, pn["log_f_e"] + pn["b_e"] * np.log10(1 + zz), "k--", lw=1.2, label="incumbent's single law")
    if s1p.exists():
        ls_, zs_ = s1["log_s_over_r200_gompertz_c0.80"], s1["z_star"]
        zb = np.array([0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 15.0])
        zc, med, lo, hi = [], [], [], []
        for a_, b_ in zip(zb[:-1], zb[1:]):
            sel = (zs_ >= a_) & (zs_ < b_) & np.isfinite(ls_)
            if sel.sum() >= 20:
                zc.append(np.median(zs_[sel])); med.append(np.median(ls_[sel]))
                lo.append(np.percentile(ls_[sel], 16)); hi.append(np.percentile(ls_[sel], 84))
        ax[1].fill_between(zc, lo, hi, color="gray", alpha=0.2, label="Stage 1 required s*/R200c (16-84)")
        ax[1].plot(zc, med, "o", color="gray")
    ax[1].set_xscale("log"); ax[1].set_xlabel("redshift of the deposit"); ax[1].set_ylabel(r"$\log_{10}(s/R_{200c})$")
    ax[1].legend(fontsize=7); ax[1].set_title("the two size laws")
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage2_channels{tag}"); paths.append(FIGDIR / f"exp63_stage2_channels{tag}.png")

    # predictions figure: bias vs epoch, both samples
    ev = None
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True)
    msk = np.asarray(S0.build(False)[2] if False else None) if False else None
    for a, key in zip(ax, TABLE_KEYS):
        for m, col in (("baseline", "#999999"), ("stage2", "#0072B2")):
            t, mm = qa.measure_all(cogs[m][good], data[good], R)[:2]
            b_all = [100 * np.nanmedian(qa.relerr(mm[key][:, j], t[key][:, j])) for j in range(5)]
            a.plot(ANCHOR_Z, b_all, "o-", color=col, lw=2, label=f"{m}, all")
        a.axhline(0, color="k", lw=0.8); a.axvspan(0.35, 0.45, color="#E69F00", alpha=0.15)
        a.set_title(_tex(key)); a.set_xlabel("z"); a.set_ylabel(f"median relative bias [{_pct()}]")
        a.legend(fontsize=7)
    fig.suptitle("fitted at z=0.4 (shaded); every other epoch is a prediction", fontsize=11)
    fig.tight_layout(); save_fig(fig, FIGDIR / f"exp63_stage2_predictions{tag}"); paths.append(FIGDIR / f"exp63_stage2_predictions{tag}.png")
    return paths


if __name__ == "__main__":
    args = sys.argv[1:]
    smoke = "--smoke" in args
    delay = "--delay" in args or "--tau-fixed" in args
    tau_fixed = float(args[args.index("--tau-fixed") + 1]) if "--tau-fixed" in args else None
    growth = "--growth" in args
    tag = (f"_tau{tau_fixed:g}" if tau_fixed is not None else ("_delay" if delay else ("_growth" if growth else ""))) + ("_smoke" if smoke else "")
    sel = None
    if "--starts" in args:
        a_, b_ = args[args.index("--starts") + 1].split("-")
        sel = (int(a_), int(b_))
    print(f"{RULE}\nexp63 STAGE 2 — the two-channel mean, fitted at z=0.4 only"
          f"{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    if "--eval-only" not in args:
        run_fit(smoke, sel, tag, delay=delay, tau_fixed=tau_fixed, growth=growth)
    if "--fit-only" not in args:
        run_eval(smoke, tag, tables_only="--tables-only" in args, delay=delay, growth=growth)
