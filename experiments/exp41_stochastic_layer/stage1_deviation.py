"""exp41 stage 1 — the per-galaxy deviation distribution.

Stage 0 located the individuality: one latent axis (carried here on
the SIG coordinate — the efficiency width; best-behaved box, strongest
f_core alignment at rho = -0.87), with a genuine second axis (carried
on Q — the migration envelope) for a ~fifth of the population. This
stage measures the raw material stage 2 will draw from:

  1-D variant  delta_sig per galaxy = the stage-0 bounded scalar refit
               (anatomy.npz), reused as-is.
  2-D variant  joint (sig, q) refit per galaxy (Nelder-Mead in the
               physical box, official z<=1.5 scope, plain loss).

Reported per variant: percentiles, robust width (1.4826*MAD), skew and
excess kurtosis, a Student-t fit (do the tails need more than a
Gaussian?), the 2-D correlation, and the loss-improvement ladder
(base -> 1-D -> 2-D; the 2-D must reproduce stage 0b's ~32%). The
layer is LOCATION-FREE by design (criterion 2: the mean model is
untouched) — deviations are drawn around zero, so the fitted medians
are reported as the (small) price of that constraint, not subtracted.

Run: PYTHONPATH=. uv run python experiments/exp41_stochastic_layer/\
stage1_deviation.py {demo|fit|report} [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUTDIR = HERE / "outputs"
E38_DIR = ROOT / "experiments/exp38_deposit_rethink"
E40_DIR = ROOT / "experiments/exp40_epoch_objective"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_FIT = [0, 1, 2, 3]
I_SIG, I_Q = 4, 2
LO6 = np.array([1.0, 0.0, 0.0, 0.0, 0.05, 1.06])
HI6 = np.array([3.0, 6.0, 3.0, 3.0, 2.0, 6.0])
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    s3 = _load_by_path("exp38_stage3", E38_DIR / "stage3_inner.py")
    s3._w_init(rows)
    _W.update(s3._W)
    _W["th0"] = np.load(E40_DIR / "outputs/latestart.npz")["theta_z15"]


def _fit2_one(args):
    """One galaxy: joint (sig, q) refit; returns the deltas + losses."""
    (gi,) = args
    s2 = _W["s2"]
    g = _W["gals"][gi]
    th0 = _W["th0"]
    l_base = s2.gal_loss(th0, g, KS_FIT, "1ch-mof")

    def f(x):
        th = th0.copy()
        th[I_SIG] = np.clip(x[0], LO6[I_SIG], HI6[I_SIG])
        th[I_Q] = np.clip(x[1], LO6[I_Q], HI6[I_Q])
        return s2.gal_loss(th, g, KS_FIT, "1ch-mof")

    best = None
    for x0 in (np.array([th0[I_SIG], th0[I_Q]]),
               np.array([max(th0[I_SIG] - 0.1, LO6[I_SIG]), th0[I_Q]])):
        r = minimize(f, x0, method="Nelder-Mead",
                     options=dict(maxiter=250, xatol=1e-4, fatol=1e-9))
        if best is None or r.fun < best.fun:
            best = r
    x = np.array([np.clip(best.x[0], LO6[I_SIG], HI6[I_SIG]),
                  np.clip(best.x[1], LO6[I_Q], HI6[I_Q])])
    if best.fun >= l_base:
        x = np.array([th0[I_SIG], th0[I_Q]])
    return (g["row"], l_base, min(best.fun, l_base),
            x[0] - th0[I_SIG], x[1] - th0[I_Q])


def cmd_fit(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    n = len(_W["gals"])
    workers = max(os.cpu_count() - 2, 2)
    t0 = time.time()
    print(f"exp41 stage 1 — joint (sig, q) per-galaxy deviations (n={n}"
          f"{', DEV' if dev else ''}):", flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        res = pool.map(_fit2_one, [(i,) for i in range(n)])
    row, l_base, l_2, d_sig, d_q = map(np.array, zip(*res))
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"stage1_dev2{tag}.npz", row=row, l_base=l_base,
             l_2=l_2, d_sig=d_sig, d_q=d_q)
    print(f"  wrote {OUTDIR / f'stage1_dev2{tag}.npz'} "
          f"({(time.time()-t0)/60:.1f} min)")
    cmd_report(dev)


def _describe(name, d):
    from scipy import stats
    q = np.percentile(d, [5, 16, 50, 84, 95])
    mad_sigma = 1.4826 * np.median(np.abs(d - np.median(d)))
    tdof, tloc, tscale = stats.t.fit(d)
    print(f"    {name}: pct 5/16/50/84/95 = "
          + "/".join(f"{v:+.3f}" for v in q)
          + f"; robust sigma (1.4826*MAD) = {mad_sigma:.3f}; "
          f"skew {stats.skew(d):+.2f}, excess kurtosis "
          f"{stats.kurtosis(d):+.2f}")
    print(f"      Student-t fit (dof, loc, scale) = ({tdof:.1f}, "
          f"{tloc:+.3f}, {tscale:.3f}) — dof <~ 5 means the tails are "
          "genuinely heavier than Gaussian")
    return mad_sigma, (tdof, tloc, tscale)


def cmd_report(dev=False):
    from scipy.stats import spearmanr, pearsonr
    tag = "_dev" if dev else ""
    d0 = np.load(OUTDIR / f"anatomy{tag}.npz")
    d2 = np.load(OUTDIR / f"stage1_dev2{tag}.npz")
    th0 = d0["th0"]
    d_sig_1d = d0["v_free"][:, I_SIG] - th0[I_SIG]
    imp1 = 1.0 - d0["l_free"][:, I_SIG] / d0["l_base"]
    imp2 = 1.0 - d2["l_2"] / d2["l_base"]
    print("exp41 stage 1 report — the deviation distributions "
          "(location-free layer: the medians below are the price of "
          "keeping the mean model untouched, not subtracted):")
    print("  [1-D variant] delta_sig (stage-0 scalar refits, "
          f"median loss improvement {100*np.median(imp1):.1f}%):")
    s1, t1 = _describe("d_sig(1D)", d_sig_1d)
    print("  [2-D variant] joint (sig, q) refits "
          f"(median loss improvement {100*np.median(imp2):.1f}%; "
          "stage-0b mark ~32%):")
    s2_, t2 = _describe("d_sig(2D)", d2["d_sig"])
    s3_, t3 = _describe("d_q  (2D)", d2["d_q"])
    pr = pearsonr(d2["d_sig"], d2["d_q"]).statistic
    sr = spearmanr(d2["d_sig"], d2["d_q"]).statistic
    print(f"    correlation d_sig x d_q: Pearson {pr:+.2f}, "
          f"Spearman {sr:+.2f} (the 2-D layer must draw them jointly)")
    at_lo = float(np.mean(np.isclose(d2["d_sig"], LO6[I_SIG] - th0[I_SIG],
                                     atol=1e-3)))
    at_hi = float(np.mean(np.isclose(d2["d_q"], LO6[I_Q] - th0[I_Q],
                                     atol=1e-3)))
    print(f"    box pile-up: {100*at_lo:.1f}% of d_sig at the sig floor, "
          f"{100*at_hi:.1f}% of d_q at q = 0 (pile-up means the box, not "
          "the data, shapes that tail — stage 2 must not draw beyond it)")
    np.savez(OUTDIR / f"stage1_dist{tag}.npz",
             d_sig_1d=d_sig_1d, d_sig_2d=d2["d_sig"], d_q_2d=d2["d_q"],
             row=d2["row"], sigma_mad=[s1, s2_, s3_],
             t_1d=t1, t_sig2d=t2, t_q2d=t3, corr=[pr, sr], th0=th0)
    print(f"  wrote {OUTDIR / f'stage1_dist{tag}.npz'}")


def demo():
    rows = np.load(POP_NPZ)["dev100"][:8]
    _w_init(rows)
    s2 = _W["s2"]
    th0 = _W["th0"]
    # (1) the joint refit never loses to the base and respects the box
    row, l_base, l_2, ds, dq = _fit2_one((0,))
    assert l_2 <= l_base + 1e-12
    assert LO6[I_SIG] - 1e-9 <= th0[I_SIG] + ds <= HI6[I_SIG] + 1e-9
    assert LO6[I_Q] - 1e-9 <= th0[I_Q] + dq <= HI6[I_Q] + 1e-9
    # (2) zero deviation reproduces the official model exactly
    g = _W["gals"][0]
    th = th0.copy()
    a = s2.model_cogs(th0, g, [0], "1ch-mof")[0]
    b = s2.model_cogs(th, g, [0], "1ch-mof")[0]
    assert np.abs(a / b - 1.0).max() < 1e-15
    print(f"exp41 stage1 demo OK: joint (sig, q) refit improves gal 0 "
          f"({l_base:.4f} -> {l_2:.4f}, d_sig {ds:+.3f}, d_q {dq:+.3f}) "
          "inside the box; zero deviation is the official model")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev)
    elif cmd == "report":
        cmd_report(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
