"""exp36 — the two-channel deposit (A) + extended theta conditioning (B).

Each MAH deposit splits into a compact IN-SITU channel (the exp35 width law)
and a wide EX-SITU channel (own width scale, shared migration), with a
mass-dependent split fraction:

    CoG_i = (1 - f_ex) B(log_s0,  g, q; t_i)  +  f_ex B(log_s0_ex, g, q; t_i)
    f_ex  = expit(fa + fb mh_std)

Nested: f_ex -> 0 (or log_s0_ex == log_s0) reproduces exp35's single-width
model EXACTLY (asserted in the demo). Everything else (lognormal efficiency,
alpha == 1 dynamical clock, M(<500) total normalization, the (3.0, 4.0)
physical box) is inherited unchanged — the wide channel takes the wide role
so the shared width law no longer has to rail toward the horizon.

Variants (theta layer):
    2ch-global : base5 + [log_s0_ex, fa, fb]                         (8 params)
    2ch-slope  : + logMh slopes on base5                             (13)
    2ch-cond   : (B) slopes on [logMh, c200c, fz2] std, phase-0 pick (23)

Run:  PYTHONPATH=. uv run python experiments/exp36_two_channel/run.py \
        {demo|fit|cv|report} [--variant 2ch-slope] [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E35_DIR = ROOT / "experiments/exp35_total_norm"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
VARIANTS = ("2ch-global", "2ch-slope", "2ch-cond")
NFOLD = 10
_W = {}                                     # per-process worker state


# --------------------------------------------------------------------------- #
# worker state (spawn-safe: children load exp35 fresh, phase0 pattern)         #
# --------------------------------------------------------------------------- #
def _w_init(rows):
    e = _load_by_path("exp35_run", E35_DIR / "run.py")
    mh_scale = tuple(np.load(ROOT / "experiments/exp32_full_population/outputs/"
                             "um_slope_diffmah.npz")["mh_scale"])
    e._init("diffmah", mh_scale, None if rows is None else np.asarray(rows))
    pop = np.load(POP_NPZ)
    row_of = {g["row"]: g for g in e._G["gals"]}
    # (B) conditioning vector per galaxy: standardized [logMh, c200c, fz2]
    z_all = np.column_stack([pop["logmh"], pop["c200c"], pop["fz2"]])
    mu = np.nanmean(z_all, axis=0)
    sd = np.nanstd(z_all, axis=0)
    for r, g in row_of.items():
        z = (z_all[r] - mu) / sd
        g["cond"] = np.where(np.isfinite(z), z, 0.0)
    _W["e"] = e
    _W["gals"] = e._G["gals"]


# --------------------------------------------------------------------------- #
# the two-channel model                                                        #
# --------------------------------------------------------------------------- #
def theta_of_2ch(p, g, variant):
    """Per-galaxy (base5, log_s0_ex, f_ex). The conditioning slopes act on the
    base5 (the phase-0 levers); the split fraction carries the mass tilt."""
    p = np.asarray(p, float)
    if variant == "2ch-global":
        base5, tail = p[:5], p[5:8]
    elif variant == "2ch-slope":
        base5 = p[:5] + p[5:10] * g["cond"][0]
        tail = p[10:13]
    elif variant == "2ch-cond":
        base5 = p[:5] + p[5:20].reshape(5, 3) @ g["cond"]
        tail = p[20:23]
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return base5, tail[0], float(expit(tail[1] + tail[2] * g["cond"][0]))


def model_cogs_2ch(base5, ls_ex, f_ex, mah, m500, ks, e):
    """Two-channel CoGs on the 24 aperture radii, M(<500)-normalized."""
    w = e.weights(mah["z"], base5[3], base5[4])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None
    dM = dM / dM.sum()
    th_in = [base5[0], base5[1], 0.0, base5[2]]
    th_ex = [ls_ex, base5[1], 0.0, base5[2]]
    out = []
    for k in ks:
        mask = mah["snap"] <= e.ANCHOR_SNAP[k]
        B_in = e.basis_ext(th_in, mah["t"], mah["t_obs"], e.pe.AT[k], e.R_EXT)
        m = (1.0 - f_ex) * (B_in @ (dM * mask))
        if f_ex > 0.0:
            B_ex = e.basis_ext(th_ex, mah["t"], mah["t_obs"], e.pe.AT[k], e.R_EXT)
            m = m + f_ex * (B_ex @ (dM * mask))
        if not np.isfinite(m[-1]) or m[-1] <= 0 or m[-2] <= 0:
            return None
        out.append(m[:-1] * (m500[k] / m[-1]))
    return out


def penalty_2ch(p, variant):
    e = _W["e"]
    base = e.penalty(p[:5])
    tail = p[{"2ch-global": 5, "2ch-slope": 10, "2ch-cond": 20}[variant]:]
    lo_t = np.array([1.0, -6.0, -4.0])
    hi_t = np.array([3.0, 6.0, 4.0])
    return base + 30.0 * float(np.sum(np.clip(lo_t - tail, 0, None) ** 2
                                      + np.clip(tail - hi_t, 0, None) ** 2))


def gal_loss_2ch(p, g, ks, variant):
    e = _W["e"]
    base5, ls_ex, f_ex = theta_of_2ch(p, g, variant)
    cogs = model_cogs_2ch(base5, ls_ex, f_ex, g["mah"], g["m500"], ks, e)
    if cogs is None:
        return 4.0
    return float(np.mean([np.sqrt(np.mean(((c - g["data"][k]) / g["data"][k]) ** 2))
                          for c, k in zip(cogs, ks)]))


def gal_eval_2ch(p, g, ks, variant, rmin=5.0):
    """[all-R max|rel|, R>rmin, 148-pinned shape R>rmin, dlog f148] per epoch."""
    e = _W["e"]
    base5, ls_ex, f_ex = theta_of_2ch(p, g, variant)
    cogs = model_cogs_2ch(base5, ls_ex, f_ex, g["mah"], g["m500"], ks, e)
    if cogs is None:
        return np.full((len(ks), 4), np.nan), np.full((len(ks), len(e.R)), np.nan)
    m = e.R > rmin
    met = []
    for c, k in zip(cogs, ks):
        rel = (c - g["data"][k]) / g["data"][k]
        cs = c * (g["data"][k][-1] / c[-1])
        rel_s = (cs - g["data"][k]) / g["data"][k]
        met.append([np.abs(rel).max(), np.abs(rel[m]).max(), np.abs(rel_s[m]).max(),
                    np.log10(c[-1] / g["data"][k][-1])])
    return np.array(met), np.array(cogs)


def _chunk(args):
    p, variant, ks, lo, hi = args
    return sum(gal_loss_2ch(p, g, ks, variant) for g in _W["gals"][lo:hi])


def _cv_fold(args):
    variant, ks, fold, warm, maxiter = args
    gals = _W["gals"]
    n = len(gals)
    train = [i for i in range(n) if i % NFOLD != fold]
    r = minimize(lambda p: np.mean([gal_loss_2ch(p, gals[i], ks, variant)
                                    for i in train]) + penalty_2ch(p, variant),
                 warm, method="Nelder-Mead",
                 options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
    return [(gals[i]["row"], *gal_eval_2ch(r.x, gals[i], ks, variant))
            for i in range(n) if i % NFOLD == fold]


# --------------------------------------------------------------------------- #
# fitting / CV / report                                                        #
# --------------------------------------------------------------------------- #
def fit_pop(variant, ks, starts, maxiter, pool, nchunk, label):
    n = len(_W["gals"])
    edges = np.linspace(0, n, nchunk + 1).astype(int)

    def loss(p):
        parts = pool.map(_chunk, [(p, variant, ks, edges[i], edges[i + 1])
                                  for i in range(nchunk)])
        return sum(parts) / n + penalty_2ch(p, variant)

    best = None
    for p0 in starts:
        t0 = time.time()
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
        print(f"  [{label}] start: loss {r.fun:.4f} ({(time.time()-t0)/60:.1f} min, "
              f"{r.nit} iters)", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def _npz(tag=""):
    return OUTDIR / f"two_channel{tag}.npz"


def _summarize_theta(variant, th):
    _w_init_cached()
    gals = _W["gals"]
    fex = np.array([theta_of_2ch(th, g, variant)[2] for g in gals])
    i0 = {"2ch-global": 5, "2ch-slope": 10, "2ch-cond": 20}[variant]
    print(f"    base5 {np.round(th[:5], 2)}  log_s0_ex {th[i0]:.2f}  "
          f"fa/fb {th[i0+1]:+.2f}/{th[i0+2]:+.2f}  "
          f"f_ex pct16/50/84 = {np.percentile(fex, [16, 50, 84]).round(2)}")


def _w_init_cached(rows=None):
    if "e" not in _W:
        _w_init(rows)


def cmd_fit(dev=False, variants=VARIANTS):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    e35 = np.load(E35_DIR / "outputs/thetas.npz")
    b_g = e35["theta_z04-global"][:5]
    b_s = e35["theta_z04-slope"][:10]
    workers = max(os.cpu_count() - 2, 2)
    tag = "_dev" if dev else ""
    scale = 0.1 if dev else 1.0
    fitted = dict(np.load(_npz(tag))) if _npz(tag).exists() else {}
    print(f"exp36 fit (n={len(_W['gals'])}{', DEV' if dev else ''}, z=0.4 only)")
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for variant in variants:
            if variant == "2ch-global":
                starts = [np.concatenate([b_g, [2.9, -1.0, 1.0]]),
                          np.concatenate([b_g, [2.5, 0.0, 0.5]])]
            elif variant == "2ch-slope":
                t = fitted.get("theta_2ch-global",
                               np.concatenate([b_g, [2.9, -1.0, 1.0]]))
                starts = [np.concatenate([b_s, t[5:8]])]
            else:                                       # 2ch-cond (B)
                t = fitted.get("theta_2ch-slope",
                               np.concatenate([b_s, [2.9, -1.0, 1.0]]))
                S = np.zeros((5, 3))
                S[:, 0] = t[5:10]                       # logMh slopes carry over
                starts = [np.concatenate([t[:5], S.ravel(), t[10:13]])]
            th, lo = fit_pop(variant, [0], starts,
                             max(int(4000 * scale), 80), pool, workers, variant)
            fitted[f"theta_{variant}"] = th
            fitted[f"loss_{variant}"] = lo
            _summarize_theta(variant, th)
            OUTDIR.mkdir(exist_ok=True)
            np.savez(_npz(tag), **fitted)
    print(f"wrote {_npz(tag)}")


def cmd_cv(dev=False, variants=VARIANTS):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    n = len(gals)
    e = _W["e"]
    tag = "_dev" if dev else ""
    fitted = np.load(_npz(tag))
    workers = min(max(os.cpu_count() - 2, 2), NFOLD)
    maxiter = 150 if dev else 1500
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    out = dict(np.load(_npz(tag)))
    print(f"exp36 {NFOLD}-fold CV (z=0.4; metrics: all-R / R>5 | pinned shape "
          "R>5 | dlog f148):")
    for variant in variants:
        t0 = time.time()
        met = np.full((n, 1, 4), np.nan)
        cogs = np.full((n, 1, len(e.R)), np.nan)
        with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
            jobs = [(variant, [0], f, fitted[f"theta_{variant}"], maxiter)
                    for f in range(NFOLD)]
            for part in pool.map(_cv_fold, jobs):
                for row, m_, c_ in part:
                    met[row_to_i[row]] = m_
                    cogs[row_to_i[row]] = c_
        med = 100 * np.nanmedian(met[:, 0], axis=0)
        print(f"  {variant:10s}: {med[0]:5.1f} / {med[1]:5.1f} | {med[2]:5.1f} "
              f"| {np.nanmedian(met[:, 0, 3]):+.4f}  ({(time.time()-t0)/60:.1f} min)")
        out[f"met_{variant}"] = met
        out[f"cogs_{variant}"] = cogs
    np.savez(_npz(tag), **out)
    print(f"wrote {_npz(tag)}")


def cmd_report(dev=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    set_style()
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    d = np.load(_npz(tag))
    tn = np.load(E35_DIR / "outputs/total_norm.npz")
    pop = np.load(POP_NPZ)
    logms = np.array([pop["logms"][g["row"]] for g in gals])
    rws = np.array([g["row"] for g in gals])
    massive = logms >= np.quantile(logms, 2.0 / 3.0)
    print("exp36 report — held-out z=0.4 (marks: statistical 15.6, "
          "unconstrained slope 16.1, exp35 z04-slope 19.6):")
    print("  variant      | pinned shape R>5 | massive dlog f148 (data=0)")
    e35_met = tn["met_z04-slope"][rws, 0]
    print(f"  exp35 slope  | {100*np.nanmedian(e35_met[:, 2]):15.1f} | "
          f"{np.nanmedian(tn['met_multi-slope'][rws][massive, 0, 3]):+.4f}")
    for variant in VARIANTS:
        if f"met_{variant}" not in d.files:
            continue
        met = d[f"met_{variant}"]
        print(f"  {variant:12s} | {100*np.nanmedian(met[:, 0, 2]):15.1f} | "
              f"{np.nanmedian(met[massive, 0, 3]):+.4f}")

    # figure: f148 vs logM* (data / exp35 / best exp36) + f_ex(logMh)
    m500 = np.stack([g["m500"][0] for g in gals])
    data148 = np.stack([g["data"][0][-1] for g in gals])
    f_data = data148 / m500
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax, bx = axes
    order = np.argsort(logms)
    for label, cog_arr, c in (("exp35 z04-slope", tn["cogs_z04-slope"][rws, 0], "#D55E00"),
                              ("exp36 2ch-cond", d.get("cogs_2ch-cond"), "#0072B2")):
        if cog_arr is None:
            continue
        cc = cog_arr[:, 0] if cog_arr.ndim == 3 else cog_arr
        f_m = cc[:, -1] / m500
        ax.plot(logms[order], np.convolve(f_m[order], np.ones(101) / 101, "same"),
                "-", c=c, lw=1.6, label=label)
    ax.plot(logms[order], np.convolve(f_data[order], np.ones(101) / 101, "same"),
            "k-", lw=2.0, label="data")
    ax.set(xlabel=r"log M$_*$", ylabel="f148 = M($<$148)/M($<$500)",
           title="aperture fraction (running median-ish, held-out)")
    ax.legend(fontsize=8)
    if "theta_2ch-cond" in d.files:
        th = d["theta_2ch-cond"]
        fex = np.array([theta_of_2ch(th, g, "2ch-cond")[2] for g in gals])
        logmh = np.array([g["logmh"] for g in gals])
        bx.scatter(logmh, fex, s=6, alpha=0.4, c="#0072B2", edgecolors="none")
        bx.set(xlabel=r"log M$_h$", ylabel=r"f$_{ex}$",
               title="fitted ex-situ split fraction")
    fig.suptitle("exp36 two-channel deposit (z=0.4)", fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    print("wrote", save_fig(fig, FIGDIR / "exp36_two_channel")[0])


# --------------------------------------------------------------------------- #
# self-check                                                                   #
# --------------------------------------------------------------------------- #
def demo():
    rows = np.load(POP_NPZ)["dev100"][:20]
    _w_init(rows)
    e = _W["e"]
    gals = _W["gals"]
    p5 = np.array([2.4, 2.0, 0.9, 1.3, 0.4])

    # (1) NESTING: f_ex -> 0 reproduces the exp35 single-width model exactly
    for g in gals[:8]:
        ref = e.model_cogs_total(p5, g["mah"], g["m500"], [0, 4])
        got = model_cogs_2ch(p5, 2.9, 0.0, g["mah"], g["m500"], [0, 4], e)
        assert ref is not None and got is not None
        err = max(np.abs(np.asarray(a) / np.asarray(b) - 1.0).max()
                  for a, b in zip(got, ref))
        assert err < 1e-12, f"f_ex=0 nesting broken: {err:.2e}"

    # (2) identical channels: any f_ex with log_s0_ex == log_s0 is the base model
    for g in gals[:8]:
        ref = e.model_cogs_total(p5, g["mah"], g["m500"], [0])
        got = model_cogs_2ch(p5, p5[0], 0.7, g["mah"], g["m500"], [0], e)
        err = np.abs(np.asarray(got[0]) / np.asarray(ref[0]) - 1.0).max()
        assert err < 1e-12, f"equal-channel nesting broken: {err:.2e}"

    # (3) a wider ex channel moves mass outward: the 148/500 fraction and the
    # 50-kpc fraction both fall as f_ex rises
    g = gals[0]
    fr = []
    for f_ex in (0.0, 0.3, 0.7):
        c = model_cogs_2ch(p5, 3.0, f_ex, g["mah"], g["m500"], [0], e)[0]
        i50 = int(np.searchsorted(e.R, 50.0))
        fr.append((c[-1] / g["m500"][0], c[i50] / c[-1]))
    assert fr[0][0] > fr[1][0] > fr[2][0], "f148 must fall with f_ex"
    assert fr[0][1] > fr[2][1], "inner fraction must fall with f_ex"

    # (4) the theta layer: zero slopes/tilt reduce every variant to global
    g = gals[3]
    p_g = np.concatenate([p5, [2.8, 0.5, 0.0]])
    b_g, ls_g, fex_g = theta_of_2ch(p_g, g, "2ch-global")
    p_s = np.concatenate([p5, np.zeros(5), [2.8, 0.5, 0.0]])
    p_c = np.concatenate([p5, np.zeros(15), [2.8, 0.5, 0.0]])
    for v, p in (("2ch-slope", p_s), ("2ch-cond", p_c)):
        b, ls, fex = theta_of_2ch(p, g, v)
        assert np.allclose(b, b_g) and ls == ls_g and abs(fex - fex_g) < 1e-12, v
    # conditioning enters: a c200c slope moves base5 only through g["cond"][1]
    p_c2 = p_c.copy()
    p_c2[5 + 1] = 0.3                        # base5[0] slope on c200c
    b2, _, _ = theta_of_2ch(p_c2, g, "2ch-cond")
    assert np.isclose(b2[0] - b_g[0], 0.3 * g["cond"][1])

    print("exp36 demo OK: f_ex=0 and equal-channel nesting exact vs exp35; "
          "wider ex channel moves mass outward; theta layer nests "
          "global->slope->cond with the phase-0 conditioning vector")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    variants = ([sys.argv[sys.argv.index("--variant") + 1]]
                if "--variant" in sys.argv else VARIANTS)
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev, variants)
    elif cmd == "cv":
        cmd_cv(dev, variants)
    elif cmd == "report":
        cmd_report(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
