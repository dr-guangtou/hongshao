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
    2ch-prune  : cond restricted to the log_s0 and sig rows          (14)
    2ch-fa     : prune + the split amplitude conditioned on the full
                 [logMh, c200c, fz2] vector — the low-mass outskirt
                 overshoot fix (f_ex can fall with c200c/fz2 at
                 fixed logMh)                                        (16)

Multi-epoch round (2026-07-14): ``fit --multi`` / ``cv --multi`` run the
joint 5-epoch fit (ks = [0..4], keys prefixed ``multi_``); ``differential``
runs exp35's differential-deposition physics test + the mu-stability check
(P4); ``overshoot`` prints the judged tercile dlog Sigma table (the +0.13
dex low-mass overshoot must fall); ``stress`` bounds-stresses log_s0_ex
(3.0 -> 3.5) on the multi fit.

Run:  PYTHONPATH=. uv run python experiments/exp36_two_channel/run.py \
        {demo|fit|cv|report|anatomy|overshoot|differential|stress} \
        [--variant 2ch-prune] [--multi] [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

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
VARIANTS = ("2ch-global", "2ch-slope", "2ch-cond", "2ch-prune", "2ch-fa")
_TAIL_I = {"2ch-global": 5, "2ch-slope": 10, "2ch-cond": 20, "2ch-prune": 11,
           "2ch-fa": 11}
_PARENT = {"2ch-fa": "2ch-prune"}           # nesting inequality checks
KS_MULTI = [0, 1, 2, 3, 4]
NFOLD = 10
S0EX_HI = 3.0                               # stress overrides (parent-side only)
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
    elif variant in ("2ch-prune", "2ch-fa"):
        # the compact best-fit: condition only log_s0 and sig — the two rows
        # the full cond fit actually used (ablation 2026-07-14); 2ch-fa adds
        # c200c/fz2 slopes on the split amplitude (tail grows 3 -> 5)
        S = np.zeros((5, 3))
        S[0] = p[5:8]
        S[4] = p[8:11]
        base5 = p[:5] + S @ g["cond"]
        tail = p[11:14] if variant == "2ch-prune" else p[11:16]
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return base5, tail[0], float(expit(tail[1]
                                       + tail[2:] @ g["cond"][:len(tail) - 2]))


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


def model_cogs_components(base5, ls_ex, f_ex, mah, m500, ks, e):
    """Per-channel CoGs (compact, wide), sharing the TOTAL's M(<500)
    normalization — their sum equals ``model_cogs_2ch`` exactly."""
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
        m_in = (1.0 - f_ex) * (e.basis_ext(th_in, mah["t"], mah["t_obs"],
                                           e.pe.AT[k], e.R_EXT) @ (dM * mask))
        m_ex = f_ex * (e.basis_ext(th_ex, mah["t"], mah["t_obs"],
                                   e.pe.AT[k], e.R_EXT) @ (dM * mask))
        tot = m_in[-1] + m_ex[-1]
        if not np.isfinite(tot) or tot <= 0:
            return None
        s = m500[k] / tot
        out.append((m_in[:-1] * s, m_ex[:-1] * s))
    return out


def penalty_2ch(p, variant):
    # NOTE: fit_pop adds this in the PARENT process, so the stress override of
    # S0EX_HI reaches the fit; spawn children (_cv_fold) see the default.
    e = _W["e"]
    base = e.penalty(p[:5])
    tail = p[_TAIL_I[variant]:]
    lo_t = np.array([1.0, -6.0, -4.0, -4.0, -4.0])[:len(tail)]
    hi_t = np.array([S0EX_HI, 6.0, 4.0, 4.0, 4.0])[:len(tail)]
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
    i0 = _TAIL_I[variant]
    slopes = "/".join(f"{v:+.2f}" for v in th[i0 + 1:])
    print(f"    base5 {np.round(th[:5], 2)}  log_s0_ex {th[i0]:.2f}  "
          f"fa+slopes {slopes}  "
          f"f_ex pct16/50/84 = {np.percentile(fex, [16, 50, 84]).round(2)}")


def _w_init_cached(rows=None):
    if "e" not in _W:
        _w_init(rows)


def _starts_z04(variant, fitted, b_g, b_s):
    if variant == "2ch-global":
        return [np.concatenate([b_g, [2.9, -1.0, 1.0]]),
                np.concatenate([b_g, [2.5, 0.0, 0.5]])]
    if variant == "2ch-slope":
        # warm from the fitted 2ch-global basin (zero slopes), NOT from
        # exp35's railed slope theta — a nested model must not lose to
        # its own special case
        t = fitted.get("theta_2ch-global",
                       np.concatenate([b_g, [2.9, -1.0, 1.0]]))
        return [np.concatenate([t[:5], np.zeros(5), t[5:8]]),
                np.concatenate([b_s, t[5:8]])]
    if variant == "2ch-cond":                           # (B)
        t = fitted.get("theta_2ch-slope",
                       np.concatenate([b_s, [2.9, -1.0, 1.0]]))
        S = np.zeros((5, 3))
        S[:, 0] = t[5:10]                               # logMh slopes carry over
        return [np.concatenate([t[:5], S.ravel(), t[10:13]])]
    if variant == "2ch-prune":                          # compact B
        t = fitted.get("theta_2ch-cond")
        if t is not None:
            S = t[5:20].reshape(5, 3)
            return [np.concatenate([t[:5], S[0], S[4], t[20:23]])]
        tg = fitted.get("theta_2ch-global",
                        np.concatenate([b_g, [2.9, -1.0, 1.0]]))
        return [np.concatenate([tg[:5], np.zeros(6), tg[5:8]])]
    # 2ch-fa: warm from the fitted prune basin + zeros for the new fa slopes
    t = fitted["theta_2ch-prune"]
    return [np.concatenate([t, [0.0, 0.0]])]


def _starts_multi(variant, fitted, mu_sig_multi):
    """Multi-epoch starts: the z=0.4 basin of the SAME variant, plus a
    foothold with exp35's multi-fit efficiency (P4: the lognormal peak is
    the one ingredient that drifts between z04 and multi basins)."""
    starts = []
    t = fitted.get(f"theta_{variant}")
    if variant == "2ch-fa" and "theta_multi_2ch-prune" in fitted:
        # the nested warm start: the fitted multi sub-model + zero fa slopes
        starts.append(np.concatenate([fitted["theta_multi_2ch-prune"],
                                      [0.0, 0.0]]))
    if t is None:
        raise SystemExit(f"multi fit needs the z04 theta_{variant} first")
    starts.append(np.asarray(t, float))
    foot = np.asarray(t, float).copy()
    foot[3:5] = mu_sig_multi
    starts.append(foot)
    return starts


def _check_nesting(variant, prefix, fitted):
    parent = _PARENT.get(variant)
    key = f"loss_{prefix}{parent}"
    if parent is None or key not in fitted:
        return
    lo, lo_p = float(fitted[f"loss_{prefix}{variant}"]), float(fitted[key])
    if lo > lo_p + 1e-9:
        print(f"    NESTING VIOLATION: {prefix}{variant} {lo:.4f} > "
              f"{prefix}{parent} {lo_p:.4f} — refit required", flush=True)
    else:
        print(f"    nesting OK: {lo:.4f} <= {prefix}{parent} {lo_p:.4f}")


def cmd_fit(dev=False, variants=VARIANTS, multi=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    e35 = np.load(E35_DIR / "outputs/thetas.npz")
    b_g = e35["theta_z04-global"][:5]
    b_s = e35["theta_z04-slope"][:10]
    ks = KS_MULTI if multi else [0]
    pf = "multi_" if multi else ""
    workers = max(os.cpu_count() - 2, 2)
    tag = "_dev" if dev else ""
    scale = 0.1 if dev else 1.0
    fitted = dict(np.load(_npz(tag))) if _npz(tag).exists() else {}
    print(f"exp36 fit (n={len(_W['gals'])}{', DEV' if dev else ''}, "
          f"{'joint 5-epoch' if multi else 'z=0.4 only'})")
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for variant in variants:
            if multi:
                starts = _starts_multi(variant, fitted,
                                       e35["theta_multi-slope"][3:5])
            else:
                starts = _starts_z04(variant, fitted, b_g, b_s)
            th, lo = fit_pop(variant, ks, starts,
                             max(int(4000 * scale), 80), pool, workers,
                             f"{pf}{variant}")
            fitted[f"theta_{pf}{variant}"] = th
            fitted[f"loss_{pf}{variant}"] = lo
            _summarize_theta(variant, th)
            _check_nesting(variant, pf, fitted)
            OUTDIR.mkdir(exist_ok=True)
            np.savez(_npz(tag), **fitted)
    print(f"wrote {_npz(tag)}")


def cmd_cv(dev=False, variants=VARIANTS, multi=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    n = len(gals)
    e = _W["e"]
    ks = KS_MULTI if multi else [0]
    pf = "multi_" if multi else ""
    tag = "_dev" if dev else ""
    fitted = np.load(_npz(tag))
    workers = min(max(os.cpu_count() - 2, 2), NFOLD)
    maxiter = 150 if dev else 1500
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    out = dict(np.load(_npz(tag)))
    print(f"exp36 {NFOLD}-fold CV ({'joint 5-epoch' if multi else 'z=0.4'}; "
          "metrics: all-R / R>5 | pinned shape R>5 | dlog f148):")
    for variant in variants:
        t0 = time.time()
        met = np.full((n, len(ks), 4), np.nan)
        cogs = np.full((n, len(ks), len(e.R)), np.nan)
        with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
            jobs = [(variant, ks, f, fitted[f"theta_{pf}{variant}"], maxiter)
                    for f in range(NFOLD)]
            for part in pool.map(_cv_fold, jobs):
                for row, m_, c_ in part:
                    met[row_to_i[row]] = m_
                    cogs[row_to_i[row]] = c_
        line = " ".join(
            f"z{e.ANCHOR_Z[k]}: {100*np.nanmedian(met[:, j, 0]):.1f}/"
            f"{100*np.nanmedian(met[:, j, 1]):.1f} | "
            f"{100*np.nanmedian(met[:, j, 2]):.1f} | "
            f"{np.nanmedian(met[:, j, 3]):+.4f}"
            for j, k in enumerate(ks))
        print(f"  {pf}{variant}: {line}  ({(time.time()-t0)/60:.1f} min)",
              flush=True)
        out[f"met_{pf}{variant}"] = met
        out[f"cogs_{pf}{variant}"] = cogs
    np.savez(_npz(tag), **out)
    print(f"wrote {_npz(tag)}")


def _binned_median(x, y, nbin=30):
    """Quantile-binned medians (a zero-padded running mean drags the curve
    down at the array edges — the sparse massive end, exactly where it counts)."""
    good = np.isfinite(x) & np.isfinite(y)
    edges = np.quantile(x[good], np.linspace(0, 1, nbin + 1))
    xc, yc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = good & (x >= a) & (x <= b)
        if m.sum() >= 5:
            xc.append(np.median(x[m]))
            yc.append(np.median(y[m]))
    return np.array(xc), np.array(yc)


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
    best = ("2ch-prune" if "cogs_2ch-prune" in d.files else "2ch-cond")
    for label, cog_arr, c in (("exp35 z04-slope", tn["cogs_z04-slope"][rws, 0], "#D55E00"),
                              (f"exp36 {best}", d.get(f"cogs_{best}"), "#0072B2")):
        if cog_arr is None:
            continue
        cc = cog_arr[:, 0] if cog_arr.ndim == 3 else cog_arr
        ax.plot(*_binned_median(logms, cc[:, -1] / m500), "-", c=c, lw=1.6,
                label=label)
    ax.plot(*_binned_median(logms, f_data), "k-", lw=2.0, label="data")
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


def cmd_anatomy(dev=False):
    """Sanity anatomy of the best fit: per-channel surface-density profiles by
    stellar-mass tercile, and the fitted parameter-halo relations (user habit
    rule: SHOW what a conditioned parameter does across the population)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.profile_emulator import density_from_cog
    set_style()
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    e = _W["e"]
    d = np.load(_npz(tag))
    variant = "2ch-prune" if "theta_2ch-prune" in d.files else "2ch-cond"
    th = d[f"theta_{variant}"]
    n = len(gals)
    R = e.R
    ci = np.full((n, len(R)), np.nan)
    cx = np.full((n, len(R)), np.nan)
    b5s = np.full((n, 5), np.nan)
    fexs = np.full(n, np.nan)
    for i, g in enumerate(gals):
        b5, ls, fx = theta_of_2ch(th, g, variant)
        b5s[i], fexs[i] = b5, fx
        parts = model_cogs_components(b5, ls, fx, g["mah"], g["m500"], [0], e)
        if parts is not None:
            ci[i], cx[i] = parts[0]
    pop = np.load(POP_NPZ)
    logms = np.array([pop["logms"][g["row"]] for g in gals])
    logmh = np.array([g["logmh"] for g in gals])
    data0 = np.stack([g["data"][0] for g in gals])

    # figure 1 — per-channel surface density by stellar-mass tercile
    def med_dens(cogs, sel):
        ok = sel & np.isfinite(cogs).all(1) & (cogs > 0).all(1)
        ls_, mid = density_from_cog(np.log10(cogs[ok]), R)
        return np.nanmedian(ls_, axis=0), mid

    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=True)
    for b, ax in enumerate(axes):
        sel = (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
        for cogs, lab, c, lw, st in ((data0, "data", "k", 2.0, "-"),
                                     (ci + cx, "model total", "0.5", 1.5, "--"),
                                     (ci, "compact channel", "#0072B2", 1.6, "-"),
                                     (cx, "wide channel", "#D55E00", 1.6, "-")):
            md, mid = med_dens(cogs, sel)
            ax.plot(mid, md, st, c=c, lw=lw, label=lab if b == 0 else None)
        ax.set(xscale="log", xlabel="R [kpc]",
               title=f"logM$_*$ {edges[b]:.2f}-{edges[b+1]:.2f}")
    axes[0].set_ylabel(r"median log$_{10}$ $\Sigma_*$ [M$_\odot$ kpc$^{-2}$]")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"exp36 [{variant}] channel anatomy at z=0.4 "
                 "(channel labels are placeholders pending hydro-sim checks)",
                 fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    print("wrote", save_fig(fig, FIGDIR / "exp36_components")[0])

    # figure 2 — the fitted parameter-halo relations + the two split fractions
    names = [r"log s$_0$ [kpc]", "g", "q", r"$\mu$", r"$\sigma_f$"]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.6))
    for j, (nm, ax) in enumerate(zip(names, axes.ravel())):
        ax.scatter(logmh, b5s[:, j], s=4, alpha=0.3, c="#0072B2",
                   edgecolors="none")
        ax.set(xlabel=r"log M$_h$", ylabel=nm)
    ax = axes.ravel()[5]
    fex148 = cx[:, -1] / (ci[:, -1] + cx[:, -1])
    ax.scatter(logmh, fexs, s=4, alpha=0.25, c="0.6", edgecolors="none",
               label="deposit share f$_{ex}$")
    ax.plot(*_binned_median(logmh, fex148), "-", c="#D55E00", lw=2.0,
            label=r"wide-channel share OF M($<$148 kpc)")
    ax.set(xlabel=r"log M$_h$", ylabel="fraction", ylim=(0, 1))
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"exp36 [{variant}] fitted parameter-halo relations "
                 "(point spread at fixed M$_h$ = the c200c/fz2 conditioning)",
                 fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / "exp36_relations")[0])


def cmd_report_multi(dev=False):
    """The multi-epoch round verdict figure: per-epoch held-out shape vs the
    marks, the overshoot terciles (baseline vs the fa fix), the massive-tercile
    differential curve, and the P4 efficiency-peak drift."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    from hongshao.profile_emulator import density_from_cog
    from hongshao.qa import _pct, _tex
    set_style()
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    e = _W["e"]
    d = np.load(_npz(tag))
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    zs = [e.ANCHOR_Z[k] for k in KS_MULTI]
    cols = {"multi_2ch-prune": "#0072B2", "multi_2ch-fa": "#D55E00"}

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.4))
    a, b, c, p4 = axes

    # (a) per-epoch held-out pinned shape vs the marks
    for key, col in cols.items():
        met = d[f"met_{key}"]
        a.plot(zs, [100 * np.nanmedian(met[:, j, 2]) for j in range(5)],
               "-o", c=col, lw=1.8, ms=4, label=key.replace("_", " "))
    a.axhline(16.4, color="0.2", ls=":", lw=1.2)
    a.text(1.55, 16.45, "z04-only 2ch-prune (16.4)", fontsize=7, color="0.2")
    a.axhline(19.1, color="0.5", ls="--", lw=1.2)
    a.text(1.55, 19.15, "exp35 unconstrained multi (19.1)", fontsize=7,
           color="0.4")
    a.set(xlabel="epoch z",
          ylabel=_tex("held-out pinned shape max|rel| R>5") + f" [{_pct()}]",
          title="joint 5-epoch fit, 10-fold CV")
    a.legend(fontsize=8)

    # (b) the judged overshoot terciles: baseline vs the multi fa fix
    ls_d, mid = density_from_cog(np.log10(data[:, 0]), e.R)
    bands = ((mid >= 30.0) & (mid < 60.0), (mid >= 60.0) & (mid <= 148.0))
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    x = np.arange(3)
    for off, (key, col) in zip((-0.18, 0.18),
                               (("cogs_2ch-prune", "0.55"),
                                ("cogs_multi_2ch-fa", "#D55E00"))):
        cogs = d[key][:, 0]
        ok = np.isfinite(cogs).all(1) & (cogs > 0).all(1)
        ls_m = np.full_like(ls_d, np.nan)
        ls_m[ok] = density_from_cog(np.log10(cogs[ok]), e.R)[0]
        dl = ls_m - ls_d
        for bi, (bm, hatch) in enumerate(zip(bands, (None, "//"))):
            v = [np.nanmedian(dl[np.ix_(ok & (logms >= edges[t])
                                        & (logms <= edges[t + 1] + 1e-9), bm)])
                 for t in range(3)]
            b.bar(x + off + (0.09 if bi else -0.09), v, width=0.16, color=col,
                  hatch=hatch, edgecolor="w",
                  label=(key.replace("cogs_", "").replace("_", " ")
                         + (" 60-148" if bi else " 30-60")))
    b.axhline(0, color="k", lw=0.8)
    b.set_xticks(x)
    b.set_xticklabels([f"T{t+1} {edges[t]:.1f}-{edges[t+1]:.1f}"
                       for t in range(3)], fontsize=8)
    b.set(ylabel=_tex("held-out median dlog Sigma (model - data) [dex]"),
          title="the outskirt overshoot, z=0.4")
    b.legend(fontsize=7)

    # (c) massive-tercile differential curve, data vs the two multi fits
    x4 = np.arange(4)
    drawn_data = False
    for key, col in cols.items():
        cogs = _model_cogs_all(key.replace("multi_", ""),
                               d[f"theta_{key}"], KS_MULTI)
        _, rows_d = e.differential(cogs, data, logms)
        if not drawn_data:
            c.plot(x4, [rows_d[("data", 2, k)][0] for k in range(4)], "-o",
                   c="0.2", lw=1.8, ms=4, label="measured")
            drawn_data = True
        c.plot(x4, [rows_d[("model", 2, k)][0] for k in range(4)], "--o",
               c=col, lw=1.6, ms=4, label=key.replace("_", " "))
    c.set_xticks(x4)
    c.set_xticklabels([f"z{e.ANCHOR_Z[k+1]}$\\to$z{e.ANCHOR_Z[k]}"
                       for k in range(4)], fontsize=8)
    c.set(ylabel="median fraction of growth beyond 50 kpc",
          title="differential deposition (massive tercile)")
    c.legend(fontsize=8)

    # (d) P4: the efficiency peak across fits
    names, peaks = [], []
    for variant in ("2ch-prune", "2ch-fa"):
        for pf in ("", "multi_"):
            key = f"theta_{pf}{variant}"
            if key in d.files:
                names.append(f"{pf or 'z04 '}{variant}".replace("_", " "))
                peaks.append(np.expm1(d[key][3]))
    p4.plot(peaks, np.arange(len(names)), "o", c="#0072B2", ms=7)
    p4.set_yticks(np.arange(len(names)))
    p4.set_yticklabels(names, fontsize=8)
    p4.set(xlabel="efficiency peak z", title="P4: peak drift z04 vs multi")
    p4.invert_yaxis()

    fig.suptitle("exp36 multi-epoch round — the four judged results",
                 fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    print("wrote", save_fig(fig, FIGDIR / "exp36_multi_round")[0])


def cmd_overshoot(dev=False):
    """The judged tercile table for the low-mass outskirt overshoot: paired
    median dlog Sigma (model - data) in the 30-60 and 60-148 kpc bands by
    logM* tercile, from the HELD-OUT z=0.4 CV cogs. Baseline (2ch-prune,
    2026-07-14): +0.13 / +0.12 dex in the lowest tercile — must fall."""
    from hongshao.profile_emulator import density_from_cog
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    e = _W["e"]
    d = np.load(_npz(tag))
    pop = np.load(POP_NPZ)
    logms = np.array([pop["logms"][g["row"]] for g in gals])
    data0 = np.stack([g["data"][0] for g in gals])
    ls_d, mid = density_from_cog(np.log10(data0), e.R)
    bands = (("30-60 kpc", (mid >= 30.0) & (mid < 60.0)),
             ("60-148 kpc", (mid >= 60.0) & (mid <= 148.0)))
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    print("exp36 outskirt overshoot — held-out paired median dlog Sigma "
          "(model - data), z=0.4, by logM* tercile:")
    for variant in VARIANTS:
        for pf in ("", "multi_"):
            key = f"cogs_{pf}{variant}"
            if key not in d.files:
                continue
            cogs = d[key][:, 0]
            ok = np.isfinite(cogs).all(1) & (cogs > 0).all(1)
            ls_m = np.full_like(ls_d, np.nan)
            ls_m[ok] = density_from_cog(np.log10(cogs[ok]), e.R)[0]
            dl = ls_m - ls_d
            cells = []
            for b in range(3):
                sel = ok & (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
                cells.append(" ".join(
                    f"{np.nanmedian(dl[np.ix_(sel, bm)]):+.3f}"
                    for _, bm in bands))
            print(f"  {pf}{variant:>10s} [30-60 / 60-148]: "
                  + "  |  ".join(f"T{b+1} {c}" for b, c in enumerate(cells)))


def _model_cogs_all(variant, th, ks):
    """In-sample model CoGs (n, len(ks), 24) from a fitted theta."""
    _w_init_cached()
    gals = _W["gals"]
    e = _W["e"]
    cogs = np.full((len(gals), len(ks), len(e.R)), np.nan)
    for i, g in enumerate(gals):
        b5, ls, fx = theta_of_2ch(th, g, variant)
        out = model_cogs_2ch(b5, ls, fx, g["mah"], g["m500"], ks, e)
        if out is not None:
            cogs[i] = out
    return cogs


def cmd_differential(dev=False):
    """The fixed judged physics test on the multi-epoch fit: exp35's
    differential-deposition statistic (massive tercile z0.7->0.4 measured
    0.37/0.11 beyond 50/100 kpc) must survive the two-channel freedom.
    Plus the P4 check: the efficiency peak mu between the z04 and multi
    basins."""
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    e = _W["e"]
    d = np.load(_npz(tag))
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    print("exp36 differential deposition (data -> model, f>50/f>100 per "
          "inter-snapshot pair; exp35 multi-slope massive z0.7->0.4: "
          "0.40/0.12 vs data 0.37/0.11):")
    for variant in VARIANTS:
        key = f"theta_multi_{variant}"
        if key not in d.files:
            continue
        cogs = _model_cogs_all(variant, d[key], KS_MULTI)
        ed3, rows_d = e.differential(cogs, data, logms)
        for b in range(3):
            cells = [f"z{e.ANCHOR_Z[k+1]}->z{e.ANCHOR_Z[k]}: "
                     f"{rows_d[('data', b, k)][0]:.2f}/"
                     f"{rows_d[('data', b, k)][1]:.2f} -> "
                     f"{rows_d[('model', b, k)][0]:.2f}/"
                     f"{rows_d[('model', b, k)][1]:.2f}"
                     for k in range(4)]
            print(f"  [{variant}] logM* {ed3[b]:.2f}-{ed3[b+1]:.2f}: "
                  + "  ".join(cells))
    print("\n  P4 efficiency-peak stability (mu, sig -> peak z):")
    for variant in VARIANTS:
        for pf in ("", "multi_"):
            key = f"theta_{pf}{variant}"
            if key not in d.files:
                continue
            th = d[key]
            print(f"    {pf or 'z04_'}{variant}: mu {th[3]:.3f} "
                  f"sig {th[4]:.3f} -> peak z {np.expm1(th[3]):.2f}  "
                  f"(log_s0_ex {th[_TAIL_I[variant]]:.2f})")


def cmd_stress(dev=False):
    """Bounds-stress log_s0_ex on the multi fit: it rails at 3.0 in every
    fit. Loosen to 3.5 (3160 kpc) and ask HOW the freedom is spent: if the
    ex channel re-opens the >500-kpc deletion channel (visibility falls)
    without moving the observables, the rail is a horizon escape again; if
    loss and observables barely move, the rail is benign (f_ex prices it)."""
    global S0EX_HI
    S0EX_HI = 3.5
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    e = _W["e"]
    d = np.load(_npz(tag))
    variant = "2ch-prune"
    base_key = f"theta_multi_{variant}"
    if base_key not in d.files:
        raise SystemExit("stress needs the multi 2ch-prune fit first")
    warm = d[base_key]
    nudge = warm.copy()
    nudge[_TAIL_I[variant]] = 3.3               # foothold past the old rail
    workers = max(os.cpu_count() - 2, 2)
    scale = 0.1 if dev else 1.0
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        th, lo = fit_pop(variant, KS_MULTI, [warm, nudge],
                         max(int(4000 * scale), 80), pool, workers,
                         "stress-multi-prune")
    print(f"\n  loss: stress {lo:.4f} vs baseline "
          f"{float(d[f'loss_multi_{variant}']):.4f}")
    _summarize_theta(variant, th)
    # HOW the freedom is spent: per-z-bin ex-channel visibility (exp35
    # physicality convention, gals[3]), weighted by the fitted efficiency
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    m500 = np.stack([g["m500"] for g in gals])
    massive = logms >= np.quantile(logms, 2 / 3)
    for label, thx in (("baseline", warm), ("stress", th)):
        b5, ls_ex, _ = theta_of_2ch(thx, gals[3], variant)
        mah = gals[3]["mah"]
        sig_ex = np.clip(10.0 ** ls_ex * (mah["t"] / mah["t_obs"]) ** b5[1],
                         1e-4, 1e5)
        w = e.weights(mah["z"], b5[3], b5[4])
        dM = w * mah["dMh"]
        dM = dM / dM.sum()
        vis148 = 1.0 - np.exp(-148.0 ** 2 / (2.0 * sig_ex ** 2))
        vis500 = 1.0 - np.exp(-e.R_TOT ** 2 / (2.0 * sig_ex ** 2))
        print(f"    [{label}] ex-channel per z-bin (weight | vis148 | vis500):")
        for zlo, zhi in ((0.4, 1), (1, 2), (2, 4), (4, 12)):
            zm = (mah["z"] >= zlo) & (mah["z"] < zhi)
            if zm.sum():
                print(f"      z [{zlo},{zhi}): {dM[zm].sum():.3f} | "
                      f"{np.median(vis148[zm]):.3f} | "
                      f"{np.median(vis500[zm]):.3f}")
        cogs = _model_cogs_all(variant, thx, KS_MULTI)
        _, rows_d = e.differential(cogs, data, logms)
        print(f"      massive f148 (z=0.4) "
              f"{np.nanmedian(cogs[massive, 0, -1] / m500[massive, 0]):.3f} "
              f"(data {np.median(data[massive, 0, -1] / m500[massive, 0]):.3f}); "
              "differential z0.7->z0.4 massive "
              + "/".join(f"{v:.2f}" for v in rows_d[("model", 2, 0)]))
    np.savez(OUTDIR / f"stress_ex{tag}.npz", theta=th, loss=lo, s0ex_hi=S0EX_HI)
    print(f"wrote {OUTDIR / f'stress_ex{tag}.npz'}")


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

    # (5) components: the per-channel CoGs sum to the total exactly, and the
    # prune variant nests (zero slopes -> global; equals cond with the other
    # slope rows zeroed)
    g = gals[1]
    tot = model_cogs_2ch(p5, 3.0, 0.6, g["mah"], g["m500"], [0, 4], e)
    parts = model_cogs_components(p5, 3.0, 0.6, g["mah"], g["m500"], [0, 4], e)
    for t_, (ci, cx) in zip(tot, parts):
        assert np.allclose(ci + cx, t_, rtol=1e-12), "components must sum to total"
    p_p = np.concatenate([p5, np.zeros(6), [2.8, 0.5, 0.0]])
    b_p, ls_p, fex_p = theta_of_2ch(p_p, g, "2ch-prune")
    b_g2, ls_g2, fex_g2 = theta_of_2ch(np.concatenate([p5, [2.8, 0.5, 0.0]]),
                                       g, "2ch-global")
    assert np.allclose(b_p, b_g2) and ls_p == ls_g2 and fex_p == fex_g2
    p_p2 = p_p.copy()
    p_p2[5:8] = [0.1, 0.2, 0.3]                        # log_s0 slopes
    p_c2 = np.concatenate([p5, np.zeros(15), [2.8, 0.5, 0.0]])
    p_c2[5:8] = [0.1, 0.2, 0.3]
    assert np.allclose(theta_of_2ch(p_p2, g, "2ch-prune")[0],
                       theta_of_2ch(p_c2, g, "2ch-cond")[0])

    # (6) 2ch-fa: zero c200c/fz2 slopes on fa reduce EXACTLY to 2ch-prune,
    # and a fa slope moves f_ex only through the matching cond component
    for g in gals[:8]:
        p_p3 = np.concatenate([p5, [0.1, 0.2, 0.3, -0.1, 0.0, 0.2],
                               [2.8, 0.5, 0.7]])
        p_f = np.concatenate([p_p3, [0.0, 0.0]])
        bp, lp, fp = theta_of_2ch(p_p3, g, "2ch-prune")
        bf, lf, ff = theta_of_2ch(p_f, g, "2ch-fa")
        assert np.allclose(bp, bf) and lp == lf and abs(fp - ff) < 1e-15
        p_f2 = p_f.copy()
        p_f2[14] = 0.4                           # fa slope on c200c
        _, _, ff2 = theta_of_2ch(p_f2, g, "2ch-fa")
        assert np.isclose(logit(ff2) - logit(ff), 0.4 * g["cond"][1])
    # penalty: the fa tail is bounded like the prune tail + the new slopes
    p_in = np.concatenate([p5, np.zeros(6), [2.8, 0.5, 0.7, 0.1, -0.2]])
    p_out = p_in.copy()
    p_out[14] = 9.0
    assert penalty_2ch(p_in, "2ch-fa") == 0.0
    assert penalty_2ch(p_out, "2ch-fa") > 1.0

    print("exp36 demo OK: f_ex=0 and equal-channel nesting exact vs exp35; "
          "wider ex channel moves mass outward; theta layer nests "
          "global->slope->cond->prune->fa with the phase-0 conditioning "
          "vector; channel components sum to the total exactly")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    multi = "--multi" in sys.argv
    variants = ([sys.argv[sys.argv.index("--variant") + 1]]
                if "--variant" in sys.argv else VARIANTS)
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev, variants, multi)
    elif cmd == "cv":
        cmd_cv(dev, variants, multi)
    elif cmd == "report":
        cmd_report(dev)
    elif cmd == "report-multi":
        cmd_report_multi(dev)
    elif cmd == "anatomy":
        cmd_anatomy(dev)
    elif cmd == "overshoot":
        cmd_overshoot(dev)
    elif cmd == "differential":
        cmd_differential(dev)
    elif cmd == "stress":
        cmd_stress(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
