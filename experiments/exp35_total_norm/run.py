"""exp35 — the TOTAL-NORMALIZED transport refit (Path B).

exp33 found the aperture-horizon degeneracy: per-epoch pinning to the 148-kpc
mass lets the optimizer delete epochs geometrically; the physical 5-param box
restores physicality at +3-4 points with g railing at its bound — because the
beyond-aperture mass is REAL (exp34: f_out 12% median, 26% massive quartile).
Here the aperture fraction becomes DATA: normalize the model to M(<500 kpc)
computed from exp34's truncation-validated power-tail fits, so the per-epoch
fraction M(<148)/M(<500) is a fitted datum and the deletion channel is
falsifiable. Widths get a LOOSE soft box (log_s0 <= 3.0, g <= 4.0).

Model = exp33 physical theta [log_s0, g, q, mu, sig]: alpha == 1 (self-similar
clock), lognormal efficiency f(z); dyntrans basis evaluated on R + {500 kpc},
model CoG scaled to the M(<500) datum at each epoch.

Judged by (exp33/34 verdicts):
  a. held-out per-epoch SHAPE (148-pinned, R>5 kpc) vs the 16.1% (z04-slope) /
     19.1% (multi-slope at z=0.4) unconstrained marks + the total-normalized
     error (the new, harder metric that includes the fraction);
  b. theta physicality (per-z-bin visible fractions) + z04-vs-multi stability;
  c. NEW physics test — does the fitted width law reproduce exp34's measured
     differential-deposition curve (37%/11% of z=0.7->0.4 growth landing
     beyond 50/100 kpc for the massive tercile)?

Run:  PYTHONPATH=. uv run python experiments/exp35_total_norm/run.py \
        {totals|fit|cv|report} [--dev]
Demo: ... run.py demo
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
for sub in ("exp29_outer_deposit", "exp30_transport_kernel", "exp32_full_population"):
    sys.path.insert(0, str(ROOT / "experiments" / sub))
sys.path.insert(0, str(ROOT))
import param_emulator as pe                                                          # noqa: E402
import universal_mass as um                                                          # noqa: E402
from run import ANCHOR_SNAP, ANCHOR_Z                                                # noqa: E402


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
M500_NPZ = OUTDIR / "m500.npz"
OUT_NPZ = OUTDIR / "total_norm.npz"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
R = pe.R
R_TOT = 500.0
R_EXT = np.append(R, R_TOT)
I50, I100 = int(np.searchsorted(R, 50.0)), int(np.searchsorted(R, 100.0))
NFOLD = 10
LO = np.array([1.0, 0.0, 0.0, 0.0, 0.05])      # loose box: log_s0, g, q, mu, sig
HI = np.array([3.0, 4.0, 3.0, 3.0, 2.0])
STARTS = [np.array([2.0, 1.5, 0.77, np.log(3.2), 0.5]),
          np.array([2.2, 1.0, 1.2, np.log(2.4), 0.8])]
MARKS = {"z04-slope": 16.1, "multi-slope": 19.1}       # unconstrained 7p, R>5 kpc
JOBS = [("z04-global", "global", [0], 2500),
        ("z04-slope", "slope", [0], 5000),
        ("multi-global", "global", [0, 1, 2, 3, 4], 2500),
        ("multi-slope", "slope", [0, 1, 2, 3, 4], 5000)]
_G = {}


# --- step 1: the M(<500 kpc) data --------------------------------------------------
def compute_totals():
    """Refit exp34 power tails, evaluate at 500 kpc (finite radius, NOT M_tot_inf:
    railed a~0 slopes make the infinity soft); expo form = systematic check."""
    e34 = _load_by_path("exp34_run", ROOT / "experiments/exp34_asymptotic_total/run.py")
    pop = np.load(POP_NPZ)
    data = pop["data"]
    n = len(data)
    m500 = np.full((n, 5, 2), np.nan)
    mtot = np.full((n, 5), np.nan)
    fallback = 0
    for i in range(n):
        for k in range(5):
            for fi, (form, f) in enumerate((("power", e34.f_power),
                                            ("expo", e34.f_expo))):
                p = e34.fit_tail(data[i, k], form)
                if p is not None:
                    m500[i, k, fi] = max(f(R_TOT, *p), data[i, k, -1])
                    if form == "power":
                        mtot[i, k] = p[0]
            if not np.isfinite(m500[i, k, 0]):
                m500[i, k, 0] = m500[i, k, 1] if np.isfinite(m500[i, k, 1]) \
                    else data[i, k, -1]
                fallback += 1
    d34 = np.load(ROOT / "experiments/exp34_asymptotic_total/outputs/asymptotic_total.npz")
    agree = np.nanmedian(np.abs(np.log10(mtot / d34["mtot"][:, :, 0])))
    sysd = np.nanmedian(np.abs(np.log10(m500[:, :, 0] / m500[:, :, 1])))
    print(f"m500: n={n}, power-fit fallbacks {fallback}/{5*n}; "
          f"M_tot_inf vs exp34 median |dlog| {agree:.4f} dex; "
          f"form systematic at 500 kpc {sysd:.4f} dex")
    print("  f(<500) beyond-148 fraction 1 - M148/M500, median per epoch: " + "  ".join(
        f"z={ANCHOR_Z[k]}: {np.median(1 - data[:, k, -1] / m500[:, k, 0]):.3f}"
        for k in range(5)))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(M500_NPZ, index=pop["index"], m500=m500[:, :, 0], m500_expo=m500[:, :, 1])
    print(f"wrote {M500_NPZ}")


# --- the total-normalized model -----------------------------------------------------
def basis_ext(th4, ti, t_obs, tk, r):
    """dyntrans unit-mass CoG basis on an arbitrary radius grid (tf.basis is
    hard-wired to the 24-point aperture grid)."""
    s0, g = 10.0 ** th4[0], th4[1]
    sig0 = np.clip(s0 * (ti / t_obs) ** g, 1e-4, 1e5)
    dt = np.clip(tk - ti, 0.0, None)
    fc = np.exp(-dt / (10.0 ** th4[2] * ti))
    sigw = np.clip(sig0 * (tk / ti) ** max(th4[3], 0.0), 1e-4, 1e5)
    core = 1.0 - np.exp(-r[:, None] ** 2 / (2.0 * sig0[None, :] ** 2))
    wide = 1.0 - np.exp(-r[:, None] ** 2 / (2.0 * sigw[None, :] ** 2))
    return fc[None, :] * core + (1.0 - fc)[None, :] * wide


def weights(z, mu, sig):
    return np.exp(-((np.log1p(z) - mu) ** 2) / (2.0 * sig ** 2))


def penalty(p):
    v = np.asarray(p[:5])
    return 30.0 * float(np.sum(np.clip(LO - v, 0, None) ** 2
                               + np.clip(v - HI, 0, None) ** 2))


def model_cogs_total(p, mah, m500, ks):
    """CoGs on the 24 aperture radii, normalized so M_model(<500) = m500[k]."""
    w = weights(mah["z"], p[3], p[4])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None
    dM = dM / dM.sum()
    th4 = [p[0], p[1], 0.0, p[2]]                       # log_alpha = 0 -> alpha = 1
    out = []
    for k in ks:
        mask = mah["snap"] <= ANCHOR_SNAP[k]
        B = basis_ext(th4, mah["t"], mah["t_obs"], pe.AT[k], R_EXT)
        m = B @ (dM * mask)
        if not np.isfinite(m[-1]) or m[-1] <= 0 or m[-2] <= 0:
            return None
        out.append(m[:-1] * (m500[k] / m[-1]))
    return out


def gal_loss(p, g, ks):
    cogs = model_cogs_total(p, g["mah"], g["m500"], ks)
    if cogs is None:
        return 4.0
    return float(np.mean([np.sqrt(np.mean(((c - g["data"][k]) / g["data"][k]) ** 2))
                          for c, k in zip(cogs, ks)]))


def gal_eval(p, g, ks, rmin=5.0):
    """Held-out metrics per epoch: [total-normalized max|rel| all-R, R>rmin,
    148-pinned SHAPE max|rel| R>rmin (the 16.1/19.1 convention), log10 aperture-
    fraction residual], plus the total-normalized cogs for QA."""
    cogs = model_cogs_total(p, g["mah"], g["m500"], ks)
    if cogs is None:
        return np.full((len(ks), 4), np.nan), np.full((len(ks), len(R)), np.nan)
    m = R > rmin
    met = []
    for c, k in zip(cogs, ks):
        rel = (c - g["data"][k]) / g["data"][k]
        cs = c * (g["data"][k][-1] / c[-1])
        rel_s = (cs - g["data"][k]) / g["data"][k]
        met.append([np.abs(rel).max(), np.abs(rel[m]).max(),
                    np.abs(rel_s[m]).max(),
                    np.log10((c[-1] / g["m500"][k]) / (g["data"][k][-1] / g["m500"][k]))])
    return np.array(met), np.array(cogs)


def theta_of(p, g, variant):
    if variant == "global":
        return np.asarray(p[:5])
    return np.asarray(p[:5]) + np.asarray(p[5:10]) * \
        (g["logmh"] - _G["mh_scale"][0]) / _G["mh_scale"][1]


# --- population fitting (physical_theta pattern) ------------------------------------
def _chunk(args):
    p, variant, ks, lo, hi = args
    return sum(gal_loss(theta_of(p, g, variant), g, ks)
               for g in _G["gals"][lo:hi])


def fit_pop(variant, ks, starts, maxiter, pool, nchunk, label):
    n = len(_G["gals"])
    edges = np.linspace(0, n, nchunk + 1).astype(int)

    def loss(p):
        parts = pool.map(_chunk, [(p, variant, ks, edges[i], edges[i + 1])
                                  for i in range(nchunk)])
        return sum(parts) / n + penalty(p)

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


def _init(config, mh_scale, rows=None):
    gals = um.load_gals(config, rows)
    m500 = np.load(M500_NPZ)["m500"]
    for g in gals:
        g["m500"] = m500[g["row"]]
    _G["gals"] = gals
    _G["mh_scale"] = mh_scale


def _setup(dev):
    config = "diffmah"
    mh_scale = tuple(np.load(ROOT / "experiments/exp32_full_population/outputs/"
                             "um_slope_diffmah.npz")["mh_scale"])
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _init(config, mh_scale, rows)
    return config, mh_scale, rows


def run_fit(dev):
    config, mh_scale, rows = _setup(dev)
    n = len(_G["gals"])
    workers = max(os.cpu_count() - 2, 2)
    scale = 0.06 if dev else 1.0
    print(f"exp35 — total-normalized transport refit (5-param physical theta, "
          f"M(<500) pinned; n={n}{' DEV' if dev else ''})\n", flush=True)
    tag = "_dev" if dev else ""
    fitted = {}
    with Pool(workers, initializer=_init, initargs=(config, mh_scale, rows)) as pool:
        for label, variant, ks, mi in JOBS:
            if variant == "global":
                starts = STARTS
            else:                                       # warm-start slopes from global
                g0 = fitted[label.replace("slope", "global")][0]
                starts = [np.concatenate([g0[:5], np.zeros(5)])]
            th, lo = fit_pop(variant, ks, starts, max(int(mi * scale), 60), pool,
                             workers, label)
            fitted[label] = (th, lo)
            physicality(th[:5], label)
            np.savez(OUTDIR / f"thetas{tag}.npz",
                     **{f"theta_{k}": v[0] for k, v in fitted.items()},
                     **{f"loss_{k}": v[1] for k, v in fitted.items()})
    print(f"wrote {OUTDIR / f'thetas{tag}.npz'}")


def _cv_fold(args):
    variant, ks, fold, warm, mh_scale, maxiter = args
    _G["mh_scale"] = mh_scale
    gals = _G["gals"]
    n = len(gals)
    train = [i for i in range(n) if i % NFOLD != fold]
    r = minimize(lambda p: np.mean([gal_loss(theta_of(p, gals[i], variant),
                                             gals[i], ks) for i in train])
                 + penalty(p),
                 warm, method="Nelder-Mead",
                 options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
    return [(gals[i]["row"], *gal_eval(theta_of(r.x, gals[i], variant), gals[i], ks))
            for i in range(n) if i % NFOLD == fold]


def run_cv(dev):
    config, mh_scale, rows = _setup(dev)
    from hongshao import qa
    n = len(_G["gals"])
    tag = "_dev" if dev else ""
    thetas = np.load(OUTDIR / f"thetas{tag}.npz")
    workers = max(os.cpu_count() - 2, 2)
    maxiter = 100 if dev else 1200
    row_to_i = {g["row"]: i for i, g in enumerate(_G["gals"])}
    data = np.stack([g["data"] for g in _G["gals"]])
    logmh = np.array([g["logmh"] for g in _G["gals"]])
    res = {}
    print(f"\n  {NFOLD}-fold CV (metrics per epoch: total-norm all-R / R>5 | "
          "148-pinned shape R>5 | dlog f148):")
    for label, variant, ks, _ in JOBS:
        t0 = time.time()
        met = np.full((n, len(ks), 4), np.nan)
        cogs = np.full((n, len(ks), len(R)), np.nan)
        with Pool(min(workers, NFOLD), initializer=_init,
                  initargs=(config, mh_scale, rows)) as pool:
            jobs = [(variant, ks, f, thetas[f"theta_{label}"], mh_scale, maxiter)
                    for f in range(NFOLD)]
            for out in pool.imap_unordered(_cv_fold, jobs):
                for row_, m_, c_ in out:
                    met[row_to_i[row_]], cogs[row_to_i[row_]] = m_, c_
        res[f"met_{label}"], res[f"cogs_{label}"] = met, cogs
        line = " ".join(
            f"z{ANCHOR_Z[k]}: {100*np.nanmedian(met[:, j, 0]):.1f}/"
            f"{100*np.nanmedian(met[:, j, 1]):.1f} | {100*np.nanmedian(met[:, j, 2]):.1f} | "
            f"{np.nanmedian(np.abs(met[:, j, 3])):.3f}"
            for j, k in enumerate(ks))
        mark = f"  [mark {MARKS[label]}%]" if label in MARKS else ""
        print(f"    {label:>13s}: {line}{mark}  ({(time.time()-t0)/60:.1f} min)",
              flush=True)
        qa.evaluate(cogs, data[:, ks], R, [ANCHOR_Z[k] for k in ks],
                    name=f"exp35-{label}{tag}", figdir=FIGDIR,
                    figures=label.endswith("slope"), bin_by=logmh,
                    bin_label="logMh")
    np.savez(OUT_NPZ if not dev else OUTDIR / "total_norm_dev.npz", **res)
    print(f"wrote {OUT_NPZ if not dev else OUTDIR / 'total_norm_dev.npz'}")


# --- diagnostics ---------------------------------------------------------------------
def physicality(p, label):
    """Per-z-bin efficiency weight + visible fractions within 148 / 500 kpc."""
    g = _G["gals"][3]
    mah = g["mah"]
    s0, gexp = 10.0 ** p[0], p[1]
    sig0 = np.clip(s0 * (mah["t"] / mah["t_obs"]) ** gexp, 1e-4, 1e5)
    w = weights(mah["z"], p[3], p[4])
    dM = w * mah["dMh"]
    dM = dM / dM.sum()
    vis148 = 1.0 - np.exp(-148.0 ** 2 / (2.0 * sig0 ** 2))
    vis500 = 1.0 - np.exp(-R_TOT ** 2 / (2.0 * sig0 ** 2))
    print(f"    [{label}] theta {np.round(p, 2)}; peak z = {np.expm1(p[3]):.1f}; "
          "per z-bin (weight | vis148 | vis500):")
    for zlo, zhi in ((0.4, 1), (1, 2), (2, 4), (4, 12)):
        m = (mah["z"] >= zlo) & (mah["z"] < zhi)
        if m.sum():
            print(f"      z [{zlo},{zhi}): {dM[m].sum():.3f} | "
                  f"{np.median(vis148[m]):.3f} | {np.median(vis500[m]):.3f}")


def differential(cogs_by_gal, data, logms):
    """exp34's differential-deposition statistic: median fraction of the
    inter-snapshot growth (within 148) landing beyond 50 / 100 kpc."""
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    rows = {}
    for b in range(3):
        m = (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
        for k in range(4):
            for nm, arr in (("data", data), ("model", cogs_by_gal)):
                d = arr[m, k, :] - arr[m, k + 1, :]
                tot = d[:, -1]
                ok = tot > 0
                rows[(nm, b, k)] = (
                    np.median(1.0 - d[ok][:, I50] / tot[ok]),
                    np.median(1.0 - d[ok][:, I100] / tot[ok])) if ok.sum() else \
                    (np.nan, np.nan)
    return edges, rows


def run_report(dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    set_style()
    config, mh_scale, rows = _setup(dev)
    tag = "_dev" if dev else ""
    thetas = np.load(OUTDIR / f"thetas{tag}.npz")
    gals = _G["gals"]
    n = len(gals)
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    m500 = np.stack([g["m500"] for g in gals])

    print(f"exp35 report (n={n}{' DEV' if dev else ''})")
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6))
    a, b, c, d = axes
    cols = {"multi-global": "#56B4E9", "multi-slope": "#CC3377"}
    for label in ("multi-global", "multi-slope"):
        variant = label.split("-")[1]
        cogs = np.full((n, 5, len(R)), np.nan)
        for i, g in enumerate(gals):
            out = model_cogs_total(theta_of(thetas[f"theta_{label}"], g, variant),
                                   g["mah"], g["m500"], list(range(5)))
            if out is not None:
                cogs[i] = out
        ed3, rows_d = differential(cogs, data, logms)
        print(f"\n  differential deposition [{label}] (data -> model, f>50/f>100 "
              "per inter-snapshot pair):")
        for bidx in range(3):
            lab = f"logM* {ed3[bidx]:.2f}-{ed3[bidx+1]:.2f}"
            cells = []
            for k in range(4):
                dd, mm = rows_d[("data", bidx, k)], rows_d[("model", bidx, k)]
                cells.append(f"z{ANCHOR_Z[k+1]}->z{ANCHOR_Z[k]}: "
                             f"{dd[0]:.2f}/{dd[1]:.2f} -> {mm[0]:.2f}/{mm[1]:.2f}")
            print(f"    {lab:>22s}: " + "  ".join(cells))
        # panel A/B: massive-tercile differential curves, data vs model
        x = np.arange(4)
        if label == "multi-global":                    # measured curve: draw once
            a.plot(x, [rows_d[("data", 2, k)][0] for k in range(4)], "-o",
                   c="0.2", lw=1.8, ms=4, label="measured")
            b.plot(x, [rows_d[("data", 2, k)][1] for k in range(4)], "-o",
                   c="0.2", lw=1.8, ms=4)
        a.plot(x, [rows_d[("model", 2, k)][0] for k in range(4)], "--o",
               c=cols[label], lw=1.6, ms=4, label=f"{label} model")
        b.plot(x, [rows_d[("model", 2, k)][1] for k in range(4)], "--o",
               c=cols[label], lw=1.6, ms=4)
        # panel C: aperture fraction vs mass, model vs data (z=0.4)
        f148_m = cogs[:, 0, -1] / m500[:, 0]
        order = np.argsort(logms)
        med = [np.nanmedian(f148_m[order][max(0, i - 150):i + 150])
               for i in range(len(order))]
        c.plot(logms[order], med, "--", c=cols[label], lw=1.7, label=f"{label}")
    for ax, ttl in ((a, "fraction of growth beyond 50 kpc"),
                    (b, "fraction of growth beyond 100 kpc")):
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels([f"z{ANCHOR_Z[k+1]}$\\to$z{ANCHOR_Z[k]}" for k in range(4)],
                           fontsize=8)
        ax.set(ylabel="median fraction (massive tercile)", title=ttl)
    a.legend(fontsize=7)
    f148_d = data[:, 0, -1] / m500[:, 0]
    order = np.argsort(logms)
    med = [np.median(f148_d[order][max(0, i - 150):i + 150])
           for i in range(len(order))]
    c.plot(logms[order], med, "-", c="0.2", lw=1.7, label="data (exp34 tails)")
    c.set(xlabel="logM* (z=0.4)", ylabel="M(<148)/M(<500)",
          title="aperture fraction: the fitted datum")
    c.legend(fontsize=7)
    cv_npz = OUT_NPZ if not dev else OUTDIR / "total_norm_dev.npz"
    if cv_npz.exists():
        cv = np.load(cv_npz)
        labels = [j[0] for j in JOBS]
        shape = [100 * np.nanmedian(cv[f"met_{la}"][:, 0, 2]) for la in labels]
        d.bar(np.arange(len(labels)), shape, color="#0072B2", label="exp35 CV shape")
        for i, la in enumerate(labels):
            if la in MARKS:
                d.plot([i - 0.4, i + 0.4], [MARKS[la]] * 2, "-", c="#D55E00", lw=2)
        d.plot([], [], "-", c="#D55E00", lw=2, label="unconstrained 7p mark")
        d.set_xticks(np.arange(len(labels)))
        d.set_xticklabels(labels, rotation=30, fontsize=8)
        d.set(ylabel="held-out z=0.4 shape max|rel| R>5 [%]",
              title="vs the exp33 marks")
        d.legend(fontsize=7)
    fig.suptitle(f"exp35 — total-normalized transport refit (n={n})", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, FIGDIR / f"exp35_total_norm{tag}")[0])


def run_stress(dev):
    """Bounds-stress test: refit multi-slope with log_s0 <= 3.5 (3160 kpc) and
    g <= 6. If loss / massive-end f148 / the differential marks barely move,
    the exp35 rail is a flat plateau; if they improve, a channel is missing."""
    global HI
    HI = np.array([3.5, 6.0, 3.0, 3.0, 2.0])
    config, mh_scale, rows = _setup(dev)
    tag = "_dev" if dev else ""
    base = np.load(OUTDIR / f"thetas{tag}.npz")
    warm = base["theta_multi-slope"]
    nudge = warm.copy()
    nudge[:2] = [3.3, 5.0]                                # foothold past the old rail
    workers = max(os.cpu_count() - 2, 2)
    ks = [0, 1, 2, 3, 4]
    with Pool(workers, initializer=_init, initargs=(config, mh_scale, rows)) as pool:
        th, lo = fit_pop("slope", ks, [warm, nudge],
                         max(int(5000 * (0.06 if dev else 1.0)), 60),
                         pool, workers, "stress-multi-slope")
    physicality(th[:5], "stress-multi-slope")
    print(f"\n  loss: stress {lo:.4f} vs baseline {base['loss_multi-slope']:.4f}")
    gals = _G["gals"]
    n = len(gals)
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    m500 = np.stack([g["m500"] for g in gals])
    massive = logms >= np.quantile(logms, 2 / 3)
    print("  massive-tercile f148 (z=0.4) data "
          f"{np.median(data[massive, 0, -1] / m500[massive, 0]):.3f}; "
          "differential marks data "
          + "/".join(f"{v:.2f}" for v in differential(data, data, logms)[1][("model", 2, 0)]))
    for label, thx in (("baseline", warm), ("stress", th)):
        cogs = np.full((n, 5, len(R)), np.nan)
        for i, g in enumerate(gals):
            out = model_cogs_total(theta_of(thx, g, "slope"), g["mah"], g["m500"], ks)
            if out is not None:
                cogs[i] = out
        _, rows_d = differential(cogs, data, logms)
        print(f"    {label:>8s}: massive f148 "
              f"{np.nanmedian(cogs[massive, 0, -1] / m500[massive, 0]):.3f}; "
              "differential z0.7->z0.4 massive "
              + "/".join(f"{v:.2f}" for v in rows_d[("model", 2, 0)]))
    np.savez(OUTDIR / f"stress{tag}.npz", theta=th, loss=lo, hi=HI)
    print(f"wrote {OUTDIR / f'stress{tag}.npz'}")


def demo():
    """Self-checks: basis_ext == tf.basis on the 24-point grid; 500-pinned model
    re-pinned at 148 == exp33 physical model (same theta, same shape); synthetic
    power-law tail recovers M(<500) exactly; penalty box sane."""
    e34 = _load_by_path("exp34_run", ROOT / "experiments/exp34_asymptotic_total/run.py")
    mtot_true, loga, aexp = 1e11, np.log10(4e11), 1.2
    cog = e34.f_power(R, mtot_true, loga, aexp)
    p = e34.fit_tail(cog, "power")
    m500_true = e34.f_power(R_TOT, mtot_true, loga, aexp)
    assert abs(np.log10(e34.f_power(R_TOT, *p) / m500_true)) < 1e-3

    config, mh_scale, _ = _setup(False)
    g = _G["gals"][0]
    th4 = [2.0, 1.5, 0.0, 0.77]
    B_ref = pe.tf.basis(th4, g["mah"]["t"], g["mah"]["t_obs"], pe.AT[0], "dyntrans")
    B_new = basis_ext(th4, g["mah"]["t"], g["mah"]["t_obs"], pe.AT[0], R)
    assert np.allclose(B_ref, B_new, rtol=1e-12)

    pth = _load_by_path("exp33_physical_theta",
                        ROOT / "experiments/exp33_single_epoch/physical_theta.py")
    p5 = np.array([2.0, 1.5, 0.77, np.log(3.2), 0.5])
    ref = pth.model_cogs_phys(p5, g["mah"], g["data"], [0, 3])
    ours = model_cogs_total(p5, g["mah"], g["m500"], [0, 3])
    for c_ref, c_new, k in zip(ref, ours, [0, 3]):
        repin = c_new * (g["data"][k][-1] / c_new[-1])
        assert np.allclose(repin, c_ref, rtol=1e-10)
        assert c_new[-1] < g["m500"][k]                # strictly inside the 500 pin
    assert penalty(p5) == 0.0 and penalty(np.array([5.0, 1.5, 0.77, 1.0, 0.5])) > 1.0
    met, cogs = gal_eval(p5, g, [0])
    assert np.isfinite(met).all() and np.isfinite(cogs).all()
    print("run.demo OK: basis_ext == tf.basis; 500-pinned shape == exp33 physical "
          "model; synthetic M(<500) recovered; penalty box sane")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fit"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    else:
        {"totals": lambda d: compute_totals(), "fit": run_fit,
         "cv": run_cv, "report": run_report, "stress": run_stress}[cmd](dev)
