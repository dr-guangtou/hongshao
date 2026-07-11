"""exp32 step 2 — the mass structure of the universal theta (full population).

Subcommands (run independently, warm-started so they parallelize as processes):
  global  --config C   one universal theta on all n (warm: exp31 n=45 fit, phys)
  bins    --config C   one theta per logMh quartile (halo-only bin assignment)
  slope   --config C   theta + linear slopes on z-scored logMh (continuous variant)
  anatomy --config C   free ONE component per galaxy around the global theta:
                       which direction carries the per-galaxy individuality?
  cv      --config C   10-fold CV (mass-stratified folds) for global/bins/slope —
                       the held-out comparison that picks the population model
  report  --config C   comparison table + figure from the cached outputs

Loss everywhere = mean over epochs of per-epoch rel-RMS over ALL radii (the
phase-3 convention); evaluation adds median profile max|rel|.

Run: PYTHONPATH=. uv run python experiments/exp32_full_population/universal_mass.py \
       {global|bins|slope|anatomy|cv|report} [--config real|diffmah] [--dev]
Demo: ... universal_mass.py demo
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
EXP30 = ROOT / "experiments" / "exp30_transport_kernel"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
sys.path.insert(0, str(EXP30))
import param_emulator as pe                                                          # noqa: E402
from run import dipfree_mah                                                          # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402

OUTDIR = HERE / "outputs"
POP_NPZ = OUTDIR / "population.npz"
MAHFUN = {"real": real_mah, "diffmah": dipfree_mah}
PHYS = np.array([2.0, 1.5, 0.0, 0.77, 4.48, 1.88, 2.23])
NBINS, NFOLD = 4, 10
_G = {}


def load_gals(config, rows=None):
    pop = np.load(POP_NPZ)
    rows = np.arange(len(pop["index"])) if rows is None else np.asarray(rows)
    gals = []
    for r in rows:
        mah = MAHFUN[config](int(pop["index"][r]))
        if mah is None:
            continue
        gals.append(dict(row=int(r), mah=mah, data=pop["data"][r],
                         logmh=float(pop["logmh"][r]), logms=float(pop["logms"][r])))
    return gals


def gal_loss(theta, g):
    cogs, _ = pe.model_cogs(theta, g["mah"], g["data"])
    if cogs is None:
        return 4.0
    D = g["data"]
    return float(np.mean(np.sqrt(np.mean(((cogs - D) / D) ** 2, axis=1))))


def gal_maxrel(theta, g):
    cogs, _ = pe.model_cogs(theta, g["mah"], g["data"])
    if cogs is None:
        return np.full(5, np.nan)
    D = g["data"]
    return np.abs((cogs - D) / D).max(axis=1)


def gal_pred(theta, g):
    """(pinned cogs (5,24), maxrel (5,)) for one galaxy; NaNs if invalid."""
    cogs, _ = pe.model_cogs(theta, g["mah"], g["data"])
    if cogs is None:
        return np.full((5, 24), np.nan), np.full(5, np.nan)
    D = g["data"]
    return cogs, np.abs((cogs - D) / D).max(axis=1)


def _chunk_loss(args):
    theta_of_key, p, lo, hi = args
    gals = _G["gals"][lo:hi]
    return sum(gal_loss(THETA_OF[theta_of_key](p, g), g) for g in gals)


def pool_loss(theta_of_key, p, pool, nchunk):
    n = len(_G["gals"])
    edges = np.linspace(0, n, nchunk + 1).astype(int)
    parts = pool.map(_chunk_loss, [(theta_of_key, p, edges[i], edges[i + 1])
                                   for i in range(nchunk)])
    return sum(parts) / n


def _init_workers(config, rows):
    _G["gals"] = load_gals(config, rows)


# --- theta parameterizations ----------------------------------------------------
def theta_global(p, g):
    return np.asarray(p[:7])


def theta_slope(p, g):
    return np.asarray(p[:7]) + np.asarray(p[7:14]) * _zmh(g["logmh"])


def _zmh(logmh):
    mu, sd = _G["mh_scale"]
    return (logmh - mu) / sd


THETA_OF = {"global": theta_global, "slope": theta_slope}


def fit_pop(theta_of_key, starts, maxiter, pool, nchunk, label=""):
    best = None
    for p0 in starts:
        t0 = time.time()
        r = minimize(lambda p: pool_loss(theta_of_key, p, pool, nchunk), p0,
                     method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
        print(f"  [{label}] start done: loss {r.fun:.4f} ({(time.time()-t0)/60:.1f} min, "
              f"{r.nit} iters)", flush=True)
        if best is None or r.fun < best.fun:
            best = r
    return best.x, best.fun


def warm_starts(config):
    d4 = np.load(EXP30 / "outputs" / "pop_forward.npz")
    return [d4[config + "_univ"], PHYS]


def mh_bins(logmh_all):
    return np.quantile(logmh_all, np.linspace(0, 1, NBINS + 1))


# --- subcommands ------------------------------------------------------------------
def run_global(config, dev):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _init_workers(config, rows)
    workers = max(os.cpu_count() - 2, 2)
    # single warm start (exp31 n=45 stratified fit): the dev run showed the PHYS
    # start converges to the same loss (0.1561 vs 0.1556) at 2x the cost
    with Pool(workers, initializer=_init_workers, initargs=(config, rows)) as pool:
        th, loss = fit_pop("global", warm_starts(config)[:1], 4000, pool, workers)
    mr = np.array([gal_maxrel(th, g) for g in _G["gals"]])
    tag = "_dev" if dev else ""
    np.savez(OUTDIR / f"um_global_{config}{tag}.npz", theta=th, loss=loss, maxrel=mr,
             rows=[g["row"] for g in _G["gals"]])
    print(f"global [{config}{tag}]: loss {loss:.4f}, in-sample epoch-avg max|rel| "
          f"{100*np.nanmean(np.nanmedian(mr, 0)):.1f}%  theta={np.round(th, 2)}")


def run_bins(config, dev):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _init_workers(config, rows)
    logmh = np.array([g["logmh"] for g in _G["gals"]])
    edges = mh_bins(logmh)
    workers = max(os.cpu_count() - 2, 2)
    thetas, losses = [], []
    all_rows = np.array([g["row"] for g in _G["gals"]])
    for b in range(NBINS):
        m = (logmh >= edges[b]) & (logmh <= edges[b + 1] + 1e-9)
        sub = all_rows[m]
        with Pool(workers, initializer=_init_workers, initargs=(config, sub)) as pool:
            _G["gals"] = load_gals(config, sub)
            th, lo = fit_pop("global", warm_starts(config)[:1], 3000, pool, workers,
                             label=f"bin{b}")
        thetas.append(th)
        losses.append(lo)
        _init_workers(config, rows)
    tag = "_dev" if dev else ""
    np.savez(OUTDIR / f"um_bins_{config}{tag}.npz", theta=np.array(thetas),
             loss=np.array(losses), edges=edges)
    print(f"bins [{config}{tag}]: losses {np.round(losses, 4)}, logMh edges "
          f"{np.round(edges, 2)}")
    print("  theta per bin:\n", np.round(np.array(thetas), 2))


def run_slope(config, dev):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _init_workers(config, rows)
    logmh = np.array([g["logmh"] for g in _G["gals"]])
    _G["mh_scale"] = (float(logmh.mean()), float(logmh.std() + 1e-9))
    g0 = np.load(OUTDIR / f"um_global_{config}{'_dev' if dev else ''}.npz")["theta"]
    workers = max(os.cpu_count() - 2, 2)
    with Pool(workers, initializer=_init_slope, initargs=(config, rows, _G["mh_scale"])) \
            as pool:
        p, loss = fit_pop("slope", [np.concatenate([g0, np.zeros(7)])], 6000, pool,
                          workers, label="slope")
    tag = "_dev" if dev else ""
    np.savez(OUTDIR / f"um_slope_{config}{tag}.npz", p=p, loss=loss,
             mh_scale=_G["mh_scale"])
    print(f"slope [{config}{tag}]: loss {loss:.4f}")
    print(f"  base {np.round(p[:7], 2)}\n  dtheta/dz(logMh) {np.round(p[7:], 3)}")


def _init_slope(config, rows, mh_scale):
    _init_workers(config, rows)
    _G["mh_scale"] = mh_scale


def _anatomy_one(args):
    row_i, = args
    g = _G["gals"][row_i]
    th0 = _G["th_global"]
    l_glob = gal_loss(th0, g)
    l_free = np.empty(7)
    for j in range(7):
        def f(x):
            th = th0.copy()
            th[j] = x
            return gal_loss(th, g)
        span = 3.0 * max(1.0, abs(th0[j]))
        r = minimize_scalar(f, bounds=(th0[j] - span, th0[j] + span),
                            method="bounded", options=dict(maxiter=80))
        l_free[j] = min(r.fun, l_glob)
    return row_i, l_glob, l_free


def run_anatomy(config, dev):
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    th0 = np.load(OUTDIR / f"um_global_{config}{tag}.npz")["theta"]
    atlas = np.load(OUTDIR / f"theta_atlas_{config}{'_dev' if dev else ''}.npz")
    workers = max(os.cpu_count() - 2, 2)
    _init_workers(config, rows)
    n = len(_G["gals"])
    l_glob = np.full(n, np.nan)
    l_free = np.full((n, 7), np.nan)
    with Pool(workers, initializer=_init_anatomy, initargs=(config, rows, th0)) as pool:
        for row_i, lg, lf in pool.imap_unordered(_anatomy_one,
                                                 [(i,) for i in range(n)], chunksize=8):
            l_glob[row_i], l_free[row_i] = lg, lf
    # per-galaxy floor loss from the atlas thetas (same loss function)
    rows_arr = np.array([g["row"] for g in _G["gals"]])
    arow = {int(r): i for i, r in enumerate(atlas["rows"])}
    l_floor = np.array([gal_loss(atlas["theta"][arow[int(r)]], _G["gals"][i])
                        if int(r) in arow else np.nan
                        for i, r in enumerate(rows_arr)])
    np.savez(OUTDIR / f"um_anatomy_{config}{tag}.npz", rows=rows_arr,
             loss_global=l_glob, loss_free=l_free, loss_floor=l_floor)
    gap = l_glob - l_floor
    closed = (l_glob[:, None] - l_free) / np.clip(gap[:, None], 1e-9, None)
    med = np.nanmedian(closed, axis=0)
    print(f"anatomy [{config}{tag}]: median gap-closure by freeing ONE component:")
    for j, nm in enumerate(pe_names()):
        print(f"    {nm:>10s}: {100*med[j]:5.1f}%")


def _init_anatomy(config, rows, th0):
    _init_workers(config, rows)
    _G["th_global"] = th0


def pe_names():
    return ["log_s0", "g", "log_alpha", "q", "b_early", "b_late", "z_c"]


def _cv_fold(args):
    variant, fold, config, rows_all, warm, edges, mh_scale = args
    _init_workers(config, rows_all)
    _G["mh_scale"] = mh_scale
    gals = _G["gals"]
    n = len(gals)
    test = [i for i in range(n) if i % NFOLD == fold]
    train = [i for i in range(n) if i % NFOLD != fold]
    out = []
    if variant == "bins":
        logmh = np.array([g["logmh"] for g in gals])
        for b in range(NBINS):
            tr = [i for i in train if edges[b] <= logmh[i] <= edges[b + 1] + 1e-9]
            te = [i for i in test if edges[b] <= logmh[i] <= edges[b + 1] + 1e-9]
            if not te:
                continue
            r = minimize(lambda p: np.mean([gal_loss(p, gals[i]) for i in tr]),
                         warm[b], method="Nelder-Mead",
                         options=dict(maxiter=1200, xatol=3e-4, fatol=1e-8))
            out += [(gals[i]["row"], *gal_pred(r.x, gals[i])) for i in te]
    else:
        key = "global" if variant == "global" else "slope"
        r = minimize(lambda p: np.mean([gal_loss(THETA_OF[key](p, gals[i]), gals[i])
                                        for i in train]),
                     warm, method="Nelder-Mead",
                     options=dict(maxiter=1200, xatol=3e-4, fatol=1e-8))
        out = [(gals[i]["row"], *gal_pred(THETA_OF[key](r.x, gals[i]), gals[i]))
               for i in test]
    return out


def run_cv(config, dev):
    tag = "_dev" if dev else ""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    pop = np.load(POP_NPZ)
    nall = len(pop["dev100"]) if dev else len(pop["index"])
    variants = {}
    g = np.load(OUTDIR / f"um_global_{config}{tag}.npz")
    variants["global"] = (g["theta"], None)
    b = OUTDIR / f"um_bins_{config}{tag}.npz"
    if b.exists():
        db = np.load(b)
        variants["bins"] = (db["theta"], db["edges"])
    s = OUTDIR / f"um_slope_{config}{tag}.npz"
    mh_scale = (0.0, 1.0)
    if s.exists():
        ds = np.load(s)
        variants["slope"] = (ds["p"], None)
        mh_scale = tuple(ds["mh_scale"])
    workers = max(os.cpu_count() - 2, 2)
    res = {}
    for variant, (warm, edges) in variants.items():
        t0 = time.time()
        jobs = [(variant, f, config, rows, warm, edges, mh_scale) for f in range(NFOLD)]
        mr = np.full((nall, 5), np.nan)
        cogs = np.full((nall, 5, 24), np.nan)
        with Pool(min(workers, NFOLD)) as pool:
            for out in pool.imap_unordered(_cv_fold, jobs):
                for row, cg, m in out:
                    ridx = row if not dev else int(np.where(pop["dev100"] == row)[0][0])
                    mr[ridx], cogs[ridx] = m, cg
        res["mr_" + variant] = mr
        res["cogs_" + variant] = cogs
        print(f"  cv {variant}: held-out epoch-avg max|rel| "
              f"{100*np.nanmean(np.nanmedian(mr, 0)):.1f}%  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    np.savez(OUTDIR / f"um_cv_{config}{tag}.npz", **res)
    print("wrote", OUTDIR / f"um_cv_{config}{tag}.npz")


def run_report(config, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    set_style()
    tag = "_dev" if dev else ""
    pop = np.load(POP_NPZ)
    rows = pop["dev100"] if dev else np.arange(len(pop["index"]))
    logms, logmh = pop["logms"][rows], pop["logmh"][rows]
    atlas = np.load(OUTDIR / f"theta_atlas_{config}{'_dev' if dev else ''}.npz")
    cv = np.load(OUTDIR / f"um_cv_{config}{tag}.npz")
    edges_ms = np.quantile(logms, np.linspace(0, 1, 5))
    print(f"\n=== exp32 step 2 [{config}{tag}] — held-out (10-fold CV) epoch-avg "
          "max|rel| by logM* quartile ===")
    floor = atlas["maxrel"].mean(1)
    rowsline = {"floor (per-galaxy)": floor}
    for k in ("mr_global", "mr_bins", "mr_slope"):
        if k in cv:
            rowsline[k[3:]] = np.nanmean(cv[k], axis=1)
    print(f"    {'model':>18s} | " + " | ".join(
        f"M*Q{q+1}".rjust(6) for q in range(4)) + " |    all")
    for nm, v in rowsline.items():
        cells = []
        for q in range(4):
            m = (logms >= edges_ms[q]) & (logms <= edges_ms[q + 1] + 1e-9)
            cells.append(f"{100*np.nanmedian(v[m]):5.1f}%")
        print(f"    {nm:>18s} | " + " | ".join(cells)
              + f" | {100*np.nanmedian(v):5.1f}%")
    # figure: floor + cv curves vs logms + anatomy summary
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    a, bx, c = axes
    order = np.argsort(logms)
    for nm, v in rowsline.items():
        med = [100 * np.nanmedian(v[order][max(0, i - 100):i + 100])
               for i in range(0, len(order))]
        a.plot(logms[order], med, lw=1.8, label=nm)
    a.set(xlabel="logM* (z=0.4)", ylabel="running-median epoch-avg max|rel| [%]",
          title="A. Held-out error vs stellar mass")
    a.legend(fontsize=8)
    bins = np.load(OUTDIR / f"um_bins_{config}{tag}.npz")
    for j, nm in enumerate(pe_names()):
        bx.plot(0.5 * (bins["edges"][:-1] + bins["edges"][1:]),
                bins["theta"][:, j], "o-", label=nm)
    bx.set(xlabel="logMh bin center", ylabel="fitted component",
           title="B. Universal theta vs halo mass (quartile fits)")
    bx.legend(fontsize=7)
    an = OUTDIR / f"um_anatomy_{config}{tag}.npz"
    if an.exists():
        d = np.load(an)
        gap = d["loss_global"] - d["loss_floor"]
        closed = (d["loss_global"][:, None] - d["loss_free"]) / \
            np.clip(gap[:, None], 1e-9, None)
        med = np.nanmedian(closed, axis=0)
        c.bar(range(7), 100 * med, color="#0072B2")
        c.set_xticks(range(7))
        c.set_xticklabels(pe_names(), rotation=45, fontsize=8)
        c.set(ylabel="median gap closed by ONE free component [%]",
              title="C. Anatomy: where does the individuality live?")
    fig.suptitle(f"exp32 step 2 [{config}] — mass structure of the universal theta "
                 f"(n={len(rows)})", fontsize=12)
    fig.tight_layout()
    print("wrote", save_fig(fig, HERE / "figures" / f"exp32_universal_mass_{config}")[0])


def demo():
    """Dev-scale self-check of the machinery: theta_slope identity at zero slope;
    pool_loss == serial loss; one anatomy galaxy closes a non-negative gap."""
    _init_workers("diffmah", np.load(POP_NPZ)["dev100"][:8])
    _G["mh_scale"] = (13.5, 0.4)
    g = _G["gals"][0]
    assert np.allclose(theta_slope(np.concatenate([PHYS, np.zeros(7)]), g), PHYS)
    serial = np.mean([gal_loss(PHYS, gg) for gg in _G["gals"]])
    with Pool(2, initializer=_init_workers, initargs=("diffmah",
                                                      np.load(POP_NPZ)["dev100"][:8])) as pool:
        par = pool_loss("global", PHYS, pool, 2)
    assert abs(serial - par) < 1e-12, (serial, par)
    _G["th_global"] = PHYS
    _, lg, lf = _anatomy_one((0,))
    assert (lf <= lg + 1e-12).all(), "freeing a component can never hurt"
    print(f"universal_mass.demo OK: slope identity, pool==serial ({serial:.4f}), "
          f"anatomy sane (global {lg:.3f} -> best single {lf.min():.3f})")


if __name__ == "__main__":
    cmd = sys.argv[1]
    config = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv \
        else "diffmah"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    else:
        {"global": run_global, "bins": run_bins, "slope": run_slope,
         "anatomy": run_anatomy, "cv": run_cv, "report": run_report}[cmd](config, dev)
