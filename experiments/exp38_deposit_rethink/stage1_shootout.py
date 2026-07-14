"""exp38 stage 1 — the single-epoch primitive shootout.

Two branches, both fit each epoch INDEPENDENTLY (the joint-history test is
stage 2; here a candidate must first prove it fits both the compact z=2.0
and the extended z=0.4 population without any parameter at a bound):

`deposit` — POPULATION-level fits (one shared theta per epoch, the regime
    where the Gaussian's width scale rails): additive deposit model with a
    lognormal efficiency f(z) and M(<500 kpc) total normalization (the
    exp35 convention — the aperture-horizon trap stays closed). Candidates
    swap only the per-deposit CoG shape:
        gauss    [log_s0, g, mu, sig]                  (the incumbent, 4p)
        sersic   + wing index n                        (5p; n=0.5 == gauss)
        shell    + off-centre exponent p               (5p; p=0 == gauss)
        moffat   + power-law tail index gamma          (5p; gamma > 1)
        gausswing+ mix weight w and scale ratio to a fixed n=1 wing (6p)
        empirical  the stage-0.2 stacked kernel shape, width-law scaled (4p)

`family` — PER-GALAXY direct profile-family fits per epoch (abandon
    deposition): the exp03 slope-sigmoid (5p), a Sersic CoG (3p), and the
    evolving-Re self-similar template (2p/epoch, shape frozen at the
    z=0.4 population median from stage 0.1). Judged on z=2-vs-z=0.4 fit
    parity and the HELD-EPOCH closure: fit 4 epochs, interpolate each
    parameter quadratically in z, predict the held epoch's curve.

Run: PYTHONPATH=. uv run python experiments/exp38_deposit_rethink/\
stage1_shootout.py {deposit|family|report|demo} [--dev] [--full]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from shapes import gauss_cog, moffat_cog, sersic_cog, shell_cog       # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO          # noqa: E402
from hongshao.qa import half_mass_radius, _pct                        # noqa: E402

set_style()
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E35_DIR = ROOT / "experiments/exp35_total_norm"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
ANCHOR_Z = [0.4, 0.7, 1.0, 1.5, 2.0]
DEPOSIT_CANDS = ("gauss", "sersic", "shell", "moffat", "gausswing",
                 "empirical")
# theta layout per candidate: [log_s0, g, mu, sig] + shape tail
_SHAPE_TAIL = {"gauss": [], "sersic": ["n"], "shell": ["p"],
               "moffat": ["gamma"], "gausswing": ["w", "log_ratio"],
               "empirical": []}
_TAIL_BOX = {"n": (0.25, 8.0), "p": (0.0, 8.0), "gamma": (1.05, 6.0),
             "w": (0.0, 1.0), "log_ratio": (0.0, 1.5)}
_TAIL_START = {"n": [0.7, 2.0], "p": [1.0], "gamma": [1.5],
               "w": [0.3], "log_ratio": [0.6]}
BASE_LO = np.array([0.3, 0.0, 0.0, 0.05])     # log_s0, g, mu, sig (exp35 box)
BASE_HI = np.array([3.0, 4.0, 3.0, 2.0])
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    """Worker state: exp35 loaders (gals with mah/data/m500) + the stage-0
    empirical kernels (spawn-safe: children rebuild everything)."""
    e = _load_by_path("exp35_run", E35_DIR / "run.py")
    mh_scale = tuple(np.load(ROOT / "experiments/exp32_full_population/"
                             "outputs/um_slope_diffmah.npz")["mh_scale"])
    e._init("diffmah", mh_scale, None if rows is None else np.asarray(rows))
    _W["e"] = e
    _W["gals"] = e._G["gals"]
    kern = np.load(OUTDIR / "stage0_kernels.npz")
    K = kern["kern"]
    # one unit-mass empirical CoG shape per logM* tercile: sum the four
    # epoch-pair kernels (weights = their positive mass), clip negatives
    cogs = []
    for b in range(3):
        s = np.clip(K[b], 0.0, None).sum(0)
        dA = np.pi * (e.R[1:] ** 2 - e.R[:-1] ** 2)
        m = np.concatenate([[0.0], np.cumsum(s * dA)])
        cogs.append(m / m[-1])
    _W["emp_cog"] = np.array(cogs)                # (3, 24) on e.R, unit mass
    _W["emp_edges"] = kern["edges"]
    pop = np.load(POP_NPZ)
    _W["logms"] = {g["row"]: pop["logms"][g["row"]] for g in _W["gals"]}


def _w_init_cached(rows=None):
    if "e" not in _W:
        _w_init(rows)


# --------------------------------------------------------------------------- #
# the deposit branch                                                           #
# --------------------------------------------------------------------------- #
def cand_cog(cand, dM, s, tail, Rgrid, g=None):
    """Per-candidate summed deposit CoG on Rgrid (unit total handled by
    caller's normalization)."""
    if cand == "gauss":
        return gauss_cog(dM, s, Rgrid)
    if cand == "sersic":
        return sersic_cog(dM, s, tail[0], Rgrid)
    if cand == "shell":
        return shell_cog(dM, s, tail[0], Rgrid)
    if cand == "moffat":
        return moffat_cog(dM, s, tail[0], Rgrid)
    if cand == "gausswing":
        w, ratio = tail[0], 10.0 ** tail[1]
        return ((1.0 - w) * gauss_cog(dM, s, Rgrid)
                + w * sersic_cog(dM, ratio * s, 1.0, Rgrid))
    # empirical: the tercile kernel shape, radially STRETCHED by s/S_REF
    # (S_REF ~ the measured kernel scale, so log_s0 ~ log10(S_REF) means
    # "use the kernel at its measured size"; the width law then only
    # modulates it in time)
    S_REF = 30.0
    cog_shape = _W["emp_cog"][g["_terc"]]
    e = _W["e"]
    base = np.interp(np.log10(np.clip(Rgrid[:, None] / (s[None, :] / S_REF),
                                      e.R[0], e.R[-1])),
                     np.log10(e.R), cog_shape)
    return (dM[None, :] * base).sum(1)


def _terc_of(g):
    if "_terc" not in g:
        ms = _W["logms"][g["row"]]
        edges = np.quantile(list(_W["logms"].values()), [1 / 3, 2 / 3])
        g["_terc"] = int(np.searchsorted(edges, ms))
    return g["_terc"]


def model_cog_dep(theta, cand, gal, k):
    """Single-epoch M(<500)-normalized model CoG for one galaxy."""
    e = _W["e"]
    mah = gal["mah"]
    w = e.weights(mah["z"], theta[2], theta[3])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None
    dM = dM / dM.sum()
    s = np.clip(10.0 ** theta[0] * (mah["t"] / mah["t_obs"]) ** theta[1],
                1e-4, 1e6)
    m = mah["snap"] <= e.ANCHOR_SNAP[k]
    if cand == "empirical":
        _terc_of(gal)
    cog = cand_cog(cand, dM * m, s, theta[4:], e.R_EXT, gal)
    if not np.isfinite(cog[-1]) or cog[-1] <= 0 or cog[-2] <= 0:
        return None
    return cog[:-1] * (gal["m500"][k] / cog[-1])


def dep_penalty(theta, cand):
    lo = np.concatenate([BASE_LO, [_TAIL_BOX[t][0] for t in _SHAPE_TAIL[cand]]])
    hi = np.concatenate([BASE_HI, [_TAIL_BOX[t][1] for t in _SHAPE_TAIL[cand]]])
    v = np.asarray(theta)
    return 30.0 * float(np.sum(np.clip(lo - v, 0, None) ** 2
                               + np.clip(v - hi, 0, None) ** 2))


def gal_loss_dep(theta, cand, gal, k):
    cog = model_cog_dep(theta, cand, gal, k)
    if cog is None:
        return 4.0
    d = gal["data"][k]
    return float(np.sqrt(np.mean(((cog - d) / d) ** 2)))


def _chunk_dep(args):
    theta, cand, k, lo, hi = args
    return sum(gal_loss_dep(theta, cand, g, k) for g in _W["gals"][lo:hi])


def _at_bound(theta, cand, frac=0.02):
    lo = np.concatenate([BASE_LO, [_TAIL_BOX[t][0] for t in _SHAPE_TAIL[cand]]])
    hi = np.concatenate([BASE_HI, [_TAIL_BOX[t][1] for t in _SHAPE_TAIL[cand]]])
    v = np.asarray(theta)
    width = hi - lo
    return [i for i in range(len(v))
            if v[i] < lo[i] + frac * width[i] or v[i] > hi[i] - frac * width[i]]


def cmd_deposit(dev=True):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    n = len(gals)
    workers = max(os.cpu_count() - 2, 2)
    edges = np.linspace(0, n, workers + 1).astype(int)
    tag = "_dev" if dev else ""
    maxiter = 1200 if dev else 4000
    out = {}
    print(f"exp38 stage 1 deposit shootout (n={n}{', DEV' if dev else ''}; "
          "independent per-epoch population fits, M(<500)-normalized; "
          "loss = mean per-galaxy relative RMS):")
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for cand in DEPOSIT_CANDS:
            names = ["log_s0", "g", "mu", "sig"] + _SHAPE_TAIL[cand]
            for k in range(5):
                def loss(th):
                    parts = pool.map(_chunk_dep,
                                     [(th, cand, k, edges[i], edges[i + 1])
                                      for i in range(workers)])
                    return sum(parts) / n + dep_penalty(th, cand)

                starts = []
                for base in ([2.0, 1.5, 1.3, 0.4], [2.4, 0.8, 1.0, 0.6]):
                    tails = _TAIL_START[_SHAPE_TAIL[cand][0]] if \
                        _SHAPE_TAIL[cand] else [None]
                    for t0 in tails:
                        extra = ([] if t0 is None else [t0]
                                 ) + ([0.6] if cand == "gausswing" else [])
                        starts.append(np.array(base + extra, float))
                best = None
                t0_ = time.time()
                for q0 in starts:
                    r = minimize(loss, q0, method="Nelder-Mead",
                                 options=dict(maxiter=maxiter, xatol=3e-4,
                                              fatol=1e-8))
                    if best is None or r.fun < best.fun:
                        best = r
                ab = _at_bound(best.x, cand)
                out[f"theta_{cand}_k{k}"] = best.x
                out[f"loss_{cand}_k{k}"] = best.fun
                out[f"atbound_{cand}_k{k}"] = np.array(ab, int)
                flag = ("" if not ab else
                        " AT BOUND: " + ",".join(names[i] for i in ab))
                print(f"  {cand:>10s} z={ANCHOR_Z[k]}: loss {best.fun:.4f}  "
                      f"theta {np.round(best.x, 2)}{flag}  "
                      f"({(time.time()-t0_)/60:.1f} min)", flush=True)
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"stage1_deposit{tag}.npz", **out)
    print(f"wrote {OUTDIR / f'stage1_deposit{tag}.npz'}")


# --------------------------------------------------------------------------- #
# the profile-family branch                                                    #
# --------------------------------------------------------------------------- #
def _fit_sersic_cog(Rg, logC):
    def resid(q):
        mod = sersic_cog(np.array([10.0 ** q[0]]), np.array([10.0 ** q[1]]),
                         np.clip(q[2], 0.3, 10.0), Rg)
        return np.log10(np.clip(mod, 1.0, None)) - logC
    r = least_squares(resid, [logC[-1] + 0.05, 1.0, 2.0],
                      bounds=([8.0, -0.5, 0.3], [14.0, 3.0, 10.0]))
    return r.x, resid(r.x)


def _fit_template(Rg, logC, template_x, template_y):
    def resid(q):
        x = Rg / (10.0 ** q[1])
        shape = np.interp(np.log10(np.clip(x, template_x[0], template_x[-1])),
                          np.log10(template_x), template_y)
        return q[0] + shape - logC
    r = least_squares(resid, [logC[-1], np.log10(10.0)],
                      bounds=([8.0, -0.5], [14.0, 2.5]))
    return r.x, resid(r.x)


def _quad_closure(zs, params, curves_fn, logC_all, hold):
    """Fit each parameter quadratically in z on the 4 non-held epochs,
    predict the held epoch's curve; return max|rel| (R>5) of the prediction."""
    ks = [k for k in range(5) if k != hold]
    pred = np.array([np.polyval(np.polyfit([zs[k] for k in ks],
                                           [params[k][j] for k in ks], 2),
                                zs[hold])
                     for j in range(len(params[0]))])
    mod = curves_fn(pred)
    d = 10.0 ** logC_all[hold]
    rel = np.abs((mod - d) / d)
    return float(rel[_W["Rmask"]].max())


def cmd_family(dev=True):
    from hongshao.profiles import fit_cog, cog_from_physical
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init_cached(rows)
    e = _W["e"]
    gals = _W["gals"]
    Rg = e.R
    _W["Rmask"] = Rg > 5.0
    tag = "_dev" if dev else ""

    # the 1j template: population-median z=0.4 shape in x = R/R_half
    xg = np.geomspace(0.15, 40.0, 30)
    shapes_ = []
    for g in gals:
        d0 = g["data"][0]
        rh = half_mass_radius(d0, Rg)
        y = np.log10(d0) - np.log10(d0[-1])
        shapes_.append(np.interp(np.log10(np.clip(xg * rh, Rg[0], Rg[-1])),
                                 np.log10(Rg), y))
    template_y = np.median(np.stack(shapes_), axis=0)

    fams = ("sigmoid", "sersiccog", "template")
    met = {f: np.full((len(gals), 5), np.nan) for f in fams}    # max|rel| R>5
    closure = {f: np.full((len(gals), 5), np.nan) for f in fams}
    pars = {f: {} for f in fams}
    t0 = time.time()
    print(f"exp38 stage 1 family shootout (n={len(gals)}"
          f"{', DEV' if dev else ''}; per-galaxy per-epoch fits + held-epoch "
          "quadratic-in-z closure):")
    for i, g in enumerate(gals):
        logC_all = [np.log10(g["data"][k]) for k in range(5)]
        fitted = {f: [] for f in fams}
        for k in range(5):
            logC = logC_all[k]
            d = fit_cog(Rg, logC, r_min=5.0)
            q_sig = [d["logMstar0"], d["beta_in"], d["beta_out"],
                     np.log10(d["R_c"]), np.log10(d["Delta"])]
            mod = 10.0 ** cog_from_physical(Rg[Rg >= 5.0], d["logMstar0"],
                                            d["beta_in"], d["beta_out"],
                                            d["R_c"], d["Delta"])
            rel = np.abs((mod - g["data"][k][Rg >= 5.0])
                         / g["data"][k][Rg >= 5.0])
            met["sigmoid"][i, k] = rel.max()
            fitted["sigmoid"].append(q_sig)
            q_ser, res = _fit_sersic_cog(Rg, logC)
            met["sersiccog"][i, k] = float(
                np.abs(10.0 ** res[_W["Rmask"]] - 1.0).max())
            fitted["sersiccog"].append(list(q_ser))
            q_tpl, res = _fit_template(Rg, logC, xg, template_y)
            met["template"][i, k] = float(
                np.abs(10.0 ** res[_W["Rmask"]] - 1.0).max())
            fitted["template"].append(list(q_tpl))
        pars_i = {f: np.array(fitted[f]) for f in fams}
        for f in fams:
            pars[f][g["row"]] = pars_i[f]

        def curves(fam, q):
            if fam == "sigmoid":
                full = 10.0 ** cog_from_physical(Rg, q[0], q[1], q[2],
                                                 10.0 ** q[3], 10.0 ** q[4])
                return full
            if fam == "sersiccog":
                return sersic_cog(np.array([10.0 ** q[0]]),
                                  np.array([10.0 ** q[1]]),
                                  float(np.clip(q[2], 0.3, 10.0)), Rg)
            x = Rg / (10.0 ** q[1])
            shape = np.interp(np.log10(np.clip(x, xg[0], xg[-1])),
                              np.log10(xg), template_y)
            return 10.0 ** (q[0] + shape)

        for f in fams:
            for hold in range(5):
                closure[f][i, hold] = _quad_closure(
                    ANCHOR_Z, list(pars_i[f]),
                    lambda q, f=f: curves(f, q), logC_all, hold)
    print(f"  ({(time.time()-t0)/60:.1f} min)")
    print("\n  per-epoch DIRECT fit, median max|rel| R>5 "
          f"[{_pct()}] (parity z=2 vs z=0.4 is the gate):")
    for f in fams:
        line = " ".join(f"z{ANCHOR_Z[k]}: {100*np.nanmedian(met[f][:, k]):5.1f}"
                        for k in range(5))
        parity = (np.nanmedian(met[f][:, 4]) / np.nanmedian(met[f][:, 0]))
        print(f"    {f:>9s}: {line}   z2/z04 ratio {parity:.2f}")
    print("\n  HELD-EPOCH closure (fit 4 epochs, quadratic-in-z params, "
          f"predict the 5th), median max|rel| R>5 [{_pct()}]:")
    for f in fams:
        line = " ".join(f"z{ANCHOR_Z[k]}: "
                        f"{100*np.nanmedian(closure[f][:, k]):5.1f}"
                        for k in range(5))
        print(f"    {f:>9s}: {line}")
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"stage1_family{tag}.npz",
             **{f"met_{f}": met[f] for f in fams},
             **{f"closure_{f}": closure[f] for f in fams},
             template_x=xg, template_y=template_y)
    print(f"wrote {OUTDIR / f'stage1_family{tag}.npz'}")


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #
def cmd_report(dev=True):
    tag = "_dev" if dev else ""
    dd = np.load(OUTDIR / f"stage1_deposit{tag}.npz")
    names_of = lambda c: ["log_s0", "g", "mu", "sig"] + _SHAPE_TAIL[c]  # noqa
    print("exp38 stage 1 verdict — deposit branch (per-epoch population "
          "loss = mean per-galaxy relative RMS; smaller is better; the "
          "incumbent is gauss):")
    print("  candidate  | " + " ".join(f"z={z:<4}" for z in ANCHOR_Z)
          + " | bounds hit (epochs)")
    for cand in DEPOSIT_CANDS:
        if f"loss_{cand}_k0" not in dd.files:
            continue
        losses = [float(dd[f"loss_{cand}_k{k}"]) for k in range(5)]
        hits = []
        for k in range(5):
            ab = dd[f"atbound_{cand}_k{k}"]
            if len(ab):
                hits.append(f"z{ANCHOR_Z[k]}:"
                            + ",".join(names_of(cand)[i] for i in ab))
        print(f"  {cand:>10s} | " + " ".join(f"{v:.4f}"[:6] for v in losses)
              + " | " + ("; ".join(hits) if hits else "none"))

    # parameter tracks vs z (smoothness read for the survivors)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for cand, col in zip(DEPOSIT_CANDS, OKABE_ITO):
        if f"theta_{cand}_k0" not in dd.files:
            continue
        th = np.stack([dd[f"theta_{cand}_k{k}"][:4] for k in range(5)])
        axes[0].plot(ANCHOR_Z, th[:, 0], "-o", ms=4, c=col, label=cand)
        axes[1].plot(ANCHOR_Z, th[:, 1], "-o", ms=4, c=col)
        if _SHAPE_TAIL[cand]:
            sh = [dd[f"theta_{cand}_k{k}"][4] for k in range(5)]
            axes[2].plot(ANCHOR_Z, sh, "-o", ms=4, c=col, label=cand)
    axes[0].axhline(3.0, c="k", ls=":", lw=1)
    axes[0].set(xlabel="epoch z", ylabel="fitted log s0 (bound at 3.0 dotted)",
                title="width scale vs epoch")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="epoch z", ylabel="fitted g",
                title="width time-exponent vs epoch")
    axes[2].set(xlabel="epoch z", ylabel="fitted shape parameter",
                title="wing/shell/tail parameter vs epoch")
    axes[2].legend(fontsize=7)
    fig.suptitle("exp38 stage 1 — independently fitted per-epoch parameters "
                 "(smooth tracks = joint-fit-ready)", fontsize=12)
    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    print("wrote", save_fig(fig, FIGDIR / f"stage1_deposit_tracks{tag}")[0])


def demo():
    # candidate CoGs: nesting + finite behavior through the model wrapper
    rows = np.load(POP_NPZ)["dev100"][:8]
    _w_init(rows)
    g = _W["gals"][0]
    th_g = np.array([2.0, 1.5, 1.3, 0.4])
    a = model_cog_dep(th_g, "gauss", g, 0)
    b = model_cog_dep(np.append(th_g + [0.5 * np.log10(2), 0, 0, 0], 0.5),
                      "sersic", g, 0)
    assert a is not None and np.abs(a / b - 1).max() < 1e-10, \
        "sersic n=0.5 must nest gauss through the model wrapper"
    c = model_cog_dep(np.append(th_g, 0.0), "shell", g, 0)
    assert np.abs(a / c - 1).max() < 1e-12, "shell p=0 must nest gauss"
    for cand, tail in (("moffat", [1.5]), ("gausswing", [0.4, 0.6]),
                       ("empirical", [])):
        m = model_cog_dep(np.concatenate([th_g, tail]), cand, g, 4)
        assert m is not None and np.isfinite(m).all() and m[-1] > 0, cand
        assert abs(m[-1] / g["m500"][4] - 1) < 0.35, (cand, "normalization")
    assert dep_penalty(np.append(th_g, 9.0), "sersic") > 1.0
    assert dep_penalty(np.append(th_g, 2.0), "sersic") == 0.0
    assert _at_bound(np.array([3.0, 1.5, 1.3, 0.4]), "gauss") == [0]
    # family fitters: recover a synthetic Sersic CoG
    Rg = _W["e"].R
    truth = sersic_cog(np.array([10.0 ** 11.2]), np.array([12.0]), 3.0, Rg)
    q, res = _fit_sersic_cog(Rg, np.log10(truth))
    assert abs(q[2] - 3.0) < 0.15 and np.abs(res).max() < 1e-3
    print("stage1 demo OK: nesting exact through the model wrapper; every "
          "candidate finite and M(<500)-normalized; penalty/at-bound sane; "
          "the Sersic-CoG family fitter recovers synthetic truth")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--full" not in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "deposit":
        cmd_deposit(dev)
    elif cmd == "family":
        cmd_family(dev)
    elif cmd == "report":
        cmd_report(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
