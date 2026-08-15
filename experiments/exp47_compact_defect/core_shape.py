"""exp47 step 2 — a core channel that knows WHICH galaxies need it.

Step 1 found that every parked component is a LOCATION fix for a SPREAD
problem: the adopted kernel under-fills compact centres by 30-44% and
over-fills extended centres by up to +21% at z=2, and the exp38 core
channel shifts both sub-populations by a similar amount, leaving the gap
at 32-33 points while failing the differential gate.

The reason is visible in the parameterization. exp38's core amplitude is

    f_core = expit(p12 + p13 * cond[0])          # cond[0] = logMh

— conditioned on halo MASS alone, so it cannot distinguish compact from
extended galaxies at fixed mass. And exp46 measured mass to be exactly
the axis that does NOT separate them (their z=2 stellar masses are
identical by construction of the matched control).

This step gives the amplitude the information stage 1b identified as the
missing ingredient — **DiffMAH shape** — and asks two questions:

  A. CAPACITY: with a plain (shape-only) loss, does the optimizer USE the
     freedom to differentiate, or does it collapse the shape slopes to
     zero because compact galaxies are a minority of the objective?
  B. OBJECTIVE: with a sub-population BALANCED loss (compact and extended
     weighted to contribute equally), does it then differentiate?

A null in A with a signal in B is the direct confirmation of proposal 3:
the model always had the capacity, and the objective was the obstacle.

FALSIFIABLE PREDICTION, fixed before fitting. exp38 diagnosed the physics
break as "the OUTER kernel re-balancing (wider late deposits) whenever any
core channel absorbs the inner mass". If the core is applied mainly to
compact galaxies — 6.3% of the sample at z=0.4 — that pressure should be
far smaller and **the differential gate should survive where the uniform
core channel failed (0.48/0.16)**. If it fails anyway, the tension is
structural and exp38's original reading was right.

Theta = the 12 stage-2 parameters + [ca, cb(logMh), cs1, cs2, cs3
(DiffMAH shape slopes), log10 rc_core]. Setting the three shape slopes to
zero nests exp38's core channel exactly.

Run: PYTHONPATH=. uv run python experiments/exp47_compact_defect/\
core_shape.py {demo|fit|report} [--dev]
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp46_highz_ridge"))

OUTDIR = HERE / "outputs"
E38 = ROOT / "experiments/exp38_deposit_rethink"
E38D = E38 / "outputs"
E40 = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
DMAH_CSV = (ROOT / "experiments/exp10_diffmah_fit/outputs/"
            "diffmah_params_tmin1gyr.csv")

KS = [0, 1, 2, 3]                 # the OFFICIAL z<=1.5 fit scope
KS_ALL = [0, 1, 2, 3, 4]
R_COMPACT = 5.0
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    """exp38 worker state plus a standardized DiffMAH SHAPE vector and the
    compact mask that the balanced loss needs."""
    s3 = _load_by_path("exp38_stage3", E38 / "stage3_inner.py")
    s3._w_init(rows)
    _W.update(s3._W)
    _W["s3"], _W["s2"] = s3, s3._W["s2"]
    _W["rows_arg"] = rows

    pop = np.load(POP_NPZ)
    raw = np.genfromtxt(DMAH_CSV, delimiter=",", names=True)
    lut = {int(i): k for k, i in enumerate(raw["index"])}
    cols = ("dmah_logtc", "dmah_early", "dmah_late")
    S = np.full((len(pop["index"]), 3), np.nan)
    for k, idx in enumerate(pop["index"]):
        j = lut.get(int(idx))
        if j is not None:
            S[k] = [raw[c][j] for c in cols]
    mu, sd = np.nanmean(S, 0), np.nanstd(S, 0)
    Z = np.where(np.isfinite(S), (S - mu) / sd, 0.0)
    for g in _W["gals"]:
        g["shape"] = Z[g["row"]]

    # compact mask at the ANCHOR epoch, from the truth CoG — the same
    # boundary exp46/exp47 use everywhere else
    from hongshao import qa
    data = np.stack([g["data"] for g in _W["gals"]])
    r50 = qa._safe_rhalf(data, _W["e"].R, 0.5)
    for i, g in enumerate(_W["gals"]):
        g["compact"] = bool(r50[i, 0] < R_COMPACT)

    # The balanced weights MUST be built here, not in the parent: Pool
    # workers are re-initialized through this function and would otherwise
    # KeyError on them (or, worse under a fork start method, silently use
    # a stale value).
    n = len(_W["gals"])
    nc = sum(g["compact"] for g in _W["gals"])
    _W["n_compact"] = nc
    _W["w_compact"] = 0.5 * n / max(nc, 1)
    _W["w_extended"] = 0.5 * n / max(n - nc, 1)


def model_cogs_coreshape(p, g, ks, variant="1ch-mof"):
    """exp38's core channel with f_core conditioned on DiffMAH SHAPE too.

    p[:12] stage-2 base; p[12]=ca, p[13]=cb (logMh), p[14:17]=shape slopes,
    p[17]=log10 rc_core. Shape slopes = 0 nests exp38's core channel.
    """
    e, s2 = _W["e"], _W["s2"]
    mah = g["mah"]
    base6, _, _ = s2.theta_of(p[:12], g, "1ch-mof")
    f_core = float(expit(p[12] + p[13] * g["cond"][0]
                         + float(np.asarray(p[14:17]) @ g["shape"])))
    rc_core = 10.0 ** float(np.clip(p[17], -0.3, 0.9))
    w = e.weights(mah["z"], base6[3], base6[4])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None
    dM = dM / dM.sum()
    gam = float(np.clip(base6[5], 1.06, 6.0))
    th4 = [base6[0], base6[1], 0.0, base6[2]]
    cog_core = 1.0 - (1.0 + (e.R_EXT / rc_core) ** 2) ** (1.0 - gam)
    out = []
    for k in ks:
        mask = mah["snap"] <= e.ANCHOR_SNAP[k]
        B = s2.basis_mof(th4, gam, mah["t"], mah["t_obs"], e.pe.AT[k],
                         e.R_EXT)
        m = ((1.0 - f_core) * (B @ (dM * mask))
             + f_core * cog_core * (dM * mask).sum())
        if not np.isfinite(m[-1]) or m[-1] <= 0 or m[-2] <= 0:
            return None
        out.append(g["m500"][k] * m[:len(e.R)] / m[-1])
    return np.array(out)


def _gal_loss(p, g, ks):
    """The PLAIN loss (shape-only relative RMS), matching s2.gal_loss."""
    cogs = model_cogs_coreshape(p, g, ks)
    if cogs is None:
        return 4.0
    return float(np.mean([
        np.sqrt(np.mean(((c - g["data"][k]) / g["data"][k]) ** 2))
        for c, k in zip(cogs, ks)]))


def _chunk_plain(args):
    p, ks, lo, hi = args
    return sum(_gal_loss(p, g, ks) for g in _W["gals"][lo:hi]), (hi - lo)


def _chunk_balanced_frozen(args):
    """Balanced loss with the SHAPE SLOPES FORCED TO ZERO — the control
    that separates proposal 3 (the objective was the obstacle) from
    proposal 2 (the conditioning was the obstacle). Without it, model B's
    gain cannot be attributed: the balanced loss re-tunes all 12 base
    parameters as well as the three slopes."""
    p, ks, lo, hi = args
    p = np.asarray(p, float).copy()
    p[14:17] = 0.0
    return _chunk_balanced((p, ks, lo, hi))


def _chunk_balanced(args):
    """Weight so compact and extended contribute EQUALLY overall."""
    p, ks, lo, hi = args
    wc, we = _W["w_compact"], _W["w_extended"]
    tot = 0.0
    for g in _W["gals"][lo:hi]:
        tot += (wc if g["compact"] else we) * _gal_loss(p, g, ks)
    return tot, (hi - lo)


def _penalty(p):
    s2 = _W["s2"]
    return s2.penalty(p[:12], "1ch-mof") + 30.0 * float(
        np.clip(-6.0 - p[12], 0, None) ** 2
        + np.clip(p[12] - 6.0, 0, None) ** 2
        + np.clip(abs(p[13]) - 4.0, 0, None) ** 2
        + np.sum(np.clip(np.abs(p[14:17]) - 4.0, 0, None) ** 2)
        + np.clip(-0.3 - p[17], 0, None) ** 2
        + np.clip(p[17] - 0.9, 0, None) ** 2)


def _simplex(p0):
    """Explicit initial simplex with a MEANINGFUL span on every axis.

    Necessary, not cosmetic. scipy's Nelder-Mead builds its own simplex by
    scaling each coordinate by 5% — except coordinates that are exactly
    zero, which it perturbs by `zdelt = 0.00025`. The three shape slopes
    warm-start at exactly 0 (to nest exp38's core channel), so the default
    simplex spans them by 2.5e-4 while they only do anything at O(1): the
    first run of this script returned |slopes| ~ 0.01 for that reason
    alone, which is a property of the start point and not a measurement.
    """
    step = np.maximum(0.05 * np.abs(p0), 0.02)
    step[12:14] = 0.3          # core amplitude and its logMh slope
    step[14:17] = 0.6          # the SHAPE slopes — the parameters on test
    step[17] = 0.15            # log10 rc_core
    sx = np.vstack([p0] + [p0 + step[i] * np.eye(len(p0))[i]
                           for i in range(len(p0))])
    return sx


def _fit(chunk_fn, p0, maxiter, label):
    n = len(_W["gals"])
    workers = max(os.cpu_count() - 2, 2)
    edges = np.linspace(0, n, workers + 1).astype(int)
    with Pool(workers, initializer=_w_init,
              initargs=(_W["rows_arg"],)) as pool:
        def loss(p):
            parts = pool.map(chunk_fn, [(p, KS, edges[i], edges[i + 1])
                                        for i in range(workers)])
            return sum(v for v, _ in parts) / n + _penalty(p)
        t0 = time.time()
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8,
                                  initial_simplex=_simplex(p0)))
    print(f"  [{label}] loss {r.fun:.4f} ({(time.time()-t0)/60:.1f} min, "
          f"{r.nit} iters)", flush=True)
    return r.x, float(r.fun)


def warm_start():
    """exp38's core theta, with the three new shape slopes at ZERO so the
    start point nests the parked model exactly."""
    core = np.load(E38D / "stage3_core.npz")["theta"]
    base = np.load(E40)["theta_z15"]            # the official z<=1.5 scope
    p = np.zeros(18)
    p[:12] = base[:12]
    p[12], p[13] = core[12], core[13]
    p[14:17] = 0.0
    p[17] = core[14]
    return p


def cmd_fit(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    n = len(_W["gals"])
    nc = _W["n_compact"]
    print(f"exp47 step 2 — core channel conditioned on DiffMAH SHAPE "
          f"(n={n}{', DEV' if dev else ''}; compact {nc}, weights "
          f"{_W['w_compact']:.2f}/{_W['w_extended']:.2f})", flush=True)

    p0 = warm_start()
    # TWO starts per objective: the nesting point, and one already
    # differentiating. A 1-D scan showed the balanced loss prefers a -1.5
    # 'early' slope at theta_A but only -0.25 at theta_B, i.e. the base
    # parameters and the shape slope are DEGENERATE for that objective —
    # so a single start cannot be trusted to find the better basin.
    p1 = p0.copy()
    p1[15] = -1.0
    it = max(int(6000 * (0.05 if dev else 1.0)), 120)
    out = {}
    for oname, chunk in (("A plain", _chunk_plain),
                         ("B balanced", _chunk_balanced),
                         ("C balanced-frozen", _chunk_balanced_frozen)):
        best = None
        for sname, ps in (("nest", p0), ("diff", p1)):
            th, lo = _fit(chunk, ps, it, f"{oname}/{sname}")
            sl = th[14:17]
            print(f"      shape slopes [logtc, early, late] = "
                  f"{sl[0]:+.3f} {sl[1]:+.3f} {sl[2]:+.3f}   "
                  f"|slopes| = {np.abs(sl).sum():.3f}", flush=True)
            if best is None or lo < best[1]:
                best = (th, lo)
        if oname.startswith("C"):
            best = (best[0].copy(), best[1])
            best[0][14:17] = 0.0     # the loss ignored them; do not report noise
        out[oname] = best
        print(f"    -> [{oname}] best loss {best[1]:.4f}, "
              f"|slopes| = {np.abs(best[0][14:17]).sum():.3f}", flush=True)
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"core_shape{tag}.npz",
             **{f"theta_{k.split()[0]}": v[0] for k, v in out.items()},
             **{f"loss_{k.split()[0]}": v[1] for k, v in out.items()},
             p0=p0)
    print(f"\n  wrote {OUTDIR / f'core_shape{tag}.npz'}")


def _build(theta, n, nr, rows):
    workers = max(os.cpu_count() - 2, 2)
    step = max(n // (4 * workers) + 1, 1)
    jobs = [(lo, min(lo + step, n), theta) for lo in range(0, n, step)]
    out = np.full((n, len(KS_ALL), nr), np.nan)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for lo, blk in pool.map(_cogs_chunk, jobs):
            out[lo:lo + len(blk)] = blk
    return out


def _cogs_chunk(args):
    lo, hi, theta = args
    out = np.full((hi - lo, len(KS_ALL), len(_W["e"].R)), np.nan)
    for i in range(lo, hi):
        c = model_cogs_coreshape(theta, _W["gals"][i], KS_ALL)
        if c is not None:
            out[i - lo] = c
    return lo, out


def cmd_report(dev=False):
    from hongshao import qa
    from judge import verdict, print_verdict, physics_gates

    rows = np.load(POP_NPZ)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    gals, e = _W["gals"], _W["e"]
    R, n = e.R, len(gals)
    data_cogs = np.stack([g["data"] for g in gals])
    pop = np.load(POP_NPZ)
    logms = np.array([pop["logms"][g["row"]] for g in gals])
    r50_t = qa._safe_rhalf(data_cogs, R, 0.5)
    d = np.load(OUTDIR / f"core_shape{tag}.npz")
    for key in ("A", "B", "C"):
        if f"theta_{key}" not in d.files:
            continue
        th = d[f"theta_{key}"]
        cogs = _build(th, n, len(R), rows)
        v = verdict(cogs, data_cogs, R, r50_truth=r50_t)
        gates = physics_gates(cogs, data_cogs, logms, e)
        sl = th[14:17]
        print_verdict(v, f"core+shape {key}  (slopes {sl[0]:+.2f} "
                         f"{sl[1]:+.2f} {sl[2]:+.2f})", gates)


def demo():
    """Nesting + capacity checks that need no data load."""
    p = np.zeros(18)
    p[12], p[13], p[17] = 0.0, 0.0, 0.2
    # f_core with zero shape slopes must not depend on the shape vector
    for shape in (np.zeros(3), np.array([2.0, -1.0, 0.5])):
        f = expit(p[12] + p[13] * 0.7 + float(p[14:17] @ shape))
        assert abs(f - 0.5) < 1e-12, f
    # ... and with non-zero slopes it MUST separate two galaxies that
    # differ only in shape — the whole point of the step
    p[14:17] = np.array([0.0, -1.2, 0.0])
    f_lo = expit(p[12] + float(p[14:17] @ np.array([0.0, -1.0, 0.0])))
    f_hi = expit(p[12] + float(p[14:17] @ np.array([0.0, +1.0, 0.0])))
    assert f_lo - f_hi > 0.4, (f_lo, f_hi)
    # the balanced weights must equalize the two groups' total contribution
    n, nc = 2397, 150
    wc, we = 0.5 * n / nc, 0.5 * n / (n - nc)
    assert abs(wc * nc - we * (n - nc)) < 1e-6
    assert abs((wc * nc + we * (n - nc)) / n - 1.0) < 1e-9
    print(f"demo OK — zero shape slopes nest exp38's core exactly; a "
          f"-1.2 'early' slope separates f_core {f_lo:.2f} vs {f_hi:.2f}; "
          f"balanced weights equalize groups ({wc:.2f}/{we:.2f}) and "
          "preserve the loss scale")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    is_dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(is_dev)
    elif cmd == "report":
        cmd_report(is_dev)
    else:
        raise SystemExit(__doc__)
