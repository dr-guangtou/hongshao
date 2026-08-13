"""exp48 step B — what should the fit actually minimize?

exp47 showed the kernel's defect is a SPREAD: it under-fills compact
galaxies' centres by 30-44% and over-fills extended ones by up to +21%.
Re-weighting the objective by a compact/extended split fixed much of it,
but that hard-wires the model to TNG300 (it needs a size threshold and
this simulation's population fractions), so it is not adoptable.

This step pursues the same gain through objective forms that carry NO
simulation-specific calibration. The current objective is

    L = (1/N) sum_g (1/K) sum_k sqrt( (1/nR) sum_i [ (m_i - d_i)/d_i ]^2 )

on the CUMULATIVE profile, in LINEAR mass, as a FRACTIONAL residual, with
all 24 radii weighted equally. Only 4 of those 24 radii lie below 5 kpc,
and because the curve of growth is cumulative, neighbouring points are
nearly the same number — an outskirt error is effectively replicated
across many points while the fast-changing centre gets few.

Five axes, all portable:

  quantity   cog | density        compare M(<R) or Sigma(R)
  residual   frac | abs | log     (m-d)/d, (m-d)/Mtot, or log m - log d
  radial     rms | absmean | max  aggregate over radii (max = KS-like)
  weight     uniform | perdex | inner
  galaxy     mean | median | cvar20

Two notes on specific options.

**abs** is the absolute LINEAR difference, normalized per galaxy by that
galaxy's own measured M(<148 kpc). Without the normalization it would also
weight massive galaxies over light ones, which is a different lever than
the intended one; dividing by the galaxy's own total keeps the radial
re-weighting and drops the cross-galaxy effect. On the density profile
this weights the centre (Sigma is largest there); on the CoG it weights
the outskirts (cumulative mass grows outward). Both are in the grid so
this is measured rather than argued.

**cvar20** — the mean of the WORST 20% of per-galaxy losses — is the
portable replacement for exp47's rebalancing. It needs no size threshold,
no population fractions, and no measurement of the galaxies at all: it
targets whatever tail is badly fit, which in TNG300 happens to be the
compact population.

The MODEL is held fixed at the adopted 12-parameter 1ch-mof throughout, so
the objective is the only thing varying. Fit scope z <= 1.5.

Run: PYTHONPATH=. uv run python experiments/exp48_objective_profile/\
objective.py {demo|screen|report} [--dev]
"""
import importlib.util
import json
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
sys.path.insert(0, str(ROOT / "experiments/exp47_compact_defect"))
sys.path.insert(0, str(ROOT / "experiments/exp46_highz_ridge"))

OUTDIR = HERE / "outputs"
E38 = ROOT / "experiments/exp38_deposit_rethink"
E40 = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"

KS = [0, 1, 2, 3]                  # the OFFICIAL z<=1.5 fit scope
KS_ALL = [0, 1, 2, 3, 4]
_W = {}

BASELINE = dict(quantity="cog", residual="frac", radial="rms",
                weight="uniform", galaxy="mean")


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    s2 = _load_by_path("exp38_stage2", E38 / "stage2_multiepoch.py")
    s2._w_init(rows)
    _W.update(s2._W)
    _W["s2"] = s2
    _W["rows_arg"] = rows
    R = _W["e"].R
    # radial weight vectors, built once. Each is normalized to sum to 1 so
    # the loss scale stays comparable across weightings.
    from hongshao.profile_emulator import density_from_cog
    _W["density_from_cog"] = density_from_cog
    mid = density_from_cog(np.log10(np.ones((1, len(R))) * np.geomspace(
        1.0, 2.0, len(R))), R)[1]
    _W["mid"] = mid
    _W["wts"] = {}
    for tag, r in (("cog", R), ("density", mid)):
        dlogr = np.gradient(np.log10(r))
        _W["wts"][tag] = {
            "uniform": np.ones(len(r)) / len(r),
            "perdex": dlogr / dlogr.sum(),
            "inner": (1.0 / r) / (1.0 / r).sum(),
        }


# --------------------------------------------------------------------------- #
# the composable objective                                                     #
# --------------------------------------------------------------------------- #
def gal_loss(p, g, ks, cfg):
    """Per-galaxy loss under one objective configuration."""
    s2 = _W["s2"]
    cogs = s2.model_cogs(p, g, ks, "1ch-mof")
    if cogs is None:
        return 4.0
    w = _W["wts"][cfg["quantity"]][cfg["weight"]]
    per_epoch = []
    for c, k in zip(cogs, ks):
        d = g["data"][k]
        if cfg["quantity"] == "density":
            lm = _W["density_from_cog"](np.log10(np.clip(c, 1.0, None))[None],
                                        _W["e"].R)[0][0]
            ld = _W["density_from_cog"](np.log10(np.clip(d, 1.0, None))[None],
                                        _W["e"].R)[0][0]
            m_v, d_v, norm = 10.0 ** lm, 10.0 ** ld, 10.0 ** np.max(ld)
        else:
            m_v, d_v, norm = c, d, d[-1]

        if cfg["residual"] == "frac":
            r = (m_v - d_v) / np.clip(d_v, 1e-30, None)
        elif cfg["residual"] == "abs":
            # linear difference, made dimensionless by the galaxy's OWN
            # total so this is a radial re-weighting and not a mass one
            r = (m_v - d_v) / max(norm, 1e-30)
        else:                                     # log10
            r = (np.log10(np.clip(m_v, 1e-30, None))
                 - np.log10(np.clip(d_v, 1e-30, None)))
        if not np.isfinite(r).all():
            return 4.0

        if cfg["radial"] == "rms":
            per_epoch.append(np.sqrt(np.sum(w * r ** 2)))
        elif cfg["radial"] == "absmean":
            per_epoch.append(np.sum(w * np.abs(r)))
        else:                                     # max: KS-like
            per_epoch.append(np.max(np.abs(r)))
    return float(np.mean(per_epoch))


def reduce_galaxies(v, how):
    """Population aggregation over the per-galaxy loss vector."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 4.0
    if how == "mean":
        return float(np.mean(v))
    if how == "median":
        return float(np.median(v))
    if how == "cvar20":
        k = max(int(np.ceil(0.20 * len(v))), 1)
        return float(np.mean(np.sort(v)[-k:]))    # the WORST 20%
    raise ValueError(how)


def _chunk(args):
    """Return the per-galaxy VECTOR, not a partial sum: cvar20 and median
    cannot be reduced chunk-wise."""
    p, ks, cfg, lo, hi = args
    return lo, np.array([gal_loss(p, g, ks, cfg)
                         for g in _W["gals"][lo:hi]], float)


def make_objective(cfg, pool, edges, n):
    def loss(p):
        parts = pool.map(_chunk, [(p, KS, cfg, edges[i], edges[i + 1])
                                  for i in range(len(edges) - 1)])
        v = np.empty(n)
        for lo, blk in parts:
            v[lo:lo + len(blk)] = blk
        return (reduce_galaxies(v, cfg["galaxy"])
                + _W["s2"].penalty(p, "1ch-mof"))
    return loss


def _simplex12(p0):
    """Initial simplex for the 12-parameter base theta.

    exp47's `_simplex` is hard-coded for the 18-dim core+shape vector, so
    this is the 12-dim counterpart carrying the same fix: scipy perturbs a
    coordinate that is exactly 0 by only zdelt=2.5e-4, which silently
    freezes it. The six conditioning slopes sit near zero, so the floor on
    the step size is load-bearing, not decorative.
    """
    p0 = np.asarray(p0, float)
    step = np.maximum(0.05 * np.abs(p0), 0.02)
    return np.vstack([p0] + [p0 + step[i] * np.eye(len(p0))[i]
                             for i in range(len(p0))])


def fit(cfg, p0, maxiter, label):
    n = len(_W["gals"])
    workers = max(os.cpu_count() - 2, 2)
    edges = np.linspace(0, n, workers + 1).astype(int)
    with Pool(workers, initializer=_w_init,
              initargs=(_W["rows_arg"],)) as pool:
        loss = make_objective(cfg, pool, edges, n)
        t0 = time.time()
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-10,
                                  initial_simplex=_simplex12(p0)))
    print(f"  [{label}] own-loss {r.fun:.5f} "
          f"({(time.time()-t0)/60:.1f} min, {r.nit} it)", flush=True)
    return r.x, float(r.fun)


# --------------------------------------------------------------------------- #
# the screen                                                                   #
# --------------------------------------------------------------------------- #
def screen_configs():
    """Baseline plus ONE axis varied at a time (10 cells)."""
    out = [("baseline", dict(BASELINE))]
    for axis, opts in (("quantity", ["density"]),
                       ("residual", ["abs", "log"]),
                       ("radial", ["absmean", "max"]),
                       ("weight", ["perdex", "inner"]),
                       ("galaxy", ["median", "cvar20"])):
        for o in opts:
            c = dict(BASELINE)
            c[axis] = o
            out.append((f"{axis}={o}", c))
    return out


SHORT = dict(quantity="q", residual="res", radial="rad", weight="w",
             galaxy="gal")       # NOT k[0]: residual and radial collide


def cname(cfg):
    """Canonical, order-stable cell name so the store caches correctly."""
    return "|".join(f"{SHORT[k]}={cfg[k]}" for k in
                    ("quantity", "residual", "radial", "weight", "galaxy"))


def factorial_configs():
    """The focused factorial, chosen by the screen.

    The screen's four best cells were residual=abs (2.49), radial=absmean
    (2.57), galaxy=median (2.59) and residual=log (2.69), all beating the
    baseline's 2.72. They share one property: each REDUCES the objective's
    sensitivity to a few extreme fractional residuals. The cell that
    increases it — galaxy=cvar20 — was the worst of the sane cells (2.89).
    So the factorial crosses the three robustness axes.

    Plus two cells giving `density` a fair test: it collapsed in the
    screen (14.37) only when paired with a FRACTIONAL residual, which
    diverges wherever Sigma is small in the outermost bins. Paired with
    abs or log it may behave.
    """
    out = []
    for residual in ("abs", "log"):
        for radial in ("rms", "absmean"):
            for galaxy in ("mean", "median"):
                out.append(dict(BASELINE, residual=residual, radial=radial,
                                galaxy=galaxy))
    for residual in ("abs", "log"):
        out.append(dict(BASELINE, quantity="density", residual=residual))
    return [(cname(c), c) for c in out]


def _run_cells(cfgs, store, dev):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    p0 = np.asarray(np.load(E40)["theta_z15"], float)[:12]
    it = max(int(4000 * (0.05 if dev else 1.0)), 100)
    OUTDIR.mkdir(exist_ok=True)
    done = dict(np.load(store, allow_pickle=True)) if store.exists() else {}
    for name, cfg in cfgs:
        if f"theta::{name}" in done:
            print(f"  [{name}] cached", flush=True)
            continue
        th, lo = fit(cfg, p0, it, name)
        done[f"theta::{name}"] = th
        done[f"loss::{name}"] = np.array([lo])
        done[f"cfg::{name}"] = np.array([json.dumps(cfg)])
        np.savez(store, **done)       # save after EVERY fit (unattended)
    print(f"\n  wrote {store}")


def cmd_factorial(dev=False):
    tag = "_dev" if dev else ""
    cfgs = factorial_configs()
    print(f"exp48 step B factorial — {len(cfgs)} cells "
          f"({', DEV' if dev else 'full'})", flush=True)
    _run_cells(cfgs, OUTDIR / f"factorial{tag}.npz", dev)


def cmd_screen(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    n = len(_W["gals"])
    p0 = np.asarray(np.load(E40)["theta_z15"], float)[:12]
    it = max(int(4000 * (0.05 if dev else 1.0)), 100)
    cfgs = screen_configs()
    print(f"exp48 step B screen — {len(cfgs)} objectives, model FIXED at the "
          f"adopted 12-param 1ch-mof (n={n}{', DEV' if dev else ''})",
          flush=True)
    OUTDIR.mkdir(exist_ok=True)
    store = OUTDIR / f"screen{tag}.npz"
    done = dict(np.load(store, allow_pickle=True)) if store.exists() else {}
    for name, cfg in cfgs:
        key = f"theta::{name}"
        if key in done:
            print(f"  [{name}] cached", flush=True)
            continue
        th, lo = fit(cfg, p0, it, name)
        done[key] = th
        done[f"loss::{name}"] = np.array([lo])
        done[f"cfg::{name}"] = np.array([json.dumps(cfg)])
        # save after EVERY fit: this runs unattended for hours
        np.savez(store, **done)
    print(f"\n  wrote {store}")


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
    d = {}
    for st in (f"screen{tag}.npz", f"factorial{tag}.npz"):
        if (OUTDIR / st).exists():
            d.update(dict(np.load(OUTDIR / st, allow_pickle=True)))
    names = [k.split("::", 1)[1] for k in d if k.startswith("theta::")]
    rowsout = {}
    for name in names:
        cogs = build_cogs(d[f"theta::{name}"], n, len(R), rows)
        v = verdict(cogs, data_cogs, R, r50_truth=r50_t)
        gates = physics_gates(cogs, data_cogs, logms, e)
        cd = qa.mass_cdf_distance(cogs, data_cogs, R, quantities=qa.CDF_KEYS)
        rowsout[name] = (v, gates, cd)
        print_verdict(v, name, gates)
    _summary(rowsout)
    np.savez(OUTDIR / f"screen_verdict{tag}.npz",
             names=np.array(list(rowsout)))


def _summary(rowsout):
    from judge import PLANES
    key = f"{PLANES[0][0]}|{PLANES[0][1]}"
    print("\n\n=== SCREEN SUMMARY (ranked by mean centered E/floor, "
          "z<=1.5) ===")
    print(f"  {'objective':<20}{'plane z0.4':>11}{'mean z<=1.5':>13}"
          f"{'M5 cmp z0.4':>13}{'M5 ext z0.4':>13}{'differential':>14}")
    rank = []
    for name, (v, g, cd) in rowsout.items():
        pl = [v[("plane", key, j, "all")] for j in range(4)]
        rank.append((float(np.mean(pl)), name, pl[0], v, g))
    for m, name, p0v, v, g in sorted(rank):
        print(f"  {name:<20}{p0v:11.1f}{m:13.2f}"
              f"{100 * v[('inner', 5, 0, 'compact')]:12.1f}%"
              f"{100 * v[('inner', 5, 0, 'extended')]:12.1f}%"
              f"{g['diff_model'][0]:9.2f}/{g['diff_model'][1]:.2f}")


def _cogs_chunk(args):
    lo, hi, theta = args
    s2 = _W["s2"]
    out = np.full((hi - lo, len(KS_ALL), len(_W["e"].R)), np.nan)
    for i in range(lo, hi):
        c = s2.model_cogs(theta, _W["gals"][i], KS_ALL, "1ch-mof")
        if c is not None:
            out[i - lo] = c
    return lo, out


def build_cogs(theta, n, nr, rows):
    workers = max(os.cpu_count() - 2, 2)
    step = max(n // (4 * workers) + 1, 1)
    jobs = [(lo, min(lo + step, n), theta) for lo in range(0, n, step)]
    out = np.full((n, len(KS_ALL), nr), np.nan)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for lo, blk in pool.map(_cogs_chunk, jobs):
            out[lo:lo + len(blk)] = blk
    return out


def demo():
    # reduce_galaxies: cvar20 must take the WORST fifth, not the best
    v = np.arange(100.0)
    assert abs(reduce_galaxies(v, "mean") - 49.5) < 1e-9
    assert abs(reduce_galaxies(v, "median") - 49.5) < 1e-9
    assert abs(reduce_galaxies(v, "cvar20") - np.mean(np.arange(80, 100))) < 1e-9
    assert reduce_galaxies(v, "cvar20") > reduce_galaxies(v, "mean")
    # and it must be INSENSITIVE to the well-fit bulk, which is the whole
    # point: improving already-good galaxies must not move it
    good = v.copy()
    good[:50] = 0.0
    assert abs(reduce_galaxies(good, "cvar20")
               - reduce_galaxies(v, "cvar20")) < 1e-9
    assert reduce_galaxies(good, "mean") < reduce_galaxies(v, "mean")
    # NaNs are dropped, not propagated
    nanv = v.copy()
    nanv[:10] = np.nan
    assert np.isfinite(reduce_galaxies(nanv, "cvar20"))
    # every screen cell differs from the baseline in exactly ONE axis
    base = dict(BASELINE)
    for name, cfg in screen_configs()[1:]:
        diff = [k for k in base if cfg[k] != base[k]]
        assert len(diff) == 1, (name, diff)
        assert name.startswith(diff[0]), (name, diff)
    assert len(screen_configs()) == 10, len(screen_configs())
    print(f"demo OK — {len(screen_configs())} screen cells, each one axis "
          f"from baseline; cvar20 takes the worst fifth ("
          f"{reduce_galaxies(v, 'cvar20'):.1f} vs mean "
          f"{reduce_galaxies(v, 'mean'):.1f}) and ignores the well-fit bulk")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    is_dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "screen":
        cmd_screen(is_dev)
    elif cmd == "factorial":
        cmd_factorial(is_dev)
    elif cmd == "report":
        cmd_report(is_dev)
    else:
        raise SystemExit(__doc__)
