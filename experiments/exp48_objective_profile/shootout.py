"""exp48 step C — the deposit-profile shootout, inside the FULL kernel.

exp38 ranked deposit profiles with a SIMPLIFIED harness: single-epoch,
pure additive deposits, one theta per epoch, no migration. That harness
cannot see how a profile interacts with the transport, and it scored
candidates with the very objective exp48 step B is testing. Moffat's
margin over gausswing at z=0.4 was 0.2001 vs 0.1997 — gausswing was
AHEAD. The incumbent was chosen inside the noise of a suspect metric,
by a harness blind to half the model.

This runs the comparison properly: each profile is fitted inside the
complete multi-epoch kernel (conditioning, migration clock, M(<500)
normalization) at the official z <= 1.5 scope.

Candidates. The profile enters the kernel only through the `cog(rc)`
closure inside `exp38.stage2_multiepoch.basis_mof`, so each candidate is
that one function swapped, with the two-state migration mixture
untouched:

  moffat      the incumbent CONTROL. 12 params. Sigma FLAT at R->0.
  cuspy       12 + alpha = 13 params. Sigma ~ R^-alpha at the centre;
              alpha=0 nests moffat exactly, so the fitted alpha IS the
              measurement of how much cusp the data want.
  sersic_r50  12 params, with the gamma slot re-read as the Sersic index
              n and the scale as R50. exp38 disqualified Sersic on an
              n-scale degeneracy, not on physics.

gausswing is DEPRIORITIZED, on a physical argument rather than an
oversight: its core is a Gaussian, which is also flat at R->0. It cannot
supply the cusp that is the whole hypothesis here, so it would test a
different question (wing shape) than the one this step asks.

Run: PYTHONPATH=. uv run python experiments/exp48_objective_profile/\
shootout.py {demo|fit|report} [--dev] [--objective NAME]
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
from scipy.special import betainc, gammainc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp47_compact_defect"))
sys.path.insert(0, str(ROOT / "experiments/exp46_highz_ridge"))

OUTDIR = HERE / "outputs"
E38 = ROOT / "experiments/exp38_deposit_rethink"
E40 = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"

KS = [0, 1, 2, 3]
KS_ALL = [0, 1, 2, 3, 4]
_W = {}

# n_extra = parameters BEYOND the 12 base ones
FAMILIES = {"moffat": 0, "cuspy": 1, "sersic_r50": 0}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    import objective as ob
    ob._w_init(rows)
    _W.update(ob._W)
    _W["ob"] = ob


def basis_family(family, th4, shape, ti, t_obs, tk, r):
    """The exp38 dyntrans basis with the deposit CoG swapped.

    Identical to ``basis_mof`` except for ``cog``: same birth radius law,
    same self-similar migration clock, same two-state mixture. Only the
    per-deposit profile changes, which is exactly the variable on test.
    """
    rc0 = np.clip(10.0 ** th4[0] * (ti / t_obs) ** th4[1], 1e-4, 1e5)
    dt = np.clip(tk - ti, 0.0, None)
    fc = np.exp(-dt / (10.0 ** th4[2] * ti))
    rcw = np.clip(rc0 * (tk / ti) ** max(th4[3], 0.0), 1e-4, 1e5)

    if family == "moffat":
        gam = float(np.clip(shape[0], 1.06, 6.0))

        def cog(rc):
            return 1.0 - (1.0 + (r[:, None] / rc[None, :]) ** 2) ** (1.0 - gam)
    elif family == "cuspy":
        gam = float(np.clip(shape[0], 1.06, 6.0))
        a = float(np.clip(shape[1], 0.0, 1.9))
        p = 1.0 - a / 2.0
        q = max(gam - 1.0 + a / 2.0, 1e-3)

        def cog(rc):
            X = (r[:, None] / rc[None, :]) ** 2
            return betainc(p, q, X / (1.0 + X))
    elif family == "sersic_r50":
        from profiles import b_n
        n = float(np.clip(shape[0], 0.35, 8.0))
        bn = b_n(n)

        def cog(rc):
            a_s = rc / (bn ** n)
            return gammainc(2.0 * n, (r[:, None] / a_s[None, :]) ** (1.0 / n))
    else:
        raise ValueError(family)
    return fc[None, :] * cog(rc0) + (1.0 - fc)[None, :] * cog(rcw)


def model_cogs_family(p, g, ks, family):
    """The exp38 forward model with ``basis_family`` in place of
    ``basis_mof``. p[:12] is the base theta; p[12:] the extra shape
    parameters (``cuspy`` alone has one: alpha)."""
    e, s2 = _W["e"], _W["s2"]
    mah = g["mah"]
    base6, _, _ = s2.theta_of(np.asarray(p)[:12], g, "1ch-mof")
    w = e.weights(mah["z"], base6[3], base6[4])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None
    dM = dM / dM.sum()
    shape = [base6[5]] + list(np.asarray(p)[12:])
    th4 = [base6[0], base6[1], 0.0, base6[2]]
    out = []
    for k in ks:
        mask = mah["snap"] <= e.ANCHOR_SNAP[k]
        B = basis_family(family, th4, shape, mah["t"], mah["t_obs"],
                         e.pe.AT[k], e.R_EXT)
        m = B @ (dM * mask)
        if not np.isfinite(m[-1]) or m[-1] <= 0 or m[-2] <= 0:
            return None
        out.append(g["m500"][k] * m[:len(e.R)] / m[-1])
    return np.array(out)


def score_cogs(cogs, g, ks, cfg):
    """Score already-built CoGs under an objective config.

    Deliberately a local copy of the residual/aggregation half of
    ``objective.gal_loss`` rather than a refactor of it: that module is
    executing the step-B factorial while this is being written, and macOS
    spawns pool workers (which re-import it), so editing it mid-run would
    corrupt a multi-hour job. Fold the two together once step B is done.
    """
    w = _W["wts"][cfg["quantity"]][cfg["weight"]]
    per_epoch = []
    for c, k in zip(cogs, ks):
        d = g["data"][k]
        if cfg["quantity"] == "density":
            dfc = _W["density_from_cog"]
            lm = dfc(np.log10(np.clip(c, 1.0, None))[None], _W["e"].R)[0][0]
            ld = dfc(np.log10(np.clip(d, 1.0, None))[None], _W["e"].R)[0][0]
            m_v, d_v, norm = 10.0 ** lm, 10.0 ** ld, 10.0 ** np.max(ld)
        else:
            m_v, d_v, norm = c, d, d[-1]
        if cfg["residual"] == "frac":
            r = (m_v - d_v) / np.clip(d_v, 1e-30, None)
        elif cfg["residual"] == "abs":
            r = (m_v - d_v) / max(norm, 1e-30)
        else:
            r = (np.log10(np.clip(m_v, 1e-30, None))
                 - np.log10(np.clip(d_v, 1e-30, None)))
        if not np.isfinite(r).all():
            return 4.0
        if cfg["radial"] == "rms":
            per_epoch.append(np.sqrt(np.sum(w * r ** 2)))
        elif cfg["radial"] == "absmean":
            per_epoch.append(np.sum(w * np.abs(r)))
        else:
            per_epoch.append(np.max(np.abs(r)))
    return float(np.mean(per_epoch))


def gal_loss_family(p, g, ks, cfg, family):
    cogs = model_cogs_family(p, g, ks, family)
    if cogs is None:
        return 4.0
    return score_cogs(cogs, g, ks, cfg)


def _chunk(args):
    p, ks, cfg, family, lo, hi = args
    return lo, np.array([gal_loss_family(p, g, ks, cfg, family)
                         for g in _W["gals"][lo:hi]], float)


def _penalty(p, family):
    s2 = _W["s2"]
    pen = s2.penalty(np.asarray(p)[:12], "1ch-mof")
    if family == "cuspy":
        a = p[12]
        pen += 30.0 * float(np.clip(-a, 0, None) ** 2
                            + np.clip(a - 1.9, 0, None) ** 2)
    return pen


def fit(family, cfg, p0, maxiter, label, pool, edges):
    """One fit against an EXISTING pool.

    The pool is created once for the whole run and passed in. Creating and
    tearing one down per fit hung the first full-scale attempt: fit 1
    finished, fit 2's pool came up with no live workers, and `pool.map`
    blocked forever at 0% CPU. Each worker's initializer path-loads the
    whole exp38 harness, so repeated spawn cycles are both the expensive
    part and the fragile one — doing it once removes both.
    """
    from objective import reduce_galaxies
    n = len(_W["gals"])
    step = np.maximum(0.05 * np.abs(p0), 0.02)
    if family == "cuspy":
        step[12] = 0.3            # alpha starts at 0 -> needs a real span
    sx = np.vstack([p0] + [p0 + step[i] * np.eye(len(p0))[i]
                           for i in range(len(p0))])

    def loss(p):
        parts = pool.map(_chunk, [(p, KS, cfg, family, edges[i], edges[i + 1])
                                  for i in range(len(edges) - 1)])
        v = np.empty(n)
        for lo, blk in parts:
            v[lo:lo + len(blk)] = blk
        return reduce_galaxies(v, cfg["galaxy"]) + _penalty(p, family)

    t0 = time.time()
    r = minimize(loss, p0, method="Nelder-Mead",
                 options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-10,
                              initial_simplex=sx))
    extra = "" if len(r.x) == 12 else f"  alpha={r.x[12]:+.3f}"
    print(f"  [{label}] own-loss {r.fun:.5f} "
          f"({(time.time()-t0)/60:.1f} min, {r.nit} it){extra}", flush=True)
    return r.x, float(r.fun)


def start_vector(family):
    base = np.asarray(np.load(E40)["theta_z15"], float)[:12]
    if family == "cuspy":
        return np.concatenate([base, [0.0]])      # nests moffat exactly
    if family == "sersic_r50":
        p = base.copy()
        p[5] = 2.0                                # the gamma slot becomes n
        return p
    return base


def cmd_fit(dev=False, objective_name=None):
    import objective as ob
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    cfgs = {"current": dict(ob.BASELINE)}
    if objective_name:
        cfgs["winner"] = json.loads(_lookup_cfg(objective_name, tag))
    it = max(int(4000 * (0.05 if dev else 1.0)), 100)
    OUTDIR.mkdir(exist_ok=True)
    store = OUTDIR / f"shootout{tag}.npz"
    done = dict(np.load(store, allow_pickle=True)) if store.exists() else {}
    print(f"exp48 step C shootout — {len(FAMILIES)} families x "
          f"{len(cfgs)} objectives, FULL kernel "
          f"(n={len(_W['gals'])}{', DEV' if dev else ''})", flush=True)
    n = len(_W["gals"])
    workers = max(os.cpu_count() - 2, 2)
    edges = np.linspace(0, n, workers + 1).astype(int)
    with Pool(workers, initializer=_w_init,
              initargs=(_W["rows_arg"],)) as pool:
        for oname, cfg in cfgs.items():
            for family in FAMILIES:
                key = f"{family}::{oname}"
                if f"theta::{key}" in done:
                    print(f"  [{key}] cached", flush=True)
                    continue
                th, lo = fit(family, cfg, start_vector(family), it, key,
                             pool, edges)
                done[f"theta::{key}"] = th
                done[f"loss::{key}"] = np.array([lo])
                done[f"cfg::{key}"] = np.array([json.dumps(cfg)])
                np.savez(store, **done)
    print(f"\n  wrote {store}")


def _lookup_cfg(name, tag):
    # fall back to the FULL stores: a --dev smoke test still needs the
    # winning config, which is only ever produced by a full step-B run
    for st in (f"factorial{tag}.npz", f"screen{tag}.npz",
               "factorial.npz", "screen.npz"):
        p = OUTDIR / st
        if p.exists():
            d = dict(np.load(p, allow_pickle=True))
            if f"cfg::{name}" in d:
                return str(d[f"cfg::{name}"][0])
    raise SystemExit(f"objective {name!r} not found in the step-B stores")


def demo():
    """Nesting and plumbing checks that need no data load."""
    r = np.logspace(np.log10(0.5), np.log10(300.0), 40)
    ti = np.array([1.0, 3.0, 6.0])
    th4 = [np.log10(50.0), 2.0, 0.0, 0.9]
    B_mof = basis_family("moffat", th4, [1.4], ti, 9.39, 8.0, r)
    B_cus = basis_family("cuspy", th4, [1.4, 0.0], ti, 9.39, 8.0, r)
    err = np.abs(B_mof - B_cus).max()
    assert err < 1e-12, f"cuspy(alpha=0) basis != moffat basis: {err:.2e}"
    # a real cusp must move mass INWARD at fixed outer normalization
    B_a = basis_family("cuspy", th4, [1.4, 1.0], ti, 9.39, 8.0, r)
    inner = np.argmin(np.abs(r - 2.0))
    assert (B_a[inner] > B_cus[inner]).all(), "alpha>0 must add central mass"
    # every basis is a CoG: non-decreasing in R, and in [0, 1]
    for name, B in (("moffat", B_mof), ("cuspy", B_a),
                    ("sersic_r50", basis_family("sersic_r50", th4, [2.0], ti,
                                                9.39, 8.0, r))):
        assert np.all(np.diff(B, axis=0) >= -1e-12), f"{name} not monotone"
        assert B.min() >= -1e-12 and B.max() <= 1.0 + 1e-9, name
    # start vectors have the right length for each family
    assert len(start_vector("cuspy")) == 13
    assert len(start_vector("moffat")) == 12
    assert abs(start_vector("cuspy")[12]) < 1e-12, "cuspy must START nested"
    print("demo OK — the cuspy BASIS nests the moffat basis exactly at "
          "alpha=0 (<1e-12) inside the full migration mixture; alpha>0 adds "
          "central mass; all three bases are monotone CoGs in [0,1]")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    is_dev = "--dev" in sys.argv
    obj = None
    if "--objective" in sys.argv:
        obj = sys.argv[sys.argv.index("--objective") + 1]
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(is_dev, obj)
    else:
        raise SystemExit(__doc__)
