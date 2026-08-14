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
FAMILIES = {"moffat": 0, "cuspy": 1, "sersic_r50": 0,
            "gompertz_log": 0, "loglogistic": 0, "richards": 1}


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
    elif family == "gompertz_log":
        # Frechet CDF as a DEPOSIT: M(<R) = exp(-(rc/R)^c).
        # Sigma ~ R^-(c+2) far out (a power-law tail, the property that
        # won exp38) but a super-exponential central hole.
        c = float(np.clip(shape[0], 0.2, 8.0))

        def cog(rc):
            return np.exp(-np.clip((rc[None, :] / r[:, None]) ** c,
                                   0.0, 700.0))
    elif family == "loglogistic":
        # power law at BOTH ends: Sigma ~ R^(k-2) inner, R^-(k+2) outer,
        # so k < 2 is a genuine CUSP. rc is exactly the half-mass radius.
        k = float(np.clip(shape[0], 0.2, 8.0))

        def cog(rc):
            X = (r[:, None] / rc[None, :]) ** k
            return X / (1.0 + X)
    elif family == "richards":
        # generalized logistic in log R; NESTS the two above
        # (nu -> 0 gompertz_log, nu = 1 loglogistic), so the fitted nu
        # says which sub-family the kernel actually wants.
        k = float(np.clip(shape[0], 0.2, 8.0))
        nu = float(np.exp(np.clip(shape[1], -6.0, 2.0)))

        def cog(rc):
            X = np.exp(np.clip(-k * np.log(r[:, None] / rc[None, :]),
                               -700.0, 700.0))
            return (1.0 + nu * X) ** (-1.0 / nu)
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
    if family == "richards":
        v = p[12]
        pen += 30.0 * float(np.clip(-6.0 - v, 0, None) ** 2
                            + np.clip(v - 2.0, 0, None) ** 2)
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
    if family in ("cuspy", "richards"):
        step[12] = 0.3            # starts at 0 -> needs a real span
    sx = np.vstack([p0] + [p0 + step[i] * np.eye(len(p0))[i]
                           for i in range(len(p0))])

    def loss(p):
        if pool is None:                       # serial: slow but reliable
            v = np.array([gal_loss_family(p, g, KS, cfg, family)
                          for g in _W["gals"]], float)
        else:
            parts = pool.map(_chunk,
                             [(p, KS, cfg, family, edges[i], edges[i + 1])
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
    if family == "richards":
        # gamma slot -> k; nu starts at 1, i.e. exactly loglogistic
        return np.concatenate([base, [0.0]])
    return base


class _NullPool:
    """Stand-in so the fit loop needs no branching: `with` yields None,
    and `fit` runs the loss serially when it sees None."""

    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def cmd_fit(dev=False, objective_name=None, one=False, use_pool=False,
            winner_only=False, store_tag="", families=None):
    import objective as ob
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    # WINNER FIRST. The ::current arm is the control, and its decisive
    # cell is already banked (cuspy::current fits alpha=+0.124 for a
    # 0.00002 loss gain -- the old objective cannot see a cusp). Serial
    # fits run ~3 h each for the special-function families, so ordering
    # decides which results exist after a given wall-clock budget.
    cfgs = {}
    if objective_name:
        cfgs["winner"] = json.loads(_lookup_cfg(objective_name, tag))
    if not winner_only:
        cfgs["current"] = dict(ob.BASELINE)
    it = max(int(4000 * (0.05 if dev else 1.0)), 100)
    OUTDIR.mkdir(exist_ok=True)
    store = OUTDIR / f"shootout{store_tag}{tag}.npz"
    done = dict(np.load(store, allow_pickle=True)) if store.exists() else {}
    fams = families or list(FAMILIES)
    print(f"exp48 step C shootout — {len(fams)} families x "
          f"{len(cfgs)} objectives, FULL kernel "
          f"(n={len(_W['gals'])}{', DEV' if dev else ''})", flush=True)
    n = len(_W["gals"])
    workers = max(os.cpu_count() - 2, 2)
    edges = np.linspace(0, n, workers + 1).astype(int)
    # SERIAL by default. Two full-scale attempts hung on the second fit of
    # a process, and the workers reported BrokenPipeError on the pool's
    # ERROR path -- something raises inside a worker at full scale that
    # does not raise at dev scale or in a minimal 3-worker pool. Rather
    # than keep chasing it, serial is ~8x slower per fit and always
    # finishes. Pass --pool to opt back in.
    ctx = (Pool(workers, initializer=_w_init, initargs=(_W["rows_arg"],))
           if use_pool else _NullPool())
    with ctx as pool:
        for oname, cfg in cfgs.items():
            for family in fams:
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
                if one:
                    # ONE fit per process, by design. Two independent
                    # full-scale attempts hung on the SECOND fit of a
                    # process — 0% CPU, no live pool workers, parent
                    # blocked in pool.map — with a fresh pool per fit and
                    # again with one shared pool. The first fit in a
                    # process has never failed. Rather than keep
                    # debugging macOS spawn semantics, the driver loop
                    # re-invokes this script until every cell is cached.
                    print("  (--one: exiting after a single fit)",
                          flush=True)
                    return
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
    # the sigmoid families: richards must NEST the other two INSIDE the
    # migration mixture, which is the property that justifies its extra
    # parameter (nu -> 0 gompertz_log, nu = 1 loglogistic)
    B_gl = basis_family("gompertz_log", th4, [1.4], ti, 9.39, 8.0, r)
    B_rg = basis_family("richards", th4, [1.4, -6.0], ti, 9.39, 8.0, r)
    e_g = np.abs(B_gl - B_rg).max()
    assert e_g < 5e-3, f"richards(nu->0) != gompertz_log: {e_g:.2e}"
    B_ll = basis_family("loglogistic", th4, [1.4], ti, 9.39, 8.0, r)
    B_r1 = basis_family("richards", th4, [1.4, 0.0], ti, 9.39, 8.0, r)
    e_l = np.abs(B_ll - B_r1).max()
    assert e_l < 1e-12, f"richards(nu=1) != loglogistic: {e_l:.2e}"
    # loglogistic with k < 2 is a CUSP: more central mass than moffat
    assert (B_ll[inner] > B_mof[inner]).all(), "k<2 loglogistic must be cuspy"

    # every basis is a CoG: non-decreasing in R, and in [0, 1]
    for name, B in (("moffat", B_mof), ("cuspy", B_a),
                    ("gompertz_log", B_gl), ("loglogistic", B_ll),
                    ("richards", B_r1),
                    ("sersic_r50", basis_family("sersic_r50", th4, [2.0], ti,
                                                9.39, 8.0, r))):
        assert np.all(np.diff(B, axis=0) >= -1e-12), f"{name} not monotone"
        assert B.min() >= -1e-12 and B.max() <= 1.0 + 1e-9, name
    # start vectors have the right length for each family
    assert len(start_vector("cuspy")) == 13
    assert len(start_vector("moffat")) == 12
    assert abs(start_vector("cuspy")[12]) < 1e-12, "cuspy must START nested"
    assert len(start_vector("richards")) == 13
    print("demo OK — inside the full migration mixture: cuspy nests moffat "
          f"at alpha=0 (<1e-12); richards nests gompertz_log ({e_g:.1e}) "
          f"and loglogistic ({e_l:.1e}); k<2 loglogistic is cuspier than "
          "moffat; all six bases are monotone CoGs in [0,1]")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    is_dev = "--dev" in sys.argv
    obj = None
    if "--objective" in sys.argv:
        obj = sys.argv[sys.argv.index("--objective") + 1]
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        fams = (sys.argv[sys.argv.index("--families") + 1].split(",")
                if "--families" in sys.argv else None)
        st = (sys.argv[sys.argv.index("--store") + 1]
              if "--store" in sys.argv else "")
        cmd_fit(is_dev, obj, one="--one" in sys.argv,
                use_pool="--pool" in sys.argv,
                winner_only="--winner-only" in sys.argv,
                store_tag=st, families=fams)
    else:
        raise SystemExit(__doc__)
