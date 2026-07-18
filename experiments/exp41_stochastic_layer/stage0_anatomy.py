"""exp41 stage 0 — anatomy of the official kernel + f_core alignment.

The exp32 anatomy protocol on the ADOPTED kernel at its OFFICIAL scope
(exp40: 1ch-mof, z<=1.5 joint fit): per galaxy, free ONE of the six
base components at a time — a bounded scalar refit inside the physical
box, plain loss over the official epochs — and record the freed loss
and the fitted per-galaxy value. The per-component deltas are the raw
material of the stochastic layer:

  (a) which components carry the per-galaxy individuality (median
      relative loss improvement per component);
  (b) the delta distributions (16/50/84 spread) and their mutual
      correlations (the anatomy subspace's shape);
  (c) ALIGNMENT: do the kernel's own leading deltas correlate with the
      exp39 per-galaxy core fractions (percore_gauss_z2.npz)? If yes,
      the measured core-diversity axis is reachable INSIDE the
      kernel's parameter space — no core channel needed for the layer;
  (d) the feature-predictable part (Spearman vs logmh/c200c/fz2/
      logms): what belongs to conditioning vs what remains stochastic.

Run: PYTHONPATH=. uv run python experiments/exp41_stochastic_layer/\
stage0_anatomy.py {demo|run|report} [--dev]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUTDIR = HERE / "outputs"
E38_DIR = ROOT / "experiments/exp38_deposit_rethink"
E39_DIR = ROOT / "experiments/exp39_core_revisit"
E40_DIR = ROOT / "experiments/exp40_epoch_objective"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_FIT = [0, 1, 2, 3]                       # the official z<=1.5 scope
NAMES = ["log_rc", "g", "q", "mu", "sig", "gamma"]
LO6 = np.array([1.0, 0.0, 0.0, 0.0, 0.05, 1.06])
HI6 = np.array([3.0, 6.0, 3.0, 3.0, 2.0, 6.0])   # g box at the stress
#                                                  value (4.0 was benign)
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


def _anatomy_one(args):
    """One galaxy: base loss + per-component (freed loss, fitted value)."""
    (gi,) = args
    s2 = _W["s2"]
    g = _W["gals"][gi]
    th0 = _W["th0"]
    l_base = s2.gal_loss(th0, g, KS_FIT, "1ch-mof")
    l_free = np.empty(6)
    v_free = np.empty(6)
    for j in range(6):
        def f(x):
            th = th0.copy()
            th[j] = x
            return s2.gal_loss(th, g, KS_FIT, "1ch-mof")
        r = minimize_scalar(f, bounds=(LO6[j], HI6[j]), method="bounded",
                            options=dict(maxiter=80, xatol=1e-4))
        if r.fun < l_base:
            l_free[j], v_free[j] = r.fun, r.x
        else:
            l_free[j], v_free[j] = l_base, th0[j]
    return gi, g["row"], l_base, l_free, v_free


def cmd_run(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    gals = _W["gals"]
    n = len(gals)
    workers = max(os.cpu_count() - 2, 2)
    t0 = time.time()
    print(f"exp41 stage 0 anatomy (n={n}{', DEV' if dev else ''}; official "
          f"z<=1.5 scope; one freed component per fit):", flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        res = pool.map(_anatomy_one, [(i,) for i in range(n)])
    gi, row, l_base, l_free, v_free = map(np.array, zip(*res))
    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"anatomy{tag}.npz", row=row, l_base=l_base,
             l_free=l_free, v_free=v_free, th0=_W["th0"][:6])
    print(f"  wrote {OUTDIR / f'anatomy{tag}.npz'} "
          f"({(time.time()-t0)/60:.1f} min)")
    cmd_report(dev)


def cmd_report(dev=False):
    from scipy.stats import spearmanr
    tag = "_dev" if dev else ""
    d = np.load(OUTDIR / f"anatomy{tag}.npz")
    row, l_base, l_free, v_free = (d["row"], d["l_base"], d["l_free"],
                                   d["v_free"])
    th0 = d["th0"]
    imp = 1.0 - l_free / l_base[:, None]        # relative loss improvement
    delta = v_free - th0[None, :]
    print("exp41 stage 0 report — (a) individuality per component "
          "(median % loss improvement when that ONE component is freed "
          "per galaxy; bigger = carries more individuality):")
    order = np.argsort(-np.median(imp, axis=0))
    for j in order:
        q = np.percentile(delta[:, j], [16, 50, 84])
        print(f"    {NAMES[j]:6s}: {100*np.median(imp[:, j]):5.1f}%  |  "
              f"delta 16/50/84: {q[0]:+.2f}/{q[1]:+.2f}/{q[2]:+.2f} "
              f"(population value {th0[j]:.2f})")
    print("  (b) mutual Spearman correlations of the per-galaxy deltas "
          "(the anatomy subspace; only pairs with |rho| >= 0.3 shown):")
    for a in range(6):
        for b in range(a + 1, 6):
            r = spearmanr(delta[:, a], delta[:, b]).statistic
            if abs(r) >= 0.3:
                print(f"    {NAMES[a]} x {NAMES[b]}: {r:+.2f}")
    # (c) alignment with the exp39 per-galaxy core fraction
    pc = np.load(E39_DIR / "outputs/percore_gauss_z2.npz")
    fmap = {int(r): v for r, v in zip(pc["row"], pc["f_core"])}
    common = np.array([i for i, r in enumerate(row) if int(r) in fmap])
    lf = np.log10(np.clip([fmap[int(r)] for r in row[common]], 1e-4, 1.0))
    print("  (c) alignment: Spearman rho of each delta vs the exp39 "
          f"per-galaxy log10 f_core (n={len(common)}; |rho| high = the "
          "core-diversity axis is reachable inside the kernel):")
    for j in order:
        r = spearmanr(delta[common, j], lf).statistic
        print(f"    {NAMES[j]:6s}: {r:+.2f}")
    # (d) feature-predictability of the deltas
    pop = np.load(POP_NPZ)
    feats = {"logmh": pop["logmh"], "c200c": pop["c200c"],
             "fz2": pop["fz2"], "logms": pop["logms"],
             "logmh_z2": pop["logmh_zk_real"][:, 4]}
    print("  (d) feature-predictability (Spearman rho of each delta vs "
          "halo/galaxy features; the predictable part belongs to "
          "conditioning, the rest is the stochastic layer):")
    hdr = "    " + " " * 8 + "  ".join(f"{k:>9s}" for k in feats)
    print(hdr)
    for j in order:
        cells = []
        for k, v in feats.items():
            vv = np.asarray(v)[row]
            ok = np.isfinite(vv)
            cells.append(f"{spearmanr(delta[ok, j], vv[ok]).statistic:+9.2f}")
        print(f"    {NAMES[j]:6s}: " + "  ".join(cells))


def demo():
    rows = np.load(POP_NPZ)["dev100"][:8]
    _w_init(rows)
    s2 = _W["s2"]
    th0 = _W["th0"]
    g = _W["gals"][0]
    # (1) the official theta loads (12 params) and its base loss is finite
    assert th0.shape == (12,)
    l0 = s2.gal_loss(th0, g, KS_FIT, "1ch-mof")
    assert np.isfinite(l0) and l0 < 4.0
    # (2) freeing a component can only improve (or match) the loss, and
    # the fitted value stays inside the box
    gi, row, l_base, l_free, v_free = _anatomy_one((0,))
    assert (l_free <= l_base + 1e-12).all()
    assert (v_free >= LO6 - 1e-9).all() and (v_free <= HI6 + 1e-9).all()
    # (3) the exp39 percore file exists and shares rows with the sample
    pc = np.load(E39_DIR / "outputs/percore_gauss_z2.npz")
    assert len(set(pc["row"].tolist())
               & {g_["row"] for g_ in _W["gals"]}) > 0
    print("exp41 stage0 demo OK: official theta loaded; freed-component "
          f"losses never exceed the base (gal 0: base {l_base:.4f}, best "
          f"freed {l_free.min():.4f} at {NAMES[int(np.argmin(l_free))]}); "
          "percore alignment rows available")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run(dev)
    elif cmd == "report":
        cmd_report(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
