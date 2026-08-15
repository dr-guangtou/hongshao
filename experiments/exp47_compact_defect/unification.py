"""exp47 stage 1 — is the plane failure and the central-mass deficit ONE defect?

exp46 showed the model's population-plane failure lives on COMPACT
galaxies at every epoch. exp45 showed compactness and central mass are
near-deterministically linked (per-galaxy Spearman between dlog R50 and
the M*(<5 kpc) bias = -0.89 to -0.97 at every epoch). The missing link is
whether the PLANE residual has the same driver.

`qa.plane_energy` is a population statistic with no per-galaxy value, so
the per-galaxy object here is the DISPLACEMENT VECTOR on the plane:
(dlog M*(<30), dlog M*(30-50)) = model - truth, decomposed against the
truth's own fitted relation into

  along-ridge   moves the galaxy along the mass sequence (the centered
                energy statistic is largely insensitive to this)
  across-ridge  moves it OFF the relation — precisely what the centered
                statistic penalizes

No fitting. Run: PYTHONPATH=. uv run python \
experiments/exp47_compact_defect/unification.py {demo|run}
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp46_highz_ridge"))

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E38 = ROOT / "experiments/exp38_deposit_rethink/outputs"
E40 = ROOT / "experiments/exp40_epoch_objective/outputs/latestart.npz"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"

KS_ALL = [0, 1, 2, 3, 4]
Z_GRID = (0.4, 0.7, 1.0, 1.5, 2.0)
PLANE = ("kpc:M(<30)", "kpc:M(30-50)")
PLANE2 = ("kpc:M(<30)", "kpc:M(50-100)")
R_COMPACT = 5.0

_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    s2 = _load_by_path("exp38_stage2", E38.parent / "stage2_multiepoch.py")
    s2._w_init(rows)
    _W.update(s2._W)
    _W["s2"] = s2


def _cogs_chunk(bounds):
    lo, hi = bounds
    s2, gals = _W["s2"], _W["gals"]
    theta = np.load(E40)["theta_z15"]
    out = np.full((hi - lo, len(KS_ALL), len(_W["e"].R)), np.nan)
    for i in range(lo, hi):
        c = s2.model_cogs(theta, gals[i], KS_ALL, "1ch-mof")
        if c is not None:
            out[i - lo] = c
    return lo, out


def adopted_model_cogs(data_cogs, rows, dev=False):
    n = len(data_cogs)
    workers = max(os.cpu_count() - 2, 2)
    step = max(n // (4 * workers) + 1, 1)
    bounds = [(lo, min(lo + step, n)) for lo in range(0, n, step)]
    out = np.full_like(data_cogs, np.nan)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for lo, blk in pool.map(_cogs_chunk, bounds):
            out[lo:lo + len(blk)] = blk
    return out


def ridge_decompose(tx, ty, mx, my):
    """Split the per-galaxy plane displacement into along/across the
    TRUTH's fitted relation.

    The basis comes from the truth, not the model, so the decomposition
    does not move when the model changes — otherwise 'across-ridge' would
    mean something different for every model tested.
    """
    ok = np.isfinite(tx) & np.isfinite(ty) & np.isfinite(mx) & np.isfinite(my)
    slope, _ = np.polyfit(tx[ok], ty[ok], 1)
    u = np.array([1.0, slope]) / np.hypot(1.0, slope)      # along the ridge
    v = np.array([-slope, 1.0]) / np.hypot(1.0, slope)     # across it
    d = np.column_stack([mx - tx, my - ty])
    along = d @ u
    across = d @ v
    for a in (along, across):
        a[~ok] = np.nan
    return along, across, float(slope)


def cmd_run(dev=False):
    from hongshao import qa
    from compact_anatomy import compactness, spearman

    rows = np.load(POP_NPZ)["dev100"] if dev else None
    tag = "_dev" if dev else ""
    _w_init(rows)
    gals, R = _W["gals"], _W["e"].R
    data_cogs = np.stack([g["data"] for g in gals])
    n = len(gals)
    print(f"exp47 stage 1 — one defect or two? (n={n}"
          f"{', DEV' if dev else ''})", flush=True)

    t0 = time.time()
    model_cogs = adopted_model_cogs(data_cogs, rows, dev)
    print(f"  adopted-theta model CoGs built ({time.time() - t0:.0f}s)",
          flush=True)

    truth, model, _, _ = qa.measure_all(model_cogs, data_cogs, R)
    r50_t = qa._safe_rhalf(data_cogs, R, 0.5)
    m5_t = np.array([[np.interp(5.0, R, data_cogs[i, j]) for j in range(5)]
                     for i in range(n)])
    m5_m = np.array([[np.interp(5.0, R, model_cogs[i, j]) for j in range(5)]
                     for i in range(n)])
    bias5 = np.log10(np.clip(m5_m, 1.0, None)) - np.log10(np.clip(m5_t, 1.0,
                                                                  None))
    c_in = np.log10(
        np.array([[np.interp(10.0, R, data_cogs[i, j]) for j in range(5)]
                  for i in range(n)])
        / np.clip(data_cogs[:, :, -1], 1.0, None))

    print("\n  READOUT 1 — is the plane residual the central-mass error?"
          "\n  Spearman of the per-galaxy plane displacement against the "
          "M*(<5 kpc) bias.")
    res = {}
    for pname, (kx, ky) in (("M(<30) vs M(30-50)", PLANE),
                            ("M(<30) vs M(50-100)", PLANE2)):
        print(f"\n    {pname}")
        print(f"      {'z':>5}{'ridge slope':>13}{'rho(across, '
                  'M5 bias)':>24}{'rho(along, M5 bias)':>21}")
        for j, z in enumerate(Z_GRID):
            tx = np.log10(np.clip(truth[kx][:, j], 1.0, None))
            ty = np.log10(np.clip(truth[ky][:, j], 1.0, None))
            mx = np.log10(np.clip(model[kx][:, j], 1.0, None))
            my = np.log10(np.clip(model[ky][:, j], 1.0, None))
            along, across, sl = ridge_decompose(tx, ty, mx, my)
            r_ac = spearman(across, bias5[:, j])
            r_al = spearman(along, bias5[:, j])
            res[(pname, j)] = (sl, r_ac, r_al,
                               float(np.nanmedian(np.abs(across))),
                               float(np.nanmedian(np.abs(along))))
            print(f"      {z:5.1f}{sl:13.2f}{r_ac:24.2f}{r_al:21.2f}")

    print("\n  READOUT 2 — does the same residual track COMPACTNESS?")
    print(f"      {'z':>5}{'rho(across, R50)':>19}{'rho(across, c_in)':>20}"
          f"{'rho(|across|, R50)':>21}")
    for j, z in enumerate(Z_GRID):
        kx, ky = PLANE
        tx = np.log10(np.clip(truth[kx][:, j], 1.0, None))
        ty = np.log10(np.clip(truth[ky][:, j], 1.0, None))
        mx = np.log10(np.clip(model[kx][:, j], 1.0, None))
        my = np.log10(np.clip(model[ky][:, j], 1.0, None))
        _, across, _ = ridge_decompose(tx, ty, mx, my)
        print(f"      {z:5.1f}{spearman(across, r50_t[:, j]):19.2f}"
              f"{spearman(across, c_in[:, j]):20.2f}"
              f"{spearman(np.abs(across), r50_t[:, j]):21.2f}")

    print("\n  READOUT 3 — how much of the plane energy do compact "
          "galaxies contribute?\n    centered E/floor with the compact "
          "group REMOVED, vs a size-matched random removal:")
    print(f"      {'z':>5}{'all':>8}{'drop compact':>15}{'drop random':>14}"
          f"{'n compact':>11}")
    rng = np.random.default_rng(0)
    kx, ky = PLANE
    for j, z in enumerate(Z_GRID):
        def score(mask):
            t = np.column_stack([
                np.log10(np.clip(truth[kx][mask, j], 1.0, None)),
                np.log10(np.clip(truth[ky][mask, j], 1.0, None))])
            m = np.column_stack([
                np.log10(np.clip(model[kx][mask, j], 1.0, None)),
                np.log10(np.clip(model[ky][mask, j], 1.0, None))])
            return qa.plane_energy(t, m)["energy_ratio_centered"]
        comp = r50_t[:, j] < R_COMPACT
        e_all = score(np.ones(n, bool))
        e_drop = score(~comp)
        rnd = []
        for _ in range(16):
            m = np.ones(n, bool)
            m[rng.choice(n, size=int(comp.sum()), replace=False)] = False
            rnd.append(score(m))
        print(f"      {z:5.1f}{e_all:8.1f}{e_drop:15.1f}"
              f"{np.median(rnd):14.1f}{int(comp.sum()):11d}")

    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / f"unification{tag}.npz",
             bias5=bias5, r50_truth=r50_t, c_in=c_in,
             keys=np.array([f"{p}|{j}" for (p, j) in res]),
             stats=np.array([res[k] for k in res]))
    print(f"\n  wrote {OUTDIR / f'unification{tag}.npz'} "
          f"({time.time() - t0:.0f}s)")


def demo():
    # the ridge decomposition must be exact on a planted displacement
    rng = np.random.default_rng(0)
    tx = rng.normal(11.5, 0.3, 800)
    ty = 1.5 * tx - 6.0 + rng.normal(0, 0.05, 800)
    slope = 1.5
    u = np.array([1.0, slope]) / np.hypot(1.0, slope)
    v = np.array([-slope, 1.0]) / np.hypot(1.0, slope)
    # push every galaxy a known amount along, and a known amount across
    a_true, c_true = 0.07, -0.04
    d = a_true * u + c_true * v
    along, across, sl = ridge_decompose(tx, ty, tx + d[0], ty + d[1])
    assert abs(sl - slope) < 0.02, sl
    assert abs(np.median(along) - a_true) < 5e-3, np.median(along)
    assert abs(np.median(across) - c_true) < 5e-3, np.median(across)
    # a pure along-ridge push must leave 'across' at zero
    d2 = 0.2 * u
    _, across2, _ = ridge_decompose(tx, ty, tx + d2[0], ty + d2[1])
    assert abs(np.median(across2)) < 5e-3, np.median(across2)
    # and the basis is the TRUTH's, so a model-only change cannot rotate it
    _, _, sl2 = ridge_decompose(tx, ty, tx + 0.5, ty - 0.5)
    assert abs(sl2 - sl) < 1e-9
    print(f"demo OK — ridge decomposition exact (recovered along "
          f"{np.median(along):+.3f} / across {np.median(across):+.3f} "
          f"against planted {a_true:+.3f} / {c_true:+.3f})")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run(dev="--dev" in sys.argv)
    else:
        raise SystemExit(__doc__)
