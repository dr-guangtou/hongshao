"""exp42 — the adopted 1ch-mof kernel fitted on the REAL de-dipped MAH.

The same 12-parameter model, the same official z<=1.5 scope and protocol
(exp40 `latestart`/`latecv`), with the loader config switched from
"diffmah" (the smooth reconstructed history) to "real" (the de-dipped
main-branch peak history, which keeps merger bursts). Model code is exp38
``stage2_multiepoch.py`` unchanged; this script only re-initializes the
worker state with the real-MAH galaxies and reuses the exp40 report
helpers (bias table, physics lines) on that state.

Run: PYTHONPATH=. uv run python experiments/exp42_real_mah/compare.py \
    {demo|fit|physics|cv} [--dev]
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
sys.path.insert(0, str(ROOT))

OUTDIR = HERE / "outputs"
E35_DIR = ROOT / "experiments/exp35_total_norm"
E38_DIR = ROOT / "experiments/exp38_deposit_rethink"
E40_DIR = ROOT / "experiments/exp40_epoch_objective"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_ALL = [0, 1, 2, 3, 4]
KS_Z15 = [0, 1, 2, 3]
NFOLD = 10
CONFIG = "real"
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    """The exp38 stage-2 worker state with config = "real": exp35's loader
    chain re-initialized on the de-dipped real MAH, the conditioning
    vector reattached, and stage2's module state pointed at it — so
    ``s2.model_cogs``/``gal_loss``/``penalty`` run unchanged. The exp40
    report helpers get the same state (spawn-safe: runs fully in each
    worker)."""
    s2 = _load_by_path("exp38_stage2", E38_DIR / "stage2_multiepoch.py")
    e = _load_by_path("exp35_run", E35_DIR / "run.py")
    mh_scale = tuple(np.load(ROOT / "experiments/exp32_full_population/"
                             "outputs/um_slope_diffmah.npz")["mh_scale"])
    e._init(CONFIG, mh_scale, None if rows is None else np.asarray(rows))
    pop = np.load(POP_NPZ)
    z_all = np.column_stack([pop["logmh"], pop["c200c"], pop["fz2"]])
    mu = np.nanmean(z_all, axis=0)
    sd = np.nanstd(z_all, axis=0)
    for g in e._G["gals"]:
        z = (z_all[g["row"]] - mu) / sd
        g["cond"] = np.where(np.isfinite(z), z, 0.0)
    s2._W["e"] = e
    s2._W["gals"] = e._G["gals"]
    t40 = _load_by_path("exp40_tests", E40_DIR / "tests.py")
    t40._W.update(e=e, gals=e._G["gals"], s2=s2)
    _W.update(e=e, gals=e._G["gals"], s2=s2, t40=t40)


def _chunk_plain(args):
    p, ks, lo, hi = args
    s2 = _W["s2"]
    return sum(s2.gal_loss(p, g, ks, "1ch-mof") for g in _W["gals"][lo:hi])


def _official_theta():
    return np.load(E40_DIR / "outputs/latestart.npz")["theta_z15"]


def cmd_fit(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    s2 = _W["s2"]
    t40 = _W["t40"]
    tag = "_dev" if dev else ""
    scale = 0.15 if dev else 1.0
    maxiter = max(int(5000 * scale), 80)
    workers = max(os.cpu_count() - 2, 2)
    n = len(_W["gals"])
    edges = np.linspace(0, n, workers + 1).astype(int)
    official = _official_theta()
    OUTDIR.mkdir(exist_ok=True)
    print(f"exp42 real-MAH z<=1.5 fit (n={n}{', DEV' if dev else ''}; "
          "marks: DiffMAH loss 0.1538, params [2.74, 4.00, 0.91, 1.52, "
          "0.24, 1.38]):", flush=True)
    nudge = official.copy()
    nudge[2] = max(official[2] - 0.5, 0.05)         # weaker envelope q
    starts = [official, nudge]
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        def loss(p):
            parts = pool.map(_chunk_plain,
                             [(p, KS_Z15, edges[i], edges[i + 1])
                              for i in range(workers)])
            return sum(parts) / n + s2.penalty(p, "1ch-mof")

        best = None
        for p0 in starts:
            t0 = time.time()
            r = minimize(loss, p0, method="Nelder-Mead",
                         options=dict(maxiter=maxiter, xatol=3e-4,
                                      fatol=1e-8))
            print(f"  [real-z15] start: loss {r.fun:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            if best is None or r.fun < best.fun:
                best = r
    th = best.x
    ab = s2._at_bound(th, "1ch-mof")
    print(f"  [real-z15] params: {np.round(th[:6], 2)}; at bound: "
          f"{', '.join(ab) if ab else 'NONE'}")
    t40._bias_table(th, "real-z15", KS_Z15)
    np.savez(OUTDIR / f"realfit{tag}.npz", theta_real_z15=th,
             loss_real_z15=best.fun)
    print(f"  wrote {OUTDIR / f'realfit{tag}.npz'}", flush=True)


def cmd_physics(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    t40 = _W["t40"]
    tag = "_dev" if dev else ""
    official = _official_theta()
    print("exp42 physics on the REAL-MAH input (marks: data 0.37/0.11; "
          "DiffMAH-input official fit 0.40/0.13, overshoot T1 "
          "+0.025/+0.028):")
    t40._bias_table(official, "official-theta TRANSFER (no refit)", KS_Z15)
    t40._physics_lines(official, "official-theta TRANSFER", KS_Z15)
    fit_npz = OUTDIR / f"realfit{tag}.npz"
    if fit_npz.exists():
        th = np.load(fit_npz)["theta_real_z15"]
        t40._physics_lines(th, "real-z15 refit", KS_Z15)
    else:
        print(f"    (no {fit_npz.name} yet — run `fit` first for the "
              "refit lines)")


def _cv_fold(args):
    fold, warm, maxiter = args
    s2 = _W["s2"]
    e = _W["e"]
    gals = _W["gals"]
    n = len(gals)
    train = [i for i in range(n) if i % NFOLD != fold]

    def tr_loss(p):
        return (np.mean([s2.gal_loss(p, gals[i], KS_Z15, "1ch-mof")
                         for i in train]) + s2.penalty(p, "1ch-mof"))
    r = minimize(tr_loss, warm, method="Nelder-Mead",
                 options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
    i5 = int(np.searchsorted(e.R, 5.0))
    i10 = int(np.searchsorted(e.R, 10.0))
    m = e.R > 5.0
    out = []
    for i in range(n):
        if i % NFOLD != fold:
            continue
        g = gals[i]
        cogs = s2.model_cogs(r.x, g, KS_ALL, "1ch-mof")
        if cogs is None:
            out.append((g["row"], np.full((5, 3), np.nan),
                        np.full((5, len(e.R)), np.nan)))
            continue
        met = []
        for c, k in zip(cogs, KS_ALL):
            d = g["data"][k]
            cs = c * (d[-1] / c[-1])
            met.append([np.abs(cs[m] / d[m] - 1).max(),
                        c[i5] / d[i5] - 1, c[i10] / d[i10] - 1])
        out.append((g["row"], np.array(met), np.array(cogs)))
    return out


def cmd_cv(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    gals = _W["gals"]
    n = len(gals)
    e = _W["e"]
    warm = np.load(OUTDIR / f"realfit{tag}.npz")["theta_real_z15"]
    workers = min(max(os.cpu_count() - 2, 2), NFOLD)
    maxiter = 150 if dev else 1500
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    met = np.full((n, 5, 3), np.nan)
    cogs = np.full((n, 5, len(e.R)), np.nan)
    t0 = time.time()
    print(f"exp42 real-MAH 10-fold CV [z15 scope] (n={n}"
          f"{', DEV' if dev else ''}; * = extrapolated epoch; DiffMAH "
          "marks 18.2/17.4/16.6/16.4/15.4*, M(<5) -9.5%):", flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        jobs = [(f, warm, maxiter) for f in range(NFOLD)]
        for part in pool.map(_cv_fold, jobs):
            for row, m_, c_ in part:
                met[row_to_i[row]] = m_
                cogs[row_to_i[row]] = c_
    line = " ".join(
        f"z{e.ANCHOR_Z[k]}{'' if k in KS_Z15 else '*'}: "
        f"{100*np.nanmedian(met[:, k, 0]):.1f} | "
        f"{100*np.nanmedian(met[:, k, 1]):+.1f} | "
        f"{100*np.nanmedian(met[:, k, 2]):+.1f}"
        for k in range(5))
    print(f"  [real-z15] held-out shape R>5 | M(<5) | M(<10): {line}  "
          f"({(time.time()-t0)/60:.1f} min)")
    np.savez(OUTDIR / f"cv_real{tag}.npz", met=met, cogs=cogs)
    print(f"  wrote {OUTDIR / f'cv_real{tag}.npz'}")


def demo():
    rows = np.load(POP_NPZ)["dev100"][:12]
    _w_init(rows)
    s2 = _W["s2"]
    e = _W["e"]
    gals = _W["gals"]
    official = _official_theta()

    # (1) the real-MAH deposit dict has the exp35 keys, and no galaxy is
    # silently dropped on the dev rows
    for key in ("z", "t", "dMh", "snap", "t_obs"):
        assert key in gals[0]["mah"], key
    assert len(gals) == 12, f"real-MAH loader dropped rows: {len(gals)}/12"

    # (2) the real history is burstier than the smooth one: the largest
    # single-step share of the deposit budget is larger for every galaxy
    um = e.um
    for g in gals[:6]:
        pop_index = int(np.load(POP_NPZ)["index"][g["row"]])
        smooth = um.MAHFUN["diffmah"](pop_index)
        frac_real = (g["mah"]["dMh"].max() / g["mah"]["dMh"].sum())
        frac_smooth = (smooth["dMh"].max() / smooth["dMh"].sum())
        assert frac_real > frac_smooth, (frac_real, frac_smooth)

    # (3) the unchanged stage-2 model runs on the real input: finite,
    # monotone, strictly inside the M(<500) pin
    for g in gals[:6]:
        cogs = s2.model_cogs(official, g, KS_ALL, "1ch-mof")
        assert cogs is not None
        for k, c in zip(KS_ALL, cogs):
            assert np.isfinite(c).all() and np.all(np.diff(c) > -1e-9)
            assert c[-1] < g["m500"][k]

    # (4) the loss decomposes exactly over the z<=1.5 subset (the exp40
    # identity, now on the real input)
    g = gals[0]
    lo = s2.gal_loss(official, g, KS_Z15, "1ch-mof")
    cogs = s2.model_cogs(official, g, KS_Z15, "1ch-mof")
    manual = np.mean([np.sqrt(np.mean(
        ((c - g["data"][k]) / g["data"][k]) ** 2))
        for c, k in zip(cogs, KS_Z15)])
    assert abs(lo - manual) < 1e-12

    print("exp42 demo OK: real-MAH loader keys match, no dev-row drops, "
          "real histories are burstier than the smooth fits on every "
          "checked galaxy, the unchanged 1ch-mof runs finite/monotone/"
          "pinned on the real input, and the subset loss decomposes "
          "exactly")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev)
    elif cmd == "physics":
        cmd_physics(dev)
    elif cmd == "cv":
        cmd_cv(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
