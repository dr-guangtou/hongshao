"""exp40 — two scope tests on the adopted kernel (user plan, 2026-07-18).

Test 1 `latestart` — fit the SAME adopted 1ch-mof model jointly from a
LATER starting epoch: z<=1.5 (epochs 0.4/0.7/1.0/1.5) and z<=1.0
(0.4/0.7/1.0), instead of the fiducial z<=2.0 five-epoch fit. The
physical motivation: at z~2 some galaxies still form stars and mergers
can be dissipative (induce star formation) — regimes the additive
collisionless transport model does not describe; forcing the model to
connect z=2 to z=0.4 may be what drains the core. The record brackets
the answer: the full z=2-start joint fit puts the z=0.4 inner bias at
M(<5) = -11.7% while the z=0.4-only population fit sits at -7.6% (the
pure population-sharing wall) — the late starts measure how much of
the ~4-point transport tax is paid to the z>=1.5 era, and whether the
z=1.5- and z=1.0-start fits agree with each other (parameters, biases,
physics marks on shared epoch pairs).

Test 2 `reaim` / `reaimcv` — complete the measurement of the exp38
"capacity" operating point: the FIDUCIAL 12-parameter kernel (no core
channel) refit under the inner-aware objective (R>5 shape +
M(<5)/M(<10) aperture terms; the observationally honest loss — the
inner profile shape is not trusted, the inner aperture masses are).
exp38 measured only the fit itself (M(<5) -4.8/-3.6%, mu railed at
3.0); the differential test, the outskirt terciles, and held-out CV
were never run. exp39's prediction: the re-balancing is
OBJECTIVE-driven, so the re-aimed fiducial should break the
differential with no added component at all. Either outcome is
informative.

Judged marks (unchanged): differential massive z0.7->0.4 data
0.37/0.11 (adopted kernel 0.39/0.12), outskirt T1 +0.026/+0.019, no
parameter at a bound, held-out shape 18.5-14.2% by epoch.

Run: PYTHONPATH=. uv run python experiments/exp40_epoch_objective/\
tests.py {demo|latestart|latephysics|reaim|reaimcv} [--dev]
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
E38_DIR = ROOT / "experiments/exp38_deposit_rethink"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS_ALL = [0, 1, 2, 3, 4]
SCOPES = {"z15": [0, 1, 2, 3], "z10": [0, 1, 2]}
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
    _W["s3"] = s3


def _i_at(r_kpc):
    return int(np.searchsorted(_W["e"].R, r_kpc))


def _chunk_plain(args):
    p, ks, lo, hi = args
    s2 = _W["s2"]
    return sum(s2.gal_loss(p, g, ks, "1ch-mof") for g in _W["gals"][lo:hi])


def _chunk_inner40(args):
    p, ks, lo, hi = args
    s3 = _W["s3"]
    return sum(s3.gal_loss_inner(p, g, ks, "1ch-mof")
               for g in _W["gals"][lo:hi])


def _bias_table(th, label, fitted_ks):
    """Inner bias + pinned shape at every epoch, marking extrapolated
    epochs (the model predicts all epochs; only fitted_ks entered the
    loss)."""
    e = _W["e"]
    s2 = _W["s2"]
    gals = _W["gals"]
    i5, i10 = _i_at(5.0), _i_at(10.0)
    m = e.R > 5.0
    rows = []
    for g in gals:
        cogs = s2.model_cogs(th, g, KS_ALL, "1ch-mof")
        if cogs is None:
            continue
        per = []
        for c, k in zip(cogs, KS_ALL):
            d = g["data"][k]
            cs = c * (d[-1] / c[-1])
            per.append([c[i5] / d[i5] - 1, c[i10] / d[i10] - 1,
                        np.abs(cs[m] / d[m] - 1).max()])
        rows.append(per)
    arr = np.median(np.array(rows), axis=0)
    print(f"    [{label}] in-sample per epoch "
          "(M<5 % | M<10 % | pinned shape R>5 %; * = extrapolated, "
          "not in the fit):")
    for j, k in enumerate(KS_ALL):
        star = " " if k in fitted_ks else "*"
        print(f"      z={e.ANCHOR_Z[k]}{star}: {100*arr[j, 0]:+.1f} | "
              f"{100*arr[j, 1]:+.1f} | {100*arr[j, 2]:.1f}")
    return arr


def _physics_lines(th, label, fitted_ks):
    """Differential pairs + z=0.4 overshoot terciles for one theta
    (in-sample), marking pairs that involve an unfitted epoch."""
    from hongshao.profile_emulator import density_from_cog
    e = _W["e"]
    s2 = _W["s2"]
    gals = _W["gals"]
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    cogs = np.full((len(gals), 5, len(e.R)), np.nan)
    for i, g in enumerate(gals):
        out = s2.model_cogs(th, g, KS_ALL, "1ch-mof")
        if out is not None:
            cogs[i] = out
    ed3, rows_d = e.differential(cogs, data, logms)
    cells = []
    for k in range(4):
        star = "" if (k in fitted_ks and (k + 1) in fitted_ks) else "*"
        cells.append(f"z{e.ANCHOR_Z[k+1]}->z{e.ANCHOR_Z[k]}{star}: "
                     f"{rows_d[('data', 2, k)][0]:.2f}/"
                     f"{rows_d[('data', 2, k)][1]:.2f} -> "
                     f"{rows_d[('model', 2, k)][0]:.2f}/"
                     f"{rows_d[('model', 2, k)][1]:.2f}")
    print(f"    [{label}] differential massive tercile "
          "(data -> model; * = pair includes an unfitted epoch):")
    print("      " + "  ".join(cells))
    cogs0 = cogs[:, 0]
    ok = np.isfinite(cogs0).all(1) & (cogs0 > 0).all(1)
    ls_d, mid = density_from_cog(np.log10(data[:, 0]), e.R)
    ls_m = np.full_like(ls_d, np.nan)
    ls_m[ok] = density_from_cog(np.log10(cogs0[ok]), e.R)[0]
    dl = ls_m - ls_d
    bands = ((mid >= 30.0) & (mid < 60.0), (mid >= 60.0) & (mid <= 148.0))
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    cells = []
    for b in range(3):
        sel = ok & (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
        cells.append(" ".join(f"{np.nanmedian(dl[np.ix_(sel, bm)]):+.3f}"
                              for bm in bands))
    print("    overshoot terciles (in-sample, z=0.4, [30-60 / 60-148 kpc]"
          "; kernel held-out marks T1 +0.026/+0.019): "
          + "  |  ".join(f"T{b+1} {c}" for b, c in enumerate(cells)))


# --------------------------------------------------------------------------- #
# test 1 — late-start joint fits                                               #
# --------------------------------------------------------------------------- #
def cmd_latestart(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    _W["rows_arg"] = rows
    s2 = _W["s2"]
    tag = "_dev" if dev else ""
    scale = 0.15 if dev else 1.0
    maxiter = max(int(5000 * scale), 80)
    workers = max(os.cpu_count() - 2, 2)
    n = len(_W["gals"])
    edges = np.linspace(0, n, workers + 1).astype(int)
    d2 = np.load(E38_DIR / "outputs/stage2_multiepoch.npz")
    adopted = d2["theta_1ch-mof"]
    names = ["log_rc", "g", "q", "mu", "sig", "gamma"]
    out_path = OUTDIR / f"latestart{tag}.npz"
    OUTDIR.mkdir(exist_ok=True)
    fitted = dict(np.load(out_path)) if out_path.exists() else {}
    print(f"exp40 late-start fits (n={n}{', DEV' if dev else ''}; plain "
          "loss, adopted-kernel structure; brackets: z=2-start M(<5) "
          "-11.7%, z=0.4-only -7.6%):", flush=True)
    print(f"    adopted (z2-start) params: {np.round(adopted[:6], 2)} "
          f"({'/'.join(names)})")
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        for scope, ks in SCOPES.items():
            nudge = adopted.copy()
            nudge[2] = max(adopted[2] - 0.5, 0.05)      # weaker envelope q
            starts = [adopted, nudge]

            def loss(p):
                parts = pool.map(_chunk_plain,
                                 [(p, ks, edges[i], edges[i + 1])
                                  for i in range(workers)])
                return sum(parts) / n + s2.penalty(p, "1ch-mof")

            best = None
            for p0 in starts:
                t0 = time.time()
                r = minimize(loss, p0, method="Nelder-Mead",
                             options=dict(maxiter=maxiter, xatol=3e-4,
                                          fatol=1e-8))
                print(f"  [{scope}] start: loss {r.fun:.4f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
                if best is None or r.fun < best.fun:
                    best = r
            th = best.x
            ab = s2._at_bound(th, "1ch-mof")
            print(f"  [{scope}] params: {np.round(th[:6], 2)}; at bound: "
                  f"{', '.join(ab) if ab else 'NONE'}")
            _bias_table(th, scope, ks)
            fitted[f"theta_{scope}"] = th
            fitted[f"loss_{scope}"] = best.fun
            np.savez(out_path, **fitted)
            print(f"  wrote {out_path} (+{scope})", flush=True)


def cmd_latephysics(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    d = np.load(OUTDIR / f"latestart{tag}.npz")
    d2 = np.load(E38_DIR / "outputs/stage2_multiepoch.npz")
    print("exp40 late-start physics (marks: data 0.37/0.11, adopted "
          "kernel 0.39/0.12):")
    _physics_lines(d2["theta_1ch-mof"], "adopted-z20", KS_ALL)
    for scope, ks in SCOPES.items():
        if f"theta_{scope}" in d.files:
            _physics_lines(d[f"theta_{scope}"], scope, ks)


def cmd_latestress(dev=False):
    """The exp35/38 bounds-stress protocol on the late-start g = 4.00
    rail: loosen the width time-exponent box (4.0 -> 6.0), refit the
    z<=1.5 scope, and ask HOW the freedom is spent — loss, parameters,
    inner bias, and the physics tests. (Penalties are parent-side, so
    the loosened box reaches the fit; workers compute raw loss only.)"""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    s2 = _W["s2"]
    e = _W["e"]
    tag = "_dev" if dev else ""
    d = np.load(OUTDIR / f"latestart{tag}.npz")
    warm = d["theta_z15"]
    nudge = warm.copy()
    nudge[1] = 5.0
    ks = SCOPES["z15"]
    scale = 0.15 if dev else 1.0
    maxiter = max(int(4000 * scale), 80)
    workers = max(os.cpu_count() - 2, 2)
    n = len(_W["gals"])
    edges = np.linspace(0, n, workers + 1).astype(int)
    e.HI = np.array([3.0, 6.0, 3.0, 3.0, 2.0])
    print(f"exp40 late-start g-rail stress (z<=1.5 scope, g box 4 -> 6; "
          f"n={n}{', DEV' if dev else ''}; baseline loss "
          f"{float(d['loss_z15']):.4f}):", flush=True)
    try:
        with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
            def loss(p):
                parts = pool.map(_chunk_plain,
                                 [(p, ks, edges[i], edges[i + 1])
                                  for i in range(workers)])
                return sum(parts) / n + s2.penalty(p, "1ch-mof")

            best = None
            for p0 in (warm, nudge):
                t0 = time.time()
                r = minimize(loss, p0, method="Nelder-Mead",
                             options=dict(maxiter=maxiter, xatol=3e-4,
                                          fatol=1e-8))
                print(f"  [stress] start: loss {r.fun:.4f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
                if best is None or r.fun < best.fun:
                    best = r
        th = best.x
        ab = s2._at_bound(th, "1ch-mof")
        print(f"  [stress] params: {np.round(th[:6], 2)}; at bound: "
              f"{', '.join(ab) if ab else 'NONE'}")
        _bias_table(th, "stress", ks)
        _physics_lines(th, "stress", ks)
        np.savez(OUTDIR / f"latestress{tag}.npz", theta=th, loss=best.fun)
        print(f"  wrote {OUTDIR / f'latestress{tag}.npz'}", flush=True)
    finally:
        e.HI = np.array([3.0, 4.0, 3.0, 3.0, 2.0])


def _cv_fold_late(args):
    fold, ks, warm, maxiter = args
    s2 = _W["s2"]
    gals = _W["gals"]
    n = len(gals)
    train = [i for i in range(n) if i % 10 != fold]

    def tr_loss(p):
        return (np.mean([s2.gal_loss(p, gals[i], ks, "1ch-mof")
                         for i in train]) + s2.penalty(p, "1ch-mof"))
    r = minimize(tr_loss, warm, method="Nelder-Mead",
                 options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
    e = _W["e"]
    i5, i10 = _i_at(5.0), _i_at(10.0)
    m = e.R > 5.0
    out = []
    for i in range(n):
        if i % 10 != fold:
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


def cmd_latecv(scope="z15", dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    gals = _W["gals"]
    n = len(gals)
    e = _W["e"]
    ks = SCOPES[scope]
    warm = np.load(OUTDIR / f"latestart{tag}.npz")[f"theta_{scope}"]
    workers = min(max(os.cpu_count() - 2, 2), 10)
    maxiter = 150 if dev else 1500
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    met = np.full((n, 5, 3), np.nan)
    cogs = np.full((n, 5, len(e.R)), np.nan)
    t0 = time.time()
    print(f"exp40 late-start 10-fold CV [{scope}] (n={n}"
          f"{', DEV' if dev else ''}; * = epoch not in the fold fits; "
          "stage-2 held-out marks 18.5/17.6/16.7/16.2/14.2, M(<5) ~-11):",
          flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        jobs = [(f, ks, warm, maxiter) for f in range(10)]
        for part in pool.map(_cv_fold_late, jobs):
            for row, m_, c_ in part:
                met[row_to_i[row]] = m_
                cogs[row_to_i[row]] = c_
    line = " ".join(
        f"z{e.ANCHOR_Z[k]}{'' if k in ks else '*'}: "
        f"{100*np.nanmedian(met[:, k, 0]):.1f} | "
        f"{100*np.nanmedian(met[:, k, 1]):+.1f} | "
        f"{100*np.nanmedian(met[:, k, 2]):+.1f}"
        for k in range(5))
    print(f"  [{scope}] held-out shape R>5 | M(<5) | M(<10): {line}  "
          f"({(time.time()-t0)/60:.1f} min)")
    np.savez(OUTDIR / f"cv_{scope}{tag}.npz", met=met, cogs=cogs)
    print(f"  wrote {OUTDIR / f'cv_{scope}{tag}.npz'}")


# --------------------------------------------------------------------------- #
# test 2 — the re-aimed fiducial, judged                                       #
# --------------------------------------------------------------------------- #
def cmd_reaim(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    s2 = _W["s2"]
    dc = np.load(E38_DIR / "outputs/stage3_capacity.npz")
    th = dc["theta_inner_1ch-mof"]
    print("exp40 re-aimed fiducial (the exp38 capacity theta: the "
          "12-param kernel under the inner-aware objective), judged:")
    ab = s2._at_bound(th, "1ch-mof")
    print(f"    params: {np.round(th[:6], 2)}; at bound: "
          f"{', '.join(ab) if ab else 'NONE'}")
    _bias_table(th, "reaim", KS_ALL)
    _physics_lines(th, "reaim", KS_ALL)


def _cv_fold_reaim(args):
    fold, warm, maxiter = args
    s3 = _W["s3"]
    s2 = _W["s2"]
    gals = _W["gals"]
    n = len(gals)
    train = [i for i in range(n) if i % 10 != fold]

    def tr_loss(p):
        return (np.mean([s3.gal_loss_inner(p, gals[i], KS_ALL, "1ch-mof")
                         for i in train]) + s2.penalty(p, "1ch-mof"))
    r = minimize(tr_loss, warm, method="Nelder-Mead",
                 options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
    e = _W["e"]
    i5, i10 = _i_at(5.0), _i_at(10.0)
    m = e.R > 5.0
    out = []
    for i in range(n):
        if i % 10 != fold:
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


def cmd_reaimcv(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    tag = "_dev" if dev else ""
    gals = _W["gals"]
    n = len(gals)
    e = _W["e"]
    warm = np.load(E38_DIR / "outputs/stage3_capacity.npz"
                   )["theta_inner_1ch-mof"]
    workers = min(max(os.cpu_count() - 2, 2), 10)
    maxiter = 150 if dev else 1500
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    met = np.full((n, 5, 3), np.nan)
    cogs = np.full((n, 5, len(e.R)), np.nan)
    t0 = time.time()
    print(f"exp40 re-aimed fiducial 10-fold CV (n={n}"
          f"{', DEV' if dev else ''}; marks: stage-2 held-out shape "
          "18.5-14.2%, M(<5) ~-11%):", flush=True)
    with Pool(workers, initializer=_w_init, initargs=(rows,)) as pool:
        jobs = [(f, warm, maxiter) for f in range(10)]
        for part in pool.map(_cv_fold_reaim, jobs):
            for row, m_, c_ in part:
                met[row_to_i[row]] = m_
                cogs[row_to_i[row]] = c_
    line = " ".join(
        f"z{e.ANCHOR_Z[k]}: {100*np.nanmedian(met[:, k, 0]):.1f} | "
        f"{100*np.nanmedian(met[:, k, 1]):+.1f} | "
        f"{100*np.nanmedian(met[:, k, 2]):+.1f}"
        for k in range(5))
    print(f"  [reaim] held-out shape R>5 | M(<5) | M(<10): {line}  "
          f"({(time.time()-t0)/60:.1f} min)")
    np.savez(OUTDIR / f"cv_reaim{tag}.npz", met=met, cogs=cogs)
    print(f"  wrote {OUTDIR / f'cv_reaim{tag}.npz'}")


def demo():
    rows = np.load(POP_NPZ)["dev100"][:8]
    _w_init(rows)
    s2 = _W["s2"]
    s3 = _W["s3"]
    d2 = np.load(E38_DIR / "outputs/stage2_multiepoch.npz")
    th = d2["theta_1ch-mof"]
    g = _W["gals"][0]
    # (1) the plain loss over an epoch subset equals the manual mean of
    # the per-epoch relative RMS terms
    for ks in SCOPES.values():
        lo = s2.gal_loss(th, g, ks, "1ch-mof")
        cogs = s2.model_cogs(th, g, ks, "1ch-mof")
        manual = np.mean([np.sqrt(np.mean(
            ((c - g["data"][k]) / g["data"][k]) ** 2))
            for c, k in zip(cogs, ks)])
        assert abs(lo - manual) < 1e-12, (ks, lo, manual)
    # (2) the subset loss is insensitive to excluded epochs: perturbing
    # the model only matters through epochs inside ks (sanity: ks=[0]
    # loss equals the k=0 term of the 5-epoch decomposition)
    lo0 = s2.gal_loss(th, g, [0], "1ch-mof")
    c0 = s2.model_cogs(th, g, [0], "1ch-mof")[0]
    manual0 = np.sqrt(np.mean(((c0 - g["data"][0]) / g["data"][0]) ** 2))
    assert abs(lo0 - manual0) < 1e-12
    # (3) the capacity theta loads, fits the 12-param layout, and the
    # inner-aware loss from stage3 is finite on it
    thr = np.load(E38_DIR / "outputs/stage3_capacity.npz"
                  )["theta_inner_1ch-mof"]
    assert thr.shape == (12,)
    li = s3.gal_loss_inner(thr, g, KS_ALL, "1ch-mof")
    assert np.isfinite(li) and li < 4.0
    print("exp40 demo OK: subset losses decompose exactly; the exp38 "
          "capacity theta loads (12 params) and the inner-aware loss is "
          "finite on it")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "latestart":
        cmd_latestart(dev)
    elif cmd == "latephysics":
        cmd_latephysics(dev)
    elif cmd == "latestress":
        cmd_latestress(dev)
    elif cmd == "latecv":
        cmd_latecv(sys.argv[sys.argv.index("--scope") + 1]
                   if "--scope" in sys.argv else "z15", dev)
    elif cmd == "reaim":
        cmd_reaim(dev)
    elif cmd == "reaimcv":
        cmd_reaimcv(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
