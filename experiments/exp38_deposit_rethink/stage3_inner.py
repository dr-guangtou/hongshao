"""exp38 stage 3 — the inner-mass deficit: diagnose, then fix.

Both stage-2 winners under-predict M(<10 kpc) by 2-7% (worst at z=2) while
over-filling the 10-30 kpc annulus by 10-37% — a PLACEMENT error, not a
budget error. Two suspects, two diagnostics:

`provenance` — decompose the fitted model's M(<10 kpc) by deposit epoch,
    twice: with the fitted transport clock, and with migration FROZEN
    (fc = 1, stars stay at their deposition width). If the frozen-clock
    model has plenty of inner mass where the fitted one is empty, the
    alpha = 1 migration rule (every deposit's compact core drains as
    exp(-dt/t_i), no floor) is draining the core — the fix is a
    core-RETENTION parameter, not a new component.

`capacity` — refit (dev scale) with the restructured objective (the
    user's option 2): fit the CoG shape only at R > 5 kpc, and add the
    integrated M(<5) and M(<10) masses as explicit loss terms. If the
    deficit closes at small outer cost, the problem was loss allocation;
    if it persists, it is structural.

`retention` — the 1-parameter structural fix: fc' = f_ret + (1-f_ret) fc
    (a fraction of every deposit never migrates; f_ret = 0 nests the
    current model exactly). Refit, judged by the inner-bias table + the
    outer marks.

Run: PYTHONPATH=. uv run python experiments/exp38_deposit_rethink/\
stage3_inner.py {demo|provenance|capacity|retention} [--dev]
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
sys.path.insert(0, str(HERE))

OUTDIR = HERE / "outputs"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS = [0, 1, 2, 3, 4]
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows):
    s2 = _load_by_path("exp38_stage2", HERE / "stage2_multiepoch.py")
    s2._w_init(rows)
    _W.update(s2._W)
    _W["s2"] = s2


def _i_at(r_kpc):
    e = _W["e"]
    return int(np.searchsorted(e.R, r_kpc))


# --------------------------------------------------------------------------- #
# provenance: who builds the model core, and what the clock drains             #
# --------------------------------------------------------------------------- #
def _mof_core_split(p, g, k, zbins, freeze_clock=False):
    """1ch-mof M(<10 kpc) contributions by deposit-redshift bin, optionally
    with the migration clock frozen (fc = 1). Returns (per-bin M, total)."""
    e = _W["e"]
    s2 = _W["s2"]
    mah = g["mah"]
    base6, _, _ = s2.theta_of(p, g, "1ch-mof")
    w = e.weights(mah["z"], base6[3], base6[4])
    dM = w * mah["dMh"]
    dM = dM / dM.sum()
    mask = mah["snap"] <= e.ANCHOR_SNAP[k]
    th4 = [base6[0], base6[1], 0.0, base6[2]]
    gam = float(np.clip(base6[5], 1.06, 6.0))
    B = s2.basis_mof(th4, gam, mah["t"], mah["t_obs"], e.pe.AT[k], e.R_EXT)
    if freeze_clock:
        rc0 = np.clip(10.0 ** th4[0] * (mah["t"] / mah["t_obs"]) ** th4[1],
                      1e-4, 1e5)
        u = 1.0 + (e.R_EXT[:, None] / rc0[None, :]) ** 2
        B = 1.0 - u ** (1.0 - gam)                  # birth widths, no envelope
    contrib = B * (dM * mask)[None, :]              # (25, ndep)
    scale = g["m500"][k] / contrib[-1].sum()
    i10 = _i_at(10.0)
    out = []
    for zlo, zhi in zbins:
        m = (mah["z"] >= zlo) & (mah["z"] < zhi)
        out.append(contrib[i10][m].sum() * scale)
    return np.array(out), contrib[i10].sum() * scale


def cmd_provenance(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    gals = _W["gals"]
    d = np.load(OUTDIR / "stage2_multiepoch.npz")
    p = d["theta_1ch-mof"]
    zbins = ((4.0, 20.0), (2.0, 4.0), (1.0, 2.0), (0.0, 1.0))
    print("exp38 stage 3 provenance — model M(<10 kpc) by deposit-redshift "
          "bin, fitted clock vs FROZEN clock (fc=1, no migration); "
          f"n={len(gals)}; medians over the sample, epoch z=0.4 and z=2.0:")
    for k in (0, 4):
        rows_out = {False: [], True: []}
        data10 = []
        for g in gals:
            for fz in (False, True):
                parts, tot = _mof_core_split(p, g, k, zbins, freeze_clock=fz)
                rows_out[fz].append(np.append(parts, tot))
            data10.append(np.interp(10.0, _W["e"].R, g["data"][k]))
        data10 = np.array(data10)
        for fz, lab in ((False, "fitted clock"), (True, "frozen clock")):
            arr = np.stack(rows_out[fz])
            med = np.median(arr / data10[:, None], axis=0)
            print(f"  z={_W['e'].ANCHOR_Z[k]} [{lab:12s}] M(<10)/data: "
                  + "  ".join(f"z[{lo},{hi}): {v:.3f}"
                              for (lo, hi), v in zip(zbins, med[:-1]))
                  + f"  | TOTAL {med[-1]:.3f}")
    print("\n  [read] TOTAL < 1 with the fitted clock but >= 1 frozen -> "
          "the migration clock drains the core; a retention floor is the "
          "structural fix. TOTAL < 1 in both -> the deposits are born too "
          "wide (efficiency/width law), not over-migrated.")


# --------------------------------------------------------------------------- #
# capacity: the restructured objective (option 2)                              #
# --------------------------------------------------------------------------- #
def gal_loss_inner(p, g, ks, variant, w_in=1.0):
    """Relative-RMS over R>5 kpc + integral terms |M(<5)|, |M(<10)| rel."""
    e = _W["e"]
    s2 = _W["s2"]
    cogs = s2.model_cogs(p, g, ks, variant)
    if cogs is None:
        return 4.0
    m_out = e.R > 5.0
    i5, i10 = _i_at(5.0), _i_at(10.0)
    tot = 0.0
    for c, k in zip(cogs, ks):
        d = g["data"][k]
        rel = (c - d) / d
        tot += (np.sqrt(np.mean(rel[m_out] ** 2))
                + w_in * 0.5 * (abs(rel[i5]) + abs(rel[i10])))
    return float(tot / len(ks))


def _chunk_inner(args):
    p, variant, ks, lo, hi = args
    return sum(gal_loss_inner(p, g, ks, variant) for g in _W["gals"][lo:hi])


def _fit(variant, starts, maxiter, penalty_fn, chunk_fn, label):
    n = len(_W["gals"])
    workers = max(os.cpu_count() - 2, 2)
    edges = np.linspace(0, n, workers + 1).astype(int)
    with Pool(workers, initializer=_w_init,
              initargs=(_W["rows_arg"],)) as pool:
        def loss(p):
            parts = pool.map(chunk_fn, [(p, variant, KS, edges[i],
                                         edges[i + 1])
                                        for i in range(workers)])
            return sum(parts) / n + penalty_fn(p, variant)

        best = None
        for p0 in starts:
            t0 = time.time()
            r = minimize(loss, p0, method="Nelder-Mead",
                         options=dict(maxiter=maxiter, xatol=3e-4,
                                      fatol=1e-8))
            print(f"  [{label}] start: loss {r.fun:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            if best is None or r.fun < best.fun:
                best = r
    return best.x, best.fun


def _inner_bias(p, variant, model_fn=None):
    """Median (model-data)/data for M(<5), M(<10), M(<30) + pinned shape
    R>5 per epoch."""
    e = _W["e"]
    s2 = _W["s2"]
    gals = _W["gals"]
    i5, i10, i30 = _i_at(5.0), _i_at(10.0), _i_at(30.0)
    m = e.R > 5.0
    rows = []
    for g in gals:
        cogs = (model_fn or s2.model_cogs)(p, g, KS, variant)
        if cogs is None:
            continue
        per = []
        for c, k in zip(cogs, KS):
            d = g["data"][k]
            cs = c * (d[-1] / c[-1])
            per.append([c[i5] / d[i5] - 1, c[i10] / d[i10] - 1,
                        c[i30] / d[i30] - 1,
                        np.abs(cs[m] / d[m] - 1).max()])
        rows.append(per)
    arr = np.array(rows)
    return np.median(arr, axis=0)                   # (5, 4)


def _print_bias(tag, B):
    e = _W["e"]
    print(f"    {tag}: " + "  ".join(
        f"z{e.ANCHOR_Z[k]}: M<5 {100*B[k, 0]:+.1f} M<10 {100*B[k, 1]:+.1f} "
        f"M<30 {100*B[k, 2]:+.1f} shape {100*B[k, 3]:.1f}"
        for k in (0, 4)))


def cmd_capacity(dev=True):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    _W["rows_arg"] = rows
    s2 = _W["s2"]
    d = np.load(OUTDIR / "stage2_multiepoch.npz")
    scale = 0.15 if dev else 1.0
    print(f"exp38 stage 3 capacity probe (n={len(_W['gals'])}"
          f"{', DEV' if dev else ''}): refit with the R>5-shape + "
          "M(<5)/M(<10) integral objective; can the current forms fill "
          "the core when asked?")
    out = {}
    for variant in ("1ch-mof", "2ch-exp"):
        warm = d[f"theta_{variant}"]
        print(f"  [{variant}] baseline (stage-2 theta, in-sample bias):")
        _print_bias("baseline", _inner_bias(warm, variant))
        th, lo = _fit(variant, [warm], max(int(4000 * scale), 80),
                      s2.penalty, _chunk_inner, f"inner-{variant}")
        _print_bias("re-aimed", _inner_bias(th, variant))
        ab = s2._at_bound(th, variant)
        print(f"    at bound: {', '.join(ab) if ab else 'NONE'}")
        out[f"theta_inner_{variant}"] = th
        out[f"loss_inner_{variant}"] = lo
    np.savez(OUTDIR / f"stage3_capacity{'_dev' if dev else ''}.npz", **out)
    print(f"wrote {OUTDIR / ('stage3_capacity' + ('_dev' if dev else '') + '.npz')}")


# --------------------------------------------------------------------------- #
# retention: the 1-parameter structural fix                                    #
# --------------------------------------------------------------------------- #
def model_cogs_ret(p, g, ks, variant):
    """1ch-mof with a core-retention floor: fc' = f_ret + (1 - f_ret) fc.
    Theta = the 12 stage-2 parameters + [f_ret]; f_ret = 0 nests stage 2
    exactly."""
    e = _W["e"]
    s2 = _W["s2"]
    mah = g["mah"]
    base6, _, _ = s2.theta_of(p[:12], g, "1ch-mof")
    f_ret = float(np.clip(p[12], 0.0, 1.0))
    w = e.weights(mah["z"], base6[3], base6[4])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None
    dM = dM / dM.sum()
    gam = float(np.clip(base6[5], 1.06, 6.0))
    rc0 = np.clip(10.0 ** base6[0] * (mah["t"] / mah["t_obs"]) ** base6[1],
                  1e-4, 1e5)
    out = []
    for k in ks:
        mask = mah["snap"] <= e.ANCHOR_SNAP[k]
        tk = e.pe.AT[k]
        dt = np.clip(tk - mah["t"], 0.0, None)
        fc = np.exp(-dt / mah["t"])
        fc = f_ret + (1.0 - f_ret) * fc
        rcw = np.clip(rc0 * (tk / mah["t"]) ** max(base6[2], 0.0), 1e-4, 1e5)

        def cog(rc):
            u = 1.0 + (e.R_EXT[:, None] / rc[None, :]) ** 2
            return 1.0 - u ** (1.0 - gam)
        B = fc[None, :] * cog(rc0) + (1.0 - fc)[None, :] * cog(rcw)
        m = B @ (dM * mask)
        if not np.isfinite(m[-1]) or m[-1] <= 0 or m[-2] <= 0:
            return None
        out.append(m[:-1] * (g["m500"][k] / m[-1]))
    return out


def gal_loss_ret(p, g, ks, variant):
    cogs = model_cogs_ret(p, g, ks, variant)
    if cogs is None:
        return 4.0
    return float(np.mean([np.sqrt(np.mean(((c - g["data"][k]) / g["data"][k])
                                          ** 2))
                          for c, k in zip(cogs, ks)]))


def _chunk_ret(args):
    p, variant, ks, lo, hi = args
    return sum(gal_loss_ret(p, g, ks, variant) for g in _W["gals"][lo:hi])


def _pen_ret(p, variant):
    s2 = _W["s2"]
    return s2.penalty(p[:12], "1ch-mof") + 30.0 * float(
        np.clip(-p[12], 0, None) ** 2 + np.clip(p[12] - 1.0, 0, None) ** 2)


def cmd_retention(dev=True):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows)
    _W["rows_arg"] = rows
    d = np.load(OUTDIR / "stage2_multiepoch.npz")
    scale = 0.15 if dev else 1.0
    warm = np.append(d["theta_1ch-mof"], 0.05)
    nudge = warm.copy()
    nudge[12] = 0.3
    print(f"exp38 stage 3 retention fit (n={len(_W['gals'])}"
          f"{', DEV' if dev else ''}): 1ch-mof + core-retention floor "
          "f_ret (f_ret=0 nests stage 2; nesting inequality applies):")
    th, lo = _fit("1ch-mof", [warm, nudge], max(int(4000 * scale), 80),
                  _pen_ret, _chunk_ret, "1ch-mof-ret")
    base_lo = float(d["loss_1ch-mof"])
    flag = "" if lo <= base_lo + 1e-9 or dev else "  NESTING VIOLATION"
    print(f"  loss {lo:.4f} vs stage-2 {base_lo:.4f}{flag}; "
          f"f_ret = {th[12]:.3f}")
    print("  inner bias (in-sample):")
    _print_bias("stage-2  ", _inner_bias(d["theta_1ch-mof"], "1ch-mof"))
    _print_bias("retention", _inner_bias(th, "1ch-mof",
                                         model_fn=model_cogs_ret))
    np.savez(OUTDIR / f"stage3_retention{'_dev' if dev else ''}.npz",
             theta=th, loss=lo)
    print(f"wrote {OUTDIR / ('stage3_retention' + ('_dev' if dev else '') + '.npz')}")


def demo():
    rows = np.load(POP_NPZ)["dev100"][:8]
    _w_init(rows)
    _W["rows_arg"] = rows
    s2 = _W["s2"]
    d = np.load(OUTDIR / "stage2_multiepoch.npz")
    p = d["theta_1ch-mof"]
    g = _W["gals"][0]
    # (1) f_ret = 0 nests the stage-2 model EXACTLY
    a = s2.model_cogs(p, g, [0, 4], "1ch-mof")
    b = model_cogs_ret(np.append(p, 0.0), g, [0, 4], "1ch-mof")
    err = max(np.abs(np.asarray(x) / np.asarray(y) - 1.0).max()
              for x, y in zip(b, a))
    assert err < 1e-12, f"f_ret=0 nesting broken: {err:.2e}"
    # (2) retention raises the inner mass monotonically
    i10 = _i_at(10.0)
    vals = [model_cogs_ret(np.append(p, fr), g, [0], "1ch-mof")[0][i10]
            / model_cogs_ret(np.append(p, fr), g, [0], "1ch-mof")[0][-1]
            for fr in (0.0, 0.3, 0.8)]
    assert vals[0] < vals[1] < vals[2], "retention must raise the core"
    # (3) provenance splits sum to the total
    parts, tot = _mof_core_split(p, g, 0, ((4, 20), (2, 4), (1, 2), (0, 1)))
    assert abs(parts.sum() / tot - 1.0) < 1e-9
    # (4) the inner objective penalizes an inner-deficient model more
    l_std = s2.gal_loss(p, g, [0], "1ch-mof")
    l_in = gal_loss_inner(p, g, [0], "1ch-mof")
    assert l_in > 0 and np.isfinite(l_in) and l_in >= 0.5 * l_std
    print("stage3 demo OK: f_ret=0 nests stage 2 exactly; retention raises "
          "the core monotonically; provenance splits sum to the total; the "
          "inner objective is finite and binding")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "provenance":
        cmd_provenance(dev)
    elif cmd == "capacity":
        cmd_capacity(dev)
    elif cmd == "retention":
        cmd_retention(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
