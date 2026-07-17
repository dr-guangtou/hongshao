"""exp39 — the core-channel revisit, lead 1: the core-form shootout.

exp38 stage 3 parked the dissipative core channel: it fixes the inner
masses (held-out M(<5) -11 -> -2.8% at z=0.4) but breaks the
differential-deposition test (0.53/0.19 vs data 0.37/0.11) and re-opens
the outskirt overshoot. The post-mortem found the core INHERITED the
kernel's power-law tail (gamma ~ 1.4 at rc ~ 2 kpc): 8.3% of "core" mass
lands beyond 30 kpc — at f_core = 0.19 that is 1.6% of ALL stellar mass
in the outskirts, matching the failed overshoot. But the quenched-core
null had shown a second suspect: the OUTER kernel re-balancing itself
once any core absorbs the inner mass.

This shootout separates the two mechanisms by swapping ONLY the core's
radial form, everything else identical to the exp38 stage-3 core model
(1ch-mof kernel + a non-migrating core channel, inner-aware objective):

  mof    the inherited power-law-tail baseline (exp38 as-parked; gamma
         taken from the kernel, exactly the parked model re-expressed)
  gauss  Sersic n=0.5 — zero wings, zero outskirt leakage by
         construction (its kernel-role failure is a core-role virtue)
  exp    Sersic n=1 — 0.0% beyond 30 kpc at the fitted R50
  ser2/ser3/ser4  cuspier Sersic n=2/3/4 — more mass inside 2-5 kpc per
         unit core mass, moderately steeper tail than the Moffat

Every form is parameterized by its HALF-MASS radius R50_core (log10, box
0.5-8 kpc), so the fitted scale is comparable across forms and the box
is form-fair. Theta = the 12 stage-2 kernel parameters + [core logit
amplitude ca, conditioning slope cb, log10 R50_core]; ca -> -inf nests
the adopted stage-2 kernel exactly (asserted).

Read: if the zero-wing forms keep the inner win AND pass the physics
tests, the leakage mechanism was the breaker (the core is rescuable);
if even the Gaussian core breaks them, the re-balancing mechanism
dominates and no core form can help — the channel stays parked.

Judged by the pre-registered criteria: differential curve (data
0.37/0.11; adopted kernel 0.39/0.12; parked core 0.53/0.19), outskirt
terciles (kernel T1 +0.026/+0.019; parked +0.054/+0.085), NO parameter
at a bound, held-out pinned shape vs the kernel's 18.5-14.2% (parked
core avg 15.0%), and the M(<5)/M(<10) bias table.

Lead 2 hook: --cond z2 conditions f_core on the standardized halo mass
at z=2 (pop logmh_zk_real[:, 4]) instead of z=0.4 logMh — the
provenance measurement showed the model core is built by z=2-4 deposits.

Run: PYTHONPATH=. uv run python experiments/exp39_core_revisit/\
shootout.py {demo|fit|cv|physics|table} [--form gauss[,ser3]] [--dev]
[--cond z2]
"""
import importlib.util
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, gammainc, gammaincinv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
E38_DIR = ROOT / "experiments/exp38_deposit_rethink"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
KS = [0, 1, 2, 3, 4]
FORMS = ("mof", "gauss", "exp", "ser2", "ser3", "ser4")
SERSIC_N = {"gauss": 0.5, "exp": 1.0, "ser2": 2.0, "ser3": 3.0, "ser4": 4.0}
R50_BOX = (-0.3, 0.9)                     # log10 kpc: 0.5 - 7.9 kpc
CA_BOX = (-6.0, 6.0)
CB_MAX = 4.0
_W = {}


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _w_init(rows, cond_key="z0"):
    s3 = _load_by_path("exp38_stage3", E38_DIR / "stage3_inner.py")
    s3._w_init(rows)
    _W.update(s3._W)
    _W["s3"] = s3
    _W["cond_key"] = cond_key
    z2 = np.load(POP_NPZ)["logmh_zk_real"][:, 4]
    mu, sd = np.nanmean(z2), np.nanstd(z2)
    for g in _W["gals"]:
        v = (z2[g["row"]] - mu) / sd
        g["cond_z2"] = float(v) if np.isfinite(v) else 0.0


def _cond_of(g):
    return g["cond_z2"] if _W["cond_key"] == "z2" else float(g["cond"][0])


# --------------------------------------------------------------------------- #
# the core forms — unit-mass CoGs at a common half-mass-radius parameter       #
# --------------------------------------------------------------------------- #
def core_cog_unit(form, log_r50, gam, r):
    """Unit-mass core CoG on radius grid r; every form at the SAME
    half-mass radius R50 = 10**log_r50 [kpc]. `gam` is the kernel's
    (population-level) Moffat exponent, used only by the inherited
    baseline."""
    r50 = 10.0 ** float(np.clip(log_r50, *R50_BOX))
    if form == "mof":
        rc = r50 / np.sqrt(2.0 ** (1.0 / (gam - 1.0)) - 1.0)
        return 1.0 - (1.0 + (r / rc) ** 2) ** (1.0 - gam)
    n = SERSIC_N[form]
    b = gammaincinv(2.0 * n, 0.5)
    a = r50 / b ** n
    return gammainc(2.0 * n, (r / a) ** (1.0 / n))


def core_leak(form, log_r50, gam, r_out=30.0):
    """Fraction of core mass beyond r_out kpc (the post-mortem metric)."""
    return 1.0 - float(core_cog_unit(form, log_r50, gam,
                                     np.array([r_out]))[0])


# --------------------------------------------------------------------------- #
# model + loss (the exp38 stage-3 core model with the form swapped)            #
# --------------------------------------------------------------------------- #
def model_cogs_form(p, g, ks, form):
    """1ch-mof kernel + a non-migrating core of the named form. Theta =
    12 stage-2 params + [ca, cb, log10 R50_core]; ca -> -inf nests the
    stage-2 kernel exactly."""
    e = _W["e"]
    s2 = _W["s2"]
    mah = g["mah"]
    base6, _, _ = s2.theta_of(p[:12], g, "1ch-mof")
    f_core = float(expit(p[12] + p[13] * _cond_of(g)))
    w = e.weights(mah["z"], base6[3], base6[4])
    dM = w * mah["dMh"]
    if not np.isfinite(dM).all() or dM.sum() <= 0:
        return None
    dM = dM / dM.sum()
    gam = float(np.clip(base6[5], 1.06, 6.0))
    th4 = [base6[0], base6[1], 0.0, base6[2]]
    cog_core = core_cog_unit(form, p[14], gam, e.R_EXT)
    out = []
    for k in ks:
        mask = mah["snap"] <= e.ANCHOR_SNAP[k]
        B = s2.basis_mof(th4, gam, mah["t"], mah["t_obs"], e.pe.AT[k],
                         e.R_EXT)
        m = ((1.0 - f_core) * (B @ (dM * mask))
             + f_core * cog_core * (dM * mask).sum())
        if not np.isfinite(m[-1]) or m[-1] <= 0 or m[-2] <= 0:
            return None
        out.append(m[:-1] * (g["m500"][k] / m[-1]))
    return out


def _i_at(r_kpc):
    return int(np.searchsorted(_W["e"].R, r_kpc))


def gal_loss_form(p, g, ks, form):
    """The exp38 inner-aware objective: R>5 shape + M(<5)/M(<10) terms."""
    cogs = model_cogs_form(p, g, ks, form)
    if cogs is None:
        return 4.0
    e = _W["e"]
    m_out = e.R > 5.0
    i5, i10 = _i_at(5.0), _i_at(10.0)
    tot = 0.0
    for c, k in zip(cogs, ks):
        d = g["data"][k]
        rel = (c - d) / d
        tot += (np.sqrt(np.mean(rel[m_out] ** 2))
                + 0.5 * (abs(rel[i5]) + abs(rel[i10])))
    return float(tot / len(ks))


def _pen_form(p):
    s2 = _W["s2"]
    return s2.penalty(p[:12], "1ch-mof") + 30.0 * float(
        np.clip(CA_BOX[0] - p[12], 0, None) ** 2
        + np.clip(p[12] - CA_BOX[1], 0, None) ** 2
        + np.clip(abs(p[13]) - CB_MAX, 0, None) ** 2
        + np.clip(R50_BOX[0] - p[14], 0, None) ** 2
        + np.clip(p[14] - R50_BOX[1], 0, None) ** 2)


def _chunk_form(args):
    p, form, ks, lo, hi = args
    return sum(gal_loss_form(p, g, ks, form) for g in _W["gals"][lo:hi])


def _at_bound_form(p):
    out = _W["s2"]._at_bound(p[:12], "1ch-mof")
    for i, nm, (lo, hi) in ((12, "ca", CA_BOX), (13, "cb", (-CB_MAX, CB_MAX)),
                            (14, "log_r50", R50_BOX)):
        w = hi - lo
        if p[i] < lo + 0.02 * w or p[i] > hi - 0.02 * w:
            out.append(f"{nm}={p[i]:.2f}")
    return out


def _model_fn_of(form):
    """Adapter to the exp38 stage-3 helpers (variant slot ignored)."""
    def fn(p, g, ks, variant):
        return model_cogs_form(p, g, ks, form)
    return fn


def _npz_tag(dev):
    return ("_z2" if _W["cond_key"] == "z2" else "") + ("_dev" if dev else "")


# --------------------------------------------------------------------------- #
# fit                                                                          #
# --------------------------------------------------------------------------- #
def _warm_starts():
    """Warm from the exp38 parked-core basin, rc converted to R50."""
    dc = np.load(E38_DIR / "outputs/stage3_core.npz")
    th38 = dc["theta"]
    gam = float(np.clip(th38[5], 1.06, 6.0))
    rc = 10.0 ** float(np.clip(th38[14], -0.3, 0.9))
    log_r50 = float(np.clip(
        np.log10(rc * np.sqrt(2.0 ** (1.0 / (gam - 1.0)) - 1.0)),
        R50_BOX[0] + 0.02, R50_BOX[1] - 0.02))
    warm = np.concatenate([th38[:12], [th38[12], th38[13], log_r50]])
    nudge = warm.copy()
    nudge[12] = -0.7                       # a bigger core
    nudge[14] = max(log_r50 - 0.25, R50_BOX[0] + 0.02)   # a smaller R50
    return [warm, nudge]


def cmd_fit(forms, dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows, _W.get("cond_key", "z0"))
    _W["rows_arg"] = rows
    tag = _npz_tag(dev)
    scale = 0.15 if dev else 1.0
    maxiter = max(int(5000 * scale), 80)
    workers = max(os.cpu_count() - 2, 2)
    n = len(_W["gals"])
    edges = np.linspace(0, n, workers + 1).astype(int)
    starts = _warm_starts()
    out_path = OUTDIR / f"shootout{tag}.npz"
    OUTDIR.mkdir(exist_ok=True)
    fitted = dict(np.load(out_path)) if out_path.exists() else {}
    gam_pop = None
    print(f"exp39 core-form shootout fit (n={n}{', DEV' if dev else ''}, "
          f"cond={_W['cond_key']}, inner-aware objective; forms: "
          f"{', '.join(forms)})", flush=True)
    with Pool(workers, initializer=_w_init,
              initargs=(rows, _W["cond_key"])) as pool:
        for form in forms:
            def loss(p):
                parts = pool.map(_chunk_form,
                                 [(p, form, KS, edges[i], edges[i + 1])
                                  for i in range(workers)])
                return sum(parts) / n + _pen_form(p)

            best = None
            for p0 in starts:
                t0 = time.time()
                r = minimize(loss, p0, method="Nelder-Mead",
                             options=dict(maxiter=maxiter, xatol=3e-4,
                                          fatol=1e-8))
                print(f"  [{form}] start: loss {r.fun:.4f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
                if best is None or r.fun < best.fun:
                    best = r
            th, lo = best.x, best.fun
            gam_pop = float(np.clip(th[5], 1.06, 6.0))
            fcs = [float(expit(th[12] + th[13] * _cond_of(g)))
                   for g in _W["gals"]]
            leak = core_leak(form, th[14], gam_pop)
            ab = _at_bound_form(th)
            print(f"  [{form}] f_core pct16/50/84 = "
                  f"{np.percentile(fcs, [16, 50, 84]).round(3)}; "
                  f"R50_core = {10.0 ** th[14]:.2f} kpc; "
                  f"core mass beyond 30 kpc = {100 * leak:.1f}%")
            print(f"  [{form}] at bound: {', '.join(ab) if ab else 'NONE'}")
            _W["s3"]._print_bias(
                f"[{form}] inner", _W["s3"]._inner_bias(
                    th, "1ch-mof", model_fn=_model_fn_of(form)))
            fitted[f"theta_{form}"] = th
            fitted[f"loss_{form}"] = lo
            np.savez(out_path, **fitted)
            print(f"  wrote {out_path} (+{form})", flush=True)


# --------------------------------------------------------------------------- #
# frozen-kernel fit — the re-balancing mechanism isolated                      #
# --------------------------------------------------------------------------- #
def _chunk_frozen(args):
    """Loss chunk with the kernel frozen: p = [ca, cb, log_r50] only; the
    12 kernel params come from the worker-side stage-2 npz (constant)."""
    p3, form, ks, lo, hi = args
    p = np.concatenate([_W["th12"], p3])
    return sum(gal_loss_form(p, g, ks, form) for g in _W["gals"][lo:hi])


def _w_init_frozen(rows, cond_key="z0"):
    _w_init(rows, cond_key)
    _W["th12"] = np.load(E38_DIR / "outputs/stage2_multiepoch.npz"
                         )["theta_1ch-mof"]


def _pen_frozen(p3):
    return 30.0 * float(
        np.clip(CA_BOX[0] - p3[0], 0, None) ** 2
        + np.clip(p3[0] - CA_BOX[1], 0, None) ** 2
        + np.clip(abs(p3[1]) - CB_MAX, 0, None) ** 2
        + np.clip(R50_BOX[0] - p3[2], 0, None) ** 2
        + np.clip(p3[2] - R50_BOX[1], 0, None) ** 2)


def cmd_frozen(forms, dev=False):
    """Fit ONLY the 3 core params with the kernel FROZEN at the adopted
    stage-2 theta. The kernel cannot re-balance: this measures how much
    inner-mass win a core channel buys on its own, and what it does to
    the physics tests when the outer kernel is pinned."""
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init_frozen(rows, _W.get("cond_key", "z0"))
    _W["rows_arg"] = rows
    tag = _npz_tag(dev)
    th12 = _W["th12"]
    scale = 0.15 if dev else 1.0
    maxiter = max(int(1500 * scale), 60)
    workers = max(os.cpu_count() - 2, 2)
    n = len(_W["gals"])
    edges = np.linspace(0, n, workers + 1).astype(int)
    starts3 = [np.array([-1.5, 0.0, 0.55]), np.array([-0.7, 0.0, 0.3])]
    out_path = OUTDIR / f"frozen{tag}.npz"
    OUTDIR.mkdir(exist_ok=True)
    fitted = dict(np.load(out_path)) if out_path.exists() else {}
    print(f"exp39 FROZEN-kernel core fit (n={n}{', DEV' if dev else ''}, "
          f"cond={_W['cond_key']}; kernel pinned at the adopted stage-2 "
          f"theta, only [ca, cb, log_r50] free; forms: {', '.join(forms)})",
          flush=True)
    with Pool(workers, initializer=_w_init_frozen,
              initargs=(rows, _W["cond_key"])) as pool:
        for form in forms:
            def loss(p3):
                parts = pool.map(_chunk_frozen,
                                 [(p3, form, KS, edges[i], edges[i + 1])
                                  for i in range(workers)])
                return sum(parts) / n + _pen_frozen(p3)

            best = None
            for p0 in starts3:
                t0 = time.time()
                r = minimize(loss, p0, method="Nelder-Mead",
                             options=dict(maxiter=maxiter, xatol=3e-4,
                                          fatol=1e-8))
                print(f"  [{form}] start: loss {r.fun:.4f} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
                if best is None or r.fun < best.fun:
                    best = r
            th = np.concatenate([th12, best.x])
            gam_pop = float(np.clip(th[5], 1.06, 6.0))
            fcs = [float(expit(th[12] + th[13] * _cond_of(g)))
                   for g in _W["gals"]]
            print(f"  [{form}] f_core pct16/50/84 = "
                  f"{np.percentile(fcs, [16, 50, 84]).round(3)}; "
                  f"R50_core = {10.0 ** th[14]:.2f} kpc; core mass beyond "
                  f"30 kpc = {100 * core_leak(form, th[14], gam_pop):.1f}%")
            _W["s3"]._print_bias(
                f"[{form}] frozen", _W["s3"]._inner_bias(
                    th, "1ch-mof", model_fn=_model_fn_of(form)))
            fitted[f"theta_{form}"] = th
            fitted[f"loss_{form}"] = best.fun
            np.savez(out_path, **fitted)
            print(f"  wrote {out_path} (+{form})", flush=True)


# --------------------------------------------------------------------------- #
# cv — 10-fold held-out (shape R>5 pinned + inner biases), exp38 cv3 protocol  #
# --------------------------------------------------------------------------- #
def _cv_fold_form(args):
    fold, form, warm, maxiter, frozen = args
    gals = _W["gals"]
    n = len(gals)
    train = [i for i in range(n) if i % 10 != fold]

    if frozen:                       # only [ca, cb, log_r50] refit per fold
        def tr_loss(p3):
            p = np.concatenate([_W["th12"], p3])
            return (np.mean([gal_loss_form(p, gals[i], KS, form)
                             for i in train]) + _pen_frozen(p3))
        warm = warm[12:15]
    else:
        def tr_loss(p):
            return (np.mean([gal_loss_form(p, gals[i], KS, form)
                             for i in train]) + _pen_form(p))
    r = minimize(tr_loss, warm, method="Nelder-Mead",
                 options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
    if frozen:
        r.x = np.concatenate([_W["th12"], r.x])
    e = _W["e"]
    i5, i10 = _i_at(5.0), _i_at(10.0)
    m = e.R > 5.0
    out = []
    for i in range(n):
        if i % 10 != fold:
            continue
        g = gals[i]
        cogs = model_cogs_form(r.x, g, KS, form)
        if cogs is None:
            out.append((g["row"], np.full((5, 3), np.nan),
                        np.full((5, len(e.R)), np.nan)))
            continue
        met = []
        for c, k in zip(cogs, KS):
            d = g["data"][k]
            cs = c * (d[-1] / c[-1])
            met.append([np.abs(cs[m] / d[m] - 1).max(),
                        c[i5] / d[i5] - 1, c[i10] / d[i10] - 1])
        out.append((g["row"], np.array(met), np.array(cogs)))
    return out


def cmd_cv(forms, dev=False, frozen=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    init_fn = _w_init_frozen if frozen else _w_init
    init_fn(rows, _W.get("cond_key", "z0"))
    tag = ("_frozen" if frozen else "") + _npz_tag(dev)
    gals = _W["gals"]
    n = len(gals)
    e = _W["e"]
    d = np.load(OUTDIR / f"{'frozen' if frozen else 'shootout'}"
                f"{_npz_tag(dev)}.npz")
    workers = min(max(os.cpu_count() - 2, 2), 10)
    maxiter = 150 if dev else 1500
    row_to_i = {g["row"]: i for i, g in enumerate(gals)}
    for form in forms:
        met = np.full((n, 5, 3), np.nan)
        cogs = np.full((n, 5, len(e.R)), np.nan)
        t0 = time.time()
        with Pool(workers, initializer=init_fn,
                  initargs=(rows, _W["cond_key"])) as pool:
            jobs = [(f, form, d[f"theta_{form}"], maxiter, frozen)
                    for f in range(10)]
            for part in pool.map(_cv_fold_form, jobs):
                for row, m_, c_ in part:
                    met[row_to_i[row]] = m_
                    cogs[row_to_i[row]] = c_
        line = " ".join(
            f"z{e.ANCHOR_Z[k]}: {100*np.nanmedian(met[:, k, 0]):.1f} | "
            f"{100*np.nanmedian(met[:, k, 1]):+.1f} | "
            f"{100*np.nanmedian(met[:, k, 2]):+.1f}"
            for k in range(5))
        print(f"  [{form}] held-out shape R>5 | M(<5) | M(<10): {line}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        np.savez(OUTDIR / f"cv_{form}{tag}.npz", met=met, cogs=cogs)
        print(f"  wrote {OUTDIR / f'cv_{form}{tag}.npz'}", flush=True)


# --------------------------------------------------------------------------- #
# physics — differential deposition (in-sample) + overshoot (held-out)         #
# --------------------------------------------------------------------------- #
def cmd_physics(forms, dev=False, frozen=False):
    from hongshao.profile_emulator import density_from_cog
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows, _W.get("cond_key", "z0"))
    tag = _npz_tag(dev)
    gals = _W["gals"]
    e = _W["e"]
    d = np.load(OUTDIR / f"{'frozen' if frozen else 'shootout'}{tag}.npz")
    data = np.stack([g["data"] for g in gals])
    logms = np.array([g["logms"] for g in gals])
    print("exp39 physics (marks: data 0.37/0.11; adopted kernel 0.39/0.12, "
          "T1 +0.026/+0.019; parked mof core 0.53/0.19, T1 +0.054/+0.085):")
    for form in forms:
        th = d[f"theta_{form}"]
        cogs = np.full((len(gals), 5, len(e.R)), np.nan)
        for i, g in enumerate(gals):
            out = model_cogs_form(th, g, KS, form)
            if out is not None:
                cogs[i] = out
        ed3, rows_d = e.differential(cogs, data, logms)
        for b in range(3):
            cells = [f"z{e.ANCHOR_Z[k+1]}->z{e.ANCHOR_Z[k]}: "
                     f"{rows_d[('data', b, k)][0]:.2f}/"
                     f"{rows_d[('data', b, k)][1]:.2f} -> "
                     f"{rows_d[('model', b, k)][0]:.2f}/"
                     f"{rows_d[('model', b, k)][1]:.2f}" for k in range(4)]
            print(f"  [{form}] logM* {ed3[b]:.2f}-{ed3[b+1]:.2f}: "
                  + "  ".join(cells))
        cv_path = OUTDIR / (f"cv_{form}{'_frozen' if frozen else ''}"
                            f"{tag}.npz")
        cogs0_src = ("held-out" if cv_path.exists() else "IN-SAMPLE")
        cogs0 = (np.load(cv_path)["cogs"][:, 0] if cv_path.exists()
                 else cogs[:, 0])
        ok = np.isfinite(cogs0).all(1) & (cogs0 > 0).all(1)
        ls_d, mid = density_from_cog(np.log10(data[:, 0]), e.R)
        ls_m = np.full_like(ls_d, np.nan)
        ls_m[ok] = density_from_cog(np.log10(cogs0[ok]), e.R)[0]
        dl = ls_m - ls_d
        bands = ((mid >= 30.0) & (mid < 60.0),
                 (mid >= 60.0) & (mid <= 148.0))
        edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
        cells = []
        for b in range(3):
            sel = ok & (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
            cells.append(" ".join(
                f"{np.nanmedian(dl[np.ix_(sel, bm)]):+.3f}" for bm in bands))
        print(f"  [{form}] overshoot terciles ({cogs0_src}, z=0.4, "
              "[30-60 / 60-148 kpc]): "
              + "  |  ".join(f"T{b+1} {c}" for b, c in enumerate(cells)))


# --------------------------------------------------------------------------- #
# table — the one-look decision summary across fitted forms                    #
# --------------------------------------------------------------------------- #
def cmd_table(dev=False):
    rows = np.load(POP_NPZ)["dev100"] if dev else None
    _w_init(rows, _W.get("cond_key", "z0"))
    tag = _npz_tag(dev)
    d = np.load(OUTDIR / f"shootout{tag}.npz")
    e = _W["e"]
    print(f"exp39 shootout summary (cond={_W['cond_key']}"
          f"{', DEV' if dev else ''}); every number defined in its column:")
    print("  form | loss (inner-aware objective, smaller better) | "
          "median f_core (fraction of each deposit into the core) | "
          "R50_core kpc (core half-mass radius) | leak30 (% of core mass "
          "beyond 30 kpc; parked mof was 8.3) | params at a bound")
    for form in FORMS:
        if f"theta_{form}" not in d.files:
            continue
        th = d[f"theta_{form}"]
        gam = float(np.clip(th[5], 1.06, 6.0))
        fcs = [float(expit(th[12] + th[13] * _cond_of(g)))
               for g in _W["gals"]]
        ab = _at_bound_form(th)
        print(f"  {form:5s} | {float(d[f'loss_{form}']):.4f} | "
              f"{np.median(fcs):.3f} | {10.0 ** th[14]:.2f} | "
              f"{100 * core_leak(form, th[14], gam):.1f} | "
              f"{', '.join(ab) if ab else 'NONE'}")
        cv_path = OUTDIR / f"cv_{form}{tag}.npz"
        if cv_path.exists():
            met = np.load(cv_path)["met"]
            shp = [100 * np.nanmedian(met[:, k, 0]) for k in range(5)]
            print(f"        held-out pinned shape R>5 by epoch "
                  f"(kernel marks 18.5-14.2, parked avg 15.0): "
                  + "/".join(f"{v:.1f}" for v in shp)
                  + f"  avg {np.mean(shp):.1f}; held-out M(<5) z=0.4 "
                  f"{100*np.nanmedian(met[:, 0, 1]):+.1f}% "
                  f"(kernel ~-11, parked -2.8)")


def demo():
    rows = np.load(POP_NPZ)["dev100"][:8]
    _w_init(rows)
    s3 = _W["s3"]
    s2 = _W["s2"]
    d2 = np.load(E38_DIR / "outputs/stage2_multiepoch.npz")
    p12 = d2["theta_1ch-mof"]
    dc = np.load(E38_DIR / "outputs/stage3_core.npz")
    th38 = dc["theta"]
    g = _W["gals"][0]

    # (1) the mof form at the converted R50 reproduces the exp38 parked
    # core model EXACTLY (same theta, rc -> R50 round trip)
    gam = float(np.clip(th38[5], 1.06, 6.0))
    rc = 10.0 ** float(np.clip(th38[14], -0.3, 0.9))
    log_r50 = np.log10(rc * np.sqrt(2.0 ** (1.0 / (gam - 1.0)) - 1.0))
    th39 = np.concatenate([th38[:14], [log_r50]])
    a = s3.model_cogs_core(th38, g, [0, 4], "1ch-mof")
    b = model_cogs_form(th39, g, [0, 4], "mof")
    err = max(np.abs(np.asarray(x) / np.asarray(y) - 1.0).max()
              for x, y in zip(b, a))
    assert err < 1e-9, f"mof form != exp38 parked core: {err:.2e}"

    # (2) ca -> -inf nests the adopted stage-2 kernel exactly, every form
    ref = s2.model_cogs(p12, g, [0, 4], "1ch-mof")
    for form in FORMS:
        got = model_cogs_form(np.concatenate([p12, [-30.0, 0.0, 0.5]]),
                              g, [0, 4], form)
        err = max(np.abs(np.asarray(x) / np.asarray(y) - 1.0).max()
                  for x, y in zip(got, ref))
        assert err < 1e-9, f"{form} ca=-30 nesting broken: {err:.2e}"

    # (3) the R50 parameterization is exact: every form's unit CoG
    # crosses 0.5 at 10**log_r50
    rr = np.logspace(-3, 4, 20000)
    for form in FORMS:
        c = core_cog_unit(form, 0.55, 1.39, rr)
        r_half = np.interp(0.5, c, rr)
        assert abs(r_half / 10.0 ** 0.55 - 1.0) < 1e-3, (form, r_half)

    # (4) the leakage ordering the shootout is built on, at the parked
    # core's R50 (3.4 kpc): mof leaks ~8% beyond 30 kpc, every Sersic
    # form leaks less, the Gaussian effectively zero
    leaks = {form: core_leak(form, log_r50, gam) for form in FORMS}
    assert all(leaks["mof"] > leaks[f] for f in FORMS if f != "mof")
    assert leaks["gauss"] < 1e-6
    assert (leaks["gauss"] < leaks["exp"] < leaks["ser2"]
            < leaks["ser3"] < leaks["ser4"])

    # (5) f_core raises the inner mass fraction monotonically (gauss form)
    i10 = _i_at(10.0)
    fr = []
    for ca in (-30.0, -1.0, 2.0):
        c = model_cogs_form(np.concatenate([p12, [ca, 0.0, 0.5]]), g, [0],
                            "gauss")[0]
        fr.append(c[i10] / c[-1])
    assert fr[0] < fr[1] < fr[2], "f_core must raise the core"

    # (6) the z2 conditioning hook: cond_z2 present, finite, standardized
    v = np.array([g_["cond_z2"] for g_ in _W["gals"]])
    assert np.isfinite(v).all()

    print("exp39 demo OK: mof form == the exp38 parked core exactly "
          "(rc -> R50 round trip); every form nests the adopted kernel at "
          "ca -> -inf; R50 parameterization exact to 0.1%; leakage beyond "
          "30 kpc at the parked R50: "
          + ", ".join(f"{k} {100*v:.2f}%" for k, v in leaks.items())
          + "; f_core monotone; z2 conditioning wired")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    _W["cond_key"] = ("z2" if "--cond" in sys.argv
                      and sys.argv[sys.argv.index("--cond") + 1] == "z2"
                      else "z0")
    forms = (sys.argv[sys.argv.index("--form") + 1].split(",")
             if "--form" in sys.argv else list(FORMS))
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(forms, dev)
    elif cmd == "frozen":
        cmd_frozen(forms, dev)
    elif cmd == "cv":
        cmd_cv(forms, dev, frozen="--frozen" in sys.argv)
    elif cmd == "physics":
        cmd_physics(forms, dev, frozen="--frozen" in sys.argv)
    elif cmd == "table":
        cmd_table(dev)
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
