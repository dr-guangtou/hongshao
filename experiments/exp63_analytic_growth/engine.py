"""exp63 Stage 0 — the deposition model as an EXACT integral along the halo's
analytic growth curve.

The incumbent (`exp54/model.forward`) deposits stars at 72 merger-tree steps
and adds them up. Every ingredient of that sum is already a closed-form
function of cosmic time — the DiffMAH curve, the redshift, the critical
density, the halo radius, the efficiency and size laws — so the sum is a
first-order Riemann approximation of

    M*(<R, t) = int_{t_start}^{t} eps(M(t'), z(t')) dM/dt' F(R / s(t')) dt'

(`doc/tech_note/04_analytic_deposition_integral.md`, Eq. 2). This module
evaluates that integral by Gauss-Legendre quadrature in ln t, batched over
galaxies, with the SAME efficiency law, size law and truncated kernel as the
incumbent (imported from `exp54/model.py`, not re-implemented), so the only
thing that changes in Stage 0 is the integration.

What a halo is to this engine: FOUR DiffMAH numbers (`logmp` anchored at
z = 0, `logtc`, `early`, `late`; transition speed fixed at k = 3.5) and the
cosmology. That is decision D1 of the exp63 plan: R200c(t) is computed from
M(t) and rho_c(z), not read from the TNG group catalog. The catalog radius
history is carried alongside ONLY for the comparison mode and the bridge to
the old sum (`discrete_sum`), which is how the engine is proved against
`exp54/model.forward`.

No stellar quantity exists on a `HaloCurve` (Contract B); the self-check
asserts it.

Run the self-check:
    HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
        experiments/exp63_analytic_growth/engine.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, gamma as Gamma, gammaincc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402  (must precede halo/model)
import halo as H                                         # noqa: E402
import model as M                                        # noqa: E402  (puts exp53 on the path)
import families                                          # noqa: E402
from hongshao.diffmah import MAH_K                       # noqa: E402
from hongshao.tng_data import load_cosmic_time           # noqa: E402

# --------------------------------------------------------------------------- #
# cosmology (TNG: Planck 2015), closed form                                    #
# --------------------------------------------------------------------------- #
H0_KMS = 67.74
OMEGA_M = 0.3089
OMEGA_L = 1.0 - OMEGA_M
H0_GYR = H0_KMS / 977.792                    # km/s/Mpc -> 1/Gyr
RHO_C0 = 2.775e11 * (H0_KMS / 100.0) ** 2   # Msun / Mpc^3, little-h removed

T_SNAP = load_cosmic_time()                  # Gyr per snapshot, (100,)
LOGT0 = float(np.log10(T_SNAP[99]))          # official DiffMAH anchor: z = 0
T_START = float(T_SNAP[1])                   # snapshot 1, z ~ 15: the grid's own start
ANCHOR_SNAP = (72, 59, 50, 40, 33)           # z = 0.4 0.7 1.0 1.5 2.0
ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
T_ANCHOR = np.array([T_SNAP[s] for s in ANCHOR_SNAP])
DIFFMAH_TABLE = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/diffmah_combined.fits"
STELLAR_FIELDS = ("m500", "data", "logms", "cog", "mstar")


def a_of_t(t):
    """Scale factor at cosmic time t [Gyr], flat LambdaCDM, exact."""
    t = np.asarray(t, float)
    return (OMEGA_M / OMEGA_L) ** (1.0 / 3.0) * np.sinh(
        1.5 * H0_GYR * np.sqrt(OMEGA_L) * t) ** (2.0 / 3.0)


def z_of_t(t):
    return 1.0 / a_of_t(t) - 1.0


def rho_c(z):
    """Critical density [Msun / Mpc^3] at redshift z."""
    z = np.asarray(z, float)
    return RHO_C0 * (OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def r200c_analytic(logm, z):
    """R200c [physical kpc] of a halo of mass 10**logm [Msun] at redshift z."""
    return (3.0 * 10.0 ** np.asarray(logm, float)
            / (4.0 * np.pi * 200.0 * rho_c(z))) ** (1.0 / 3.0) * 1e3


# --------------------------------------------------------------------------- #
# the halo, as the engine reads it                                             #
# --------------------------------------------------------------------------- #
@dataclass
class HaloCurve:
    """One halo's growth curve: four DiffMAH numbers, plus the catalog radius
    history and the old step grid for the comparison mode and the bridge.
    Contains NO stellar information."""
    row: int
    index: int
    logmp: float          # log10 Mpeak at z = 0 (official anchor)
    logtc: float
    early: float
    late: float
    t_cat: np.ndarray     # (nc,) Gyr where the catalog R200c is finite
    r200c_cat: np.ndarray  # (nc,) physical kpc
    # the incumbent's own step grid, for `discrete_sum` (the bridge)
    t_steps: np.ndarray
    logmh_steps: np.ndarray
    dmh_steps: np.ndarray
    r200c_steps: np.ndarray
    ok_steps: np.ndarray
    epoch_mask_steps: np.ndarray   # (5, n_steps)


def build_curves(recs, verbose=True):
    """`HaloCurve` per exp54 `Halo` record, from the OFFICIAL DiffMAH fit.

    Refuses any galaxy whose DiffMAH source is not the official catalog, or
    whose analytic curve fails to reproduce the record's own step masses to
    1e-5 dex — the two ways the four numbers could silently be the wrong ones.
    """
    from astropy.table import Table
    tab = Table.read(DIFFMAH_TABLE)
    row_of = {int(i): k for k, i in enumerate(tab["index"])}
    out, worst = [], 0.0
    for h in recs:
        r = tab[row_of[int(h.index)]]
        if str(r["diffmah_source"]) != "official":
            raise ValueError(f"galaxy index {h.index}: DiffMAH source is "
                             f"{r['diffmah_source']!r}, not 'official'")
        ok = np.isfinite(h.r200c) & (h.r200c > 0)
        hc = HaloCurve(int(h.row), int(h.index),
                       float(r["official_logmp_z0"]), float(r["official_logtc"]),
                       float(r["official_early"]), float(r["official_late"]),
                       np.asarray(h.t[ok], float), np.asarray(h.r200c[ok], float),
                       np.asarray(h.t, float), np.asarray(h.logmh, float),
                       np.asarray(h.dmh, float), np.asarray(h.r200c, float),
                       ok, np.asarray(h.epoch_mask, bool))
        dev = float(np.max(np.abs(log_mah(np.log10(hc.t_steps), hc) - hc.logmh_steps)))
        worst = max(worst, dev)
        if dev > 1e-5:
            raise ValueError(f"galaxy index {h.index}: the official DiffMAH "
                             f"curve misses the record's masses by {dev:.2e} dex")
        out.append(hc)
    if verbose:
        print(f"  built {len(out)} halo curves from the official DiffMAH fit; "
              f"worst |dlogM| vs the records {worst:.1e} dex")
    return out


def log_mah(lt, hc):
    """log10 M(t) on lt = log10 t [Gyr]; vectorised over lt."""
    lt = np.asarray(lt, float)
    s = expit(MAH_K * (lt - hc.logtc))
    alpha = hc.early + (hc.late - hc.early) * s
    return hc.logmp + alpha * (lt - LOGT0)


def dm_dlnt(lt, hc):
    """dM/d ln t [Msun] on lt = log10 t."""
    lt = np.asarray(lt, float)
    s = expit(MAH_K * (lt - hc.logtc))
    alpha = hc.early + (hc.late - hc.early) * s
    dalpha = (hc.late - hc.early) * MAH_K * s * (1.0 - s)
    return 10.0 ** (hc.logmp + alpha * (lt - LOGT0)) * (alpha + dalpha * (lt - LOGT0))


def r200c_of(hc, lt, mode="analytic"):
    """R200c [kpc] at lt = log10 t: from M(t) and rho_c(z), or the catalog."""
    if mode == "analytic":
        t = 10.0 ** np.asarray(lt, float)
        return r200c_analytic(log_mah(lt, hc), z_of_t(t))
    if mode == "catalog":
        return 10.0 ** np.interp(np.asarray(lt, float), np.log10(hc.t_cat),
                                 np.log10(hc.r200c_cat))
    raise ValueError(f"mode must be 'analytic' or 'catalog', got {mode!r}")


# --------------------------------------------------------------------------- #
# the incumbent's laws, in the engine's variables                              #
# --------------------------------------------------------------------------- #
def _check_spec(spec):
    """Stage 0 serves the incumbent's class only: E2 efficiency, S2 size."""
    if (spec.eff != "E2" or spec.size != "S2" or spec.e3 or spec.conditioning
            or spec.size_time or spec.compact):
        raise NotImplementedError(
            f"the exp63 engine implements gompertz_log/moffat/sersic-E2-S2 "
            f"only; got {spec.label}")


def log_eps_e2(eff, logmh, z):
    m, x = logmh - M.M_PIV, np.log1p(z)
    return eff[0] + eff[1] * m + eff[2] * x + eff[3] * m * x


def r50_s2(size, z, r200):
    return 10.0 ** (size[0] + size[1] * np.log10(1.0 + z)) * r200


# --------------------------------------------------------------------------- #
# quadrature nodes shared by all galaxies and all five epochs                  #
# --------------------------------------------------------------------------- #
def nodes(t_start=T_START, t_anchor=T_ANCHOR, n_early=8, n_node=24):
    """Gauss-Legendre nodes in ln t: `n_early` equal panels from t_start to the
    earliest anchor (z=2), then one panel per anchor interval.

    Returns (lt, w, include) with lt = log10 t at the nodes (N,), w the
    quadrature weights for d ln t (N,), and include (5, N) boolean: which nodes
    belong to each epoch's integral (epoch order = ANCHOR_Z: z = 0.4 first).
    """
    ta = np.asarray(t_anchor, float)             # descending in z: t72 > ... > t33
    edges = list(np.linspace(np.log(t_start), np.log(ta[-1]), n_early + 1))
    for k in range(len(ta) - 2, -1, -1):          # t33 -> t40 -> ... -> t72
        edges.append(np.log(ta[k]))
    edges = np.array(edges)
    xg, wg = np.polynomial.legendre.leggauss(n_node)
    tau, w, panel = [], [], []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        tau.append(0.5 * (b - a) * xg + 0.5 * (b + a))
        w.append(0.5 * (b - a) * wg)
        panel.append(np.full(n_node, i))
    tau, w, panel = np.concatenate(tau), np.concatenate(w), np.concatenate(panel)
    n_panel = len(edges) - 1
    include = np.zeros((len(ta), len(tau)), bool)
    for k in range(len(ta)):                      # epoch k ends at anchor ta[k]
        last_panel = n_early + (len(ta) - 1 - k)  # panels are numbered by increasing t
        include[k] = panel < last_panel
    assert include[0].all() and include[-1].sum() == n_early * n_node
    return tau / np.log(10.0), w, include, n_panel


# --------------------------------------------------------------------------- #
# the model                                                                    #
# --------------------------------------------------------------------------- #
def _deposits(spec, theta, curves, lt, mode):
    """(logmh, z, r200, dm_dlnt, r50, r_trunc) as (n, N) arrays at the nodes."""
    eff, size, shape = spec.unpack(theta)
    n, N = len(curves), len(lt)
    lm = np.empty((n, N)); dm = np.empty((n, N)); r200 = np.empty((n, N))
    for i, hc in enumerate(curves):
        lm[i] = log_mah(lt, hc)
        dm[i] = dm_dlnt(lt, hc)
        r200[i] = r200c_of(hc, lt, mode)
    z = np.broadcast_to(z_of_t(10.0 ** lt), (n, N))
    dmstar = 10.0 ** np.clip(log_eps_e2(eff, lm, z), -30, 10) * dm
    r50 = r50_s2(size, z, r200)
    return lm, z, r200, dmstar, r50, spec.trunc_C * r200, shape


def predict(spec, theta, curves, R, mode="analytic", truncate=True,
            t_start=T_START, n_early=8, n_node=24, block=400):
    """M*(<R) [Msun] at the five anchor epochs for every curve: (n, 5, len(R)).

    Batched: the nodes are shared, the kernel matrix is built once per block of
    galaxies, and each epoch's integral is a masked weighted sum over nodes.
    """
    _check_spec(spec)
    R = np.asarray(R, float)
    lt, w, include, _ = nodes(t_start, T_ANCHOR, n_early, n_node)
    out = np.empty((len(curves), len(include), len(R)))
    for lo in range(0, len(curves), block):
        cv = curves[lo:lo + block]
        _, _, _, dmstar, r50, r_tr, shape = _deposits(spec, theta, cv, lt, mode)
        n, N = dmstar.shape
        if truncate:
            B = M.cog_truncated(spec.family, shape, r50.ravel(), r_tr.ravel(), R)
        else:
            B = families.cog_unit(spec.family, r50.ravel(), shape, R)
        B = B.reshape(len(R), n, N)                        # (R, n, N)
        wd = dmstar * w[None, :]                           # (n, N)
        for k in range(len(include)):
            out[lo:lo + n, k] = np.einsum("rgn,gn->gr", B, wd * include[k][None, :])
    return out


def step_start(hc):
    """log10 of the time one snapshot BEFORE the first step `forward` keeps:
    the record dropped it, so it is recovered by inverting the curve."""
    j0 = int(np.argmax(hc.ok_steps))
    target = np.log10(10.0 ** hc.logmh_steps[j0] - hc.dmh_steps[j0])
    lt0 = np.log10(hc.t_steps[j0])
    return brentq(lambda lt: float(log_mah(lt, hc)) - target, lt0 - 1.0, lt0)


def discrete_sum(spec, theta, hc, R, sub=1, mode="catalog"):
    """The incumbent's end-of-step Riemann sum on the record's own kept steps,
    each split `sub` times on the analytic curve: (5, len(R)).

    `sub=1` with `mode="catalog"` IS `exp54/model.forward` (to rounding) for a
    galaxy without a mid-history R200c gap; larger `sub` walks toward `predict`.
    The first kept step starts one snapshot before the first kept time, which
    the record no longer carries, so it is recovered by inverting the curve.
    """
    _check_spec(spec)
    eff, size, shape = spec.unpack(theta)
    ok = hc.ok_steps
    lt_prev = step_start(hc)
    out = np.empty((len(ANCHOR_SNAP), len(R)))
    for k in range(len(ANCHOR_SNAP)):
        keep = hc.epoch_mask_steps[k] & ok
        tt = hc.t_steps[keep]
        lt_edges = np.r_[lt_prev, np.log10(tt)]
        lt_all = np.concatenate([np.linspace(a, b, sub + 1)[1:]
                                 for a, b in zip(lt_edges[:-1], lt_edges[1:])])
        lm_e = log_mah(np.r_[lt_edges[0], lt_all], hc)
        dmh = np.diff(10.0 ** lm_e)
        z = z_of_t(10.0 ** lt_all)
        r200 = r200c_of(hc, lt_all, mode)
        dm = 10.0 ** np.clip(log_eps_e2(eff, lm_e[1:], z), -30, 10) * dmh
        B = M.cog_truncated(spec.family, shape, r50_s2(size, z, r200),
                            spec.trunc_C * r200, R)
        out[k] = B @ dm
    return out


def mass_beyond_truncation(spec, theta, curves, t_start=T_START, n_early=8,
                           n_node=24):
    """Fraction of each galaxy's z=0.4 deposited stellar mass that an
    UNTRUNCATED kernel would place beyond 3 R200c(t') of its own deposit (the
    D3 measurement): (n,)."""
    _check_spec(spec)
    lt, w, include, _ = nodes(t_start, T_ANCHOR, n_early, n_node)
    _, _, _, dmstar, r50, r_tr, shape = _deposits(spec, theta, curves, lt, "analytic")
    n, N = dmstar.shape
    # unit-mass cumulative profile of every deposit evaluated at its own 3 R200c
    f_in = np.empty((n, N))
    for i in range(n):
        f_in[i] = np.diag(families.cog_unit(spec.family, r50[i], shape, r_tr[i]))
    wd = dmstar * w[None, :] * include[0][None, :]
    return ((1.0 - f_in) * wd).sum(1) / wd.sum(1)


# --------------------------------------------------------------------------- #
# closed forms (tech note 04) — used by the gates                              #
# --------------------------------------------------------------------------- #
def upper_gamma(q, x):
    """Gamma(q, x) for any real q, by the recurrence when q <= 0."""
    x = np.asarray(x, float)
    if q > 0:
        return gammaincc(q, x) * Gamma(q)
    return (upper_gamma(q + 1.0, x) - x ** q * np.exp(-x)) / q


def powerlaw_beta(alpha, a_M, a_z, b):
    """Slope of the deposit-size distribution for a power-law MAH in EdS."""
    return (3.0 * (1.0 + a_M) * alpha - 2.0 * a_z * np.log(10.0)) / (alpha + 2.0 * (1.0 - b))


def gompertz_closed_form(R, A, beta, c, s_min, s_max):
    """Eq. 5 of tech note 04: the untruncated gompertz_log CoG of W = A s^beta."""
    q = beta / c
    return (A * R ** beta * np.log(2.0) ** (-q) / c
            * (upper_gamma(q, np.log(2.0) * (s_min / R) ** c)
               - upper_gamma(q, np.log(2.0) * (s_max / R) ** c)))


def mellin_constant(family, beta, shape):
    """int_0^inf y^(-beta-1) F(y) dy for the three kernel families."""
    from scipy.special import beta as Beta
    if family == "gompertz_log":
        c = shape[0]
        return np.log(2.0) ** (-beta / c) * Gamma(beta / c) / c
    if family == "moffat":
        g = shape[0]
        return (g - 1.0) / beta * Beta(1.0 - beta / 2.0, g - 1.0 + beta / 2.0)
    if family == "sersic":
        n = shape[0]
        return Gamma(2.0 * n - n * beta) / (beta * Gamma(2.0 * n))
    raise ValueError(family)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time
    import stage33 as S33
    import stage37_size_epoch as S37
    from scipy.integrate import quad
    from hongshao.tng_data import TNG_COSMO

    R = F.R_GRID
    recs = H.build_records(rows=np.arange(0, 2397, 40), verbose=False)
    curves = build_curves(recs)
    spec = S33.spec_from_label("gompertz_log-E2-S2")
    theta = S37.incumbent()

    # (a) the curve reproduces the records — asserted inside build_curves
    # (b) closed-form z(t) against astropy
    t = np.geomspace(0.2, 13.8, 200)
    za = np.array([float(v) for v in np.interp(
        t, TNG_COSMO.age(np.linspace(0, 20.5, 4000)).value[::-1],
        np.linspace(0, 20.5, 4000)[::-1])])
    dz = float(np.max(np.abs(z_of_t(t) - za)))
    print(f"  (b) closed-form z(t) vs astropy: max |dz| = {dz:.1e}")
    assert dz < 1e-4

    # (c) the bridge: discrete_sum(sub=1, catalog) == exp54 forward on no-gap galaxies
    worst, n_gap = 0.0, 0
    for h, hc in zip(recs, curves):
        j0 = int(np.argmax(hc.ok_steps))
        if (~hc.ok_steps[j0:]).any():
            n_gap += 1
            continue
        ref = M.forward(spec, theta, h, [0, 1, 2, 3, 4], R)
        got = discrete_sum(spec, theta, hc, R, sub=1)
        worst = max(worst, float(np.max(np.abs(np.log10(got / ref)))))
    print(f"  (c) discrete_sum(sub=1) vs exp54 forward, {len(curves) - n_gap} "
          f"no-gap galaxies: max |dlog| = {worst:.1e} dex ({n_gap} with a "
          f"mid-history gap skipped)")
    # The record stores the official curve as float32 (`diffmah_log_mah_fit`),
    # which is 2.7e-6 dex at these masses; the bridge cannot beat the precision
    # of what it reproduces, so the gate is three times that, not 1e-6.
    assert worst < 1e-5

    # (d) predict(catalog) is the limit of the refined sums; (e) node convergence
    hc = curves[3]
    pc = predict(spec, theta, [hc], R, mode="catalog",
                 t_start=10.0 ** step_start(hc))[0]
    sums = {s: discrete_sum(spec, theta, hc, R, sub=s) for s in (2, 4, 8, 16)}
    seq = [float(np.max(np.abs(np.log10(sums[s] / pc)))) for s in (2, 4, 8, 16)]
    ratios = np.array(seq[:-1]) / np.array(seq[1:])
    # first-order convergence: the error halves per doubling, so the Richardson
    # extrapolation 2 S16 - S8 removes the leading term and must land on predict
    rich = float(np.max(np.abs(np.log10((2.0 * sums[16] - sums[8]) / pc))))
    print(f"  (d) refined sums vs predict(catalog): sub 2/4/8/16 -> "
          + " / ".join(f"{v:.2e}" for v in seq) + f" dex; error ratios per "
          f"doubling {ratios.round(2)}; Richardson-extrapolated limit vs "
          f"predict {rich:.1e} dex")
    assert np.all((ratios > 1.8) & (ratios < 2.2)) and rich < 5e-4
    p1 = predict(spec, theta, curves, R, n_early=8, n_node=24)
    p2 = predict(spec, theta, curves, R, n_early=16, n_node=32)
    dn = float(np.max(np.abs(np.log10(p1 / p2))))
    print(f"  (e) nodes (8x24) vs (16x32): max |dlog| = {dn:.1e} dex")
    # The quadrature itself is converged (the Richardson limit above lands on
    # predict to ~1e-5); what remains between node sets is the linear-table
    # interpolation inside exp54's `cog_truncated` (1024-point kernel table,
    # 512-point truncation inversion), which moves with the node positions.
    # That floor is inherited with the kernel, so the gate is 1e-5, not 1e-6.
    assert dn < 1e-5

    # (f) Contract B
    assert not (set(HaloCurve.__dataclass_fields__) & set(STELLAR_FIELDS))
    print("  (f) HaloCurve carries no stellar field")

    # (g) closed forms
    eff, size, shape = spec.unpack(theta)
    worst_g = 0.0
    for alpha in (0.5, 1.0, 2.0, 3.0):
        beta = powerlaw_beta(alpha, eff[1], eff[2], size[1])
        tau = np.linspace(np.log(0.3), np.log(9.39), 40001); tt = np.exp(tau)
        Mt = 10 ** 13.5 * (tt / 9.39) ** alpha; opz = (13.8 / tt) ** (2.0 / 3.0)
        r200 = (3 * Mt / (4 * np.pi * 200 * RHO_C0 * opz ** 3)) ** (1 / 3) * 1e3
        s = 10 ** (size[0] + size[1] * np.log10(opz)) * r200
        eps = 10 ** (eff[0] + eff[1] * (np.log10(Mt) - 13.5) + eff[2] * np.log(opz))
        Wtau = eps * alpha * Mt
        brute = np.trapezoid(families.cog_unit("gompertz_log", s, shape, R) * Wtau[None, :], tau, axis=1)
        gr = alpha / 3 + 2 * (1 - size[1]) / 3
        A = (Wtau[-1] / gr) / s[-1] ** beta
        cf = gompertz_closed_form(R, A, beta, shape[0], s[0], s[-1])
        worst_g = max(worst_g, float(np.max(np.abs(np.log10(cf / brute)))))
    for fam, sh, bt in (("moffat", (1.3771,), 0.7), ("sersic", (4.0,), 0.5),
                        ("sersic", (1.0,), 1.2), ("gompertz_log", (0.8,), 0.5)):
        # the Mellin constants are for the kernels in their NATIVE scale
        # (moffat rc, sersic a); `cog_unit` is parametrised by R50, so pass the
        # R50 that makes the native scale equal to 1
        r50_native = (families.scale_from_r50(fam, np.array([1.0]), sh) ** -1
                      if fam != "gompertz_log" else np.array([1.0]))
        num = quad(lambda y: y ** (-bt - 1) * families.cog_unit(fam, r50_native, sh, np.array([y]))[0, 0],
                   0, np.inf, limit=400)[0]
        worst_g = max(worst_g, abs(num / mellin_constant(fam, bt, sh) - 1))
    print(f"  (g) closed forms (Eq. 5 at four MAH slopes; Mellin constants of "
          f"three kernels): worst {worst_g:.1e}")
    assert worst_g < 1e-8

    # timing, informational
    t0 = time.time(); predict(spec, theta, curves, R); dt = time.time() - t0
    print(f"  timing: {len(curves)} galaxies x 5 epochs in {dt:.2f} s "
          f"({1e3 * dt / len(curves):.1f} ms per galaxy)")
    print("\nengine.py self-check OK")
