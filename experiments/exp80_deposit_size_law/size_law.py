"""exp80 — the deposit SIZE LAW, generalised, for the frozen-theta tests
(Stage 0 C) and the Stage 1 fit.

The baseline (exp63's two-channel model, `model2._deposits2`) sizes each
deposit made at time t' (redshift z') by

    s_c = 10^log_f_c (1+z')^b_c                    [kpc]        the compact channel
    s_e = 10^log_f_e (1+z')^b_e R200c(t')          [kpc]        the extended channel

(`compact_in_kpc=True` in the adopted spec). This module adds the candidate
changes of the exp80 plan as named knobs, EVERY ONE NESTING AT THE BASELINE
(`LAW_DEFAULT`), so that `predict_law(..., LAW_DEFAULT)` reproduces
`model2.predict2` bit for bit (`selfcheck` asserts it):

  g_e, g_c        (b) a halo-mass exponent at fixed z': (M(t') / 10^13)^g on
                  s_e / s_c. g = -1/3 makes the deposit a FIXED PHYSICAL size
                  at every halo mass (R200c is proportional to M^(1/3) at
                  fixed z). model2's own levers, carried here.
  q_e             (a) the extended deposit's coordinate expands with the halo
                  after deposition: s_e is multiplied by
                  (R200c(t_obs) / R200c(t'))^q_e, so at q_e = 1 the size is set
                  by the halo radius at the OBSERVED epoch. "No transport" is
                  kept for the mass (nothing moves between deposits), not for
                  the coordinate. With q_e != 0 the extended kernel depends on
                  the epoch, so predict_law evaluates it per epoch.
  b_e2, z_brk_e   (c) a second (1+z') exponent for the extended channel above
  b_c2, z_brk_c   a break redshift: log s gains b2 * max(log(1+z') - log(1+z_brk), 0).
  early_kpc,      (d) the extended channel's EARLY deposits at a fixed physical
  early_b, z_sw   size: for deposits made before z_sw the size is
                  10^early_kpc (1+z')^early_b kpc instead of the R200c-scaled
                  law, blended over 0.1 dex in log(1+z') (`SWITCH_WIDTH`).
                  early_kpc = None switches it off.
  s_floor_kpc     a FLOOR on the extended deposit size in kpc (0 = off): early
                  deposits in small haloes cannot be smaller than it.
  a_early,        (e) exp83, the EARLY-MASS term. The conditioning variable of a
  s_early         deposit made at t' is the fraction of the halo's mass at t'
                  that was already assembled by T_EARLY_GYR = 2 Gyr,
                      phi(t') = min(log M(2 Gyr) - log M(t'), 0)      [dex, <= 0]
                  (zero for every deposit made before 2 Gyr; for the last
                  deposit before an epoch it is the per-galaxy early-mass
                  fraction log M(2 Gyr) - log Mh(z_k) that the residual is
                  regressed on), taken RELATIVE to the population's median at
                  that time, phi(t') - phi_ref(t') (`set_early_ref`, the same
                  device as model2's alpha_ref for g_rel): phi is monotone in
                  time for every halo, so the raw variable is a second time
                  law and its population mean belongs to a_z; the relative
                  variable says how much earlier than the typical halo of that
                  moment this one assembled. `a_early` multiplies the deposit's stellar
                  efficiency by 10^(a_early phi): a_early < 0 makes deposits in
                  haloes that grew much since 2 Gyr MORE efficient (early-formed
                  haloes lighter), a_early > 0 the reverse. `s_early` adds
                  s_early phi to the LOGIT of the compact share: s_early < 0
                  makes the same deposits more compact. `c_early` multiplies
                  the COMPACT deposit's size by 10^(c_early phi): c_early > 0
                  makes the deposits of haloes that grew more than typical
                  since 2 Gyr smaller. All three nest at zero.
  q_c, q_ch       (f) exp85, the COMPACT channel's post-deposition expansion (the
                  centre's mechanism). A compact deposit made at t' evaluated at
                  the observed epoch t_obs is multiplied by (t_obs / t')^q_c — an
                  AGE-driven expansion, the analogue of q_e for the in-situ
                  stars, decoupled from the halo's growth — or, the control
                  form, by (R200c(t_obs) / R200c(t'))^q_ch, coupled to it. The
                  oldest deposits expand the most, and less so at z = 2 than at
                  z = 0.4: a decline of the central mass at fixed total. With
                  either on the compact kernel is epoch dependent. The expanded
                  size is capped at the compact channel's truncation as in
                  exp84 (the cap never binds at the mean). Both nest at zero.

Parameter count: each knob used in a fit is one parameter (a break needs
two: the exponent and the break redshift; (d) needs three).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp63_analytic_growth", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402
import model as M                                        # noqa: E402
import model2 as M2                                      # noqa: E402
import families                                          # noqa: E402

LAW_DEFAULT = dict(g_e=0.0, g_c=0.0, q_e=0.0, b_e2=0.0, z_brk_e=2.0, b_c2=0.0, z_brk_c=2.0,
                   early_kpc=None, early_b=0.0, z_sw=2.0, s_floor_kpc=0.0, smooth_delay=False,
                   a_early=0.0, s_early=0.0, c_early=0.0, q_c=0.0, q_ch=0.0)
#: the reference time of the early-mass term (e): the mass assembled by 2 Gyr
#: (z = 3.2), the variable of exp81's residual regressions
T_EARLY_GYR = 2.0
#: `smooth_delay=True` with a spec carrying tau_d (exp63 Stage 2b): instead of
#: model2's STEP arrival (a node's extended mass has arrived or not, which makes
#: the loss a staircase in tau_d with no finite-difference gradient — seen
#: 2026-09-12: tau_d did not move from its start in 3000 evaluations), the
#: accreted mass arrives with an EXPONENTIAL distribution of merger times of
#: mean tau_d Hubble times at accretion: the fraction arrived by the epoch is
#: 1 - exp(-(t_k - t') H(z') / tau_d). Nests at tau_d -> 0; smooth in tau_d.
SWITCH_WIDTH = 0.1                                       # dex in log(1+z') for the (d) blend
CAP_OF_TRUNCATION = 0.3                                  # s_e <= 0.3 R_trunc (gompertz c = 0.8 allows 0.42)
LAW_KNOBS = tuple(LAW_DEFAULT)


def with_law(**kw):
    bad = [k for k in kw if k not in LAW_DEFAULT]
    if bad:
        raise ValueError(f"unknown size-law knobs {bad}; choose from {LAW_KNOBS}")
    law = dict(LAW_DEFAULT)
    law.update(kw)
    return law


def arrival_weights(spec2, p, law, lt, include):
    """(5, N) weights of the extended deposits that have arrived by each epoch:
    model2's boolean step, or the exponential arrival when `smooth_delay`."""
    if not spec2.delay or p["tau_d"] <= 0.0:
        return include.astype(float)
    if not law.get("smooth_delay", False):
        return M2.arrival_include(spec2, p, lt, include).astype(float)
    t = 10.0 ** np.asarray(lt, float)
    h = M2.hubble_gyr(E.z_of_t(t))
    out = np.zeros((len(include), len(t)))
    for k in range(len(include)):
        dt = np.clip(E.T_ANCHOR[k] - t, 0.0, None)
        out[k] = include[k] * (1.0 - np.exp(-dt * h / p["tau_d"]))
    return out


def describe(law):
    on = {k: v for k, v in law.items() if v != LAW_DEFAULT[k] and not (k in ("z_brk_e",) and law["b_e2"] == 0)
          and not (k in ("z_brk_c",) and law["b_c2"] == 0) and not (k in ("early_b", "z_sw") and law["early_kpc"] is None)}
    return ", ".join(f"{k}={v:+.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in on.items()) or "the baseline law"


def n_extra_parameters(law):
    """How many fitted parameters the knobs in use would add to the twelve."""
    n = 0
    n += law["g_e"] != 0.0
    n += law["g_c"] != 0.0
    n += law["q_e"] != 0.0
    n += 2 * (law["b_e2"] != 0.0)
    n += 2 * (law["b_c2"] != 0.0)
    n += 3 * (law["early_kpc"] is not None)
    n += law["s_floor_kpc"] != 0.0
    n += law.get("a_early", 0.0) != 0.0
    n += law.get("s_early", 0.0) != 0.0
    n += law.get("c_early", 0.0) != 0.0
    n += law.get("q_c", 0.0) != 0.0
    n += law.get("q_ch", 0.0) != 0.0
    n += law.get("smooth_delay", False) and 0     # tau_d is the spec's parameter, not a knob
    return int(n)


def sizes_at_nodes(spec2, p, law, lm, lz, r200, lr200_obs=None, phi=None):
    """(s_c, s_e) [kpc] at the nodes for one epoch, in model2's own arithmetic
    (the fraction's log clipped to (-8, 2), then multiplied by R200c) so the
    default law nests bit for bit. `lr200_obs` (n, 1) is log R200c at the
    observed epoch, needed only when q_e != 0; `phi` (n, N) the early-mass
    variable, needed only when c_early != 0."""
    dlm = lm - M2.LEVER_PIVOT_LOGM
    lf_c = p["log_f_c"] + p["b_c"] * lz + p.get("g_c", 0.0) * dlm
    if law["g_c"] != 0.0:
        lf_c = lf_c + law["g_c"] * dlm
    if law.get("c_early", 0.0) != 0.0:
        if phi is None:
            raise ValueError("c_early != 0 needs the early-mass variable phi")
        lf_c = lf_c + law["c_early"] * phi
    if law["b_c2"] != 0.0:
        lf_c = lf_c + law["b_c2"] * np.maximum(lz - np.log10(1.0 + law["z_brk_c"]), 0.0)
    s_c = 10.0 ** np.clip(lf_c, -8, 2) * (1.0 if spec2.compact_in_kpc else r200)
    lf_e = p["log_f_e"] + p["b_e"] * lz + p.get("g_e", 0.0) * dlm
    if law["g_e"] != 0.0:
        lf_e = lf_e + law["g_e"] * dlm
    if law["b_e2"] != 0.0:
        lf_e = lf_e + law["b_e2"] * np.maximum(lz - np.log10(1.0 + law["z_brk_e"]), 0.0)
    if law["q_e"] != 0.0:
        # the coordinate expands with the halo: the deposit's size AND its
        # truncation radius (C x R200c(t'), C untouched) grow by the same factor
        if lr200_obs is None:
            raise ValueError("q_e != 0 needs the observed epoch's R200c")
        r200 = r200 * 10.0 ** (law["q_e"] * (lr200_obs - np.log10(r200)))
    s_e = 10.0 ** np.clip(lf_e, -8, 2) * r200
    if law["early_kpc"] is not None:
        early = law["early_kpc"] + law["early_b"] * lz
        f = expit((lz - np.log10(1.0 + law["z_sw"])) / SWITCH_WIDTH)      # 1 for deposits before z_sw
        s_e = 10.0 ** (f * early + (1.0 - f) * np.log10(s_e))
    if law["s_floor_kpc"] > 0.0:
        s_e = np.maximum(s_e, law["s_floor_kpc"])
    r_tr = spec2.trunc_C * r200
    # a deposit cannot be more extended than its own truncation allows
    # (`model.solve_u`: R50 below 0.5^(1/c) x R_trunc); a fixed-kpc early size
    # in a tiny early halo would ask for it. The baseline never reaches the
    # cap (its s_e is at most 0.19 R200c = 0.063 R_trunc), so nesting holds.
    s_e = np.minimum(s_e, CAP_OF_TRUNCATION * r_tr)
    return s_c, s_e, r_tr


def epoch_dependent(law):
    return law["q_e"] != 0.0


def compact_epoch_dependent(law):
    return law.get("q_c", 0.0) != 0.0 or law.get("q_ch", 0.0) != 0.0


def compact_sizes_at_epoch(law, s_c, lt, k, r200_nodes, lr200_obs_k):
    """exp85: the compact deposits' sizes at the observed epoch k, (n, N):
    s_c(t') expanded by (t_k / t')^q_c (age driven) and by
    (R200c(t_k) / R200c(t'))^q_ch (halo driven). Deposits after t_k (excluded
    from the epoch's integral anyway) are left at s_c(t'). Uncapped here."""
    q_c, q_ch = law.get("q_c", 0.0), law.get("q_ch", 0.0)
    if q_c == 0.0 and q_ch == 0.0:
        return s_c
    lt_k = np.log10(E.T_ANCHOR[k])
    lgrow = np.zeros_like(s_c)
    if q_c != 0.0:
        lgrow = lgrow + q_c * np.maximum(lt_k - np.asarray(lt, float), 0.0)[None, :]
    if q_ch != 0.0:
        lgrow = lgrow + q_ch * np.maximum(lr200_obs_k[:, None] - np.log10(r200_nodes), 0.0)
    return s_c * 10.0 ** lgrow


_EARLY_REF = {"lt": None, "phi": None}


def early_fraction_raw(curves, lm):
    """phi(t') = min(log M(T_EARLY_GYR) - log M(t'), 0) at the nodes, (n, N):
    the fraction (in dex, <= 0) of the halo's mass at the deposit that was
    already in place at 2 Gyr; zero before 2 Gyr."""
    lm2 = np.array([E.log_mah(np.array([np.log10(T_EARLY_GYR)]), hc)[0] for hc in curves])
    return np.minimum(lm2[:, None] - lm, 0.0)


def set_early_ref(curves, nodes=None):
    """Tabulate the population-median phi(t') at the quadrature nodes from the
    fitting sample's curves (a fixed function stored with the fit, not a
    fitted quantity). Returns (lt, phi_ref)."""
    lt = E.nodes(**(nodes or M2.FULL_NODES))[0]
    lm = np.array([E.log_mah(lt, hc) for hc in curves])
    _EARLY_REF["lt"], _EARLY_REF["phi"] = lt, np.median(early_fraction_raw(curves, lm), axis=0)
    return lt, _EARLY_REF["phi"]


def early_ref(lt):
    if _EARLY_REF["lt"] is None:
        raise RuntimeError("call size_law.set_early_ref(curves) before using a_early / s_early")
    return np.interp(np.asarray(lt, float), _EARLY_REF["lt"], _EARLY_REF["phi"])


def early_fraction(curves, lm, lt):
    """The early-mass term's variable: phi(t') - phi_ref(t'), (n, N); positive
    for a halo that had assembled more of its mass by 2 Gyr than the typical
    halo of the sample had at the same t'."""
    return early_fraction_raw(curves, lm) - early_ref(lt)[None, :]


def deposits_law(spec2, theta, law, curves, lt, epochs=(0, 1, 2, 3, 4)):
    """(dm_c, dm_e, s_c, s_e_by_epoch, r_trunc_by_epoch): (n, N) arrays; the
    by-epoch ones are {k: (n, N)} (one shared array when the law is not epoch
    dependent). The compact channel's truncation radius is always C R200c(t')."""
    p = spec2.unpack(theta)
    n, N = len(curves), len(lt)
    lm = np.empty((n, N)); dm = np.empty((n, N)); r200 = np.empty((n, N))
    for i, hc in enumerate(curves):
        lm[i] = E.log_mah(lt, hc)
        dm[i] = E.dm_dlnt(lt, hc)
        r200[i] = E.r200c_of(hc, lt, "analytic")
    z = np.broadcast_to(E.z_of_t(10.0 ** lt), (n, N))
    eff = (p["a0"], p["a_M"], p["a_z"], p["a_Mz"])
    log_eps = E.log_eps_e2(eff, lm, z)
    a_early, s_early, c_early = law.get("a_early", 0.0), law.get("s_early", 0.0), law.get("c_early", 0.0)
    phi = early_fraction(curves, lm, lt) if (a_early != 0.0 or s_early != 0.0 or c_early != 0.0) else None
    if a_early != 0.0:
        log_eps = log_eps + a_early * phi
    dmstar = 10.0 ** np.clip(log_eps, -30, 10) * dm
    wc = M2.compact_share(p, lm, alpha=dm / 10.0 ** lm, lt_nodes=lt,
                          logit_extra=(s_early * phi if s_early != 0.0 else None))
    lz = np.log10(1.0 + z)
    s_e, r_tr = {}, {}
    if epoch_dependent(law):
        lt_k = np.log10(E.T_ANCHOR)
        for k in epochs:
            lr_obs = np.array([np.log10(E.r200c_of(hc, lt_k[k], "analytic")) for hc in curves])[:, None]
            s_c, s_e[k], r_tr[k] = sizes_at_nodes(spec2, p, law, lm, lz, r200, lr_obs, phi=phi)
    else:
        s_c, shared, tr = sizes_at_nodes(spec2, p, law, lm, lz, r200, phi=phi)
        s_e = {k: shared for k in epochs}
        r_tr = {k: tr for k in epochs}
    return dmstar * wc, dmstar * (1.0 - wc), s_c, s_e, r_tr


def predict_law(spec2, theta, law, curves, R, epochs=(0, 1, 2, 3, 4), nodes=M2.FULL_NODES, block=300,
                size_dev=None):
    """M*(<R) [Msun] at the requested epochs under the size law: (n, len(epochs), len(R)).

    `size_dev` (exp84, the stochastic layer): per-galaxy, PER-EPOCH deviations
    of the two deposit sizes, `dict(c=(n, 5), e=(n, 5))` in dex, indexed by
    the epoch number (columns 0..4 = z 0.4 .. 2). At the observed epoch k
    every compact deposit's size is multiplied by 10^dev_c[:, k] and every
    extended deposit's by 10^dev_e[:, k]; the sizes are re-capped at the
    truncation limit. The deviation acts at the EVALUATION of an epoch, not on
    the deposits' history, so a galaxy may be drawn compact at one epoch and
    extended at another (the layer's cross-epoch correlation is the draw's).
    `None` (or an all-zero array) reproduces the mean bit for bit.
    """
    p = spec2.unpack(theta)
    R = np.asarray(R, float)
    lt, w, include, _ = E.nodes(**nodes)
    epochs = list(epochs)
    out = np.empty((len(curves), len(epochs), len(R)))
    for lo in range(0, len(curves), block):
        cv = curves[lo:lo + block]
        dm_c, dm_e, s_c, s_e, r_tr = deposits_law(spec2, theta, law, cv, lt, epochs)
        n, N = dm_c.shape
        r200_nodes = np.array([E.r200c_of(hc, lt, "analytic") for hc in cv])
        r_tr_c = spec2.trunc_C * r200_nodes
        dev_c = dev_e = None
        if size_dev is not None:
            dev_c = np.asarray(size_dev["c"], float)[lo:lo + n]
            dev_e = np.asarray(size_dev["e"], float)[lo:lo + n]
            assert dev_c.shape == dev_e.shape == (n, 5), (dev_c.shape, dev_e.shape, n)
        expanding = compact_epoch_dependent(law)
        lr200_obs = None
        if law.get("q_ch", 0.0) != 0.0:
            lt_k = np.log10(E.T_ANCHOR)
            lr200_obs = {k: np.array([np.log10(E.r200c_of(hc, lt_k[k], "analytic")) for hc in cv]) for k in epochs}
        Bc_shared = None
        if dev_c is None and not expanding:
            Bc_shared = M.cog_truncated(M2.COMPACT_FAMILY, (p["n_c"],), s_c.ravel(), r_tr_c.ravel(), R).reshape(len(R), n, N)
        wc_ = dm_c * w[None, :]; we_ = dm_e * w[None, :]
        inc_e = arrival_weights(spec2, p, law, lt, include)
        Be_shared = None
        for j, k in enumerate(epochs):
            if Bc_shared is not None:
                Bc = Bc_shared
            else:
                # the cap never goes below the mean's own size (the baseline does not
                # cap the compact channel), so a zero deviation nests bit for bit
                s_c_k = compact_sizes_at_epoch(law, s_c, lt, k, r200_nodes, None if lr200_obs is None else lr200_obs[k])
                if dev_c is not None:
                    s_c_k = s_c_k * 10.0 ** dev_c[:, k][:, None]
                s_c_k = np.minimum(s_c_k, np.maximum(s_c, CAP_OF_TRUNCATION * r_tr_c))
                Bc = M.cog_truncated(M2.COMPACT_FAMILY, (p["n_c"],), s_c_k.ravel(), r_tr_c.ravel(), R).reshape(len(R), n, N)
            if Be_shared is None or epoch_dependent(law) or dev_e is not None:
                s_e_k = s_e[k]
                if dev_e is not None:
                    s_e_k = np.minimum(s_e_k * 10.0 ** dev_e[:, k][:, None], np.maximum(s_e_k, CAP_OF_TRUNCATION * r_tr[k]))
                Be = M.cog_truncated(spec2.extended_family, (p["c_e"],), s_e_k.ravel(), r_tr[k].ravel(), R).reshape(len(R), n, N)
                if not epoch_dependent(law) and dev_e is None:
                    Be_shared = Be
            else:
                Be = Be_shared
            out[lo:lo + n, j] = (np.einsum("rgn,gn->gr", Bc, wc_ * include[k][None, :])
                                 + np.einsum("rgn,gn->gr", Be, we_ * inc_e[k][None, :]))
    return out


def selfcheck_size_dev(spec2, theta, law, curves, R):
    """exp84: a zero deviation reproduces the mean bit for bit; a deviation of
    one axis at one epoch moves that epoch only, in the expected direction
    (a larger compact size lowers the mass inside 5 kpc; a larger extended
    size lowers the mass inside 30 kpc), and leaves the total at the grid's
    end nearly unchanged (mass is conserved, only the truncation cap moves it)."""
    n = len(curves)
    ref = predict_law(spec2, theta, law, curves, R)
    zero = dict(c=np.zeros((n, 5)), e=np.zeros((n, 5)))
    assert np.array_equal(predict_law(spec2, theta, law, curves, R, size_dev=zero), ref), "a zero size_dev does not nest"
    i5, i30 = int(np.argmin(np.abs(R - 4.92))), int(np.argmin(np.abs(R - 30.0)))
    for axis, idx, label in (("c", i5, "M(<5 kpc)"), ("e", i30, "M(<30 kpc)")):
        for k in (0, 4):
            dev = dict(c=np.zeros((n, 5)), e=np.zeros((n, 5)))
            dev[axis][:, k] = 0.3
            got = predict_law(spec2, theta, law, curves, R, size_dev=dev)
            other = [j for j in range(5) if j != k]
            assert np.array_equal(got[:, other], ref[:, other]), (axis, k, "another epoch moved")
            moved = np.median(np.log10(got[:, k, idx] / ref[:, k, idx]))
            assert moved < -1e-3, (axis, k, moved)
            end = float(np.max(np.abs(np.log10(got[:, k, -1] / ref[:, k, -1]))))
            print(f"    size_dev {axis} +0.3 dex at epoch {k}: {label} moves {moved:+.3f} dex (median); "
                  f"the grid-end total by at most {end:.3f} dex; the other epochs untouched")
    print("  size_law selfcheck_size_dev OK: a zero deviation nests bit for bit")


def selfcheck(spec2, theta, curves, R):
    """The default law reproduces predict2 bit for bit; each knob at its
    nesting value changes nothing; each knob switched on changes something."""
    ref = M2.predict2(spec2, theta, curves, R)
    got = predict_law(spec2, theta, LAW_DEFAULT, curves, R)
    assert np.array_equal(got, ref), "LAW_DEFAULT does not nest"
    for kw in (dict(g_e=-1.0 / 3.0), dict(g_c=-1.0 / 3.0), dict(q_e=1.0), dict(b_e2=0.5, z_brk_e=2.0),
               dict(b_c2=0.5, z_brk_c=2.0), dict(early_kpc=np.log10(4.0), early_b=0.0, z_sw=2.0), dict(s_floor_kpc=4.0),
               dict(a_early=-0.5), dict(s_early=-2.0), dict(c_early=0.5), dict(q_c=0.5), dict(q_ch=0.5)):
        d = predict_law(spec2, theta, with_law(**kw), curves, R)
        assert np.all(np.isfinite(d)) and np.all(np.diff(d, axis=2) >= -1e-9), kw
        moved = float(np.max(np.abs(np.log10(d / ref))))
        assert moved > 1e-4, (kw, moved)
        print(f"    knob {describe(with_law(**kw)):<44} moves the profile by up to {moved:.3f} dex; monotone in R")
    print("  size_law selfcheck OK: LAW_DEFAULT == model2.predict2 bit for bit")


def selfcheck_compact_expansion(spec2, theta, law, curves, R):
    """exp85: q_c lowers the mass inside 5 kpc at every epoch, MORE at z = 0.4
    than at z = 2 (the same deposit is older there), and leaves the grid-end
    total nearly unchanged; the halo form q_ch likewise. With a size_dev the
    expansion and the deviation compose (a zero deviation nests)."""
    ref = predict_law(spec2, theta, law, curves, R)
    i5 = int(np.argmin(np.abs(R - 4.92)))
    n = len(curves)
    for knob in ("q_c", "q_ch"):
        got = predict_law(spec2, theta, with_law(**{**{k: v for k, v in law.items() if k != knob}, knob: 0.5}), curves, R)
        d5 = [float(np.median(np.log10(got[:, k, i5] / ref[:, k, i5]))) for k in range(5)]
        end = float(np.max(np.abs(np.log10(got[:, :, -1] / ref[:, :, -1]))))
        assert all(v < -1e-3 for v in d5), (knob, d5)
        assert d5[0] < d5[4], (knob, "z=0.4 must expand more than z=2", d5)
        print(f"    {knob} = 0.5: M(<5 kpc) moves (median) z=0.4..2 " + " / ".join(f"{v:+.3f}" for v in d5)
              + f" dex; the grid-end total by at most {end:.3f} dex")
        zero = dict(c=np.zeros((n, 5)), e=np.zeros((n, 5)))
        law_on = with_law(**{**{k: v for k, v in law.items() if k != knob}, knob: 0.5})
        assert np.array_equal(predict_law(spec2, theta, law_on, curves, R, size_dev=zero), got), (knob, "size_dev=0 does not nest")
    print("  size_law selfcheck_compact_expansion OK")


if __name__ == "__main__":
    import halo as H
    import stage2_fit as S2F
    fz = np.load(ROOT / "experiments/exp63_analytic_growth/outputs/stage2_fit_joint_kpc_free_sane.npz", allow_pickle=True)
    spec2 = S2F.spec_from_fit(fz)
    th = np.asarray(np.load(ROOT / "experiments/exp74_c19_history_leak/outputs/stage1_refit_measured.npz",
                            allow_pickle=True)["theta_best"], float)
    import fit as F
    recs = H.build_records(rows=np.arange(0, 2397, 60), verbose=False)
    curves = E.build_curves(recs, verbose=False)
    set_early_ref(curves)
    selfcheck(spec2, th, curves, F.R_GRID)
    spec_d = M2.Spec2(theta_names=M2.THETA_NAMES_DELAY, extended_family=spec2.extended_family,
                      compact_in_kpc=spec2.compact_in_kpc)
    th_d = np.append(th, 0.15)                              # the adopted mean's structure: tau_d held, q_e on
    selfcheck_size_dev(spec_d, th_d, with_law(q_e=0.15), curves, F.R_GRID)
    selfcheck_compact_expansion(spec_d, th_d, with_law(q_e=0.15), curves, F.R_GRID)
