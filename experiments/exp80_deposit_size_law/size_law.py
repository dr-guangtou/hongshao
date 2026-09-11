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
                   early_kpc=None, early_b=0.0, z_sw=2.0, s_floor_kpc=0.0, smooth_delay=False)
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
    n += law.get("smooth_delay", False) and 0     # tau_d is the spec's parameter, not a knob
    return int(n)


def sizes_at_nodes(spec2, p, law, lm, lz, r200, lr200_obs=None):
    """(s_c, s_e) [kpc] at the nodes for one epoch, in model2's own arithmetic
    (the fraction's log clipped to (-8, 2), then multiplied by R200c) so the
    default law nests bit for bit. `lr200_obs` (n, 1) is log R200c at the
    observed epoch, needed only when q_e != 0."""
    dlm = lm - M2.LEVER_PIVOT_LOGM
    lf_c = p["log_f_c"] + p["b_c"] * lz + p.get("g_c", 0.0) * dlm
    if law["g_c"] != 0.0:
        lf_c = lf_c + law["g_c"] * dlm
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
    dmstar = 10.0 ** np.clip(E.log_eps_e2(eff, lm, z), -30, 10) * dm
    wc = M2.compact_share(p, lm, alpha=dm / 10.0 ** lm, lt_nodes=lt)
    lz = np.log10(1.0 + z)
    s_e, r_tr = {}, {}
    if epoch_dependent(law):
        lt_k = np.log10(E.T_ANCHOR)
        for k in epochs:
            lr_obs = np.array([np.log10(E.r200c_of(hc, lt_k[k], "analytic")) for hc in curves])[:, None]
            s_c, s_e[k], r_tr[k] = sizes_at_nodes(spec2, p, law, lm, lz, r200, lr_obs)
    else:
        s_c, shared, tr = sizes_at_nodes(spec2, p, law, lm, lz, r200)
        s_e = {k: shared for k in epochs}
        r_tr = {k: tr for k in epochs}
    return dmstar * wc, dmstar * (1.0 - wc), s_c, s_e, r_tr


def predict_law(spec2, theta, law, curves, R, epochs=(0, 1, 2, 3, 4), nodes=M2.FULL_NODES, block=300):
    """M*(<R) [Msun] at the requested epochs under the size law: (n, len(epochs), len(R))."""
    p = spec2.unpack(theta)
    R = np.asarray(R, float)
    lt, w, include, _ = E.nodes(**nodes)
    epochs = list(epochs)
    out = np.empty((len(curves), len(epochs), len(R)))
    for lo in range(0, len(curves), block):
        cv = curves[lo:lo + block]
        dm_c, dm_e, s_c, s_e, r_tr = deposits_law(spec2, theta, law, cv, lt, epochs)
        n, N = dm_c.shape
        r_tr_c = spec2.trunc_C * np.array([E.r200c_of(hc, lt, "analytic") for hc in cv])
        Bc = M.cog_truncated(M2.COMPACT_FAMILY, (p["n_c"],), s_c.ravel(), r_tr_c.ravel(), R).reshape(len(R), n, N)
        wc_ = dm_c * w[None, :]; we_ = dm_e * w[None, :]
        inc_e = arrival_weights(spec2, p, law, lt, include)
        Be_shared = None
        for j, k in enumerate(epochs):
            if Be_shared is None or epoch_dependent(law):
                Be = M.cog_truncated(spec2.extended_family, (p["c_e"],), s_e[k].ravel(), r_tr[k].ravel(), R).reshape(len(R), n, N)
                if not epoch_dependent(law):
                    Be_shared = Be
            else:
                Be = Be_shared
            out[lo:lo + n, j] = (np.einsum("rgn,gn->gr", Bc, wc_ * include[k][None, :])
                                 + np.einsum("rgn,gn->gr", Be, we_ * inc_e[k][None, :]))
    return out


def selfcheck(spec2, theta, curves, R):
    """The default law reproduces predict2 bit for bit; each knob at its
    nesting value changes nothing; each knob switched on changes something."""
    ref = M2.predict2(spec2, theta, curves, R)
    got = predict_law(spec2, theta, LAW_DEFAULT, curves, R)
    assert np.array_equal(got, ref), "LAW_DEFAULT does not nest"
    for kw in (dict(g_e=-1.0 / 3.0), dict(g_c=-1.0 / 3.0), dict(q_e=1.0), dict(b_e2=0.5, z_brk_e=2.0),
               dict(b_c2=0.5, z_brk_c=2.0), dict(early_kpc=np.log10(4.0), early_b=0.0, z_sw=2.0), dict(s_floor_kpc=4.0)):
        d = predict_law(spec2, theta, with_law(**kw), curves, R)
        assert np.all(np.isfinite(d)) and np.all(np.diff(d, axis=2) >= -1e-9), kw
        moved = float(np.max(np.abs(np.log10(d / ref))))
        assert moved > 1e-4, (kw, moved)
        print(f"    knob {describe(with_law(**kw)):<44} moves the profile by up to {moved:.3f} dex; monotone in R")
    print("  size_law selfcheck OK: LAW_DEFAULT == model2.predict2 bit for bit")


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
    selfcheck(spec2, th, curves, F.R_GRID)
