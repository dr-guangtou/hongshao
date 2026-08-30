"""exp63 Stage 2 — the two-channel deposition model on the exact-integral engine.

Stage 1 read, from the z=0.4 curves of growth alone, that the stellar mass of a
massive galaxy sits in TWO deposit-size modes: a compact one at a fixed few
kpc, and an extended one at ~0.1 R200c whose share rises with halo mass, with
almost nothing between them. This is that finding as a forward model:

    M*(<R, t) = int dM/dt' eps(M, z) [ w_c(M) F_c(R / s_c(t'))
                                     + (1 - w_c(M)) F_e(R / s_e(t')) ] dt'

  eps        the incumbent's E2 efficiency (a0, a_M, a_z, a_Mz), unchanged
  w_c(M)     the COMPACT share of each deposit, a logistic in the halo mass
             AT THE DEPOSIT'S OWN TIME: expit((m_half - log10 M) / d_split)
             — falling toward high mass, the ex-situ picture
  s_c, s_e   the two size laws, f (1+z)^b R200c(t') each (log_f_c, b_c,
             log_f_e, b_e)
  F_c        a Sersic deposit with index n_c (Stage 1: near 1; n = 4 is
             rejected by the data), truncated at 3 R200c like everything else
  F_e        the incumbent's gompertz_log deposit with shape c_e

Twelve parameters. NESTING: `nested_theta(theta_incumbent)` sets the compact
share to exactly zero (m_half = -1000) and the extended channel to the
incumbent's size law and shape, so `predict2` reproduces `engine.predict` bit
for bit — the self-check asserts it. That is the warrant for reading any fit
of this model as "the incumbent plus a compact channel".

Run the self-check:
    HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
        experiments/exp63_analytic_growth/model2.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude", HERE):
    sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402  (imports fit first)
import model as M                                        # noqa: E402
import families                                          # noqa: E402

THETA_NAMES = ("a0", "a_M", "a_z", "a_Mz", "m_half", "d_split",
               "log_f_c", "b_c", "log_f_e", "b_e", "n_c", "c_e")
#: Stage 2b: the extended channel's ACCRETION-TO-DEPOSITION DELAY. A satellite's
#: stars join the central only when the merger completes, a dynamical-friction
#: time after the halo accreted the satellite; the model credits the extended
#: deposit of halo mass accreted at t' at t' + tau_d / H(z(t')) — tau_d in
#: Hubble times at accretion. tau_d = 0 is the 12-parameter model exactly.
#: Not a radial transport: nothing moves after it is deposited; mass accreted
#: but not yet arrived is "in transit", which is what a central's measured
#: stellar mass excludes.
THETA_NAMES_DELAY = THETA_NAMES + ("tau_d",)
#: Stage 2c: the split also depends on the halo's GROWTH RATE at the deposit,
#: alpha = dlnM/dlnt: w_c = expit((m_half - log M + g_split (alpha - 1)) / d).
#: g_split < 0 makes deposits laid down during fast growth (mergers) more
#: extended and those during slow growth (smooth accretion) more compact —
#: the in-situ / ex-situ picture. g_split = 0 is the 12-parameter model.
THETA_NAMES_GROWTH = THETA_NAMES + ("g_split",)
#: Stage 2d: the same split on the growth rate RELATIVE to the population's
#: typical rate at that time, alpha - alpha_ref(t). At high redshift every
#: halo grows fast, so an absolute rate makes all early deposits extended
#: (Stage 2c's failure); mergers are growth in excess of the typical. The
#: reference is a fixed function set once from the sample (`set_alpha_ref`)
#: and stored with the fit; it is not a fitted quantity.
THETA_NAMES_GROWTH_REL = THETA_NAMES + ("g_rel",)
#: The single-epoch round (2026-08-28): optional LEVERS, each nesting at zero.
#:   g_c, g_e  halo-mass exponents on the two sizes, s = f (1+z)^b R200c(t')
#:             (M(t') / 10^13)^g. Since R200c is proportional to M^(1/3) at
#:             fixed z, g = -1/3 makes the deposit a FIXED PHYSICAL size at
#:             every halo mass (Stage 1: the compact mode does not move with
#:             halo mass; the model's s_c does, by R200c).
#:   w_min     a floor on the compact share: w_c = w_min + (1 - w_min) sigma(.)
#:             (the mass-only split sends the heaviest haloes' share to zero;
#:             Stage 1 wants 0.43 in the top tercile).
LEVER_NAMES = ("g_c", "g_e", "w_min")
LEVER_PIVOT_LOGM = 13.0
#: `Spec2(compact_in_kpc=True)`: the compact size is a PHYSICAL scale,
#: s_c = 10^log_f_c (1+z)^b_c kpc, not a fraction of R200c(t'). Stage 1 found
#: the compact mode at a fixed few kpc at every halo mass; tied to R200c(t')
#: the model spreads a galaxy's compact deposits over ~10x in size along its
#: history. Then `log_f_c` is log10(kpc) with bounds (-0.5, 1.5), and the
#: nested incumbent is unchanged (compact share zero).
COMPACT_KPC_BOUNDS = (-0.5, 1.5)
_ALPHA_REF = {"lt": None, "alpha": None}


def set_alpha_ref(curves, nodes=None):
    """Tabulate the population-median dlnM/dlnt at the quadrature nodes."""
    lt = E.nodes(**(nodes or FULL_NODES))[0]
    a = np.array([E.dm_dlnt(lt, hc) / 10.0 ** E.log_mah(lt, hc) for hc in curves])
    _ALPHA_REF["lt"], _ALPHA_REF["alpha"] = lt, np.median(a, axis=0)
    return lt, _ALPHA_REF["alpha"]


def alpha_ref(lt):
    if _ALPHA_REF["lt"] is None:
        raise RuntimeError("call model2.set_alpha_ref(curves) before using g_rel")
    return np.interp(np.asarray(lt, float), _ALPHA_REF["lt"], _ALPHA_REF["alpha"])
BOUNDS = {"a0": (-6.0, 2.0), "a_M": (-3.0, 3.0), "a_z": (-3.0, 3.0), "a_Mz": (-3.0, 3.0),
          "m_half": (10.5, 15.0), "d_split": (0.1, 3.0),
          "log_f_c": (-3.0, -0.5), "b_c": (-3.0, 3.0),
          "log_f_e": (-2.5, 0.0), "b_e": (-3.0, 3.0),
          "n_c": (0.5, 2.5), "c_e": (0.2, 8.0), "tau_d": (0.0, 3.0), "g_split": (-3.0, 3.0),
          "g_rel": (-3.0, 3.0), "g_c": (-1.5, 1.5), "g_e": (-1.5, 1.5), "w_min": (0.0, 0.95)}
#: the extended kernel may also be a Sersic deposit; then `c_e` is its INDEX
EXTENDED_SHAPE_BOUNDS = {"gompertz_log": (0.2, 8.0), "sersic": (0.5, 6.0)}
TRUNC_C = 3.0
COMPACT_FAMILY, EXTENDED_FAMILY = "sersic", "gompertz_log"
#: quadrature for the FIT (checked against the full nodes in stage2_fit.py)
FIT_NODES = dict(n_early=6, n_node=16)
FULL_NODES = dict(n_early=8, n_node=24)


@dataclass(frozen=True)
class Spec2:
    theta_names: tuple = THETA_NAMES
    trunc_C: float = TRUNC_C
    extended_family: str = EXTENDED_FAMILY
    compact_in_kpc: bool = False

    @staticmethod
    def with_levers(levers=(), extended_family=EXTENDED_FAMILY, compact_in_kpc=False):
        """The twelve parameters plus the named levers (subset of LEVER_NAMES)."""
        bad = [l for l in levers if l not in LEVER_NAMES]
        if bad:
            raise ValueError(f"unknown levers {bad}; choose from {LEVER_NAMES}")
        return Spec2(theta_names=THETA_NAMES + tuple(levers), extended_family=extended_family,
                     compact_in_kpc=compact_in_kpc)

    @property
    def levers(self):
        return tuple(n for n in self.theta_names if n in LEVER_NAMES)

    @staticmethod
    def with_delay():
        return Spec2(theta_names=THETA_NAMES_DELAY)

    @staticmethod
    def with_growth_split():
        return Spec2(theta_names=THETA_NAMES_GROWTH)

    @staticmethod
    def with_growth_rel():
        return Spec2(theta_names=THETA_NAMES_GROWTH_REL)

    @property
    def growth_rel(self):
        return "g_rel" in self.theta_names

    @property
    def delay(self):
        return "tau_d" in self.theta_names

    @property
    def growth_split(self):
        return "g_split" in self.theta_names or "g_rel" in self.theta_names

    @property
    def n_theta(self):
        return len(self.theta_names)

    def bounds(self):
        out = []
        for n in self.theta_names:
            if n == "c_e":
                out.append(EXTENDED_SHAPE_BOUNDS[self.extended_family])
            elif n == "log_f_c" and self.compact_in_kpc:
                out.append(COMPACT_KPC_BOUNDS)
            else:
                out.append(BOUNDS[n])
        return out

    def unpack(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (self.n_theta,):
            raise ValueError(f"Spec2 wants {self.n_theta} parameters, got {theta.shape}")
        return dict(zip(self.theta_names, theta))

    def index(self, name):
        return self.theta_names.index(name)


def with_levers_theta(theta12, spec2):
    """Append the spec's levers at their nesting value (zero)."""
    return np.r_[np.asarray(theta12, float), np.zeros(len(spec2.levers))]


def nested_theta(theta_incumbent, delay=False, growth=False):
    """The incumbent (a0, a_M, a_z, a_Mz, log_f0, b, c) embedded: compact
    share exactly zero (expit(-1000) underflows to 0.0; -10 leaves 3e-10),
    extended channel = the incumbent's size law and shape."""
    a0, aM, az, aMz, lf0, b, c = np.asarray(theta_incumbent, float)
    th = np.array([a0, aM, az, aMz, -1000.0, 1.0, np.log10(0.04), 0.0, lf0, b, 1.0, c])
    return np.r_[th, 0.0] if (delay or growth) else th


def default_theta(theta_incumbent, delay=False, growth=False):
    """Stage 1's starting values: a compact channel at 0.04 R200c with n = 1,
    the extended one at 0.2 R200c, the split at 10^13.2."""
    a0, aM, az, aMz, lf0, b, c = np.asarray(theta_incumbent, float)
    th = np.array([a0, aM, az, aMz, 13.2, 0.5, np.log10(0.04), 0.0,
                   np.log10(0.2), -0.5, 1.0, c])
    if growth:
        return np.r_[th, -0.5]
    return np.r_[th, 0.5] if delay else th


def clip_to_bounds(spec2, theta):
    lo, hi = np.array(spec2.bounds()).T
    return np.clip(np.asarray(theta, float), lo, hi)


def compact_share(theta_dict, logmh, alpha=None, lt_nodes=None):
    """The compact share of a deposit; `alpha` = dlnM/dlnt at the deposit,
    used only when the spec carries `g_split`. `w_min` (a lever) floors it."""
    x = theta_dict["m_half"] - np.asarray(logmh, float)
    if "g_split" in theta_dict and alpha is not None:
        x = x + theta_dict["g_split"] * (np.asarray(alpha, float) - 1.0)
    if "g_rel" in theta_dict and alpha is not None:
        x = x + theta_dict["g_rel"] * (np.asarray(alpha, float) - alpha_ref(lt_nodes))
    w = expit(x / theta_dict["d_split"])
    w_min = theta_dict.get("w_min", 0.0)
    return w_min + (1.0 - w_min) * w


def hubble_gyr(z):
    """H(z) in 1/Gyr for the TNG cosmology."""
    return E.H0_GYR * np.sqrt(E.OMEGA_M * (1.0 + np.asarray(z, float)) ** 3 + E.OMEGA_L)


def arrival_include(spec2, p, lt, include):
    """(5, N) masks of the extended deposits that have ARRIVED by each epoch:
    the node's accretion time plus tau_d Hubble times at accretion is at or
    before the anchor. With tau_d = 0 this is `include` exactly."""
    if not spec2.delay or p["tau_d"] <= 0.0:
        return include
    t = 10.0 ** lt
    t_arr = t + p["tau_d"] / hubble_gyr(E.z_of_t(t))
    return np.stack([include[k] & (t_arr <= E.T_ANCHOR[k]) for k in range(len(include))])


def _deposits2(spec2, theta, curves, lt, mode="analytic"):
    """(dm_c, dm_e, s_c, s_e, r_trunc) as (n, N) arrays at the nodes."""
    p = spec2.unpack(theta)
    n, N = len(curves), len(lt)
    lm = np.empty((n, N)); dm = np.empty((n, N)); r200 = np.empty((n, N))
    for i, hc in enumerate(curves):
        lm[i] = E.log_mah(lt, hc)
        dm[i] = E.dm_dlnt(lt, hc)
        r200[i] = E.r200c_of(hc, lt, mode)
    z = np.broadcast_to(E.z_of_t(10.0 ** lt), (n, N))
    eff = (p["a0"], p["a_M"], p["a_z"], p["a_Mz"])
    dmstar = 10.0 ** np.clip(E.log_eps_e2(eff, lm, z), -30, 10) * dm
    wc = compact_share(p, lm, alpha=dm / 10.0 ** lm, lt_nodes=lt)
    lz = np.log10(1.0 + z)
    dlm = lm - LEVER_PIVOT_LOGM
    s_c = 10.0 ** np.clip(p["log_f_c"] + p["b_c"] * lz + p.get("g_c", 0.0) * dlm, -8, 2) * (
        1.0 if spec2.compact_in_kpc else r200)
    s_e = 10.0 ** np.clip(p["log_f_e"] + p["b_e"] * lz + p.get("g_e", 0.0) * dlm, -8, 2) * r200
    return dmstar * wc, dmstar * (1.0 - wc), s_c, s_e, spec2.trunc_C * r200


def predict2(spec2, theta, curves, R, epochs=(0, 1, 2, 3, 4), truncate=True,
             mode="analytic", nodes=FULL_NODES, block=300):
    """M*(<R) [Msun] at the requested epochs: (n, len(epochs), len(R))."""
    p = spec2.unpack(theta)
    R = np.asarray(R, float)
    lt, w, include, _ = E.nodes(**nodes)
    epochs = list(epochs)
    out = np.empty((len(curves), len(epochs), len(R)))
    for lo in range(0, len(curves), block):
        cv = curves[lo:lo + block]
        dm_c, dm_e, s_c, s_e, r_tr = _deposits2(spec2, theta, cv, lt, mode)
        n, N = dm_c.shape
        if truncate:
            Bc = M.cog_truncated(COMPACT_FAMILY, (p["n_c"],), s_c.ravel(), r_tr.ravel(), R)
            Be = M.cog_truncated(spec2.extended_family, (p["c_e"],), s_e.ravel(), r_tr.ravel(), R)
        else:
            Bc = families.cog_unit(COMPACT_FAMILY, s_c.ravel(), (p["n_c"],), R)
            Be = families.cog_unit(spec2.extended_family, s_e.ravel(), (p["c_e"],), R)
        Bc = Bc.reshape(len(R), n, N); Be = Be.reshape(len(R), n, N)
        wc_ = dm_c * w[None, :]; we_ = dm_e * w[None, :]
        inc_e = arrival_include(spec2, p, lt, include)
        for j, k in enumerate(epochs):
            out[lo:lo + n, j] = (np.einsum("rgn,gn->gr", Bc, wc_ * include[k][None, :])
                                 + np.einsum("rgn,gn->gr", Be, we_ * inc_e[k][None, :]))
    return out


def channel_masses(spec2, theta, curves, nodes=FULL_NODES):
    """Stellar mass deposited by z=0.4 by the compact and extended channels: (n, 2)."""
    lt, w, include, _ = E.nodes(**nodes)
    p = spec2.unpack(theta)
    dm_c, dm_e, _, _, _ = _deposits2(spec2, theta, curves, lt)
    inc_e = arrival_include(spec2, p, lt, include)
    return np.column_stack([(dm_c * include[0][None, :] * w[None, :]).sum(1),
                            (dm_e * inc_e[0][None, :] * w[None, :]).sum(1)])


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import halo as H
    import stage33 as S33
    import stage37_size_epoch as S37
    import fit as F

    R = F.R_GRID
    recs = H.build_records(rows=np.arange(0, 2397, 40), verbose=False)
    curves = E.build_curves(recs)
    spec_inc = S33.spec_from_label("gompertz_log-E2-S2")
    th_inc = S37.incumbent()
    s2 = Spec2()

    # (1) nesting: the embedded incumbent reproduces engine.predict bit for bit
    ref = E.predict(spec_inc, th_inc, curves, R)
    got = predict2(s2, nested_theta(th_inc), curves, R)
    rel = float(np.max(np.abs(got / ref - 1.0)))
    print(f"  (1) nested theta vs engine.predict, 5 epochs x {len(curves)} galaxies: "
          f"max relative difference {rel:.1e}")
    assert rel < 1e-12

    # (2) the default theta is a different model
    d = predict2(s2, default_theta(th_inc), curves, R)
    moved = float(np.max(np.abs(np.log10(d / ref))))
    print(f"  (2) default theta moves the profile by up to {moved:.3f} dex")
    assert moved > 0.02

    # (3) mass conservation to the truncation boundary; (4) monotone in R and t
    th = default_theta(th_inc)
    big = predict2(s2, th, curves[:5], np.array([1e9]))[:, 0, 0]
    cm = channel_masses(s2, th, curves[:5]).sum(1)
    cons = float(np.max(np.abs(big / cm - 1.0)))
    print(f"  (3) M*(<1e9 kpc) vs the two channels' deposited mass: max |ratio - 1| = {cons:.1e}")
    assert cons < 1e-6
    assert np.all(np.diff(d, axis=2) >= -1e-9) and np.all(np.diff(d, axis=1) <= 1e-9)
    print("  (4) monotone in radius and in time")

    # (5) the fit nodes agree with the full nodes
    df = predict2(s2, th, curves, R, nodes=FIT_NODES)
    dn = float(np.max(np.abs(np.log10(df / d))))
    print(f"  (5) fit nodes (6x16) vs full nodes (8x24): max |dlog| = {dn:.1e} dex")
    assert dn < 1e-3

    # (6) the compact share behaves: falls with mass, 0..1
    p = s2.unpack(th)
    wc = compact_share(p, np.array([11.0, 13.2, 15.0]))
    print(f"  (6) compact share at logMh 11 / 13.2 / 15: {wc.round(3)}")
    assert wc[0] > 0.9 and abs(wc[1] - 0.5) < 1e-9 and wc[2] < 0.1

    # (7) the delay nests exactly at tau_d = 0 and moves mass at tau_d > 0
    sd = Spec2.with_delay()
    thd = np.r_[th, 0.0]
    assert np.array_equal(predict2(sd, thd, curves, R), d), "tau_d = 0 does not nest"
    thd[-1] = 0.5
    dd_ = predict2(sd, thd, curves, R)
    print(f"  (7) delay: tau_d = 0 nests bit for bit; tau_d = 0.5 Hubble times lowers M*(<100) at "
          f"z=0.4 by {np.median(1 - dd_[:, 0, F.I100] / d[:, 0, F.I100]) * 100:.1f}% and at z=2 by "
          f"{np.median(1 - dd_[:, 4, F.I100] / d[:, 4, F.I100]) * 100:.1f}% (median); still monotone")
    assert np.all(np.diff(dd_, axis=2) >= -1e-9) and np.all(np.diff(dd_, axis=1) <= 1e-9)
    assert np.all(dd_ <= d + 1e-9)
    # (8) the growth split nests exactly at g_split = 0 and moves the split at g_split < 0
    sg = Spec2.with_growth_split()
    thg = np.r_[th, 0.0]
    assert np.array_equal(predict2(sg, thg, curves, R), d), "g_split = 0 does not nest"
    thg[-1] = -0.5
    cg = channel_masses(sg, thg, curves); c0 = channel_masses(s2, th, curves)
    print(f"  (8) growth split: g_split = 0 nests bit for bit; g_split = -0.5 moves the median compact "
          f"share from {np.median(c0[:, 0] / c0.sum(1)):.3f} to {np.median(cg[:, 0] / cg.sum(1)):.3f}")
    # (9) the levers nest at zero and move what they say they move
    sl = Spec2.with_levers(LEVER_NAMES)
    thl = with_levers_theta(th, sl)
    assert np.array_equal(predict2(sl, thl, curves, R), d), "levers at zero do not nest"
    thl_c = thl.copy(); thl_c[sl.index("g_c")] = -1.0 / 3.0
    dl = predict2(sl, thl_c, curves, R)
    lmh0 = np.array([E.log_mah(np.log10(E.T_ANCHOR[0]), hc) for hc in curves])
    hi_, lo_ = lmh0 > np.median(lmh0), lmh0 <= np.median(lmh0)
    r_hi = np.median(np.log10(dl[hi_, 0, 0] / d[hi_, 0, 0])); r_lo = np.median(np.log10(dl[lo_, 0, 0] / d[lo_, 0, 0]))
    print(f"  (9) levers nest at zero; g_c = -1/3 moves M*(<2 kpc) at z=0.4 by {r_hi:+.3f} dex (heavier half) "
          f"and {r_lo:+.3f} dex (lighter half)")
    assert r_hi > r_lo
    thl_w = thl.copy(); thl_w[sl.index("w_min")] = 0.5
    pw = sl.unpack(thl_w)
    assert compact_share(pw, np.array([15.0])) >= 0.5 - 1e-9
    sk = Spec2.with_levers((), compact_in_kpc=True)
    thk = th.copy(); thk[sk.index("log_f_c")] = np.log10(4.0)
    dk = predict2(sk, thk, curves, R)
    assert np.all(np.isfinite(dk)) and np.all(np.diff(dk, axis=2) >= -1e-9)
    assert np.array_equal(predict2(sk, nested_theta(th_inc), curves, R), ref), "kpc compact: nested broken"
    print("  (9) compact_in_kpc: the nested incumbent is unchanged; a 4 kpc compact channel evaluates and is monotone")
    ss = Spec2.with_levers((), extended_family="sersic")
    ths = th.copy(); ths[ss.index("c_e")] = 1.0
    ds = predict2(ss, ths, curves[:5], R)
    assert np.all(np.isfinite(ds)) and np.all(np.diff(ds, axis=2) >= -1e-9)
    print("  (9) w_min floors the share; a Sersic extended kernel evaluates and is monotone")
    import time
    t0 = time.time(); predict2(s2, th, curves, R, epochs=(0,), nodes=FIT_NODES); dt = time.time() - t0
    print(f"  timing: z=0.4 only, fit nodes, {len(curves)} galaxies: {dt:.2f} s "
          f"({1e3 * dt / len(curves):.2f} ms per galaxy)")
    print("\nmodel2.py self-check OK")
