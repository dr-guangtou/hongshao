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
BOUNDS = {"a0": (-6.0, 2.0), "a_M": (-3.0, 3.0), "a_z": (-3.0, 3.0), "a_Mz": (-3.0, 3.0),
          "m_half": (10.5, 15.0), "d_split": (0.1, 3.0),
          "log_f_c": (-3.0, -0.5), "b_c": (-3.0, 3.0),
          "log_f_e": (-2.5, 0.0), "b_e": (-3.0, 3.0),
          "n_c": (0.5, 2.5), "c_e": (0.2, 8.0)}
TRUNC_C = 3.0
COMPACT_FAMILY, EXTENDED_FAMILY = "sersic", "gompertz_log"
#: quadrature for the FIT (checked against the full nodes in stage2_fit.py)
FIT_NODES = dict(n_early=6, n_node=16)
FULL_NODES = dict(n_early=8, n_node=24)


@dataclass(frozen=True)
class Spec2:
    theta_names: tuple = THETA_NAMES
    trunc_C: float = TRUNC_C

    @property
    def n_theta(self):
        return len(self.theta_names)

    def bounds(self):
        return [BOUNDS[n] for n in self.theta_names]

    def unpack(self, theta):
        theta = np.asarray(theta, float)
        if theta.shape != (self.n_theta,):
            raise ValueError(f"Spec2 wants {self.n_theta} parameters, got {theta.shape}")
        return dict(zip(self.theta_names, theta))

    def index(self, name):
        return self.theta_names.index(name)


def nested_theta(theta_incumbent):
    """The incumbent (a0, a_M, a_z, a_Mz, log_f0, b, c) embedded: compact
    share exactly zero (expit(-1000) underflows to 0.0; -10 leaves 3e-10),
    extended channel = the incumbent's size law and shape."""
    a0, aM, az, aMz, lf0, b, c = np.asarray(theta_incumbent, float)
    return np.array([a0, aM, az, aMz, -1000.0, 1.0, np.log10(0.04), 0.0, lf0, b, 1.0, c])


def default_theta(theta_incumbent):
    """Stage 1's starting values: a compact channel at 0.04 R200c with n = 1,
    the extended one at 0.2 R200c, the split at 10^13.2."""
    a0, aM, az, aMz, lf0, b, c = np.asarray(theta_incumbent, float)
    return np.array([a0, aM, az, aMz, 13.2, 0.5, np.log10(0.04), 0.0,
                     np.log10(0.2), -0.5, 1.0, c])


def clip_to_bounds(spec2, theta):
    lo, hi = np.array(spec2.bounds()).T
    return np.clip(np.asarray(theta, float), lo, hi)


def compact_share(theta_dict, logmh):
    return expit((theta_dict["m_half"] - np.asarray(logmh, float)) / theta_dict["d_split"])


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
    wc = compact_share(p, lm)
    lz = np.log10(1.0 + z)
    s_c = 10.0 ** np.clip(p["log_f_c"] + p["b_c"] * lz, -8, 2) * r200
    s_e = 10.0 ** np.clip(p["log_f_e"] + p["b_e"] * lz, -8, 2) * r200
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
            Be = M.cog_truncated(EXTENDED_FAMILY, (p["c_e"],), s_e.ravel(), r_tr.ravel(), R)
        else:
            Bc = families.cog_unit(COMPACT_FAMILY, s_c.ravel(), (p["n_c"],), R)
            Be = families.cog_unit(EXTENDED_FAMILY, s_e.ravel(), (p["c_e"],), R)
        Bc = Bc.reshape(len(R), n, N); Be = Be.reshape(len(R), n, N)
        wc_ = dm_c * w[None, :]; we_ = dm_e * w[None, :]
        for j, k in enumerate(epochs):
            inc = include[k][None, :]
            out[lo:lo + n, j] = (np.einsum("rgn,gn->gr", Bc, wc_ * inc)
                                 + np.einsum("rgn,gn->gr", Be, we_ * inc))
    return out


def channel_masses(spec2, theta, curves, nodes=FULL_NODES):
    """Stellar mass deposited by z=0.4 by the compact and extended channels: (n, 2)."""
    lt, w, include, _ = E.nodes(**nodes)
    dm_c, dm_e, _, _, _ = _deposits2(spec2, theta, curves, lt)
    inc = include[0][None, :] * w[None, :]
    return np.column_stack([(dm_c * inc).sum(1), (dm_e * inc).sum(1)])


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

    import time
    t0 = time.time(); predict2(s2, th, curves, R, epochs=(0,), nodes=FIT_NODES); dt = time.time() - t0
    print(f"  timing: z=0.4 only, fit nodes, {len(curves)} galaxies: {dt:.2f} s "
          f"({1e3 * dt / len(curves):.2f} ms per galaxy)")
    print("\nmodel2.py self-check OK")
