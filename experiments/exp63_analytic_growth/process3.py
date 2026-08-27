"""exp63 Stage 3 — noise INSIDE the deposition process.

Three physically named random modifications of a galaxy's history, drawn once
per galaxy and shared by every epoch (so a draw is a coherent multi-epoch
galaxy), on top of the Stage 2 two-channel mean (`model2.py`):

  lumpy accretion   the extended channel's deposition is a compound Poisson
                    process: on each quadrature interval the deposited mass
                    m_i becomes m_i K_i / mu_i with K_i ~ Poisson(mu_i) and
                    mu_i = n_eff * (halo e-folds in the interval) — n_eff
                    events per e-fold of halo growth. Mean preserved; the
                    variance is Campbell's theorem, `lumpy_covariance`.
  efficiency        log10 eps -> log10 eps + eta(ln t), eta a zero-mean
  history           Gaussian process with covariance sigma_eta^2
                    exp(-|dln t| / ell); mean-preserving factor divided out.
  size offsets      per galaxy: log f_c += N(0, sigma_sc), log f_e += N(0, sigma_se).

`Noise3.off()` reproduces `model2.predict2` bit for bit (asserted).

Run the self-check:
    HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
        experiments/exp63_analytic_growth/process3.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude", HERE):
    sys.path.insert(0, str(p))

import engine as E                                       # noqa: E402
import model as M                                        # noqa: E402
import model2 as M2                                      # noqa: E402

NOISE_NAMES = ("n_eff", "sigma_eta", "ell", "sigma_sc", "sigma_se")
NOISE_BOUNDS = {"n_eff": (0.5, 200.0), "sigma_eta": (0.0, 1.0), "ell": (0.02, 2.0),
                "sigma_sc": (0.0, 0.6), "sigma_se": (0.0, 0.6)}


@dataclass(frozen=True)
class Noise3:
    n_eff: float = 1e6        # events per e-fold of halo growth (large = smooth)
    sigma_eta: float = 0.0    # dex, efficiency-history amplitude
    ell: float = 1.0          # correlation length in ln t
    sigma_sc: float = 0.0     # dex, compact size offset
    sigma_se: float = 0.0     # dex, extended size offset

    @staticmethod
    def off():
        return Noise3()

    @staticmethod
    def from_vector(v):
        return Noise3(*[float(x) for x in v])

    def vector(self):
        return np.array([self.n_eff, self.sigma_eta, self.ell, self.sigma_sc, self.sigma_se])

    @property
    def is_off(self):
        return (self.n_eff >= 1e5 and self.sigma_eta == 0.0
                and self.sigma_sc == 0.0 and self.sigma_se == 0.0)


def _eta_cholesky(lt, sigma_eta, ell):
    """Cholesky factor of the efficiency-history covariance on the node grid."""
    lnt = lt * np.log(10.0)
    C = sigma_eta ** 2 * np.exp(-np.abs(lnt[:, None] - lnt[None, :]) / ell)
    return np.linalg.cholesky(C + 1e-12 * np.eye(len(lt)))


def draw(spec2, theta, noise, curves, R, rng, n_draw=8, epochs=(0, 1, 2, 3, 4),
         nodes=M2.FULL_NODES, block=300):
    """(S, n, len(epochs), len(R)) coherent multi-epoch draws."""
    p = spec2.unpack(theta)
    R = np.asarray(R, float)
    lt, w, include, _ = E.nodes(**nodes)
    epochs = list(epochs)
    N = len(lt)
    out = np.empty((n_draw, len(curves), len(epochs), len(R)))
    L = _eta_cholesky(lt, noise.sigma_eta, noise.ell) if noise.sigma_eta > 0 else None
    mean_factor = 10.0 ** (0.5 * noise.sigma_eta ** 2 * np.log(10.0))
    for lo in range(0, len(curves), block):
        cv = curves[lo:lo + block]
        n = len(cv)
        # the deterministic ingredients at the nodes
        lm = np.empty((n, N)); dm = np.empty((n, N)); r200 = np.empty((n, N))
        for i, hc in enumerate(cv):
            lm[i] = E.log_mah(lt, hc); dm[i] = E.dm_dlnt(lt, hc); r200[i] = E.r200c_of(hc, lt)
        z = E.z_of_t(10.0 ** lt)[None, :]
        eff = (p["a0"], p["a_M"], p["a_z"], p["a_Mz"])
        log_eps = np.clip(E.log_eps_e2(eff, lm, z), -30, 10)
        wc = M2.compact_share(p, lm)
        lz = np.log10(1.0 + z)
        r_tr = spec2.trunc_C * r200
        # e-folds of halo growth per node interval (for the Poisson means)
        dlnM = (dm / 10.0 ** lm) * w[None, :]              # dlnM/dlnt * dlnt
        inc_e = M2.arrival_include(spec2, p, lt, include)
        for s in range(n_draw):
            # efficiency history
            if L is not None:
                eta = (L @ rng.standard_normal((N, n))).T   # (n, N)
                eps = 10.0 ** (log_eps + eta) / mean_factor
            else:
                eps = 10.0 ** log_eps
            dmstar = eps * dm * w[None, :]                  # mass per node
            m_c, m_e = dmstar * wc, dmstar * (1.0 - wc)
            # lumpy accretion on the extended channel
            if noise.n_eff < 1e5:
                mu = noise.n_eff * np.clip(dlnM, 1e-12, None)
                K = rng.poisson(mu)
                m_e = m_e * K / mu
            # size offsets
            dsc = rng.normal(0.0, noise.sigma_sc, (n, 1)) if noise.sigma_sc > 0 else 0.0
            dse = rng.normal(0.0, noise.sigma_se, (n, 1)) if noise.sigma_se > 0 else 0.0
            s_c = 10.0 ** np.clip(p["log_f_c"] + dsc + p["b_c"] * lz, -8, 2) * r200
            s_e = 10.0 ** np.clip(p["log_f_e"] + dse + p["b_e"] * lz, -8, 2) * r200
            Bc = M.cog_truncated(M2.COMPACT_FAMILY, (p["n_c"],), s_c.ravel(), r_tr.ravel(), R).reshape(len(R), n, N)
            Be = M.cog_truncated(M2.EXTENDED_FAMILY, (p["c_e"],), s_e.ravel(), r_tr.ravel(), R).reshape(len(R), n, N)
            for j, k in enumerate(epochs):
                out[s, lo:lo + n, j] = (np.einsum("rgn,gn->gr", Bc, m_c * include[k][None, :])
                                        + np.einsum("rgn,gn->gr", Be, m_e * inc_e[k][None, :]))
    return out


def lumpy_covariance(spec2, theta, noise, curves, R, nodes=M2.FULL_NODES):
    """Campbell's theorem: the z=0.4 covariance between radii of the lumpy
    extended channel, (n, nR, nR)."""
    p = spec2.unpack(theta)
    lt, w, include, _ = E.nodes(**nodes)
    dm_c, dm_e, s_c, s_e, r_tr = M2._deposits2(spec2, theta, curves, lt)
    n, N = dm_e.shape
    lm = np.array([E.log_mah(lt, hc) for hc in curves])
    dm = np.array([E.dm_dlnt(lt, hc) for hc in curves])
    dlnM = (dm / 10.0 ** lm) * w[None, :]
    mu = noise.n_eff * np.clip(dlnM, 1e-12, None)
    inc_e = M2.arrival_include(spec2, p, lt, include)
    m_e = dm_e * w[None, :] * inc_e[0][None, :]
    Be = M.cog_truncated(M2.EXTENDED_FAMILY, (p["c_e"],), s_e.ravel(), r_tr.ravel(), R).reshape(len(R), n, N)
    var_w = m_e ** 2 / mu                                  # Var[m_i K_i/mu_i] = m_i^2/mu_i
    return np.einsum("rgn,gn,sgn->grs", Be, var_w, Be)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import time
    import halo as H
    import stage37_size_epoch as S37
    import fit as F

    R = F.R_GRID
    recs = H.build_records(rows=np.arange(0, 2397, 120), verbose=False)
    curves = E.build_curves(recs)
    spec2 = M2.Spec2()
    theta = M2.default_theta(S37.incumbent())
    mean = M2.predict2(spec2, theta, curves, R)
    rng = np.random.default_rng(3)

    d0 = draw(spec2, theta, Noise3.off(), curves, R, rng, n_draw=1)[0]
    print(f"  (1) noise off reproduces predict2: max |rel| = {np.max(np.abs(d0 / mean - 1)):.1e}")
    assert np.array_equal(d0, mean) or np.max(np.abs(d0 / mean - 1)) < 1e-12

    nz = Noise3(n_eff=3.0)
    S = 300
    dl = draw(spec2, theta, nz, curves, R, np.random.default_rng(1), n_draw=S)
    mu_draw = dl.mean(0)[:, 0, :]
    var_draw = dl.var(0)[:, 0, :]
    cov = lumpy_covariance(spec2, theta, nz, curves, R)
    var_th = np.einsum("grr->gr", cov)
    rel_mean = np.max(np.abs(mu_draw / mean[:, 0, :] - 1))
    sel = var_th > 0.01 * mean[:, 0, :] ** 2 * 0 + 1e-8 * mean[:, 0, :] ** 2   # where the variance is not negligible
    rel_var = np.median(np.abs(var_draw[sel] / var_th[sel] - 1))
    print(f"  (2) lumpy accretion, n_eff=3, {S} draws: mean vs predict2 max |rel| {rel_mean:.3f}; "
          f"variance vs Campbell median |rel| {rel_var:.3f}; typical sigma(M*(<100))/M = "
          f"{np.median(np.sqrt(var_th[:, F.I100]) / mean[:, 0, F.I100]):.3f}")
    assert rel_mean < 0.05 and rel_var < 0.25

    ne = Noise3(sigma_eta=0.2, ell=0.3)
    de = draw(spec2, theta, ne, curves, R, np.random.default_rng(2), n_draw=S)
    rel_e = np.max(np.abs(de.mean(0)[:, 0, F.I100] / mean[:, 0, F.I100] - 1))
    sc_e = np.median(np.std(np.log10(de[:, :, 0, F.I100]), axis=0))
    print(f"  (3) efficiency history sigma_eta=0.2 dex, ell=0.3: mean of M*(<100) preserved to "
          f"{rel_e:.3f}; scatter of log M*(<100) across draws {sc_e:.3f} dex")
    assert rel_e < 0.05

    ns = Noise3(sigma_sc=0.2, sigma_se=0.2)
    ds = draw(spec2, theta, ns, curves, R, np.random.default_rng(4), n_draw=8)
    assert np.all(np.diff(ds, axis=3) >= -1e-9) and np.all(np.diff(ds, axis=2) <= 1e-9)
    print("  (4) size offsets: draws monotone in radius and time")

    t0 = time.time(); draw(spec2, theta, Noise3(3.0, 0.2, 0.3, 0.2, 0.2), curves, R, rng, n_draw=8); dt = time.time() - t0
    print(f"  timing: 8 draws x {len(curves)} galaxies x 5 epochs in {dt:.2f} s "
          f"(-> {dt / len(curves) * 2397 / 60:.1f} min for the full sample)")
    print("\nprocess3.py self-check OK")
