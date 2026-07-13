"""The multi-epoch emulator: continuous-z profiles with an AR(1) epoch latent.

Graduated from exp37 (the exp33-vi blueprint; every ingredient measured there,
n=2395): five per-epoch heteroscedastic cores (``hongshao.emulator``) on the
shared portable features, coefficients interpolated quadratically in z
(held-epoch closure within +-4% / <=0.9 max|rel| points), and a per-galaxy
AR(1)-in-epoch latent for generative sampling (the cross-epoch residual
correlation is Markovian, rho ~ 0.62 measured with poly2 cores).

Two graduated profile products (user decision 2026-07-14: both available):

* ``fit_block_profile`` — the DEFAULT. Predicts the log masses of the kpc
  BLOCKS directly (2-10/10-30/30-50/50-100/100-R_max, snapped to grid radii),
  plus the central fraction and pooled density-shape scores that distribute
  mass WITHIN blocks. Every sampled CoG is monotone by construction, drawn
  block masses reproduce near-empty high-z annuli (draw planes 0.4-0.9x the
  sampling floor through z=1), and the mean path allocates the total-vs-parts
  gap by each block's predictive uncertainty.
* ``fit_shared_profile`` — the log-CoG product (one pooled PCA shape basis
  over all epochs, K=3). Best inner-region mean accuracy at z=0.4 (+1.8% vs
  block +3.5% inside 10 kpc); its per-radius draws are NOT guaranteed
  monotone.

``fit_multi_epoch`` dispatches between them (``mode="block"`` default).
Numbers above are exp37's records (its README holds the full evidence table);
the self-check here (``python -m hongshao.multi_epoch``) pins the
representation-independent contracts: AR(1) sampler exactness, snapshot/held-z
coefficient interpolation, the exact block round-trip, draw monotonicity, and
the uncertainty-weighted deficit allocation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .emulator import Emulator, fit as emu_fit, _chol_psd
from .profile_emulator import ProfileEmulator, density_from_cog


# --------------------------------------------------------------------------- #
# the epoch machinery                                                          #
# --------------------------------------------------------------------------- #
def ar1_corr(rho, idx):
    """AR(1) correlation matrix over epoch-index positions ``idx``: rho^|di-dj|.

    ``idx`` is in epoch-INDEX units (the exp33-vi measurement: residual
    correlation decays per snapshot STEP, not per dz), so arbitrary redshifts
    map in via ``MultiEpochEmulator.epoch_index``.
    """
    idx = np.asarray(idx, float)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _sample_eps(rho, corrs, n_draws, rng, idx=None):
    """Standardized residual draws (n_draws, E, T).

    Construction: chol(AR1(rho)) mixes iid normals ACROSS epochs per target,
    then chol(R_k) mixes WITHIN each epoch across targets. Within-epoch
    correlation is exactly ``corrs[k]``; cross-epoch same-target correlation is
    rho^sep exactly when the R_k agree (A_kj * (L_k L_j^T) in general); PSD by
    construction.
    """
    E = len(corrs)
    T = corrs[0].shape[0]
    La = _chol_psd(ar1_corr(rho, np.arange(E) if idx is None else idx))
    Z = rng.standard_normal((n_draws, E, T))
    eps = np.einsum("kj,njt->nkt", La, Z)
    for k in range(E):
        eps[:, k, :] = eps[:, k, :] @ _chol_psd(corrs[k]).T
    return eps


def fit_rho(C, idx=None):
    """Least-squares AR(1) rho from a cross-epoch correlation matrix:
    log C_ab regressed on |idx_a - idx_b| through the origin."""
    E = len(C)
    idx = np.arange(E) if idx is None else np.asarray(idx, float)
    sep, val = [], []
    for a in range(E):
        for b in range(a + 1, E):
            if C[a, b] > 0:
                sep.append(abs(idx[a] - idx[b]))
                val.append(np.log(C[a, b]))
    sep, val = np.asarray(sep), np.asarray(val)
    return float(np.exp((sep @ val) / (sep @ sep)))


def interp_emulator(emus, z_src, z_tgt):
    """Element-wise quadratic-in-z least-squares interpolation of fitted
    emulator coefficients (beta, gamma, corr), exp33-vi's validated form."""
    def q(vals):
        vals = np.stack(vals)
        flat = vals.reshape(len(z_src), -1)
        out = np.empty(flat.shape[1])
        for j in range(flat.shape[1]):
            out[j] = np.polyval(np.polyfit(z_src, flat[:, j], 2), z_tgt)
        return out.reshape(vals.shape[1:])
    e0 = emus[0]
    return Emulator(e0.mean, e0.mu_x, e0.sd_x, q([e.beta for e in emus]),
                    q([e.gamma for e in emus]), q([e.corr for e in emus]))


@dataclass
class MultiEpochEmulator:
    """Per-epoch heteroscedastic cores, continuous in z by quadratic
    coefficient interpolation, generative via an AR(1)-in-epoch latent."""
    z_grid: np.ndarray        # (E,) fitted snapshot redshifts
    emus: list                # E per-epoch Emulator cores (shared features)
    rho: float                # AR(1) epoch-latent correlation (measured)

    def at_z(self, z):
        """The DIRECT core at a fitted snapshot; elsewhere the element-wise
        quadratic-in-z least-squares interpolation of its coefficients
        (exp33-vi closure: within +-4% of a direct fit)."""
        k = np.where(np.isclose(self.z_grid, z))[0]
        if len(k):
            return self.emus[int(k[0])]
        return interp_emulator(self.emus, self.z_grid, z)

    def predict(self, X, z):
        return self.at_z(z).predict(X)

    def epoch_index(self, z):
        """Map z to epoch-INDEX units (the AR(1) separation coordinate)."""
        return np.interp(z, self.z_grid, np.arange(len(self.z_grid)))

    def sample_epochs(self, X, z=None, size=1, rng=None):
        """(size, N, E, T) draws, AR(1)-coherent across the epochs ``z``
        (default: the fitted snapshot grid)."""
        rng = np.random.default_rng(rng)
        z = self.z_grid if z is None else np.asarray(z, float)
        cores = [self.at_z(zz) for zz in z]
        mu = np.stack([c.predict(X)[0] for c in cores], axis=1)   # (N, E, T)
        sig = np.stack([c.predict(X)[1] for c in cores], axis=1)
        n = len(mu)
        eps = _sample_eps(self.rho, [c.corr for c in cores], size * n, rng,
                          idx=self.epoch_index(z))
        return mu[None] + sig[None] * eps.reshape(size, n, len(z), -1)


# --------------------------------------------------------------------------- #
# product 1 (default): the block-pinned profile emulator                       #
# --------------------------------------------------------------------------- #
@dataclass
class MultiEpochBlock:
    """The block-pinned profile product (exp37 option b, the DEFAULT).

    The emulator predicts the log masses of the kpc BLOCKS directly — the
    representation the aperture emulator fits at <=1.4% bias — and the density
    shape only distributes each block's mass across its own shells. Compressed
    vector per epoch: [anchor, cfrac, log block-fraction masses (one per
    block), K density-shape scores]. Block masses are positive by construction
    and shells within a block are positive, so every reconstructed CoG is
    monotone; radial detail is unchanged (the full shell grid, shape modes
    acting within blocks).
    """
    me: MultiEpochEmulator
    mean_shape: np.ndarray    # (R-1,) pooled amplitude-removed mean log density
    modes: np.ndarray         # (K, R-1) shared PCA density-shape modes
    radii: np.ndarray         # (R,) kpc — the CoG EDGE grid
    block_idx: np.ndarray     # (nb+1,) indices into radii: block boundaries

    @property
    def _dA(self):
        r = self.radii
        return np.pi * (r[1:] ** 2 - r[:-1] ** 2)

    def compress(self, cogs_log):
        """(..., R) log CoGs -> (..., 2+nb+K) [anchor, cfrac, bfracs, scores]."""
        cogs_log = np.asarray(cogs_log, float)
        lead = cogs_log.shape[:-1]
        anchor = cogs_log[..., -1]
        cfrac = cogs_log[..., 0] - anchor
        cum = 10.0 ** cogs_log
        # a near-empty block can underflow to a zero linear difference; floor
        # at 1 Msun (the _cum_to_bins convention) — negligible mass, keeps the
        # target finite and the round-trip within float tolerance
        bfracs = [np.log10(np.clip(cum[..., b] - cum[..., a], 1.0, None)) - anchor
                  for a, b in zip(self.block_idx[:-1], self.block_idx[1:])]
        log_sig, _ = density_from_cog(cogs_log.reshape(-1, cogs_log.shape[-1]),
                                      self.radii)
        shape = log_sig.reshape(*lead, -1) - anchor[..., None]
        scores = (shape - self.mean_shape) @ self.modes.T
        return np.concatenate([anchor[..., None], cfrac[..., None],
                               np.stack(bfracs, axis=-1), scores], axis=-1)

    def reconstruct(self, compressed, sigma=None):
        """(..., 2+nb+K) -> (..., R) log CoGs, monotone by construction.

        The sum of the (median-predicted) parts falls short of the predicted
        total by the lognormal median-vs-mean gap. Without ``sigma`` the
        deficit is spread uniformly (the draw path — each draw is a
        realization, the rescale just enforces its own total). With ``sigma``
        (the per-target predictive dex scatter, the MEAN path) the deficit is
        allocated in proportion to each block's expected gap,
        B_j (e^(sigma_ln^2 / 2) - 1): tight blocks stay at their direct
        predictions, the uncertainty-dominated blocks absorb the missing mass.
        """
        compressed = np.asarray(compressed, float)
        nb = len(self.block_idx) - 1
        anchor = compressed[..., 0:1]
        cfrac = compressed[..., 1:2]
        bfracs = compressed[..., 2:2 + nb]
        scores = compressed[..., 2 + nb:]
        blocks = 10.0 ** (anchor + bfracs)                       # (..., nb)
        cen = 10.0 ** (anchor + cfrac)
        total = 10.0 ** anchor
        w = 10.0 ** (anchor + self.mean_shape + scores @ self.modes) * self._dA
        shells = np.empty_like(w)
        for j, (a, b) in enumerate(zip(self.block_idx[:-1], self.block_idx[1:])):
            wj = w[..., a:b]
            shells[..., a:b] = (wj / wj.sum(axis=-1, keepdims=True)
                                * blocks[..., j:j + 1])
        cum = np.concatenate([cen, cen + np.cumsum(shells, axis=-1)], axis=-1)
        if sigma is None:
            return np.log10(cum * total / cum[..., -1:])
        sig_ln = np.log(10.0) * np.asarray(sigma, float)[..., 2:2 + nb]
        gap_w = blocks * (np.exp(0.5 * sig_ln ** 2) - 1.0) + 1e-30
        deficit = total[..., 0] - cum[..., -1]
        alloc = gap_w / gap_w.sum(-1, keepdims=True) * deficit[..., None]
        factor = np.clip((blocks + alloc) / blocks, 1e-6, None)  # keep positive
        for j, (a, b) in enumerate(zip(self.block_idx[:-1], self.block_idx[1:])):
            shells[..., a:b] = shells[..., a:b] * factor[..., j:j + 1]
        cum = np.concatenate([cen, cen + np.cumsum(shells, axis=-1)], axis=-1)
        # positivity clamps can leave a residual; close it with the uniform
        # rescale (tiny once the weighted allocation has done the work)
        return np.log10(cum * total / cum[..., -1:])

    def predict_cog(self, X, z):
        """(n, R) mean log CoG at redshift ``z`` (uncertainty-weighted)."""
        mu, sig, _ = self.me.predict(X, z)
        return self.reconstruct(mu, sigma=sig)

    def sample_cogs(self, X, z=None, size=1, rng=None):
        """(size, n, E, R) AR(1)-coherent log CoG draws, every one monotone."""
        return self.reconstruct(self.me.sample_epochs(X, z=z, size=size, rng=rng))


def fit_block_profile(X, cogs_log, radii, z_grid, n_modes=6, rho=0.0,
                      mean="poly2", block_kpc=(10.0, 30.0, 50.0, 100.0)):
    """Fit the block-pinned profile product (see ``MultiEpochBlock``).

    ``cogs_log`` (n, E, R) log CoGs on the ``radii`` EDGE grid. Block
    boundaries snap to the nearest grid radii; the outermost block ends at the
    grid edge and the innermost starts at the central disk boundary.
    """
    cogs_log = np.asarray(cogs_log, float)
    n, E, R_ = cogs_log.shape
    radii = np.asarray(radii, float)
    idx = [0] + [int(np.argmin(np.abs(radii - e))) for e in block_kpc] + [R_ - 1]
    block_idx = np.array(sorted(set(idx)))
    if len(block_idx) != len(idx):
        raise ValueError(f"block edges collide on this grid: {idx}")
    anchor = cogs_log[..., -1]
    log_sig, _ = density_from_cog(cogs_log.reshape(-1, R_), radii)
    shape = log_sig.reshape(n, E, R_ - 1) - anchor[..., None]
    mean_shape = shape.mean(axis=(0, 1))
    _, _, Vt = np.linalg.svd((shape - mean_shape).reshape(-1, R_ - 1),
                             full_matrices=False)
    modes = Vt[:n_modes]
    mpb = MultiEpochBlock(None, mean_shape, modes, radii, block_idx)
    comp = mpb.compress(cogs_log)                      # (n, E, 2+nb+K)
    emus = [emu_fit(X, comp[:, k], mean=mean) for k in range(E)]
    mpb.me = MultiEpochEmulator(np.asarray(z_grid, float), emus, rho)
    return mpb


# --------------------------------------------------------------------------- #
# product 2: the shared-basis log-CoG profile emulator                         #
# --------------------------------------------------------------------------- #
@dataclass
class MultiEpochProfile:
    """The log-CoG profile product: ONE pooled PCA shape basis (so the
    compressed coefficients interpolate in z) + a MultiEpochEmulator over
    [anchor, shared-PC scores]. Best inner-region mean accuracy at z=0.4;
    per-radius draws are NOT guaranteed monotone (use the block product for
    generative work)."""
    me: MultiEpochEmulator
    mean_shape: np.ndarray    # (R,) pooled amplitude-removed mean shape
    modes: np.ndarray         # (K, R) shared PCA modes
    radii: np.ndarray         # (R,) kpc

    def at_z(self, z):
        return ProfileEmulator(self.me.at_z(z), self.mean_shape, self.modes,
                               self.radii)

    def sample_epochs(self, X, z=None, size=1, rng=None):
        """(size, N, E, R) correlated log-profile draws across epochs."""
        draws = self.me.sample_epochs(X, z=z, size=size, rng=rng)
        A = np.column_stack([np.ones(len(self.radii)), self.modes.T])
        return draws @ A.T + self.mean_shape


def fit_shared_profile(X, profs, anchors, radii, z_grid, n_modes=3, rho=0.0,
                       mean="poly2"):
    """Fit the shared-basis log-CoG profile product.

    ``profs`` (n, E, R) per-epoch log profiles; ``anchors`` (n, E) the
    amplitudes factored out (logMtot per epoch). One PCA basis from the POOLED
    amplitude-removed shapes of every epoch, then a per-epoch core on
    [anchor, shared scores].
    """
    profs = np.asarray(profs, float)
    anchors = np.asarray(anchors, float)
    shape = profs - anchors[..., None]
    mean_shape = shape.mean(axis=(0, 1))
    flat = (shape - mean_shape).reshape(-1, shape.shape[-1])
    _, _, Vt = np.linalg.svd(flat, full_matrices=False)
    modes = Vt[:n_modes]
    emus = [emu_fit(X, np.column_stack([anchors[:, k],
                                        (shape[:, k] - mean_shape) @ modes.T]),
                    mean=mean)
            for k in range(shape.shape[1])]
    me = MultiEpochEmulator(np.asarray(z_grid, float), emus, rho)
    return MultiEpochProfile(me, mean_shape, modes, np.asarray(radii, float))


def fit_multi_epoch(X, cogs_log, radii, z_grid, mode="block", **kwargs):
    """Fit a multi-epoch profile product; ``mode`` = "block" (default) or
    "cog". ``cogs_log`` (n, E, R) log CoGs on the ``radii`` edge grid; the
    cog mode compresses ``cogs_log[..., :-1]`` with the total as anchor."""
    if mode == "block":
        return fit_block_profile(X, cogs_log, radii, z_grid, **kwargs)
    if mode == "cog":
        cogs_log = np.asarray(cogs_log, float)
        return fit_shared_profile(X, cogs_log[..., :-1], cogs_log[..., -1],
                                  np.asarray(radii, float)[:-1], z_grid,
                                  **kwargs)
    raise ValueError(f"mode must be 'block' or 'cog', got {mode!r}")


# --------------------------------------------------------------------------- #
# self-check                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ZK = np.array([0.4, 0.7, 1.0, 1.5, 2.0])

    # (1) AR(1) sampler: within-epoch correlation exactly R, cross-epoch
    # same-target correlation exactly rho^sep
    E, T, rho = 5, 3, 0.6
    R_t = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.4], [0.2, 0.4, 1.0]])
    eps = _sample_eps(rho, [R_t] * E, 200_000, rng)
    for k in range(E):
        assert np.allclose(np.corrcoef(eps[:, k, :].T), R_t, atol=0.02)
    sep = np.abs(np.arange(E)[:, None] - np.arange(E)[None, :])
    for t in range(T):
        assert np.allclose(np.corrcoef(eps[:, :, t].T), rho ** sep, atol=0.02)

    # (2) continuous z: direct cores at snapshots, quadratic coefficient
    # recovery at a held z, AR(1)-coherent draws, rho recovery
    n = 3000
    X = rng.normal(size=(n, 5))
    emus, z_h = [], 0.85
    for z in ZK:
        B = np.array([[10 + z ** 2, 0.3 * z, 0.1, 0.0, 0.05, -0.2 * z ** 2],
                      [9 - 0.5 * z, -0.1 * z, 0.2, 0.1, 0.0, 0.1 * z]])
        Y = (np.column_stack([np.ones(n), X]) @ B.T
             + np.array([0.05, 0.10]) * rng.standard_normal((n, 2)))
        emus.append(emu_fit(X, Y, mean="linear"))
    me = MultiEpochEmulator(ZK, emus, rho=0.6)
    assert me.at_z(ZK[2]) is emus[2]
    B_h = np.array([[10 + z_h ** 2, 0.3 * z_h, 0.1, 0.0, 0.05, -0.2 * z_h ** 2],
                    [9 - 0.5 * z_h, -0.1 * z_h, 0.2, 0.1, 0.0, 0.1 * z_h]])
    rms = float(np.sqrt(((me.predict(X, z_h)[0]
                          - np.column_stack([np.ones(n), X]) @ B_h.T) ** 2).mean()))
    assert rms < 0.02, rms
    draws = me.sample_epochs(X[:400], size=250, rng=1)
    mu_all = np.stack([e.predict(X[:400])[0] for e in emus], axis=1)
    sig_all = np.stack([e.predict(X[:400])[1] for e in emus], axis=1)
    res = (draws - mu_all[None]) / sig_all[None]
    C_d = np.corrcoef(res.transpose(2, 0, 1, 3).reshape(E, -1))
    assert np.allclose(C_d, 0.6 ** sep, atol=0.04)
    assert abs(fit_rho(C_d) - 0.6) < 0.03

    # (3) block product: exact round-trip with the full mode set (the
    # integrate_density identity), block components exact, monotone draws
    ng = 400
    Xd = rng.normal(size=(ng, 5))
    rg = np.geomspace(2.0, 148.0, 16)
    cogs = np.empty((ng, len(ZK), len(rg)))
    for k, z in enumerate(ZK):
        amp = 11.0 + 0.35 * Xd[:, 0] - 0.25 * z + 0.05 * rng.standard_normal(ng)
        size_kpc = np.clip(10.0 * (1.0 + 0.3 * Xd[:, 1]) / (1.0 + z), 2.0, None)
        cogs[:, k] = amp[:, None] + np.log10(
            1.0 - np.exp(-(rg[None, :] / size_kpc[:, None])))
    mpb = fit_block_profile(Xd, cogs, rg, ZK, n_modes=len(rg) - 1,
                            rho=0.5, mean="linear")
    cb = mpb.compress(cogs)
    assert np.abs(mpb.reconstruct(cb) - cogs).max() < 1e-9, "round-trip"
    e0, e1 = mpb.block_idx[0], mpb.block_idx[1]
    true_b0 = np.log10(10.0 ** cogs[..., e1] - 10.0 ** cogs[..., e0])
    assert np.allclose(cb[..., 2] + cogs[..., -1], true_b0, atol=1e-9)
    mpb3 = fit_block_profile(Xd, cogs, rg, ZK, n_modes=3, rho=0.5, mean="linear")
    dcb = mpb3.sample_cogs(Xd[:100], size=30, rng=7)
    assert dcb.shape == (30, 100, E, len(rg))
    assert np.all(np.diff(10.0 ** dcb, axis=-1) > -1e-6), "draws must be monotone"

    # (4) uncertainty-weighted deficit allocation (the mean path): with equal
    # block masses and ONE uncertain block, the total-vs-parts gap must land
    # there, leaving the tight blocks at their direct predictions
    nb = len(mpb3.block_idx) - 1
    comp_t = np.zeros((20, 2 + nb + 3))
    comp_t[:, 0] = 11.0
    comp_t[:, 1] = -2.0
    comp_t[:, 2:2 + nb] = np.log10(0.18)
    sig_t = np.full_like(comp_t, 0.02)
    sig_t[:, 2 + nb - 1] = 1.0
    rec = mpb3.reconstruct(comp_t, sigma=sig_t)
    cum = 10.0 ** rec
    assert np.allclose(rec[:, -1], comp_t[:, 0], atol=1e-9), "total must be exact"
    assert np.all(np.diff(cum, axis=-1) > 0)
    bl = np.stack([np.log10(cum[:, b] - cum[:, a]) - comp_t[:, 0]
                   for a, b in zip(mpb3.block_idx[:-1], mpb3.block_idx[1:])], -1)
    assert np.allclose(bl[:, :-1], np.log10(0.18), atol=0.005), \
        "tight blocks must stay at their direct predictions"
    assert np.all(bl[:, -1] > np.log10(0.18) + 0.15), \
        "the uncertain block must absorb the deficit"

    # (5) the cog product + the dispatcher: block is the default; the cog
    # product recovers a shared-basis profile at snapshot and held z
    mp_cog = fit_multi_epoch(Xd, cogs, rg, ZK, mode="cog", mean="linear")
    assert isinstance(mp_cog, MultiEpochProfile)
    mu_p, _ = mp_cog.at_z(ZK[1]).predict(Xd)
    assert float(np.sqrt(((mu_p - cogs[:, 1, :-1]) ** 2).mean())) < 0.12
    assert isinstance(fit_multi_epoch(Xd, cogs, rg, ZK, mean="linear",
                                      n_modes=3), MultiEpochBlock)

    print("multi_epoch self-check OK: AR(1) sampler exact; held-z coefficient "
          f"recovery rms {rms:.4f}; draws AR(1)-coherent; block round-trip "
          "exact, draws monotone, weighted allocation keeps tight blocks and "
          "the total; dispatcher defaults to the block product")
