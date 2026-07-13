"""The profile/target layer: graduate all four Ultimate-SHMR prediction modes.

Every mode predicts a target vector from the portable features ``DiffMAH + c_200c``
through the *same* heteroscedastic core (``hongshao.emulator.Emulator``). The
modes differ only in how the target is built (and, for profiles, how it is
reconstructed):

  (1) kpc aperture/outskirt masses   -> ``aperture_targets`` + ``Emulator``      (exp20)
  (2) Re-based aperture masses       -> ``re_targets`` + ``Emulator``            (exp21)
  (3) the cumulative curve of growth -> ``ProfileEmulator`` (cumulative)         (exp22)
  (4) the 1-D surface-density profile-> ``ProfileEmulator`` (density)            (exp22b)

(1)/(2) are direct: the masses *are* the prediction. (3)/(4) compress each
profile to ``[anchor=logMtot, PC1..PCK]`` (in-sample PCA, K=3 by default; >3 PCs
do not help the predictor, exp22 ``pca_n_components``), predict that vector, and
reconstruct the per-radius profile linearly — so the predictive Gaussian
propagates analytically to a per-radius mean + sigma, no sampling.

Which mode? Apertures (1)/(2) for directly-observable masses with clean
uncertainties. The **density profile (4)** is the most complete target: it is the
most halo-predictable in shape (esp. the outskirts) and integrates *stably*
(outward from the centre) to the CoG and any aperture — so it yields the others.
For aperture masses alone the four are equivalent (total-mass-dominated).

Self-check (``python -m hongshao.profile_emulator``) reproduces exp20/21/22/22b.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .emulator import Emulator, fit


# --------------------------------------------------------------------------- #
# target builders (the curve of growth -> a modeling target)                  #
# --------------------------------------------------------------------------- #
def aperture_targets(cog, radii, edges_kpc):
    """Mode (1): cumulative + annulus log-masses at fixed kpc ``edges_kpc``.

    ``cog`` (N, R) is the log curve of growth on ``radii`` (kpc). Returns (N, E):
    column 0 is the cumulative ``log M(<edges[0])``; the rest are annuli
    ``log[M(<edge_k) - M(<edge_{k-1})]``.
    """
    radii = np.asarray(radii, float)
    edges = np.asarray(edges_kpc, float)
    cum = np.column_stack([10.0 ** np.array([np.interp(e, radii, c) for c in cog])
                           for e in edges])
    return _cum_to_bins(cum)


def re_targets(cog, radii, re_edges, total_kpc=120.0):
    """Mode (2): aperture/outskirt masses in half-mass-radius (Re) units.

    Re = the radius enclosing half of ``M(<total_kpc)``, read off each galaxy's
    log CoG. Bins are the consecutive ``re_edges`` (multiples of Re). Returns
    ``(Y, Re, mask)``: ``Y`` (N, len(re_edges)) the bin masses, ``Re`` (N,) kpc,
    and a boolean ``mask`` dropping galaxies whose outermost edge leaves the
    measured CoG (apply it: ``Y[mask]``, ``X[mask]``).
    """
    radii = np.asarray(radii, float)
    re_edges = np.asarray(re_edges, float)
    log_total = np.array([np.interp(total_kpc, radii, c) for c in cog])
    re = np.array([np.interp(lt - np.log10(2.0), c, radii) for c, lt in zip(cog, log_total)])
    edge_kpc = re[:, None] * re_edges[None, :]                       # (N, E) kpc
    cum = np.array([[10.0 ** np.interp(e, radii, c) for e in edges]
                    for c, edges in zip(cog, edge_kpc)])
    Y = _cum_to_bins(cum)
    mask = (np.isfinite(Y).all(1) & np.isfinite(re) & (re > radii[0])
            & (edge_kpc[:, -1] <= radii[-1]) & np.all(np.diff(cum, axis=1) > 0, axis=1))
    return Y, re, mask


def density_from_cog(cog, radii):
    """Mode (4) input: log surface density ``Sigma(R) = dM/dA`` by differencing
    the log CoG. Returns ``(log_sigma (N, R-1), mid_radii (R-1,))``. (Valid only
    where the profile is reliable — e.g. a noiseless simulation; on noisy data
    this differencing amplifies noise.)
    """
    radii = np.asarray(radii, float)
    cum = 10.0 ** np.asarray(cog, float)
    dM = cum[:, 1:] - cum[:, :-1]
    dA = np.pi * (radii[1:] ** 2 - radii[:-1] ** 2)
    mid = np.sqrt(radii[:-1] * radii[1:])
    return np.log10(np.clip(dM, 1.0, None) / dA[None, :]), mid


def _cum_to_bins(cum):
    """(N, E) linear cumulative masses -> (N, E) log [first cumulative, rest annuli]."""
    Y = np.empty_like(cum)
    Y[:, 0] = np.log10(np.clip(cum[:, 0], 1.0, None))
    for k in range(1, cum.shape[1]):
        Y[:, k] = np.log10(np.clip(cum[:, k] - cum[:, k - 1], 1.0, None))
    return Y


# --------------------------------------------------------------------------- #
# profile emulator (modes 3 & 4): PCA-compress -> predict -> reconstruct       #
# --------------------------------------------------------------------------- #
@dataclass
class ProfileEmulator:
    """Predicts a full 1-D log profile (CoG or density) from the halo.

    Wraps a core ``Emulator`` over ``[anchor, PC1..PCK]``; reconstruction is
    linear, so ``predict`` returns an analytic per-radius mean and sigma.
    """
    emu: Emulator
    mean_shape: np.ndarray     # (R,) population mean shape (amplitude removed)
    modes: np.ndarray          # (K, R) PCA shape modes
    radii: np.ndarray          # (R,) kpc

    @property
    def _design(self):
        return np.column_stack([np.ones(len(self.radii)), self.modes.T])     # (R, K+1)

    def predict(self, X):
        """Per-radius predictive ``(mu, sigma)``, each (N, R)."""
        mu, _, cov = self.emu.predict(X)                                     # on [anchor, PCs]
        A = self._design
        mu_prof = mu @ A.T + self.mean_shape
        var = np.einsum("rk,nkl,rl->nr", A, cov, A)
        return mu_prof, np.sqrt(var)

    def sample(self, X, size=1, rng=None):
        """Draw ``size`` correlated profiles per halo; returns (size, N, R)."""
        draws = self.emu.sample(X, size=size, rng=rng)                       # (size, N, K+1)
        return draws @ self._design.T + self.mean_shape

    def scores(self, X):
        """Predicted ``[anchor, PC1..PCK]`` for halos ``X`` (the compressed vector)."""
        return self.emu.predict(X)[0]


def fit_profile(X, profile, anchor, radii, n_modes=3, mean="linear", ridge=2.0):
    """Fit a ``ProfileEmulator``.

    ``profile`` (N, R) is the per-radius log quantity (``log M(<R)`` for a CoG,
    ``log Sigma(R)`` for a density); ``anchor`` (N,) is the amplitude factored
    out (``logMtot``). PCA the shape ``profile - anchor`` in-sample to ``n_modes``,
    then fit the core emulator on ``[anchor, PC scores]``.
    """
    profile = np.asarray(profile, float)
    anchor = np.asarray(anchor, float)
    shape = profile - anchor[:, None]
    mean_shape = shape.mean(0)
    _, _, Vt = np.linalg.svd(shape - mean_shape, full_matrices=False)
    modes = Vt[:n_modes]
    scores = (shape - mean_shape) @ modes.T
    emu = fit(X, np.column_stack([anchor, scores]), mean=mean, ridge=ridge)
    return ProfileEmulator(emu, mean_shape, modes, np.asarray(radii, float))


def integrate_density(log_sigma, radii, central_log_mass):
    """Mode (4) -> cumulative: integrate ``Sigma(R)`` OUTWARD from the centre.

    ``radii`` (R,) is the SAME edge grid handed to ``density_from_cog``, so
    ``log_sigma`` (N, R-1) holds one annulus per pair of consecutive edges and
    the shell areas here match the ones the densities were built with — the
    round-trip is then exact on the grid (do NOT pass the annulus mid radii;
    mid-derived areas bias the steep inner region by up to ~0.24 dex).
    ``central_log_mass`` (N,) is ``log M(<radii[0])`` — the real,
    resolution-limited centre, which still counts toward larger apertures.
    Integrating outward (rather than inward from the total, or from R_min with
    no centre) is the only stable direction for the small inner cumulative (the
    near-cancellation lesson, exp22). Returns log cumulative ``M(<radii[1:])``
    (N, R-1).
    """
    radii = np.asarray(radii, float)
    dA = np.pi * (radii[1:] ** 2 - radii[:-1] ** 2)         # true annulus areas
    shells = 10.0 ** np.asarray(log_sigma, float) * dA[None, :]
    cum = 10.0 ** np.asarray(central_log_mass, float)[:, None] + np.cumsum(shells, axis=1)
    return np.log10(np.clip(cum, 1.0, None))


# --------------------------------------------------------------------------- #
# self-check: reproduce exp20 / exp21 / exp22 / exp22b                         #
# --------------------------------------------------------------------------- #
def _cv_profile(X, profile, anchor, radii, n_modes=3, k=5, seed=0):
    """Out-of-fold (mu, sigma) profiles + (pred, true) PC scores (in-fold PCA)."""
    n, nr = profile.shape
    order = np.random.default_rng(seed).permutation(n)
    MU = np.empty((n, nr))
    SIG = np.empty((n, nr))
    PRED = np.empty((n, n_modes))
    TRUE = np.empty((n, n_modes))
    for fold in np.array_split(order, k):
        tr = np.setdiff1d(np.arange(n), fold)
        pe = fit_profile(X[tr], profile[tr], anchor[tr], radii, n_modes=n_modes)
        MU[fold], SIG[fold] = pe.predict(X[fold])
        mu_sh = (profile[tr] - anchor[tr][:, None]).mean(0)
        TRUE[fold] = (profile[fold] - anchor[fold][:, None] - mu_sh) @ pe.modes.T
        PRED[fold] = pe.scores(X[fold])[:, 1:]
    return MU, SIG, PRED, TRUE


if __name__ == "__main__":
    from pathlib import Path

    from .emulator import _cv_oof
    from .metrics import crps_gaussian, interval_coverage

    # synthetic mechanics: fit_profile recovers a profile, predict/sample shape & calibration
    rng = np.random.default_rng(0)
    n, nr, K = 3000, 12, 3
    Xs = rng.normal(size=(n, 5))
    rad = np.linspace(2.0, 100.0, nr)
    modes_t = np.linalg.svd(rng.normal(size=(nr, nr)))[0][:K]            # orthonormal modes
    anchor_t = 11.0 + 0.4 * Xs[:, 0] + 0.05 * rng.normal(size=n)
    score_t = Xs[:, 1:4] * np.array([0.3, 0.2, 0.1]) + 0.10 * rng.normal(size=(n, 3))
    prof = anchor_t[:, None] + score_t @ modes_t                         # exactly K modes
    pe = fit_profile(Xs, prof, anchor_t, rad, n_modes=K)
    mu_p, sig_p = pe.predict(Xs)
    assert mu_p.shape == (n, nr) and sig_p.shape == (n, nr)
    draws = pe.sample(Xs, size=200, rng=1)
    assert draws.shape == (200, n, nr)
    _, cov = interval_coverage((prof - mu_p).ravel(), 0.0, sig_p.ravel())
    assert np.allclose(cov, [0.5, 0.68, 0.9, 0.95], atol=0.04), cov     # calibrated
    assert np.allclose(draws.std(0).mean(0), sig_p.mean(0), rtol=0.1)   # sample spread = sigma

    # density_from_cog -> integrate_density is a discrete identity: with the shell
    # areas taken between the SAME grid edges, the round-trip must reproduce the
    # input CoG at radii[1:] to float precision, for any profile shape.
    r_grid = np.arange(2 ** 0.25, 150 ** 0.25, 0.1) ** 4                # the 24-point CoG grid
    cog_uniform = np.log10(np.pi * r_grid ** 2 * 1e8)[None, :]          # uniform Sigma
    cog_steep = (11.5 + 0.8 * np.log10(1.0 - np.exp(-(r_grid / 8.0) ** 0.5)))[None, :]
    for cog_t in (cog_uniform, cog_steep):
        log_sig_t, _ = density_from_cog(cog_t, r_grid)
        cog_rt = integrate_density(log_sig_t, r_grid, cog_t[:, 0])
        err_rt = np.abs(cog_rt - cog_t[:, 1:]).max()
        assert err_rt < 1e-9, f"round-trip not exact: max err {err_rt:.4f} dex"

    # real-data reproduction of all four modes
    table = Path(__file__).resolve().parents[1] / "data" / "processed" / "tng300_072_z0p4.fits"
    if not table.exists():
        print("profile_emulator self-check OK (synthetic only; catalog FITS not found)")
    else:
        from astropy.table import Table

        from .tng_data import COG_RAD_KPC
        t = Table.read(table)
        t = t[t["use"]]
        cog = np.asarray(t["logmstar_cog"], float)
        aper = np.asarray(t["logmstar_aper"], float)
        feat = np.column_stack([np.asarray(t[c], float) for c in
                                ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late", "c_200c")])
        R = COG_RAD_KPC
        good = np.isfinite(cog).all(1) & np.isfinite(feat).all(1) & np.isfinite(aper).all(1)
        cog, aper, X = cog[good], aper[good], feat[good]
        logMtot = cog[:, -1]

        def _ann(a_o, a_i):
            return np.log10(np.clip(10.0 ** a_o - 10.0 ** a_i, 1.0, None))

        # (1) kpc apertures — reproduce exp20 (CRPS ~0.083)
        Ykpc = np.column_stack([aper[:, 0], _ann(aper[:, 1], aper[:, 0]),
                                _ann(aper[:, 2], aper[:, 1]), _ann(aper[:, 4], aper[:, 2])])
        mu, sig, _ = _cv_oof(X, Ykpc, "linear")
        crps_kpc = crps_gaussian(Ykpc, mu, sig).mean()

        # (2) Re bins — reproduce exp21 (well-calibrated)
        Yre, _, m = re_targets(cog, R, [0.5, 1, 2, 4, 6, 9])
        mu_re, sig_re, cov_re = _cv_oof(X[m], Yre[m], "linear")
        crps_re = crps_gaussian(Yre[m], mu_re, sig_re).mean()
        _, cov_cal = interval_coverage(Yre[m].ravel(), mu_re.ravel(), sig_re.ravel())

        # (3) CoG profile — exp22: per-radius CRPS, recon RMS, PC1 R^2 ~0.39
        MUc, SIGc, Pc, Tc = _cv_profile(X, cog[:, :-1], logMtot, R[:-1])
        r2_cog = 1.0 - ((Tc[:, 0] - Pc[:, 0]) ** 2).mean() / Tc[:, 0].var()
        recon_cog = np.sqrt(((MUc - cog[:, :-1]) ** 2).mean())

        # (4) density profile — exp22b: PC1 more halo-predictable (R^2 ~0.54 > CoG)
        logSig, mid = density_from_cog(cog, R)
        _, _, Pd, Td = _cv_profile(X, logSig, logMtot, mid)
        r2_den = 1.0 - ((Td[:, 0] - Pd[:, 0]) ** 2).mean() / Td[:, 0].var()

        print(f"profile_emulator self-check OK  n={len(X)}")
        print(f"  (1) kpc apertures   CRPS={crps_kpc:.4f}        (exp20 ~0.083)")
        print(f"  (2) Re bins (n={m.sum()}) CRPS={crps_re:.4f}  cov={'/'.join(f'{c:.2f}' for c in cov_cal)}")
        print(f"  (3) CoG profile     recon RMS={recon_cog:.3f}  PC1 R2={r2_cog:+.2f}  (exp22 ~0.39)")
        print(f"  (4) density profile PC1 R2={r2_den:+.2f}  (exp22b ~0.54, > CoG)")
        assert abs(crps_kpc - 0.083) < 0.004, crps_kpc
        assert crps_re < 0.085 and np.allclose(cov_cal, [0.5, 0.68, 0.9, 0.95], atol=0.05), (crps_re, cov_cal)
        assert recon_cog < 0.13 and r2_cog > 0.30, (recon_cog, r2_cog)
        assert r2_den > r2_cog + 0.08, (r2_den, r2_cog)                 # density PC1 more predictable
