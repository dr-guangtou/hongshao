"""exp37 — the multi-epoch statistical emulator (Path A, the product).

Builds the exp33-vi blueprint, every ingredient measured there (n=2397):
continuous-z mean/scatter by quadratic coefficient interpolation (held-epoch
closure +-4%), an AR(1)-in-epoch per-galaxy latent (cross-epoch OOF residual
correlation is Markovian, rho ~ 0.67 — measured in-fit here, never hardcoded),
and generative sampling (the mean alone is under-dispersed, exp15).

Subcommands:
  demo    — synthetic self-checks, no data (run first; TDD)
  fit     — per-epoch cores + shared-basis profile cores + rho -> outputs/
  qa      — OOF model CoGs -> hongshao.qa.evaluate per epoch + generative
            planes + cross-epoch coherence figures
  closure — held-epoch closure re-run in the shared-PCA-basis profile space
  report  — verdict tables vs the exp33-vi marks

Run: PYTHONPATH=. uv run python experiments/exp37_multi_epoch/run.py <cmd>
"""
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hongshao.emulator import fit as emu_fit, _chol_psd                      # noqa: E402
from hongshao.profile_emulator import ProfileEmulator                        # noqa: E402


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
ZK = np.array([0.4, 0.7, 1.0, 1.5, 2.0])
_EPOCHS = _load_by_path("exp33_epochs", ROOT / "experiments/exp33_single_epoch/epochs.py")


# --------------------------------------------------------------------------- #
# the model layer                                                              #
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


def _fit_rho(C, idx=None):
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


@dataclass
class MultiEpochEmulator:
    """The product: per-epoch heteroscedastic cores, continuous in z by
    quadratic coefficient interpolation, generative via an AR(1)-in-epoch
    latent (exp33-vi blueprint)."""
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
        return _EPOCHS.interp_emulator(self.emus, self.z_grid, z)

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


@dataclass
class MultiEpochProfile:
    """Profile product: ONE pooled PCA shape basis (so the compressed
    coefficients interpolate in z) + a MultiEpochEmulator over
    [anchor, shared-PC scores]."""
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


def fit_shared_profile(X, profs, anchors, radii, z_grid, n_modes=3, rho=0.0):
    """Fit the shared-basis profile product.

    ``profs`` (n, E, R) per-epoch log profiles; ``anchors`` (n, E) the
    amplitudes factored out (logMtot per epoch). One PCA basis from the POOLED
    amplitude-removed shapes of every epoch, then a per-epoch core on
    [anchor, shared scores] — unlike per-epoch ``fit_profile`` bases, the
    compressed coefficients then live in one space and interpolate in z.
    """
    profs = np.asarray(profs, float)
    anchors = np.asarray(anchors, float)
    shape = profs - anchors[..., None]
    mean_shape = shape.mean(axis=(0, 1))
    flat = (shape - mean_shape).reshape(-1, shape.shape[-1])
    _, _, Vt = np.linalg.svd(flat, full_matrices=False)
    modes = Vt[:n_modes]
    emus = [emu_fit(X, np.column_stack([anchors[:, k],
                                        (shape[:, k] - mean_shape) @ modes.T]))
            for k in range(shape.shape[1])]
    me = MultiEpochEmulator(np.asarray(z_grid, float), emus, rho)
    return MultiEpochProfile(me, mean_shape, modes, np.asarray(radii, float))


# --------------------------------------------------------------------------- #
# real data: fit / qa / closure / report                                       #
# --------------------------------------------------------------------------- #
N_DRAW = 3
NPZ = OUTDIR / "multi_epoch.npz"


def _load_data(dev=False):
    X, Y, data = _EPOCHS.load()          # (n,5feat), (n,5,4), (n,5,24) linear
    if dev:
        X, Y, data = X[:400], Y[:400], data[:400]
    profs = np.log10(data[..., :-1])
    anchors = np.log10(data[..., -1])
    return X, Y, data, profs, anchors


def _spearman_corr(res):
    """exp33-vi convention: mean-over-targets Spearman cross-epoch matrix."""
    from scipy.stats import spearmanr
    E = len(res)
    C = np.eye(E)
    for a in range(E):
        for b in range(a + 1, E):
            C[a, b] = C[b, a] = np.mean([spearmanr(res[a][:, j], res[b][:, j])[0]
                                         for j in range(res[a].shape[1])])
    return C


def cmd_fit(dev=False):
    from hongshao.metrics import crps_gaussian
    X, Y, data, profs, anchors = _load_data(dev)
    n = len(X)
    R = _EPOCHS.R
    print(f"exp37 fit (n={n}{', DEV' if dev else ''})")

    # (1) kpc mode: OOF per epoch (exp33-vi setup) + CRPS + cross-epoch rho
    mus = np.empty_like(Y)
    sigs = np.empty_like(Y)
    for k in range(5):
        mus[:, k], sigs[:, k] = _EPOCHS.cv_epoch(X, Y[:, k])
    crps = crps_gaussian(Y.reshape(-1, 4), mus.reshape(-1, 4),
                         sigs.reshape(-1, 4)).reshape(n, 5, 4).mean(axis=(0, 2))
    C = _spearman_corr([mus[:, k] - Y[:, k] for k in range(5)])
    rho = _fit_rho(C)
    print("  kpc CRPS per epoch: " + " ".join(f"{c:.4f}" for c in crps)
          + f"  (exp33-vi marks 0.081 -> 0.200)\n  rho = {rho:.3f} "
          f"(exp33-vi 0.67); adjacent C: "
          + " ".join(f"{C[k, k+1]:+.2f}" for k in range(4)))

    # (2) shared-basis profile mode: fold-clean OOF per-radius + anchor, and
    # OOF generative CoG draws from the SAME fold-fitted products
    MU = np.empty_like(profs)
    SIG = np.empty_like(profs)
    AMU = np.empty_like(anchors)
    ASIG = np.empty_like(anchors)
    cog_draws = np.empty((N_DRAW, n, 5, 24))
    for fi, fold in enumerate(_EPOCHS.folds_of(n)):
        tr = np.setdiff1d(np.arange(n), fold)
        mp = fit_shared_profile(X[tr], profs[tr], anchors[tr], R[:-1], ZK,
                                rho=rho)
        for k in range(5):
            MU[fold, k], SIG[fold, k] = mp.at_z(ZK[k]).predict(X[fold])
            m, s, _ = mp.me.emus[k].predict(X[fold])
            AMU[fold, k], ASIG[fold, k] = m[:, 0], s[:, 0]
        dc = mp.me.sample_epochs(X[fold], size=N_DRAW, rng=100 + fi)
        A = np.column_stack([np.ones(len(mp.radii)), mp.modes.T])
        dprof = dc @ A.T + mp.mean_shape                    # (S, f, 5, 23)
        cog_draws[:, fold] = np.concatenate(
            [10.0 ** dprof, 10.0 ** dc[..., 0:1]], axis=-1)
    OUTDIR.mkdir(exist_ok=True)
    np.savez(NPZ, crps=crps, C=C, rho=rho, mu_kpc=mus, sig_kpc=sigs,
             MU=MU, SIG=SIG, AMU=AMU, ASIG=ASIG, cog_draws=cog_draws,
             dev=dev, n=n)
    print(f"  wrote {NPZ}")


def _model_cogs(d):
    """OOF mean model CoGs (n, 5, 24), linear, from the saved fit."""
    return np.concatenate([10.0 ** d["MU"], 10.0 ** d["AMU"][..., None]],
                          axis=-1)


def cmd_qa(dev=False):
    from hongshao import qa
    X, Y, data, profs, anchors = _load_data(dev)
    d = np.load(NPZ)
    assert int(d["n"]) == len(X), "saved fit does not match the loaded sample"
    R = _EPOCHS.R
    model = _model_cogs(d)
    FIGDIR.mkdir(exist_ok=True)
    res = qa.evaluate(model, data, R, list(ZK), name="exp37_oof",
                      figdir=FIGDIR, bin_by=X[:, 0], bin_label="dmah logmp",
                      draw_cogs=d["cog_draws"])

    # generative: per-epoch observational planes on the OOF draws
    print("\n  generative planes (energy/floor full | centered), "
          f"{N_DRAW} OOF drawn populations vs truth:")
    truth_m, _, rhalf, _ = qa.measure_all(data, data, R)
    for di in range(N_DRAW):
        dm, _, _, _ = qa.measure_all(d["cog_draws"][di], d["cog_draws"][di], R)
        for (kx, ky) in qa.PLANES:
            cells = []
            for k in range(5):
                pe = qa.plane_energy(
                    np.column_stack([np.log10(np.clip(truth_m[kx][:, k], 1.0, None)),
                                     np.log10(np.clip(truth_m[ky][:, k], 1.0, None))]),
                    np.column_stack([np.log10(np.clip(dm[kx][:, k], 1.0, None)),
                                     np.log10(np.clip(dm[ky][:, k], 1.0, None))]))
                cells.append(f"{pe['energy_ratio']:4.1f}|{pe['energy_ratio_centered']:4.1f}")
            print(f"    draw {di} {kx} vs {ky}: " + "  ".join(cells))

    # cross-epoch coherence: (a) draw residual correlation vs the measured C;
    # (b) the growth plane logM*tot(z=2) vs logM*tot(z=0.4), truth vs draws
    la = np.log10(d["cog_draws"][..., -1])                   # (S, n, 5)
    res_d = (la - d["AMU"][None]) / d["ASIG"][None]
    C_draw = np.corrcoef(res_d.transpose(2, 0, 1).reshape(5, -1))
    print("\n  cross-epoch coherence: measured C (kpc OOF) vs draws:")
    for a in range(5):
        print("    " + " ".join(f"{d['C'][a, b]:+.2f}" for b in range(5))
              + "   |   " + " ".join(f"{C_draw[a, b]:+.2f}" for b in range(5)))
    print(f"    rho: measured {float(d['rho']):.3f} -> draws "
          f"{_fit_rho(C_draw):.3f}")
    gt = np.column_stack([anchors[:, 4], anchors[:, 0]])
    ge = [qa.plane_energy(gt, np.column_stack([la[i, :, 4], la[i, :, 0]]))
          for i in range(N_DRAW)]
    gm = qa.plane_energy(gt, np.column_stack([d["AMU"][:, 4], d["AMU"][:, 0]]))
    print("  growth plane logMtot(z=2) vs logMtot(z=0.4), energy/floor "
          "full | centered:")
    print(f"    mean prediction: {gm['energy_ratio']:.1f} | "
          f"{gm['energy_ratio_centered']:.1f}")
    for i, g in enumerate(ge):
        print(f"    draw {i}:          {g['energy_ratio']:.1f} | "
              f"{g['energy_ratio_centered']:.1f}")
    _coherence_figure(d, C_draw, anchors, la)
    return res


def _coherence_figure(d, C_draw, anchors, la):
    import matplotlib.pyplot as plt
    from hongshao.plotting import set_style, save_fig
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    a, b, c = axes
    for M, ax, title in ((d["C"], a, "measured cross-epoch C (kpc OOF)"),
                         (C_draw, b, "draw residual correlation")):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="cividis")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(ZK)
        ax.set_yticklabels(ZK)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w" if M[i, j] < 0.6 else "k", fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    c.scatter(anchors[:, 4], anchors[:, 0], s=6, alpha=0.4, c="#0072B2",
              edgecolors="none", label="truth")
    c.scatter(la[0, :, 4], la[0, :, 0], s=8, facecolors="none",
              edgecolors="0.3", lw=0.5, label="draw 0")
    c.set(xlabel="log Mtot (z=2)", ylabel="log Mtot (z=0.4)",
          title="growth plane")
    c.legend(fontsize=8)
    fig.suptitle(f"exp37 cross-epoch coherence (rho={float(d['rho']):.2f})",
                 fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp37_coherence")[0])


def cmd_closure(dev=False):
    """Held-epoch closure in the SHARED-basis profile space (fold-clean over
    galaxies AND epochs): basis + cores from the other four epochs, cores
    interpolated to the held z. The product's continuous-z claim for profiles."""
    from hongshao import qa
    X, Y, data, profs, anchors = _load_data(dev)
    n = len(X)
    R = _EPOCHS.R
    d = np.load(NPZ)
    model_direct = _model_cogs(d)
    print(f"exp37 closure — shared-basis profile space (n={n})")
    print("  held z: CoG max|rel| median, direct shared-basis fit -> "
          "coefficient-interpolated (all R | R>5 kpc)")
    out = {}
    for k in (1, 2, 3):
        others = [j for j in range(5) if j != k]
        MU_i = np.empty((n, profs.shape[-1]))
        AMU_i = np.empty(n)
        for fold in _EPOCHS.folds_of(n):
            tr = np.setdiff1d(np.arange(n), fold)
            mp = fit_shared_profile(X[tr], profs[tr][:, others],
                                    anchors[tr][:, others], R[:-1],
                                    ZK[others])
            pe = mp.at_z(ZK[k])
            MU_i[fold] = pe.predict(X[fold])[0]
            AMU_i[fold] = mp.me.at_z(ZK[k]).predict(X[fold])[0][:, 0]
        cog_i = np.concatenate([10.0 ** MU_i, 10.0 ** AMU_i[:, None]],
                               axis=1)[:, None, :]
        truth_k = data[:, k][:, None, :]
        direct_k = model_direct[:, k][:, None, :]
        rows = []
        for cog_m in (direct_k, cog_i):
            mr_a = 100 * np.nanmedian(qa.profile_maxrel(cog_m, truth_k, R))
            mr_o = 100 * np.nanmedian(qa.profile_maxrel(cog_m, truth_k, R,
                                                        rmin=qa.RMIN_KPC))
            rows.append((mr_a, mr_o))
        out[k] = rows
        print(f"    z={ZK[k]}: {rows[0][0]:5.1f}% | {rows[0][1]:5.1f}%  ->  "
              f"{rows[1][0]:5.1f}% | {rows[1][1]:5.1f}%")
    np.savez(OUTDIR / "closure.npz",
             held=np.array([[out[k][i][j] for i in range(2) for j in range(2)]
                            for k in (1, 2, 3)]))
    print(f"  wrote {OUTDIR / 'closure.npz'}")


def cmd_report():
    d = np.load(NPZ)
    print("exp37 report — the multi-epoch statistical emulator "
          f"(n={int(d['n'])}{', DEV' if bool(d['dev']) else ''})")
    print("  kpc CRPS per epoch : "
          + " ".join(f"{c:.4f}" for c in d["crps"])
          + "   (exp33-vi independent ceiling 0.081 -> 0.200)")
    print(f"  AR(1) rho          : {float(d['rho']):.3f}   (exp33-vi 0.67)")
    cl = OUTDIR / "closure.npz"
    if cl.exists():
        held = np.load(cl)["held"]
        print("  profile closure (held z, direct -> interpolated, "
              "all R | R>5):")
        for row, k in zip(held, (1, 2, 3)):
            print(f"    z={ZK[k]}: {row[0]:5.1f}%|{row[1]:5.1f}% -> "
                  f"{row[2]:5.1f}%|{row[3]:5.1f}%")


# --------------------------------------------------------------------------- #
# self-check                                                                   #
# --------------------------------------------------------------------------- #
def demo():
    rng = np.random.default_rng(0)

    # (1) the AR(1) cross-epoch sampler: within-epoch correlation must be
    # exactly R_k, cross-epoch same-target correlation must decay as rho^sep
    E, T, rho = 5, 3, 0.6
    R_t = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.4], [0.2, 0.4, 1.0]])
    eps = _sample_eps(rho, [R_t] * E, 200_000, rng)               # (n, E, T)
    assert eps.shape == (200_000, E, T)
    for k in range(E):
        C_in = np.corrcoef(eps[:, k, :].T)
        assert np.allclose(C_in, R_t, atol=0.02), f"within-epoch corr off at {k}"
    for t in range(T):
        C_x = np.corrcoef(eps[:, :, t].T)
        sep = np.abs(np.arange(E)[:, None] - np.arange(E)[None, :])
        assert np.allclose(C_x, rho ** sep, atol=0.02), "cross-epoch corr not AR(1)"

    # (2) MultiEpochEmulator: coefficients quadratic in z are recovered at a
    # held z; at a snapshot, at_z returns the DIRECT per-epoch core
    n = 3000
    X = rng.normal(size=(n, 5))
    emus = []
    Ys = []
    for z in ZK:
        B = np.array([[10 + z ** 2, 0.3 * z, 0.1, 0.0, 0.05, -0.2 * z ** 2],
                      [9 - 0.5 * z, -0.1 * z, 0.2, 0.1, 0.0, 0.1 * z]])
        Y = (np.column_stack([np.ones(n), X]) @ B.T
             + np.array([0.05, 0.10]) * rng.standard_normal((n, 2)))
        Ys.append(Y)
        emus.append(emu_fit(X, Y))
    me = MultiEpochEmulator(ZK, emus, rho=0.6)
    assert me.at_z(ZK[2]) is emus[2], "at_z at a snapshot must be the direct core"
    z_held = 0.85
    B_true = np.array([[10 + z_held ** 2, 0.3 * z_held, 0.1, 0.0, 0.05,
                        -0.2 * z_held ** 2],
                       [9 - 0.5 * z_held, -0.1 * z_held, 0.2, 0.1, 0.0,
                        0.1 * z_held]])
    mu_h, _, _ = me.predict(X, z_held)
    Yt = np.column_stack([np.ones(n), X]) @ B_true.T
    rms = float(np.sqrt(((mu_h - Yt) ** 2).mean()))
    assert rms < 0.02, f"held-z quadratic recovery failed: rms {rms:.4f}"

    # (3) generative draws: shapes; per-epoch spread matches sigma; the draws'
    # cross-epoch residual correlation reproduces the AR(1) target
    draws = me.sample_epochs(X[:400], size=250, rng=1)            # (250, 400, E, T)
    assert draws.shape == (250, 400, E, 2)
    mu_all = np.stack([e.predict(X[:400])[0] for e in emus], axis=1)
    sig_all = np.stack([e.predict(X[:400])[1] for e in emus], axis=1)
    res = (draws - mu_all[None]) / sig_all[None]
    assert np.allclose(res.std(axis=0).mean(axis=0), 1.0, atol=0.05), \
        "draw spread != sigma"
    flat = res.transpose(2, 0, 1, 3).reshape(E, -1)
    C_d = np.corrcoef(flat)
    sep = np.abs(np.arange(E)[:, None] - np.arange(E)[None, :])
    assert np.allclose(C_d, 0.6 ** sep, atol=0.04), "draws not AR(1)-coherent"

    # (4) rho measurement: AR(1) fit to synthetic residual correlations
    # recovers the generating rho
    rho_hat = _fit_rho(C_d)
    assert abs(rho_hat - 0.6) < 0.03, f"rho measurement off: {rho_hat:.3f}"

    # (5) shared-PCA-basis profile mode: with one shared mode set and scores
    # quadratic in z, the held-z profile is recovered through the shared basis
    nr, K = 18, 2
    rad = np.linspace(2.0, 120.0, nr)
    modes_t = np.linalg.svd(rng.normal(size=(nr, nr)))[0][:K]
    profs = np.empty((n, len(ZK), nr))
    anchors = np.empty((n, len(ZK)))
    for k, z in enumerate(ZK):
        anchors[:, k] = 11.0 + 0.4 * X[:, 0] - 0.3 * z
        scores = (X[:, 1:3] * np.array([0.3 * z, 0.15])
                  + 0.05 * rng.standard_normal((n, K)))
        profs[:, k] = anchors[:, k][:, None] + scores @ modes_t
    mp = fit_shared_profile(X, profs, anchors, rad, ZK, n_modes=K, rho=0.6)
    pe_snap = mp.at_z(ZK[1])
    mu_p, sig_p = pe_snap.predict(X)
    rms_snap = float(np.sqrt(((mu_p - profs[:, 1]) ** 2).mean()))
    assert rms_snap < 0.06, f"shared-basis snapshot recon rms {rms_snap:.4f}"
    mu_h, _ = mp.at_z(0.85).predict(X)
    prof_h = ((11.0 + 0.4 * X[:, 0] - 0.3 * 0.85)[:, None]
              + (X[:, 1:3] * np.array([0.3 * 0.85, 0.15])) @ modes_t)
    rms_h = float(np.sqrt(((mu_h - prof_h) ** 2).mean()))
    assert rms_h < 0.06, f"shared-basis held-z rms {rms_h:.4f}"

    print("exp37 demo OK: AR(1) sampler exact (within R, cross rho^sep); "
          f"held-z coefficient recovery rms {rms:.4f}; draws AR(1)-coherent; "
          f"rho_hat {rho_hat:.3f}; shared-basis profile snapshot/held-z rms "
          f"{rms_snap:.4f}/{rms_h:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dev = "--dev" in sys.argv
    if cmd == "demo":
        demo()
    elif cmd == "fit":
        cmd_fit(dev)
    elif cmd == "qa":
        cmd_qa(dev)
    elif cmd == "closure":
        cmd_closure(dev)
    elif cmd == "report":
        cmd_report()
    else:
        sys.exit(f"unknown subcommand {cmd!r}")
