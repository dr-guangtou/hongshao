"""exp33 step v-prep — representation repairs for the single-epoch CoG emulator.

Same inputs, same folds, same metrics; only the TARGET REPRESENTATION changes:
  A  baseline    mode-3 as shipped: [logMtot, PC1-3] of the kpc-space log CoG
  B  size-aware  [logMtot, logR50, PC1-3 of the shape in R/R50 coordinates];
                 reconstruction uses the PREDICTED R50 (exp08 lesson: aligned
                 coordinates beat capacity; targets the Re-plane defect)
  C  core-split  [logM(<5kpc), logMtot, PC1-3 of the R>5kpc shape]; the
                 marginally-resolved inner region (exp07) is one number, not a
                 contaminant of the shape basis; inner CoG filled with the
                 train-median inner shape scaled to the predicted core mass
  D  density     [logM(<Rmin), logMtot, PC1-3 of the log-density shape],
                 integrated OUTWARD to the CoG (exp22b: density PC1 is the
                 most halo-predictable shape mode, R2 0.54 vs 0.39)

Scores: shape max|rel| (pinned; all R and R>5 kpc), amplitude scatter, and the
Re plane M(<2Re) vs M(2-4Re) (slope/scatter + energy/floor) — the defect B is
built to fix.

Run: PYTHONPATH=. uv run python experiments/exp33_single_epoch/repr_fix.py
Demo: ... repr_fix.py demo
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hongshao import qa                                                              # noqa: E402
from hongshao.emulator import fit as emu_fit                                         # noqa: E402
from hongshao.profile_emulator import density_from_cog, integrate_density            # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                            # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR = HERE / "figures"
R = COG_RAD_KPC
NPZ = HERE / "outputs" / "single_epoch.npz"
UGRID = np.geomspace(0.1, 8.0, 22)               # R/R50 grid for architecture B
CORE_KPC = 5.0
KFOLD, SEED, NPC = 5, 0, 3


def folds_of(n):
    order = np.random.default_rng(SEED).permutation(n)
    return np.array_split(order, KFOLD)


def pca_fit(shape_tr):
    mean_shape = shape_tr.mean(0)
    _, _, Vt = np.linalg.svd(shape_tr - mean_shape, full_matrices=False)
    modes = Vt[:NPC]
    return mean_shape, modes


# ---- architecture builders: fit on train rows, predict log CoG (24,) on test ----
def arch_B(X, cog, r50, tr, te):
    shape = np.array([np.interp(UGRID * r50[i], R, cog[i]) - cog[i, -1]
                      for i in range(len(cog))])
    mean_shape, modes = pca_fit(shape[tr])
    scores = (shape - mean_shape) @ modes.T
    Y = np.column_stack([cog[:, -1], np.log10(r50), scores])
    emu = emu_fit(X[tr], Y[tr])
    mu, _, _ = emu.predict(X[te])
    out = np.empty((len(te), len(R)))
    for k in range(len(te)):
        shp = mean_shape + mu[k, 2:] @ modes
        u = R / 10.0 ** mu[k, 1]                             # predicted R50
        out[k] = mu[k, 0] + np.interp(u, UGRID, shp)
    return out


def arch_C(X, cog, tr, te):
    m_out = R > CORE_KPC
    core = np.array([np.interp(CORE_KPC, R, c) for c in cog])
    shape_out = cog[:, m_out] - cog[:, -1:]
    mean_shape, modes = pca_fit(shape_out[tr])
    scores = (shape_out - mean_shape) @ modes.T
    inner_shape = np.median(cog[tr][:, ~m_out] - core[tr][:, None], axis=0)
    Y = np.column_stack([core, cog[:, -1], scores])
    emu = emu_fit(X[tr], Y[tr])
    mu, _, _ = emu.predict(X[te])
    out = np.empty((len(te), len(R)))
    out[:, m_out] = mu[:, 1:2] + mean_shape + mu[:, 2:] @ modes
    out[:, ~m_out] = mu[:, 0:1] + inner_shape[None, :]
    return out


def arch_D(X, cog, tr, te):
    log_sig, mid = density_from_cog(cog, R)
    shape = log_sig - cog[:, -1:]
    mean_shape, modes = pca_fit(shape[tr])
    scores = (shape - mean_shape) @ modes.T
    Y = np.column_stack([cog[:, 0], cog[:, -1], scores])     # central, anchor, PCs
    emu = emu_fit(X[tr], Y[tr])
    mu, _, _ = emu.predict(X[te])
    sig_pred = mu[:, 1:2] + mean_shape + mu[:, 2:] @ modes
    cum = integrate_density(sig_pred, mid, mu[:, 0])
    return np.column_stack([mu[:, 0], np.array([np.interp(R[1:], mid, c)
                                                for c in cum])])


def cv_arch(builder, X, cog, *extra):
    n = len(cog)
    out = np.empty((n, len(R)))
    for fold in folds_of(n):
        tr = np.setdiff1d(np.arange(n), fold)
        out[fold] = builder(X, cog, *extra, tr, fold)
    return out


# ---- scoring --------------------------------------------------------------------
def score(name, pred_log, cog):
    truth = 10.0 ** cog[:, None, :]
    pred = 10.0 ** pred_log[:, None, :]
    pin = pred * (truth[:, :, -1:] / pred[:, :, -1:])
    mr_all = qa.profile_maxrel(pin, truth, R)
    mr_out = qa.profile_maxrel(pin, truth, R, rmin=qa.RMIN_KPC)
    amp = np.std(pred_log[:, -1] - cog[:, -1])
    t_, m_, _, _ = qa.measure_all(pred, truth, R)
    kx, ky = "Re:M(<2Re)", "Re:M(2-4Re)"
    lt = np.column_stack([np.log10(np.clip(t_[kx][:, 0], 1, None)),
                          np.log10(np.clip(t_[ky][:, 0], 1, None))])
    lm = np.column_stack([np.log10(np.clip(m_[kx][:, 0], 1, None)),
                          np.log10(np.clip(m_[ky][:, 0], 1, None))])
    st = qa.plane_stats(lt[:, 0], lt[:, 1])
    sm = qa.plane_stats(lm[:, 0], lm[:, 1])
    en = qa.plane_energy(lt, lm)
    print(f"  {name:>12s}: shape max|rel| {100*np.median(mr_out):5.1f}% (R>5) "
          f"{100*np.median(mr_all):5.1f}% (all) | amp {amp:.3f} dex | "
          f"Re-plane slope {st['slope']:.2f}->{sm['slope']:.2f} scatter "
          f"{st['scatter']:.3f}->{sm['scatter']:.3f} E/floor {en['energy_ratio']:.1f} "
          f"(centered {en['energy_ratio_centered']:.1f})")
    return dict(mr_out=100 * np.median(mr_out), mr_all=100 * np.median(mr_all),
                amp=amp, e=en["energy_ratio"], ec=en["energy_ratio_centered"],
                dscatter=abs(sm["scatter"] - st["scatter"]))


def main():
    d = np.load(NPZ)
    X, cog, mu3 = d["X"], d["cog"], d["mu3"]
    n = len(cog)
    r50 = np.array([qa.half_mass_radius(10.0 ** c, R) for c in cog])
    print(f"exp33 representation repairs (n={n}, same folds/metrics; "
          "shape pinned; Re plane = the known defect)\n")
    results = {}
    results["A baseline"] = score("A baseline", mu3, cog)
    results["B size-aware"] = score("B size-aware", cv_arch(arch_B, X, cog, r50), cog)
    results["C core-split"] = score("C core-split", cv_arch(arch_C, X, cog), cog)
    results["D density"] = score("D density", cv_arch(arch_D, X, cog), cog)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    names = list(results)
    for ax, key, title in ((axes[0], "mr_out", "shape max|rel| R>5 kpc [%]"),
                           (axes[1], "dscatter", "Re-plane |Δscatter| [dex]"),
                           (axes[2], "ec", "Re-plane energy/floor (centered)")):
        ax.bar(range(len(names)), [results[nm][key] for nm in names],
               color=["0.55", "#0072B2", "#009E73", "#D55E00"])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([nm.split()[0] for nm in names])
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"exp33 — target-representation repairs (n={n}, OOF mean)",
                 fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp33_repr_fix")[0])


def demo():
    """Self-check: with targets == truth each architecture's reconstruction is
    near-exact where it should be (B on-grid, C outer radii, D by integration)."""
    d = np.load(NPZ)
    cog = d["cog"][:200]
    r50 = np.array([qa.half_mass_radius(10.0 ** c, R) for c in cog])
    log_sig, mid = density_from_cog(cog, R)
    cum = integrate_density(log_sig, mid, cog[:, 0])
    back = np.array([np.interp(R[1:], mid, c) for c in cum])
    err = np.abs(back - cog[:, 1:])
    # the shell-area discretization is biased at BOTH ends: ~0.11 dex at the
    # steep innermost radius and an accumulating ~0.05 dex by the outermost
    # (architecture D's structural handicap; exp22's "stable" = monotonic,
    # not exact); the mid radii (~5-10 kpc) are clean
    per_rad = np.median(err, axis=0)
    assert per_rad.min() < 0.01 and np.median(per_rad) < 0.05, per_rad.round(3)
    shape = np.array([np.interp(UGRID * r50[i], R, cog[i]) - cog[i, -1]
                      for i in range(len(cog))])
    mean_shape, modes = pca_fit(shape)
    rec = mean_shape + ((shape - mean_shape) @ modes.T) @ modes
    u_ok = (UGRID > 0.3) & (UGRID < 4.0)
    rms = np.sqrt(((rec - shape)[:, u_ok] ** 2).mean())
    assert rms < 0.05, ("R/R50-basis reconstruction", rms)
    print(f"repr_fix.demo OK: density roundtrip per-radius median "
          f"{per_rad.min():.4f}-{per_rad.max():.4f} dex (biased ends); "
          f"R/R50 PCA-3 recon RMS {rms:.3f} dex")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        sys.exit(main())
