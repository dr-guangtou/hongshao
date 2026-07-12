"""exp33 step vi — single-epoch models at every snapshot + the epoch connection.

Five INDEPENDENT single-epoch emulators (z=0.4/0.7/1.0/1.5/2.0), one per
snapshot, same portable features [DiffMAH(4), c200c], same galaxies, same
folds. Targets: the mode-1 kpc masses of each epoch's progenitor CoG
[M(<10), M(10-30), M(30-50), M(50-100)] (+ a per-epoch CoG-profile mode for
the shape metric). Three questions:
  1  performance vs epoch — the independent-single-epoch ceiling the
     multi-epoch transport model can be compared against;
  2  cross-epoch residual correlation — does ONE per-galaxy latent persist
     through cosmic time? (decides the correlated dimension of any future
     multi-epoch stochastic layer);
  3  the CLOSURE test of the user's epoch-connection path: hold out one
     epoch entirely, interpolate the other epochs' model COEFFICIENTS in z,
     and predict the held epoch. Interpolated ~ directly-fitted => a
     continuous-z emulator by coefficient interpolation is viable.

Run: PYTHONPATH=. uv run python experiments/exp33_single_epoch/epochs.py
Demo: ... epochs.py demo
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from astropy.table import Table                                                      # noqa: E402
from hongshao import qa                                                              # noqa: E402
from hongshao.emulator import Emulator, fit as emu_fit                               # noqa: E402
from hongshao.profile_emulator import fit_profile                                    # noqa: E402
from hongshao.metrics import crps_gaussian                                           # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                            # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR = HERE / "figures"
EXP32 = ROOT / "experiments" / "exp32_full_population" / "outputs"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
R = COG_RAD_KPC
ZK = np.array([0.4, 0.7, 1.0, 1.5, 2.0])
EDGES = [10.0, 30.0, 50.0, 100.0]
TNAMES = ["M(<10)", "M(10-30)", "M(30-50)", "M(50-100)"]
KFOLD, SEED = 5, 0


def folds_of(n):
    order = np.random.default_rng(SEED).permutation(n)
    return np.array_split(order, KFOLD)


def load():
    pop = np.load(EXP32 / "population.npz")
    t = Table.read(TABLE)
    row = {int(g): i for i, g in enumerate(np.asarray(t["index"]))}
    feats = np.array([[t[c][row[int(g)]] for c in
                       ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late",
                        "c_200c")] for g in pop["index"]], float)
    ok = np.isfinite(feats).all(1)
    X, data = feats[ok], pop["data"][ok]
    cum = np.stack([np.column_stack([np.interp(e, R, data[i, k])
                                     for e in EDGES])
                    for i in range(len(data)) for k in range(5)])
    cum = cum.reshape(len(data), 5, 4)
    Y = np.log10(np.concatenate([cum[:, :, 0:1],
                                 np.clip(np.diff(cum, axis=2), 1.0, None)],
                                axis=2))
    return X, Y, data                                  # Y: (n, 5 epochs, 4 targets)


def cv_epoch(X, Yk):
    mu = np.empty_like(Yk)
    sig = np.empty_like(Yk)
    for fold in folds_of(len(Yk)):
        tr = np.setdiff1d(np.arange(len(Yk)), fold)
        emu = emu_fit(X[tr], Yk[tr])
        mu[fold], sig[fold], _ = emu.predict(X[fold])
    return mu, sig


def interp_emulator(emus, z_src, z_tgt):
    """Element-wise quadratic-in-z interpolation of fitted emulator params."""
    def q(vals):
        vals = np.stack(vals)                                # (E, ...)
        flat = vals.reshape(len(z_src), -1)
        out = np.empty(flat.shape[1])
        for j in range(flat.shape[1]):
            out[j] = np.polyval(np.polyfit(z_src, flat[:, j], 2), z_tgt)
        return out.reshape(vals.shape[1:])
    e0 = emus[0]
    return Emulator(e0.mean, e0.mu_x, e0.sd_x, q([e.beta for e in emus]),
                    q([e.gamma for e in emus]), q([e.corr for e in emus]))


def main():
    X, Y, data = load()
    n = len(X)
    print(f"exp33 step vi — independent single-epoch models (n={n}, shared "
          "features, same folds)\n")

    # (1) per-epoch performance + per-epoch CoG-profile shape metric
    mus, sigs, crps, shape_mr = [], [], [], []
    for k in range(5):
        mu, sig = cv_epoch(X, Y[:, k])
        mus.append(mu)
        sigs.append(sig)
        crps.append(float(crps_gaussian(Y[:, k], mu, sig).mean()))
        prof = np.log10(data[:, k, :-1])
        anchor = np.log10(data[:, k, -1])
        MU = np.empty((n, 23))
        for fold in folds_of(n):
            tr = np.setdiff1d(np.arange(n), fold)
            pe = fit_profile(X[tr], prof[tr], anchor[tr], R[:-1])
            MU[fold] = pe.predict(X[fold])[0]
        pred = np.column_stack([10.0 ** MU, data[:, k, -1:]])
        pin = pred * (data[:, k, -1:] / pred[:, -1:])
        shape_mr.append(100 * np.median(qa.profile_maxrel(
            pin[:, None, :], data[:, k][:, None, :], R, rmin=qa.RMIN_KPC)))
    print("  (1) per-epoch: CRPS (4 kpc masses) | CoG shape max|rel| R>5 kpc")
    for k in range(5):
        print(f"    z={ZK[k]}: {crps[k]:.4f} | {shape_mr[k]:5.1f}%   "
              f"dex " + " ".join(f"{np.std(mus[k][:, j] - Y[:, k, j]):.3f}"
                                 for j in range(4)))

    # (2) cross-epoch residual correlation (per-galaxy latent persistence)
    print("\n  (2) cross-epoch OOF residual correlation (mean over 4 targets):")
    res = [mus[k] - Y[:, k] for k in range(5)]
    C = np.eye(5)
    for a in range(5):
        for b in range(a + 1, 5):
            r = np.mean([spearmanr(res[a][:, j], res[b][:, j])[0]
                         for j in range(4)])
            C[a, b] = C[b, a] = r
    for a in range(5):
        print("    " + " ".join(f"{C[a, b]:+.2f}" for b in range(5)))

    # (3) closure: interpolate coefficients over the other epochs -> held epoch
    print("\n  (3) coefficient-interpolation closure (held epoch, fold-clean):")
    closure = {}
    for k in (1, 2, 3):                                # interior epochs
        others = [j for j in range(5) if j != k]
        mu_i = np.empty_like(Y[:, k])
        sig_i = np.empty_like(Y[:, k])
        mu_nn = np.empty_like(Y[:, k])
        nn = others[int(np.argmin(np.abs(ZK[others] - ZK[k])))]
        for fold in folds_of(n):
            tr = np.setdiff1d(np.arange(n), fold)
            emus = [emu_fit(X[tr], Y[tr][:, j]) for j in others]
            emu_int = interp_emulator(emus, ZK[others], ZK[k])
            mu_i[fold], sig_i[fold], _ = emu_int.predict(X[fold])
            mu_nn[fold], _, _ = emus[others.index(nn)].predict(X[fold])
        c_dir = crps[k]
        c_int = float(crps_gaussian(Y[:, k], mu_i, sig_i).mean())
        d_nn = float(np.mean(np.abs(mu_nn - Y[:, k])))
        d_int = float(np.mean(np.abs(mu_i - Y[:, k])))
        d_dir = float(np.mean(np.abs(mus[k] - Y[:, k])))
        closure[k] = (c_dir, c_int)
        print(f"    z={ZK[k]}: CRPS direct {c_dir:.4f} -> interpolated {c_int:.4f} "
              f"({100*(c_int/c_dir-1):+.1f}%) | MAE dir/int/nearest "
              f"{d_dir:.3f}/{d_int:.3f}/{d_nn:.3f}")

    _figure(crps, shape_mr, C, closure, X, Y, n)


def _figure(crps, shape_mr, C, closure, X, Y, n):
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.6))
    a, b, c, dax = axes
    a.plot(ZK, crps, "o-", c="#0072B2", label="CRPS (4 kpc masses)")
    a2 = a.twinx()
    a2.plot(ZK, shape_mr, "s--", c="#D55E00", label="CoG shape max|rel| R>5")
    a.set(xlabel="z", ylabel="CRPS", title="A. Per-epoch performance")
    a2.set_ylabel("max|rel| [%]", color="#D55E00")
    a.legend(loc="upper left", fontsize=7)
    a2.legend(loc="lower right", fontsize=7)

    im = b.imshow(C, vmin=0, vmax=1, cmap="cividis")
    b.set_xticks(range(5))
    b.set_yticks(range(5))
    b.set_xticklabels(ZK)
    b.set_yticklabels(ZK)
    for i in range(5):
        for j in range(5):
            b.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                   color="w" if C[i, j] < 0.6 else "k", fontsize=8)
    b.set_title("B. Cross-epoch residual correlation")
    plt.colorbar(im, ax=b, fraction=0.046)

    # C: coefficient evolution of the full-sample fits (beta of each target)
    emus = [emu_fit(X, Y[:, k]) for k in range(5)]
    for j, nm in enumerate(TNAMES):
        for f, fnm in enumerate(["logmp", "logtc", "early", "late", "c200c"]):
            v = [e.beta[j, 1 + f] for e in emus]
            if j == 0:
                c.plot(ZK, v, "o-", lw=1.2, label=fnm)
            elif f == 0:
                c.plot(ZK, v, "o-", lw=1.2, c="0.6", alpha=0.5)
    c.set(xlabel="z", ylabel="standardized mean coefficient",
          title="C. Coefficient evolution (M(<10); others grey: logmp)")
    c.legend(fontsize=7)

    ks = sorted(closure)
    w = 0.35
    dax.bar(np.arange(len(ks)) - w / 2, [closure[k][0] for k in ks], w,
            label="direct fit", color="#0072B2")
    dax.bar(np.arange(len(ks)) + w / 2, [closure[k][1] for k in ks], w,
            label="coeff-interpolated", color="#D55E00")
    dax.set_xticks(range(len(ks)))
    dax.set_xticklabels([f"z={ZK[k]}" for k in ks])
    dax.set(ylabel="CRPS", title="D. Held-epoch closure")
    dax.legend(fontsize=8)
    fig.suptitle(f"exp33 step vi — the epoch connection (n={n})", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp33_epochs")[0])


def demo():
    """Self-check: interp_emulator is exact when params vary quadratically in z."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(600, 5))
    emus = []
    z_src = [0.4, 0.7, 1.5, 2.0]
    for z in z_src:
        B = np.array([[10 + z ** 2, 0.3 * z, 0.1, 0.0, 0.05, -0.2 * z ** 2]])
        Y = np.column_stack([np.ones(600), X]) @ B.T + 0.05 * rng.standard_normal((600, 1))
        emus.append(emu_fit(X, Y))
    emu_i = interp_emulator(emus, np.array(z_src), 1.0)
    B_true = np.array([[10 + 1.0, 0.3, 0.1, 0.0, 0.05, -0.2]])
    Yt = np.column_stack([np.ones(600), X]) @ B_true.T
    mu, _, _ = emu_i.predict(X)
    rms = float(np.sqrt(((mu - Yt) ** 2).mean()))
    assert rms < 0.02, rms
    print(f"epochs.demo OK: quadratic coefficient interpolation exact "
          f"(held-z RMS {rms:.4f})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        sys.exit(main())
