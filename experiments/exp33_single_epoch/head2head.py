"""exp33 step iv — physical vs statistical CoG head-to-head at z=0.4.

Same galaxies (exp32 population ∩ exp33 sample), both held-out, same QA:
  statistical  mode-3 ProfileEmulator OOF mean (PCA-3 on [DiffMAH+c200c];
               all freedom spent on z=0.4; never sees the MAH curve)
  physical     exp32 transport emulator, theta(logMh) 10-fold-CV CoGs (the
               adopted multi-epoch mean model; one consistent history through
               five epochs; consumes the full DiffMAH deposit history)

Two amplitude treatments so the settled amplitude question cannot pollute it:
  e2e     halo-only end-to-end (statistical: own predicted total; physical:
          LOO SHMR at z=0.4)
  pinned  both rescaled to the true total (pure shape) — DIAGNOSTIC only.

Questions: (1) the single-epoch price of multi-epoch consistency; (2) do the
two model classes fail on the SAME galaxies (residual correlation: high =
shared information wall, low = complementary -> hybrid headroom); (3) where in
radius the physics helps/hurts.

Run: PYTHONPATH=. uv run python experiments/exp33_single_epoch/head2head.py
Demo: ... head2head.py demo
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
from hongshao.tng_data import COG_RAD_KPC                                            # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR = HERE / "figures"
EXP32 = ROOT / "experiments" / "exp32_full_population" / "outputs"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
R = COG_RAD_KPC
R_PROBE = [5.0, 30.0, 100.0]                     # correlation probe radii [kpc]


def loo_linear(X, y):
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    h = np.einsum("ij,jk,ik->i", A, np.linalg.pinv(A.T @ A), A)
    return y - (y - A @ beta) / np.clip(1.0 - h, 1e-9, None)


def exp33_indices():
    """Reproduce run.py's sample mask -> galaxy indices (deterministic)."""
    t = Table.read(TABLE)
    t = t[t["use"]]
    cog = np.asarray(t["logmstar_cog"], float)
    aper = np.asarray(t["logmstar_aper"], float)
    X = np.column_stack([np.asarray(t[c], float) for c in
                         ("dmah_logmp", "dmah_logtc", "dmah_early", "dmah_late",
                          "c_200c")])
    g = np.isfinite(cog).all(1) & np.isfinite(X).all(1) & np.isfinite(aper).all(1)
    return np.asarray(t["index"])[g]


def load_common():
    """Common-sample truth + both models' held-out z=0.4 CoGs (linear masses)."""
    d33 = np.load(HERE / "outputs" / "single_epoch.npz")
    idx33 = exp33_indices()
    assert len(idx33) == len(d33["cog"])
    pop = np.load(EXP32 / "population.npz")
    cv = np.load(EXP32 / "um_cv_diffmah.npz")
    row33 = {int(g): i for i, g in enumerate(idx33)}
    keep32 = [i for i, g in enumerate(pop["index"]) if int(g) in row33]
    sel33 = [row33[int(pop["index"][i])] for i in keep32]
    truth = pop["data"][keep32, 0, :]                       # (n, 24) z=0.4
    assert np.allclose(truth, 10.0 ** d33["cog"][sel33], rtol=1e-6), \
        "exp32/exp33 truth CoGs must agree on the common sample"
    stat = 10.0 ** d33["mu3"][sel33]                        # OOF mean, own amplitude
    phys = cv["cogs_slope"][keep32, 0, :]                   # pinned CV prediction
    shmr = loo_linear(pop["logmh_zk_diffmah"][keep32, [0]].reshape(-1, 1),
                      np.log10(truth[:, -1]))
    phys_e2e = phys * (10.0 ** shmr / truth[:, -1])[:, None]
    logmh = pop["logmh"][keep32]
    ok = np.isfinite(phys).all(1)
    return (truth[ok], stat[ok], phys_e2e[ok], phys[ok], logmh[ok])


def pin(cogs, truth):
    return cogs * (truth[:, -1:] / cogs[:, -1:])


def maxrel(cogs, truth, rmin=5.0):
    m = R > rmin
    return np.abs((cogs[:, m] - truth[:, m]) / truth[:, m]).max(axis=1)


def main():
    truth, stat, phys_e2e, phys_pin, logmh = load_common()
    n = len(truth)
    stat_pin = pin(stat, truth)
    print(f"exp33 step iv — physical vs statistical z=0.4 CoG (common n={n})\n")

    print("  (1) worst-radius error, median max|rel| (R>5 kpc | all R):")
    for nm, c in (("statistical e2e", stat), ("physical e2e", phys_e2e),
                  ("statistical shape", stat_pin), ("physical shape", phys_pin)):
        ma = 100 * np.median(maxrel(c, truth, 0.0))
        m5 = 100 * np.median(maxrel(c, truth, 5.0))
        print(f"    {nm:>18s}: {m5:5.1f}% | {ma:5.1f}%")
    d_pair = maxrel(phys_pin, truth) - maxrel(stat_pin, truth)
    print(f"    paired (shape): physical-statistical median {100*np.median(d_pair):+.1f} "
          f"points; statistical better for {100*np.mean(d_pair > 0):.0f}% of galaxies")

    print("\n  (2) residual correlation between the models (shape, pinned):")
    rel_s = (stat_pin - truth) / truth
    rel_p = (phys_pin - truth) / truth
    for r in R_PROBE:
        j = int(np.argmin(np.abs(R - r)))
        rho, p = spearmanr(rel_s[:, j], rel_p[:, j])
        print(f"    R={R[j]:6.1f} kpc: Spearman rho={rho:+.2f} (p={p:.1e})")
    rho_mx, _ = spearmanr(maxrel(stat_pin, truth), maxrel(phys_pin, truth))
    print(f"    per-galaxy max|rel|: rho={rho_mx:+.2f}")
    amp_s = np.log10(stat[:, -1] / truth[:, -1])
    amp_p = np.log10(phys_e2e[:, -1] / truth[:, -1])
    rho_a, _ = spearmanr(amp_s, amp_p)
    print(f"    amplitude residuals: rho={rho_a:+.2f} "
          f"(scatter {np.std(amp_s):.3f} vs {np.std(amp_p):.3f} dex)")

    print("\n  (3) per-radius median |rel| (shape), % :")
    med_s = 100 * np.median(np.abs(rel_s), axis=0)
    med_p = 100 * np.median(np.abs(rel_p), axis=0)
    for j in range(0, 24, 4):
        print(f"    R={R[j]:6.1f} kpc: statistical {med_s[j]:5.1f}  "
              f"physical {med_p[j]:5.1f}")
    _figure(truth, rel_s, rel_p, stat_pin, phys_pin, logmh, med_s, med_p, n)


def _figure(truth, rel_s, rel_p, stat_pin, phys_pin, logmh, med_s, med_p, n):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    a, b, c = axes
    for nm, med, col in (("statistical (mode 3)", med_s, "#0072B2"),
                         ("physical (transport)", med_p, "#D55E00")):
        a.plot(R, med, "o-", c=col, lw=1.8, ms=4, label=nm)
    a.axvline(5.0, c="0.7", ls=":", lw=1)
    a.set(xscale="log", xlabel="R [kpc]", ylabel="median |model-data|/data [%]",
          title="A. Shape error vs radius (both pinned)")
    a.legend(fontsize=8)

    j30 = int(np.argmin(np.abs(R - 30.0)))
    rho, _ = spearmanr(rel_s[:, j30], rel_p[:, j30])
    b.scatter(100 * rel_s[:, j30], 100 * rel_p[:, j30], s=7, alpha=0.3,
              c="#0072B2", edgecolors="none")
    b.axhline(0, c="0.6", lw=0.7)
    b.axvline(0, c="0.6", lw=0.7)
    lim = np.percentile(100 * np.abs(np.r_[rel_s[:, j30], rel_p[:, j30]]), 99)
    b.set(xlim=(-lim, lim), ylim=(-lim, lim),
          xlabel="statistical residual at 30 kpc [%]",
          ylabel="physical residual at 30 kpc [%]",
          title=f"B. Do they fail on the same galaxies?\nSpearman rho={rho:+.2f}")

    edges = np.quantile(logmh, np.linspace(0, 1, 5))
    x = np.arange(4)
    for nm, cg, col in (("statistical", stat_pin, "#0072B2"),
                        ("physical", phys_pin, "#D55E00")):
        v = [100 * np.median(maxrel(cg, truth)[(logmh >= edges[q])
                                               & (logmh <= edges[q + 1] + 1e-9)])
             for q in range(4)]
        c.plot(x, v, "o-", c=col, lw=1.8, label=nm)
    c.set_xticks(x)
    c.set_xticklabels([f"{edges[q]:.2f}-{edges[q+1]:.2f}" for q in range(4)],
                      fontsize=8)
    c.set(xlabel="logMh quartile", ylabel="median max|rel| R>5 kpc [%]",
          title="C. Shape error vs halo mass")
    c.legend(fontsize=8)
    fig.suptitle(f"exp33 step iv — physical vs statistical z=0.4 CoG "
                 f"(held-out, common n={n})", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp33_head2head")[0])


def demo():
    """Self-check: pin() exact at the outer radius; maxrel masks radii; the
    common-sample loader agrees between exp32 and exp33 truths."""
    truth, stat, phys_e2e, phys_pin, logmh = load_common()
    assert np.allclose(pin(stat, truth)[:, -1], truth[:, -1], rtol=1e-12)
    assert np.allclose(phys_pin[:, -1], truth[:, -1], rtol=1e-6)
    full = maxrel(truth * 1.1, truth, 0.0)
    assert np.allclose(full, 0.1, atol=1e-12)
    print(f"head2head.demo OK: common n={len(truth)}, truths agree, "
          "pin exact, maxrel sane")


if __name__ == "__main__":
    if sys.argv[1:2] == ["demo"]:
        demo()
    else:
        sys.exit(main())
