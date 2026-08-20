"""exp52 — what the DATA does: differential density-profile evolution by halo mass.

The empirical counterpart to `decompose.py`. That measures how the MODEL grows a
galaxy; this measures how the galaxies actually grow, straight from the measured
curves of growth, with no model involved.

Prior record: **exp26** established the headline on a wider sample (n=3380) --
growth is inside-out and multiplicative, `Sigma_low/Sigma_high ~ R^b`, with
b_stack = +0.15/+0.16/+0.28/+0.34 for the four adjacent pairs and +0.95 over the
long baseline, where the density at 8 kpc grows x4.2 and at 60 kpc x29. exp26
did NOT split by halo mass and showed the per-galaxy spread only for the long
baseline. This adds both, on the n=2397 modeling sample so the numbers line up
with everything else in the model thread.

Sigma comes from differencing the measured CoG (`density_from_cog`), the same
operator the objective uses -- NOT the isophote `intensity` column exp26 used,
so small offsets from exp26 are expected and are a definition difference, not a
disagreement.

Run: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \
experiments/exp52_growth_mechanism/data_differential.py [--dev]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import stage2_multiepoch as s2               # noqa: E402
from hongshao.plotting import OKABE_ITO, save_fig, set_style   # noqa: E402
from hongshao.profile_emulator import density_from_cog         # noqa: E402

ANCHOR_Z = [0.4, 0.7, 1.0, 1.5, 2.0]
# (earlier k, later k) -- k=0 is z=0.4, k=4 is z=2.0
PAIRS = [(4, 3), (3, 2), (2, 1), (1, 0), (4, 0)]
FIT_LO, FIT_HI = 6.0, 100.0          # exp26's resolved range, kept for comparison
N_SHOW = 70                          # individual galaxies drawn per panel


def main(dev=False):
    pop = np.load(ROOT / "experiments/exp32_full_population/outputs/"
                  "population.npz")
    rows = pop["dev100"] if dev else None
    s2._w_init(rows)
    gals, R = s2._W["gals"], s2._W["e"].R

    cogs = np.stack([np.stack([g["data"][k] for k in range(5)]) for g in gals])
    n, ne, nr = cogs.shape
    lsig, mid = density_from_cog(
        np.log10(np.clip(cogs.reshape(-1, nr), 1.0, None)), R)
    lsig = lsig.reshape(n, ne, nr - 1)
    logmh = np.array([pop["logmh"][g["row"]] for g in gals])

    edges = [logmh.min() - 1e-6] + list(np.percentile(logmh, [33.3, 66.7])) \
        + [logmh.max() + 1e-6]
    bins = [(edges[i], edges[i + 1]) for i in range(3)]
    print(f"exp52 data differential — n={n}, Sigma from the measured CoG")
    for lo, hi in bins:
        print(f"  halo mass bin {lo:.2f}-{hi:.2f}: "
              f"{int(((logmh >= lo) & (logmh < hi)).sum())} galaxies")

    set_style()
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(3, 5, figsize=(19.0, 9.4), sharex=True,
                             sharey=True)
    print(f"\n  power-law slope b of dlogSigma = a + b logR, fitted "
          f"{FIT_LO:g}-{FIT_HI:g} kpc, on the MEDIAN profile:")
    print(f"    {'halo mass bin':<18}" + "".join(
        f"{f'z{ANCHOR_Z[a]}->{ANCHOR_Z[b]}':>14}" for a, b in PAIRS))
    fitm = (mid >= FIT_LO) & (mid <= FIT_HI)

    for i, (lo, hi) in enumerate(bins):
        sel = (logmh >= lo) & (logmh < hi)
        slopes = []
        for j, (ka, kb) in enumerate(PAIRS):
            ax = axes[i, j]
            d = lsig[sel, kb] - lsig[sel, ka]            # later minus earlier
            ok = np.isfinite(d).all(axis=1)
            d = d[ok]
            idx = rng.choice(len(d), min(N_SHOW, len(d)), replace=False)
            for row in d[idx]:
                ax.plot(mid, row, color="0.7", lw=0.4, alpha=0.45, zorder=1)
            med = np.median(d, axis=0)
            q16, q84 = np.percentile(d, [16, 84], axis=0)
            ax.fill_between(mid, q16, q84, color=OKABE_ITO[1], alpha=0.35,
                            zorder=2, lw=0)
            ax.plot(mid, med, color=OKABE_ITO[5], lw=2.2, zorder=3)
            ax.axhline(0.0, color="k", lw=0.8, ls=":", zorder=0)
            ax.set_xscale("log")
            # A few per cent of galaxies have an unreliable outermost annulus
            # (near-flat CoG -> the 1 Msun floor in density_from_cog), which
            # sends individual curves off scale. Fixed limits keep the medians
            # and the 16-84 band legible; the outliers run off the top.
            ax.set_ylim(-0.35, 2.05)
            ax.set_xlim(mid.min(), 130.0)
            b = np.polyfit(np.log10(mid[fitm]), med[fitm], 1)[0]
            slopes.append(b)
            ax.text(0.04, 0.93, f"$b={b:+.2f}$", transform=ax.transAxes,
                    fontsize=9, va="top")
            if i == 0:
                ax.set_title(f"$z={ANCHOR_Z[ka]}\\rightarrow{ANCHOR_Z[kb]}$",
                             fontsize=10)
            if j == 0:
                ax.set_ylabel(f"$\\log M_h={lo:.2f}$–${hi:.2f}$\n"
                              r"$\Delta\log\Sigma_*$", fontsize=9)
            if i == 2:
                ax.set_xlabel("$R$ [kpc]")
        print(f"    {lo:.2f}-{hi:.2f} (n={int(sel.sum()):4d})"
              + "".join(f"{b:14.3f}" for b in slopes))

    fig.suptitle("exp52 — differential stellar-density growth by halo mass: "
                 "grey = individual galaxies, band = 16–84 per cent, "
                 "orange = median", fontsize=12)
    fig.tight_layout()
    for p in save_fig(fig, HERE / "figures" / "exp52_data_differential"):
        print(f"\n  wrote {p}")

    print("\n  median dlogSigma at 8 and 60 kpc, long baseline z=2.0 -> 0.4:")
    i8 = int(np.argmin(np.abs(mid - 8.0)))
    i60 = int(np.argmin(np.abs(mid - 60.0)))
    for lo, hi in bins:
        sel = (logmh >= lo) & (logmh < hi)
        d = lsig[sel, 0] - lsig[sel, 4]
        d = d[np.isfinite(d).all(axis=1)]
        m8, m60 = np.median(d[:, i8]), np.median(d[:, i60])
        print(f"    {lo:.2f}-{hi:.2f}:  8 kpc {m8:+.3f} (x{10**m8:.1f})   "
              f"60 kpc {m60:+.3f} (x{10**m60:.1f})")


if __name__ == "__main__":
    main("--dev" in sys.argv)
