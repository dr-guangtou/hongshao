"""Sanity-check figure for the assembled TNG300 z=0.4 dataset.

Confirms the data is loadable and physically sensible before experiment 1.

    PYTHONPATH=. uv run python scripts/qc_overview.py \
        --table data/processed/tng300_072_z0p4.fits --out figures/qc_overview.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

from hongshao.tng_data import COG_RAD_KPC, SMA_KPC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="data/processed/tng300_072_z0p4.fits")
    ap.add_argument("--out", default="figures/qc_overview.png")
    args = ap.parse_args()

    t = Table.read(args.table)
    use = t[t["use"]]
    print(f"loaded {len(t)} rows; {len(use)} pass the 'use' cut")

    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    # (1) M0 distribution
    ax[0, 0].hist(use["logm0_halo"], bins=40, color="0.4")
    ax[0, 0].set(xlabel=r"$\log_{10} M_{\rm peak}(z{=}0.4)\ [M_\odot]$",
                 ylabel="N", title="Halo mass distribution")

    # (2) CoG examples (20 random galaxies)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(use), size=20, replace=False):
        ax[0, 1].plot(COG_RAD_KPC, use["logmstar_cog"][i], lw=0.8, alpha=0.6)
    ax[0, 1].set(xscale="log",
                 xlabel="R [kpc]", ylabel=r"$\log_{10} M_\star(<R)\ [M_\odot]$",
                 title="Curve-of-growth examples")

    # (3) Aperture-mass SHMR: Mstar(<100 kpc) vs M0
    j100 = int(np.where(SMA_KPC == 100)[0][0])
    ax[1, 0].scatter(use["logm0_halo"], use["logmstar_aper"][:, j100],
                     s=4, alpha=0.3, color="C0")
    ax[1, 0].set(xlabel=r"$\log_{10} M_{\rm peak}(z{=}0.4)$",
                 ylabel=r"$\log_{10} M_\star(<100\,{\rm kpc})$",
                 title="Aperture-mass SHMR (sanity check)")

    # (4) MAH diversity at fixed M0: Mpeak(z)/M0 in a narrow mass bin
    z_anchor = np.array([0.4, 0.7, 1.0, 1.5, 2.0])
    cols = ["logmpeak_z0p4", "logmpeak_z0p7", "logmpeak_z1",
            "logmpeak_z1p5", "logmpeak_z2"]
    sel = use[(use["logm0_halo"] > 13.3) & (use["logm0_halo"] < 13.5)]
    for row in sel[:60]:
        frac = [10 ** (row[c] - row["logm0_halo"]) for c in cols]
        ax[1, 1].plot(z_anchor, frac, lw=0.7, alpha=0.5, color="C3")
    ax[1, 1].set(xlabel="z", ylabel=r"$M_{\rm peak}(z)/M_{\rm peak}(z{=}0.4)$",
                 title=f"MAH diversity at fixed $M_0$ (n={len(sel)})", yscale="log")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
