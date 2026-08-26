"""exp58 P-D — the monotonicity compromise, measured per group.

The unified-reading claim is arithmetic: for a galaxy whose measured central
mass DECLINES between redshift 2 and 0.4, a monotone-in-time model's two
endpoint errors must sum to at least the decline, so the fit must choose where
to put the error. This prints the incumbent's actual choice, per group and per
epoch, at the centre (the 4.92 kpc aperture, exp54's standard central shell)
and at the amplitude anchor (103 kpc) — nothing else, so the claim rests on
numbers rather than on the story.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp58_reassessment/pd_decliner_split.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term"):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage4_target as S4                               # noqa: E402

ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
I5 = 3                     # the 4.92 kpc grid point, exp54's central shell
RULE = "=" * 96


def main():
    print(f"{RULE}\nexp58 P-D — where the monotone model puts the decliners' "
          f"unavoidable error\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    pred = slow.predict(theta)
    finite = np.isfinite(pred).all(axis=2) & (pred[:, :, F.I100] > 0)
    use = np.asarray(mask, bool) & finite & (data > 0).all(axis=2)
    fell, ok = S4.decline_flag(data)
    groups = (("declining centres", fell & ok),
              ("non-declining", (~fell) & ok))

    with np.errstate(invalid="ignore", divide="ignore"):
        e5 = np.log10(pred[:, :, I5] / data[:, :, I5])
        e100 = np.log10(pred[:, :, F.I100] / data[:, :, F.I100])
    e5 = np.where(use, e5, np.nan)
    e100 = np.where(use, e100, np.nan)

    for lab, err, r in (("M*(<4.92 kpc) — the centre", e5, 4.92),
                        ("M*(<103 kpc) — the amplitude", e100, 103.45)):
        print(f"  median log10(model/measured) {lab}")
        print(f"  {'group':<22}" + "".join(f"{f'z={z}':>10}" for z in ANCHOR_Z)
              + f"{'z2 err + z0.4 err':>20}")
        for gname, g in groups:
            row = [np.nanmedian(err[g, j]) for j in range(5)]
            print(f"  {gname:<22}" + "".join(f"{v:>+10.4f}" for v in row)
                  + f"{abs(row[4]) + abs(row[0]):>20.4f}")
        print()

    # the arithmetic floor itself, for the declining group
    dec = np.log10(data[:, 0, I5] / data[:, 4, I5])
    g = groups[0][1] & use[:, 0] & use[:, 4]
    print(f"  measured central decline (z=2 to 0.4) among decliners in the "
          f"fit sample:\n    median {np.nanmedian(-dec[g]):+.4f} dex over "
          f"{int(g.sum())} galaxies — the FLOOR on the sum of the model's two "
          f"central\n    endpoint errors for these galaxies, whatever the "
          f"parameters, in this model class.")
    # and the model's predicted change for both groups, vs the measured one
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = np.log10(pred[:, 0, I5] / pred[:, 4, I5])
    print(f"\n  central CHANGE z=2 -> 0.4, median [dex]:")
    print(f"  {'group':<22}{'measured':>12}{'model':>12}")
    for gname, gg in groups:
        m2 = gg & use[:, 0] & use[:, 4]
        print(f"  {gname:<22}{np.nanmedian(dec[m2]):>+12.4f}"
              f"{np.nanmedian(dm[m2]):>+12.4f}")


if __name__ == "__main__":
    main()
