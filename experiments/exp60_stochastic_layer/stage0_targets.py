"""exp60 Stage 0 — freeze the stochastic layer's targets. Run once; the gates
never read the data again.

Everything gate S1/S1b/S5 of the plan will judge against is computed here, on
exp57's sample, and stored: the central-evolution distribution at 4.92 kpc
over z=2→0.4 (overall and per formation-time quartile, for the tilt test),
and the incumbent's amplitude-residual scatter with its cross-epoch
correlation (what the amplitude draw must reproduce). Also stored, because
the layer must NOT disturb them: the incumbent's median CoG per epoch (gate
S3) and its G1b outskirt shares (gate S4 reads exp57's machinery at judging
time; the medians here are the S3 reference).

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp60_stochastic_layer/stage0_targets.py
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

I5 = 3                       # R_GRID[3] = 4.92 kpc, the central shell
OUT = HERE / "outputs" / "stage0_targets.npz"
RULE = "=" * 96


def main():
    print(f"{RULE}\nexp60 STAGE 0 — the frozen targets\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    n = len(recs)
    pred = slow.predict(theta)
    finite = np.isfinite(pred).all(axis=2) & (pred[:, :, F.I100] > 0)
    use = np.asarray(mask, bool) & finite & (data > 0).all(axis=2)

    # --- S1: the central-evolution distribution --------------------------- #
    fell, ok = S4.decline_flag(data)
    with np.errstate(invalid="ignore", divide="ignore"):
        dc = np.log10(data[:, 0, I5] / data[:, 4, I5])       # z=2 -> 0.4
    g = ok & np.isfinite(dc)
    frac = float((dc[g] < 0).mean())
    med_all = float(np.median(dc[g]))
    med_dec = float(np.median(dc[g & (dc < 0)]))
    p10, p90 = (float(v) for v in np.percentile(dc[g], [10, 90]))
    print(f"  S1 — measured central change at 4.92 kpc, z=2 -> 0.4 "
          f"({int(g.sum())} galaxies):")
    print(f"    declining fraction {100 * frac:.1f}%   overall median "
          f"{med_all:+.4f} dex   decliner median {med_dec:+.4f} dex")
    print(f"    10th-90th percentile {p10:+.4f} to {p90:+.4f} dex")

    # per formation-time quartile, for the mixture-tilt test
    fform = np.array([h.f_form[-1] for h in recs])
    qs = np.nanpercentile(fform[g], [25, 50, 75])
    edges = [-np.inf, *qs, np.inf]
    q_frac, q_med = [], []
    print(f"    per f_form(z=0.4) quartile (earliest-assembled first):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = g & (fform > lo) & (fform <= hi)
        q_frac.append(float((dc[m] < 0).mean()))
        q_med.append(float(np.median(dc[m])))
        print(f"      fraction declining {100 * q_frac[-1]:5.1f}%   "
              f"median {q_med[-1]:+.4f} dex   (n={int(m.sum())})")

    # --- S5: the amplitude residual's scatter and coherence --------------- #
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.log10(pred[:, :, F.I100] / data[:, :, F.I100])
    a = np.where(use, a, np.nan)
    sig = np.array([(np.nanpercentile(a[:, k], 84)
                     - np.nanpercentile(a[:, k], 16)) / 2 for k in range(5)])
    full = np.isfinite(a).all(axis=1)
    corr = np.corrcoef(a[full].T)
    print(f"\n  S5 — the incumbent's amplitude residual (what the draw must "
          f"reproduce):")
    print(f"    per-epoch scatter (half 16-84) [dex]: "
          + " ".join(f"{v:.4f}" for v in sig))
    print(f"    cross-epoch correlation ({int(full.sum())} galaxies with all "
          f"five epochs):")
    for j in range(5):
        print(f"      " + " ".join(f"{corr[j, i]:+.3f}" for i in range(5)))

    # --- S3 reference: the incumbent's median CoG per epoch --------------- #
    with np.errstate(invalid="ignore", divide="ignore"):
        med_cog = np.nanmedian(np.where(use[:, :, None],
                                        np.log10(np.clip(pred, 1.0, None)),
                                        np.nan), axis=0)

    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, frac_decline=frac, med_all=med_all, med_dec=med_dec,
             p10=p10, p90=p90, q_frac=np.array(q_frac), q_med=np.array(q_med),
             q_edges=np.array(qs), amp_sigma=sig, amp_corr=corr,
             med_cog_incumbent=med_cog, n_central=int(g.sum()))
    print(f"\n  frozen -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
