"""exp74 addendum — the 2 x 2 re-scored with the binned term's halo-mass bins
defined by the MEASURED (catalog) mass instead of the official DiffMAH mass.

exp63's objective bins galaxies into halo-mass terciles by the DiffMAH curve's
mass at each epoch. The honest model does not use that curve, so scoring it
in those bins is scoring it on the official model's variable. Here both models
are scored with terciles by the measured M200c(z_k) (NaN filled from DiffMAH
for the few per cent without a catalog entry), references rebuilt at the
nested incumbent on the official curves in the same bins.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", ROOT / "experiments/exp63_analytic_growth", HERE):
    sys.path.insert(0, str(p))
import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import selection as SEL                                  # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage2_fit as S2F                                 # noqa: E402
import history as HI                                     # noqa: E402
from stage1_refit import PreEpochJoint, FIT_NPZ, OUTDIR  # noqa: E402

ANCHOR_Z = list(E.ANCHOR_Z)
recs, data, mask_legacy, lmh_dm, spec_inc, th_inc, _ = S0.build(False)
mask, _ = S2F.fit_masks(data, mask_legacy, lmh_dm, "sane")
curves = E.build_curves(recs, verbose=False)
fz = np.load(FIT_NPZ, allow_pickle=True); spec2 = S2F.spec_from_fit(fz)
th63 = np.asarray(fz["theta_best"], float)
thr = np.asarray(np.load(OUTDIR / "stage1_refit.npz", allow_pickle=True)["theta_best"], float)
pre = {k: HI.load_curves(curves, k) for k in range(5)}
hs = np.load(SEL.HS_NPZ, allow_pickle=True)
rows = np.array([h.row for h in recs])
lmh_cat = SEL.sample_masses(hs)[rows]
filled = np.where(np.isfinite(lmh_cat), lmh_cat, lmh_dm)
print(f"  catalog mass missing for {int((~np.isfinite(lmh_cat)).sum())} galaxy-epochs, filled from DiffMAH")
for lab, lmh in (("bins by DiffMAH mass (exp63's)", lmh_dm), ("bins by MEASURED mass", filled)):
    pr = PreEpochJoint(spec2, curves, pre, data, mask, lmh, F.R_GRID, th_inc, binned=True)
    print(f"\n  {lab}")
    print(f"  {'theta / curves':<28}" + "".join(f"{f'z={z}':>9}" for z in ANCHOR_Z) + f"{'total':>10}")
    for tl, th in (("exp63", th63), ("exp74 refit", thr)):
        for cl in ("official", "pre-epoch"):
            per = (S2F.JointProblem2.per_epoch(pr, th, nodes=M2.FULL_NODES) if cl == "official"
                   else pr.per_epoch(th, nodes=M2.FULL_NODES))
            row = np.array([sum(x * x for x in (per[k][0], per[k][1], per[k][2], per[k][4])) for k in pr.epochs])
            print(f"  {tl + ' on ' + cl:<28}" + "".join(f"{v:>9.3f}" for v in row) + f"{row.sum():>10.3f}")
