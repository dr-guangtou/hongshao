"""exp73 Block D, first use — does the new tier 2d SIZE GATE discriminate
something we already understand?

The gate (`hongshao.qa.size_gate`, the user's D1) is new. Before it judges a new
fit it is run on three models whose size behaviour is already known:

  exp63 joint         holds R50 to 0-5 per cent at every epoch (exp72 Step 3)
  chi2 refit (exp72)  makes z = 2 galaxies 27 per cent too big -- the failure
                      the size gate exists to catch
  nested incumbent    the null

If the gate passes the first, fails the second ON OFFSET at high redshift, and
says something sensible about the third, it is measuring what it claims to.

Two grids, because the user's density-based curve of growth reaches 0.673 kpc:
on the standard 24-radius grid (2-148 kpc) the truth's R20 falls inside the
innermost measured radius for compact high-z galaxies and is EXTRAPOLATED, which
the gate reports; on the merged 30-radius grid it is measured. The difference
between the two R20 rows is what the density profile buys for QA.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \\
    PYTHONPATH=. uv run python -u \\
        experiments/exp73_size_relative/size_gate_check.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term",
          ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp72_error_objective", HERE):
    sys.path.insert(0, str(p))

import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model2 as M2                                      # noqa: E402
import fit_re as FR                                      # noqa: E402
from hongshao import qa                                  # noqa: E402

RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
EXP72_FIT = ROOT / "experiments/exp72_error_objective/outputs/fit_gls.npz"


def main(smoke=False):
    print(f"{RULE}\nexp73 Block D — first use of the tier 2d SIZE GATE on three "
          f"known models\n{RULE}\n")
    (spec2, th_inc, th_nested, fz63, curves, truth_m, Rm, lmh0,
     rows, data, lmh, good) = FR.build(smoke)
    thetas = {"exp63 joint": np.asarray(fz63["theta_best"], float),
              "chi2 refit (exp72)": np.asarray(np.load(EXP72_FIT, allow_pickle=True)["theta_best"], float),
              "nested incumbent": th_nested}
    lmh_ep = lmh[good]                                    # (n, 5) DiffMAH per epoch
    for gname, R, truth in (("standard 24-radius grid, 2-148 kpc", F.R_GRID, data[good]),
                            ("merged 30-radius grid, 0.673-148 kpc", Rm, truth_m)):
        print(f"\n{RULE}\nGRID: {gname}\n{RULE}")
        for name, th in thetas.items():
            cogs = M2.predict2(spec2, th, curves, R, nodes=M2.FULL_NODES)
            ok = np.isfinite(cogs).all(axis=(1, 2)) & (cogs > 0).all(axis=(1, 2))
            print(f"\n  --- {name} ({int(ok.sum())} galaxies) ---")
            out = qa.evaluate(cogs[ok], truth[ok], R, ANCHOR_Z, name=name,
                              figdir=None, figures=False, verbose=False,
                              bin_by=lmh0[ok], halo_mass_epochs=lmh_ep[ok])
            qa.print_size_gate(out["size_gate_ms"], ANCHOR_Z, "STELLAR mass")
            qa.print_size_gate(out["size_gate_mh"], ANCHOR_Z, "HALO mass")
    print(f"\n  Read: exp63 should pass R50 everywhere; the chi2 refit should fail "
          f"R50 ON OFFSET at\n  z >= 1.5 (it is 22-27% too big there); and the R20 "
          f"row should carry an extrapolation\n  warning on the standard grid "
          f"that disappears on the merged one.")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
