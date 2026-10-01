"""exp87 Stage 0 — certificates and the frozen split. STOPS (exit code 1) unless
the record's numbers reproduce:

  1. the parent is the box: TNG300 centrals with M200c(z = 0.4) >= 10^13 number
     3388, the parent table's length;
  2. the parent's curve of growth equals the multi-epoch population's z = 0.4
     curve of growth on the shared galaxies;
  3. the two scatters of the record, in-sample and linear on the curated
     sample with M*(<103 kpc): sigma(Mh | M*) = 0.175 and sigma(M* | Mh) =
     0.140 dex, each within 0.01.

Then the lockbox (1 galaxy in 5) and the five development folds are written
once (`outputs/splits.npz`) and every sample's counts are printed.

Run:  PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/stage0_setup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import data as D                                         # noqa: E402

RULE = "=" * 100


def linear_scatter(x, y):
    """Residual standard deviation of the in-sample straight line y on x, and its slope."""
    ok = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return float(np.std(y[ok] - (slope * x[ok] + intercept))), float(slope)


def main():
    print(f"{RULE}\nexp87 STAGE 0 — certificates and the frozen split (config {C.config_hash()})\n{RULE}\n")
    cert = {"config": C.config_hash()}
    parent = D.load_parent()
    prior = D.box_prior()
    from astropy.table import Table
    n_table = len(Table.read(D.Z0P4_FITS))
    cert["box_centrals_above_cut"] = int(prior["n_above_cut"][0])
    cert["parent_table_rows"] = int(n_table)
    ok1 = cert["box_centrals_above_cut"] == C.CERT_PARENT_N == n_table
    print(f"  1. the parent is the box: {cert['box_centrals_above_cut']} box centrals with M200c >= 10^{C.PARENT_CUT:g} at snapshot "
          f"{C.ANCHOR_SNAP[0]}, {n_table} rows in the parent table, {len(parent)} with a finite curve of growth  "
          f"{'OK' if ok1 else 'FAIL'}")
    print("     box centrals above the per-epoch completeness cuts (z = 0.4 .. 2): "
          + " / ".join(str(int(v)) for v in prior["n_above_cut"]))

    curated = D.load_curated(0)
    pos = {int(g): i for i, g in enumerate(parent.index)}
    rows = np.array([pos[int(g)] for g in curated.index])
    diff = float(np.max(np.abs(parent.logcog[rows] - curated.logcog)))
    dmh = float(np.nanmax(np.abs(parent.targets["mh"][rows] - curated.targets["mh"])))
    cert["cog_identity_max_dex"], cert["mh_identity_max_dex"] = diff, dmh
    ok2 = diff < 1e-6 and dmh < 2e-3
    print(f"  2. the curve of growth is the same object in both tables: max |difference| {diff:.2e} dex over {len(rows)} shared "
          f"galaxies; halo mass max |difference| {dmh:.1e} dex  {'OK' if ok2 else 'FAIL'}")

    i103 = int(np.argmin(np.abs(curated.radii - 103.45)))
    s_inv, b_inv = linear_scatter(curated.logcog[:, i103], curated.targets["mh"])
    s_fwd, b_fwd = linear_scatter(curated.targets["mh"], curated.logcog[:, i103])
    cert.update(sigma_mh_given_mstar=s_inv, sigma_mstar_given_mh=s_fwd, slope_inverse=b_inv, slope_forward=b_fwd)
    ok3 = abs(s_inv - C.CERT_SIGMA_MH_GIVEN_MSTAR) <= C.CERT_TOL and abs(s_fwd - C.CERT_SIGMA_MSTAR_GIVEN_MH) <= C.CERT_TOL
    print(f"  3. the record's scatters on the curated sample, M*(<{curated.radii[i103]:.0f} kpc), in-sample straight lines: "
          f"sigma(Mh | M*) = {s_inv:.3f} dex (record {C.CERT_SIGMA_MH_GIVEN_MSTAR}; slope {b_inv:.2f}), "
          f"sigma(M* | Mh) = {s_fwd:.3f} dex (record {C.CERT_SIGMA_MSTAR_GIVEN_MH}; slope {b_fwd:.2f})  {'OK' if ok3 else 'FAIL'}")
    sp_inv, _ = linear_scatter(parent.logcog[:, i103], parent.targets["mh"])
    sp_fwd, _ = linear_scatter(parent.targets["mh"], parent.logcog[:, i103])
    cert.update(parent_sigma_mh_given_mstar=sp_inv, parent_sigma_mstar_given_mh=sp_fwd)
    print(f"     the same on the PARENT: sigma(Mh | M*) = {sp_inv:.3f}, sigma(M* | Mh) = {sp_fwd:.3f} dex "
          f"(the curated sample is tighter: its cuts depend on stellar mass at fixed halo mass)")

    print(f"\n  THE FROZEN SPLIT ({D.SPLIT_NPZ.name}, seed {C.SEED}): one galaxy in five to the lockbox, five development folds, "
          f"stratified in halo mass, shared by every sample and epoch")
    print(f"     parent: lockbox {int(parent.lockbox.sum())}, folds " + " / ".join(str(int((parent.fold == f).sum())) for f in range(C.N_FOLDS)))
    for k in range(5):
        s = D.load_curated(k, verbose=(k > 0))
        print(f"     curated z = {C.ANCHOR_Z[k]}: n {len(s)}, lockbox {int(s.lockbox.sum())}, folds "
              + " / ".join(str(int((s.fold == f).sum())) for f in range(C.N_FOLDS))
              + f"; halo mass min / median / max {np.nanmin(s.targets['mh']):.2f} / {np.nanmedian(s.targets['mh']):.2f} / {np.nanmax(s.targets['mh']):.2f}")
        cert[f"n_curated_{k}"] = len(s)
    cert["passed"] = bool(ok1 and ok2 and ok3)
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    (C.OUTDIR / "stage0_certificates.json").write_text(json.dumps(cert, indent=1))
    print(f"\n  STAGE 0 {'PASSED' if cert['passed'] else 'FAILED — STOP'}; wrote {C.OUTDIR / 'stage0_certificates.json'}")
    return 0 if cert["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
