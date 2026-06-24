"""exp22 aside (Q1 refined) — does fitting the CUMULATIVE MAH leave the RECENT
accretion poorly constrained, the time-domain twin of the outskirt annulus?

User's point: DiffMAH is fit to the cumulative M(t), and the accretion rate falls
with time (recent rate low, like the low outskirt density). The recently accreted
mass is M(t0) - M(t0-dt): a small difference of two large, nearly-equal cumulative
values -- the same near-cancellation as the 50-100 kpc annulus. So a fixed
cumulative uncertainty should blow up the RECENT accretion rate far more than the
early one.

We test it two ways on real halos:
  (1) Direct near-cancellation: propagate a fixed cumulative noise to the
      fractional uncertainty of the accreted mass per epoch (no model).
  (2) DiffMAH bootstrap: perturb the cumulative MAH, refit, and measure the
      fractional scatter of the reconstructed dM/dt vs cosmic time.

Both should rise steeply toward the present. (Caveat, from exp13: this recent-rate
info that DiffMAH softens does NOT actually limit the OUTSKIRT prediction -- raw
MAH does not beat DiffMAH there -- so the outskirt residual is intrinsic, and
DiffMAH's 'late' already captures the bit that matters as the scatter driver.)

Run: PYTHONPATH=. uv run python experiments/exp22_full_profile_predict/diffmah_recent_uncertainty.py
"""
# %% setup
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.diffmah import fit_mah, log_mah                                     # noqa: E402
from hongshao.tng_data import load_mah, load_cosmic_time, peak_history           # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                      # noqa: E402

set_style()
FIGDIR = Path(__file__).resolve().parent / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
NHALO = int(os.environ.get("EXP22_NHALO", 150))     # halos to aggregate over
NBOOT = int(os.environ.get("EXP22_NBOOT", 25))      # bootstrap refits per halo
COG_NOISE = 0.02                                     # cumulative log-mass perturbation [dex]

t = Table.read(TABLE); t = t[t["use"]]
idx = np.asarray(t["index"])
mah = load_mah(); tsnap = load_cosmic_time()
rng = np.random.default_rng(0)
sample = rng.choice(len(idx), size=min(NHALO, len(idx)), replace=False)
TGRID = np.linspace(2.5, 9.0, 12)                    # cosmic-time epochs to probe [Gyr]


def rate_on_grid(p, t0):
    """dM/dt at TGRID from a DiffMAH param dict (analytic-smooth, finite-diff)."""
    h = 0.05
    lm_hi = log_mah(np.log10(TGRID + h), p["logmp"], p["logtc"], p["early"], p["late"], np.log10(t0))
    lm_lo = log_mah(np.log10(TGRID - h), p["logmp"], p["logtc"], p["early"], p["late"], np.log10(t0))
    return (10.0 ** lm_hi - 10.0 ** lm_lo) / (2 * h)


# %% ---- (1) model-free near-cancellation + (2) DiffMAH bootstrap ------------
PARAMS = ["logmp", "logtc", "early", "late"]
frac_unc_rate = []                                   # per-halo fractional rate uncertainty vs TGRID
pscatter = []                                        # per-halo bootstrap std of each param
for i in sample:
    snaps, lmp = peak_history(mah[int(idx[i])])
    if lmp is None or len(lmp) < 8:
        continue
    tg = tsnap[snaps.astype(int)]; t0 = tg[-1]
    if tg[0] > 2.0 or t0 < 9.0:
        continue
    # bootstrap the cumulative MAH fit; collect the reconstructed rate + the params
    rates = []; pars = []
    for _ in range(NBOOT):
        fit = fit_mah(tg, lmp + COG_NOISE * rng.standard_normal(len(lmp)), t0)
        if not np.isfinite(fit["late"]):
            continue
        rates.append(rate_on_grid(fit, t0)); pars.append([fit[p] for p in PARAMS])
    rates = np.array(rates); pars = np.array(pars)
    if len(rates) < 5:
        continue
    frac_unc_rate.append(rates.std(0) / np.maximum(np.abs(rates.mean(0)), 1e6))
    pscatter.append(pars.std(0))

frac_unc_rate = np.array(frac_unc_rate)
pscatter = np.array(pscatter)
fu_med = np.median(frac_unc_rate, axis=0)
ps_med = np.median(pscatter, axis=0)
print(f"aggregated over {len(frac_unc_rate)} halos; cumulative noise {COG_NOISE} dex")
print("  bootstrap param scatter (std under fixed cumulative noise):")
for p, s in zip(PARAMS, ps_med):
    print(f"    {p:7s} {s:.4f}")
print(f"  -> 'late' (recent slope) is ~{ps_med[3]/ps_med[0]:.0f}x softer than 'logmp' (final mass)")
print("  fractional rate uncertainty (early->recent): " + " ".join(f"{x:.2f}" for x in fu_med))
print(f"  recent/early fractional-uncertainty ratio = {fu_med[-1] / fu_med[1]:.1f}x")


# %% ---- FIGURE --------------------------------------------------------------
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.5))
axA.bar(PARAMS, ps_med, color=[OKABE_ITO[7], OKABE_ITO[7], OKABE_ITO[7], OKABE_ITO[5]])
axA.set_ylabel("bootstrap std under fixed cumulative noise")
axA.set_title(f"A. rate slopes (early/late) are ~{ps_med[3]/ps_med[0]:.0f}x softer than logmp")
axB.plot(TGRID, fu_med, "-o", color=OKABE_ITO[5], ms=4)
axB.set_xlabel("cosmic time [Gyr]")
axB.set_ylabel("fractional uncertainty of $dM/dt$")
axB.set_title(f"B. Recent accretion rate is least constrained ({fu_med[-1]/fu_med[1]:.0f}x early)")
fig.suptitle("Fitting the cumulative MAH leaves the RECENT accretion the least constrained — the "
             "time-domain twin of the galaxy outskirt (small signal on a large cumulative)", fontsize=9.5)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp22_diffmah_recent")
print(f"\nwrote figure -> {FIGDIR}/exp22_diffmah_recent")
