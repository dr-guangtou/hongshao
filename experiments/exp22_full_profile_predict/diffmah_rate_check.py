"""exp22 aside — Does DiffMAH (a fit to the *cumulative* MAH) suffer the same
differencing instability as predicting-the-CoG-then-differentiating?

Q1 showed: predicting a cumulative profile and DIFFERENCING it (to get annuli /
the density) amplifies noise — near-cancellation of two large, close numbers.
DiffMAH fits the cumulative peak-mass history log M(t). So if we want the
accretion rate dM/dt, we differentiate it — same operation. Does it blow up?

Answer (demonstrated here): NO. DiffMAH is a SMOOTH 4-parameter analytic fit, so
its derivative is smooth and stable — the regularization happens at the fitting
step. The instability needs *noisy* data to difference; DiffMAH has already
removed the noise. Finite-differencing the RAW peak-mass history is jumpy (merger
steps); the DiffMAH-model rate is clean. (The flip side is information loss, not
blow-up: bursts are smoothed away — but exp21 showed that costs no profile skill.)

Run: PYTHONPATH=. uv run python experiments/exp22_full_profile_predict/diffmah_rate_check.py
"""
# %% setup
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

t = Table.read(TABLE); t = t[t["use"]]
idx = np.asarray(t["index"])
logmp = np.asarray(t["dmah_logmp"], float)
mah = load_mah(); tsnap = load_cosmic_time()

# three example halos at low / median / high final mass
order = np.argsort(logmp)
picks = [order[int(f * len(order))] for f in (0.1, 0.5, 0.9)]


def rate(tg, M):
    """dM/dt at bin midpoints (Msun/Gyr)."""
    return (tg[:-1] + tg[1:]) / 2.0, np.diff(M) / np.diff(tg)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.6))
for c, row in zip([OKABE_ITO[0], OKABE_ITO[4], OKABE_ITO[5]], picks):
    snaps, lmp = peak_history(mah[int(idx[row])])
    tg = tsnap[snaps.astype(int)]
    t0 = tg[-1]
    fit = fit_mah(tg, lmp, t0)
    tf = np.linspace(tg[1], tg[-1], 400)
    lm_smooth = log_mah(np.log10(tf), fit["logmp"], fit["logtc"], fit["early"],
                        fit["late"], np.log10(t0))
    # cumulative MAH: raw points vs smooth DiffMAH fit
    axA.plot(tg, lmp, "o", color=c, ms=3, alpha=0.7)
    axA.plot(tf, lm_smooth, "-", color=c, lw=1.8, label=f"logMp={fit['logmp']:.1f}")
    # accretion rate: finite-difference the RAW history vs the SMOOTH model
    tr, rr = rate(tg, 10.0 ** lmp)
    ts, rs = rate(tf, 10.0 ** lm_smooth)
    axB.plot(tr, np.clip(rr, 1e7, None), "o--", color=c, ms=3, lw=0.8, alpha=0.7)
    axB.plot(ts, rs, "-", color=c, lw=1.8)

axA.set_xlabel("cosmic time [Gyr]"); axA.set_ylabel(r"$\log M_{\rm peak}(<t)$ (cumulative)")
axA.set_title("A. DiffMAH fits the CUMULATIVE MAH (smooth)"); axA.legend(fontsize=8)
axB.set_yscale("log"); axB.set_xlabel("cosmic time [Gyr]"); axB.set_ylabel(r"$dM/dt$ [$M_\odot$/Gyr]")
axB.set_title("B. Accretion rate: raw finite-diff (jumpy) vs DiffMAH (smooth)")
axB.plot([], [], "o--", color="0.4", ms=3, label="finite-diff of raw MAH")
axB.plot([], [], "-", color="0.4", lw=1.8, label="derivative of DiffMAH fit")
axB.legend(fontsize=8)
fig.suptitle("DiffMAH fits the cumulative MAH, but its derivative is STABLE — it is a smooth "
             "4-param fit, so there is no noise to amplify (cf. Q1's differencing of noisy data)",
             fontsize=10)
fig.tight_layout()
save_fig(fig, FIGDIR / "exp22_diffmah_rate")
print("Verdict: DiffMAH fits the cumulative log M(t); differentiating it is STABLE because the "
      "fit is smooth/parametric (the noise was removed at the fit step). The Q1 instability needs "
      "noisy data to difference. DiffMAH's 'late' param already IS a denoised recent-accretion "
      "rate, so we never differentiate raw data.")
print(f"wrote figure -> {FIGDIR}/exp22_diffmah_rate")
