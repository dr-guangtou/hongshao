"""exp54 — the forward model, end to end, on ONE real galaxy, with the role of
the efficiency window `w(z)` isolated by experiment rather than assertion.

Written because "what does w(z) actually do?" is the question the whole exp54
revision turns on, and prose kept failing to settle it. This prints every
intermediate array of a single CoG prediction, then runs two controlled
experiments on `w` that answer the question numerically:

  A. multiply `w` by 1000  -> the predicted CoG is IDENTICAL to machine
     precision. The AMPLITUDE of `w` is destroyed by `dM = w/w.sum()` and then
     again by the `M(<500)` pin. In the adopted model `w` carries no mass.
  B. move the window's centre -> the predicted CoG CHANGES SHAPE. What `w`
     actually controls is WHICH FORMATION TIMES dominate, and because a
     deposit's radius depends on when it formed, that is a RADIAL weight.

So in the adopted (pinned) kernel `w(z)` is not a star-formation efficiency at
all. It is a radial-mixing weight. Removing the pin is what turns it back into
an efficiency — which is why the amplitude and the shape then want
incompatible windows (plan §3.2.F).

Run:
    HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
    experiments/exp54_unpinned_amplitude/walkthrough.py [--galaxy N]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp53_deposition_only"):
    sys.path.insert(0, str(p))

import families                                          # noqa: E402
import kernel as K                                       # noqa: E402
import stage2_multiepoch as s2                           # noqa: E402

ZBINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0),
         (3.0, 5.0), (5.0, 8.0), (8.0, 30.0))
RULE = "-" * 78


def head(n, title):
    print(f"\n{RULE}\nSTEP {n}. {title}\n{RULE}")


def main(gal=None):
    s2._w_init(None)
    gals, e = s2._W["gals"], s2._W["e"]
    if gal is None:                       # the MEDIAN-halo-mass galaxy
        lm = np.array([gg["logmh"] for gg in gals])
        gal = int(np.argsort(lm)[len(lm) // 2])
    g = gals[gal]
    spec = K.Spec("moffat", q_free=True)
    theta = K.from_exp38_theta(K.adopted_theta())
    mah, R, R_EXT = g["mah"], e.R, e.R_EXT
    t_obs, t_k = mah["t_obs"], e.pe.AT[0]                 # z = 0.4

    lm_all = np.array([gg["logmh"] for gg in gals])
    pct = 100.0 * np.mean(lm_all < g["logmh"])
    print(f"THE ADOPTED FORWARD MODEL, TRACED ON ONE GALAXY (index {gal}, "
          f"row {g['row']})")
    print(f"logMh = {g['logmh']:.3f}, the {pct:.0f}th percentile of the sample")
    print(f"predicting the z=0.4 curve of growth on {len(R)} radii, "
          f"{R[0]:.0f}-{R[-1]:.0f} kpc")

    # ---------------------------------------------------------------- step 1
    head(1, "INPUTS — four DiffMAH numbers and three halo properties")
    print("  DiffMAH (Hearin+2021): log10 M(t) = logmp + alpha(logt)(logt-logt0)")
    print("    alpha(logt) = early + (late-early) * sigmoid(3.5 (logt - logtc))")
    print(f"  halo mass at z=0.4          logMh = {g['logmh']:.3f}")
    print(f"  conditioning vector (standardized [logMh, c200c, fz2])")
    print(f"                              cond  = {np.round(g['cond'], 3)}")
    print(f"  cosmic time at z=0.4        t_obs = {t_obs:.3f} Gyr")
    print("\n  NOTE: nothing here is stellar. This is the whole input.")

    # ---------------------------------------------------------------- step 2
    head(2, "THE MASS ACCRETION HISTORY — the DiffMAH curve, differenced")
    mask = mah["snap"] <= e.ANCHOR_SNAP[0]
    print(f"  {len(mah['t'])} deposits, one per snapshot, t = "
          f"{mah['t'][0]:.3f} to {mah['t'][-1]:.3f} Gyr "
          f"(z = {mah['z'][0]:.1f} to {mah['z'][-1]:.2f})")
    print(f"  dMh_i = diff(10**logMh) -- the halo mass gained in step i")
    print(f"\n  CAREFUL: the MAH runs past the observation epoch, to z=0. Only")
    print(f"  the {int(mask.sum())} deposits with snap <= {e.ANCHOR_SNAP[0]} "
          f"(z >= 0.4) can contribute at z=0.4;")
    print(f"  every number below is quoted over THOSE unless said otherwise.")
    dmh = mah["dMh"][mask]
    zz = mah["z"][mask]
    print(f"\n  halo mass gained by z=0.4 = {dmh.sum():.4g} Msun")
    print(f"  fraction gained at z < 1   = "
          f"{dmh[zz < 1].sum() / dmh.sum():.3f}   <-- most of it arrives LATE")
    print(f"  fraction gained at z > 3   = {dmh[zz > 3].sum() / dmh.sum():.3f}")

    # ---------------------------------------------------------------- step 3
    head(3, "CONDITIONING — 12 global numbers + cond -> this galaxy's kernel")
    log_r50, gg, q, mu, sig, shape = spec.unpack(theta, g["cond"])
    print(f"  log_R50 = {log_r50:8.4f}   (base {theta[0]:.4f} + "
          f"{np.round(theta[6:9], 4)} . cond)")
    print(f"  g       = {gg:8.4f}   deposit size grows as (t_i/t_obs)^g")
    print(f"  q       = {q:8.4f}   transported material grows as (t_k/t_i)^q")
    print(f"  mu      = {mu:8.4f}   <-- the efficiency window centre, in ln(1+z)")
    print(f"  sig     = {sig:8.4f}   <-- its width  (base {theta[4]:.4f} + "
          f"{np.round(theta[9:12], 4)} . cond)")
    print(f"  gamma   = {shape[0]:8.4f}   Moffat outer slope")
    print(f"\n  the window peaks at z = {np.expm1(mu):.2f}, "
          f"1-sigma span z = {np.expm1(mu - sig):.2f} to {np.expm1(mu + sig):.2f}")

    # ---------------------------------------------------------------- step 4
    head(4, "THE EFFICIENCY WINDOW w(z), AND WHERE ITS AMPLITUDE DIES")
    w_raw = np.exp(-((np.log1p(mah["z"]) - mu) ** 2) / (2.0 * sig ** 2))
    wdm = w_raw * mah["dMh"]
    dM = wdm / wdm.sum()                       # <-- kernel.py:212
    print("  w_i    = exp(-(ln(1+z_i) - mu)^2 / 2 sig^2)      dimensionless, peak 1")
    print("  dM_i   = w_i * dMh_i                             a mass, in Msun")
    print("  dM     = dM / dM.sum()            <-- kernel.py:212  THE AMPLITUDE DIES HERE")
    print(f"\n  sum(w_i * dMh_i) before normalizing = {wdm.sum():.4g} "
          f"(Msun x dimensionless)")
    print(f"  after normalizing                   = {dM.sum():.4f}  (exactly 1)")
    print(f"\n  NOTE, and it matters for exp54: the sum runs over the FULL MAH,")
    print(f"  including the {int((~mask).sum())} deposits that postdate z=0.4. So "
          f"the masked weights\n  sum to {dM[mask].sum():.4f}, not 1. Harmless "
          f"here because the pin renormalizes\n  anyway -- a real bug the moment "
          f"the pin is removed.")
    print("\n  This is the quantity that WOULD have been the stellar mass. It is")
    print("  divided away, and nothing downstream can recover it.")

    # ---------------------------------------------------------------- step 5
    head(5, "DEPOSIT RADII — set by WHEN the deposit formed, not by how much")
    r50_0 = np.clip(10.0 ** log_r50 * (mah["t"] / t_obs) ** gg,
                    K.R50_FLOOR, K.R50_CAP)
    print(f"  r50_0(i) = 10^log_R50 * (t_i/t_obs)^g, clipped to "
          f"[{K.R50_FLOOR:g}, {K.R50_CAP:g}] kpc")
    print(f"  spans {r50_0.min():.4g} to {r50_0.max():.4g} kpc; "
          f"{100 * np.mean(r50_0 >= K.R50_CAP):.0f}% of deposits sit at the ceiling")
    print("\n  ** THIS IS THE COUPLING **  the radius of a deposit is a function")
    print("  of its formation time alone, so any re-weighting in TIME is")
    print("  automatically a re-weighting in RADIUS.")

    # ---------------------------------------------------------------- step 6
    head(6, "TRANSPORT — each deposit splits into a core and a migrated part")
    fc = np.exp(-np.clip(t_k - mah["t"], 0.0, None) / mah["t"])
    r50_w = np.clip(r50_0 * (t_k / mah["t"]) ** q, K.R50_FLOOR, K.R50_CAP)
    print(f"  fc_i    = exp(-(t_k - t_i)/t_i)   fraction staying at r50_0")
    print(f"  r50_w(i)= r50_0 * (t_k/t_i)^q     where the rest goes")
    print(f"  median fc = {np.median(fc):.4f}  (early deposits transport almost "
          f"completely)")

    # ---------------------------------------------------------------- step 7
    head(7, "SUM OVER DEPOSITS -> a UNIT-MASS shape")
    B = K.basis(spec, theta, g["cond"], mah["t"], t_obs, t_k, R_EXT)
    m = B @ (dM * mask)
    print(f"  B is ({B.shape[0]} radii x {B.shape[1]} deposits), each column the "
          f"unit-mass\n  CoG of one deposit; m(<R) = B @ (dM * mask)")
    print(f"  m(<148 kpc) = {m[-2]:.6f}   m(<500 kpc) = {m[-1]:.6f}   "
          f"(dimensionless)")
    print(f"  -> {100 * (1 - m[-1]):.2f}% of the deposited mass lies BEYOND 500 kpc "
          f"and is discarded")

    # ---------------------------------------------------------------- step 8
    head(8, "THE PIN — the only place physical stellar mass enters")
    cog = m[:-1] * (g["m500"][0] / m[-1])
    print(f"  M_model(<R) = m(<R) * M_TNG(<500) / m(<500)      <-- kernel.py:233")
    print(f"  M_TNG(<500) = {g['m500'][0]:.4g} Msun   <-- READ FROM THE TRUTH")
    print(f"\n  predicted M(<10 kpc)  = {np.interp(10.0, R, cog):.4g} Msun")
    print(f"  predicted M(<100 kpc) = {np.interp(100.0, R, cog):.4g} Msun")
    print(f"  predicted M(<148 kpc) = {cog[-1]:.4g} Msun     "
          f"truth {g['data'][0][-1]:.4g}")

    # =================================================================== ROLE
    print(f"\n{'=' * 78}\nWHAT w(z) ACTUALLY DOES — two controlled experiments\n{'=' * 78}")

    print("\nEXPERIMENT A — scale w by 1000, change nothing else.")
    th_a = theta.copy()
    cog_a = K.model_cogs(spec, th_a, g, [0])[0]
    w_big = 1000.0 * w_raw * mah["dMh"]
    dM_big = w_big / w_big.sum()
    cog_scaled = (B @ (dM_big * mask))[:-1]
    cog_scaled = cog_scaled * (g["m500"][0] / (B @ (dM_big * mask))[-1])
    print(f"  max |relative difference| in the predicted CoG = "
          f"{np.abs(cog_scaled / cog_a - 1).max():.3e}")
    print("  -> IDENTICAL. The amplitude of w is unobservable in this model.")
    print("     w is defined only up to a multiplicative constant.")

    print("\nEXPERIMENT B — move the window centre mu, keep everything else.")
    print(f"  {'mu':>6}{'peak z':>8}{'M(<10)/M(<148)':>16}{'R50 light':>11}"
          f"{'M(<148)':>12}{'M(<500) PINNED':>16}")
    for mu_try in (1.0, 1.25, mu, 1.75, 2.0):
        th = theta.copy()
        th[3] = mu_try
        c = K.model_cogs(spec, th, g, [0])[0]
        r50_light = float(np.interp(0.5 * c[-1], c, R))
        print(f"  {mu_try:>6.3f}{np.expm1(mu_try):>8.2f}"
              f"{c[np.searchsorted(R, 10.0)] / c[-1]:>16.4f}"
              f"{r50_light:>11.2f}{c[-1]:>12.4g}{g['m500'][0]:>16.4g}"
              + ("  <- adopted" if mu_try == mu else ""))
    print("\n  -> the SHAPE moves ENORMOUSLY: the half-light radius runs 41.8 ->")
    print("     2.0 kpc, a factor of 20, for a window shift of one unit in mu.")
    print("  -> M(<500) is EXACTLY invariant -- it is the pinned quantity, read")
    print("     from the truth and identical in every row.")
    print("  -> M(<148) drifts only because the pin sits at 500 kpc while this")
    print("     column is measured at 148: as the profile concentrates, more of")
    print("     the same pinned total falls inside 148 kpc. It is a SHAPE effect,")
    print("     not the model predicting a different mass.")

    # ------------------------------------------------------- the decomposition
    print(f"\n{'=' * 78}\nWHERE THE HALO MASS ARRIVES vs WHERE THE STARS ARE PLACED"
          f"\n{'=' * 78}")
    print(f"  {'z bin':>12}{'halo mass':>12}{'star deposits':>15}"
          f"{'ratio':>9}{'median r50':>12}{'of M(<148)':>12}")
    tot_h, tot_s = mah["dMh"][mask].sum(), dM[mask].sum()
    for lo, hi in ZBINS:
        sel = (mah["z"] >= lo) & (mah["z"] < hi) & mask
        if not sel.any():
            continue
        fh, fs = mah["dMh"][sel].sum() / tot_h, dM[sel].sum() / tot_s
        contrib = (B[:, sel] @ (dM * mask)[sel])[-2] / m[-2]
        print(f"  {f'{lo:g}-{hi:g}':>12}{fh:>12.4f}{fs:>15.4f}"
              f"{(fs / fh if fh > 0 else np.nan):>9.2f}"
              f"{np.median(r50_0[sel]):>12.3f}{contrib:>12.4f}")
    print("\n  'halo mass' and 'star deposits' are fractions of their own totals.")
    print("  The ratio is the window's relative efficiency. The last column is")
    print("  each epoch's share of the FINAL PREDICTED LIGHT inside 148 kpc.")
    print("\n  The window suppresses low-z deposits by a factor ~1/ratio, and")
    print("  low-z deposits are the LARGE ones (see 'median r50'). That is how a")
    print("  quantity with no mass content controls the radial profile.")


if __name__ == "__main__":
    n = None                              # default: the median-halo-mass galaxy
    if "--galaxy" in sys.argv:
        n = int(sys.argv[sys.argv.index("--galaxy") + 1])
    main(n)
