"""exp29 — deposition kernel on a dip-free MAH, tested against the *multi-epoch*
profiles. Headline: a **centred Gaussian** already reproduces inside-out growth;
the handover's "outer-weighted shell" is unnecessary and counterproductive.

Two changes from exp25 (2026-06-27 handover):
  1. **Dip-free MAH.** Drop the dippy main-branch ``peak_history`` (merger-tree
     drop-outs) for the smooth official **DiffMAH fit** curve (same quantity —
     main-branch SubhaloMass Mpeak — but monotonic, all galaxies). ``dipfree_mah``.
  2. **Outer-weighted primitive.** ``deposit.py`` generalizes the centred Gaussian
     to a one-parameter family; ``p`` slides the deposit peak from R=0 (p=0,
     Gaussian) out to R=sigma*sqrt(p) (a shell).

The test is the **stacked** differential profile (exp26): for each adjacent epoch
pair and the long baseline (z=2->0.4), the inside-out slope ``b`` of
``log Sigma_lo - log Sigma_hi ~ b log R``. We integrate the deposits only to each
t(z_k), stack the differential over ~2400 galaxies, and compare model vs data b,
restricted to the radii where the data are valid (no floor artifact).

Efficiency (the temporal budget) is fixed at the exp25 population median — this is
a *spatial* test. Only the width-growth ``g`` is fit (to the long-baseline b);
``p`` is then scanned to show p>0 makes it worse.

RESULT: with sigma(t) = sigma_0 (t/t_obs)^0.55 the **centred Gaussian** matches the
whole b(z) trend (data 0.13/0.14/0.27/0.33/0.82 -> model 0.11/0.13/0.28/0.31/0.82);
p=2,4 monotonically undershoot. The forward model never deposits one Gaussian — it
sums Gaussians whose *width grows with cosmic time*, so the cumulative added mass is
outer-weighted (b~0.8) even though every primitive is centred. The exp26 "not a
centred Gaussian" result was about a single deposit; sigma(t) is the inside-out
mechanism, not the deposit's own off-centredness.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/run.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from deposit import deposit_sigma, width_t, eff_two_epoch, deposited                # noqa: E402
from hongshao.tng_data import load_mah, peak_history, _time_to_redshift             # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                        # noqa: E402
from hongshao.provenance import write_manifest                                      # noqa: E402

set_style()
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
OFFICIAL = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/official_mah.npz"
DIFFPROF = ROOT / "experiments/exp26_differential_profiles/outputs/differential_profiles.npz"
ANCHOR_SNAP = [72, 59, 50, 40, 33]             # exp26 sigma axis: z=0.4,0.7,1.0,1.5,2.0
ANCHOR_Z = [0.4, 0.7, 1.0, 1.5, 2.0]
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]   # 4 adjacent + long baseline
PAIR_LABEL = ["0.4-0.7", "0.7-1.0", "1.0-1.5", "1.5-2.0", "2.0->0.4"]
RES_LO, RES_HI = 6.0, 100.0
EFF_MEDIAN = (3.6, 1.15, 2.4)                  # exp25 population-median two-epoch efficiency
S0_FIXED = 60.0                                # b(z) depends on g & p, not the overall size s0


# %% ---- dip-free MAH from the official DiffMAH fit curve --------------------
def dipfree_mah(gi, _cache={}):
    """Smooth DiffMAH-fit MAH for galaxy index ``gi``: per-deposit snap/t/z/dMh.

    Same physical quantity as the dippy ``peak_history`` (main-branch SubhaloMass
    Mpeak, h-free Msun) but monotonic by construction. None if unavailable."""
    if "mz" not in _cache:
        _cache["mz"] = np.load(OFFICIAL); _cache["zt"] = _time_to_redshift()
    mz = _cache["mz"]; hit = np.where(mz["index"] == gi)[0]
    if hit.size == 0:
        return None
    lm = mz["diffmah_log_mah_fit"][int(hit[0])]; m = np.isfinite(lm)
    snap = np.arange(lm.size)[m]; tg = mz["cosmic_time_gyr"][m]; lm = lm[m]
    if 72 not in snap:
        return None
    ta, za = _cache["zt"]
    return dict(snap_full=snap, logMh_full=lm, t_full=tg, snap=snap[1:], t=tg[1:],
                z=np.interp(tg[1:], ta, za), dMh=np.clip(np.diff(10.0 ** lm), 0.0, None),
                t_obs=float(tg[snap == 72][0]))


# %% ---- load the population once (dip-free MAH + cached 5-epoch profiles) ----
def load_population():
    t = Table.read(TABLE); cog = np.asarray(t["logmstar_cog"], float)
    idx = np.asarray(t["index"]); use = np.asarray(t["use"])
    cogmap = {int(g): 10.0 ** float(cog[k, -1]) for k, g in enumerate(idx) if np.isfinite(cog[k, -1])}
    ok = np.isfinite(cog).all(1) & use
    bcg = int(idx[int(np.argmax(np.where(ok, cog[:, -1], -np.inf)))])

    mz = np.load(OFFICIAL); dp = np.load(DIFFPROF)
    R = dp["R"]; tg = mz["cosmic_time_gyr"]; snap = np.arange(100)
    ta, za = _time_to_redshift()
    logmah = mz["diffmah_log_mah_fit"]; mzm = {int(g): r for r, g in enumerate(mz["index"])}
    Sigall = dp["sigma"]; flall = dp["flags"]; dpm = {int(g): r for r, g in enumerate(dp["index"])}

    gals = {}
    for i in np.where(ok)[0]:
        gi = int(idx[i])
        if gi not in mzm or gi not in dpm or gi not in cogmap:
            continue
        lm = logmah[mzm[gi]]; m = np.isfinite(lm)
        if 72 not in snap[m]:
            continue
        sd, td = snap[m][1:], tg[m][1:]
        gals[gi] = dict(snap=sd, t=td, z=np.interp(td, ta, za),
                        dMh=np.clip(np.diff(10.0 ** lm[m]), 0.0, None),
                        t_obs=float(tg[m][snap[m] == 72][0]), Mstar_tot=cogmap[gi],
                        Sig=Sigall[dpm[gi]], flags=flall[dpm[gi]])
    return gals, R, bcg


# %% ---- forward model + stacked differential slope --------------------------
def model_sigma_epochs(g_, s0, gexp, p, R):
    """Predicted Sigma(R) at the 5 anchor snaps for one galaxy dict ``g_``."""
    dMstar = deposited(eff_two_epoch(g_["z"], *EFF_MEDIAN), g_["dMh"], g_["Mstar_tot"])
    sigma = width_t(s0, gexp, g_["t"], g_["t_obs"])
    return {sa: deposit_sigma(dMstar[g_["snap"] <= sa], sigma[g_["snap"] <= sa], p, R)
            for sa in ANCHOR_SNAP}


def stacked_dlog(gals, R, s0, gexp, p):
    """Median model & data Delta-log-Sigma per pair (over valid radii) and slopes b."""
    res = (R >= RES_LO) & (R <= RES_HI)
    accm = {k: [] for k in PAIRS}; accd = {k: [] for k in PAIRS}
    for g_ in gals.values():
        Sm = model_sigma_epochs(g_, s0, gexp, p, R); Sig, fl = g_["Sig"], g_["flags"]
        for lo, hi in PAIRS:
            ok = fl[lo] & fl[hi] & (Sig[lo] > 0) & (Sig[hi] > 0)     # same valid radii both sides
            ml = np.log10(np.clip(Sm[ANCHOR_SNAP[lo]], 1e-3, None)) - \
                np.log10(np.clip(Sm[ANCHOR_SNAP[hi]], 1e-3, None))
            dl = np.log10(Sig[lo]) - np.log10(Sig[hi])
            accm[(lo, hi)].append(np.where(ok, ml, np.nan))
            accd[(lo, hi)].append(np.where(ok, dl, np.nan))
    mstack, dstack, bm, bd = {}, {}, {}, {}
    for k in PAIRS:
        mstack[k] = np.nanmedian(np.array(accm[k]), 0); dstack[k] = np.nanmedian(np.array(accd[k]), 0)
        v = res & np.isfinite(mstack[k]) & np.isfinite(dstack[k])
        bm[k] = float(np.polyfit(np.log10(R[v]), mstack[k][v], 1)[0])
        bd[k] = float(np.polyfit(np.log10(R[v]), dstack[k][v], 1)[0])
    return dict(R=R, mstack=mstack, dstack=dstack, b_model=bm, b_data=bd)


def fit_g(gals, R):
    """One-parameter fit: width-growth g matching the long-baseline data b (p=0)."""
    def loss(gexp):
        s = stacked_dlog(gals, R, S0_FIXED, gexp, 0.0)
        return (s["b_model"][(0, 4)] - s["b_data"][(0, 4)]) ** 2
    return float(minimize_scalar(loss, bounds=(0.0, 3.0), method="bounded",
                                 options=dict(xatol=2e-3)).x)


# %% ---- main ----------------------------------------------------------------
def main():
    gals, R, bcg = load_population()
    print(f"exp29: {len(gals)} galaxies with dip-free MAH + 5-epoch profiles; BCG index={bcg}")

    gexp = fit_g(gals, R)
    scans = {p: stacked_dlog(gals, R, S0_FIXED, gexp, p) for p in (0.0, 2.0, 4.0)}
    s0 = scans[0.0]
    print(f"\n  best width-growth g={gexp:.2f} (fit to long-baseline b, p=0); "
          f"inside-out slope b per epoch pair:")
    print(f"    {'pair':10s} {'data':>6s} " + " ".join(f"p={p:.0f}".rjust(7) for p in scans))
    for k, lab in zip(PAIRS, PAIR_LABEL):
        print(f"    {lab:10s} {s0['b_data'][k]:+6.2f} " +
              " ".join(f"{scans[p]['b_model'][k]:+7.2f}" for p in scans))

    _figures(gals, R, bcg, gexp, scans)
    _save_and_check(bcg, gexp, scans)


# %% ---- figures -------------------------------------------------------------
def _figures(gals, R, bcg, gexp, scans):
    s0 = scans[0.0]
    fig, axes = plt.subplots(1, 4, figsize=(20.5, 4.7))

    a = axes[0]    # A. dip-free vs dippy MAH for the BCG
    raw = np.asarray(load_mah()[bcg], float); order = np.argsort(raw[0]); u = 1e10 / 0.6774
    mz = np.load(OFFICIAL); r = int(np.where(mz["index"] == bcg)[0][0])
    lm = mz["diffmah_log_mah_fit"][r]; sn = np.arange(100)[np.isfinite(lm)]
    a.plot(raw[0][order], np.log10(raw[1][order] * u), ":", c="0.6", lw=1.3, label="old: main-branch (dips)")
    a.plot(sn, lm[np.isfinite(lm)], "-", c=OKABE_ITO[0], lw=2.4, label="new: DiffMAH fit (dip-free)")
    a.set(xlabel="snapshot", ylabel=r"$\log M_{\rm halo}\ [M_\odot]$", title="A. Dip-free MAH (input)")
    a.legend(fontsize=8, loc="lower right")

    b = axes[1]    # B. stacked Delta log Sigma per pair: data vs centred-Gaussian model
    cmap = matplotlib.colormaps["viridis"]; res = (R >= RES_LO) & (R <= RES_HI)
    for j, (k, lab) in enumerate(zip(PAIRS[:4], PAIR_LABEL[:4])):
        c = cmap(j / 3)
        b.plot(R[res], s0["dstack"][k][res], "o", c=c, ms=3, alpha=0.7)
        b.plot(R[res], s0["mstack"][k][res], "-", c=c, lw=1.8, label=f"z={lab}")
    b.set(xscale="log", xlabel="R [kpc]", ylabel=r"$\Delta\log\Sigma_*$ (stacked)",
          title="B. Differential growth: data (dots) vs Gaussian model (lines)")
    b.legend(fontsize=8, title="epoch pair")

    c = axes[2]    # C. inside-out slope b(z): data vs model p=0/2/4 -> p>0 is worse
    x = np.arange(len(PAIRS))
    c.plot(x, [s0["b_data"][k] for k in PAIRS], "o-", c="k", lw=2, ms=7, label="TNG data")
    for p, col in zip(scans, (OKABE_ITO[1], OKABE_ITO[5], OKABE_ITO[2])):
        c.plot(x, [scans[p]["b_model"][k] for k in PAIRS], "s--", c=col, ms=5,
               label=f"model p={p:.0f}" + (" (Gaussian)" if p == 0 else " (outer)"))
    c.set_xticks(x); c.set_xticklabels(PAIR_LABEL, rotation=30, fontsize=7)
    c.set(ylabel=r"inside-out slope $b$", title=f"C. Centred Gaussian (p=0, g={gexp:.2f}) wins")
    c.legend(fontsize=8, loc="upper left")

    d = axes[3]    # D. why: deposit width grows with cosmic time -> late mass lands wide
    g_ = gals[bcg]; sigma = width_t(S0_FIXED, gexp, g_["t"], g_["t_obs"])
    dMstar = deposited(eff_two_epoch(g_["z"], *EFF_MEDIAN), g_["dMh"], g_["Mstar_tot"])
    sc = d.scatter(g_["z"], sigma, c=np.log10(np.clip(dMstar, 1, None)), cmap="plasma", s=14)
    d.set(xlabel="deposition redshift", ylabel=r"centred-Gaussian width $\sigma(t)$ [kpc]",
          title="D. Why: late epochs (low z) deposit WIDE")
    d.invert_xaxis(); fig.colorbar(sc, ax=d, label=r"$\log dM_*$ [$M_\odot$]", pad=0.01)
    fig.suptitle(f"exp29 — dip-free MAH + a CENTRED-Gaussian kernel reproduces exp26 inside-out "
                 f"growth; outer-weighting (p>0) is unnecessary  (n={len(gals)})", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp29_centred_gaussian_wins")[0])


def _save_and_check(bcg, gexp, scans):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    s0 = scans[0.0]
    write_manifest(OUTDIR, params={
        "bcg_index": bcg, "g_best": gexp, "s0_fixed": S0_FIXED, "eff_median": list(EFF_MEDIAN),
        "b_data": {PAIR_LABEL[i]: s0["b_data"][k] for i, k in enumerate(PAIRS)},
        "b_model_p0": {PAIR_LABEL[i]: scans[0.0]["b_model"][k] for i, k in enumerate(PAIRS)},
        "b_model_p2": {PAIR_LABEL[i]: scans[2.0]["b_model"][k] for i, k in enumerate(PAIRS)},
        "b_model_p4": {PAIR_LABEL[i]: scans[4.0]["b_model"][k] for i, k in enumerate(PAIRS)}})
    long = (0, 4)
    err0 = np.mean([abs(scans[0.0]["b_model"][k] - s0["b_data"][k]) for k in PAIRS])
    # centred Gaussian matches the b(z) trend, and outer-weighting only makes the long-baseline worse
    assert err0 < 0.08, err0
    assert scans[0.0]["b_model"][long] > scans[2.0]["b_model"][long] > scans[4.0]["b_model"][long], \
        "p>0 should monotonically reduce inside-out slope"
    assert abs(scans[0.0]["b_model"][long] - s0["b_data"][long]) < 0.06, "Gaussian must match long-baseline b"
    print(f"\n[verdict] dip-free MAH + a CENTRED-Gaussian deposit (sigma(t)=s0(t/t_obs)^{gexp:.2f}) "
          f"reproduces exp26 inside-out growth across all epochs (mean |b| error {err0:.3f}). "
          f"Outer-weighting p=2,4 monotonically undershoots (long-baseline b "
          f"{scans[0.0]['b_model'][long]:.2f}->{scans[2.0]['b_model'][long]:.2f}->"
          f"{scans[4.0]['b_model'][long]:.2f}). sigma(t), not deposit off-centredness, is the "
          f"inside-out mechanism. Next: a width set by the accretion *event*, and the shared-kernel "
          f"population CoG fit on this dip-free MAH.")


if __name__ == "__main__":
    sys.exit(main())
