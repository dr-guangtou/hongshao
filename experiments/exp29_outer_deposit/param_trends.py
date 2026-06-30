"""exp29 — patterns in the independent single-epoch best-fit parameters.

single_epoch_all.py fit the centred-Gaussian kernel to each epoch's CoG alone and
cached the best-fit params (outputs/single_epoch_params.npz). Here we ask: do the
params show natural, epoch-ordered trends? Two questions:

  1. Which parts of the kernel are epoch-STABLE (a shared shape) and which must
     CHANGE with the observation epoch (the parametric fingerprint of the
     multi-epoch tension)?
  2. The puff-up test: a clump of stellar mass formed before z=2 — does the
     independent fit make it OCCUPY A LARGER RADIUS when observed later (puffing,
     sigma grows post-deposition) or a SMALLER one (anti-puff)? We use a
     mass-weighted, degeneracy-robust width, not a single poorly-constrained
     deposit width.

Raw params per epoch: (log10 sigma_0, g, b_early, b_late, z_c). The width law
sigma(t)=sigma_0 (t/t_obs)^g is anchored at t_obs=t(z=0.4), so sigma_0 inflates for
high-z fits (extrapolation) and is NOT comparable across epochs — derived widths
evaluated at fixed cosmic times are.

Run: PYTHONPATH=. uv run python experiments/exp29_outer_deposit/param_trends.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from deposit import width_t, eff_two_epoch, deposited, deposit_cog                   # noqa: E402
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z                                   # noqa: E402
from hongshao.tng_data import COG_RAD_KPC                                            # noqa: E402
from hongshao.plotting import set_style, save_fig, OKABE_ITO                         # noqa: E402

set_style()
FIGDIR = HERE / "figures"
NPZ = HERE / "outputs" / "single_epoch_params.npz"
R = COG_RAD_KPC


def r50_of_cog(cog):
    """Half-mass radius of a (monotonic) model CoG by interpolation on the R grid."""
    return float(np.interp(0.5 * cog[-1], cog, R))


def derive(d):
    """Per-galaxy, per-epoch derived widths from the cached params. Returns a dict
    of (n, 5) arrays; widths in kpc, evaluated at fixed cosmic times so they are
    comparable across the observation epoch."""
    P, idx, at = d["params"], d["index"], d["anchor_t"]
    t_obs, t_z2 = at[0], at[4]
    n = len(idx)
    out = {k: np.full((n, 5), np.nan) for k in
           ("sig_fresh", "sig_all", "sig_early", "r50_early", "frac_early")}
    for i, gi in enumerate(idx):
        mah = dipfree_mah(int(gi))
        for k in range(5):
            s0, g, be, bl, zc = 10.0 ** P[i, k, 0], P[i, k, 1], P[i, k, 2], P[i, k, 3], P[i, k, 4]
            dM = deposited(eff_two_epoch(mah["z"], be, bl, zc), mah["dMh"], 1.0)
            sig = width_t(s0, g, mah["t"], t_obs)
            inc = mah["snap"] <= ANCHOR_SNAP[k]                  # deposits up to t(z_k)
            early = inc & (mah["t"] <= t_z2)                     # formed before z=2 (fixed set)
            wI = dM[inc] / dM[inc].sum()
            wE = dM[early] / dM[early].sum()
            out["sig_fresh"][i, k] = s0 * (at[k] / t_obs) ** g   # freshest deposit width
            out["sig_all"][i, k] = (wI * sig[inc]).sum()         # mass-wtd width, whole galaxy
            out["sig_early"][i, k] = (wE * sig[early]).sum()     # mass-wtd width, pre-z2 mass
            out["r50_early"][i, k] = r50_of_cog(deposit_cog(dM[early], sig[early], 0.0, R))
            out["frac_early"][i, k] = dM[early].sum() / dM[inc].sum()
    return out


def main():
    d = np.load(NPZ)
    P, logms, az = d["params"], d["logms"], d["anchor_z"]
    der = derive(d)
    n = len(logms)
    print(f"exp29 — single-epoch best-fit parameter trends (n={n})\n")

    print("  raw params, median per epoch:")
    print(f"    {'z':>4s} {'g':>6s} {'b_early':>8s} {'b_late':>7s} {'z_c':>6s}")
    for k, z in enumerate(az):
        print(f"    {z:4.1f} {np.median(P[:,k,1]):6.2f} {np.median(P[:,k,2]):8.2f} "
              f"{np.median(P[:,k,3]):7.2f} {np.median(P[:,k,4]):6.2f}")

    print("\n  derived widths [kpc] & early-mass extent, median per epoch:")
    print(f"    {'z':>4s} {'sig_fresh':>10s} {'sig_all':>8s} {'sig_early':>10s} "
          f"{'R50_early':>10s} {'frac_early':>11s}")
    for k, z in enumerate(az):
        print(f"    {z:4.1f} {np.median(der['sig_fresh'][:,k]):10.1f} "
              f"{np.median(der['sig_all'][:,k]):8.1f} {np.median(der['sig_early'][:,k]):10.1f} "
              f"{np.median(der['r50_early'][:,k]):10.2f} {np.median(der['frac_early'][:,k]):11.2f}")

    # the puff-up verdict: R50 of the SAME pre-z2 mass, late vs at formation
    r50 = der["r50_early"]
    ratio = np.median(r50[:, 0] / r50[:, 4])             # z=0.4 / z=2
    print(f"\n[puff-up test] R50 of pre-z=2 mass:  z=2 {np.median(r50[:,4]):.2f} kpc  ->  "
          f"z=0.4 {np.median(r50[:,0]):.2f} kpc   (ratio {ratio:.2f})")
    if ratio > 1.15:
        print("  -> early-formed mass occupies a LARGER radius when observed later: the\n"
              "     independent fits DEMAND puffing. A monotonic post-deposition sigma-growth\n"
              "     law (PUFF_MODEL_PLAN) is the right shape and this sets its rate.")
    elif ratio < 0.87:
        print("  -> early-formed mass is fit MORE COMPACT when observed later (anti-puff).\n"
              "     Puffing is the wrong sign; the epochs disagree on the early mass via the\n"
              "     efficiency, not the width. Re-think before building puff-up.")
    else:
        print("  -> early-mass extent is ~epoch-stable; the tension lives elsewhere (efficiency).")

    _figure(P, logms, az, der)


def _figure(P, logms, az, der):
    x = np.arange(5)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    a = axes[0]                                          # what is stable vs what rotates
    a.plot(x, [np.median(P[:, k, 1]) for k in range(5)], "o-", c=OKABE_ITO[0], lw=2, label="g (width growth)")
    a.plot(x, [np.median(P[:, k, 2]) for k in range(5)], "s-", c=OKABE_ITO[1], lw=2, label=r"$b_{\rm early}$")
    a.plot(x, [np.median(P[:, k, 3]) for k in range(5)], "^-", c=OKABE_ITO[5], lw=2, label=r"$b_{\rm late}$")
    a.axhline(np.median(P[:, :, 1]), c=OKABE_ITO[0], ls=":", lw=1)
    a.set_xticks(x); a.set_xticklabels([f"{z}" for z in az])
    a.set(xlabel="observation epoch z", ylabel="best-fit value",
          title="A. g is epoch-stable (~%.1f); the efficiency rotates" % np.median(P[:, :, 1]))
    a.legend(fontsize=8)

    b = axes[1]                                          # characteristic widths vs epoch
    for key, c, lab in (("sig_fresh", OKABE_ITO[2], "freshest deposit"),
                        ("sig_all", OKABE_ITO[0], "mass-wtd (whole)"),
                        ("sig_early", OKABE_ITO[3], "mass-wtd (pre-z2)")):
        med = [np.median(der[key][:, k]) for k in range(5)]
        b.plot(x, med, "o-", c=c, lw=2, label=lab)
    b.set_xticks(x); b.set_xticklabels([f"{z}" for z in az])
    b.set(xlabel="observation epoch z", ylabel="Gaussian width [kpc]",
          title="B. Late accretion lands wide (inside-out)")
    b.legend(fontsize=8)

    c = axes[2]                                          # the puff-up test
    r50 = der["r50_early"]
    for idx, col, lab in ((slice(None), "0.6", "all"),
                          (np.argsort(logms)[-20:], OKABE_ITO[5], "high-mass 1/3")):
        med = [np.median(r50[idx, k]) for k in range(5)]
        c.plot(x, med, "o-", c=col, lw=2, label=lab)
    c.set_xticks(x); c.set_xticklabels([f"{z}" for z in az])
    c.set(xlabel="observation epoch z", ylabel=r"$R_{50}$ of pre-z=2 mass [kpc]",
          title="C. Puff-up test: extent of the SAME early mass")
    c.legend(fontsize=8, title="sample")
    fig.suptitle("exp29 — single-epoch best-fit parameter trends: a stable spatial kernel (g), a "
                 "rotating efficiency, and the early-mass extent across epochs", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp29_param_trends")[0])


def demo():
    """Self-check: R50 of a single Gaussian CoG is sigma*sqrt(2 ln 2)."""
    s = 5.0
    cog = deposit_cog(np.array([1.0]), np.array([s]), 0.0, R)   # leak beyond 148 kpc negligible
    assert abs(r50_of_cog(cog) - s * np.sqrt(2.0 * np.log(2.0))) < 0.08 * s, r50_of_cog(cog)
    print("param_trends.demo OK: R50(Gaussian) = sigma*sqrt(2 ln2)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
