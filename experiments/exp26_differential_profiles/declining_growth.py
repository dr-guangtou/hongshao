"""exp26 — do the declining-MAH halos still grow their STELLAR mass after the halo
turnover, and do the four archetypes differ?

Stellar mass is centrally bound and far more robust to stripping than halo mass.
For each declining-MAH group we track the total stellar mass M*(z) at the 5
profile epochs and the differential stellar-density profile of the latest
interval [0.4,0.7] (mostly post-turnover), and compare to a control of
non-declined galaxies.

Metrics per galaxy:
  dlogM*_late  = logM*(z=0.4) - logM*(z=0.7)            late (post-turnover) growth
  dlogM*_post  = logM*(z=0.4) - logM*(t_peak)           growth since the HALO peak
  dlogM*_total = logM*(z=0.4) - logM*(z=2.0)
  + the [0.4,0.7] differential density profile (inner 8 kpc / outer 60 kpc).

Run: PYTHONPATH=. uv run python experiments/exp26_differential_profiles/declining_growth.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from hongshao.tng_data import load_aper, load_cosmic_time                      # noqa: E402
from hongshao.plotting import set_style, save_fig                             # noqa: E402
import declining_mah as dm                                                    # noqa: E402

set_style()
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
ZKEYS = ["z0p4", "z0p7", "z1", "z1p5", "z2"]
ZSNAP = [72, 59, 50, 40, 33]                       # snapshots for z=0.4..2.0
GROUPS = dm.GROUPS                                 # [(name,color), ...]


def mstar_track(index):
    """Total log10 M*(z) at the 5 epochs (CoG outer point); NaN if missing."""
    a = load_aper(int(index))
    out = np.full(5, np.nan)
    for k, zk in enumerate(ZKEYS):
        if zk not in a:
            continue
        try:
            v = np.atleast_1d(np.asarray(a[zk].get("cog"), float))
        except (TypeError, ValueError):
            continue
        if v.size >= 24 and np.isfinite(v[-1]) and v[-1] > 0:
            out[k] = np.log10(v[-1])
    return out


def main():
    tsnap = load_cosmic_time()
    t_ep = tsnap[ZSNAP]                              # epoch times (descending z -> ascending? no)
    order = np.argsort(t_ep)                         # ascending cosmic time: z2..z0.4
    t_asc = t_ep[order]
    t_obs = tsnap[72]

    rows, _ = dm.features()
    tpeak = {r["index"]: r["t_peak"] for r in rows}
    klass = {r["index"]: r["klass"] for r in rows}

    # differential density (exp26) for the [0.4,0.7] pair, by index
    d = np.load(OUTDIR / "differential_profiles.npz")
    R = d["R"]; sig = d["sigma"]; didx = d["index"]
    row_of = {int(ix): i for i, ix in enumerate(didx)}
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog04 = np.log10(sig[:, 0, :]) - np.log10(sig[:, 1, :])   # z0.4 - z0.7, all exp26 gals
    iin, iout = int(np.argmin(np.abs(R - 8))), int(np.argmin(np.abs(R - 60)))

    # control: non-declined galaxies in the exp26 sample
    t = Table.read(TABLE)
    decl_set = set(int(i) for i in tpeak)
    control_idx = [int(i) for i in didx if int(i) not in decl_set]
    rng = np.random.default_rng(0)
    control_idx = list(rng.choice(control_idx, min(500, len(control_idx)), replace=False))

    def collect(indices, is_decl):
        recs = []
        for ix in indices:
            lm = mstar_track(ix)
            if not np.isfinite(lm[0]):                # need z=0.4
                continue
            lm_asc = lm[order]
            rec = dict(index=ix, lm=lm, lm_asc=lm_asc,
                       dlate=lm[0] - lm[1], dtot=lm[0] - lm[4])
            if is_decl:
                tp = np.clip(tpeak[ix], t_asc[0], t_obs)
                rec["dpost"] = lm[0] - float(np.interp(tp, t_asc, lm_asc))
                rec["klass"] = klass[ix]; rec["t_peak"] = tpeak[ix]
            if ix in row_of:
                dl = dlog04[row_of[ix]]
                rec["din"], rec["dout"] = dl[iin], dl[iout]
                rec["dprof"] = dl
            recs.append(rec)
        return recs

    decl = collect(list(tpeak), True)
    ctrl = collect(control_idx, False)

    print(f"declining galaxies with M* track: {len(decl)};  control: {len(ctrl)}\n")
    print(f"  {'group':12s} {'n':>4s} {'dlogM*_late':>12s} {'dlogM*_post':>12s} "
          f"{'dlogM*_total':>13s} {'dSig_in(8)':>11s} {'dSig_out(60)':>12s}")
    summ = {}
    for kl, _ in GROUPS:
        g = [r for r in decl if r["klass"] == kl]
        med = lambda key: np.nanmedian([r.get(key, np.nan) for r in g])
        summ[kl] = g
        print(f"  {kl:12s} {len(g):>4d} {med('dlate'):>+12.3f} {med('dpost'):>+12.3f} "
              f"{med('dtot'):>+13.3f} {med('din'):>+11.3f} {med('dout'):>+12.3f}")
    med = lambda key: np.nanmedian([r.get(key, np.nan) for r in ctrl])
    print(f"  {'control':12s} {len(ctrl):>4d} {med('dlate'):>+12.3f} {'--':>12s} "
          f"{med('dtot'):>+13.3f} {med('din'):>+11.3f} {med('dout'):>+12.3f}")

    f_grow = np.mean([r["dlate"] > 0 for r in decl])
    print(f"\n  fraction of declining-MAH galaxies that STILL grow M* in [0.7->0.4]: {f_grow:.2f}")

    make_figure(summ, ctrl, R, t_asc, t_obs, tpeak)
    print(f"\nwrote figure -> {FIGDIR}/exp26_declining_growth.png")
    assert f_grow > 0.5, "expected most declining-MAH galaxies to still grow stars"


def make_figure(summ, ctrl, R, t_asc, t_obs, tpeak):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5.2))

    # A: cumulative stellar growth logM*(t) - logM*(z=2) per group + control
    for kl, col in GROUPS:
        g = summ[kl]
        tracks = np.array([r["lm_asc"] - r["lm_asc"][0] for r in g])
        a1.plot(t_asc, np.nanmedian(tracks, axis=0), "-o", color=col, lw=2, ms=4, label=f"{kl} (n={len(g)})")
        a1.axvline(np.median([tpeak[r["index"]] for r in g]), color=col, ls=":", lw=1, alpha=0.6)
    ct = np.array([r["lm_asc"] - r["lm_asc"][0] for r in ctrl])
    a1.plot(t_asc, np.nanmedian(ct, axis=0), "-o", color="k", lw=2, ms=4, label="control (non-declined)")
    a1.axvline(t_obs, color="0.5", ls="--", lw=0.8)
    a1.set_xlabel("cosmic time [Gyr]"); a1.set_ylabel(r"median $\log M_*(t)-\log M_*(z{=}2)$")
    a1.set_title("A. Stars keep growing (dotted = median halo peak)"); a1.legend(fontsize=8)

    # B: late stellar growth distribution per group + control
    data = [[r["dlate"] for r in summ[kl]] for kl, _ in GROUPS] + [[r["dlate"] for r in ctrl]]
    labels = [kl for kl, _ in GROUPS] + ["control"]
    cols = [c for _, c in GROUPS] + ["0.4"]
    bp = a2.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, whis=(10, 90))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    a2.axhline(0, color="k", lw=0.8, ls=":")
    a2.set_ylabel(r"$\Delta\log M_*$ in [0.7$\to$0.4] (late growth)")
    a2.set_title("B. Late stellar growth by archetype"); a2.tick_params(axis="x", rotation=20)

    # C: median differential density profile [0.4,0.7] per group + control
    for kl, col in GROUPS:
        prof = np.array([r["dprof"] for r in summ[kl] if "dprof" in r])
        if len(prof):
            a3.plot(R, np.nanmedian(prof, axis=0), "-", color=col, lw=2, label=kl)
    cprof = np.array([r["dprof"] for r in ctrl if "dprof" in r])
    a3.plot(R, np.nanmedian(cprof, axis=0), "-", color="k", lw=2, label="control")
    a3.axhline(0, color="0.5", lw=0.8, ls=":"); a3.axvspan(0, 6, color="0.9")
    a3.set_xscale("log"); a3.set_xlim(2, 130)
    a3.set_xlabel("R [kpc]"); a3.set_ylabel(r"median $\Delta\log\Sigma_*$ [0.4,0.7]")
    a3.set_title("C. Where the late stars are added"); a3.legend(fontsize=8)

    fig.suptitle("exp26 — declining-MAH halos still grow stars (inside-out) after the halo turns over, "
                 "MORE than the non-declined control; deep/sustained (mergers) grow most, with elevated INNER growth",
                 fontsize=11)
    fig.tight_layout(); save_fig(fig, FIGDIR / "exp26_declining_growth")


if __name__ == "__main__":
    main()
