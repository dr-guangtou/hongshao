"""exp26 side-analysis — visualise the halos with a DECLINING main-branch MAH.

`mah_declined` (1 - M_final/M_peak > 5%) flags 695/3388 galaxies whose tracked
main-branch halo mass sits below its historical peak by z=0.4. exp25 excludes
them (its kernel integrates the MAH); exp26 does not. Here we just look at WHAT
these declining histories are, classified by the properties that separate the
likely causes:

  - depth     : how much mass was lost (1 - M_final/M_peak)
  - timing    : how long ago it peaked (t_obs - t_peak)  -> recent dip vs long decline
  - smoothness: fraction of post-peak steps that decrease -> smooth loss vs noisy

Four archetypes (priority order):
  deep        : decline >= 20%  (strong stripping / major disruption)
  fluctuating : noisy post-peak (mono < 0.7)  -> merger-tree / halo-finder artifacts
  sustained   : smooth, peaked >= 1.5 Gyr ago  -> genuine turnover / slow stripping
  recent_dip  : smooth, peaked < 1.5 Gyr ago   -> benign recent downturn / merger

Run: PYTHONPATH=. uv run python experiments/exp26_differential_profiles/declining_mah.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hongshao.tng_data import load_mah, load_cosmic_time, PICKLE_MASS_UNIT     # noqa: E402
from hongshao.plotting import set_style, save_fig                              # noqa: E402

set_style()
HERE = Path(__file__).resolve().parent
OUTDIR, FIGDIR = HERE / "outputs", HERE / "figures"
TABLE = ROOT / "data" / "processed" / "tng300_072_z0p4.fits"
DEEP, MONO_LO, RECENT_GYR = 0.20, 0.70, 1.5
GROUPS = [("deep", "#d55e00"), ("fluctuating", "#0072b2"),
          ("sustained", "#009e73"), ("recent_dip", "#cc79a7")]


def features():
    """Per declining-MAH halo: raw history + depth / timing / smoothness, and class."""
    t = Table.read(TABLE)
    decl = np.asarray(t["mah_declined"])
    idx = np.asarray(t["index"]); lmh = np.asarray(t["logmh_z0p4"], float)
    mah = load_mah(); tsnap = load_cosmic_time()
    t_obs = tsnap[72]
    rows = []
    for k in np.where(decl)[0]:
        arr = np.asarray(mah[int(idx[k])], float)
        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] < 5:
            continue
        o = np.argsort(arr[0])
        snaps = arr[0][o].astype(int)
        m = arr[1][o] * PICKLE_MASS_UNIT
        ip = int(np.argmax(m))
        decline = float(1 - m[-1] / m[ip])
        post = m[ip:]
        mono = float((np.diff(post) <= 0).mean()) if len(post) > 1 else 1.0
        t_since = float(t_obs - tsnap[snaps[ip]])
        if decline >= DEEP:
            klass = "deep"
        elif mono < MONO_LO:
            klass = "fluctuating"
        elif t_since >= RECENT_GYR:
            klass = "sustained"
        else:
            klass = "recent_dip"
        rows.append(dict(index=int(idx[k]), logmh=lmh[k], decline=decline,
                         t_peak=float(tsnap[snaps[ip]]), t_since_peak=t_since,
                         mono=mono, klass=klass,
                         t_gyr=tsnap[snaps], logm=np.log10(m), logm_peak=float(np.log10(m[ip]))))
    return rows, t_obs


def main():
    rows, t_obs = features()
    OUTDIR.mkdir(parents=True, exist_ok=True); FIGDIR.mkdir(parents=True, exist_ok=True)
    print(f"declining-MAH halos with usable history: {len(rows)}")
    print(f"  {'class':12s} {'n':>4s} {'med decline':>11s} {'med t_since':>11s} "
          f"{'med mono':>8s} {'med logMh':>9s}")
    for kl, _ in GROUPS:
        g = [r for r in rows if r["klass"] == kl]
        if not g:
            print(f"  {kl:12s} {0:>4d}"); continue
        print(f"  {kl:12s} {len(g):>4d} {np.median([r['decline'] for r in g]):>11.2f} "
              f"{np.median([r['t_since_peak'] for r in g]):>11.2f} "
              f"{np.median([r['mono'] for r in g]):>8.2f} {np.median([r['logmh'] for r in g]):>9.2f}")

    # save the class assignment for downstream use
    Table(rows={"index": [r["index"] for r in rows], "klass": [r["klass"] for r in rows],
                "decline": [r["decline"] for r in rows], "t_since_peak": [r["t_since_peak"] for r in rows],
                "mono": [r["mono"] for r in rows], "logmh": [r["logmh"] for r in rows]}
          ).write(OUTDIR / "declining_mah_classes.fits", overwrite=True)

    # FIGURE: 2 rows x 4 archetypes. Top = full MAH (context); bottom = decline
    # shape, peak-aligned (t - t_peak), which is what defines the groups.
    xg = np.linspace(-1.5, 4.5, 60)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.4))
    for c, (kl, col) in enumerate(GROUPS):
        g = [r for r in rows if r["klass"] == kl]
        show = list(g) if len(g) <= 70 else [g[i] for i in rng.choice(len(g), 70, replace=False)]
        # top: absolute MAH vs cosmic time (context)
        a = axes[0, c]
        for r in show:
            a.plot(r["t_gyr"], r["logm"], "-", color=col, lw=0.5, alpha=0.25)
        a.axvline(t_obs, color="0.5", lw=0.8, ls="--")
        a.set_xlim(1, t_obs + 0.2); a.set_ylim(12.0, 15.2)
        a.set_title(f"{kl}  (n={len(g)})", fontsize=11, color=col)
        if c == 0:
            a.set_ylabel(r"$\log_{10} M_{\rm halo}$ (absolute)")
        # bottom: peak-aligned decline shape, normalised to peak
        b = axes[1, c]
        stacks = []
        for r in g:
            x = r["t_gyr"] - r["t_peak"]
            y = r["logm"] - r["logm_peak"]
            stacks.append(np.interp(xg, x, y, left=np.nan, right=np.nan))
        for r in show:
            b.plot(r["t_gyr"] - r["t_peak"], r["logm"] - r["logm_peak"], "-",
                   color=col, lw=0.5, alpha=0.25)
        med = np.nanmedian(np.array(stacks), axis=0)
        b.plot(xg, med, "-", color="k", lw=2.6, label="median")
        b.axvline(0, color="0.5", lw=0.8, ls=":"); b.axhline(0, color="0.5", lw=0.8, ls=":")
        b.set_xlim(-1.5, 4.5); b.set_ylim(-0.45, 0.05)
        b.set_xlabel(r"cosmic time since peak  $t-t_{\rm peak}$ [Gyr]")
        b.legend(fontsize=8, loc="lower left")
        d = np.median([r["decline"] for r in g])
        b.set_title(f"median decline {100*d:.0f}%", fontsize=9)
        if c == 0:
            b.set_ylabel(r"$\log_{10}(M/M_{\rm peak})$ (peak-aligned)")
    fig.suptitle("exp26 — declining main-branch MAHs by archetype.  Top: full history (z=0.4 dashed). "
                 "Bottom: decline shape, peak-aligned (deep=steep, fluctuating=noisy, "
                 "sustained=long smooth, recent_dip=short)", fontsize=12)
    fig.tight_layout(); save_fig(fig, FIGDIR / "exp26_declining_mah")
    print(f"\nwrote figure -> {FIGDIR}/exp26_declining_mah.png")
    assert sum(r["klass"] == "deep" for r in rows) > 0 and len(rows) > 500


if __name__ == "__main__":
    main()
