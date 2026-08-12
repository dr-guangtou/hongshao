"""exp46 stage 1f — spherical-overdensity mass histories, and a fit-free
concentration proxy at the RIGHT epoch.

Stage 1e found halo concentration carries essentially nothing for the
compact z=2 progenitors (Cliff's delta -0.02). But that `c200c` is
measured at **z=0.4**, so it says nothing about concentration at z=2. The
user's point: additional halo-property HISTORIES may help, provided they
are clearly defined, reliably measured, and portable across N-body
simulations and halo finders.

Scope (user, 2026-08-12): **MAH via different SO masses, and concentration
history only.** Centre offset, spin and substructure counts are excluded
deliberately — their definitions are tied to a specific simulation + halo
finder combination.

Spherical-overdensity masses qualify on all three criteria and are already
on disk, at all 72 snapshots, in exp27's cached SubLink main branches. The
ratio M500c/M200c is a concentration proxy needing **no NFW fit, no Vmax,
and no profile fitting** — just two directly measured SO masses. Validated
here at Spearman +0.700 against the catalog's own c200c at z=0.4, which
matches the best NFW-based proxy (+0.709).

The payoff test: does the fit-free proxy, evaluated AT z=2, separate the
compact progenitors that z=0.4 concentration could not?

Run: PYTHONPATH=. uv run python experiments/exp46_highz_ridge/\
so_history.py {demo|run}
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

OUTDIR = HERE / "outputs"
MPB = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/mpb_cache"
XMATCH = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/crossmatch.fits"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"

N_SNAP = 73          # snapshots 0..72; the root of each tree IS snap 72
EPOCHS = ((0.4, 72), (0.7, 59), (1.0, 50), (1.5, 40), (2.0, 33))
SO_FIELDS = ("Group_M_Crit200", "Group_M_Crit500", "Group_M_Mean200",
             "Group_M_TopHat200", "Group_R_Crit200", "Group_R_Crit500")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def extract(indices):
    """(n, 72) arrays per SO field, on a fixed snapshot axis 0..71.

    Galaxies enter the tree at different snapshots, so absent snapshots stay
    NaN rather than being silently filled — a zero SO mass means "the halo
    was not identified", not "the halo was empty".
    """
    import h5py
    from astropy.table import Table
    xm = Table.read(XMATCH)
    i2s = {int(i): int(s)
           for i, s in zip(xm["index"], xm["subhalo_id_snap72"])}
    out = {k: np.full((len(indices), N_SNAP), np.nan) for k in SO_FIELDS}
    missing = 0
    for row, idx in enumerate(indices):
        sid = i2s.get(int(idx))
        if sid is None:
            missing += 1
            continue
        try:
            with h5py.File(MPB / f"{sid}.hdf5", "r") as f:
                sn = f["SnapNum"][:]
                cols = {k: f[k][:] for k in SO_FIELDS}
        except (OSError, KeyError):
            missing += 1
            continue
        keep = (sn >= 0) & (sn < N_SNAP)
        for k in SO_FIELDS:
            v = np.asarray(cols[k], float)[keep]
            out[k][row, sn[keep]] = np.where(v > 0, v, np.nan)
    return out, missing


def ratios(so):
    """Fit-free concentration proxies. Each is a ratio of two directly
    measured SO masses, so units, h and the scale factor all cancel."""
    return {
        "M500c/M200c": so["Group_M_Crit500"] / so["Group_M_Crit200"],
        "M200m/M200c": so["Group_M_Mean200"] / so["Group_M_Crit200"],
        "MTopHat/M200c": so["Group_M_TopHat200"] / so["Group_M_Crit200"],
        "R500c/R200c": so["Group_R_Crit500"] / so["Group_R_Crit200"],
    }


def cmd_run():
    from compact_anatomy import compactness, cliffs_delta
    from compact_figure import mass_matched
    from hongshao.tng_data import COG_RAD_KPC as R

    pop = np.load(POP_NPZ)
    so, missing = extract(pop["index"])
    rat = ratios(so)
    n = len(pop["index"])
    print(f"exp46 stage 1f — SO mass histories (n={n}, {N_SNAP} snapshot slots; "
          f"{missing} unresolved)\n")

    cov = np.isfinite(so["Group_M_Crit200"]).mean(0)
    print("  snapshot coverage of Group_M_Crit200 at the profile epochs:")
    for z, s in EPOCHS:
        print(f"    z={z:3.1f} (snap {s:2d}): {100 * cov[s]:5.1f}% of galaxies")

    print("\n  Validation at z=0.4 against the catalog's own c200c "
          "(the only epoch where a fitted concentration exists):")
    c0 = pop["c200c"]
    for name, v in rat.items():
        print(f"    {name:<16s} Spearman = {spearman(v[:, 72], c0):+.3f}")
    print("    for reference: NFW proxies gave +0.709 (R_max), "
          "+0.551 (V_max)")

    print("\n  Median by epoch (a history, not a single number):")
    print(f"    {'proxy':<16s}" + "".join(f"z={z:<8.1f}" for z, _ in EPOCHS))
    for name, v in rat.items():
        print(f"    {name:<16s}"
              + "".join(f"{np.nanmedian(v[:, s]):<10.3f}" for _, s in EPOCHS))

    # --- the payoff: does the proxy AT z=2 do what c200c(z=0.4) could not?
    r50, c_in = compactness(pop["data"], R)
    lms2 = np.log10(np.clip(pop["data"][:, 4, -1], 1.0, None))
    comp = r50 <= np.quantile(r50, 0.10)
    matched = mass_matched(lms2, comp)
    print("\n  THE TEST — compact decile vs its stellar-mass-matched control"
          "\n  (Cliff's delta; stage 1e got -0.02 for c200c measured at "
          "z=0.4):")
    print(f"    {'quantity':<28s}{'delta':>8s}{'rho(R50)':>10s}"
          f"{'rho(c_in)':>11s}")
    print(f"    {'c200c (z=0.4, catalog)':<28s}"
          f"{cliffs_delta(c0[comp], c0[matched]):+8.2f}"
          f"{spearman(r50, c0):+10.2f}{spearman(c_in, c0):+11.2f}")
    res = {}
    for name, v in rat.items():
        for z, s in ((0.4, 72), (2.0, 33)):
            a = v[:, s]
            d = cliffs_delta(a[comp], a[matched])
            res[(name, z)] = d
            print(f"    {name + f' (z={z})':<28s}{d:+8.2f}"
                  f"{spearman(r50, a):+10.2f}{spearman(c_in, a):+11.2f}")

    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / "so_history.npz",
             **{k.replace("/", "_over_"): v for k, v in so.items()},
             **{k.replace("/", "_over_"): v for k, v in rat.items()},
             index=pop["index"], compact=comp, matched=matched)
    print(f"\n  wrote {OUTDIR / 'so_history.npz'}")


def demo():
    # ratios must be unit-, h- and a-free: scaling both masses leaves them fixed
    so = {k: np.full((4, N_SNAP), 2.0) for k in SO_FIELDS}
    so["Group_M_Crit500"][:] = 1.0
    r = ratios(so)
    assert np.allclose(r["M500c/M200c"], 0.5)
    so2 = {k: v * 1e10 / 0.6774 for k, v in so.items()}
    assert np.allclose(ratios(so2)["M500c/M200c"], r["M500c/M200c"])
    # spearman sanity
    x = np.random.default_rng(0).normal(size=200)
    assert abs(spearman(x, np.exp(x)) - 1.0) < 1e-9
    # NaN discipline: a zero SO mass must not become a real ratio
    so["Group_M_Crit200"][0, 0] = np.nan
    assert np.isnan(ratios(so)["M500c/M200c"][0, 0])
    print("demo OK — SO ratios are unit/h/a-free and NaN-safe")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run()
    else:
        raise SystemExit(__doc__)
