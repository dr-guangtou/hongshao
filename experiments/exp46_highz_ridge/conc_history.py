"""exp46 stage 1g — the FITTED concentration history, extracted and tested.

The TNG supplementary "Halo Structure" catalog (Header: `CatalogName =
HaloStructure`, `Reference = Dhayaa et al. (2021)`, i.e. Anbajagane et al.,
arXiv:2109.02713) fits an NFW profile with a free scale radius and reports
c200c for every FoF group, at twenty snapshots. Five of them are exactly
our profile epochs: **72 / 59 / 50 / 40 / 33** for z = 0.4 / 0.7 / 1.0 /
1.5 / 2.0.

CONFIRMED, not assumed: the project's existing z=0.4 `c200c` IS this
catalog at snapshot 72 — joining on `catgrp_id` reproduces all 2397 values
with **max |difference| = 0.0**. So the other four epochs drop straight
into the same feature pipeline, on the same definition.

The join key across snapshots is the FoF group index at that snapshot,
`SubhaloGrNr`, read from exp27's cached SubLink main branches (verified:
`SubhaloGrNr` at snap 72 equals `catgrp_id` for 400/400 spot-checked).
`GroupFlag == 1` marks the 459 972 of 18.5M halos with a valid structural
fit; every one of our galaxies is flagged valid at z=0.4.

The source files are ~2.7 GB EACH. This script reduces them to a
(2397 x 5) table so they can be deleted afterwards.

Run: PYTHONPATH=. uv run python experiments/exp46_highz_ridge/\
conc_history.py {demo|run}
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

OUTDIR = HERE / "outputs"
SUPP = OUTDIR / "supplementary"
MPB = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/mpb_cache"
XMATCH = ROOT / "experiments/exp27_tng_api_crossmatch/outputs/crossmatch.fits"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"

EPOCHS = ((0.4, 72), (0.7, 59), (1.0, 50), (1.5, 40), (2.0, 33))
# c200c is the target; the rest are the SAME five properties the project
# already carries at z=0.4, taken along because the download is paid for.
FIELDS = ("c200c", "M200c", "GroupFlag", "q", "s", "sigma_3D", "M_acc_dyn")


def group_ids():
    """(n, 5) FoF group index per galaxy per epoch, -1 where absent."""
    import h5py
    from astropy.table import Table
    xm = Table.read(XMATCH)
    pop = np.load(POP_NPZ)
    i2s = {int(i): int(s)
           for i, s in zip(xm["index"], xm["subhalo_id_snap72"])}
    gid = np.full((len(pop["index"]), len(EPOCHS)), -1, dtype=np.int64)
    for row, idx in enumerate(pop["index"]):
        sid = i2s.get(int(idx))
        if sid is None:
            continue
        with h5py.File(MPB / f"{sid}.hdf5", "r") as f:
            sn = f["SnapNum"][:]
            gr = f["SubhaloGrNr"][:]
        for j, (_, snap) in enumerate(EPOCHS):
            m = sn == snap
            if m.any():
                gid[row, j] = int(gr[m][0])
    return gid, pop


def cmd_run():
    import h5py
    from compact_anatomy import compactness, cliffs_delta, spearman
    from compact_figure import mass_matched
    from hongshao.tng_data import COG_RAD_KPC as R

    gid, pop = group_ids()
    n = len(gid)
    out = {k: np.full((n, len(EPOCHS)), np.nan) for k in FIELDS}
    print(f"exp46 stage 1g — fitted NFW concentration history (n={n})\n")

    for j, (z, snap) in enumerate(EPOCHS):
        path = SUPP / f"halo_structure.{snap}.hdf5"
        if not path.exists():
            print(f"  z={z}: MISSING {path.name} — run fetch_halo_structure.py")
            continue
        have = gid[:, j] >= 0
        with h5py.File(path, "r") as f:
            ncat = f["c200c"].shape[0]
            inrange = have & (gid[:, j] < ncat)
            idx = gid[inrange, j]
            order = np.argsort(idx)          # h5py needs increasing indices
            for k in FIELDS:
                vals = np.asarray(f[k][:])[idx[order]] if k in f else None
                if vals is None:
                    continue
                buf = np.full(inrange.sum(), np.nan)
                buf[order] = vals
                out[k][inrange, j] = buf
        valid = out["GroupFlag"][:, j] == 1
        print(f"  z={z:3.1f} (snap {snap:2d}): {have.sum():4d} matched, "
              f"{valid.sum():4d} with a valid structural fit "
              f"(GroupFlag==1), median c200c = "
              f"{np.nanmedian(np.where(valid, out['c200c'][:, j], np.nan)):.3f}")

    # only trust flagged-valid fits from here on
    c = np.where(out["GroupFlag"] == 1, out["c200c"], np.nan)

    d = c[:, 0] - pop["c200c"]
    print(f"\n  JOIN VALIDATION at z=0.4 against the project's own c200c: "
          f"max |diff| = {np.nanmax(np.abs(d)):.3e}  "
          f"({np.mean(np.abs(d) < 1e-9):.1%} exact)")

    # --- THE TEST ---------------------------------------------------------
    r50, c_in = compactness(pop["data"], R)
    lms2 = np.log10(np.clip(pop["data"][:, 4, -1], 1.0, None))
    comp = r50 <= np.quantile(r50, 0.10)
    matched = mass_matched(lms2, comp)
    print("\n  THE TEST — FITTED c200c, compact decile vs its "
          "stellar-mass-matched control:")
    print(f"    {'epoch':<22s}{'delta':>8s}{'rho(R50)':>10s}"
          f"{'rho(c_in)':>11s}{'n valid':>9s}")
    for j, (z, _) in enumerate(EPOCHS):
        a = c[:, j]
        print(f"    fitted c200c z={z:<9.1f}"
              f"{cliffs_delta(a[comp], a[matched]):+8.2f}"
              f"{spearman(r50, a):+10.2f}{spearman(c_in, a):+11.2f}"
              f"{np.isfinite(a).sum():9d}")
    print("    (for reference: the fit-free M500c/M200c proxy gave +0.01 at "
          "z=2;\n     the MAH head start at t=1 Gyr gives +0.35)")

    np.savez(OUTDIR / "conc_history.npz", gid=gid,
             epochs=np.array([z for z, _ in EPOCHS]),
             snaps=np.array([s for _, s in EPOCHS]),
             **{k: out[k] for k in FIELDS})
    tot = sum(p.stat().st_size for p in SUPP.glob("halo_structure.*.hdf5"))
    print(f"\n  wrote {OUTDIR / 'conc_history.npz'} "
          f"({(OUTDIR / 'conc_history.npz').stat().st_size / 1e3:.0f} kB)")
    print(f"  the {tot / 1e9:.1f} GB of source files in {SUPP} are now "
          "redundant and can be deleted")


def demo():
    # the sorted-index round trip must preserve galaxy order — the one
    # place this script could silently scramble everything, since h5py
    # requires increasing indices for fancy selection.
    rng = np.random.default_rng(0)
    cat = rng.normal(size=5000)
    idx = rng.choice(5000, size=300, replace=False)
    order = np.argsort(idx)
    buf = np.empty(300)
    buf[order] = cat[idx[order]]
    assert np.allclose(buf, cat[idx]), "index round trip scrambled the order"
    # duplicated group ids (two galaxies in one FoF group) must still work
    idx2 = np.array([7, 7, 3, 99, 3])
    o2 = np.argsort(idx2)
    b2 = np.empty(5)
    b2[o2] = cat[idx2[o2]]
    assert np.allclose(b2, cat[idx2])
    print("demo OK — sorted-index round trip preserves order, "
          "including duplicate group ids")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run()
    else:
        raise SystemExit(__doc__)
