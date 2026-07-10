"""exp32 step 1a — the full-population dataset cache.

One npz for the entire eligible sample (use flag + finite z=0.4 CoG + QUALITY-
MATCHED official DiffMAH crossmatch [the exp29 standard] + valid real de-dipped
MAH + valid 5-epoch measured CoG):
  index, logms, logmh, c200c              identity + z=0.4 halo props
  data (n,5,24)                           linear 5-epoch CoG masses [Msun]
  t50, fz2, burst                         real-MAH halo-only summaries
  logmh_zk_real, logmh_zk_diffmah (n,5)   epoch-matched halo masses per config
  dev100                                  stratified 100-galaxy dev subsample rows

Run: PYTHONPATH=. uv run python experiments/exp32_full_population/dataset.py [n_max]
Demo: ... dataset.py demo   (rebuild must reproduce the exp30 sample exactly)
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
from run import dipfree_mah, ANCHOR_SNAP, TABLE, OFFICIAL                            # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402
from cog_extrapolate import measured_cog                                             # noqa: E402

OUTDIR = HERE / "outputs"
OUT_NPZ = OUTDIR / "population.npz"


def burstiness(mah):
    s = mah["dMh"] / 10.0 ** mah["logMh_full"][1:]
    return float(mah["dMh"][s > 0.10].sum() / mah["dMh"].sum())


def build(n_max=None):
    from astropy.table import Table
    t = Table.read(TABLE)
    cog0 = np.asarray(t["logmstar_cog"], float)
    ok = np.asarray(t["use"]) & np.isfinite(cog0).all(axis=1)
    idx_all = np.asarray(t["index"])
    mz = np.load(OFFICIAL)
    matched = {int(g) for g, m in zip(mz["index"], mz["matched"]) if m}
    order = np.argsort(np.where(ok, cog0[:, -1], -np.inf))[::-1]   # mass-ranked
    rows = dict(index=[], logms=[], logmh=[], c200c=[], data=[], t50=[], fz2=[],
                burst=[], logmh_zk_real=[], logmh_zk_diffmah=[])
    n_skip = 0
    for i in order:
        if not ok[i]:
            break                                       # sorted: rest are invalid
        gi = int(idx_all[i])
        if gi not in matched:
            n_skip += 1
            continue
        rm, dm, logC = real_mah(gi), dipfree_mah(gi), measured_cog(gi)
        if rm is None or dm is None or logC is None:
            n_skip += 1
            continue
        Mh, tf = 10.0 ** rm["logMh_full"], rm["t_full"]
        rows["index"].append(gi)
        rows["logms"].append(float(logC[0, -1]))
        rows["logmh"].append(float(t["logmh_z0p4"][i]))
        rows["c200c"].append(float(t["c_200c"][i]))
        rows["data"].append(10.0 ** logC)
        rows["t50"].append(float(np.interp(0.5 * Mh[-1], Mh, tf)))
        rows["fz2"].append(float(10.0 ** (np.interp(33, rm["snap_full"], rm["logMh_full"])
                                          - np.log10(Mh[-1]))))
        rows["burst"].append(burstiness(rm))
        rows["logmh_zk_real"].append(np.interp(ANCHOR_SNAP, rm["snap_full"],
                                               rm["logMh_full"]))
        rows["logmh_zk_diffmah"].append(np.interp(ANCHOR_SNAP, dm["snap_full"],
                                                  dm["logMh_full"]))
        if n_max and len(rows["index"]) >= n_max:
            break
    out = {k: np.array(v) for k, v in rows.items()}
    # stratified dev subsample: 100 galaxies spread evenly through the mass ranking
    n = len(out["index"])
    out["dev100"] = np.unique(np.linspace(0, n - 1, 100).round().astype(int))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **out)
    print(f"wrote {OUT_NPZ}: n={n} (skipped {n_skip}), "
          f"logM* {out['logms'].min():.2f}-{out['logms'].max():.2f}, "
          f"logMh {out['logmh'].min():.2f}-{out['logmh'].max():.2f}")
    return out


def demo():
    """Check the built cache contains the exp30 sample with identical data.
    (The historical n=45 was a stratified every-~41st subsample of the mass
    ranking, NOT the top-45 — discovered here; rows 0,41,...,1787.)"""
    d30 = np.load(ROOT / "experiments/exp30_transport_kernel/outputs/param_emulator.npz")
    d = np.load(OUT_NPZ)
    assert np.all(np.diff(d["logms"]) <= 1e-9), "cache must be mass-ranked"
    row = {int(g): i for i, g in enumerate(d["index"])}
    sel = [row[int(g)] for g in d30["index"]]           # KeyError if any missing
    assert np.allclose(d["data"][sel], d30["data"], rtol=1e-8), "CoGs must match exp30"
    assert np.allclose(d["burst"][sel], d30["burst"], atol=1e-12)
    assert len(d["dev100"]) == 100 and len(d["index"]) > 2000
    print(f"dataset.demo OK: n={len(d['index'])}, exp30 sample embedded exactly "
          f"(rows {sel[0]},{sel[1]},...,{sel[-1]}), mass-ranked, dev100 ready")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        n_max = int(sys.argv[1]) if len(sys.argv) > 1 else None
        build(n_max)
