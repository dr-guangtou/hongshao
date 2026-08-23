"""exp54 Stage 3 — the per-galaxy HALO RECORD: everything the forward model
may read, and nothing else.

This module is the enforcement point for the Contract B interface. A halo
record contains **only halo quantities**, assembled per deposit (one MAH step),
so `model.forward` cannot accidentally reach a stellar array: there is none in
the object it is handed. Stage 0's poisoned-data test then reduces to "does
`build_records` refuse to read stellar columns", which is checkable by
inspection as well as by test.

Per-deposit fields, all evaluated at the deposit's OWN formation time `t_j`:

    t, z, snap     the epoch of the step
    dMh            halo mass gained in the step               [Msun]
    logMh          log10 halo mass AFTER the step             [log Msun]
    r200c          R200c, PHYSICAL kpc  (raw is ckpc/h -> /h/(1+z))
    c200c          fitted NFW concentration, NaN where absent
    dlogc          log10[ c200c / c_Diemer19(Mh, z) ], 0 where absent
    has_c          whether dlogc came from a measurement (never a predictor)
    c_eff          c_Diemer19(Mh(t_j), z_j) * 10**dlogc -- ALWAYS defined, and
                   the model's working concentration. Where nothing was
                   measured it degrades gracefully to the Diemer19 mean
                   relation, i.e. "typical for this mass and redshift".
    f_form         t_half(t_j)/t_j, the epoch-local formation-time fraction

Per-galaxy: `epoch_mask` (n_epoch, n_deposit) selecting deposits laid down by
each observation epoch, and `r200c_at` at those epochs.

`f_form` in words: `t_half(t_j)` is when the halo first reached HALF the mass it
has at `t_j`, so `f_form` answers "of the mass this halo has now, how long ago
did it acquire half of it, as a fraction of its current age?". Near 0 = early
assembly. Unlike `fz2` and `t50` it never divides by the FINAL mass, so it is
computable for a halo at any epoch without knowing its descendant.

Run the self-check: HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python \\
    experiments/exp54_unpinned_amplitude/halo.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink"):
    sys.path.insert(0, str(p))

H_LITTLE = 0.6774
ANCHOR_SNAP = (72, 59, 50, 40, 33)
ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
HS_NPZ = HERE / "outputs" / "halo_structure_history.npz"
SO_NPZ = ROOT / "experiments/exp46_highz_ridge/outputs/so_history.npz"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
CACHE = HERE / "outputs" / "halo_records.npz"


@dataclass
class Halo:
    """One galaxy's halo history. Contains NO stellar information."""
    row: int
    index: int
    t: np.ndarray          # (nd,) cosmic time of each deposit [Gyr]
    z: np.ndarray          # (nd,)
    dmh: np.ndarray        # (nd,) [Msun]
    logmh: np.ndarray      # (nd,) log10 Msun, post-deposit
    r200c: np.ndarray      # (nd,) PHYSICAL kpc
    dlogc: np.ndarray      # (nd,) concentration excess, 0 where unmeasured
    has_c: np.ndarray      # (nd,) bool
    c_eff: np.ndarray      # (nd,) working concentration, always finite
    f_form: np.ndarray     # (nd,) t_half(t_j)/t_j
    epoch_mask: np.ndarray # (5, nd) bool
    r200c_at: np.ndarray   # (5,) PHYSICAL kpc at the observation epochs


def _diemer19(logmh, z):
    """c200c(M, z) from Diemer & Joyce (2019); h-free log-mass input."""
    from colossus.cosmology import cosmology
    from colossus.halo import concentration
    if "tng300" not in cosmology.cosmologies:
        cosmology.addCosmology("tng300", flat=True, H0=67.74, Om0=0.3089,
                               Ob0=0.0486, sigma8=0.8159, ns=0.9667)
    cosmology.setCosmology("tng300")
    out = np.full(np.shape(logmh), np.nan, float)
    ok = np.isfinite(logmh)
    if np.any(ok):
        out[ok] = concentration.concentration(
            10.0 ** np.asarray(logmh)[ok] * H_LITTLE, "200c", float(z),
            model="diemer19")
    return out


def _assert_m200c_is_log10(hs, pop):
    """Guard the units of the halo-structure `M200c` column.

    The Anbajagane catalog stores `M200c` as **log10(M/Msun)**, not as a linear
    mass in 1e10 Msun/h. Treating it as linear evaluates the Diemer19 reference
    concentration ~2 dex too low and distorts the concentration excess's
    dependence on halo mass -- a bug this project shipped once already. The
    column is checkable against the project's own halo mass, which is derived
    independently, so check it rather than trust a comment.
    """
    j = list(hs["snaps"]).index(72)                        # snap 72 == z = 0.4
    cat, own = np.asarray(hs["M200c"], float)[:, j], np.asarray(pop["logmh"], float)
    ok = np.isfinite(cat) & np.isfinite(own)
    dev = float(np.max(np.abs(cat[ok] - own[ok])))
    if dev > 1e-3:
        raise ValueError(
            f"halo-structure M200c is not log10(M/Msun): it disagrees with "
            f"population.npz logmh by up to {dev:.4f} dex at snap 72. Do NOT "
            f"'fix' this by rescaling -- find out what the column actually is.")


def _f_form(t, logmh):
    """t_half(t_j)/t_j: when the halo first had half its t_j mass, over t_j.

    Uses the CUMULATIVE MAXIMUM of the mass history so the reference is
    monotone even if the fitted curve dips; `t_half` is then well defined as a
    first-crossing and cannot jump backwards.
    """
    m = np.maximum.accumulate(10.0 ** np.asarray(logmh, float))
    t = np.asarray(t, float)
    out = np.empty_like(t)
    for j in range(len(t)):
        half = 0.5 * m[j]
        k = int(np.searchsorted(m[: j + 1], half))
        if k == 0:
            out[j] = t[0] / t[j]
        else:                                     # linear-in-t interpolation
            m0, m1 = m[k - 1], m[k]
            w = 0.0 if m1 == m0 else (half - m0) / (m1 - m0)
            out[j] = (t[k - 1] + w * (t[k] - t[k - 1])) / t[j]
    return np.clip(out, 0.0, 1.0)


def build_records(rows=None, verbose=True):
    """Assemble a `Halo` per galaxy. Reads MAH + halo structure ONLY."""
    import stage2_multiepoch as s2
    s2._w_init(None)
    gals = s2._W["gals"]
    if rows is not None:
        keep = set(int(r) for r in rows)
        gals = [g for g in gals if g["row"] in keep]
    pop = np.load(POP_NPZ)
    hs = np.load(HS_NPZ, allow_pickle=True)
    so = np.load(SO_NPZ)
    hs_i = {int(v): i for i, v in enumerate(hs["index"])}
    so_i = {int(v): i for i, v in enumerate(so["index"])}
    hs_snap, hs_z = list(hs["snaps"]), np.asarray(hs["z"], float)
    c_grid = np.where(hs["GroupFlag"] == 1, hs["c200c"], np.nan)
    _assert_m200c_is_log10(hs, pop)
    m_grid = np.where(np.isfinite(c_grid), hs["M200c"], np.nan)
    exc_grid = np.full_like(c_grid, np.nan)
    for j, zz in enumerate(hs_z):
        exc_grid[:, j] = np.log10(c_grid[:, j]) - np.log10(
            _diemer19(m_grid[:, j], zz))
    xg = np.log1p(hs_z)
    order = np.argsort(xg)

    out, rec = [], []
    for g in gals:
        idx = int(pop["index"][g["row"]])
        mah = g["mah"]
        snap, t, z = mah["snap"], mah["t"], mah["z"]
        logmh = mah["logMh_full"][1:]
        r_raw = so["Group_R_Crit200"][so_i[idx]]
        keep = snap < r_raw.shape[0]
        snap, t, z, logmh = snap[keep], t[keep], z[keep], logmh[keep]
        dmh = mah["dMh"][keep]
        r200 = r_raw[snap] / H_LITTLE / (1.0 + z)          # -> physical kpc
        v = exc_grid[hs_i[idx]][order]
        good = np.isfinite(v)
        x = np.log1p(z)
        if good.sum() >= 2:
            lo, hi = xg[order][good][[0, -1]]
            inside = (x >= lo) & (x <= hi)
            dlogc = np.where(inside, np.interp(x, xg[order][good], v[good]), 0.0)
        else:
            inside = np.zeros(len(x), bool)
            dlogc = np.zeros(len(x))
        em = np.stack([snap <= s for s in ANCHOR_SNAP])
        r_at = np.array([r_raw[s] / H_LITTLE / (1.0 + zz)
                         for s, zz in zip(ANCHOR_SNAP, ANCHOR_Z)])
        rec.append((g["row"], idx, snap, t, z, dmh, logmh, r200, dlogc,
                    inside, em, r_at))

    # Diemer19 is deterministic in (M, z), and every galaxy shares the snapshot
    # grid, so evaluate it ONCE PER SNAPSHOT over all galaxies rather than
    # per deposit -- 73 vectorized calls instead of ~170,000 scalar ones.
    z_of_snap = {}
    for _, _, snap, _, z, *_ in rec:
        for s, zz in zip(snap, z):
            z_of_snap.setdefault(int(s), float(zz))
    stack = {s: [] for s in z_of_snap}
    for gi, (_, _, snap, _, _, _, logmh, *_) in enumerate(rec):
        for k, s in enumerate(snap):
            stack[int(s)].append((gi, k, logmh[k]))
    cd19 = [np.full(len(r[3]), np.nan) for r in rec]
    for s, items in stack.items():
        if not items:
            continue
        gi, kk, lm = (np.array([x[i] for x in items]) for i in range(3))
        c = _diemer19(lm, z_of_snap[s])
        for a, b, v in zip(gi.astype(int), kk.astype(int), c):
            cd19[a][b] = v

    for (row, idx, snap, t, z, dmh, logmh, r200, dlogc, inside, em, r_at), cd \
            in zip(rec, cd19):
        out.append(Halo(row, idx, t, z, dmh, logmh, r200, dlogc, inside,
                        cd * 10.0 ** dlogc, _f_form(t, logmh), em, r_at))
    if verbose:
        nd = np.array([len(h.t) for h in out])
        print(f"  built {len(out)} halo records, {nd.min()}-{nd.max()} deposits "
              f"each; R200c finite {100 * np.mean([np.isfinite(h.r200c).mean() for h in out]):.1f}%, "
              f"dlogc measured {100 * np.mean([h.has_c.mean() for h in out]):.1f}%")
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    recs = build_records(rows=np.arange(0, 2397, 40))
    h = recs[3]
    print(f"\n  galaxy row {h.row}: {len(h.t)} deposits, "
          f"t {h.t[0]:.2f}-{h.t[-1]:.2f} Gyr, z {h.z[0]:.1f}-{h.z[-1]:.2f}")
    # (1) NO stellar field may exist on the record
    stellar = {"m500", "data", "logms", "mstar", "cog"}
    assert not (set(vars(h)) & stellar), set(vars(h)) & stellar
    # (2) R200c physical and sane, growing with time
    assert np.all(h.r200c[np.isfinite(h.r200c)] > 0)
    fin = np.isfinite(h.r200c)
    assert h.r200c[fin][-1] > h.r200c[fin][0], "R200c should grow with time"
    # (3) f_form bounded, and small (early assembly) => large t_half fraction? no:
    assert np.all((h.f_form >= 0) & (h.f_form <= 1))
    # a halo that has just doubled has t_half/t near 1 only if it doubled recently
    print(f"  f_form: {h.f_form[0]:.3f} (first) -> {h.f_form[-1]:.3f} (last), "
          f"median {np.median(h.f_form):.3f}")
    # (4) f_form is EPOCH-LOCAL: truncating the history must not change earlier values
    from copy import deepcopy
    cut = len(h.t) // 2
    ff_cut = _f_form(h.t[:cut], h.logmh[:cut])
    assert np.allclose(ff_cut, h.f_form[:cut], atol=1e-12), \
        "f_form leaked future information"
    # (5) epoch masks nest
    for k in range(4):
        assert np.all(h.epoch_mask[k + 1] <= h.epoch_mask[k]), "masks must nest"
    # (6) dlogc is 0 exactly where unmeasured
    assert np.all(h.dlogc[~h.has_c] == 0.0)
    print(f"  dlogc: measured on {h.has_c.sum()}/{len(h.t)} deposits, "
          f"median where measured {np.median(h.dlogc[h.has_c]):+.3f} dex")
    # (7) c_eff always finite, and equals the D19 mean where nothing was measured
    assert np.all(np.isfinite(h.c_eff)) and np.all(h.c_eff > 0)
    assert np.allclose(h.dlogc[~h.has_c], 0.0)
    print(f"  c_eff: {h.c_eff.min():.2f}-{h.c_eff.max():.2f}, "
          f"median {np.median(h.c_eff):.2f} (always finite)")
    print("\nhalo.py self-check OK: records carry no stellar field; R200c is "
          "physical and grows; f_form is bounded and provably epoch-local "
          "(truncating the MAH leaves earlier values identical); epoch masks "
          "nest; dlogc is exactly 0 where unmeasured; c_eff is always finite")
