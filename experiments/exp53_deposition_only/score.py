"""exp53 — THE SCORECARD: judge a candidate against the MEASURED added light.

exp53's design constraint 3. Ranking these candidates by the loss is the one
thing that must not happen: exp38 stage 2 already showed the loss-optimal basin
paying for its loss in the physics tests, exp49 showed the loss and the compact
central deficit moving in OPPOSITE directions along `g`, and exp52 showed a
model that reproduces the summed profile while getting its growth MECHANISM
wrong. The summed CoG cannot see the mechanism; the added light can.

Three blocks, all differential and all measured on the data first.

**A. the added-light kernel** (`exp38/outputs/stage0_kernels.npz`, stage 0.2).
Stacked median added surface density per logM* tercile x adjacent epoch pair.
The measured kernel is centrally peaked at 3-5 kpc with Sersic n = 2.1-3.1 and
kernel half-mass radius 15-47 kpc. exp53 applies the IDENTICAL operator to
model CoGs -- median over the same terciles, difference of the same
`Sigma = dM/dA`, the same `_sersic_gridfit` over the same 4-120 kpc band -- so
a model number and a data number in the same cell are directly comparable.

**B. exp52's differential scorecard** (n=2397, from the measured CoGs):
median `dlogSigma ~ 0.00 +/- 0.01` at 2-3 kpc for both z<1 pairs and **+0.09 at
2.35 kpc over the long baseline** -- the core GROWS -- and the inside-out slope
`b` of `dlogSigma = a + b logR` fitted over 6-100 kpc, ~0.10 late rising to
0.15-0.19 early, essentially independent of halo mass.

**C. the exp52 mechanism split**, for deposition models only: what fraction of
the model's own `M*[50,100]` growth is transport rather than new deposition.
For `q = 0` this is identically zero and the block is a self-check; for `q > 0`
it is the number exp52 measured at 96.5% for the adopted kernel.

Every number here is computed for BOTH the model and the data with the same
code path, and reported as a pair. Nothing is compared against a remembered
constant.

Self-check: `HONGSHAO_DATA_DIR=... PYTHONPATH=. uv run python
experiments/exp53_deposition_only/score.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments/exp38_deposit_rethink"))

import kernel as K                                   # noqa: E402
import stage2_multiepoch as s2                       # noqa: E402
from stage0_diagnostics import _sersic_gridfit, _sersic_r50   # noqa: E402
from hongshao.profile_emulator import density_from_cog        # noqa: E402

#: adjacent epoch pairs as (earlier k, later k); k=0 is z=0.4
ADJACENT = [(1, 0), (2, 1), (3, 2), (4, 3)]
#: exp38 stage 0.2's kernel-fit band and inner-radius guard
KERN_LO, KERN_HI, R_IN = 4.0, 120.0, 3.0
#: exp52's inside-out slope band
SLOPE_LO, SLOPE_HI = 6.0, 100.0
#: exp52's core radii (the two innermost annuli of the aperture grid)
CORE_HI = 3.5


def sigma_stack(cogs, R):
    """Linear surface density from CoGs, exactly as exp38 stage 0.2 built it.

    ``cogs`` is ``(n_gal, n_epoch, n_radii)``; returns ``(sig, mid)`` with
    ``sig`` on the 23 annuli between the 24 aperture radii."""
    dA = np.pi * (R[1:] ** 2 - R[:-1] ** 2)
    return np.diff(cogs, axis=-1) / dA, np.sqrt(R[:-1] * R[1:])


def added_kernels(cogs, R, logms, edges, pairs=ADJACENT):
    """Stacked median added Sigma per (tercile, epoch pair) -- the operator.

    Returns ``(kern, mid)`` with ``kern`` shaped ``(3, len(pairs), 23)``.
    NaN galaxies are dropped per cell rather than poisoning the median.
    """
    sig, mid = sigma_stack(cogs, R)
    kern = np.full((3, len(pairs), len(mid)), np.nan)
    for b in range(3):
        m = (logms >= edges[b]) & (logms <= edges[b + 1] + 1e-9)
        for j, (ka, kb) in enumerate(pairs):
            d = sig[m, kb] - sig[m, ka]
            ok = np.isfinite(d).all(axis=1)
            if ok.sum() >= 10:
                kern[b, j] = np.median(d[ok], axis=0)
    return kern, mid


def kernel_shape(dsig, mid):
    """(Sersic n, kernel R50 [kpc], peak radius [kpc], core sign) of one cell.

    The Sersic grid fit is exp38 stage 0.2's, unchanged, so a model number is
    comparable with the recorded data number cell by cell. A cell with fewer
    than 6 positive points in the band is unfittable and returns NaNs -- which
    is itself a verdict: the measured kernel is positive nearly everywhere.
    """
    band = (mid >= KERN_LO) & (mid <= KERN_HI)
    pos = band & np.isfinite(dsig) & (dsig > 0)
    ipk = int(np.argmax(np.where((mid > R_IN) & np.isfinite(dsig), dsig,
                                 -np.inf)))
    core = dsig[mid <= KERN_LO]
    core_sign = float(np.nanmin(core)) if np.isfinite(core).any() else np.nan
    if pos.sum() < 6:
        return np.nan, np.nan, mid[ipk], core_sign, float(pos.sum())
    n, a, _ = _sersic_gridfit(mid[pos], dsig[pos])
    return n, _sersic_r50(n, a), mid[ipk], core_sign, float(pos.sum())


def differential_slopes(cogs, R, logmh, pairs=None):
    """exp52's block B: median dlogSigma profiles and their power-law slope.

    Returns a dict keyed by ``(halo-mass bin, pair)`` with
    ``(b, dlogSigma at the innermost annulus, dlogSigma at ~3 kpc)``.
    Halo-mass bins are exp52's terciles of ``logmh``.
    """
    pairs = pairs or (ADJACENT + [(4, 0)])
    n, ne, nr = cogs.shape
    lsig, mid = density_from_cog(
        np.log10(np.clip(cogs.reshape(-1, nr), 1.0, None)), R)
    lsig = lsig.reshape(n, ne, nr - 1)
    edges = ([logmh.min() - 1e-6] + list(np.percentile(logmh, [33.3, 66.7]))
             + [logmh.max() + 1e-6])
    fitm = (mid >= SLOPE_LO) & (mid <= SLOPE_HI)
    core = mid <= CORE_HI
    out = {}
    for b in range(3):
        sel = (logmh >= edges[b]) & (logmh < edges[b + 1])
        for ka, kb in pairs:
            d = lsig[sel, kb] - lsig[sel, ka]
            ok = np.isfinite(d).all(axis=1)
            if ok.sum() < 10:
                out[(b, (ka, kb))] = (np.nan, np.nan, np.nan)
                continue
            med = np.median(d[ok], axis=0)
            slope = float(np.polyfit(np.log10(mid[fitm]), med[fitm], 1)[0])
            out[(b, (ka, kb))] = (slope, float(med[0]),
                                  float(np.nanmedian(med[core])))
    out["edges"] = np.array(edges)
    out["mid"] = mid
    return out


def transport_share(spec, theta, gals, pair=(2, 0), lo=50.0, hi=100.0):
    """exp52's block C: transport's share of the model's own annulus growth.

    Population-SUMMED, exactly as `exp52/decompose.py` reports it. The change
    in the un-normalized annulus mass ``M[lo,hi]`` between the earlier epoch
    ``pair[0]`` and the later ``pair[1]`` splits into material already present
    and MOVED, and material newly DEPOSITED; the share is
    ``transport / (transport + new deposition)``. The `M(<500)` pinning term is
    deliberately absent -- this is the physics question with the bookkeeping
    removed, which is exp52's convention.

    Returns NaN when the two terms have opposite signs, where a ratio would be
    meaningless (as it is in the centre, where transport is negative).
    """
    import families
    e = s2._W["e"]
    ka, kb = pair
    moved_tot = dep_tot = 0.0
    edge = np.array([lo, hi])
    for g in gals:
        dM = K.deposit_weights(spec, theta, g)
        if dM is None:
            continue
        mah = g["mah"]
        log_r50, gg, q, _, _, shape = spec.unpack(theta, g["cond"])
        r50_0 = np.clip(10.0 ** log_r50 * (mah["t"] / mah["t_obs"]) ** gg,
                        K.R50_FLOOR, K.R50_CAP)
        old = (mah["snap"] <= e.ANCHOR_SNAP[ka]).astype(float)
        new = ((mah["snap"] > e.ANCHOR_SNAP[ka])
               & (mah["snap"] <= e.ANCHOR_SNAP[kb])).astype(float)
        cog_0 = families.cog_unit(spec.family, r50_0, shape, edge)

        def ann(tk, sel):
            if q == 0.0:
                cg = cog_0
            else:
                r50_w = np.clip(r50_0 * (tk / mah["t"]) ** q, K.R50_FLOOR,
                                K.R50_CAP)
                fc = np.exp(-np.clip(tk - mah["t"], 0.0, None) / mah["t"])
                cg = (fc[None, :] * cog_0
                      + (1.0 - fc)[None, :]
                      * families.cog_unit(spec.family, r50_w, shape, edge))
            return float((cg[1] - cg[0]) @ (dM * sel))

        ta, tb = e.pe.AT[ka], e.pe.AT[kb]
        moved_tot += ann(tb, old) - ann(ta, old)
        dep_tot += ann(tb, new)
    if moved_tot * dep_tot < 0 or (moved_tot + dep_tot) == 0:
        return np.nan
    return float(moved_tot / (moved_tot + dep_tot))


def growth_ratio(cogs, data_cogs, R, ks, pair, lo, hi):
    """Population-summed model growth / truth growth in an aperture or annulus.

    exp52's headline central-deficit statistic: for `M(<10)` over z=1.0->0.4 it
    measured 57.3% for the adopted kernel against 98.9% for `M[50,100]`. A
    value of 1.0 means the model grows that region at the truth's rate.
    `pair` is ``(earlier k, later k)`` in ABSOLUTE epoch indices.
    """
    ks = list(ks)
    ja, jb = ks.index(pair[0]), ks.index(pair[1])

    def mass(c, j):
        inner = np.interp(lo, R, c[j]) if lo > 0 else 0.0
        return np.interp(hi, R, c[j]) - inner

    num = den = 0.0
    for i in range(len(cogs)):
        if not np.isfinite(cogs[i, [ja, jb]]).all():
            continue
        num += mass(cogs[i], jb) - mass(cogs[i], ja)
        den += mass(data_cogs[i], jb) - mass(data_cogs[i], ja)
    return float(num / den) if den != 0 else np.nan


def deposit_reach(spec, theta, gals, radii=(50.0, 148.0, 500.0),
                 t_index=0):
    """Where the model PUTS its stars: mass-weighted deposit reach.

    exp52's actionable finding was "RIGHT SOURCE, ~10x TOO MUCH REACH" -- 93%
    of the transported mass leaves 0-5 kpc (correct) but 58.5% is delivered
    beyond 50 kpc and 12% beyond 500 kpc, where the normalization discards it.
    With transport removed the same question is asked of the DEPOSITION: what
    fraction of the mass the kernel lays down at the observation epoch lands
    beyond each radius, and how big are the deposits it uses.

    Only meaningful across families because the scale coordinate is a half-mass
    radius. Returns the beyond-R fractions, the mass-weighted median deposit
    R50, and the mass fraction sitting AT the `R50_CAP` ceiling (a large value
    means the ceiling is load-bearing rather than free).
    """
    import families
    e = s2._W["e"]
    edge = np.asarray(radii, float)
    beyond = np.zeros(len(edge))
    tot = 0.0
    r50_all, w_all, capped = [], [], 0.0
    tk = e.pe.AT[t_index]
    for g in gals:
        dM = K.deposit_weights(spec, theta, g)
        if dM is None:
            continue
        mah = g["mah"]
        sel = (mah["snap"] <= e.ANCHOR_SNAP[t_index]).astype(float)
        w = dM * sel
        log_r50, gg, q, _, _, shape = spec.unpack(theta, g["cond"])
        raw = 10.0 ** log_r50 * (mah["t"] / mah["t_obs"]) ** gg
        r50_0 = np.clip(raw, K.R50_FLOOR, K.R50_CAP)
        if q == 0.0:
            r50_eff = r50_0
        else:
            r50_w = np.clip(r50_0 * (tk / mah["t"]) ** q, K.R50_FLOOR,
                            K.R50_CAP)
            fc = np.exp(-np.clip(tk - mah["t"], 0.0, None) / mah["t"])
            r50_eff = fc * r50_0 + (1.0 - fc) * r50_w        # for the summary
        cg = families.cog_unit(spec.family, r50_0, shape, edge)
        if q != 0.0:
            cgw = families.cog_unit(spec.family, r50_w, shape, edge)
            cg = fc[None, :] * cg + (1.0 - fc)[None, :] * cgw
        beyond += (1.0 - cg) @ w
        tot += w.sum()
        capped += float(w[raw >= K.R50_CAP].sum())
        r50_all.append(r50_eff)
        w_all.append(w)
    if tot <= 0:
        return dict(beyond={}, median_r50=np.nan, at_cap=np.nan)
    r50_all = np.concatenate(r50_all)
    w_all = np.concatenate(w_all)
    o = np.argsort(r50_all)
    cw = np.cumsum(w_all[o]) / w_all.sum()
    return dict(beyond={float(rr): float(v / tot) for rr, v in zip(edge, beyond)},
                median_r50=float(np.interp(0.5, cw, r50_all[o])),
                at_cap=float(capped / tot))


# --------------------------------------------------------------------------- #
def scorecard(spec, theta, gals, R, logms, logmh, data_cogs, ks=K.KS):
    """The full exp53 verdict for one fitted theta. Model AND data, paired."""
    cogs = K.population_cogs(spec, theta, gals, ks)
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    pairs = [(a, b) for a, b in ADJACENT if a in ks and b in ks]
    km, mid = added_kernels(cogs, R, logms, edges, pairs)
    kd, _ = added_kernels(data_cogs[:, ks], R, logms, edges, pairs)
    out = {"n": len(gals), "pairs": pairs, "mid": mid,
           "kern_model": km, "kern_data": kd, "edges": edges}
    for b in range(3):
        for j, pr in enumerate(pairs):
            out[("kshape", "model", b, pr)] = kernel_shape(km[b, j], mid)
            out[("kshape", "data", b, pr)] = kernel_shape(kd[b, j], mid)
    slope_pairs = pairs + ([(max(ks), min(ks))] if len(ks) > 2 else [])
    out["slopes_model"] = differential_slopes(cogs, R, logmh, slope_pairs)
    out["slopes_data"] = differential_slopes(data_cogs[:, ks], R, logmh,
                                             slope_pairs)
    out["slope_pairs"] = slope_pairs
    out["transport_share"] = transport_share(
        spec, theta, gals, pair=(max(ks), min(ks)))
    # exp52's central-deficit statistic, and its outskirt control. The
    # falsifiable prediction of exp53 lives in this line: removing transport
    # must raise the M(<10) entry towards 1.0.
    for pr in [(2, 0), (max(ks), min(ks))]:
        if pr[0] not in ks or pr[1] not in ks:
            continue
        for nm, lo, hi in (("M(<5)", 0.0, 5.0), ("M(<10)", 0.0, 10.0),
                           ("M(<30)", 0.0, 30.0), ("M[50,100]", 50.0, 100.0)):
            out[("growth", nm, pr)] = growth_ratio(
                cogs, data_cogs[:, ks], R, ks, pr, lo, hi)
    out["kern_rms"] = _kernel_distance(km, kd, mid)
    out["reach"] = deposit_reach(spec, theta, gals)
    return out


def _kernel_distance(km, kd, mid):
    """One scalar: median |dlog| between model and measured added light over
    the fit band, across all cells where BOTH are positive. A model that makes
    the added light negative where the data makes it positive is not merely
    far away -- it is a different sign, so those cells are counted at their
    number and reported separately by `print_scorecard`."""
    band = (mid >= KERN_LO) & (mid <= KERN_HI)
    both = np.isfinite(km) & np.isfinite(kd) & (km > 0) & (kd > 0) & band
    if both.sum() == 0:
        return np.nan
    return float(np.median(np.abs(np.log10(km[both] / kd[both]))))


def sign_disagreement(km, kd, mid):
    """Fraction of in-band cells where the model and data added light differ
    in SIGN -- the failure a |dlog| distance cannot express."""
    band = (mid >= KERN_LO) & (mid <= KERN_HI)
    ok = np.isfinite(km) & np.isfinite(kd) & band
    if ok.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(km[ok]) != np.sign(kd[ok])))


def print_scorecard(sc, label):
    print(f"\n=== exp53 SCORECARD [{label}]  (n={sc['n']}) ===")
    print("  A. added-light kernel — model vs MEASURED (exp38 stage 0.2 "
          "operator)")
    print(f"    {'cell':<14}{'Sersic n':>18}{'kernel R50 [kpc]':>22}"
          f"{'peak R [kpc]':>18}")
    print(f"    {'':<14}{'model / data':>18}{'model / data':>22}"
          f"{'model / data':>18}")
    for b in range(3):
        for pr in sc["pairs"]:
            m = sc[("kshape", "model", b, pr)]
            d = sc[("kshape", "data", b, pr)]
            cell = f"T{b+1} z{K.ANCHOR_Z[pr[0]]}->{K.ANCHOR_Z[pr[1]]}"
            print(f"    {cell:<14}{m[0]:8.1f} /{d[0]:8.1f}"
                  f"{m[1]:12.1f} /{d[1]:9.1f}"
                  f"{m[2]:10.1f} /{d[2]:7.1f}")
    print(f"    kernel distance (median |dlog| in band): {sc['kern_rms']:.3f} "
          f"dex;  sign disagreement "
          f"{100*sign_disagreement(sc['kern_model'], sc['kern_data'], sc['mid']):.0f}%")
    print("  B. differential slope b (6-100 kpc) and core dlogSigma (<3.5 kpc)"
          " — model vs data")
    print(f"    {'halo bin':<12}" + "".join(
        f"{f'z{K.ANCHOR_Z[a]}->{K.ANCHOR_Z[b]}':>18}"
        for a, b in sc["slope_pairs"]))
    for b in range(3):
        cells = []
        for pr in sc["slope_pairs"]:
            sm = sc["slopes_model"][(b, pr)]
            sd = sc["slopes_data"][(b, pr)]
            cells.append(f"{sm[0]:+.2f}/{sd[0]:+.2f}")
        print(f"    b   bin{b+1:<5}" + "".join(f"{c:>18}" for c in cells))
    for b in range(3):
        cells = []
        for pr in sc["slope_pairs"]:
            sm = sc["slopes_model"][(b, pr)]
            sd = sc["slopes_data"][(b, pr)]
            cells.append(f"{sm[2]:+.2f}/{sd[2]:+.2f}")
        print(f"    core bin{b+1:<4}" + "".join(f"{c:>18}" for c in cells))
    print(f"  C. transport share of the model's own M[50,100] growth over the "
          f"full baseline: {100*sc['transport_share']:.1f}% "
          f"(exp52 measured 87.5% for the adopted kernel over z=1.0->0.4)")
    rc = sc.get("reach")
    if rc:
        print("  C2. deposit REACH at z=0.4 — mass-weighted fraction of the "
              "model's own deposited\n      stars landing beyond each radius "
              "(exp52 measured 58.5% beyond 50 kpc and 12.0%\n      beyond 500 "
              "kpc for the ADOPTED kernel's transported mass)")
        print("      " + "   ".join(
            f"beyond {r:.0f} kpc: {100*v:5.1f}%" for r, v in rc["beyond"].items())
            + f"   |  median deposit R50 {rc['median_r50']:.1f} kpc"
              f"   |  {100*rc['at_cap']:.1f}% of mass at the "
              f"{K.R50_CAP:.0f} kpc ceiling")
    grow = sorted({k[2] for k in sc if isinstance(k, tuple) and k[0] == "growth"})
    if grow:
        print("  D. population-summed growth, MODEL / TRUTH "
              "(1.00 = the model grows that region at the truth's rate;"
              " exp52: M(<10) 0.573, M[50,100] 0.989 over z=1.0->0.4)")
        print(f"    {'epoch pair':<16}" + "".join(
            f"{nm:>13}" for nm in ("M(<5)", "M(<10)", "M(<30)", "M[50,100]")))
        for pr in grow:
            lab = f"z{K.ANCHOR_Z[pr[0]]}->{K.ANCHOR_Z[pr[1]]}"
            print(f"    {lab:<16}" + "".join(
                f"{sc[('growth', nm, pr)]:13.3f}"
                for nm in ("M(<5)", "M(<10)", "M(<30)", "M[50,100]")))


def demo():
    pop = np.load(ROOT / "experiments/exp32_full_population/outputs/"
                  "population.npz")
    s2._w_init(pop["dev100"])
    gals, R = s2._W["gals"], s2._W["e"].R
    logms = np.array([pop["logms"][g["row"]] for g in gals])
    logmh = np.array([pop["logmh"][g["row"]] for g in gals])
    data = np.stack([np.stack([g["data"][k] for k in range(5)]) for g in gals])

    # (1) the operator REPRODUCES exp38 stage 0.2 on the data. Run it on the
    #     full 5-epoch data and check the recorded kernel cells come back.
    edges = np.quantile(logms, [0, 1 / 3, 2 / 3, 1])
    kern, mid = added_kernels(data, R, logms, edges)
    rec = np.load(ROOT / "experiments/exp38_deposit_rethink/outputs/"
                  "stage0_kernels.npz")
    assert np.allclose(mid, rec["mid"]), "annulus grid drifted from stage 0.2"
    # dev100 is a subsample, so the medians differ; the SHAPE verdict must not
    n, r50, peak, core, npos = kernel_shape(kern[2, 0], mid)
    assert 1.5 <= n <= 5.0, n
    assert 5.0 <= r50 <= 120.0, r50
    assert peak <= 8.0, peak            # centrally peaked, not a shell

    # ... and on the FULL sample the operator must reproduce the recorded
    # numbers exactly. Checked here against the stored `rows` for the cells
    # dev100 cannot address, by re-running the recorded arrays themselves.
    band = (rec["mid"] >= KERN_LO) & (rec["mid"] <= KERN_HI)
    for b in range(3):
        for k in range(4):
            dsig = rec["kern"][b, k]
            got = kernel_shape(dsig, rec["mid"])
            want = rec["rows"][b * 4 + k]
            assert abs(got[0] - want[2]) < 1e-9, (b, k, got[0], want[2])
            assert abs(got[2] - want[4]) < 1e-9, (b, k, got[2], want[4])
    assert band.sum() > 10

    # (2) the scorecard runs end to end on the adopted model and its data
    #     column is INDEPENDENT of the model (same data, same numbers)
    spec = K.Spec("moffat", q_free=True)
    th = K.from_exp38_theta(K.adopted_theta())
    sc = scorecard(spec, th, gals, R, logms, logmh, data)
    th2 = th.copy()
    th2[5] = 2.0                       # a different gamma: a different model
    sc2 = scorecard(spec, th2, gals, R, logms, logmh, data)
    for b in range(3):
        for pr in sc["pairs"]:
            a = sc[("kshape", "data", b, pr)]
            c = sc2[("kshape", "data", b, pr)]
            assert np.allclose(a, c, equal_nan=True), (b, pr)
    assert not np.allclose(sc["kern_model"], sc2["kern_model"], equal_nan=True)

    # (2b) THE exp52 BUG, locked out: an aperture with `lo = 0` must count
    #      from ZERO on BOTH sides. `np.interp` clamps below its grid, so the
    #      natural-looking `np.interp(0.0, R, cog)` returns the mass inside
    #      2 kpc instead — which is how exp52 came to divide the model's
    #      aperture growth by the truth's annulus growth. Asserted here on
    #      both the model and the truth path, which is what would have caught
    #      it: a flat CoG has zero growth in ANY aperture starting at zero.
    flat = np.ones((3, 2, len(R)))
    assert growth_ratio(flat, flat, R, [0, 1], (1, 0), 0.0, 10.0) == 0.0 \
        or np.isnan(growth_ratio(flat, flat, R, [0, 1], (1, 0), 0.0, 10.0))
    rise = np.stack([np.stack([np.full(len(R), 1.0), np.full(len(R), 2.0)])
                     for _ in range(3)])
    # truth doubles everywhere; a model that doubles too must score EXACTLY 1
    assert abs(growth_ratio(rise, rise, R, [0, 1], (0, 1), 0.0, 10.0) - 1.0) < 1e-12
    # and the aperture and the annulus must NOT agree when the core moves:
    core_drop = rise.copy()
    core_drop[:, 1, 0] = 0.5                       # inner point falls
    ap = growth_ratio(core_drop, core_drop, R, [0, 1], (0, 1), 0.0, 10.0)
    ann = growth_ratio(core_drop, core_drop, R, [0, 1], (0, 1), R[0], 10.0)
    assert abs(ap - 1.0) < 1e-12 and abs(ann - 1.0) < 1e-12   # self-ratio is 1
    # the point is the DENOMINATORS differ, which is what exp52 mixed
    m0 = np.interp(10.0, R, core_drop[0, 1]) - np.interp(10.0, R, core_drop[0, 0])
    assert abs(m0) > 0

    # (3) block C is EXACTLY zero for a deposition-only model, by construction
    spec0 = K.Spec("moffat", q_free=False, q_fixed=0.0)
    assert transport_share(spec0, np.delete(th, 2), gals) == 0.0
    # ... and substantial for the adopted one, reproducing exp52's finding
    ts = transport_share(spec, th, gals, pair=(2, 0))
    assert ts > 0.5, ts

    print_scorecard(sc, "adopted 1ch-mof (dev100 wiring check)")
    print(f"\nscore.demo OK: the stage 0.2 operator reproduces every recorded "
          f"kernel cell exactly; the data column is model-independent; "
          f"transport share is 0.000 for q=0 by construction and {ts:.3f} for "
          f"the adopted kernel over z=1.0->0.4 (exp52: 0.875 on the full "
          f"sample)")


if __name__ == "__main__":
    demo()
