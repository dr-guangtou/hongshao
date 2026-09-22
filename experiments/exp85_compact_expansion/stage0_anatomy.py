"""exp85 Stage 0, the anatomy of the centre in THE ADOPTED MEAN: which channel
and which deposit AGES make up M*(<4.9 kpc) at z = 0.4 and z = 2, for the
galaxies whose true centre declined from z = 2 to 0.4 and for the rest.

An age-driven expansion of the compact deposits can only produce the
decliners' central loss if the decliners' centres hold OLDER compact stars
than the non-decliners' (and the aperture sees them expand). This script
measures that premise directly, per galaxy, from the adopted model's
deposits (`size_law.deposits_law` + the compact kernel):
  - the compact and extended channels' shares of M(<4.9);
  - the mass-weighted mean log10(t_obs / t') of the compact deposits INSIDE
    4.9 kpc (their age in the expansion's own variable), and the same for
    the whole compact channel;
  - the fraction of the compact mass inside 4.9 kpc deposited before 2 Gyr;
  - the compact deposits' sizes inside 4.9 kpc (mass-weighted median), i.e.
    whether a x1.5 expansion of a 2 kpc deposit is visible to the aperture;
  - the model's per-galaxy response of log M(<4.9) to q_c = 0.3 at frozen
    constants, and its rank correlation with the truth's central change.

Run:
    HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 PYTHONPATH=. \\
        uv run python -u experiments/exp85_compact_expansion/stage0_anatomy.py [--smoke]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", ROOT / "experiments/exp63_analytic_growth",
          ROOT / "experiments/exp73_size_relative", ROOT / "experiments/exp74_c19_history_leak",
          ROOT / "experiments/exp78_size_aware_objective", ROOT / "experiments/exp80_deposit_size_law", HERE):
    sys.path.insert(0, str(p))
import fit as F                                          # noqa: E402
import engine as E                                       # noqa: E402
import model as M                                        # noqa: E402
import model2 as M2                                      # noqa: E402
import size_law as SL                                    # noqa: E402


def _by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S1 = _by_path("exp80_stage1_fit", ROOT / "experiments/exp80_deposit_size_law/stage1_fit.py")
RB = S1.RB
RULE = "=" * 100
ANCHOR_Z = list(E.ANCHOR_Z)
KNOBS = ("q_e", "q_c", "q_ch")
R_CENTRE = 4.92
READ_EPOCHS = (0, 4)


def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, 0.5 * c[-1])]) if c[-1] > 0 else np.nan


def main(smoke=False):
    print(f"{RULE}\nexp85 STAGE 0 — the anatomy of the centre in the adopted mean: channels and deposit ages inside 4.9 kpc, by the central decline"
          f"{' (SMOKE)' if smoke else ''}\n{RULE}\n")
    recs, data, mask, lmh_dm, lmh_bins, meas, fz, spec2, th_inc, th_nested, pr, lp, bounds = S1.build(smoke, KNOBS, True, "step")
    ad = RB.adopted_mean()
    th_ad = np.r_[ad["theta_full"], 0.0, 0.0]
    th13, law = lp.split(th_ad)
    p = spec2.unpack(th13)
    rows = pr.all_rows
    truth_all = data[rows]
    good = np.isfinite(truth_all).all(axis=(1, 2)) & (truth_all > 0).all(axis=(1, 2))
    truth = truth_all[good]
    cvg = [c for c, g in zip([meas[i] for i in rows], good) if g]
    i_c = int(np.argmin(np.abs(F.R_GRID - R_CENTRE)))
    dec = truth[:, 4, i_c] > truth[:, 0, i_c]
    dch_t = np.log10(truth[:, 0, i_c] / truth[:, 4, i_c])
    n = len(cvg)
    print(f"  {n} galaxies; {100 * dec.mean():.0f}% declined at 4.9 kpc from z=2 to 0.4; the compact kernel n_c {p['n_c']:.2f}, "
          f"log_f_c {p['log_f_c']:+.3f}, b_c {p['b_c']:+.3f}\n")
    lt, w, include, _ = E.nodes(**M2.FULL_NODES)
    R1 = np.array([R_CENTRE])
    out = {}
    for lo in range(0, n, 300):
        cv = cvg[lo:lo + 300]
        dm_c, dm_e, s_c, s_e, r_tr = SL.deposits_law(spec2, th13, law, cv, lt, READ_EPOCHS)
        nb, N = dm_c.shape
        r200 = np.array([E.r200c_of(hc, lt, "analytic") for hc in cv])
        r_tr_c = spec2.trunc_C * r200
        Bc = M.cog_truncated(M2.COMPACT_FAMILY, (p["n_c"],), s_c.ravel(), r_tr_c.ravel(), R1).reshape(nb, N)
        inc_e = SL.arrival_weights(spec2, p, law, lt, include)
        for k in READ_EPOCHS:
            Be = M.cog_truncated(spec2.extended_family, (p["c_e"],), s_e[k].ravel(), r_tr[k].ravel(), R1).reshape(nb, N)
            mc_node = dm_c * w[None, :] * include[k][None, :] * Bc          # compact mass inside 4.9 kpc per node
            me_node = dm_e * w[None, :] * inc_e[k][None, :] * Be
            mc_all = dm_c * w[None, :] * include[k][None, :]
            age = np.maximum(np.log10(E.T_ANCHOR[k]) - lt, 0.0)[None, :] * np.ones((nb, 1))
            before2 = (lt < np.log10(SL.T_EARLY_GYR))[None, :] * np.ones((nb, 1))
            d = out.setdefault(k, {kk: [] for kk in ("mc", "me", "age_in", "age_all", "f_before2", "s_in", "s_all")})
            d["mc"].append(mc_node.sum(1))
            d["me"].append(me_node.sum(1))
            d["age_in"].append((mc_node * age).sum(1) / mc_node.sum(1))
            d["age_all"].append((mc_all * age).sum(1) / mc_all.sum(1))
            d["f_before2"].append((mc_node * before2).sum(1) / mc_node.sum(1))
            d["s_in"].append([wmedian(s_c[i], mc_node[i]) for i in range(nb)])
            d["s_all"].append([wmedian(s_c[i], mc_all[i]) for i in range(nb)])
    for k in READ_EPOCHS:
        d = {kk: np.concatenate([np.asarray(v, float) for v in vals]) for kk, vals in out[k].items()}
        out[k] = d
    # the model's own response to q_c at frozen constants
    m0 = lp.predict(th_ad, F.R_GRID, nodes=M2.FULL_NODES)[good]
    th_q = th_ad.copy()
    th_q[list(lp.names).index("q_c")] = 0.3
    m1 = lp.predict(th_q, F.R_GRID, nodes=M2.FULL_NODES)[good]
    resp = {k: np.log10(m1[:, k, i_c] / m0[:, k, i_c]) for k in READ_EPOCHS}
    print(f"  {'reading (median; decliners | non-decliners)':<70}{'z = 0.4':>22}{'z = 2':>22}")

    def row(label, arrs, fmt="+.3f"):
        cells = []
        for k in READ_EPOCHS:
            a = arrs[k]
            cells.append(f"{np.nanmedian(a[dec]):{fmt}} | {np.nanmedian(a[~dec]):{fmt}}")
        print(f"  {label:<70}{cells[0]:>22}{cells[1]:>22}")
    row("compact share of M(<4.9), M_c / (M_c + M_e)", {k: out[k]["mc"] / (out[k]["mc"] + out[k]["me"]) for k in READ_EPOCHS}, ".2f")
    row("log M_c(<4.9) [Msun]", {k: np.log10(out[k]["mc"]) for k in READ_EPOCHS})
    row("log M_e(<4.9) [Msun]", {k: np.log10(out[k]["me"]) for k in READ_EPOCHS})
    row("mean log10(t_obs / t') of the compact stars INSIDE 4.9 kpc (the expansion's variable)", {k: out[k]["age_in"] for k in READ_EPOCHS})
    row("mean log10(t_obs / t') of the WHOLE compact channel", {k: out[k]["age_all"] for k in READ_EPOCHS})
    row("fraction of the compact mass inside 4.9 kpc deposited before 2 Gyr", {k: out[k]["f_before2"] for k in READ_EPOCHS}, ".2f")
    row("compact deposit size inside 4.9 kpc, mass-weighted median [kpc]", {k: out[k]["s_in"] for k in READ_EPOCHS}, ".2f")
    row("compact deposit size, whole channel, mass-weighted median [kpc]", {k: out[k]["s_all"] for k in READ_EPOCHS}, ".2f")
    row("the model's response of log M(<4.9) to q_c = 0.3 at frozen constants [dex]", resp)
    print()
    for k in READ_EPOCHS:
        print(f"  z={ANCHOR_Z[k]}: rank correlation of the q_c response with the truth's central change z=2->0.4 (a decline is negative): "
              f"{spearmanr(resp[k], dch_t)[0]:+.2f}; with the compact share {spearmanr(resp[k], out[k]['mc'] / (out[k]['mc'] + out[k]['me']))[0]:+.2f}; "
              f"with the inside age {spearmanr(resp[k], out[k]['age_in'])[0]:+.2f}; with the inside size {spearmanr(resp[k], out[k]['s_in'])[0]:+.2f}")
    print(f"  the DIFFERENTIAL response z=0.4 minus z=2 (what a decline needs to be negative): decliners {np.median(resp[0][dec] - resp[4][dec]):+.4f}, "
          f"non-decliners {np.median(resp[0][~dec] - resp[4][~dec]):+.4f}; rank correlation with the truth's change {spearmanr(resp[0] - resp[4], dch_t)[0]:+.2f}")
    print(f"  the truth's central change: decliners {np.median(dch_t[dec]):+.3f}, non-decliners {np.median(dch_t[~dec]):+.3f}; "
          f"the inside age's rank correlation with the truth's change at z=0.4: {spearmanr(out[0]['age_in'], dch_t)[0]:+.2f}")
    np.savez(HERE / "outputs" / f"stage0_anatomy{'_smoke' if smoke else ''}.npz", dec=dec, dch_truth=dch_t,
             **{f"{kk}_{k}": out[k][kk] for k in READ_EPOCHS for kk in out[k]}, **{f"resp_{k}": resp[k] for k in READ_EPOCHS})


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
