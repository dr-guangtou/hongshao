"""exp30 phase 4 — the population/forward step: halo-only inputs -> theta -> profile.

Turns the per-galaxy-fitted 7-param transport emulator (param_emulator.npz) into the
goal emulator (MAH, halo props) -> theta -> shape x SHMR -> M*(<R, z_k):
  (i)   UNIVERSAL theta: one global 7-param set, leave-galaxy-out (LOGO) evaluated;
        the train-median-theta predictor is the no-refit reference. Measures whether
        the MAH alone carries the per-galaxy individuality.
  (ii)  HALO-CONDITIONING only where the 45 fitted theta correlate with halo-only
        props (logMh, c200c, t50, fz2, burstiness) — exp29 phase4e pattern.
  (iii) END-TO-END LOGO error: predicted-theta shape x per-epoch SHMR amplitude
        (logM*_k <- logMh(z_k) interpolated from the MAH itself), vs the per-galaxy
        in-sample floor (9.7%). LOEO 24% was epoch-generalization; THIS is the
        new-halo number.
Two configs: real de-dipped MAH = REFERENCE; DiffMAH-fit-curve input = differentiable
variant to quantify (validate, don't assume equivalence).

Run: PYTHONPATH=. uv run python experiments/exp30_transport_kernel/pop_forward.py \
       [n] [--refit] [--config real|diffmah|both]
Demo: ... pop_forward.py demo
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP29 = ROOT / "experiments" / "exp29_outer_deposit"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP29))
import param_emulator as pe                                                          # noqa: E402
from run import dipfree_mah, ANCHOR_SNAP, ANCHOR_Z, TABLE                            # noqa: E402
from real_mah_test import real_mah                                                   # noqa: E402
import mass_qa                                                                        # noqa: E402
from hongshao.plotting import set_style, save_fig                                    # noqa: E402

set_style()
FIGDIR, OUTDIR = HERE / "figures", HERE / "outputs"
PARAM_NPZ = OUTDIR / "param_emulator.npz"
OUT_NPZ = OUTDIR / "pop_forward.npz"
Z = np.array(ANCHOR_Z)
CONFIGS = ["real", "diffmah"]
MAHFUN = {"real": real_mah, "diffmah": dipfree_mah}
PROPS = ["logmh", "c200c", "t50", "fz2", "burst"]
THETA_NAMES = ["log_s0", "g", "log_alpha", "q", "b_early", "b_late", "z_c"]
# liberal selection, strict promotion: with 35 tests p<0.05 admits ~2 chance pairs,
# but a pair only survives if it improves the LEAVE-GALAXY-OUT error
P_SELECT, MAX_PAIRS = 0.05, 3


# %% ---- sample assembly ------------------------------------------------------
def load_gals(config, n):
    """Galaxy dicts for one MAH config: MAH + 5-epoch CoG + halo-only props."""
    d = np.load(PARAM_NPZ)
    from astropy.table import Table
    t = Table.read(TABLE)
    row = {int(g): k for k, g in enumerate(np.asarray(t["index"]))}
    gals = []
    for i, gi in enumerate(d["index"][:n]):
        gi = int(gi)
        mah = MAHFUN[config](gi)
        if mah is None or gi not in row:
            print(f"  [skip] galaxy {gi}: no {config} MAH")
            continue
        Mh, tf = 10.0 ** mah["logMh_full"], mah["t_full"]
        data = d["data"][i]
        gals.append(dict(
            gi=gi, mah=mah, data=data, theta_fit=d["params"][i],
            logmh=float(t["logmh_z0p4"][row[gi]]), c200c=float(t["c_200c"][row[gi]]),
            t50=float(np.interp(0.5 * Mh[-1], Mh, tf)),
            fz2=float(10.0 ** (np.interp(33, mah["snap_full"], mah["logMh_full"])
                               - np.log10(Mh[-1]))),
            burst=float(d["burst"][i]),
            logmh_zk=np.interp(ANCHOR_SNAP, mah["snap_full"], mah["logMh_full"]),
            logms_zk=np.log10(data[:, -1])))
    for p in PROPS:                                        # z-scores for conditioning
        v = np.array([g[p] for g in gals])
        mu, sd = np.nanmean(v), np.nanstd(v) + 1e-9
        for g in gals:
            g["z_" + p] = float((g[p] - mu) / sd) if np.isfinite(g[p]) else 0.0
        gals[0].setdefault("prop_scale", {})[p] = (mu, sd)
    return gals


# %% ---- shared-theta fitting -------------------------------------------------
def shape_maxrel(theta, gal):
    """Pinned-amplitude CoGs + per-epoch profile max|rel| (all radii)."""
    cogs, _ = pe.model_cogs(theta, gal["mah"], gal["data"])
    if cogs is None:
        return None, np.full(5, np.nan)
    D = gal["data"]
    return cogs, np.abs((cogs - D) / D).max(axis=1)


def pop_loss(theta_of, gals, ids):
    """Mean over galaxies of mean per-epoch rel-RMS (same loss as param_emulator)."""
    tot = 0.0
    for i in ids:
        g = gals[i]
        cogs, _ = pe.model_cogs(theta_of(g), g["mah"], g["data"])
        if cogs is None:
            return 1e3
        D = g["data"]
        tot += float(np.mean(np.sqrt(np.mean(((cogs - D) / D) ** 2, axis=1))))
    return tot / len(ids)


def theta_cond(p, gal, pairs):
    """Base theta (p[:7]) + one linear slope per (theta_col, prop) pair."""
    th = np.array(p[:7], float).copy()
    for m, (j, prop) in enumerate(pairs):
        th[j] += p[7 + m] * gal["z_" + prop]
    return th


def box_penalty(p):
    """Population-informed soft box on the f(z) efficiency params: keeps z_c inside
    the deposit-redshift span and the slopes non-degenerate (the unbounded universal
    fit railed z_c at the -1 guard, collapsing f(z) to one power law)."""
    lo = np.array([0.0, 0.0, 1.0])                      # b_early, b_late, z_c
    hi = np.array([10.0, 10.0, 5.0])
    v = np.asarray(p[4:7], float)
    return 10.0 * float(np.sum(np.clip(lo - v, 0, None) ** 2
                               + np.clip(v - hi, 0, None) ** 2))


def fit_shared(gals, ids, pairs, starts, maxiter, penalty=None):
    """Nelder-Mead over base theta (+slopes); returns best params."""
    def loss(p):
        v = pop_loss(lambda g: theta_cond(p, g, pairs), gals, ids)
        return v + (penalty(p) if penalty else 0.0)
    best = None
    for p0 in starts:
        r = minimize(loss, p0, method="Nelder-Mead",
                     options=dict(maxiter=maxiter, xatol=3e-4, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def logo_eval(gals, pairs, full_fit, maxiter=1500, label="", penalty=None):
    """LOGO: refit on n-1 (warm from the full-sample fit), evaluate the held galaxy.
    Returns (maxrel (n,5), cogs (n,5,24), thetas (n,7))."""
    n = len(gals)
    mr = np.full((n, 5), np.nan)
    cogs = np.full((n, 5, 24), np.nan)
    thetas = np.full((n, 7), np.nan)
    t0 = time.time()
    for i in range(n):
        ids = [j for j in range(n) if j != i]
        p = fit_shared(gals, ids, pairs, [full_fit], maxiter, penalty=penalty)
        th = theta_cond(p, gals[i], pairs)
        thetas[i] = th
        ci, mi = shape_maxrel(th, gals[i])
        mr[i] = mi
        if ci is not None:
            cogs[i] = ci
        if i == 0:
            print(f"    [{label}] fold 0: {time.time()-t0:.1f}s "
                  f"(x{n} folds ~ {(time.time()-t0)*n/60:.1f} min)")
    return mr, cogs, thetas


# %% ---- conditioning selection ------------------------------------------------
def corr_table(gals, verbose=True):
    """Spearman rho of each fitted theta column vs each halo-only prop.
    Selection: strongest prop per theta column, p < P_SELECT, at most MAX_PAIRS."""
    P = np.array([g["theta_fit"] for g in gals])
    cand = []
    if verbose:
        print("\n  (ii) fitted-theta vs halo props, Spearman rho (p):")
        print(f"    {'theta':>9s} | " + " | ".join(f"{p:>14s}" for p in PROPS))
    for j in range(7):
        row, best = [], None
        for prop in PROPS:
            v = np.array([g[prop] for g in gals])
            r, pv = spearmanr(P[:, j], v)
            row.append(f"{r:+.2f} ({pv:.3f})")
            if best is None or pv < best[2]:
                best = (j, prop, pv)
        if verbose:
            print(f"    {THETA_NAMES[j]:>9s} | " + " | ".join(f"{s:>14s}" for s in row))
        if best[2] < P_SELECT:
            cand.append(best)
    cand.sort(key=lambda b: b[2])
    pairs = [(j, prop) for j, prop, _ in cand[:MAX_PAIRS]]
    if verbose:
        print(f"    -> selected pairs (p<{P_SELECT}, max {MAX_PAIRS}): "
              + (", ".join(f"{THETA_NAMES[j]}<-{p}" for j, p in pairs) or "NONE"))
    return pairs


# %% ---- SHMR amplitude + end-to-end -------------------------------------------
def shmr_logo(gals):
    """Per-epoch LOGO linear SHMR logM*(z_k) <- logMh(z_k) (MAH-derived, halo-only).
    Returns (predicted logM* (n,5), residual scatter per epoch (5,))."""
    X = np.array([g["logmh_zk"] for g in gals])
    Y = np.array([g["logms_zk"] for g in gals])
    n = len(gals)
    pred = np.zeros_like(Y)
    for k in range(5):
        for i in range(n):
            m = np.arange(n) != i
            pred[i, k] = np.polyval(np.polyfit(X[m, k], Y[m, k], 1), X[i, k])
    return pred, np.std(pred - Y, axis=0)


def end_to_end(gals, logo_cogs, shmr_pred):
    """Absolute-CoG max|rel|: LOGO shape rescaled from data-pinned to SHMR amplitude."""
    n = len(gals)
    mr = np.full((n, 5), np.nan)
    for i in range(n):
        D = gals[i]["data"]
        scale = 10.0 ** shmr_pred[i] / D[:, -1]
        mr[i] = np.abs((logo_cogs[i] * scale[:, None] - D) / D).max(axis=1)
    return mr


# %% ---- pipeline ---------------------------------------------------------------
def compute(n, configs):
    out = {}
    for config in configs:
        print(f"\n=== config: {config} MAH ===")
        gals = load_gals(config, n)
        ng = len(gals)
        med_theta = np.median(np.array([g["theta_fit"] for g in gals]), axis=0)

        # (i) universal
        print(f"  (i) universal theta fit (n={ng}) ...")
        # med_theta is a poor start (per-galaxy fits are degenerate) but free; the
        # physical start = phase-3 population medians (alpha=1, q=0.77, f(z) medians)
        phys = np.array([2.0, 1.5, 0.0, 0.77, 4.48, 1.88, 2.23])
        univ = fit_shared(gals, list(range(ng)), [], [med_theta, phys, pe.STARTS[0]], 6000)
        mr_med = np.array([shape_maxrel(np.median(np.array(
            [gals[j]["theta_fit"] for j in range(ng) if j != i]), axis=0), gals[i])[1]
            for i in range(ng)])
        mr_u, cogs_u, th_u = logo_eval(gals, [], univ, label=f"{config}/univ")

        # (ii) conditioned
        pairs = corr_table(gals)
        mr_c, cogs_c, th_c = (mr_u, cogs_u, th_u)
        cond = np.array(univ)
        if pairs:
            cond = fit_shared(gals, list(range(ng)), pairs,
                              [np.concatenate([univ, np.zeros(len(pairs))])], 6000)
            mr_c, cogs_c, th_c = logo_eval(gals, pairs, cond, label=f"{config}/cond")

        # optional refinement: f(z) box (the unbounded fit rails z_c at the guard)
        bnd = fit_shared(gals, list(range(ng)), [], [phys, univ], 6000,
                         penalty=box_penalty)
        mr_b, cogs_b, th_b = logo_eval(gals, [], bnd, label=f"{config}/bound",
                                       penalty=box_penalty)

        # (iii) SHMR + end-to-end
        shmr_pred, shmr_sc = shmr_logo(gals)
        e2e_u = end_to_end(gals, cogs_u, shmr_pred)
        e2e_c = end_to_end(gals, cogs_c, shmr_pred)
        e2e_b = end_to_end(gals, cogs_b, shmr_pred)

        pre = config + "_"
        out.update({pre + "index": np.array([g["gi"] for g in gals]),
                    pre + "univ": univ, pre + "cond": cond, pre + "bound": bnd,
                    pre + "pairs": np.array([f"{THETA_NAMES[j]}<-{p}" for j, p in pairs]),
                    pre + "mr_med": mr_med, pre + "mr_univ": mr_u, pre + "mr_cond": mr_c,
                    pre + "mr_bound": mr_b,
                    pre + "cogs_univ": cogs_u, pre + "cogs_cond": cogs_c,
                    pre + "cogs_bound": cogs_b,
                    pre + "theta_univ": th_u, pre + "theta_cond": th_c,
                    pre + "theta_bound": th_b,
                    pre + "shmr_scatter": shmr_sc, pre + "shmr_pred": shmr_pred,
                    pre + "e2e_univ": e2e_u, pre + "e2e_cond": e2e_c,
                    pre + "e2e_bound": e2e_b})
    OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ, **out)
    print(f"\nwrote {OUT_NPZ}")
    return np.load(OUT_NPZ)


def med_epoch(mr):
    return np.array([100 * np.nanmedian(mr[:, k]) for k in range(5)])


def main():
    refit = "--refit" in sys.argv
    n = next((int(a) for a in sys.argv[1:] if a.isdigit()), 45)
    config_arg = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "both"
    configs = CONFIGS if config_arg == "both" else [config_arg]
    d = compute(n, configs) if (refit or not OUT_NPZ.exists()) else np.load(OUT_NPZ)

    dp = np.load(PARAM_NPZ)
    ngf = len(dp["index"])
    floor = np.array([pe.tf.maxrel(dp["cogs"][i], dp["data"][i]) for i in range(ngf)])
    avg = lambda m: float(np.mean(med_epoch(m)))

    print("\nexp30 phase 4 — population/forward step, LOGO median max|rel| per epoch "
          "(all radii)\n")
    print(f"    {'model':>28s} | " + " | ".join(f"z={z}".rjust(6) for z in ANCHOR_Z) + " |  avg")
    print(f"    {'per-galaxy fit (floor)':>28s} | "
          + " | ".join(f"{100*np.median(floor[:,k]):5.1f}%" for k in range(5))
          + f" | {100*np.mean([np.median(floor[:,k]) for k in range(5)]):4.1f}%")
    for config in CONFIGS:
        if config + "_mr_univ" not in d:
            continue
        gals_cfg = load_gals(config, ngf)
        assert [g["gi"] for g in gals_cfg] == list(d[config + "_index"])
        mr_ins = np.array([shape_maxrel(d[config + "_univ"], g)[1] for g in gals_cfg])
        rows = [("median-theta", d[config + "_mr_med"]),
                ("universal in-sample", mr_ins),
                ("universal", d[config + "_mr_univ"]),
                ("conditioned", d[config + "_mr_cond"]),
                ("e2e universal", d[config + "_e2e_univ"]),
                ("e2e conditioned", d[config + "_e2e_cond"])]
        if config + "_mr_bound" in d:
            rows[4:4] = [("universal bounded", d[config + "_mr_bound"])]
            rows.append(("e2e bounded", d[config + "_e2e_bound"]))
        for tag, m in rows:
            print(f"    {config + ' ' + tag:>28s} | "
                  + " | ".join(f"{v:5.1f}%" for v in med_epoch(m)) + f" | {avg(m):4.1f}%")
        print(f"    {'':>28s}   SHMR scatter [dex]: "
              + "  ".join(f"z={Z[k]}:{d[config+'_shmr_scatter'][k]:.3f}" for k in range(5))
              + f"   pairs: {list(d[config + '_pairs'])}")

    if "real_mr_univ" in d:
        fl = 100 * np.mean([np.median(floor[:, k]) for k in range(5)])
        au, ac = avg(d["real_mr_univ"]), avg(d["real_mr_cond"])
        ab = avg(d["real_mr_bound"]) if "real_mr_bound" in d else np.inf
        print(f"\n[verdict] floor {fl:.1f}% -> universal {au:.1f}% -> conditioned {ac:.1f}%"
              + (f" -> bounded {ab:.1f}%" if np.isfinite(ab) else "")
              + (f" | diffmah universal {avg(d['diffmah_mr_univ']):.1f}%"
                 if "diffmah_mr_univ" in d else ""))
        gain = "conditioning helps" if ac < au - 0.5 else "conditioning does NOT help"
        boxg = "; the f(z) box helps" if ab < min(au, ac) - 0.5 else \
               "; the f(z) box does NOT help" if np.isfinite(ab) else ""
        print(f"  {gain}{boxg}; gap to the per-galaxy floor {min(au, ac, ab) - fl:+.1f} "
              "points = the individuality the MAH+props do not carry.")
    _figure(d, floor)
    for config in CONFIGS:
        if config + "_cogs_cond" in d:
            ok = np.isfinite(d[config + "_cogs_cond"]).all(axis=(1, 2))
            gidx = {int(g): i for i, g in enumerate(dp["index"])}
            sel = [gidx[int(g)] for g in d[config + "_index"][ok]]
            mass_qa.evaluate(d[config + "_cogs_cond"][ok], dp["data"][sel], pe.R,
                             ANCHOR_Z, name=f"pop-cond-{config}", figdir=FIGDIR)


def _figure(d, floor):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    a, b, c = axes
    x = np.arange(5)
    a.plot(x, [100 * np.median(floor[:, k]) for k in range(5)], "o-", c="0.35", lw=2,
           label="per-galaxy fit (floor)")
    styles = {"real": ("-", 2.0), "diffmah": ("--", 1.5)}
    for config in CONFIGS:
        if config + "_mr_univ" not in d:
            continue
        ls, lw = styles[config]
        a.plot(x, med_epoch(d[config + "_mr_med"]), ls, marker="s", c="#E69F00", lw=lw,
               label=f"{config} median-theta")
        a.plot(x, med_epoch(d[config + "_mr_univ"]), ls, marker="o", c="#CC3377", lw=lw,
               label=f"{config} universal")
        a.plot(x, med_epoch(d[config + "_mr_cond"]), ls, marker="^", c="#009E73", lw=lw,
               label=f"{config} conditioned")
        if config + "_mr_bound" in d:
            a.plot(x, med_epoch(d[config + "_mr_bound"]), ls, marker="d", c="#0072B2",
                   lw=lw, label=f"{config} bounded f(z)")
    a.set_xticks(x); a.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    ymax = 1.3 * max(med_epoch(d[c + "_mr_univ"]).max() for c in CONFIGS
                     if c + "_mr_univ" in d)          # median-theta may run off-scale
    a.set(xlabel="epoch z", ylabel="LOGO median max|rel| [%]", ylim=(0, max(ymax, 40)),
          title="A. New-halo shape error vs the per-galaxy floor")
    a.legend(fontsize=7)

    pairs = list(d.get("real_pairs", []))
    if pairs:
        j, prop = next((jj, pp) for jj, pp in
                       [(THETA_NAMES.index(s.split("<-")[0]), s.split("<-")[1])
                        for s in pairs][:1])
        dp = np.load(PARAM_NPZ)
        gals = load_gals("real", len(dp["index"]))
        v = np.array([g[prop] for g in gals])
        P = np.array([g["theta_fit"] for g in gals])
        b.scatter(v, P[:, j], s=30, c="#CC3377", edgecolor="0.3", lw=0.4)
        b.set(xlabel=prop, ylabel=THETA_NAMES[j],
              title=f"B. Strongest conditioning pair: {THETA_NAMES[j]} <- {prop}")
    else:
        b.text(0.5, 0.5, "no pair passed p < %.2f" % P_SELECT, ha="center", va="center",
               transform=b.transAxes)
        b.set(title="B. Conditioning selection")

    for config in CONFIGS:
        if config + "_e2e_cond" not in d:
            continue
        ls, lw = styles[config]
        c.plot(x, med_epoch(d[config + "_mr_cond"]), ls, marker="^", c="#009E73", lw=lw,
               label=f"{config} shape (M* given)")
        c.plot(x, med_epoch(d[config + "_e2e_cond"]), ls, marker="o", c="#D55E00", lw=lw,
               label=f"{config} end-to-end (SHMR M*)")
    c.set_xticks(x); c.set_xticklabels([f"{z}" for z in ANCHOR_Z])
    c.set(xlabel="epoch z", ylabel="LOGO median max|rel| [%]", ylim=(0, None),
          title="C. End-to-end: SHMR amplitude added")
    c.legend(fontsize=7)
    fig.suptitle("exp30 phase 4 — population/forward step: halo-only -> theta -> profile "
                 "(LOGO, all radii)", fontsize=12)
    fig.tight_layout()
    print("\nwrote", save_fig(fig, FIGDIR / "exp30_pop_forward")[0])


def demo():
    """Self-check: own-theta reproduces the stored in-sample cogs; median-theta CoGs
    structurally valid; theta_cond identity; SHMR LOGO exact on a noiseless linear
    relation. (Median-theta QUALITY is not asserted: the per-galaxy 7-param fits are
    degenerate, so raw-theta medians are legitimately poor — the universal set must
    be refit jointly.)"""
    gals = load_gals("real", 6)
    assert len(gals) >= 4
    dp = np.load(PARAM_NPZ)
    own, _ = shape_maxrel(gals[0]["theta_fit"], gals[0])
    assert np.allclose(own, dp["cogs"][0], rtol=1e-8), \
        "own theta must reproduce the stored in-sample cogs"
    med = np.median(np.array([g["theta_fit"] for g in gals]), axis=0)
    cogs, mr = shape_maxrel(med, gals[0])
    assert cogs is not None and np.isfinite(cogs).all()
    assert np.all(np.diff(cogs, axis=1) >= -1e-9), "CoG must be monotonic"
    for k in range(5):
        assert abs(cogs[k][-1] / gals[0]["data"][k][-1] - 1) < 1e-9, "must be pinned"
    p = np.concatenate([med, [0.0]])
    assert np.allclose(theta_cond(p, gals[0], [(3, "c200c")]), med), \
        "zero slope must reduce to the base theta"
    for g in gals:                                        # noiseless linear SHMR
        g["logms_zk"] = 0.5 * g["logmh_zk"] + 1.0
    _, sc = shmr_logo(gals)
    assert sc.max() < 1e-8, ("LOGO SHMR must be exact on a linear relation", sc)
    v = pop_loss(lambda g: med, gals, range(len(gals)))
    assert 0 < v < 1.0, ("pop_loss out of range", v)
    print(f"pop_forward.demo OK: own-theta reproduces stored cogs; pinned monotonic "
          f"median-theta CoGs (max|rel| {100*mr.max():.0f}%, degeneracy expected), "
          "cond identity, exact LOGO SHMR")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo()
    else:
        sys.exit(main())
