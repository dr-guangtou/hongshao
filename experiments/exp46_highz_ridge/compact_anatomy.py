"""exp46 stage 1e — are the very compact z=2 progenitors special in their HALOES?

Stage 1d found the model's plane failure lives on compact galaxies at
every epoch, and that high-z epochs are simply made almost entirely of
compact galaxies (the R50 < 5 kpc fraction runs 6 / 14 / 27 / 61 / 85%
from z=0.4 to z=2). The kernel is a function of halo properties ONLY, so
whether that failure is addressable at all turns on one question:

    do compact high-z progenitors have distinguishable haloes?

If yes, a halo-conditioned term can single them out. If no, three
possibilities remain, and this script measures evidence for each:
  (i)  they are genuine outliers of the profile distribution;
  (ii) their MAHs are unreliable (the user's hypothesis: e.g. the halo
       mass is overestimated) — tested via the DiffMAH fit residual and
       the real-vs-DiffMAH halo-mass disagreement at z=2;
  (iii) compactness is real but simply not a halo-driven quantity, in
       which case only a stochastic layer (or a state-dependent
       transport term) can produce it.

Two compactness measures, deliberately:
  R50      the half-mass radius — the quantity stage 1d cut on, but
           BELOW the 2.0 kpc CoG grid start for 13% of z=2 galaxies, so
           its value there comes from log-log extrapolation;
  c_in     log10 M*(<10 kpc) / M*(<148 kpc) — a concentration index that
           is entirely on-grid and needs no extrapolation. Anything that
           shows up in R50 but not in c_in is suspect.

No fitting. Run: PYTHONPATH=. uv run python \
experiments/exp46_highz_ridge/compact_anatomy.py {demo|run}
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

OUTDIR = HERE / "outputs"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
DMAH_CSV = (ROOT / "experiments/exp10_diffmah_fit/outputs/"
            "diffmah_params_tmin1gyr.csv")
Z_GRID = (0.4, 0.7, 1.0, 1.5, 2.0)
K_HIGHZ = 4                      # the z=2.0 column
COMPACT_Q = 0.10                 # "very compact" = smallest decile at z=2


def spearman(a, b):
    """Rank correlation, NaN-safe (no scipy dependency in this repo)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def cliffs_delta(x, y):
    """P(x > y) - P(x < y) via ranks: +1 = x wholly above y, 0 = identical.

    Reported instead of a t-statistic because these features are skewed and
    the two groups differ ~9x in size; it is a pure ordering statistic and
    needs no distributional assumption.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    both = np.concatenate([x, y])
    r = np.argsort(np.argsort(both)).astype(float) + 1.0
    rx = r[:len(x)].sum()
    u = rx - len(x) * (len(x) + 1) / 2.0
    return float(2.0 * u / (len(x) * len(y)) - 1.0)


def load():
    p = np.load(POP_NPZ)
    d = p["data"]
    feats = {
        "logmh (z=0.4)": p["logmh"],
        "c200c": p["c200c"],
        "fz2 = Mh(z2)/Mh": p["fz2"],
        "t50 (halo)": p["t50"],
        "burst": p["burst"],
        "logms (z=0.4)": p["logms"],
        "logmh at z=2 (real)": p["logmh_zk_real"][:, K_HIGHZ],
        "logmh at z=2 (diffmah)": p["logmh_zk_diffmah"][:, K_HIGHZ],
    }
    # the user's hypothesis (ii): is the halo history itself unreliable here?
    feats["|real - diffmah| Mh(z=2)"] = np.abs(
        p["logmh_zk_real"][:, K_HIGHZ] - p["logmh_zk_diffmah"][:, K_HIGHZ])
    feats["Mh growth z2->z0.4 [dex]"] = (
        p["logmh"] - p["logmh_zk_real"][:, K_HIGHZ])

    if DMAH_CSV.exists():
        raw = np.genfromtxt(DMAH_CSV, delimiter=",", names=True)
        lut = {int(i): k for k, i in enumerate(raw["index"])}
        for name, col in (("dmah_logtc", "dmah_logtc"),
                          ("dmah_early", "dmah_early"),
                          ("dmah_late", "dmah_late"),
                          ("dmah_rms (fit quality)", "dmah_rms")):
            v = np.full(len(d), np.nan)
            for k, idx in enumerate(p["index"]):
                j = lut.get(int(idx))
                if j is not None:
                    v[k] = raw[col][j]
            feats[name] = v
    return p, d, feats


def compactness(d, R):
    """(R50, c_in) at z=2. c_in needs no below-grid extrapolation."""
    from hongshao import qa
    r50 = np.array([qa.enclosed_radius(d[i, K_HIGHZ], R, 0.5)
                    for i in range(len(d))])
    m10 = np.array([np.interp(10.0, R, d[i, K_HIGHZ]) for i in range(len(d))])
    c_in = np.log10(np.clip(m10, 1.0, None) /
                    np.clip(d[:, K_HIGHZ, -1], 1.0, None))
    return r50, c_in


def cmd_run():
    from hongshao import qa
    from hongshao.tng_data import COG_RAD_KPC as R
    p, d, feats = load()
    n = len(d)
    r50, c_in = compactness(d, R)

    print(f"exp46 stage 1e — are compact z=2 progenitors special in their "
          f"HALOES? (n={n})\n")

    # --- is the compact tail smooth, or a distinct clump? -----------------
    qs = np.quantile(r50, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    print("  R50 at z=2 [kpc] by percentile "
          "(1/5/10/25/50/75/90):  " + " ".join(f"{v:.2f}" for v in qs))
    edges = np.array([0, 1, 1.5, 2, 3, 4, 5, 7, 10, 1e3])
    hist, _ = np.histogram(r50, bins=edges)
    print("  counts in [0,1,1.5,2,3,4,5,7,10,inf) kpc:      "
          + " ".join(f"{h:5d}" for h in hist))
    print(f"  Spearman(R50, c_in) = {spearman(r50, c_in):+.2f}   "
          "(the two compactness measures agree if strongly negative)\n")

    cut = np.quantile(r50, COMPACT_Q)
    comp = r50 <= cut
    print(f"  'very compact' = smallest {COMPACT_Q:.0%} of R50 at z=2 "
          f"(R50 <= {cut:.2f} kpc, n={comp.sum()}); rest n={(~comp).sum()}\n")

    # --- the halo comparison ----------------------------------------------
    print("  Halo/MAH properties: compact vs rest. delta = Cliff's delta")
    print("  (+1 compact wholly above, 0 no separation); |rho| vs the")
    print("  CONTINUOUS measures is the same signal without a threshold.\n")
    print(f"    {'feature':<26s}{'compact':>10s}{'rest':>10s}"
          f"{'delta':>8s}{'rho(R50)':>10s}{'rho(c_in)':>11s}")
    rows = {}
    for name, v in feats.items():
        mc, mr = np.nanmedian(v[comp]), np.nanmedian(v[~comp])
        dl = cliffs_delta(v[comp], v[~comp])
        rows[name] = (mc, mr, dl, spearman(r50, v), spearman(c_in, v))
        print(f"    {name:<26s}{mc:10.3f}{mr:10.3f}{dl:+8.2f}"
              f"{rows[name][3]:+10.2f}{rows[name][4]:+11.2f}")

    # --- what are these galaxies at z=0.4? ---------------------------------
    r50_z0 = np.array([qa.enclosed_radius(d[i, 0], R, 0.5) for i in range(n)])
    tot = np.log10(np.clip(d[:, :, -1], 1.0, None))
    print(f"\n  What are they at z=0.4?  R50 median "
          f"{np.median(r50_z0[comp]):.2f} vs {np.median(r50_z0[~comp]):.2f} "
          f"kpc (delta {cliffs_delta(r50_z0[comp], r50_z0[~comp]):+.2f})")
    print(f"    log M*(<148) at z=0.4: {np.median(tot[comp, 0]):.2f} vs "
          f"{np.median(tot[~comp, 0]):.2f} "
          f"(delta {cliffs_delta(tot[comp, 0], tot[~comp, 0]):+.2f})")
    print(f"    log M*(<148) at z=2.0: {np.median(tot[comp, 4]):.2f} vs "
          f"{np.median(tot[~comp, 4]):.2f} "
          f"(delta {cliffs_delta(tot[comp, 4], tot[~comp, 4]):+.2f})")
    print(f"    stellar growth z2->z0.4 [dex]: "
          f"{np.median(tot[comp, 0] - tot[comp, 4]):.2f} vs "
          f"{np.median(tot[~comp, 0] - tot[~comp, 4]):.2f}")

    # --- the ceiling: how much of compactness is halo-predictable? ---------
    # The number that decides whether a halo-conditioned term can work: how
    # much compactness survives halo features AT FIXED PROGENITOR MASS. The
    # kernel is normalized to the measured total, so the mass trend is
    # already free to it; only the residual is a real modelling target.
    keys = ["logmh (z=0.4)", "c200c", "fz2 = Mh(z2)/Mh", "t50 (halo)",
            "logmh at z=2 (real)", "dmah_early", "dmah_late", "dmah_logtc"]
    keys = [k for k in keys if k in feats]
    X = np.column_stack([feats[k] for k in keys])
    lms2 = np.log10(np.clip(d[:, K_HIGHZ, -1], 1.0, None))
    ok = np.isfinite(X).all(1) & np.isfinite(c_in) & np.isfinite(r50)

    def _fit(y, cols):
        A = np.column_stack([np.ones(len(y))]
                            + [(c - c.mean()) / (c.std() + 1e-12)
                               for c in cols])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return beta, 1.0 - np.var(y - A @ beta) / np.var(y)

    cols = [X[ok, i] for i in range(X.shape[1])]
    # sign convention: "more compact" = SMALLER R50 but LARGER c_in, so the
    # compact decile is the low tail of one and the high tail of the other.
    for label, y0, compact_is_low in (("log R50", np.log10(r50), True),
                                      ("c_in", c_in, False)):
        y = y0[ok]
        _, r2_raw = _fit(y, cols)
        bm, r2_mass = _fit(y, [lms2[ok]])
        yres = y - np.column_stack([np.ones(len(y)),
                                    (lms2[ok] - lms2[ok].mean())
                                    / (lms2[ok].std() + 1e-12)]) @ bm
        _, r2_fixed = _fit(yres, cols)
        print(f"\n  How halo-predictable is {label} at z=2?")
        print(f"    R^2 from all halo features, raw          {r2_raw:.3f}")
        print(f"    R^2 from the progenitor's own mass alone {r2_mass:.3f}")
        print(f"    R^2 from halo features AT FIXED MASS     {r2_fixed:.3f}"
              "   <- the modelling target")
        q = np.quantile(yres, COMPACT_Q if compact_is_low else 1 - COMPACT_Q)
        cmp_res = yres <= q if compact_is_low else yres >= q
        print("    at fixed mass, compact-decile delta: " + "  ".join(
            f"{k.split(' ')[0]}={cliffs_delta(feats[k][ok][cmp_res], feats[k][ok][~cmp_res]):+.2f}"
            for k in keys))

    OUTDIR.mkdir(exist_ok=True)
    np.savez(OUTDIR / "compact_anatomy.npz", r50_z2=r50, c_in_z2=c_in,
             compact=comp, feature_names=np.array(list(rows)),
             stats=np.array([rows[k] for k in rows]))
    print(f"\n  wrote {OUTDIR / 'compact_anatomy.npz'}")


def demo():
    rng = np.random.default_rng(0)
    # spearman: perfect monotone -> +1, reversed -> -1
    x = rng.normal(size=200)
    assert abs(spearman(x, np.exp(x)) - 1.0) < 1e-9
    assert abs(spearman(x, -np.exp(x)) + 1.0) < 1e-9
    # cliffs_delta: disjoint groups -> +-1, identical -> ~0
    assert abs(cliffs_delta(np.arange(50) + 100.0, np.arange(50.0)) - 1) < 1e-9
    assert abs(cliffs_delta(np.arange(50.0), np.arange(50) + 100.0) + 1) < 1e-9
    assert abs(cliffs_delta(rng.normal(size=400), rng.normal(size=400))) < 0.15
    # a KNOWN planted signal is recovered end to end at the right strength
    feat = rng.normal(size=1000)
    size = 0.8 * feat + 0.6 * rng.normal(size=1000)
    comp = size <= np.quantile(size, 0.10)
    d_known = cliffs_delta(feat[comp], feat[~comp])
    assert d_known < -0.5, d_known
    # ... and a feature with NO relation shows none
    noise = rng.normal(size=1000)
    assert abs(cliffs_delta(noise[comp], noise[~comp])) < 0.25
    print(f"demo OK — planted signal recovered (delta {d_known:+.2f}), "
          "null feature flat")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        demo()
    elif cmd == "run":
        cmd_run()
    else:
        raise SystemExit(__doc__)
