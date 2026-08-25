"""exp57 Stage 7 — the scoreboard, and one attribution test.

**THE SCOREBOARD** collects every fit and every gate into one table, ordered by
the thing that decides — the gates — and NOT by the loss. That ordering is the
point: `X3`'s search found two basins and the **loss prefers the one the gates
reject**, which is the failure mode this program has recorded repeatedly
(exp38's loss-optimal basin paid for its loss in the physics tests; exp49 found
the loss and the compact central deficit moving in opposite directions; exp52
found a model that reproduced the summed profile while getting the mechanism
wrong).

**THE ATTRIBUTION TEST.** Gate G5(c) — the median core surface-density growth
against the measurement — behaves in opposite directions for long-reach and
short-reach expansion, and a fitted model is not a controlled experiment: its
base parameters have moved too. So for each fit this switches the expansion OFF
at the fitted base parameters and re-measures. That separates

  * what the EXPANSION did, from
  * what REFITTING the other seven parameters around it did,

which is the difference between a mechanism and a reparameterisation. exp53
found a mechanism that was real in a model's own bookkeeping and was not the
cause of that model's error, with a frozen slice and an honest refit 57 points
apart; this is the cheap version of that check.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp57_expansion_term/stage7_summary.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP54 = ROOT / "experiments/exp54_unpinned_amplitude"
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink", EXP54, HERE):
    sys.path.insert(0, str(p))

import expand as X                                       # noqa: E402
import gates as G                                        # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage35_time_law as S35                           # noqa: E402
import stage4_target as S4                               # noqa: E402

OUT = HERE / "outputs"
FITDIR = OUT / "stage3_fits"
RULE = "=" * 112
_P = S35._PC


def collect():
    rows = []
    for f in sorted(FITDIR.glob("*.npz")):
        if ".PARTIAL" in f.name:
            continue
        z = np.load(f, allow_pickle=True)
        tag = str(z["label"])
        g = OUT / f"stage2_gates_{tag}.npz"
        t = OUT / f"stage4_target_{tag}.npz"
        rows.append(dict(tag=tag, law=str(z["law"]),
                         loss=float(z["loss"]),
                         theta=np.asarray(z["theta"], float),
                         names=[str(v) for v in z["names"]],
                         shape=float(z["shape_pct"]),
                         gates=np.load(g, allow_pickle=True) if g.exists() else None,
                         target=np.load(t, allow_pickle=True) if t.exists() else None))
    return rows


def main():
    print(f"{RULE}\nexp57 STAGE 7 — the scoreboard, ordered by the GATES and "
          f"not by the loss\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(False)
    rows = collect()
    gn = OUT / "stage2_gates_null.npz"
    tn = OUT / "stage4_target_null.npz"
    null_g = np.load(gn, allow_pickle=True) if gn.exists() else None
    null_t = np.load(tn, allow_pickle=True) if tn.exists() else None

    print(f"  {'model':<34}{'loss':>10}{'shape':>9}{'G2 >50':>9}{'G2 >148':>9}"
          f"{'G5a dec':>9}{'G5b not':>9}{'G5c gap':>9}   verdict")
    print(f"  {'incumbent (expansion OFF)':<34}{1.874253:>10.6f}"
          f"{13.787:>8.3f}{_P}{'—':>9}{'—':>9}"
          f"{float(null_t['span_dec']) if null_t is not None else np.nan:>9.3f}"
          f"{float(null_t['span_not']) if null_t is not None else np.nan:>9.3f}"
          f"{float(null_t['core_gap']) if null_t is not None else np.nan:>+9.4f}"
          f"   the reference")
    for r in sorted(rows, key=lambda r: -(r["gates"] is not None
                                          and bool(r["gates"]["passed"]))):
        g, t = r["gates"], r["target"]
        b50 = float(np.nanmax(g["g2_b50"])) if g is not None else np.nan
        b148 = float(np.nanmax(g["g2_b148"])) if g is not None else np.nan
        v = []
        if g is not None:
            v.append("G2 " + ("PASS" if bool(g["passed"]) else "FAIL"))
        if t is not None:
            v.append("G5 " + "".join(
                k if bool(t[f"{k}_ok"]) else k.upper()
                for k in ("a", "b", "c")))
        print(f"  {r['tag'].replace('gompertz_log-E2-S2', ''):<34}"
              f"{r['loss']:>10.6f}{r['shape']:>8.3f}{_P}"
              f"{100 * b50:>8.1f}{_P}{100 * b148:>8.1f}{_P}"
              f"{float(t['span_dec']) if t is not None else np.nan:>9.3f}"
              f"{float(t['span_not']) if t is not None else np.nan:>9.3f}"
              f"{float(t['core_gap']) if t is not None else np.nan:>+9.4f}"
              f"   {'; '.join(v)}")
    print(f"\n  G5 letters: lower case = that part passes, UPPER CASE = it "
          f"fails.")

    # ---------------------------------------------------------------- #
    print(f"\n{RULE}\n  THE ATTRIBUTION TEST — what did the EXPANSION do, and "
          f"what did REFITTING do?\n{RULE}")
    print(f"  For each fit, the expansion is switched off AT THE FITTED BASE "
          f"PARAMETERS. The gap\n  between that and the full fit is the "
          f"expansion; the gap between it and the incumbent\n  is what "
          f"re-optimising the other seven bought on its own.\n")
    print(f"  {'model':<34}{'core gap':>11}{'expansion':>11}{'refit':>11}"
          f"{'span dec':>11}{'expansion':>11}{'refit':>11}")
    print(f"  {'':<34}{'(full fit)':>11}{'alone':>11}{'alone':>11}"
          f"{'(full fit)':>11}{'alone':>11}{'alone':>11}")
    use = mask[:, 4] & mask[:, 0]
    d_dat, _, _ = S4.core_dlogsigma(data, use)
    fell, ok_flag = S4.decline_flag(data)
    saved = []
    for r in rows:
        if r["target"] is None:
            continue
        sp = X.XSpec(base, r["law"])
        pr = X.ExpandingProblem(sp, recs, data, mask, epochs=(0, 1, 2, 3, 4))
        th_off = r["theta"].copy()
        for i, n in enumerate(r["names"]):
            if n in ("A", "alpha"):
                th_off[i] = 0.0
        p_off = pr.predict(th_off)
        c_off, _, _ = S4.core_dlogsigma(p_off, use)
        _, s_off = S4.central_span(p_off, data, mask, fell & ok_flag)
        c_full = float(r["target"]["core_gap"])
        s_full = float(r["target"]["span_dec"])
        c_null, s_null = float(null_t["core_gap"]), float(null_t["span_dec"])
        print(f"  {r['tag'].replace('gompertz_log-E2-S2', ''):<34}"
              f"{c_full:>+11.4f}{c_full - (c_off - d_dat):>+11.4f}"
              f"{(c_off - d_dat) - c_null:>+11.4f}"
              f"{s_full:>11.3f}{s_full - s_off:>+11.3f}{s_off - s_null:>+11.3f}")
        saved.append(dict(tag=r["tag"], core_full=c_full,
                          core_off=c_off - d_dat, span_full=s_full,
                          span_off=s_off))
    print(f"\n  A model whose 'expansion alone' column is near zero is a "
          f"REPARAMETERISATION:\n  everything it bought came from moving the "
          f"other seven parameters, and the expansion\n  term is carrying the "
          f"credit for it.")

    np.savez(OUT / "stage7_summary.npz",
             tags=np.array([r["tag"] for r in rows]),
             loss=np.array([r["loss"] for r in rows]),
             shape=np.array([r["shape"] for r in rows]),
             **{f"attr_{k}": np.array([s[k] for s in saved])
                for k in ("core_full", "core_off", "span_full", "span_off")})
    print(f"\n  wrote stage7_summary.npz")


if __name__ == "__main__":
    main()
