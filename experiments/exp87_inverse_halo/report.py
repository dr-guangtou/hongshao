"""exp87 — reading the scoreboard: the ladder tables, the paired comparisons
and the controller's decisions.

For one GROUP (sample, epoch, population, target) every method is scored on
the same development galaxies, so two methods are compared by a paired
bootstrap of their per-galaxy CRPS. "Significant" is the plan's rule: the 95%
interval of the difference excludes zero AND the relative gain reaches delta
(the mechanics gate's value, 1%).

    report.py tables                 every group's ladder, sorted by CRPS
    report.py decide                 the Stage 4 decision (nonlinearity N) and
                                     the comparisons the README quotes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import harness as HN                                     # noqa: E402
import scoring as S                                      # noqa: E402

RULE = "=" * 100
GROUP_KEYS = ("sample", "epoch", "population", "target")


def delta():
    return float(json.loads((C.OUTDIR / "gate_mechanics.json").read_text())["delta"])


def groups(rows=None):
    rows = rows if rows is not None else HN.read_scoreboard()
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in GROUP_KEYS), []).append(r)
    return out


def per_galaxy(row, key="crps"):
    z = HN.load_cell(row["cell"])
    return z[key][z["used"]], z["index"][z["used"]]


def compare(row_a, row_b, d=None):
    """Paired bootstrap of CRPS(a) - CRPS(b) on the galaxies both cells used."""
    a, ia = per_galaxy(row_a)
    b, ib = per_galaxy(row_b)
    if len(ia) != len(ib) or not np.array_equal(ia, ib):
        common, pa, pb = np.intersect1d(ia, ib, return_indices=True)
        a, b = a[pa], b[pb]
    ok, boot = S.significant_gain(a, b, delta() if d is None else d)
    return ok, boot


def best(rows, rungs=None, extra="none", features=None, exclude=()):
    cand = [r for r in rows if (rungs is None or r["rung"] in rungs) and r["extra"] == extra
            and (features is None or r["feature"] in features) and r["method"] not in exclude]
    return min(cand, key=lambda r: r["crps"]) if cand else None


def find(rows, feature, method, extra="none"):
    hit = [r for r in rows if r["feature"] == feature and r["method"] == method and r["extra"] == extra]
    return hit[0] if hit else None


def label(r):
    return f"{r['feature']}{'' if r['extra'] == 'none' else '+true ' + r['extra']} / {r['method']}"


def print_table(key, rows):
    clim = find(rows, "mtot", "climatology")
    unit = "dex" if key[3] in ("mh", "logc") else "Gyr"
    print(f"\n--- {key[0]} z = {C.ANCHOR_Z[key[1]]}, population '{key[2]}', target {key[3]} [{unit}]: n = {rows[0]['n']}, "
          f"cut {rows[0]['cut']} ---")
    print(f"  {'rung':<5}{'inputs / method':<44}{'CRPS':>8}{'skill':>8}{'RMSE':>8}{'MAE':>8}{'nats':>8}{'cata%':>7}{'cov68':>7}{'cov90':>7}{'slope':>7}{'width':>7}")
    for r in sorted(rows, key=lambda r: r["crps"]):
        skill = 1.0 - r["crps"] / clim["crps"] if clim else np.nan
        nats = (clim["logscore"] - r["logscore"]) if clim else np.nan
        print(f"  {r['rung']:<5}{label(r):<44}{r['crps']:>8.4f}{skill:>8.3f}{r['rmse']:>8.4f}{r['mae_median']:>8.4f}{nats:>8.3f}"
              f"{100 * r['catastrophic']:>7.2f}{r['cover68']:>7.2f}{r['cover90']:>7.2f}{r['reliability_slope']:>7.2f}{r['width_ratio']:>7.2f}")


def tables():
    for key, rows in sorted(groups().items(), key=lambda kv: (kv[0][3], kv[0][0], kv[0][1], kv[0][2])):
        print_table(key, rows)


def describe(name, a, b, d):
    ok, boot = compare(a, b, d)
    print(f"   {name}: {label(a)} ({a['crps']:.4f}) vs {label(b)} ({b['crps']:.4f}): gain {100 * -boot['rel']:+.2f}% "
          f"[{100 * -boot['rel_hi']:+.2f}, {100 * -boot['rel_lo']:+.2f}]  {'SIGNIFICANT' if ok else 'not significant'}")
    return dict(name=name, a=a["cell"], b=b["cell"], gain=-boot["rel"], lo=-boot["rel_hi"], hi=-boot["rel_lo"], significant=ok)


def decide():
    d = delta()
    g = groups()
    out = {"delta": d, "comparisons": []}
    print(f"{RULE}\nexp87 — the ladder's paired comparisons (delta = {100 * d:.2f}%)\n{RULE}")
    for key in sorted(g, key=lambda k: (k[3], k[0], k[1], k[2])):
        rows = g[key]
        clim, scalar = find(rows, "mtot", "climatology"), find(rows, "mtot", "line")
        if clim is None or scalar is None:
            continue
        print(f"\n {key[0]} z = {C.ANCHOR_Z[key[1]]} '{key[2]}' target {key[3]} (n {rows[0]['n']}):")
        ap = best(rows, rungs=("L0",), exclude=("climatology",))
        l1 = best(rows, rungs=("L1",))
        l23 = best(rows, rungs=("L2", "L3"))
        l5 = best(rows, rungs=("L5",))
        recs = [describe("stellar mass over nothing", scalar, clim, d)]
        if ap is not None and ap["cell"] != scalar["cell"]:
            recs.append(describe("best single aperture over M*(<148)", ap, scalar, d))
        if l1 is not None:
            recs.append(describe("the PROFILE over M*(<148) (best linear)", l1, scalar, d))
            ms = find(rows, "mass_size", "linear")
            if ms is not None:
                recs.append(describe("mass + sizes over M*(<148)", ms, scalar, d))
                if l1["cell"] != ms["cell"]:
                    recs.append(describe("the full profile over mass + sizes", l1, ms, d))
        if l23 is not None and l1 is not None:
            recs.append(describe("NONLINEARITY N (best L2/L3 over best L1)", l23, l1, d))
        if l5 is not None and l1 is not None:
            recs.append(describe("generative inverse over best L1", l5, l1, d))
        oracle_rows = [r for r in rows if r["extra"] != "none"]
        if oracle_rows:
            base = find(rows, "none", "poly-mh", extra="mh")
            cog_only = best(rows, rungs=("L1", "L2", "L3"))
            with_cog = min([r for r in oracle_rows if r["feature"] != "none"], key=lambda r: r["crps"])
            if base is not None:
                recs.append(describe("ORACLE: true halo mass over nothing", base, clim, d))
                recs.append(describe("ORACLE: profile + true halo mass over true halo mass", with_cog, base, d))
                if cog_only is not None:
                    recs.append(describe("ORACLE: profile + true halo mass over profile", with_cog, cog_only, d))
        for r in recs:
            r["group"] = list(key)
        out["comparisons"] += recs
    key = ("parent", 0, "parent", "mh")
    n_rec = [r for r in out["comparisons"] if r["group"] == list(key) and r["name"].startswith("NONLINEARITY")]
    out["nonlinearity_significant"] = bool(n_rec and n_rec[0]["significant"])
    out["stage4_mode"] = "S1,S2(,S3)" if out["nonlinearity_significant"] else "S1 only, residual mode"
    print(f"\n STAGE 4 DECISION: nonlinearity at z = 0.4 on the parent is "
          f"{'SIGNIFICANT -> escalate S1, S2, and S3 if S2 gains' if out['nonlinearity_significant'] else 'not significant -> S1 only, residual mode; report no detectable nonlinearity'}")
    (C.OUTDIR / "stage3_decisions.json").write_text(json.dumps(out, indent=1))
    with open(C.OUTDIR / "decisions.jsonl", "a") as fh:
        fh.write(json.dumps(dict(stage=3, nonlinearity_significant=out["nonlinearity_significant"], mode=out["stage4_mode"], delta=d)) + "\n")
    return out


if __name__ == "__main__":
    {"tables": tables, "decide": decide}[sys.argv[1] if len(sys.argv) > 1 else "tables"]()
