"""exp87 Stage 4 — the symbolic-regression controller: which searches run, in
which order, decided by the plan's rules from the scoreboard.

For one CELL (a sample, epoch, population and target):

  1. S1 (+ - * square), RESIDUAL mode, on the three input sets (summaries,
     pca4, raw24).
  2. If the ladder's nonlinearity N for this cell (best L2/L3 over best L1 in
     latent space, Stage 3) was NOT significant: stop. "No detectable
     nonlinearity"; the S1 formulas are reported as they are.
  3. Otherwise S2 (adds / sqrt log exp): the residual mode on S1's best input
     set, and the direct mode on the readable summaries.
  4. S3 (adds pow tanh max min) on the same two configurations only if S2's
     best beats S1's best significantly (paired interval excluding zero and a
     gain of at least delta).

The plan enters at halo mass, z = 0.4, on the parent (`mh-parent`). The other
cells are run with the SAME rule because Stage 3 found N significant there;
they are labelled BEYOND THE ENTRY RULE in the report (exploratory).

Run (detached; one job per group of cells):
    PYTHONPATH=. uv run python -u experiments/exp87_inverse_halo/stage4_driver.py mh-parent t50-parent [--smoke]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import harness as HN                                     # noqa: E402
import scoring as S                                      # noqa: E402
import sr as SR                                          # noqa: E402

RULE = "=" * 100
INPUTS = ("summaries", "pca4", "raw24")
CELLS = {
    "mh-parent": dict(sample="parent", epoch=0, population="parent", target="mh"),
    "t50-parent": dict(sample="parent", epoch=0, population="parent", target="t50"),
    "logc-parent": dict(sample="parent", epoch=0, population="parent", target="logc"),
    "mh-z2-complete": dict(sample="curated", epoch=4, population="complete", target="mh"),
    "mh-z1p5-complete": dict(sample="curated", epoch=3, population="complete", target="mh"),
    "mh-z2-asis": dict(sample="curated", epoch=4, population="asis", target="mh"),
}


def nonlinearity(spec):
    out = json.loads((C.OUTDIR / "stage3_decisions.json").read_text())
    group = [spec["sample"], spec["epoch"], spec["population"], spec["target"]]
    rec = [r for r in out["comparisons"] if r["group"] == group and r["name"].startswith("NONLINEARITY")]
    return (bool(rec[0]["significant"]), rec[0]["gain"]) if rec else (False, float("nan"))


def better(cell_a, cell_b, delta):
    a, b = HN.load_cell(cell_a), HN.load_cell(cell_b)
    return S.significant_gain(a["crps"][a["used"]], b["crps"][b["used"]], delta)


def run_cell(name, smoke, t_start):
    spec = CELLS[name]
    delta = float(json.loads((C.OUTDIR / "gate_mechanics.json").read_text())["delta"])
    sig, gain = nonlinearity(spec)
    log = dict(cell=name, spec=spec, nonlinearity_significant=sig, nonlinearity_gain=gain, stages=[])
    print(f"\n{RULE}\nCELL {name}: Stage 3's nonlinearity N = {100 * gain:+.2f}% ({'significant' if sig else 'not significant'})"
          f"{'' if name == 'mh-parent' else '  [beyond the entry rule: exploratory]'}\n{RULE}", flush=True)
    s1 = [SR.run("S1", "residual", inp, spec, smoke=smoke) for inp in INPUTS]
    log["stages"].append(dict(stage="S1", verdicts=s1))
    best1 = min(s1, key=lambda v: v["crps"])
    if not sig:
        log["stopped"] = "N not significant: S1 only, residual mode"
    elif time.time() - t_start > C.BUDGET["stage4"]:
        log["stopped"] = "budget"
    else:
        s2 = [SR.run("S2", "residual", best1["inputs"], spec, smoke=smoke), SR.run("S2", "direct", "summaries", spec, smoke=smoke)]
        log["stages"].append(dict(stage="S2", verdicts=s2))
        best2 = min(s2, key=lambda v: v["crps"])
        ok, boot = better(best2["cell"], best1["cell"], delta)
        log["s2_over_s1"] = dict(gain=-boot["rel"], significant=ok)
        print(f"  S2's best ({best2['mode']} on {best2['inputs']}, {best2['crps']:.4f}) over S1's best ({best1['inputs']}, {best1['crps']:.4f}): "
              f"{100 * -boot['rel']:+.2f}% {'SIGNIFICANT -> S3' if ok else 'not significant -> stop'}", flush=True)
        if ok and time.time() - t_start <= C.BUDGET["stage4"]:
            s3 = [SR.run("S3", "residual", best1["inputs"], spec, smoke=smoke), SR.run("S3", "direct", "summaries", spec, smoke=smoke)]
            log["stages"].append(dict(stage="S3", verdicts=s3))
        else:
            log["stopped"] = "S2 did not gain over S1" if not ok else "budget"
    accepted = [v for st in log["stages"] for v in st["verdicts"] if v["accepted"]]
    log["accepted"] = [dict(stage=v["stage"], mode=v["mode"], inputs=v["inputs"], gain=v["gain"], equations=v["equations"]) for v in accepted]
    print(f"  CELL {name}: {len(accepted)} accepted formula configuration(s)", flush=True)
    if not smoke:
        with open(C.OUTDIR / "stage4_cells.jsonl", "a") as fh:
            fh.write(json.dumps(log) + "\n")
    return log


if __name__ == "__main__":
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    t0 = time.time()
    for n in names:
        run_cell(n, "--smoke" in sys.argv, t0)
    print(f"STAGE4 DRIVER DONE {names} in {time.time() - t0:.0f} s")
