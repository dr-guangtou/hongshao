"""exp53 — does ANY deposition-only model beat the fiducial, in ANY aspect?

Direct answer to the user's question (2026-08-21). Every judged metric exp53
recorded, with each deposition-only candidate scored by its DISTANCE TO THE
TARGET rather than by its raw value, so "better" means "closer to what the
data say" and not "larger" or "smaller".

Targets: 0 for a bias or an offset; 1.0 for a growth ratio and for a plane
E/floor (the truth's own split-half floor); the MEASURED value for the two
quantities that have one (the differential gate and the kernel Sersic index).

Reads `outputs/verdict_shootout.json`, written by `report.py table`.

Run: `PYTHONPATH=. uv run python \
experiments/exp53_deposition_only/headtohead.py [--baseline NAME]`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: (label, json key, target). None = look the target up from the data columns.
METRICS = [("added-light kernel [dex]", "kern_dex", 0.0),
           ("kernel Sersic n vs data", "n_model", None),
           ("M(<10) growth vs truth", "grow_m10", 1.0),
           ("M(<5) bias, all", "m5_all", 0.0),
           ("M(<5) bias, COMPACT", "m5_compact", 0.0),
           ("M(<5) bias, extended", "m5_extended", 0.0),
           ("GATE differential", "gate_diff", None),
           ("GATE outskirt overshoot", "gate_over", 0.0),
           ("dlog R50 at z=0.4", "r50_z04", 0.0),
           ("dlog R50 at z=2.0", "r50_z20", 0.0),
           ("tier 2b plane z=0.4", "plane_z04", 1.0),
           ("tier 2b plane z=2.0", "plane_z20", 1.0),
           ("compact-galaxy tilt rho", "tilt_m5", 0.0)]


def main(baseline="moffat-qfree", path=None):
    path = path or HERE / "outputs/verdict_shootout.json"
    V = {r["name"]: r for r in json.loads(Path(path).read_text())}
    if baseline not in V:
        raise SystemExit(f"baseline {baseline!r} not in {path}")
    base = V[baseline]
    cands = [n for n in V if n.endswith("-q0")]
    print(f"exp53 — does ANY deposition-only model beat the fiducial "
          f"({baseline})?")
    print("  scored by DISTANCE TO TARGET; * = closer to the data than the "
          "fiducial\n")
    print(f"  {'metric':<28}{'fiducial':>10}"
          + "".join(f"{n.replace('-q0', ''):>14}" for n in cands))
    wins = {n: [] for n in cands}
    for lab, k, tgt in METRICS:
        if tgt is None:
            tgt = base["gate_diff_data"] if k == "gate_diff" else base["n_data"]
        b = abs(base[k] - tgt)
        row = f"  {lab:<28}{base[k]:10.3f}"
        for n in cands:
            d = abs(V[n][k] - tgt)
            if d < b - 1e-9:
                wins[n].append(lab)
                row += f"{V[n][k]:13.3f}*"
            else:
                row += f"{V[n][k]:13.3f} "
        print(row)
    print("\n  WINS per deposition-only model:")
    for n in cands:
        print(f"    {n:<18}{len(wins[n]):>2}/{len(METRICS)}")
        for w in wins[n]:
            print(f"        + {w}")
    return wins


if __name__ == "__main__":
    bl = "moffat-qfree"
    if "--baseline" in sys.argv:
        bl = sys.argv[sys.argv.index("--baseline") + 1]
    main(bl)
