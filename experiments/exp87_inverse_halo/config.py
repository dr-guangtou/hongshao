"""exp87 — every constant, threshold, budget and seed of the inverse experiment.

Nothing here is tuned after seeing a real-data score: the plan
(`doc/plans/2026-10-02-exp87-inverse-halo.md`) fixes these values, and
`config_hash()` stamps every scoreboard row so a later change is visible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTDIR = HERE / "outputs"
FIGDIR = HERE / "figures/qa"

SEED = 87
ANCHOR_Z = (0.4, 0.7, 1.0, 1.5, 2.0)
ANCHOR_SNAP = (72, 59, 50, 40, 33)
EPOCH_TAG = ("z0p4", "z0p7", "z1", "z1p5", "z2")

#: the halo-mass selection edge of the parent sample at z = 0.4 [log10 Msun, M200c]
PARENT_CUT = 13.0
#: per-epoch halo-mass-completeness cuts (exp54 `selection.npz`): the lower edge
#: of the "complete above c_k" population
COMPLETE_CUTS = (13.0, 13.0, 13.0, 12.9, 12.8)

#: honesty devices
LOCKBOX_FRACTION = 0.2
N_FOLDS = 5
N_INNER_FOLDS = 4
N_BOOTSTRAP = 2000

#: scoring
CATASTROPHIC_DEX = 0.5
#: the threshold-weighted CRPS reads the massive end: weight 1[t >= this]
TW_THRESHOLD_LOGMH = 14.0
#: gridded predictives: step in the target's own unit
GRID_STEP = 0.0025
GRID_HI = {"mh": 16.0, "logc": 1.8, "t50": 14.0, "t75": 14.0, "t90": 14.0}
GRID_LO = {"mh": 10.5, "logc": -0.3, "t50": 0.0, "t75": 0.0, "t90": 0.0}
#: a gain is "significant" when the paired-bootstrap 95% interval excludes zero
#: AND the relative CRPS gain is at least max(this floor, the synthetic gate's
#: 95th-percentile null gain)
GAIN_FLOOR = 0.01

#: the fixed list of single apertures / annuli for the L0 reference [kpc]
APERTURES = {"M(<10)": (0.0, 10.0), "M(<30)": (0.0, 30.0), "M(<100)": (0.0, 100.0),
             "M(50-100)": (50.0, 100.0), "M(>50)": (50.0, None)}

#: Stage 0 certificates: the record's numbers, to be reproduced within this tolerance
CERT_SIGMA_MH_GIVEN_MSTAR = 0.175
CERT_SIGMA_MSTAR_GIVEN_MH = 0.140
CERT_TOL = 0.01
CERT_PARENT_N = 3388

#: Stage 2: the independence test's tolerance, in units of the forward scatter per coordinate
INDEPENDENCE_TOL = 0.2

#: budgets [seconds]
BUDGET = {"stage0": 1800, "stage1": 3600, "stage2": 1800, "stage3": 5400, "stage4": 8 * 3600,
          "stage5": 7200, "stage6": 3600}
MAX_HEAVY_JOBS = 2
GP_MAX_POINTS = 1500

#: symbolic regression stages: operators, complexity, wall time per fit [s]
SR_STAGES = {
    "S1": dict(binary=["+", "-", "*"], unary=["square"], maxsize=20, timeout=90),
    "S2": dict(binary=["+", "-", "*", "/"], unary=["square", "sqrt", "log", "exp"], maxsize=30, timeout=180),
    "S3": dict(binary=["+", "-", "*", "/", "pow", "max", "min"], unary=["square", "sqrt", "log", "exp", "tanh"],
               maxsize=40, timeout=420),
}
SR_MIN_FOLD_RECURRENCE = 4

#: raw box catalogue (27 GB; only the M200c of centrals is read and cached)
HALO_STRUCTURE_DIR_DEFAULT = Path.home() / "Desktop" / "tng300_halo_structure"


def config_hash():
    """A short hash of every public constant, stamped on each scoreboard row."""
    public = {k: (str(v) if isinstance(v, Path) else v) for k, v in globals().items()
              if k.isupper() and k not in ("HERE", "ROOT", "OUTDIR", "FIGDIR")}
    return hashlib.sha256(json.dumps(public, sort_keys=True, default=str).encode()).hexdigest()[:12]
