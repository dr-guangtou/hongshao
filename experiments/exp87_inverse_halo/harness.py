"""exp87 — the harness: one CELL = (sample, epoch, population, target, feature
set, method) scored out of fold on the frozen development folds.

`oof_scores` is the core (used by the synthetic gate as well): for each outer
fold the method is fitted on the other four and predicts the held-out one;
per-galaxy scores come back aligned to the rows. `run_cell` wraps it for the
real samples, writes `outputs/cells/<id>.npz` (per-galaxy scores and the
predictive's summaries) and appends one row to `outputs/scoreboard.jsonl`.
A cell already on disk is not recomputed (`force=True` to redo).

Populations (which galaxies, and the cut the predictive must respect):
  parent     the complete z = 0.4 population; halo mass cut at 13.0
  asis       the curated main-progenitor sample as it is; no cut
  complete   the curated sample above the epoch's completeness cut; cut there
  fit2356    the repo's fitting sample at z = 0.4 (sensitivity); cut at 13.0
The lockbox is never touched here (`lockbox.py` scores it once, at the end).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import config as C                                       # noqa: E402
import data as D                                         # noqa: E402
import scoring as S                                      # noqa: E402

CELL_DIR = C.OUTDIR / "cells"
SCOREBOARD = C.OUTDIR / "scoreboard.jsonl"
PER_KEYS = ("crps", "logscore", "pit", "err_mean", "err_median", "pred_mean", "pred_median", "twcrps", "latent_m", "latent_s")


def oof_scores(method, X, y, fold, cut, t0=None, grid=None):
    """Per-galaxy out-of-fold scores of `method`. `fold` (n,) ints, negative =
    not used. Returns (dict of (n,) arrays, NaN where unused; list of fit infos)."""
    X, y, fold = np.asarray(X, float), np.asarray(y, float), np.asarray(fold)
    per = {k: np.full(len(y), np.nan) for k in PER_KEYS}
    infos = []
    for f in sorted(set(fold[fold >= 0].tolist())):
        tr, te = (fold >= 0) & (fold != f), fold == f
        pred = method.fit_predict(X[tr], y[tr], X[te], cut)
        per["crps"][te], per["logscore"][te], per["pit"][te] = pred.crps(y[te]), pred.logscore(y[te]), pred.pit(y[te])
        mean, median = pred.mean(), pred.median()
        per["pred_mean"][te], per["pred_median"][te] = mean, median
        per["err_mean"][te], per["err_median"][te] = mean - y[te], median - y[te]
        if isinstance(pred, S.TruncNormal):
            per["latent_m"][te], per["latent_s"][te] = pred.m, pred.s
        if t0 is not None:
            per["twcrps"][te] = pred.twcrps(y[te], t0) if isinstance(pred, S.Gridded) else pred.twcrps(y[te], t0, grid)
        infos.append(dict(method.info))
    return per, infos


def summarise_per(per, y, used):
    """The scoreboard numbers of one cell from its per-galaxy arrays."""
    pit, err, errm = per["pit"][used], per["err_mean"][used], per["err_median"][used]
    out = dict(n=int(used.sum()), crps=float(np.mean(per["crps"][used])), logscore=float(np.mean(per["logscore"][used])),
               rmse=float(np.sqrt(np.mean(err ** 2))), mae_median=float(np.mean(np.abs(errm))), bias=float(np.mean(err)),
               catastrophic=float(np.mean(np.abs(errm) > C.CATASTROPHIC_DEX)),
               cover68=float(np.mean((pit > 0.16) & (pit < 0.84))), cover90=float(np.mean((pit > 0.05) & (pit < 0.95))),
               pit_mean=float(np.mean(pit)), pit_var=float(np.var(pit)),
               width_ratio=float(np.std(per["pred_mean"][used]) / np.std(y[used])))
    if np.isfinite(per["twcrps"][used]).all():
        out["twcrps"] = float(np.mean(per["twcrps"][used]))
    rel = S.reliability(per["pred_mean"][used], y[used])
    out.update(reliability_slope=rel["slope"], reliability_max_offset=rel["max_offset"])
    return out


# --------------------------------------------------------------------------- #
# real-data cells                                                               #
# --------------------------------------------------------------------------- #
_SAMPLE_CACHE = {}


def get_sample(sample, epoch):
    key = (sample, epoch)
    if key not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[key] = D.load_parent(verbose=False) if sample == "parent" else D.load_curated(epoch, verbose=False)
    return _SAMPLE_CACHE[key]


def population_rows(sample_obj, population, target):
    """(rows used, cut on the target). The cut applies to the halo-mass target only."""
    y = sample_obj.targets[target]
    ok = np.isfinite(y)
    mh = sample_obj.targets["mh"]
    if population == "parent":
        rows, cut = ok, C.PARENT_CUT
    elif population == "asis":
        rows, cut = ok & np.isfinite(mh), None
    elif population == "complete":
        rows, cut = ok & sample_obj.flags["complete"], C.COMPLETE_CUTS[sample_obj.epoch]
    elif population == "fit2356":
        rows, cut = ok & sample_obj.flags["fit2356"], C.COMPLETE_CUTS[sample_obj.epoch]
    else:
        raise KeyError(population)
    return rows, (cut if target == "mh" else None)


def build_design(sample_obj, feature, extra=None):
    """The design of a feature set, optionally with extra target columns appended
    (`extra="mh"` is the oracle extension: the true halo mass as an input)."""
    X, labels = D.feature_set(sample_obj, feature)
    if extra:
        for name in extra.split("+"):
            X = np.column_stack([X, sample_obj.targets[name]])
            labels = labels + [f"true_{name}"]
    return X, labels


def cell_id(spec):
    return "__".join(str(spec[k]) for k in ("sample", "epoch", "population", "target", "feature", "extra", "method")).replace("/", "-")


def run_cell(spec, method, force=False, smoke=False, verbose=True):
    """Score one cell and record it. `spec`: dict(sample, epoch, population,
    target, feature, extra, method). Returns the scoreboard row."""
    spec = dict(spec, extra=spec.get("extra") or "none", method=method.name)
    cid = cell_id(spec) + ("__smoke" if smoke else "")
    path = CELL_DIR / f"{cid}.npz"
    if path.exists() and not force:
        return json.loads(str(np.load(path, allow_pickle=True)["row"]))
    t0 = time.time()
    s = get_sample(spec["sample"], spec["epoch"])
    rows, cut = population_rows(s, spec["population"], spec["target"])
    X, labels = build_design(s, spec["feature"], None if spec["extra"] == "none" else spec["extra"])
    rows = rows & np.isfinite(X).all(1)
    y = s.targets[spec["target"]]
    fold = np.where(rows & ~s.lockbox, s.fold, -1)
    if smoke:
        keep = np.flatnonzero(fold >= 0)[::max(1, int((fold >= 0).sum() // 300))]
        thin = np.full(len(fold), -1)
        thin[keep] = fold[keep]
        fold = thin
    tw = C.TW_THRESHOLD_LOGMH if spec["target"] == "mh" else None
    grid = S.target_grid("mh", cut) if tw is not None else None
    per, infos = oof_scores(method, X, y, fold, cut, t0=tw, grid=grid)
    used = fold >= 0
    row = dict(spec, cell=cid, rung=method.rung, cut=cut, n_features=int(X.shape[1]), seconds=round(time.time() - t0, 2),
               config=C.config_hash(), **summarise_per(per, y, used))
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, row=json.dumps(row), index=s.index, used=used, y=y, fold=fold, infos=json.dumps(infos, default=float),
             **{k: per[k] for k in PER_KEYS})
    if not smoke:
        with open(SCOREBOARD, "a") as fh:
            fh.write(json.dumps(row) + "\n")
    if verbose:
        print(f"  [{row['rung']}] {cid:<78} n {row['n']:>4}  CRPS {row['crps']:.4f}  RMSE {row['rmse']:.4f}  "
              f"logscore {row['logscore']:+.3f}  cover68 {row['cover68']:.2f}  ({row['seconds']:.0f} s)", flush=True)
    return row


def load_cell(cid):
    z = np.load(CELL_DIR / f"{cid}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def read_scoreboard():
    if not SCOREBOARD.exists():
        return []
    rows = {}
    for line in SCOREBOARD.read_text().splitlines():
        r = json.loads(line)
        rows[r["cell"]] = r                              # the latest row of a cell wins
    return list(rows.values())
