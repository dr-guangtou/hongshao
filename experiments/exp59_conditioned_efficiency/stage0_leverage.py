"""exp59 Stage 0b — can a conditioned efficiency law EVER pass the B gates?
Measured by evaluation, before any fit.

THE QUESTION. Gate B2 requires the decliner-minus-rest amplitude split at z=2
(measured **−0.122 dex** — the decliners are the shortchanged group, so the
signed split is negative) to close to within ±0.060 dex WITHOUT the z=0.4
split (**+0.082 dex**, opposite sign — the crossing exp58 P-D measured)
getting worse. A per-deposit modulation `log_eps += e * v_j` shifts each
galaxy's amplitude at epoch k by roughly its deposit-weighted mean of
`e * v_j` up to k, so whether ANY coefficient serves both epochs is a
property of the VARIABLE, measurable without fitting. The loss cannot referee
this (exp58 P-B: the z=2 bias costs ~0.6% of the loss), so the sweep runs
first and the fits are spent only on candidates that can qualify.

THE QUALIFICATION RULE, declared before the numbers are read:

  A (split fix): some grid coefficient gives |z=2 split| ≤ 0.060 dex while
    |z=0.4 split| ≤ 0.092 dex (its size today, 0.082, plus a 0.010 guard).
  B (median fix): some grid coefficient gives |z=2 median bias| ≤ 0.020 dex —
    after removing the constant-plus-linear-in-ln(1+z) piece a base-law refit
    (`a0`, `a_z`) could absorb, an approximation and stated as one — while
    BOTH splits stay within their today's size + 0.010.

A candidate qualifying under neither is disqualified before any fit.

A SIGN BUG THIS FILE'S FIRST DRAFT HAD, recorded openly (exp59 P1): the z=2
split target was transcribed as +0.122 from exp58's magnitude table; the
SIGNED value is −0.122. The bug flipped the required coefficient's sign and
declared both candidates "worth a fit" when the correctly-signed reading shows
the trade. Caught by checking the sweep's e=0 row against exp58 P-D's group
medians — which is why the null row is printed first and compared.

Run: HONGSHAO_DATA_DIR=... OMP_NUM_THREADS=1 PYTHONPATH=. uv run python -u \\
     experiments/exp59_conditioned_efficiency/stage0_leverage.py [--smoke]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "experiments/exp38_deposit_rethink",
          ROOT / "experiments/exp54_unpinned_amplitude",
          ROOT / "experiments/exp57_expansion_term", HERE):
    sys.path.insert(0, str(p))

import conditioned as C                                  # noqa: E402
import fit as F                                          # noqa: E402
import stage0_cost as S0                                 # noqa: E402
import stage4_target as S4                               # noqa: E402

#: coefficient sweep, symmetric so both signs are searched
GRID = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)
#: the measured state this sweep must improve (exp58 P-D, SIGNED):
SPLIT_Z2, SPLIT_Z04 = -0.122, +0.082
#: gate thresholds (frozen in the plan): B2 |z=2 split|; the no-harm guard on
#: the z=0.4 split; B1's per-epoch median bias
B2_LIM, GUARD, B1_LIM = 0.060, 0.010, 0.020
RULE = "=" * 96

#: the candidates. F1/C1 are the model's own laws; the rest are sweep-only
#: variables served by `SweepProblem.cond_le` — promoted into `conditioned.py`
#: only if they qualify.
CANDS = ("F1", "C1", "R1", "F2", "H1")
DESCR = {
    "F1": "f_form_j - 0.5                  (assembly state at deposit)",
    "C1": "dlogc_j                         (concentration excess at deposit)",
    "R1": "log10(dmh_j/Mh_j), centred      (how burst-like the deposit is)",
    "F2": "(f_form_j - 0.5) * ln(1+z_j)    (assembly state x deposit epoch)",
    "H1": "max(0, 0.45 - f_form_j)         (active only once the halo stalls)",
}


class SweepProblem(C.ConditionedProblem):
    """`ConditionedProblem` with `cond_le` swapped for a sweep-only variable.
    Diagnostic only — `predict` itself is inherited, never copied."""

    def __init__(self, var, base, recs, data, mask):
        super().__init__(C.CSpec(base, "F1"), recs, data, mask,
                         epochs=(0, 1, 2, 3, 4))
        self.var = var

    def cond_le(self, le, cond, d):
        if self.var in ("F1", "C1"):
            return super().cond_le(le, cond, d) if self.var == "F1" else \
                le + cond[0] * d.dlogc
        if self.var == "R1":
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.log10(np.clip(d.dmh, 1e-30, None)) - d.logmh
            real = np.isfinite(v) & (d.dmh > 0)
            return le + cond[0] * np.where(real, v - np.median(v[real]), 0.0)
        if self.var == "F2":
            return le + cond[0] * (d.f_form - C.F_PIV) * np.log1p(d.z)
        if self.var == "H1":
            return le + cond[0] * np.maximum(0.0, 0.45 - d.f_form)
        raise ValueError(self.var)


def measure(pred, data, use, fell, ok, i100):
    """(z=2 split, z=0.4 split, per-epoch medians) — splits are decliner-
    minus-rest medians (global and a_z-like shifts cancel in the difference);
    the medians are adjusted by removing their best constant + linear-in-
    ln(1+z) piece, the freedom a base-law refit's (a0, a_z) could absorb."""
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.log10(pred[:, :, i100] / data[:, :, i100])
    a = np.where(use, a, np.nan)
    s2 = float(np.nanmedian(a[fell & ok, 4]) - np.nanmedian(a[(~fell) & ok, 4]))
    s0 = float(np.nanmedian(a[fell & ok, 0]) - np.nanmedian(a[(~fell) & ok, 0]))
    med = np.array([np.nanmedian(a[:, k]) for k in range(5)])
    x = np.log1p(np.array([0.4, 0.7, 1.0, 1.5, 2.0]))
    D = np.column_stack([np.ones(5), x])
    med_adj = med - D @ np.linalg.lstsq(D, med, rcond=None)[0]
    return s2, s0, med_adj


def main(smoke=False):
    print(f"{RULE}\nexp59 STAGE 0b — leverage: what each conditioning "
          f"variable can do to the B gates\n{RULE}\n")
    recs, data, mask, lmh, base, theta, slow = S0.build(smoke)
    i100 = F.I100
    fell, ok = S4.decline_flag(data)
    print(f"  sample: {len(recs)} galaxies; decliners "
          f"{int((fell & ok).sum())}, non-decliners {int(((~fell) & ok).sum())}")
    print(f"  state to improve (exp58 P-D, signed): z=2 split "
          f"{SPLIT_Z2:+.3f} dex, z=0.4 split {SPLIT_Z04:+.3f} dex")
    print(f"  RULE A: |z=2 split| <= {B2_LIM} with |z=0.4 split| <= "
          f"{abs(SPLIT_Z04) + GUARD:.3f}")
    print(f"  RULE B: |adjusted z=2 median| <= {B1_LIM} with both splits "
          f"within today's size + {GUARD}\n")

    verdicts = {}
    for var in CANDS:
        pr = SweepProblem(var, base, recs, data, mask)
        print(f"  {var}: v_j = {DESCR[var]}")
        print(f"    {'coef':>6} {'z=2 split':>11} {'z=0.4 split':>12} "
              f"{'adj z=2 median':>15}   qualifies")
        best = None
        for e in GRID:
            th = pr.cspec.nest(theta).copy()
            th[-1] = e
            pred = pr.predict(th)
            finite = np.isfinite(pred).all(axis=2) & (pred[:, :, i100] > 0)
            use = np.asarray(mask, bool) & finite & (data[:, :, i100] > 0)
            s2, s0, med_adj = measure(pred, data, use, fell, ok, i100)
            qa_ = abs(s2) <= B2_LIM and abs(s0) <= abs(SPLIT_Z04) + GUARD
            qb = (abs(med_adj[4]) <= B1_LIM
                  and abs(s2) <= abs(SPLIT_Z2) + GUARD
                  and abs(s0) <= abs(SPLIT_Z04) + GUARD)
            tag = ("A" if qa_ else "") + ("B" if qb else "")
            print(f"    {e:>+6.1f} {s2:>+11.4f} {s0:>+12.4f} "
                  f"{med_adj[4]:>+15.4f}   {tag or '-'}")
            if tag and (best is None or abs(s2) < abs(best[1])):
                best = (e, s2, s0, tag)
        verdicts[var] = best
        if best:
            print(f"    VERDICT: QUALIFIES ({best[3]}) at coef "
                  f"{best[0]:+.1f} — a fit is warranted.\n")
        else:
            print(f"    VERDICT: no grid point satisfies either rule — "
                  f"DISQUALIFIED before any fit.\n")

    print(f"{RULE}\n  SUMMARY: "
          + ", ".join(f"{v} {'QUALIFIES ' + verdicts[v][3] if verdicts[v] else 'disqualified'}"
                      for v in CANDS))
    if not smoke:
        np.savez_compressed(HERE / "outputs" / "stage0_leverage.npz",
                            cands=np.array(CANDS),
                            qualified=np.array([bool(verdicts[v])
                                                for v in CANDS]))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
