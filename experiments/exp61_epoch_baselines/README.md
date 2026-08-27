# exp61 — single-epoch baselines and the z ≤ 1.0 scope

Branch `exp60-stochastic-layer` (this experiment runs alongside exp60; both
follow the user's 2026-08-26 direction). One script, `stage0_fits.py`; full
log `outputs/stage0_fits.log`. Seven fits of `gompertz_log-E2-S2` on the
standard sample under the standard mask: five independent single-epoch fits,
one z ≤ 1.0 fit, against the shared incumbent.

## The baseline table

Per-epoch score_A / score_F (lower is better; ~1 means as good as the
halo-mass regression benchmark refitted at that epoch):

| model | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| shared incumbent | 1.115 / 0.934 | 1.056 / 0.908 | 1.028 / 0.896 | 1.042 / 0.845 | 0.941 / 0.775 |
| single-epoch (each its own) | 1.109 / 0.888 | 1.050 / 0.890 | 1.015 / 0.891 | 1.011 / 0.827 | **0.835 / 0.744** |
| z ≤ 1.0 refit | 1.110 / 0.899 | 1.054 / 0.890 | 1.026 / 0.898 | 1.059 / 0.903 * | 1.011 / 0.906 * |

(* out of that fit's scope, evaluated for information only.) Single-epoch
loss gains over the incumbent's theta at that epoch: +4.5 / +2.3 / +1.8 /
+5.2 / **+15.9** percent; the z ≤ 1.0 refit gains +1.9% in scope.
Multi-start spreads 1e-9 to 1e-11, no parameter at a bound.

## The two answers

**1. The z ≤ 1.0 retreat is a REPORTING decision, not a refit — the shared
model is already at its class ceiling there.** Refitting to z ≤ 1.0 moves the
in-scope scores by under 0.01 while costing z=2 seven points of score_A. The
five-epoch fit was NOT compromising the low-redshift epochs to serve the high
ones; hongshao's z ≤ 1.0 performance is what the model class delivers, and
can be claimed as such with the incumbent unchanged.

**2. z=2 is the one epoch where the shared law's FORM binds.** Freed from
serving the other epochs, the same seven parameters reach score_A 0.835 —
better than the per-epoch benchmark — recovering 15.9% of the loss. So
roughly that much of the z=2 gap is not missing information but a functional
form that cannot bend enough: in the drift table every efficiency parameter
walks monotonically with epoch (`a0` −2.57 → −4.30, `a_M` −0.64 → −1.48,
`a_z` +0.57 → +1.68, `a_Mz` +0.42 → +0.86 from z=0.4 to z=2), and a drifting
`a_Mz` is curvature in the mass×time plane that the single cross term cannot
express. The size parameters drift oppositely (`log_f0` −1.09 → −0.72, `b`
−0.19 → −0.97) with the deposit shape `c` stable at 0.66–0.80.

**Caveats, stated**: the drift is INDICATIVE — single-epoch fits can slide
along the parameter degeneracies exp54 Stage 3.9 measured (the four
efficiency parameters absorb each other 12–18×), and no profile likelihood
was run. What is solid is the per-epoch score ceiling, which is a property of
the fits' losses, not of individual parameters.

## What follows from it

- The scope statement for hongshao v1: **validated at z ≤ 1.0 with the
  incumbent as adopted** (plus the exp60 layer if it graduates); z ≥ 1.5
  reported with the known gap.
- The z=2 recoverable component (~16% of per-epoch loss) is a MEAN-MODEL
  form question — a candidate for a mass×time-curvature term (`m·x²`), which
  no E-law variant has tried (E6/E7/E8 bend time only, at fixed mass slope).
  Parked as an open question: the user's stated read is that z ≥ 1.5 likely
  needs additional data (the real merger tree, 3-D information), and the
  stochastic layer leads the current work.

## Stage 1 — the M2 mass x time-curvature fit (2026-08-27): a clean NULL

The one-parameter curvature term the drift table pointed at
(`a_mz2 (logmh − 13.5) ln(1+z)²` per deposit, exp59's exact-nesting
machinery): all four starts — including one launched from the wrong sign and
one from strong curvature — converge to the same interior optimum,
`a_mz2 = −0.056`, loss −0.04%, shape error 13.754%, and the B gates exactly
at the null's values (z=2 bias −0.047 against the null's −0.046). Two things
are established, and the pre-registered sign (`a_mz2 > 0`) is FALSIFIED as a
loss-optimum prediction:

1. **The shared loss will not spend curvature on the z=2 bias** — the same
   0.6%-of-loss blindness exp58 P-B measured, now demonstrated on the exact
   term built to exploit it. The fit is well-identified; the null is real.
2. Together with exp58 P-C (a fit-sample median fix OVERSHOOTS the full QA
   population by +4% at z=2, because the mh-complete and full samples carry
   different biases), the z=2-median question is now fully mapped: reachable
   per epoch (the single-epoch fit), not adoptable by the shared loss (this
   fit), and ambiguous across samples (P-C). It stays parked; any future
   attempt must first decide WHICH z=2 population the model owes its median
   to, which is a scope question for the user, not a fitting question.
