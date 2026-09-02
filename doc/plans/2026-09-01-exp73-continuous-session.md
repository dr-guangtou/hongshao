# exp73 — a 6–7 hour continuous session plan

Written 2026-09-01 for the next session, at the user's request: *"test the exp73
first, if the conclusion is clear and nothing special that needs my attention,
you can continue to explore the model and try to improve it based on our
discussion earlier."*

Branch **`exp73-size-relative-objective`**, already cut from merged master
(`5d142f4`). The science plan is
`doc/plans/2026-09-01-exp73-size-relative-objective.md`; this file is the
schedule and the decision rules.

## How to run this without supervision

**Every block ends in a written gate with a numeric threshold fixed in advance.**
A gate decides whether the next block runs, is skipped, or is replaced. Record
the gate's verdict in the log before acting on it. If a gate is ambiguous, take
the more conservative branch and say so.

**Long fits go in the MAIN shell with `nohup` and per-start checkpoints**, never
in a subagent. While a fit runs, do the block marked *parallel*.

**Standing constraints, none of which may be relaxed to make a result work:**

- `cog_provided` stays the fitting target (the user's D1). Nothing switches to
  the isophote density.
- THE FITTING SAMPLE rule is unchanged; profile-quality flags compose with it.
- A size-relative coordinate uses the **TRUTH's** R50, never the model's — a
  model scored on its own size scale can satisfy the objective by shrinking.
  Assert this in a selftest.
- Report the cumulative profile before any shell; state the axis range when
  pointing at a residual panel; lead with what a number means physically.
- Do not merge. Do not push to master.

**Wake the user (stop and write it up) if any of these happen:**

1. A result contradicts something already in the record — C18's answer, the
   mass–size relation holding to 5 per cent, or the fitting-sample rule.
2. A gate is passed only by changing the gate.
3. Anything suggests a published number is wrong.
4. Block E's fit rails a parameter that exp63 did not rail, or lands in more than
   one basin.

---

## Block A — the coordinate, and the risk test (target 0:00–1:15, no fit)

The risk test comes first because it can make Blocks B–D pointless.

**A1. Build the size-relative machinery** (~30 min).
`experiments/exp73_size_relative/coordinate.py`: shells in units of the truth's
R50; `M*(<k·R50)` by interpolation; a mass-weighted residual (error over the
galaxy's TOTAL mass); a coverage gate for shells where the measurement ran out.
Selftest: the coordinate uses truth-only inputs; a self-similar synthetic gives
identical shell fractions at every epoch; the mass-weighted and relative
residuals agree when all shells hold equal mass.

**A2. THE RISK TEST** (~45 min, no fit). Two questions, both answerable from fits
that already exist (`exp63/outputs/stage2_fit_frozen_binned_kpc_sane_e{0..4}.npz`
and `stage2_fit_joint_kpc_free_sane.npz`):

- **Does the joint-versus-per-epoch gap shrink under the new coordinate?** Score
  the joint fit and each single-epoch fit under (i) the production objective on
  fixed kpc and (ii) the mass-weighted objective on R50 shells. If the gap
  narrows materially, the coordinate was part of the problem. If it does not,
  the shared law is, and no objective change will fix it.
- **Do the per-epoch optima below z=2 lie on a smooth curve in time?** exp63
  Stage 5i found the sequence BREAKS at z=2 and read that as a mass-selection
  law (C18 then confirmed it). Below z=2 it looked smooth. Plot and fit the four
  z≤1.5 optima per parameter against log(1+z); report the residual scatter
  against the parameter's own conditional width.

> **GATE A.** If the gap narrows by less than 25 per cent under the new
> coordinate AND the z≤1.5 per-epoch optima lie within their conditional width of
> a smooth curve, then **the coordinate is not the limit and the time law is** —
> **skip Blocks B–D, go straight to Block E**, and record that exp73's premise
> was tested and did not survive. That is a real result, not a failure.

---

## Block B — the weighting sweep (target 1:15–2:30, no fit)

Only if Gate A did not divert. Implements the user's **D2**: schemes **A**
(smooth mass-fraction weights, no free parameter) and **B** (threshold gate, one
parameter `t`) as siblings, on identical ground, with the production objective as
the common reference.

Reuse exp72's Step 1 machinery wholesale — the participation ratio, the
share-of-loss-against-share-of-mass table, and the symmetric probes — and add a
**size probe** (perturb R50 at fixed total mass) alongside the amplitude and
shape ones.

> **GATE B.** A scheme proceeds only if, at every epoch, (i) no shell is charged
> more than **2x** its mass share, and (ii) its participation ratio is no worse
> than the production objective's. **And the D2 rule, fixed before any number is
> seen: if B's verdict moves materially as `t` sweeps 0.5 / 1 / 2 / 4 per cent,
> B is disqualified in favour of A.** If neither survives, skip Block C and go
> to Block E.

---

## Block C — one refit (target 2:30–5:00, ~2.5 h, background)

The winner of Gate B only. Same twelve parameters, bounds, starts, sample and
engine as `stage2_fit_joint_kpc_free_sane`, so the objective is again the only
change. Multi-start with the named nested start and the null asserted to 1.000
per epoch before any start runs. `nohup` in the main shell, per-start checkpoint.

Judge with exp72's `eval_gls.py` pattern: the transfer matrix, the standard
battery on both samples, shape and amplitude separated, **the mass–size relation
explicitly**, and exp71's tilt and leakage diagnostics.

---

## Block D — the QA additions (parallel with Block C, ~1:30)

The user's **D1**, independent of any fit, so it runs while Block C computes.

1. **Add R20** to `hongshao/qa.py` (`SIZE_FRACTIONS` currently `(0.5, 0.8, 0.9)`).
2. **Score the DISTRIBUTION, not the median**: at fixed stellar mass and at fixed
   halo mass, compare the model's and the truth's spread of R20/R50/R80 — the
   user's exact words. This is the detector for
   `fitting-harder-narrows-the-population`.
3. **Promote it to a numbered gate** with a threshold, reported for every
   product, rather than a figure that has to be read.
4. Extend `qa.py`'s selftest: a self-similar synthetic must pass; a model with
   deliberately narrowed sizes must fail.

Then re-score the three models already in hand (exp63 joint, chi2 refit, nested
incumbent) on the new gate — the first use of it is a check that it discriminates
something we already understand.

---

## Block E — the richer size-time law (target 5:00–6:15)

The user's **D3**, and the one part of the joint-versus-per-epoch gap that
neither selection nor overfitting explains. Register entry `D2` closed *richer
time dependence for the EFFICIENCY law* — but on the pre-exp63 seven-parameter
model, with no compact channel and no size-law time dependence. exp63 gave the
sizes one power of (1+z) each (`b_c`, `b_e`) and **nobody has asked whether a
richer size-time law closes the z≤1 gap.**

**Evaluation first** (~30 min): does the z≤1.5 per-epoch sequence actually demand
more freedom than one power of (1+z)? Block A2 already measures this; if the
smooth curve fits within the parameters' conditional widths, a richer law has
nothing to collect and Block E stops there — the same shape of answer register
`D2` got.

**Then, only if it does** (~45 min fit): the smallest nested extension that could
collect it — a second term in log(1+z) for `b_c` and `b_e` — fitted at **z ≤ 1.0
only**, per D3, with the z=2 epoch reported and not targeted. Two extra
parameters, both nesting at zero.

> **GATE E.** The extension must beat the nested form by more than the 0.2
> percentage points that register `D2` found was the entire budget for the
> efficiency law's time dependence. Below that it is the same negative result and
> should be recorded as one.

---

## Block F — consolidate (target 6:15–7:00)

README for exp73 with every table; `doc/open_questions.md` (C21 resolved or
sharpened; C20 updated with whatever Gate A said about the shared law); lessons;
memory; `doc/todo.md`; a handover appended to
`doc/journal/2026-09-01_handover.md` or a new dated file. Commit throughout, not
only here. **Do not merge.**

## What "improving the model" means here, and what it does not

The user asked to *"continue to explore the model and try to improve it"*. On the
evidence now in the record, the honest reading is:

- **Improving the z=2 numbers by fitting them harder is not improvement.** C18
  showed the z=2 halo-mass dependence is the selection; exp61 showed per-epoch
  fits overshoot the population; exp63's own per-epoch sequence breaks at z=2.
  Report z=2 as scope.
- **The live levers are the coordinate (Blocks A–C) and the size-time law
  (Block E)**, in that order, and both may return negative. Two objectives have
  now been re-weighted and each moved the compromise without raising the ceiling;
  a third negative would be worth stating plainly as a class result rather than
  dressed as progress.
- **C19 is the largest open item and is deliberately NOT in this schedule.** It
  needs a fit, it touches every high-redshift number in the programme, and it
  deserves the user's attention before it is spent — 6 hours is not enough to do
  it and its consequences properly.
