# Session Handover — 2026-08-23 (third session)

## Goal

Build the **profile-shape representational ceiling** agreed at the end of the
previous session, and then — contingent on what it said — test one nested
parameter, a redshift-dependent deposit shape `c = c0 + c_z ln(1+z)`.

**Both are done.** The ceiling answered the question, and the answer was not
one of the three options the plan anticipated. The contingent parameter was
tested at the level of the bound and rejected without a fit.

## THE HEADLINE

> **The basis can draw the profiles. It cannot draw the CHANGES between them.**

Given per-epoch freedom, the model's own deposits reproduce every galaxy's
curve of growth to about **one per cent at every radius** — so neither the
profile family, nor the size law, nor the truncation is the limitation. What
costs a factor of 3.3 in the bound, with no change of basis at all, is the
requirement that **one causal non-negative weight vector serve all five
epochs**. That is the deposition-only premise, and it is now the binding
constraint on the shape half of the model.

## What was built

`experiments/exp54_unpinned_amplitude/stage34_ceiling.py` — for each galaxy the
deposits form a fixed basis (one truncated unit-mass cumulative profile per MAH
step, `R50` from the fitted size law, truncated at `3 R200c(t_j)`). Replace the
seven-parameter weight law with non-negative least squares over all 71 weights,
respecting causality. Convex, no optimiser, **5 ms per galaxy**, 2397 galaxies
in 13 s.

`stage34_basis_scan.py` — the same bound recomputed over a grid of GLOBAL basis
parameters with the per-galaxy freedom held fixed, so a gain cannot come from
degrees of freedom. Snapshot-vectorised (all galaxies share one 72-step deposit
grid, which the run asserts), 49 grid points in 0.3 min on 799 galaxies.

`stage34_figures.py` — two figures.

**Two gates, both at 0.00e+00.** `_score_masked` must reproduce
`stage33_perepoch.MaskedProblem.per_epoch`, and `B @ (eps dMh)` must reproduce
`model.forward`. Without the first, a bound and the model it bounds are not
scored by the same arithmetic; without the second, the bound is on a different
model. Both refuse to run on failure.

## The ladder — `gompertz_log-E2-S2`, `score_F` on the per-epoch mh-complete mask

| rung | freedom/galaxy | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | mean |
|---|---|---|---|---|---|---|---|
| `monotone` (enclosed mass may not fall) | none, basis-free | 0.349 | 0.293 | 0.263 | 0.255 | 0.309 | 0.294 |
| `epoch-alone` (each epoch fitted alone) | 5 x ~71 | 0.094 | 0.134 | 0.184 | 0.220 | 0.212 | 0.169 |
| **`ceiling`** (one causal weight vector) | **71** | **0.604** | **0.542** | **0.541** | **0.530** | **0.581** | **0.559** |
| `ceiling_fb` (same, `eps_surv <= f_b`) | 71, capped | 0.609 | 0.546 | 0.543 | 0.532 | 0.582 | 0.563 |
| `interval5` (one weight per interval) | 5 | 0.856 | 0.785 | 0.783 | 0.697 | 0.777 | 0.780 |
| `model` (the fitted law) | 0 (7 global) | 0.934 | 0.909 | 0.897 | 0.846 | 0.777 | 0.872 |
| *control*: ceiling fitted to ANOTHER galaxy | 71 | 0.756 | 0.709 | 0.686 | 0.597 | 0.613 | 0.672 |
| *reference*: another galaxy's truth, unfitted | — | 1.728 | 1.717 | 1.675 | 1.459 | 1.215 | 1.559 |

`score_A` (mean over epochs): `monotone` 0.026, `epoch-alone` 0.072, `ceiling`
0.153, `interval5` 0.273, **`model` 1.237**, *permuted control* 0.260.

`moffat-E2-S2` reproduces the whole ladder to within 0.03, so none of this
belongs to one profile family.

## Where the constraint bites, exactly

The causal design is block-triangular: the deposits of one epoch interval are
the only things that can build that interval's rise in `M*(<R)`. Two ways that
fails, both priced as an RMS error in the cumulative curve as a fraction of
`M*(<R)`:

| interval | radii where the truth DECREASES | cost of clipping that to zero | cost of the basis on the rise |
|---|---|---|---|
| z=2.0 to 1.5 | 31.4% | 2.88% | 6.29% |
| z=1.5 to 1.0 | 31.8% | 4.26% | 8.32% |
| z=1.0 to 0.7 | 36.3% | 2.48% | 5.67% |
| z=0.7 to 0.4 | 34.3% | 2.67% | 5.79% |

## Three things this retires or changes

### 1. The compact second deposit channel — RETIRED

At 2 kpc and z=2 on the mh-complete sample: **model −25.4%, ceiling −17.1%,
basis-free monotone floor −17.4%.** About two thirds of the model's central
deficit at z=2 is irreducible for **any** deposition-only model of **any**
basis, and the ceiling — free to put its 71 weights on sub-kpc deposits laid
down at z > 10 — recovers essentially nothing beyond it. The obstacle is not
that compact mass is unavailable; it is that mass, once deposited, cannot be
removed, and TNG's enclosed mass at 2 kpc genuinely falls between z=2 and
z=0.4. A compact channel would be fitting a structural residual.

### 2. `c = c0 + c_z ln(1+z)` — REJECTED AT THE BOUND, not built

| scan | result |
|---|---|
| the size law `(log_f0, b)` at the ceiling, 7x7 | ceiling-optimal `(-0.646, -1.286)` against fitted `(-0.846, -0.886)`, worth **+0.4%**. Already at its optimum along a degenerate ridge. |
| `(c0, c_z)` at the 71-weight ceiling, 7x7 | optimum at **`c_z` = 0**, `c0` = 0.798 (the fitted value). Interior minimum, surface rises both ways. |
| `(c0, c_z)` at the 5-weight `interval5` rung, 7x7 | optimum at **`c_z` = 0**, `c0` = 0.798. |

Scan C exists because a 71-weight bound could in principle be insensitive to a
basis change the fitted model would still feel; it is not. **What the scan does
NOT formally exclude** is a `c_z` that helps by compensating for the rigidity of
the `E2` efficiency law rather than by improving the basis — a mechanism nobody
has proposed and which the "too diffuse" argument does not supply. If you want
it built anyway, that is the argument you would have to make first.

### 3. The R^(1/4) outer-slope headline is HALF SURVEY, and the rest is the objective

Two separate corrections to the previous session's motivating figure.

**It is half survey.** The truth's outer density slope steepens −2.78 to −3.38
on the FULL progenitor-selected sample; on the **per-epoch mh-complete sample
it steepens only to −2.95**, and on one fixed galaxy set −2.56 to −2.95. The
model's z=2 error is **+0.21 dex/dex** (`gompertz_log`) or **+0.06**
(`moffat`), not +0.48. This is the same trap as tech-note §6.4b, one day later,
in a new figure.

**The rest is the objective, not the model.** `epoch-alone` matches the
cumulative curve to ~1% at every radius and is still 0.2 dex/dex too shallow,
because at z=2 only **6.3%** of the stellar mass lies outside 52 kpc — a
sub-per-cent error on the cumulative curve is a thirty-per-cent error in that
annulus, and a fractional loss on `M*(<R)` cannot see it. **Fitting harder will
not fix the outskirts. Only changing the objective will.**

## THE CAUTION, MEASURED RATHER THAN ASSERTED

The user asked for the ceiling to be reported as a bound, and specifically that
a small ceiling must not be read as a target. It is quantified three ways:

1. **The permuted control** — one galaxy's deposits fitted to a *different*
   galaxy's measured profile — reaches `score_F` **0.672** and `score_A`
   **0.260**, against the ceiling's 0.559 and 0.153 and the model's 0.872 and
   1.237. **A 71-weight solve fitted to the WRONG galaxy beats a five-weight
   oracle fitted to the RIGHT one, and is four to five times better than the
   fitted model in amplitude while carrying no information about the galaxy at
   all.** Most of the ceiling's margin is generic flexibility.
2. **Sparsity**: NNLS activates a median of **6 of 71** weights (10th-90th
   percentile 4-7), so it is not using its nominal freedom evenly.
3. **Physical admissibility**: capping `eps_surv` at `f_b = Omega_b/Omega_m`
   costs **+0.004** in mean `score_F`, so the ceiling is not an arithmetic
   curiosity — but 2.6% of its deposits do ask for more stars than baryons, and
   the cap is what keeps the bound honest.

**The defensible headroom for a richer efficiency law is the gap to
`interval5`: 0.09 in mean `score_F`, and zero at z=2.** Not the 0.31 gap to
the ceiling.

One more honest caveat, recorded in the module docstring: NNLS minimises the
summed squared fractional residual of the ABSOLUTE curve of growth, which is
close to but not identical with the production objective. So every reported
NNLS `score_F` is an UPPER bound on that rung — which makes "the ceiling is
this low" conservative, and is why "the ceiling is this high" statements are
checked against the basis-free `monotone` rung instead.

## In Progress

**Nothing is running.**

## NEXT — three candidates, in the order the ceiling supports

1. **Change the objective before changing the model.** The blindness is now
   quantified (6.3% of the mass beyond 52 kpc at z=2). exp48 already measured a
   density-plus-log-residual objective that improves the compact-centre defect;
   it was not adopted for unrelated reasons (it steepens the stellar-mass
   decline with redshift the wrong way) and deserves re-examination now that we
   know the production objective cannot see the outskirts at all. **This is the
   only move the ceiling actively endorses.**
2. **The amplitude, and why the model degrades on complete massive samples.**
   `score_A` 1.237 against a ceiling of 0.153 — but the permuted control
   reaches 0.260 while fitting a different galaxy, so most of that gap is
   per-galaxy flexibility. The honest headroom is the gap to `interval5`, 0.27.
   Still unexplained and still the deployment case.
3. **Hold z=2.0 out of the fit** (unchanged from the last two handovers). The
   catch is also unchanged: the Stage 3.2 factorial fits only the two
   ENDPOINTS, so a clean hold-out must re-select as well as re-fit.

Deliberately NOT next: the compact second channel (retired above), `c_z`
(rejected above), and the delivery-delay kernel (still a last resort, still
needs an empirical halo-mass dependence).

## Figures

- `figures/stage34_ladder_<label>.png` — the ladder, plus where each bound's
  residual sits in radius at every epoch, with the permuted control drawn
  alongside so the margin can be read for what it is.
- `figures/stage34_cz_<label>.png` — the `(c0, c_z)` surface at both freedom
  levels and a cut at the fitted `c0`.

A house bug was fixed on the way: `savefig.bbox = "tight"` means matplotlib's
own `wrap=True` silently WIDENS the canvas instead of wrapping, which is why
the existing QA figures render at 3.4:1 rather than their requested 2.15:1.
`stage34_figures._caption` pre-wraps the string; the older figures still have
the issue.

## Warnings carried forward (new ones first)

- **Bound the framework before enriching it, and price the bound's own
  freedom.** A ceiling with per-object freedom always looks reachable. It ships
  with a permuted-target control, a count of how much of its nominal freedom it
  actually uses, and a rung at a freedom level a global law could reach.
- **Test a proposed parameter at the BOUND before fitting it.** Minutes of
  compute against a day of editing shared code. Scan at more than one freedom
  level: the widest bound can be insensitive to a basis change the model feels.
- **A metric can be blind in a region and still look converged there.** Ask
  whether the objective could have detected a defect before naming it.
- **The completeness trap recurs in every new view.** Apply the mh-complete
  mask the FIRST time a diagnostic is plotted, not after it makes a headline.
- A benchmark and the thing it benchmarks must be scored on the same objects —
  assert it. Classify what a fix invalidates from what the code READS. Never
  wrap an import in a bare `except`; import a same-named sibling BY FILE PATH,
  and import `fit` before `halo`/`model` in this experiment. An unbounded
  metric can be dominated by one object. `setsid` does not exist on macOS and a
  `ps` snapshot is not proof; watch the LOG ADVANCE. Long runs checkpoint per
  unit of work.
- Rank by the judge, NEVER by loss. Truncating a profile moves its half-mass
  radius. A shuffle control must REFIT. Multi-start is mandatory. `np.interp`
  CLAMPS. A SLICE is not a PROFILE. usetex eats `%` — use `hongshao.qa._pct()`.
  Never create a term without defining it in place.
- Always `export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof`;
  KEEP the 27 GB cache at `~/Desktop/tng300_halo_structure` — `selection.py`
  needs the RAW catalogues.

## Branch State

Branch **`exp54-unpinned-amplitude`**, clean, **48 commits ahead of master, NOT
pushed**. master @ `3b376f1`. Read `doc/tech_note/deposition_model_2026-08.md`
§8.5 for the full write-up (also in the PDF, rebuilt).

---
*Session: eb8b15e6 — resume with `claude --resume eb8b15e6`*
