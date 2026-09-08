# exp76 — let the deposit's shape depend on how fast the halo was growing

Branch `exp76-growth-rate-split` (worktree `hongshao_exp76_growth_rate_split`),
from `master` `360d2d6`. Plan `doc/plans/2026-09-07-exp76-growth-rate-split.md`.
Started 2026-09-08 00:20, the user's overnight run, after the adoption of the
measured halo history (`exp74/rebaseline.py`, branch `adopt-measured-input`).

## The idea in plain language

The model builds a galaxy by depositing stars along its halo's growth, and
each deposit is either compact (a few kpc) or extended (tens of kpc). Until
now the choice depended only on how massive the halo was at that moment.
exp63's open problem P9 found that this gets the assembly history backwards:
in the data, galaxies whose haloes formed early hold more of their stars
compactly; in the model it is the late formers. A no-fit probe showed that
one extra number fixes the signs — let the choice also depend on how FAST the
halo was growing when the stars arrived, so that mass arriving in a burst
(a merger) lands extended and mass arriving during quiet growth lands
compact. It was never fitted, because the growth rate then came from the
DiffMAH curve, which we now know is a smooth function of the halo's final
mass (C19). Under the measured history the growth rate at every moment is the
real one. exp76 asks whether the extra number earns its place there.

## Stage 0 — leverage by evaluation, no fit (`stage0_leverage.py`, `outputs/stage0_leverage.log`)

Everything held at exp74's measured-input optimum and only `g_split` swept:
a conditional slice, read for the shape of the trade, never for an optimum.

### The assembly correlations, re-measured on the honest input

Partial Spearman correlation at fixed z = 0.4 halo mass (measured) between a
galaxy's compact share and its assembly history. Two sets of assembly
variables: MEASURED from the running-peak history (t50, t80 = when the halo
reached half / 80 per cent of its z = 0.4 mass; growth over the last 2.1 and
3.5 Gyr) and the DiffMAH-derived set P9 used (`late`, `f_form`, `logtc`,
`t50`). The data's compact share is exp63 Stage 1's deconvolved inner share
(mass of the z = 0.4 deposit-size distribution inside 5 kpc).

| compact share | t50 | t80 | growth 2 Gyr | growth 3.5 Gyr | late | f_form | logtc | t50 (DiffMAH) |
|---|---|---|---|---|---|---|---|---|
| **the data** | −0.07 | +0.02 | +0.02 | +0.02 | +0.25 | −0.26 | −0.19 | −0.06 |
| model, g = 0 | +0.31 | +0.30 | +0.30 | +0.30 | −0.03 | +0.19 | +0.11 | +0.30 |
| model, g = −1 | +0.15 | +0.15 | +0.16 | +0.15 | −0.01 | +0.04 | +0.15 | +0.17 |
| model, g = −1.5 | +0.04 | +0.08 | +0.10 | +0.08 | +0.11 | −0.10 | +0.00 | +0.05 |
| model, g = −2 | −0.10 | −0.02 | +0.01 | −0.02 | +0.19 | −0.22 | −0.12 | −0.08 |

Three things to read off it.

1. **Against the measured assembly variables the data show no correlation
   at all** (|ρ| ≤ 0.07), while the g = 0 model's compact share rises with
   late formation and recent growth at +0.30. The model has a dependence the
   simulation does not have.
2. **Against the DiffMAH variables the data DO correlate** (+0.25 with
   `late`, −0.26 with `f_form`) and the g = 0 model has the wrong signs, as
   P9 found. That the data correlate with the smooth-fit parameters but not
   with the measured formation time is itself a finding: those parameters
   carry something the measured history does not — the whole-history fit's
   response to the future (`late` is the slope to z = 0) — and P9's target
   was partly a property of the parametrisation. Recorded as a new open
   question, not resolved here.
3. **At g ≈ −1.5 to −2 the model's correlations sit within ~0.1 of the
   data's on all eight variables** (rms distance 0.33 at g = 0 → 0.06 at
   g = −2): the measured-variable dependence goes to zero and the DiffMAH
   ones take the data's signs and sizes. The gate as first written ("all
   measured-variable signs match") was ill-posed, because the data's signs
   there are noise around zero; the honest criterion is the distance, and it
   is met.

### What the slice does to the rest (everything else fixed)

| g | loss (adopted refs; null 20.85) | R50 width ratio z=0.4 / z=2 | centre at 2 kpc, z=0.4 / z=1.5 / z=2 | z=2 mh-complete total |
|---|---|---|---|---|
| 0 | 15.56 | 0.66 / 0.35 | +5.6 / −9.8 / −12.6 % | −1.0 % |
| −1 | 16.67 | 0.66 / 0.41 | +11.7 / −4.2 / −6.5 % | −1.1 % |
| −2 | 16.64 | 0.70 / 0.42 | +13.0 / −3.6 / −6.2 % | −1.1 % |

On the slice, g = −2 **halves the high-redshift central deficit** (z = 2:
−12.6 → −6.2 per cent; z = 1.5: −9.8 → −3.6) and **widens the size
distribution at fixed mass by a fifth at z = 2** (0.35 → 0.42), at the price
of a z = 0.4 centre that overshoots (+5.6 → +13.0 per cent) and a loss 7 per
cent worse. That is the expected shape of a slice: the compact share rises
everywhere; the fit will lower the split's mass scale to compensate, and the
question for Stage 1 is whether the model can keep the high-z gain without
the low-z overshoot.

Figure `figures/exp76_stage0_sweep.png`: left, the model's correlations with
the measured variables against g, the data's values dotted; middle, the R50
width ratio per epoch; right, the loss and the z = 2 centre.

**Decision: Stage 1 runs** (the plan's Gate A, read as the distance criterion,
is met at g ≈ −1.5 to −2).

## Stage 1 — the fit, and the verdict (`stage1_fit.py`, `stage1_eval.py`; `outputs/stage1_eval.log`)

exp63's model with `g_split` free (13 parameters) on the measured curves
under the adopted references. Seven starts, one process each: four from
exp74's measured optimum (g = 0, −1, −2, and a near start at −1.5) and, once
the adoption's re-baseline had found its lower basin (14.63; C23), three from
that basin (g = 0, −1, −2).

| start | from | loss | fitted g |
|---|---|---|---|
| g = 0 | exp74's point | 15.560 | 0.00 (stationary) |
| g = −2 / near −1.5 | exp74's point | 15.134 / 15.141 | −0.31 / −0.30 |
| **g = 0** | the lower basin | **14.635** | **0.00** (stays exactly) |
| g = −1 / −2 | the lower basin | 14.998 / 15.057 | −0.27 / −0.26 |

**The objective rejects the split.** From the baseline optimum the loss has
no gradient in g: the fit leaves it at 0.000 and moves nothing else. Started
at g = −1 or −2, it walks back to g ≈ −0.27 and stops 2.5 per cent worse. The
best solution with the split active is therefore a start-dependent one, not
an optimum of the loss.

### What the split does when it is on (g = −0.27 against the baseline, same objective)

| | baseline (g = 0) | split, g = −0.27 | the data |
|---|---|---|---|
| loss (adopted refs) | 14.635 | 14.998 (+2.5 %) | |
| compact share of the deposits (median) | 0.44 | 0.14 | 0.21 (inner share) |
| rms distance of the 8 assembly correlations to the data's | 0.46 | 0.27 | 0 |
| compact share vs recent growth (measured), partial ρ | +0.55 | +0.22 | +0.02 |
| centre at 2 kpc, z = 0.4 / z = 2 | +9.0 / −13.2 % | +5.4 / −11.1 % | |
| R50 width ratio at fixed M*, z = 0.4 / z = 2 | 0.57 / 0.42 | 0.66 / 0.49 | 1 |
| **R80 width ratio, z = 0.4 / z = 2** | 0.48 / 0.22 | **0.63 / 0.48** | 1 |
| R50 offset, z = 1.5 / 2 | +17 / +21 % | +13 / +18 % | |
| R50 offset, z = 0.4 / 0.7 | −4 / −3 % | −8 / −7 % | |
| size-gate OFFSET passes | 10 of 15 | 11 of 15 | |
| z = 2 massive progenitors, total | +3.8 % | +2.6 % | |
| z = 1.5 massive progenitors, inside 10 kpc | +4.2 % | +7.5 % | |

**The positive side.** With the split on, the model's galaxies of a given
mass become far more diverse in size — the outer size (R80) at z = 2 goes
from 0.22 to 0.48 of the real spread, more than doubling, the largest single
gain on the width gate the programme has seen from a mean model (exp73
Block D: every mean model 0.2–0.6, collapsing with z). Both centres improve
at once (z = 0.4 +9.0 → +5.4 per cent, z = 2 −13.2 → −11.1), the high-z sizes
come down, and the compact share's spurious dependence on recent growth
drops by more than half. The mechanism does what Stage 0 said: mass
arriving in bursts is spread out, mass arriving quietly is kept compact,
and each galaxy's own history then sets its size.

**The negative side.** The loss does not want it (+2.5 per cent), so it is
not an optimum; low-z sizes overshoot the other way (−8 per cent at z = 0.4);
the z = 1.5 massive progenitors get heavier inside 10 kpc; and the fitted g
(−0.27) is far from the g ≈ −2 that Stage 0 needed to reproduce the
DiffMAH-variable correlations — at −0.27 those signs are still mostly the
model's, not the data's (`f_form` +0.05 vs the data's −0.26, `logtc` +0.20 vs
−0.19).

### The verdict

**Not adopted, and not closed.** The growth-rate split is a real lever on
the thing the model most lacks — size diversity at fixed mass — and it works
by a mechanism the data support (the compact share should not follow recent
growth, and with the split it stops doing so). But exp63's objective cannot
see sizes and ranks it as a 2.5 per cent loss, so a fit will always switch
it off. That is C20 and C23 again from the other side: the lever the gates
want is one the loss rejects. Two ways forward, both the user's call:

1. **Give the objective a size term** (the R50/R80 offset and width at
   fixed mass, from the gate itself), then refit with g free. If the split
   is real, a loss that can see sizes will keep it.
2. **Keep g as a fixed, physics-set parameter** (−0.3 to −2; Stage 0's
   sweep says the correlations want −2, the fit says −0.3) and let the
   stochastic layer own the rest of the width.

And C22 first: the data's compact share follows DiffMAH's `late`/`f_form`,
not the measured formation time — until that is understood, "the assembly
correlations" are not a safe target.

Figures: `figures/exp76_stage0_sweep` (the slice), `figures/qa/qa_bins_exp76_*`
(the halo-mass-binned curves of growth for the baseline, the g = 0 fit and
the g = −0.27 solution; the third shows the z = 1.5 outskirts in the most
massive third still +8 per cent, and the z = 2 centre in that bin −18 per
cent against the baseline's −20).
