# Session Handover — 2026-08-23 (second session)

## Goal

Answer the user's three Stage 3.4 questions before building anything: the
progenitor-selection confound, the delivery-delay kernel, and the inner
deficit. **All three are answered.** Two defects in the SCORING were found on
the way, and both change what the previous session's results mean.

## THE HEADLINE, AND ITS CORRECTION

**The previous handover's headline does not survive.** It claimed
`gompertz_log-E2-S2` beats the best halo-only regression at four of five
epochs (`score_A` 1.128 / 0.907 / 0.899 / 0.929 / 0.970). Under a corrected
benchmark it is **1.131 / 1.100 / 1.059 / 1.058 / 1.050 — worse than the
regression at every epoch, by 5-13%.**

The physics results are untouched. What was wrong was the comparison.

## The two scoring defects

### 1. `M200c` read as linear when it is already log10 (`dc226b0`)

`halo.py:132` and `scoreboard.py:136` converted the halo-structure catalog's
`M200c` column as a linear mass in `1e10 Msun/h`. It is already
`log10(M/Msun)` — verified against `population.npz["logmh"]`, which it
reproduces to **max |diff| = 0.0000** at snap 72. The Diemer19 reference
concentration was therefore evaluated **2.0006 dex too low**.

Effect on `dlogc = log10[c200c / c_Diemer19]`: a spurious 0.15 dex redshift
ramp between z=2 and z=0.4, and a halo-mass slope that changes sign at z=0.4
(−0.053 as coded against +0.047 correct, dex/dex).

Scope: after removing a linear trend in `log10 Mh` the buggy and correct
variables agree to correlation 0.9987–0.9995, so forms with a linear mass term
absorb most of it. The consumers are `E3`, **`S2a`** (`r50_of` uses
`R200c/c_eff` for every size except exactly `S2`) and `S3` — **33 of 45**
factorial cells. `_assert_m200c_is_log10` now makes the units a test that
fires on the shipped transformation.

### 2. `SIGMA_A` inflated by one galaxy the model never saw (`91a26f3`)

Row 181's `M*(<100 kpc)` collapses from 10^11.712 at z=0.4 to a flat ~10^7.96
at all four higher-redshift epochs — 2.5 dex below the sample's 0.1st
percentile, a broken cross-match. The next-largest backward drop is 2.03 dex
(row 2372, a genuine fast grower) and moves the baseline by ≤0.0015 dex.

**Row 181 is in Stage 1's 2397, which sets `SIGMA_A`, and in NONE of the
model-fitting subsamples** (Stage 3.1's 300, 3.2's 240, 3.3's 1199 — all take
every second galaxy, and 181 is odd). Every `score_A` divided a model error
measured on clean galaxies by a benchmark charged for a corrupt one.

| best halo-only row | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| as published | 0.1047 | 0.1359 | 0.1417 | 0.1456 | 0.1744 |
| row 181 rejected | 0.1044 | 0.1120 | 0.1204 | 0.1279 | 0.1611 |

`scoreboard.py` now rejects such histories at source (`MAX_BACKWARD_DEX = 3.0`)
and stores the `rows` it kept.

**The correction reaches the amplitude metric only.** Row 181 moves the
population shape loss by **+0.043%** against 8–18% for `SIGMA_A`, because the
amplitude residual is `log10(model/truth)` (unbounded, and its was 3.76 dex)
while the shape residual is `(m−d)/d` on a profile normalized to 1 at 100 kpc.
So `L_F_REF` and every `score_F` stand, including Stage 3.3's
0.919 / 0.916 / 0.875 / 0.797 / 0.716.

**There is no generalization gap.** The NMAD of the amplitude residual is
identical between the fitted and held-out halves at every epoch
(0.1136/0.1178/0.1226/0.1310/0.1530 against 0.1156/0.1202/0.1218/0.1305/0.1507).
Seven global parameters generalize, as they should.

## The user's three questions, answered

### 1. The progenitor-selection confound — REAL, and it explains defect (b)

`selection.py` measures the selection function against the FULL TNG300
halo-structure catalog at each epoch (346k–460k centrals). **Completeness** =
`N(ours)/N(all centrals)` per bin of `log10 M200c`. The z=0.4 ceiling is
**0.781**, set by the CoG and quality flags, which are evaluated on the z=0.4
galaxy and so cost the same at every epoch. The **mh-complete cut** is where
completeness first reaches 60% of that ceiling and stays there.

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| cut, `log10 M200c` | >13.00 | >13.00 | >13.00 | >12.90 | >12.80 |
| galaxies kept | 2397 | 1780 | 1435 | 1144 | 839 |

**The z=2 halo-mass tilt is an artifact.** With theta frozen:

| tilt at z=2 [dex/dex] | 5 kpc | 10 kpc | 75 kpc | 148 kpc |
|---|---|---|---|---|
| full sample | −0.135 | −0.075 | −0.083 | −0.094 |
| mh-complete sample | −0.011 | **+0.066** | **+0.099** | **+0.089** |

It flips, not shrinks, and lands on the same positive tilt the mh-complete sample
shows at z=0.7–1.5.

**The amplitude deficit is real and LARGER**: median `log10[model/truth]` at
100 kpc goes from −0.0255 to **−0.0421 dex** at z=2.

**The mechanism is confirmed and its size is PREDICTED.** The residual depends
on future growth `G_k = log10 Mh(z=0.4) − log10 Mh(z_k)` at fixed halo mass:
partial r = +0.19/+0.22/+0.23/+0.21, **dy/dG = +0.18/+0.16/+0.15/+0.15 dex per
dex**. Feeding that and the completeness curve into a truncated-Gaussian model
predicts the tilt shift with no fitted model quantity in between:

| tilt shift at 148 kpc | observed | pred (uncorrected s_G) | pred (truncation-corrected s_G) |
|---|---|---|---|
| z=0.7 | −0.0148 | −0.0170 | −0.0443 |
| z=1.0 | −0.0331 | −0.0296 | −0.0844 |
| z=1.5 | −0.0667 | −0.0429 | −0.1364 |
| z=2.0 | −0.1830 | −0.0542 | −0.1918 |

The observation is **bracketed at every epoch**; the uncorrected estimate
matches where truncation is mildest and the corrected one matches z=2 where it
is most severe. **The size of the tilt shift needs no physics.**

### 2. The delivery-delay kernel — ARGUED AGAINST, and it has lost its target

Analytic: for a 200ρ_c halo, `V200c = 10 H(z) R200c`, so
**`t_dyn = R200c/V200c = 1/(10 H(z))` is a function of redshift alone** — the
halo mass cancels identically. `tau = eta * t_dyn` is mass-independent by
construction and cannot produce a halo-mass tilt. Worse, over our five epochs
`t_dyn/t_age` is 0.127 → 0.146, so it is a near-constant fractional shift of
the time axis, close to what `a_z ln(1+z)` already does.

**The user's counter-argument is accepted and recorded**: the "wrong sign /
worsens the inner deficit" part of the argument is a single-parameter reading
of a model that would re-converge, since the efficiency and deposit radius
refit jointly. The user also holds that the delay is genuinely tied to the
merger / dynamical-friction channel and is worth testing empirically with
`t_dyn` as a baseline scale and an **empirical halo-mass dependence** — but as
a LAST RESORT, not a priority.

**Stage 3.4 now removes its main target**: the mass-dependent tilt it was meant
to explain is the selection. What remains is the amplitude deficit, which a
mass-independent clock is least suited to.

### 3. The inner deficit — the only well-posed extension left

Untouched by any of this: the shape metric was never contaminated. Still
−21% inside 3 kpc, +5% at 20–70 kpc, total nearly right — misplaced mass. What
the earlier experiments falsified is the profile FAMILY (the shape of one
deposited component); **a second component has not been tried**, and is a
different move. It is not a resolution artifact in the direction that would
explain it away: softening makes TNG's cores less concentrated than reality.

**The leash**, so it is falsifiable rather than a spline: tie the compact
channel's fraction to `f_form` (`t_half(t_j)/t_j`, already on the halo record
and provably epoch-local) and redshift, and let the `f_form` dependence come
out null if it wants to. The compact-progenitor result gives the prior.

## The re-judged factorial (`b148761`)

45 cells, 204.7 min, both corrections. Three predictions recorded before the
run; all three held.

- **`gompertz_log-E2-S2` is still rank 1** (mean rank 12.86).
- **E1+E3 improves sharply but stops at rank 13.** The four largest upward
  moves in the factorial are all E1+E3 cells (29→13, 34→18, 37→23, 36→22), so
  "E1+E3 never reaches the top 15" was partly an artifact of the corrupted
  variable — correcting it does not make concentration competitive with the
  cross-term.
- **E2 still replaces transport; E3 still does not.** z=0.4 tilt is
  +0.003..+0.012 for every E2 cell against +0.0396 for the best E1+E3 cell.
  (Measured at z=0.4, where `selection.py` shows no progenitor bias.)
- **E4 is still the worst family** (last at 35.14, `ident` ~1.5e-07).
- **The judge still overrules the loss** (loss picks the three E5-S2+S3 cells,
  `ident` ~8e-04 against the winner's 7.8e-03).
- Every loss rose a near-uniform 0.07–0.09 — the `SIGMA_A` signature, not the
  concentration fix. A uniform shift cannot reorder a ranking, so the rank
  moves are attributable to the concentration fix.

## Stage 3.3 re-run (`2994086`)

Six fits: top three of the re-judged factorial, on the published 1199 and on
the 751 above the mh-complete cut at all five epochs.

**The parameters do NOT depend on the selected regime.** `score_A` cost of
using the fair-fitted theta on the full sample:

| model | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| `gompertz_log-E2-S2` | +0.006 | +0.018 | +0.025 | +0.024 | +0.018 |
| `moffat-E5-S2+S3` | +0.015 | +0.010 | +0.011 | +0.007 | +0.022 |
| `moffat-E2-S2` | +0.006 | +0.017 | +0.023 | +0.021 | +0.015 |

At most 2.5%. The efficiency is the stable part (`a0` +0.06, `a_M` +0.03,
`a_z` −0.06, `a_Mz` −0.06); the size law moves more (`log_f0` +0.20, `b` −0.22).

**A NEW LEAD**: in `moffat-E5-S2+S3` all three S3 conditioning slopes collapse
toward zero on the mh-complete sample — `f:logMh` 0.099→0.003, `f:dlogc`
−0.338→−0.017, `f:fform` −0.334→+0.004. **S3 may be absorbing selection
structure rather than physics.** With the factorial's finding that S3 does not
collapse the tilt the way E2 does, that is two independent reasons to distrust
the size-law conditioning terms.

**A FLAW IN THAT CHECK, recorded in `stage33_refit.py`'s docstring**: the
intersection is **assembly-biased**. At fixed z=0.4 halo mass (13.2–13.6,
n=987) it has `t50` = 4.14 Gyr against 5.59 (p = 2e-53) and `fz2` = 0.350
against 0.175 (p = 1e-83). This does not weaken the robustness answer — the
second sample differs in two ways, so transferring parameters passed a harder
test — but the `score_A` values ON the mh-complete sample describe an early-forming
massive subset and must not be quoted as a complete-sample number.

## The per-epoch mh-complete mask (`2168c84`) — DONE, and it changes the short list

`stage33_perepoch.py` repairs the intersection's assembly bias: each epoch is
scored on its own fair galaxies, **7599 galaxy-epochs against 3755**, no galaxy
required to be massive at an epoch where it is not scored. `MaskedProblem`
subclasses `fit.Problem` behind an equivalence gate — with an all-true mask it
must reproduce the parent, and does (**|Δ| = 0.00e+00** on `score_A`, 4.4e-16
on `score_F`).

**1. THE S3 COLLAPSE IS REAL.** Two mh-complete samples of different construction, one
assembly-biased and one not, both drive all three slopes to near zero:

| parameter | full | intersection | per-epoch |
|---|---|---|---|
| `f:logMh` | 0.0992 | 0.0032 | **0.0202** |
| `f:dlogc` | −0.3380 | −0.0165 | **−0.0242** |
| `f:fform` | −0.3342 | +0.0043 | **+0.0051** |

**S3 absorbs progenitor-selection structure, not physics.** Drop the size-law
conditioning.

**2. AND SO DOES E5.** On the per-epoch fit `moffat-E5-S2+S3` has a z=2 tilt of
+0.005/+0.033/−0.002/−0.015 against `gompertz_log-E2-S2`'s
−0.126/−0.076/−0.089/−0.099. E5's whole advantage was closing that tilt, and
the tilt is the survey. **E5's four extra parameters buy a fit to a selection
artifact.** The case for seven-parameter `E2-S2` is STRONGER after the control.

**3. THE MODEL IS WEAKEST WHERE THE DATA ARE BEST.** Against a regression
refitted on each epoch's own fair galaxies:

| `gompertz_log-E2-S2` score_A | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| full (progenitor-selected) | 1.131 | 1.100 | 1.059 | 1.058 | 1.050 |
| fair, same regression | 1.116 | **1.356** | **1.339** | **1.360** | **1.371** |

~36% worse at z ≥ 0.7 on a complete sample of massive haloes, against 5–10% on
the full one. The regression barely notices the restriction (scatter falls
3–15%); the model degrades sharply. **The progenitor-selected sample was
flattering the model.** The adopted model's own parameters barely move
(`a0` −0.004, `a_z` +0.012); what moves is the mass dependence (`a_M` −0.110,
`a_Mz` +0.080) — exactly the part the selection acts on.

## The R^(1/4) density view, and what it changed (`ddc15a4`, `9500a67`)

`qa_dens_*` plots `log10 Sigma(R) = dM/dA` against `R^(1/4)`, the coordinate in
which a de Vaucouleurs (n = 4) profile is a straight line. It is now part of the
project-wide `qa.evaluate()` suite, so every experiment gets it.

**It exposes what the cumulative view hides.** Outer logarithmic slope
`d log10 Sigma / d log10 R` over 50-148 kpc:

| epoch | truth | model | model - truth |
|---|---|---|---|
| z = 0.4 | -2.78 | -2.74 | +0.04 |
| z = 1.0 | -2.91 | -2.81 | +0.10 |
| **z = 2.0** | **-3.38** | **-2.90** | **+0.48** |

The truth steepens with redshift; the model barely does. Yet the z = 2 curve of
growth is only -0.024 dex low at 148 kpc -- a shallow outer slope adds mass in
the outermost annuli while the total stays low, so a CoG figure cannot see it.

**The full four-zone slope error** (model - truth; positive = model shallower):

| band | z=0.4 | z=1.0 | z=2.0 |
|---|---|---|---|
| 2-6 kpc | +0.37 | +0.43 | +0.47 |
| 6-25 kpc | -0.12 | -0.17 | +0.09 |
| 25-70 kpc | -0.13 | -0.11 | -0.13 |
| 70-148 kpc | +0.08 | +0.13 | **+0.64** |

The truth steepens monotonically and continuously (-1.80 -> -1.90 -> -2.42 ->
-2.84 at z = 0.4); the model steepens too fast through the middle and then
flattens. **This is a CURVATURE error, not a slope offset**, and
`moffat-E5-S2+S3` shows the same four zones, so it belongs to the framework.

**All the high-redshift failures are one statement: the model's z = 2 galaxies
are TOO DIFFUSE.** It makes roughly the right total mass and spreads it too far
-- 29% too light inside 3 kpc in the most massive z = 2 haloes, 0.48 dex/dex too
shallow beyond 70 kpc.

## In Progress

**Nothing is running.**

## NEXT -- the profile-shape representational ceiling (AGREED WITH THE USER)

This mirrors Stage 3.0, which computed a ceiling for the AMPLITUDE (0.0456) and
correctly killed the survival term before anyone built it. The same question has
never been asked about the SHAPE.

**The experiment.** For each galaxy the deposits form a fixed BASIS: one
truncated unit profile `F_j(<R)` per MAH step, `R50` from the fitted size law,
truncated at `3 R200c(t_j)`. The model chooses non-negative weights
`dM*_j = eps_surv * dMh_j`. Ask instead:

> over ALL non-negative weights `w_j >= 0`, respecting causality (`w_j` enters
> epoch `k` only if `t_j <= t_k`), what is the best fit to that galaxy's
> measured profile at all five epochs?

That is **non-negative least squares** -- convex, no optimiser tuning, no
multi-start, seconds per galaxy. Its residual is the SHAPE CEILING: the best any
deposition model with this basis could do given a PERFECT efficiency law.

**Why it is decisive.** It separates three culprits that currently look
identical in the loss:

| if the ceiling is | the limitation is | the fix is |
|---|---|---|
| near zero | the **efficiency law** -- the basis can make the right profile, `eps_surv` is not choosing the right weights | enrich `eps_surv`, now knowing it is worth it despite E4/E5 failing |
| large, concentrated at small R | the **basis** -- no non-negative sum of these deposits is compact enough | a compact second channel, or a redshift-dependent shape parameter |
| large at all radii | the **size law** -- deposits are the wrong size everywhere | a different functional form for `R50`, not another additive term |

**Cost**: an afternoon. NNLS over ~72 deposits x 24 radii x 5 epochs per galaxy,
a few hundred galaxies. No global fitting.

**The caution to report with it.** The ceiling gives the basis unconstrained
PER-GALAXY weights, far more freedom than a seven-parameter global law. So a
large ceiling PROVES the basis is inadequate; a small ceiling does NOT prove any
global efficiency law can reach it. Report it as a bound, never as a target.

**Contingent follow-up.** If the basis is the limit (my expectation, ~70%), test
one nested parameter before any new component: a redshift-dependent deposit
shape, `c = c0 + c_z * ln(1+z)`, so high-redshift deposits are intrinsically
more concentrated. It targets "too diffuse at high z" directly and goes through
the same identifiability check as everything else. A compact second channel is
the fallback.

## ALSO PENDING

1. **Drop `S3` from the short list**, and treat `E5` as suspect for the same
   reason. The short list should become `gompertz_log-E2-S2` and `moffat-E2-S2`
   until something else earns a place.
2. Score any new extension on the **per-epoch mh-complete mask**, not the full
   sample -- the full sample is now known to flatter.
3. The delivery-delay kernel is a last resort, and if built must carry an
   empirical halo-mass dependence (the user's condition), not a bare
   `eta * t_dyn`.
4. **Hold z = 2.0 out of the fit** and score it as a genuine extrapolation in
   redshift (`doc/todo.md`). The catch: the Stage 3.2 factorial fits only the
   two ENDPOINTS (z=0.4 and z=2.0), so a model selected there has already seen
   z=2 -- a clean hold-out must re-select as well as re-fit.
5. **Why does the model degrade on massive complete samples while the
   regression barely does?** ~36% worse at z >= 0.7. Unexplained, and it is the
   deployment case.

## Warnings carried forward (new ones first)

- **A benchmark and the thing it benchmarks must be scored on the same
  galaxies.** Assert it with a `set` comparison; it is one line.
- **Classify which code a fix invalidates from what the code READS**, as a
  predicate over the spec, not a hand-written list. Any partial re-run must
  include controls asserted NOT to move and must refuse to merge if they do.
- **Never wrap an import in a bare `except`.** Three experiments ship a
  `scoreboard.py`; import a same-named sibling BY FILE PATH.
- **An unbounded metric can be dominated by one object; a bounded one cannot.**
  Audit the unbounded ones and MEASURE the leverage on the rest.
- **`setsid` does not exist on macOS**, and a `ps` snapshot one second after
  launch does not prove a job started. Confirm by watching the LOG ADVANCE, and
  delete the log before relaunching so a stale completion marker cannot satisfy
  a wait condition.
- **Any run measured in hours must checkpoint per unit of work.**
- Rank by the judge, NEVER by loss. Truncating a profile moves its half-mass
  radius. A shuffle control must REFIT. Multi-start is mandatory.
- `np.interp` CLAMPS. A SLICE is not a PROFILE. usetex eats `%` — use
  `hongshao.qa._pct()`. Don't name a new module `run.py`.
- Never create a term without defining it in place (the user, three times).
- Always `export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof`;
  KEEP the 27 GB cache at `~/Desktop/tng300_halo_structure` — `selection.py`
  needs the RAW catalogs, not just the reduced table.

## Branch State

Branch **`exp54-unpinned-amplitude`**, clean, **37 commits ahead of master, NOT
pushed**. master @ `3b376f1`.

---
*Session: eb8b15e6 — resume with `claude --resume eb8b15e6`*
