# exp73 plan — score the galaxy where the galaxy is

**Status: PROPOSED, awaiting the user's decision. Nothing is built.**
Raised by the user on 2026-09-01; the evidence below is measured, not assumed
(`experiments/exp72_error_objective/radial_evidence.py`,
`outputs/radial_evidence.{npz,log}`).

## The user's two claims, and what the measurement says

> **1.** The same physical radial bins are used at all five epochs. At high
> redshift the outer bins hold very little mass, in the simulation as in
> observation, so scoring them equally over-weights a part of the galaxy that
> barely exists. **2.** Every residual here is relative, `(model − data)/data`.
> A z = 2 model 50 per cent wrong at 100–150 kpc sounds bad, but if almost no
> mass is there then almost no mass has been misplaced.

**Both are confirmed, and the size of the effect is larger than the argument
suggested.**

### The galaxy shrinks by a factor 3.9; the grid does not move

| median half-mass radius R50 | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| kpc | 11.6 | 9.1 | 7.1 | 4.4 | **3.0** |
| so 148 kpc is | 13 R50 | 16 R50 | 21 R50 | 34 R50 | **49 R50** |

The outermost bin is scoring the mid-outskirts of a z = 0.4 galaxy against the
far envelope of a z = 2 one, under one shared law.

### Where the stellar mass actually is

Median fraction of `M*(<148 kpc)` in each shell:

| epoch | 2–5 | 5–10 | 10–23 | 23–52 | 52–103 | 103–148 kpc |
|---|---|---|---|---|---|---|
| 0.4 | 17.9% | 16.2% | 19.5% | 18.3% | 10.6% | 3.8% |
| 1.0 | 22.3% | 16.9% | 17.2% | 13.4% | 7.0% | 2.2% |
| **2.0** | **27.2%** | 14.4% | 9.6% | 5.8% | **2.6%** | **0.7%** |

At z = 2, **96.7 per cent of the galaxy is already inside 52 kpc.**

### What the objective charges, against where the mass is

Share of the production objective's loss, against the shell's share of the mass:

| z = 2.0 | 2–5 | 5–10 | 10–23 | 23–52 | 52–103 | 103–148 kpc |
|---|---|---|---|---|---|---|
| share of the loss | 17.7% | 15.7% | 16.8% | 18.6% | 17.5% | 10.3% |
| share of the mass | **27.2%** | 14.4% | 9.6% | 5.8% | **2.6%** | **0.7%** |

**At z = 2 the production objective spends 27.8 per cent of its attention on the
outer two shells, which together hold 3.3 per cent of the stellar mass — an
eight-fold over-weighting — while the 2–5 kpc shell holds 27.2 per cent of the
mass and gets 17.7 per cent of the attention.** Claim 1 is confirmed with a
number.

### The relative residual amplifies exactly there

The chi-square refit at z = 2, per shell — relative error beside the mass it
actually misplaces, as a fraction of the whole galaxy:

| shell | relative error | mass misplaced |
|---|---|---|
| 10–23 kpc | +32.2% | **+3.00%** |
| 23–52 kpc | +37.6% | +2.14% |
| 52–103 kpc | +42.8% | +1.08% |
| **103–148 kpc** | **+81.1%** | **+0.53%** |

The shell with the *smallest* relative error misplaces **six times more mass**
than the shell with the largest. Claim 2 is confirmed. And the measurement runs
out there too: at z = 2, **3.9 per cent** of galaxies have no positive measured
mass at all in the outermost shell, so a relative residual is undefined for them
while a fit still charges the model.

### In R50 units the galaxy is very nearly self-similar

Median fraction of the galaxy's mass in each shell, shells defined by the
galaxy's own half-mass radius:

| epoch | 0.25–0.5 | 0.5–1 | 1–2 | 2–4 | 4–8 R50 |
|---|---|---|---|---|---|
| 0.4 | 14.3% | 17.1% | 18.1% | 16.0% | 11.0% |
| 1.0 | 14.6% | 17.6% | 17.6% | 14.6% | 10.1% |
| 2.0 | 13.7% | 18.7% | 19.9% | 12.9% | 7.4% |

| spread across epochs, percentage points | worst column |
|---|---|
| fixed kpc shells | **12.4** |
| R50 shells | **3.7** |

**In units of its own half-mass radius, a massive galaxy's mass distribution
barely changes from z = 2 to z = 0.4.** Almost all of the evolution is in R50
itself (a factor 3.9) and in the total mass. That is a structural fact about the
data, and it reframes what a shared five-epoch law should be asked to do.

## The finding that decides the plan

The mass–size relation, model against TNG300, median R50:

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| TNG300 [kpc] | 11.60 | 9.13 | 7.10 | 4.38 | 3.04 |
| exp63 joint | 11.82 | 9.02 | 6.78 | 4.43 | 3.17 |
| exp63 error | **+0%** | +1% | −1% | +3% | **+5%** |
| chi2 refit | 10.80 | 8.91 | 7.26 | 5.17 | 3.78 |
| chi2 error | −2% | +4% | +9% | +22% | **+27%** |

**The incumbent objective already delivers the mass–size relation to 5 per cent
at every epoch.** The error-weighted chi-square breaks it, making z = 2 galaxies
**27 per cent too big** — which is the physical statement of everything else it
did wrong (too little mass at 2 kpc, too much at 100 kpc). Stated in the user's
own terms: the chi-square traded away the thing that matters most at high
redshift to buy the z = 0.4 centre.

So exp73 is **not** "re-weight the chi-square". The chi-square's weighting is
downstream of a coordinate choice, and the coordinate is the thing to change.

## The proposal

**One question: can a shared five-epoch law hold the mass–size relation AND the
inner profile, if it is scored where the galaxy is rather than at fixed kpc?**

### Stage 1 — evaluation only, no fit (hours)

Following the method that worked in exp72 Step 1: sweep by evaluation, and let
the sweep disqualify candidates before any fit is spent.

Build three objective ingredients and measure what each charges, at frozen
theta, for the models already in hand (exp63 joint, chi2 refit, nested
incumbent, and the single-epoch z = 0.4 fit):

1. **A size-relative radial coordinate.** Shells in units of each galaxy's own
   measured R50. The truth's R50 is used, not the model's, so the coordinate is
   fixed data and not a moving target the fit can game — that has to be
   asserted, because a model that scores itself on its own size scale can
   satisfy the objective by shrinking.
2. **A mass-weighted residual.** The error in a shell divided by the galaxy's
   TOTAL mass rather than by the shell's own mass, so a shell is charged in
   proportion to the mass it actually holds. Section 3's two columns are exactly
   this contrast.
3. **A coverage gate.** Shells where the measurement has run out (3.9 per cent
   of galaxies at z = 2 in the outermost) excluded rather than charged.

Report for each combination: the concentration statistic (the exp72 Step 1
usability gate), the share of the loss against the share of the mass, and the
amplitude/shape/size sensitivity probes — with a **new probe for size**, since
R50 is now a first-class quantity.

**The gate this stage must pass before any fit:** an objective that charges the
outer two shells at z = 2 more than about twice their mass share is not fixing
the problem, and one whose concentration is worse than the production
objective's is disqualified outright.

### Stage 2 — one refit, only if Stage 1 clears something (about 2.5 h)

The same twelve parameters, bounds, starts, sample and engine, so the objective
is again the only change. Judge on the exp72 transfer matrix, the standard
battery, **the mass–size relation explicitly**, and the exp71 tilt diagnostics.

### What would make this worth doing, and what would sink it

**Worth doing if** Stage 1 shows a size-relative, mass-weighted objective
allocating attention roughly in proportion to mass at every epoch, while still
being sensitive to the inner profile.

**Sinks it if** the evidence above turns out to be reachable more simply. Two
cheap checks belong in Stage 1, before anything else:

- The exp63 objective **already** gets R50 to 5 per cent. If it is not broken,
  the coordinate change may buy nothing measurable.
- The z = 0.4 centre — the one thing the chi-square improved — is already solved
  by fitting a single epoch (−0.1 per cent inside 2 kpc). If the real limit is
  the shared law rather than the coordinate ([[shared-law-is-a-close-compromise]],
  and exp63 Stage 5i's per-epoch gate rms 1.2–1.4 against 3.3 joint), then a
  better coordinate will not raise the ceiling either, and the honest conclusion
  is that hongshao needs a per-epoch or time-interpolated law, not a new
  objective.

**The second bullet is the real risk and it should be tested first.** Everything
exp72 measured points the same way: the objective has been re-weighted twice now
and each time it moved the compromise without raising the ceiling.

## The user's decisions (2026-09-01)

### D1 — the fitting target does NOT change; the SIZES become a validation target

> *"I still think the CoGs themselves should be the main fitting target, but
> mass-size relation or the difference of the R20, R50, R80 fractional sizes'
> distributions at fixed stellar mass (or halo mass) should be a major target
> during the model validation phase and the QA process."*

exp73 keeps `cog_provided` as the thing minimised. The coordinate work is about
WHERE the curve of growth is scored, not about scoring something else.

What changes is QA. `hongshao/qa.py` already computes sizes at
`SIZE_FRACTIONS = (0.5, 0.8, 0.9)` and plots the mass-size plane (tier 2d).
Three concrete additions are owed:

1. **Add R20.** The user named R20/R50/R80; the module has R50/R80/R90. R20
   probes the inner region the central defect lives in, which is exactly where
   the objectives disagree.
2. **Score the DISTRIBUTION, not just the median.** The requirement is the
   *difference of the fractional sizes' distributions at fixed stellar mass (or
   halo mass)*, so at fixed mass compare the model's and the truth's SPREAD of
   R20/R50/R80, not their medians alone. This bears directly on
   [[fitting-harder-narrows-the-population]]: harder fitting narrows the
   predicted population, and a distribution comparison is what detects it.
3. **Promote it to a first-class tier** with a numbered gate and a threshold,
   reported for every product, rather than a figure that has to be read.

The measurement that motivates this is already in hand: the incumbent objective
holds R50 to 5 per cent at every epoch while the error-weighted chi-square makes
z = 2 galaxies 27 per cent too big — a difference invisible in the loss and
obvious in one size number.

### D2 — test both down-weighting schemes against each other

> *"Honestly, I don't know which is better. Can you devise a plan to test both
> and let the results speak for themselves?"*

Both enter Stage 1's sweep as siblings, on identical ground, with the production
objective as the common reference:

| scheme | definition | free parameters | what it risks |
|---|---|---|---|
| **A, smooth** | each shell's residual weighted by that shell's median mass fraction at that epoch | **none** — measured from the data | may leave a long tail of near-zero weights contributing noise |
| **B, gated** | shells whose median mass fraction falls below a threshold `t` are dropped outright | **one**, `t` | a threshold is a cliff; the result may hinge on where it sits |

Judged on four things, all already built in exp72:

1. **Allocation** — share of the loss against share of the mass, per shell per
   epoch. The target is proportionality; the production objective is off by 8x
   at z = 2.
2. **Usability** — the participation ratio. Neither scheme may concentrate the
   loss on fewer galaxies than the production objective does.
3. **Sensitivity probes** — amplitude, shape, and a NEW size probe (perturb R50
   at fixed total mass), so the two are compared on what they charge for the
   failure modes that matter.
4. **Threshold sensitivity for B** — sweep `t` over 0.5, 1, 2 and 4 per cent.
   **If B's verdict moves materially with `t`, B is disqualified in favour of A**,
   because a conclusion resting on an arbitrary cliff is not a conclusion. That
   rule is fixed here, before any number is seen.

Only the winner is fitted, and only if it also clears the Stage 1 gate.

### D3 — the shared law: KEEP it, and stop treating z = 2 per-epoch as a target

**Recommendation, from measurements already in the record.**

The gap to close was joint 3.33 / 2.36 / 3.57 / 3.28 / 6.88 against per-epoch
1.42 / 1.17 / 1.37 / 1.23 / 2.86 (exp63 Stage 5i, gate rms). Three things say
the z = 2 half of that gap is not a legitimate target:

- **exp63's own reading of its per-epoch sequence** (Stage 5i point 3): the
  parameter sequence BREAKS at z = 2 — the split moves *up* in mass instead of
  flattening, `n_c` jumps 1.25 → 2.1, `c_e` 1.2 → 1.6 — and exp63 concluded "the
  z = 2 fit is describing a different population, and the time law was partly a
  **mass-selection law**".
- **exp71 (C18) then proved that reading right**: the z = 2 halo-mass dependence
  IS the z = 0.4 progenitor selection, predicted quantitatively from the
  completeness curve with no free parameter.
- **exp61**: single-epoch fits erase the z = 2 bias on the mask they are fitted
  on and then OVERSHOOT the population to +12.4 per cent at 100 kpc
  ([[per-epoch-ceiling-does-not-transfer]]), and fitting harder narrows the
  predicted population ([[fitting-harder-narrows-the-population]]).

So chasing the z = 2 per-epoch number means fitting a selection artefact and
narrowing the population while doing it. **Drop it as a target and report it as
scope**, which is what C18 concluded independently.

**But the low-redshift half of the gap is real and unexplained.** At z = 0.4 the
completeness is at its plateau, so no selection acts, and the joint fit is still
2.3x worse than the single-epoch fit (3.33 against 1.42). That is the shared law
genuinely costing something, and it has NOT been explained away.

**What is untested there.** `D2` in the register closed "a richer time dependence
for the EFFICIENCY law" — but that was the pre-exp63 seven-parameter incumbent,
with no compact channel and no size-law time dependence. exp63 gave the sizes
their own time law, one power of (1+z) each (`b_c`, `b_e`), and **nobody has
asked whether a richer time law for the SIZES closes the z <= 1 gap.** Below
z = 2 exp63 found the per-epoch size parameters moving smoothly, which is exactly
the condition under which a better time parametrisation should be able to reach
them.

**So the recommendation is:** keep one shared theta; drop the z = 2 per-epoch
ceiling as a target; and make "a richer time law for the two size parameters,
judged at z <= 1.0 only" the next model question after the coordinate work — it
is cheap, it is nested (both reduce to the current form), and it is the one part
of the joint-versus-per-epoch gap that neither selection nor overfitting
explains.
