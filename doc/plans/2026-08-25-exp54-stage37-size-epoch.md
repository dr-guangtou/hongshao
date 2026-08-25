# exp54 Stage 3.7 — the size law's dependence on epoch

**Status: PLAN. Not started.** Written 2026-08-25, after Stage 3.5 (the
efficiency law's time dependence, negative) and Stage 3.6 (the objective,
negative).

## The defect this targets, stated as a number

The model's error in the stellar mass inside 2 kpc, as a median over the
completeness-controlled galaxies at each epoch, in dex — negative meaning the
model puts too little mass there:

| | z = 0.4 | z = 0.7 | z = 1.0 | z = 1.5 | z = 2.0 | span |
|---|---|---|---|---|---|---|
| the adopted model | +0.020 | −0.003 | −0.021 | −0.062 | −0.126 | **0.145** |

The model is right in the centre today and **25% too light in the centre at
redshift 2**. Two of the three levers we have are now known not to touch that
swing:

- **The efficiency law's dependence on time cannot** (Stage 3.5). Five richer
  forms and a free shared curve all land within 0.2 percentage points of the
  incumbent's profile-shape error, and the whole budget for a shared time law
  is 0.2 points.
- **The objective cannot** (Stage 3.6). Five objectives, including two that see
  the innermost aperture far better than the production one, shift this column
  by a nearly CONSTANT offset: the shift varies by 0.006 dex across five
  epochs, and the span stays 0.145–0.152 dex in every one. A constant offset is
  absorbable by the efficiency normalisation and is not the defect; the span
  is.

That leaves the third lever, which has never been enriched: **where each
deposit puts its mass, as a function of when it was deposited.**

## The current size law, and the single number that controls all of this

`model.r50_of` gives every deposit a half-mass radius

    R50 = 10^(log_f0 + b * log10(1+z_j)) * R200c(t_j)

so the deposit's size relative to its halo evolves as a **single power law in
redshift with one exponent `b` shared by every galaxy and every halo mass**.
The one previously-tested size extension (`S3`) varies the size NORMALISATION
with halo mass, concentration and formation time. **It never varies the
exponent.**

## Step 0 — the pre-check that can cancel this step

**Is the epoch swing real, or is it the sample changing?** The
completeness-controlled sample runs from 2397 galaxies at redshift 0.4 to 839
at redshift 2, and the survivors are more massive. If the central error is a
function of halo mass rather than of epoch, then the 0.145 dex "swing" is the
sample's mass composition moving and this whole stage is aimed at nothing.

Measure the central error **in fixed bins of halo mass, epoch by epoch**, and
on the fixed cohort followed at all five epochs. Proceed only if the swing
survives at fixed mass. This is lesson 22 — "the completeness trap recurs in
every new view" — applied before a headline rather than after one.

**If the swing does NOT survive**: the defect is a mass-dependent central error,
not an epoch-dependent one, and the right lever is `S3`-like conditioning or the
efficiency law's mass dependence, not anything below.

## Step 1 — the bound, before any candidate

Stage 3.5's most useful product was not its five candidates, it was `E8`: a
free shared curve that bounded them all and showed the budget was 0.2 points
before any of them were fitted. **Do the same here first.**

`S6` — give the size law's redshift dependence a FREE curve, using the same
construction as `model._pl_basis`: a piecewise-linear function of `ln(1+z)` on
six knots, with the constant and linear directions projected out analytically
so it cannot trade against `log_f0` and `b`. Four extra shared parameters. Not
a candidate model; a ceiling.

**What it answers**: how much of the 0.145 dex span a shared size law of ANY
shape can remove. If the answer is "almost none", this stage stops here and the
conclusion is that the deposition model's radial freedom is structurally unable
to follow the centre's evolution — which is a real result and points directly
at a second component.

## Step 2 — the candidates, only if Step 1 leaves room

All nested on the current law, all sharing every parameter across galaxies:

| name | what it adds | extra parameters |
|---|---|---|
| `S4` | `b -> b + b_M * (logMh - 13.5)`: the redshift exponent depends on halo mass | 1 |
| `S5` | a low-order polynomial in `log10(1+z)`, orthogonal to {1, log10(1+z)} | 1–2 |
| `S4+S5` | both | 2–3 |

`S5` is the exact analogue of Stage 3.5's `E6` moved from the efficiency law to
the size law. The orthogonalisation is not optional and is the reason `E4`
failed: an added curve that is not orthogonal to what is already there is
degenerate with it.

## How it is judged

Same discipline as Stage 3.5 and 3.6, for the same reason.

1. **The seven-criterion judge**, ranked, never the loss.
2. **The objective-independent diagnostics**, which are what this stage is
   actually about: the central error per epoch and **its span**, the error
   beyond 52 kpc, the outer surface-density slope, the halo-mass tilt, and the
   predicted growth.
3. **The identifiability report** — the binding constraint in Stage 3.5, and
   likely again here.
4. Fitted on the per-epoch completeness-controlled mask **with
   `selection.sane_history_mask` applied**, which is new as of 2026-08-25 and
   is why Stage 3.5 is being re-run.

**Success looks like**: the 0.145 dex span materially reduced — say below
0.10 dex — with the error beyond 52 kpc no worse than the incumbent's −0.001 to
−0.008 dex, the outer slope no shallower, the amplitude unchanged, and the new
parameters determined.

**Failure looks like**: the span unmoved, or moved only by a uniform offset —
which is what every objective in Stage 3.6 did and is not a repair.

**The trap**: do not read a drop in the profile-shape error as success. That
score is dominated by radii where the model is already right; the whole point
of this stage is a defect the aggregate score barely feels.

## What comes after, either way

If a shared size law cannot follow the centre's evolution, the remaining
hypothesis is a **second deposit component**, and there is now outside evidence
about its shape. A parallel experiment (exp55) finds that an **exponential core
plus an extended Moffat** reproduces measured curves of growth as well as a
three-component decomposition, and can be connected to halo growth when
constrained at a single epoch. Two cautions before importing it:

- exp55 constrains a **single epoch**; this model must serve five. Its
  conclusion cannot be carried over, only its hint about the inner component's
  shape.
- "Two components" need **not** mean every deposit carries both. The more
  natural form here, and the one that matches this stage's defect, is **an
  early exponential deposition and a late Moffat-like deposition** — a mixture
  whose weight depends on when the mass was laid down, not on radius within a
  deposit. That is a different object from the redshift-dependent deposit SHAPE
  parameter (`c_z`) that was already rejected, and it must not be conflated
  with it: `c_z` drifted one family's shape continuously, this switches between
  two families.

An early-compact, late-extended mixture is exactly the shape of the defect
above — too little central mass early, correct central mass late — so it should
be tried on the non-declining galaxies of Stage 3.4's split.

## Open questions this stage should record, not resolve

- Does the deposit truncation at three halo radii interact with a
  redshift-dependent size? `C = 3` is a convention with no scan behind it.
- Stage 3.6's objectives moved the central level by 0.1 dex without changing
  the span. Is that level a free parameter of the model (absorbed by
  `log_f0`/`a0`) or does it cost something elsewhere?
- Does `c_z`'s rejection survive on the cleaned sample and under a
  shell-based objective? It was rejected with row 181 inside the sample.
