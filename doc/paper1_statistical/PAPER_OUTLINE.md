# Paper outline: an assembly-resolved stellar-profile relation

Updated September 27 after user discussion. Forward prediction is the goal;
inverse recovery is optional supporting evidence. The five figure concepts
below are organizational suggestions, not a figure or text limit.

## Recommended scope and message

A focused TNG300 proof-of-principle paper, centered on massive central
galaxies at z=0.4. Keep the physical deposition prescription, analytic-family
search, and multi-epoch engine outside this paper. The statistical emulator
is a way to measure and express the relation, not a proposed simulation-
independent physical law.

Provisional title:

**Beyond the stellar–halo mass relation: halo assembly information in the
stellar mass profiles of massive galaxies**

More conservative alternative:

**An assembly-dependent stellar-profile–halo relation for massive central
galaxies in IllustrisTNG**

Avoid “the full MAH encoded in stellar profiles” and “the ultimate SHMR” as
claims of completeness. A defensible result can be an informative but noisy
population relation.

### The logic in one paragraph

Halo mass predicts much of a massive galaxy's stellar mass distribution,
but not all galaxies in similar halos have the same profile. First establish
that assembly history predicts part of those differences. Then determine
whether the gain is in the amount of stellar mass, the size, or additional
radial structure. Show where the surviving signal resides without imposing
an early-inner/late-outer interpretation. Finally demonstrate that a small
probabilistic profile model can retain the measured relations and their
scatter. The conclusion is an evidence-based extension of the SHMR, with
explicit limits on individual-history recovery and simulation dependence.

## Proposed section structure

### 1. Introduction: what does a single stellar mass leave out?

- Motivate extended stellar envelopes and the observational value of inner
  and outer stellar masses. Acknowledge the literature in
  [positioning](LITERATURE_POSITIONING.md), rather than claiming this motivation
  as a new discovery.
- Separate two questions: better tracing halo mass, and tracing assembly
  **at the same halo mass**.
- State the core incremental question beyond stellar mass, the additional
  size-controlled test, and the
  TNG300 scope. Close with the conditional distribution to be measured.

### 2. Data and definitions: what population and what mass profile?

- TNG300 parent selection, central definition, exact epoch, resolution,
  profile extraction, h-free masses, and physical semi-major-axis radii.
- Anchor at the native outer CoG aperture near 148 kpc. Define finite-aperture
  stellar mass rather than calling it an
  asymptotic total. Account for low-z and all-epoch quality selections.
- Define halo mass and the measured MAH endpoint; distinguish `M200c`, bound
  subhalo mass, peak mass, and fitted normalization.
- One concise sample table: each cut, remaining count, and scientific reason.
  No halo-mass completeness assertion without measured support.
- Fixed galaxy-level folds; training-only preprocessing; small set of
  predeclared comparison models and metrics.

### 3. How many profile coordinates are needed?

- Introduce amplitude plus normalized shape, then PCA as a description of
  smooth radial variation. Show held-out reconstruction, not only training
  explained variance.
- Make size visible: compare mass alone, mass plus size, and remaining shape.
- Check coarse annular masses alongside CoGs so cumulative smoothness does
  not hide poor outer-envelope reconstruction.
- Keep PCA algebra and detailed component-count sensitivity in an appendix.
- Compare the existing cubic-logit analytic CoG as a bounded alternative,
  judged by downstream profile prediction as well as reconstruction. Do not
  make the paper depend on developing a new mathematical family.

### 4. Where does the assembly information reside? — the core result

1. Compare halo-mass-only and halo-mass-plus-MAH predictions on identical
   objects with identical target representation and covariance treatment.
   Demonstrate both measured and DiffMAH histories on reconciled definitions.
2. Add stellar mass as a control to ask about profile shape; add size to ask
   about finer shape. These are information audits, not halo-only predictors.
3. Repeat the relevant increment with concentration controlled. Do not
   attribute a joint MAH-plus-concentration gain entirely to MAH.
4. Show radial patterns and one or two interpretable assembly summaries.
   Include a matched shuffled-history reference and uncertainty.
5. If run, show a limited reverse test: does shape improve recovery of an
   assembly summary beyond halo mass, stellar mass, and size? A small gain
   or a null result is reported, not hidden behind the forward score.

The central figure should decompose the evidence, rather than rank many
algorithms. It should make a mass-size-only explanation easy to see if that
is what the data support.

### 5. A compact probabilistic extension of the SHMR

- Describe `P(stellar mass, profile shape | halo mass, MAH)` and its residual
  covariance. State the number of coordinates and the full fitted complexity.
- Compare true profiles, conditional means, and correlated random draws.
  A narrow distribution of means is expected; a narrow sampled population
  or incorrect assembly-conditioned trends are not.
- Demonstrate aperture/annular masses and size–mass/inner–outer planes for
  held-out galaxies. Use a positive-mass representation if PCA draws fail
  physical consistency, rather than clipping the output silently.
- This empirical TNG relation is not the compact physical prescription
  being pursued elsewhere in HongShao. No downstream lensing, clustering,
  likelihood, or sampler is needed here.

### 6. Application, discussion, and conclusions

- If a suitable catalog is supplied, demonstrate generation of stellar
  populations from N-body MAHs with the frozen statistical mapping. Show
  input-domain checks; without stellar truth this is application, not accuracy
  validation. Keep downstream inference outside HongShao.
- What must a galaxy–halo connection retain beyond a scalar SHMR?
- How much information survives after removing size? Does a few-number
  aperture/size summary suffice, or does the full radial profile add value?
- Limits: selected centrals, finite radial range, mass definition, projection,
  resolution, measurement procedure, one hydro model, and repeated use of
  this sample during development.
- Observational implications are conditional on halo-mass uncertainty,
  stellar M/L gradients, sky/PSF effects, and sample selection. Do not claim
  an observational MAH measurement from noise-free simulation scores.
- End with two or three findings and one next step, not the experiment history.

## Five starting figure concepts, without a fixed limit

| Figure | Content | Question answered | Existing starting point |
|---|---|---|---|
| 1. Profile diversity | CoGs and coarse annular masses in narrow halo-mass bins; normalized profiles and reconstruction examples | What structure is omitted by a scalar mass? | Exp02; Exp62 visual records |
| 2. Assembly gain | Matched held-out mass-only, mass+MAH, mass+concentration, and full comparisons; conditional shuffle reference | Does assembly add predictive information beyond current halo properties? | Exp01/06/08/16 |
| 3. Information beyond amplitude and size | Radial incremental skill under successively stricter controls; same-object radial-support check | Is the signal more than stellar mass and size? | Exp62 Stage 7 |
| 4. An interpretable assembly connection | Matched early/late assembly profile residuals and predicted responses, with an optional reverse-summary diagnostic | How does assembly change the predicted profile distribution? | Exp01/06 associations |
| 5. The probabilistic relation | Measured versus mean and sampled inner–outer mass and size–mass planes; calibration inset | Does the statistical relation preserve population diversity and covariance? | Exp08/15/22; Exp37 lessons |

All figures need self-contained captions naming the sample, measurement,
control variables, held-out status, and meaning of bands. Error bars on a
mean must not be drawn as though they were galaxy-to-galaxy scatter. Plot
absolute errors alongside fractional improvements when the baseline scatter
is small. Do not reuse an old figure unchanged as a confirmatory paper figure.

Figure 4 defaults to the controlled radial assembly-association figure;
practical individual-history recovery is not claimed. Add separate figures
for representation comparisons or an N-body application if they improve
clarity. Do not constrain the science to a predetermined figure/page budget.

## Supporting material, not extra headline results

- Full standard QA: average CoGs in halo/stellar-mass bins, density and
  annular residuals, aperture/outskirt masses, cumulative residual
  distributions, mass and size planes, and best/typical/worst individual
  galaxies selected by a fixed rule.
- Sample attrition and sensitivity; measured MAH versus DiffMAH; component
  count; flexible mass baseline; radial cut and geometry; outlier influence.
- Independent epoch fits may enter an appendix without jointly describing
  evolution. Cross-epoch QA is needed only for any actual coherence claim.
  Do not quietly
  describe progenitors of selected low-z centrals as an epoch-complete sample.

## Draft abstract skeleton — deliberately without frozen numbers

> A scalar stellar–halo mass relation omits the radial distribution of a
> galaxy's stellar mass. Using projected stellar mass profiles of massive
> central galaxies in TNG300 at z=0.4, we test whether halo assembly history
> predicts profile variation beyond present halo mass. We describe stellar
> profiles with a small set of coordinates and compare nested models on
> held-out galaxies, separately controlling stellar-mass amplitude, size,
> and concentration. [Insert confirmed effect sizes and where they occur.]
> [State how much survives the size control, including a null if appropriate.]
> A probabilistic profile relation [state validated population properties].
> These results motivate an assembly-dependent extension of the SHMR while
> quantifying the limits of stellar profiles as tracers of individual halo
> histories. Transfer to observed galaxies remains to be established.

## What would change the paper's conclusion?

- **Gain beyond mass and size survives:** emphasize additional radial
  assembly information, with its limited amplitude and radial support.
- **Gain beyond mass survives but size absorbs it:** emphasize an
  assembly-dependent mass–size relation embedded in a profile model.
- **Only stellar-mass amplitude survives:** a full-profile novelty claim is
  weak; narrow the paper or obtain additional evidence before proceeding.
- **Gain disappears under matched mass definitions or selection checks:**
  stop the planned positive thesis. Publish no stronger conclusion than the
  corrected evidence supports; decide with the user whether a null/methods
  paper is worth pursuing.
