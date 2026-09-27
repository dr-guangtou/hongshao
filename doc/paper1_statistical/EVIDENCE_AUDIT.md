# Evidence audit: what can support Paper 1?

Scope update September 27: this is an audit of historical evidence, not a
requirement to recover individual MAHs. Forward prediction is the paper's
goal. The user approved an epoch-relevant sample exception and confirmed Xu's
data/satellite-treatment lineage; references below to pending approval reflect
the September 26 audit and are superseded by [the decision note](DECISIONS_20260927.md).

Reviewed September 25–26, 2026, against the frozen checkout identified in
[README](README.md). This is a scientific planning audit, not a reproduction
of the fits. Values below are historical measurements. Selected score files,
manifests, and nine figures were copied into the private worktree, with source
hashes checked before/after copying and byte-for-byte agreement verified.
[Artifact inventory](ARTIFACTS.md) distinguishes these checks from a rerun.

## Assessment

There is enough material to design and begin writing a focused paper. There
is not yet one internally consistent, publication-ready measurement of its
strongest claim. The missing work is a controlled consolidation, not another
search for a physical profile family.

The strongest established result is that halo assembly history improves
predictions of the stellar mass distributed across apertures at fixed halo
mass. The more distinctive result—information in profile shape beyond total
stellar mass and size—has encouraging but smaller, radius-dependent evidence.
Practical recovery of an individual halo's history from a profile has not
been established by the forward-emulator results audited here.

The paper should distinguish these claims:

| Claim | Present evidence | Defensible interpretation |
|---|---|---|
| Profiles can be compressed into a few coordinates | Strong descriptive PCA evidence | A useful representation, not evidence of halo information by itself |
| Assembly adds to halo-mass-only stellar-profile prediction | Repeated positive results, with shuffled controls | Strong within-TNG evidence worth confirming on one declared sample |
| Shape adds information beyond halo mass and stellar mass | Normalized-profile results plus later explicit-control tests | Normalization alone does not prove this; the later tests are the better starting point |
| Detailed shape adds beyond mass and size | Exp62 Stage 7 finds modest, localized gains | Promising, but baseline flexibility, conditional nulls, and radial selection need stronger checks |
| Profiles accurately recover individual MAHs | Not demonstrated by the audited results | Do not claim a detailed assembly chronometer |
| A calibrated emulator reproduces the population | Good marginal/aperture demonstrations; known joint-profile defects | Validate correlated draws and derived quantities, not only mean curves |
| The relation transfers to observations or another simulation | Not established here | An implication and next step, not a result of Paper 1 |

## 1. The evidence worth retaining

### 1.1 Direct aperture evidence: Exp01, Exp04, Exp06

[Exp01](../../experiments/exp01_aperture_mah_corr/README.md) is the cleanest
opening scientific demonstration. In its archived five-fold predictions for
2,538 usable targets, adding MAH summaries reduced the RMS error of
`log M*(<10 kpc)` from 0.1433 to 0.1155 dex and of
`log M*(50–100 kpc)` from 0.2304 to 0.1829 dex. Shuffled-history errors were
0.1434 and 0.2306 dex, respectively: essentially the halo-mass-only values.
These are prediction errors against TNG measurements, not measurement errors
or the scatter of halo mass inferred from stellar mass.

[Exp04](../../experiments/exp04_conditional_model/README.md) extended this to
the whole CoG. Its reported RMS changed from 0.152 to 0.118 dex when MAH
summaries were added, while the normalized-shape RMS changed less, from 0.082
to 0.076 dex. This already suggests that amplitude carries much of the gain.

[Exp06](../../experiments/exp06_mah_pca/README.md) found essentially the same
whole-CoG performance with four MAH PCs as with hand-selected history
summaries: archived RMS 0.1173 versus 0.1177 dex, compared with 0.1521 dex
for halo mass only and 0.1522 dex after shuffling MAH PCs.

**What this contributes:** a simple, direct result that the SHMR's residuals
are structured by history. **What it does not establish:** how much
additional information comes from radial structure rather than stellar-mass
amplitude, or how much history is recoverable in the inverse direction.

### 1.2 Compression: Exp02 and Exp06

[Exp02](../../experiments/exp02_profile_pca/README.md) describes normalized
log-CoGs for 2,545 galaxies. Its manifest assigns 92.51%, 6.39%, and 0.83% of
sample shape variance to the first three PCs. This is variance explained in
the chosen cumulative, normalized representation; it is not a fraction of
halo-history information, and the descriptive PCA used the full sample.

The mode figure is a useful teaching figure, but a paper should also show
actual profiles displaced along a mode. A normalized eigenvector is not
itself a typical physical perturbation of that amplitude. Cumulative masses
are strongly correlated by construction, which helps make their PCA compact.

Exp06's MAH and stellar PCA association is suggestive: it reports partial
rank correlation 0.46 between one MAH mode and the leading CoG mode after
controlling for halo mass. This does not control for stellar mass or size.
PC labels/signs are specific to a fitted basis, not physical clocks.

**Contribution:** a compact, intelligible coordinate system for the relation.
It must serve the science rather than become the headline by itself.

### 1.3 A probabilistic relation: Exp07–Exp09, Exp14–Exp19

[Exp08](../../experiments/exp08_emulator/README.md) supplies the strongest
archived probabilistic comparison. Averaged over four stellar mass targets
(`<10`, `10–30`, `30–50`, `50–100 kpc`), its continuous ranked probability
score—CRPS, a score of the whole predictive distribution, lower is better—
was 0.11177 dex for halo mass only, 0.08506 dex with four MAH PCs, and
0.11181 dex for shuffled MAH PCs. The joint residual covariance matters;
independent aperture draws are not an adequate population model.

[Exp14](../../experiments/exp14_scatter_model/README.md) and
[Exp19](../../experiments/exp19_emulator_c200c/README.md) show why the residual
width should be allowed to depend on halo properties. Their role is to
support a distribution-valued galaxy–halo connection, not merely a best-fit
mean. [Exp15](../../experiments/exp15_outskirt_bias/README.md) illustrates
that conditional means naturally occupy a narrower range than galaxies.
This explains some residual-versus-truth trends; it does not excuse arbitrary
conditional biases or prove a noise floor.

[Exp16](../../experiments/exp16_secondary_c200c/README.md) is essential for
attribution. On its matched sample, archived four-target CRPS was 0.11280 dex
with halo mass alone, 0.10148 dex with mass and concentration, 0.08498 dex
with mass and MAH PCs, and 0.08269 dex with MAH PCs and concentration.
Thus concentration cannot simply be dismissed as redundant with MAH.
Conversely, the paper needs an explicit **MAH gain beyond mass and
concentration**, using matched inputs and model flexibility.

[Exp09](../../experiments/exp09_ceiling_check/README.md),
[Exp13](../../experiments/exp13_outskirt_limit/README.md),
[Exp17](../../experiments/exp17_c200c_nonlinear/README.md), and
[Exp18](../../experiments/exp18_secondary_more/README.md) constrain how much
complexity is worthwhile within the tested models. They do not establish an
information-theoretic ceiling or that every unexplained residual is intrinsic.

### 1.4 Full profiles: Exp22 and Exp37

[Exp22](../../experiments/exp22_full_profile_predict/README.md) demonstrates
prediction of a stellar-mass anchor plus three CoG shape PCs, with stellar
PCA learned inside each training fold. Its displayed mean per-radius CRPS
improves from 0.0715 to 0.0643 dex against its own reference model.

That reference predicts amplitude using DiffMAH plus concentration but uses
the **unconditional population-average normalized shape**. It is not a
flexible halo-mass-dependent shape model, and not a model conditioned on
measured stellar mass. The reference variance also adds amplitude and shape
variances without their covariance. Therefore, the reported improvement is
not a clean measure of MAH information beyond a conventional SHMR.

The attractive Exp22 population histogram uses **direct annular-mass draws**,
not differences of the generated PCA CoGs; the driver explicitly says this.
Its good agreement cannot certify the full-profile generator. Differencing
some PCA CoG draws produced unstable annular masses in the experiment.

[Exp37](../../experiments/exp37_multi_epoch/README.md) provides useful later
machinery for positive mass blocks and correlated multi-epoch draws. This is
a source of implementation lessons, not a reason to put five epochs and a
temporal stochastic model into the first paper.

### 1.5 The most relevant later evidence: Exp62 Stage 7

[Exp62](../../experiments/exp62_cog_fit_atlas/README.md), Stage 7, explicitly
asks whether scale-free profile diversity contains assembly information.
It uses `g(x) = log10[M*(<x R50)/M*(<148 kpc)]`, omits the fixed point at
`x=1`, and compares direct radial regression with training-fold PCA.

For 1,257 galaxies with common radial support across all five epochs, at
z=0.4 the additive-quadratic halo-feature model reduced inner-profile
residual squared error by **6.81% relative to its halo-mass-only baseline**.
With explicit stellar-mass and size controls, the corresponding reduction
was **9.62% relative to that stricter baseline**. These percentages have
different denominators; the latter does not mean that conditioning created
more absolute information. The strict model adds stellar mass and size
linearly while retaining quadratic halo mass.

This is the closest existing test of the distinctive paper claim. However:

- The effect is modest, not precise reconstruction of individual shapes.
- The primary inner interval is `0.7–1 R50`; changing the lower radius to
  `0.85 R50` also changes the sample to 1,743 galaxies. The z=0.4 inner gain
  then falls to 3.59% against that analysis's baseline. Radius and sample
  effects must be disentangled with a common-object comparison.
- The shuffled features were conditioned on halo mass, not jointly on halo
  mass, stellar mass, size, and concentration. The displayed shuffled band
  is not a calibrated null for the strict-control curve.
- The reported bootstrap resamples fixed held-out predictions. It measures
  uncertainty conditional on the trained models, not all refitting variation.
- The common five-epoch support is unnecessarily restrictive for a low-z
  paper and can select profile sizes and histories.
- Very small average residuals in the normalized CoG figure are not the
  fraction of individual diversity explained. Its models almost overlap
  in the means while explaining only a small part of individual variation.

Stage 6's amplitude-pinned profile decoding is a weaker control than Stage
7's explicit stellar-mass regression. Holding an amplitude fixed in a decoder
does not by itself remove statistical dependence on stellar mass.

**Interpretation:** enough evidence to justify a targeted confirmatory
shape-information study, not enough to promise a full radial clock.

## 2. Audit findings that affect publication

### Training/test separation and reproducibility

- Exp06 and Exp08 construct the MAH PCA basis on the full sample before
  cross-validation. This uses held-out feature distributions, although not
  held-out stellar labels. Redo all standardization, PCA, and tuning inside
  training folds; do not speculate about the size of the correction.
- Exp22 and Exp62 Stage 7 fit stellar PCA inside folds. Preserve that pattern.
- All seven copied historical manifests record `git_dirty: true`. Their
  source SHA does not uniquely identify the executed code. Archived values
  are credible historical evidence but not a fully pinned reproducibility
  package. The current committed code is what this audit inspected.
- The Exp22 figure, manifest, and score table carry differently summarized
  RMS values. Do not turn the field named `recon_rms` into a representation-
  only error without tracing the driver: the table stores halo-prediction
  RMS. Rebuild the final table from one saved set of predictions.
- More random seeds on these repeatedly explored galaxies do not create an
  untouched confirmation sample. Report the exploratory history honestly.

### Halo definitions: a potentially important confounder

`logm0_halo` in the early analyses is a latest-snapshot peak proxy, while
`logmh_z0p4` is the exact epoch mass. Own DiffMAH fits and official catalog
fits also differ in anchor epoch. Critically, Exp28 and the later lesson
correct Exp27: the official DiffMAH simulated history follows bound
**SubhaloMass**, not identically `M200c`. Agreement in the sample median
had hidden per-halo differences.

The paper must certify every source quantity, epoch, mass definition, and
little-h conversion. If a history feature supplies a second estimate or a
different definition of present mass, its prediction gain is not purely
growth information. Use the same measured endpoint in nested comparisons;
separately control for current peak/bound mass when comparing mixed sources.
Do not merge all fields labeled `M0` or `Mpeak`.

### Selection is part of the scientific question

The legacy `use` mask excludes declining histories and has a halo-mass cut.
Later fitting uses all-epoch stellar-history checks, including distance from
the stellar–halo mass relation and adjacent-epoch mass changes. These choices
may remove real unusual objects as well as broken measurements. Selection on
the outcome can change precisely the scatter and correlations of interest.

The existing rule in [SPEC](../../docs/SPEC.md) applies to every new fit.
This plan does not override it. A low-z paper sample or a refitted permissive
sample needs explicit approval. Frozen-model reporting on flagged subsets is
useful but does not alone demonstrate the absence of fitting-selection bias.
Do not claim the older and newer samples are interchangeable.

### What the profiles actually measure

- CoGs use elliptical apertures with a constant adopted shape; radius is
  semi-major axis, not automatically circularized radius.
- A CoG and its complete vector of annular differences plus central mass
  contain the same discretized mass information. Their different predictive
  scores need not imply different intrinsic information.
- The old `density_from_cog` uses circular area `pi * delta R^2` and floors
  nonpositive differences. It is not the measured isophote-density product.
- `sigma_shared` is a separate along-isophote measurement. The current data
  note attributes its difference from aperture-derived density to clipping.
  That difference alone does not identify the clipped material as satellites
  or ICL; do not assert that smooth ICL is removed. Xu et al.'s published
  map pipeline removes bound satellite particles before photometry. Whether
  this exact drop follows that pipeline needs author/source confirmation.
- An unmeasured, zero-filled density bin is not evidence of zero physical
  density. No extrapolated outer radius or forced positivity of the truth
  may silently manufacture an information signal.
- Nonoverlapping annuli can still have correlated errors from sky, geometry,
  normalization, and orientation. A diagonal annular covariance is an
  assumption, not a general identity. A near-unity reduced chi-square from
  a flexible profile fit is a consistency check, not proof of a complete
  measurement-error model.

### Interpretations to retire or qualify

1. **“Normalizing by stellar mass controls stellar mass.”** It removes the
   coordinate's amplitude, not all dependence of shape on stellar mass.
2. **“PCA is non-portable.”** A frozen basis can project new input histories
   on a compatible grid. Its calibration may fail under a population or
   simulation change; the same domain issue applies to DiffMAH regressions.
3. **“The residual is the intrinsic ceiling.”** Failure of several regressors
   to improve it is not an information bound. Later projection measurements
   account for only a small part of population variance in cumulative masses.
4. **“Density has more information than CoG.”** Different losses, compression,
   and/or measurements are being compared. An invertible change of variables
   cannot increase the underlying information.
5. **“An early inner / late outer clock is proven.”** MAH epochs are strongly
   correlated; PCA signs are arbitrary; stellar formation age differs from
   halo assembly time. Partial associations are not deposition histories.
6. **“A TNG-calibrated emulator is a general physical forward prescription.”**
   It is an empirical conditional distribution. This paper can publish that
   scientific relation without reversing the Exp75/Exp77 production decision.

## 3. What stays outside this paper

The symbolic-profile searches, damped-cosine reparameterizations, deposition
kernel development, and Exp79 convergence history are not necessary evidence
for this thesis. Exp75/Exp77 contribute QA lessons (especially annular masses
and the danger of a good pooled CoG score) but their learned corrections need
not appear as paper models. Exp74 contributes input/anchor lessons, not a
requirement to revisit the physical engine. Full halo histories remain valid
inputs under the user's prediction contract; a low-z paper's optional
history cutoff is a scientific scope choice, not a new prohibition.

## 4. Readiness decision

**Start the manuscript structure now; freeze the headline numbers later.**
The paper becomes ready when one reproducible matched analysis confirms the
increment beyond a flexible halo-mass reference, separates amplitude, size,
and finer shape, survives the declared null/selection checks, and passes
direct profile and population QA. If finer shape adds little, the honest
paper is an assembly-dependent mass-and-size/profile relation—not a claim
that detailed MAHs can be recovered from outskirts.
