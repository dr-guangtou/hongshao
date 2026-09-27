# ushmr1 execution plan

Repository name: **`ushmr1` — The Ultimate Stellar-Halo Mass Relation: I**.
Approved direction, September 28, 2026. This is the next-session execution
contract, not a record of analyses already run.

## 1. Scientific question and argument

Given a massive central halo, does its assembly history improve prediction
of the galaxy's projected stellar mass distribution beyond what halo mass
alone supplies? How much of that gain remains after accounting for stellar
mass, and then stellar mass plus size?

The argument should follow this order:

1. Define the supplied massive-central population and the finite-aperture
   1-D measurement; show the diversity that a scalar stellar mass omits.
2. Establish that MAH adds held-out predictive information to a flexible
   halo-mass reference, with concentration tested separately.
3. Separate gain in stellar-mass amplitude, size, and remaining radial shape.
   These stellar-controlled tests are information audits, not the forward model.
4. Locate the robust radial association with simple observables and measured
   assembly summaries; do not impose an early-inner/late-outer narrative.
5. Show that correlated forward draws reproduce the relevant population
   distributions and scatter, not just individual conditional means.
6. State what is intrinsic to TNG300 and what remains untested observationally
   or in an N-body application. Report a weak or null fine-shape result honestly.

This uses existing methods: direct radial/annular regression, training-only
PCA, DiffMAH, a low-complexity mean/scatter emulator, and one established
analytic profile alternative. No new statistical framework, broad algorithm
competition, symbolic search, or physical-model development is required.
The new work is a matched, leakage-controlled analysis and evidence package.

## 2. Scope and authority

- Main epoch: z=0.4 in TNG300; all qualifying centrals. Preserve the delivered
  parent IDs while certifying peak selection. No star-formation/morphology/
  stellar-mass cuts; no unrelated all-epoch stellar-history exclusions.
- Stellar anchor: the native outer CoG aperture, listed in the source note as
  148.22006438 kpc. Verify the array's exact value before use. It is a finite
  mass, not an extrapolated infinite total. Use physical semi-major-axis radii.
- Main measurements: CoG-derived masses and profiles. Isophotal density is
  complementary, not assumed identical to differentiated aperture mass.
- Compare measured MAH and DiffMAH on the same mass definition, time window,
  objects, endpoints, target representation, folds, and scatter prescription.
  Full halo history is legitimate input; measured galaxy stellar properties
  are forbidden prediction inputs but allowed in explicitly labeled audits.
- Work on analysis, summaries, visualizations, and Markdown notes. No LaTeX
  manuscript yet, except figure captions. No publication figure/page limit.
- Independent-epoch appendix, reverse summary diagnostic, realistic mock
  observation tests, and N-body generation are optional follow-ups. None
  may delay the core z=0.4 evidence package or trigger a new data query without
  an explicit decision. No lensing/clustering/inference machinery.
- Next session may create the approved sibling repository. This closeout
  creates neither it nor a remote. Never publish it by default.

## 3. Workspace and preserved evidence

Destination: `/Users/shuang/Dropbox/work/project/massive/ushmr1`.
If it already exists, inspect ownership and contents and stop before
overwriting. Initialize an independent local Git repository; use a feature
branch for implementation. Do not use a HongShao-linked worktree for the
paper repository, and do not change Claude's checkout to create it.

Proposed structure (create only when needed):

```text
ushmr1/
  README.md                     # purpose, status, reproduction commands
  AGENTS.md                     # scientific, isolation, and figure rules
  pyproject.toml
  uv.lock
  config/
    sample.yaml                 # exact definitions and exclusions
    analysis.yaml               # nested models, grids, folds, seeds, scores
    figures.yaml                # dimensions, typography, labels, colors
  notes/
    paper_outline.md
    decisions.md                # dated amendments, before inspecting results
    data_dictionary.md
    sample_provenance.md
    methods.md
    claims.md                   # claim -> evidence -> limitations
    results_review.md
    run_status.md               # stage, artifact IDs, next action, blocker
    writing_handoff.md
    references/                 # curated literature/source notes
  src/ushmr1/
    sample.py
    representations.py
    prediction.py
    evaluation.py
    plotting.py
  scripts/
    prepare_sample.py
    run_analysis.py
    prepare_figure_data.py
  data/
    README.md                   # external source paths and permissions
    manifests/
    processed/                  # private frozen inputs; not casually committed
  results/
    runs/<run_id>/               # configs, models, predictions, tables, logs
    manifests/
  figures/
    fig01/
      fig01.py
      fig01.tex
      fig01_data.npz
      fig01_manifest.json
      fig01.pdf
      fig01.png
    qa/<diagnostic_name>/        # same reproducibility/caption contract
  tests/                        # small focused scientific-contract checks
```

There is deliberately no manuscript tree yet. Names above describe future
files, not commands or implementations that already exist.

Transfer the curated planning notes, source citations, and checksum inventory.
Retain necessary controls and negative evidence, not just favorable results.
Preserve the current private archive before any planning-worktree deletion:

`hongshao_codex_paper1_plan_20260925/data/processed/paper1_audit_snapshot_20260925/`.

It contains 22 historical artifacts inventoried in `ARTIFACTS.md`. Verify
hashes on transfer; label them historical, not newly reproduced paper results.
Source manifests record dirty trees, so their old SHA alone is insufficient
to reproduce those fits.

Reuse mature HongShao code from a recorded immutable revision, either through
a pinned dependency or a minimal attributed copy with focused tests. Never
import a mutable checkout or execute another agent's experiment driver.
Keep the paper's environment, caches, outputs, and snapshots private. Use
`uv` exclusively; do not alter any existing HongShao environment.

Track code, small configurations, notes, captions, and manifests in Git.
Small redistributable figure data may be tracked; large inputs/results need
an explicit private backup and restore path, not merely `.gitignore`.
Remote creation, data redistribution, and release are separate decisions.

## 4. Frozen comparison contract

Use the definitions and equations in `ANALYSIS_PLAN.md`. In brief, `H` is
the certified halo-mass control, `A` assembly shape, `C` concentration,
`S = log Mstar(<R_ap)`, `q = log CoG - S`, and `r = log R50` for the same
finite-aperture mass. When history and halo-control mass definitions differ,
include the history's endpoint in BOTH members of each nested pair.

| Test | Reference | Added information | Output |
|---|---|---|---|
| Forward | H | A | Complete stellar profile, amplitude included |
| Forward with concentration | H, C | A | Same profile on a matched sample |
| Beyond stellar mass | H, S | A | Normalized fixed-kpc shape |
| Beyond stellar mass and size | H, S, r | A | Remaining shape |
| Attribution with concentration | H, S, r, C | A | Remaining shape |

Use the same reference terms and scatter flexibility in each pair. Primary
starting representation: four MAH PCs, stellar amplitude plus three shape
PCs, and a direct coarse-annulus reference. These are declared historical
starting choices, not performance claims. All preprocessing belongs inside
training folds. Include concentration-available sample attrition separately;
do not reduce the entire paper sample merely because C is missing.

Freeze before fitting:

- Exact halo/history definitions, endpoint handling, usable common time grid,
  raw versus running-peak roles, missing-history policy, and DiffMAH fitting
  constraints. Reconstruction failures remain visible; they are not silent
  sample exclusions. Sampling DiffMAH curves through the same history basis
  separates fitted-curve smoothing from parameter-coordinate effects.
- Exact radial support, masks, interpolation, central mass, coarse edges,
  and finite-aperture R50 calculation. Do not interpolate outside support.
- Five galaxy-grouped outer folds and training-only tuning. Keep all views
  and epochs of each halo together. Store explicit fold membership.
- One simple nonlinear mass reference and one bounded flexibility sensitivity;
  use the same permitted reference terms for the assembly-added model.
  Specify regularization grids before scores; no broad tuning campaign.
- Equal galaxy weighting and fixed log-radius weighting; paired squared-error
  reduction plus absolute RMS in dex, annular errors, marginal CRPS, interval
  coverage, and joint-population diagnostics. No likelihood comparison in
  singular truncated-PC coordinates.
- Primary global radial statistic and uncertainty procedure; radial searches
  get simultaneous uncertainty, not selected-radius nominal significance.
- Whole-history conditional null for each control set, support diagnostics,
  synthetic null validation, and a residual-based sensitivity. Do not
  randomize epochs independently or call approximate matching an exact test.

Remaining data-dependent settings must be resolved from input support and
small mechanics checks, not by maximizing the scientific gain. Save the
completed configuration and a dated decision note before the first science run.

## 5. Staged execution and decisions

| Stage | Work | Durable output | Gate for continuing |
|---|---|---|---|
| 0. Establish isolated home | Check ownership; create local repo/environment; transfer notes and verified evidence | README, rules, dependency lock, source inventory | No shared writable paths or unexplained existing destination |
| 1. Certify data and selection | Trace query; reconcile IDs, mass definitions, units, endpoints, measurement masks | Data dictionary, sample/attrition tables, source hashes, selection certificate | No unresolved ambiguity affecting membership or mass controls |
| 2. Freeze and check protocol | Synthetic/null checks, folds, representations, nested feature contract | Frozen configs, tests, stage report | Contracts pass and no stellar leakage into forward inputs |
| 3. Operational pilot | One end-to-end small fit, prediction, null calculation, and captioned QA figure | Timings, memory, pilot outputs clearly labeled non-scientific | Complete in under a minute on reduced operational scope; inspect QA |
| 4. Matched forward evidence | Measured/DiffMAH, mass-only and concentration-controlled, common targets | Per-galaxy out-of-fold predictions/losses; aggregate and radial tables | Inspect mean CoGs and stellar-mass planes before a promising verdict |
| 5. Information separation | Stellar-mass, mass+size, and concentration controls; matched assembly trends | Shape-gain tables, controlled-profile panels, overlap diagnostics | Robustness/null checks determine strength of claim, not whether to keep result |
| 6. Robustness and uncertainty | Conditional nulls, influence, old strict-mask sensitivity, radial support, refit/split sensitivity | Uncertainty bands, influence list, all declared negative results | Failed null calibration or support overlap blocks a significance claim |
| 7. Representation and population | Correlated forward draws; bounded analytic alternative after stability check | Standard QA, physical-validity rates, model complexity table | No silent clipping, measured-amplitude pinning, or unexplained scatter collapse |
| 8. Freeze figure/evidence release | Build self-contained figure packs; rerun plotting; inspect output | Final figure index, captions, claim ledger, writing handoff | Every claim links to an artifact and every figure reproduces from saved inputs |

Stages 4 and 5 produce preliminary diagnostics; no headline numbers are
final until stages 6 and 7 are reviewed. Stage 7's cubic-logit comparison is
bounded: if compression, coordinate stability, or halo predictability fails,
record the negative result and stop that branch. Do not search another family.

Before the pilot, write checks for ID joins, units, group separation,
training-only PCA/scalers, finite-anchor normalization, cumulative/annular
round trips, reference-model nesting, and no measured stellar input to
forward predictions. Perturb held-out stellar labels and verify trained
forward parameters and that object's prediction do not change. Test null
calibration with correlated synthetic controls; one successful shuffle is
not sufficient validation of the null procedure.

Measure larger-run cost after the sub-minute gate. Choose resampling counts
from measured cost before looking at significance; the detailed plan's 999
randomizations is a target, not permission to exceed resources or skip checks.
Record attainable tail resolution. Never change the science sample to satisfy
the operational time gate. Start with one worker; respect other agents' CPU,
memory, and disk use. No runtime or sample-count estimates are certified yet.

## 6. Restart, automatic progression, and stop rules

Use a small stage dispatcher or explicit scripts, not a new workflow framework.
The future entry point can be `uv run python scripts/run_analysis.py --stage
<name> --config config/analysis.yaml`; this is a proposed interface only.
Each stage validates inputs, writes to a new run directory, completes its
manifest, and then marks itself complete. A half-written result is not complete.

`notes/run_status.md` and a machine-readable run manifest record: stage,
configuration hash, source/input hashes, exact code revision and dirty state,
environment, folds, seeds, start/end times, failures, dependencies, output
hashes, verification status, and the next action. Reuse outputs only if this
identity matches; changed inputs invalidate downstream stages. Never overwrite
an accepted result set with a partially completed retry.

The agent may automatically advance through the declared stages after their
checks pass, including inspecting and summarizing QA. It must surface figures
for the user's review and retain that review as a distinct pending/completed
item. Automatic checks do not replace visual inspection or scientific judgment.

Pause and ask for direction on:

- ambiguous parent selection or mass/ID mapping that affects the comparison;
- required external data, publication/remote permissions, or a nontrivial new
  observational-error assumption;
- a change to the scientific target, selection, control set, or model family;
- corrupted inputs, repeated numerical failures, unavailable LaTeX rendering,
  unsafe resource contention, or a figure that cannot reproduce;
- a result whose explanation requires more than the bounded checks.

A null or modest result is NOT a reason to pause, tune until positive, or
drop the comparison. Complete the declared checks and narrow the conclusion.
Stop when the core evidence package is complete. Optional extensions need
a separate decision; accurate individual-MAH recovery is never a gate.

## 7. Figure contract and logical sequence

Every new scientific figure, including QA, receives:

1. A standalone script such as `figures/fig01/fig01.py` that resolves data
   relative to its own location and outputs PNG and vector PDF. No fit,
   network request, mutable worktree dependency, notebook state, or manual edit.
2. A saved compact data bundle containing the plotted quantities, uncertainties,
   model/sample identifiers, selected object IDs, and fixed random draws where
   relevant. The full pipeline that creates this bundle is documented separately.
3. A matching `.tex` caption stating quantities, sample, controls, held-out
   status, meaning of lines/bands, interpretation, and important limitations.
   Keep captions human-editable; plotting reruns do not overwrite them.
4. A manifest with upstream run ID, data/code/configuration hashes, environment,
   and rendering commands. Avoid circular self-hashes. Scientific reproduction
   means identical plotted values/layout within declared tolerances; timestamp
   metadata can prevent byte-identical PDF hashes.
5. Publication-size visual review of both formats: typography, clipping,
   legibility, panel labels, units, color accessibility, and correct uncertainty
   interpretation. All text, including ticks, legends, annotations and math,
   uses the same actual LaTeX font/preamble. Verify LaTeX availability before
   expensive work; no silent fallback to the HongShao sans-serif default.

Use a consistent colorblind-safe palette and dimensional conventions. Keep
galaxy scatter distinct from uncertainty in means and distinguish conditional
means from stochastic population draws. Display every new figure in the
generation turn, give its full PNG path, note its PDF companion, and explain
it in one sentence. Read the plotting/visualization skills before implementation.

| Initial figure | Evidence required | Main interpretive check |
|---|---|---|
| 1. Sample and profile diversity | Certified selection; observed CoGs/annuli; held-out compression examples | What is lost by reducing a galaxy to one stellar mass? |
| 2. Matched forward gain | Mass-only, MAH, concentration, and joint predictions, measured and DiffMAH | Does assembly add beyond a fair halo-mass reference? |
| 3. Beyond amplitude and size | Same-object conditional gains, absolute errors, null/uncertainty bands | Is the signal more than mass or size? |
| 4. Radial assembly association | Matched control distributions, profile residuals, support/count panels | Where is a simple, interpretable assembly association supported? |
| 5. Population relation | True profiles, means and draws; inner/outer and size/mass planes; coverage | Does the model preserve covariance and conditional scatter? |

These are logical roles, not a fixed count. Split crowded panels. Give a
representation comparison its own figure if warranted. Full standard QA also
includes mean CoGs in halo/stellar-mass bins, measured and derived density
clearly distinguished, aperture/annular/outskirt residuals, residual
distributions, and fixed-rule best/typical/worst individuals. If a model only
predicts coarse annuli, do not manufacture an isophotal-density prediction.

## 8. Evidence notes and completion criterion

For each candidate claim in `notes/claims.md`, record:

- exact plain-language statement and whether supported, weak, null, or blocked;
- matched reference and sample definition;
- effect size, absolute scale/units, uncertainty, and all relevant caveats;
- table row, figure panel, out-of-fold prediction file, run ID, and source hash;
- sensitivity that most challenges it, and whether that changes the conclusion.

`notes/results_review.md` answers motivation, method, findings, problems, and
implication without bare numbers or unexplained experiment labels. Keep a
figure index with captions and direct reproduction commands. Preserve the
historical audit separately from new results so a writing agent cannot mistake
an archived exploratory number for the new matched result.

The analysis phase is complete when the core comparisons and bounded checks
are finished, numerical claims trace to verified outputs, population QA is
reviewed, and all proposed publication figures reproduce with captions.
`notes/writing_handoff.md` then provides the argument, claim ledger, caveats,
figure order, source bibliography, and remaining author decisions. It is not
a manuscript draft. No further model search is needed merely because the
fine-shape signal is smaller than hoped.
