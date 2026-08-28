# Session Handover — 2026-08-28 (exp63: the analytic, two-scale, stochastic deposition model)

## Goal

The user asked for a fresh-eyes pivot: (1) prove that the "follow the halo mass
increments, then sum" recipe compresses to an analytic integral; (2) a new
single-epoch (z=0.4) Direction-3 model that grows a galaxy along its MAH and
explains the data better without unphysical transport; (3) a model that is
statistical from the start. Plan approved with five decisions (D1 analytic R200c,
D2 c200c only with leverage, D3 truncation measured first, D4 production loss +
shells/log term, D5 both samples at every predicted epoch); the user then said to
run every stage autonomously and adjust on the fly.

## Completed (all on branch `exp63-analytic-growth`, ~25 commits, NOT pushed)

- **Tech note 04** (`doc/tech_note/04_analytic_deposition_integral.md`): the sum is
  a first-order Riemann approximation of an exact integral (bias −0.027 dex at 2 kpc);
  the model is a convolution in log radius of the deposit-size distribution;
  incomplete-gamma closed form for power-law histories, verified to 1e-10.
- **Stage 0**: the engine (0.32 s per full-sample 5-epoch evaluation), gates
  G0a–G0d pass, D3 → truncation kept (3.2 % beyond 3 R200c); Result 0: the
  incumbent's central shape defect is not numerical.
- **Stage 1**: NNLS deconvolution — W bimodal (compact ~5 kpc fixed; extended
  43–74 kpc, share 0.35→0.57 with halo mass), n=4 deposits rejected, n=1 best,
  c200c no leverage (D2: no); G1: two scales warranted.
- **Stage 2**: two-channel mean fitted at z=0.4 only, one optimum from seven
  starts; inner slope 1.018 (data 1.015); the non-declining 58 % predicted to
  0.016 dex across five epochs; the extended channel arrives too early at z≥0.7;
  assembly correlations wrong-signed.
- **Stage 2b/2c/2d** (adjustments on the fly): accretion delay, absolute and
  relative growth-rate splits — each fixes one z=0.4 symptom and breaks the
  predicted epochs; recorded as the class tension P9. Stage 2 mean carried.
- **Stage 3**: five noise parameters inside the history; held-out G3a/G3b pass
  (planes 1.2–1.3× floor; tier-2e widths 0.94–1.02 from draws — C16 closed);
  widths hold at every predicted epoch; persistence +0.68 (truth +0.33).
- **Stage 4**: the battery on three products; P10 (mean- vs median-preserving
  noise) found.
- Docs: README (all stages + closing summary), PROBLEMS P1–P10, todo, lessons,
  memory (`exp63-results`, `deposit-size-distribution-is-bimodal`,
  `deposition-sum-is-an-analytic-integral`, `halo-history-knows-a-little-about-the-core`).

## Problems / self-corrections

- The tech-note probe first used a geomspace radius grid (P1); the "1.45 vs 0.88"
  inner-slope headline was wrong for a few hours; corrected everywhere.
- Tech note 04's "half the central deficit is numerics" overstated (P2); corrected.
- Two bookkeeping issues in the incumbent filed in `doc/todo.md` (dropped steps
  at catalog gaps; the z=0.4-anchored `diffmah_*` columns of
  `diffmah_combined.fits` disagree with the official curve at z>8).

## Decisions owed by the user

1. Merge/push `exp63-analytic-growth` (nothing merged; hongshao v1 unchanged).
2. P10: mean- or median-preserving noise convention for the reported biases.
3. Whether exp63's mean + process replaces the exp60 Option A layer as v1.
4. Whether to pursue P9 (a mechanism on the extended channel's LATE history).

## Branch State

`exp63-analytic-growth` at the last commit of this session, tree clean, level with
nothing remote (never pushed). `master` untouched at `a43d823`. No background
processes left running. Outputs/figures gitignored; regenerable (Stage 3's
full-sample Nelder–Mead took 225 min — lower `maxiter` if rerun).
