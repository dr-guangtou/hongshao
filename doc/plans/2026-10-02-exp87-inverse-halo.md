# exp87 — the inverse problem: what the stellar curve of growth knows about its halo

## Context

Every experiment so far predicts the stellar curve of growth (CoG, log M*(<R) at
24 radii, 2–148 kpc, five epochs) FROM the halo's assembly history. The user
wants the reverse as a related but INDEPENDENT study: knowing the CoG
(including the stellar-mass normalisation), how well can we predict (1) halo
mass, (2) halo concentration, (3) the mass accretion history (MAH) or its
DiffMAH parameters. It must run in autopilot: a pre-declared ladder of methods
from linear regression to symbolic regression with rich operators, an
iteration logic, and a scoring metric that is statistically rigorous given
that the sample is selected by halo mass at z = 0.4.

Scope guard: the 2026-09-27 rule keeps inverse recovery out of Paper 1's
gates. exp87 touches no Paper-1 file or gate (`doc/paper1_statistical/` is
read-only for it).

User decisions (2026-10-02): headline at z = 0.4 on the complete PARENT
sample, curated sample elsewhere, the 2356 fitting sample as a labelled
sensitivity (an exception to the fitting-sample rule, scoped to exp87);
two labelled high-z populations; full ~14 h autopilot including the
oracle-Mh extension, hand back at the end, no merge without the user.

## What exploration established (facts the design rests on)

- **The parent is exactly complete.** Box centrals with M200c(z=0.4) ≥ 10^13.000
  number 3388 = the parent; 3380 have a finite z = 0.4 CoG
  (`data/processed/tng300_072_z0p4.fits`: CoG, `logmh_z0p4`, `c_200c`,
  `t50/t75/t90`, `dmah_*`). The lower edge is a hard step (20% of the sample
  within 0.1 dex of the cut). Raw box catalogue: `~/Desktop/tng300_halo_structure/`.
- **The curated 2397 / 2356 thinning depends on x at fixed y** (declined-MAH
  haloes +0.065 dex in M* at fixed Mh; flag A of `stellar_history_flags`
  removes exactly the catastrophic inverse errors).
- **Per-epoch halo data exist only for the 2397**:
  `experiments/exp54_unpinned_amplitude/outputs/halo_structure_history.npz`
  (`M200c` log10, `c200c`, `GroupFlag`), CoGs in
  `experiments/exp32_full_population/outputs/population.npz` (`data` (2397,5,24)),
  dense history `experiments/exp46_highz_ridge/outputs/so_history.npz`,
  leak-free per-epoch DiffMAH `experiments/exp74_c19_history_leak/outputs/history_curves.npz`.
  The official DiffMAH sees the future (C19) and is NOT a target.
- **Scale-setting checks** (single splits, to be redone properly): M*(<148)
  alone → Mh RMSE 0.173 dex; ridge on the 24 points 0.142; gradient boosting
  0.1415 (shape adds information, and it looks linear). Truncated-normal
  likelihood improves CRPS 0.104 → 0.088 (16%), larger than any ladder step.
  c200c and t50 from the CoG: R² ≈ 0.1; from the CoG plus the true Mh: 0.34 / 0.44.

## The statistics (the core of the plan)

1. **Estimand.** At z = 0.4 on the parent, sample = population of centrals
   above the cut, so ordinary proper scores are unbiased for that population.
   Truncation does not bias scoring; it breaks (a) the predictive family
   (support [c, ∞), a mean that bends like an inverse-Mills hockey stick),
   (b) transfer to differently selected populations, (c) comparability of
   variance-normalised numbers (R², r) across epochs.
2. **One predictive contract for every method**: a truncated normal on
   [c, ∞) with LATENT (m, s), fitted by truncated likelihood. Renormalising a
   naive Gaussian double-counts the cut. Flexible learners get a common head
   fitted inside the training folds on inner out-of-fold point predictions f
   (m = α0 + α1 f + α2 (f − f̄)², log s = β0 + β1 f) and optional
   Mills-corrected refits; the linear family also gets direct truncated
   regression, and the two routes must agree.
3. **Nonlinearity is judged in latent space or forward (x on y)**, where
   selection on y adds no bias; otherwise symbolic regression would be funded
   to rediscover the selection function.
4. **Primary metric: truncated CRPS in dex** (closed form; verified against
   quadrature). Co-reported: log score as information gain over climatology in
   nats; RMSE of the predictive mean and MAE of its median; PIT / coverage
   under the truncated CDF; reliability binned by PREDICTION, offset and width
   separately; threshold-weighted CRPS for the massive end (weight 1[t ≥ 14]),
   never a truth-selected subset; catastrophic rate (|error| > 0.5 dex);
   the prior-free forward resolution (bᵀΣ⁻¹b)^(−1/2), the only number
   comparable across epochs. No R² or r in headlines.
5. **Prior made explicit.** The generative inverse p(y|x) ∝ p(x|y) π(y)
   (forward fit on all galaxies, frozen; prior from the box counts) is exact
   under selection on y and is the method that moves between populations by
   changing π only. The mean shift γ ln10 (σ_x/b)² ≈ 0.12 dex (γ = 1.03–1.28
   measured) is reported.
6. **High z, two labelled populations.** (a) progenitors as-is: plain
   heteroscedastic Gaussian, no assumption. (b) complete above c_k
   (13.0/13.0/13.0/12.9/12.8): truncated at c_k, valid only if the CoG does
   not know future growth G at fixed y_k — tested multivariately with a
   pre-declared tolerance (shift < 0.2 σ_x per coordinate); failure demotes
   (b) to exploratory. Inverse-completeness weighting is a sensitivity row
   only (effective size 11–55 over the whole high-z sample).
7. **Honesty devices.** A 20% lockbox frozen at Stage 0 and scored once;
   five outer folds grouped by galaxy, identical for all methods, tuning
   inside folds; paired galaxy bootstrap on every score difference;
   "significant" = the 95% interval of ΔCRPS excludes zero AND the gain ≥
   δ = max(1%, the 95th-percentile null gain from the synthetic gate);
   controls: permuted target, shape shuffled within stellar-mass bins,
   cross-projection jitter at z = 0.4 (`logmstar_aper_proj`).

## Layout — `experiments/exp87_inverse_halo/` (branch `exp87-inverse-halo`)

| file | role | reuses |
| --- | --- | --- |
| `config.py` | constants, thresholds, budgets, seeds (hash in every row) | — |
| `data.py` | `load_parent()`, `load_curated()`, feature sets (scalar mass, 5 apertures, sizes/shape, fold-internal PCA, raw 24), x-only masks, lockbox + folds, cached box priors | `selection.sample_masses`, `sane_history_mask`, `stellar_history_flags`, `catalog_masses` (exp54 `selection.py`); `fit.R_GRID`; `qa.enclosed_radius`; fold pattern of `exp75/correction.py::make_folds` |
| `scoring.py` | truncated-normal + gridded predictives, CRPS / log score / PIT / tw-CRPS / energy score, paired bootstrap, reliability, forward resolution, quadrature self-test | `hongshao/metrics.py`, `hongshao/stats.py` |
| `heads.py` | common head, direct truncated regression, Mills iteration | — |
| `synthetic.py` | generator + `mechanics_gate()` | — |
| `methods.py` | registry L0–L3 | sklearn |
| `generative.py` | L5 forward fit on PCA scores, posterior on a y grid (also joint Mh–c) | `hongshao/emulator.py::fit` (`mean="linear"`) |
| `harness.py` | one cell → OOF file + one row in `outputs/scoreboard.jsonl`; idempotent; `--smoke` | `hongshao/provenance.write_manifest` |
| `controller.py` | state machine applying the rules, `decisions.jsonl`, ≤ 2 heavy jobs, budget guard | `exp82_delay_expansion/launch.py` |
| `sr.py` | staged PySR, per-fold checkpoints, `timeout_in_seconds` | `exp50_direct_cog_map/symbolic.py` (`new_symbolic_regressor`, `nested_symbolic`) |
| `mah.py` | history targets + vector methods | `so_history.npz`, exp74 `history_curves.npz`, `hongshao/tng_data.py::formation_times` |
| `lockbox.py`, `figures.py`, `README.md` | one-shot final score; figures to `figures/qa/` | `hongshao/plotting`, `qa._tex/_pct` |

Repo plan file `doc/plans/2026-10-02-exp87-inverse-halo.md` is committed
before any real-data score. No new dependency (sklearn + PySR only).

## Method ladder (per target, per epoch)

- **L0 references**: climatology; M*(<148) alone; best of a FIXED aperture
  list (<10, <30, <100, 50–100, >50 kpc).
- **L1 linear**: mass + sizes; ridge on the 24 points; PCA-k; PLS — each with
  the truncated-likelihood version.
- **L2**: poly-2 / splines on PCA scores.
- **L3 nonparametric**: kNN, GP (≤ 1500 points), HistGradientBoosting,
  random forest, MLP — all through the common head.
- **L4 symbolic regression (PySR)**: S1 `+ − × square` (size 20);
  S2 adds `/ sqrt log exp` (size 30, 180 s per fit, positive features);
  S3 adds `pow tanh max min` (size 40, 420 s per fit). Residual-of-linear
  mode nested in folds and direct mode; complexity chosen inside the fold;
  an equation is accepted if it beats L1 by δ and its skeleton recurs in
  ≥ 4 of 5 folds.
- **L5 generative inverse** with an explicit prior (Gaussian and Student-t
  forward scatter).

## Autopilot protocol

| stage | cap | produces | rule |
| --- | --- | --- | --- |
| 0 setup | 30 min | branch, certificates, frozen folds + lockbox | STOP unless parent = box (3388), CoG identity holds, σ(Mh\|M*) ≈ 0.175 and forward scatter 0.140 reproduce within 0.01 |
| 1 mechanics gate | 1 h | `gate_mechanics.json`, δ | STOP unless on synthetic truth: oracle beats naive / double-counted / shifted predictives; truncated ML recovers slope and scale within 3%; PIT uniform; a flexible learner + head shows no gain on a linear latent truth and detects an injected 3% curvature; the x-dependent-thinning variant shows its known bias |
| 2 selection report | 30 min | independence test, parent vs curated, effective sizes | failure of the tolerance demotes high-z (b) to exploratory; STOP only if it fails above c_k too |
| 3 ladder L0–L3 + L5 | 1.5 h | Mh and c200c: parent + curated at z = 0.4, both high-z populations; oracle-Mh extension (CoG + true Mh → c, t50) | fixed registry, no adaptivity |
| 4 symbolic regression | 8 h, 2 detached jobs | equations, stability, OOF scores | measure peak memory in a smoke run first. N = gain of best L2/L3 over best L1 in latent space at z = 0.4. Not significant → S1 only, residual mode, report "no detectable nonlinearity". Significant → S2; S3 only if S2 gained significantly. Other epochs only for accepted cells |
| 5 histories (vector → vector) | 2 h | scores at z = 0.4, then other epochs | targets pre-epoch only: Δlog M at fixed lookbacks, their PCA modes, exp74 DiffMAH parameters scored in curve space, t50/t75/t90 (parent). Methods: ridge, PLS, reduced-rank, per-mode gradient boosting. Baselines: climatology, mediated by CoG-predicted Mh, oracle Mh. If the gain over the mediated baseline on the first mode is not significant for both ridge and boosting, stop at the linear rung and report the null |
| 6 close | 1 h | lockbox once, figures, README (plain language first), appends to `doc/lessons.md`, `doc/todo.md`, `doc/open_questions.md`, handover, memory | hand back; no merge |

**Stop and wait for the user if**: a gate fails; a result contradicts the
record; the lockbox falls outside the development 99% interval; a budget or
memory limit is exceeded; anything outside exp87 would need editing; a gate
could pass only by being changed. No gate is relaxed after seeing results.

## Risks carried into the report

Truncated-normal tails may be wrong (PIT, Student-t forward model, quantile
boosting check); the relation below the cut is unobserved (no claim for
unselected populations); c200c is noisy for unrelaxed haloes (unknown skill
ceiling); one box, clustered galaxies (intervals slightly optimistic); PySR
with non-positive inputs under log / sqrt; the parent = box finding bears on
`doc/paper1_statistical/SAMPLE_PROVENANCE.md`, which exp87 does not edit —
reported to the user for them to route.

## Verification

- `uv run python -m experiments.exp87_inverse_halo.scoring` — closed-form
  truncated CRPS / log score against quadrature.
- `harness.py --smoke` on 200 galaxies, sub-minute, before every stage's
  full run; `uv run ruff check experiments/exp87_inverse_halo`.
- Stage 1's synthetic gate is the end-to-end test of the metric: known
  slope, scatter and skill recovered under truncation, known biases shown
  by the naive scores.
- Stage 0 certificates reproduce the record's numbers (parent count, CoG
  identity, 0.175 / 0.140 dex).
- Permuted-target control scores zero skill on real data; the lockbox score
  falls inside the development interval.
- Scoreboard rows carry config hash, fold ids and git SHA
  (`provenance.write_manifest`); figures regenerate from saved artifacts.
