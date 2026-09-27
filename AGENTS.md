# AGENTS.md — HongShao

## Codex paper-planning isolation (user instructions, September 2026)

- (2026-09-28, user decision) Prepare Paper 1 in the separate sibling
  repository `ushmr1` (The Ultimate Stellar-Halo Mass Relation: I), not a
  gitignored directory inside this public repository. Do not create a public
  remote or publish unpublished results without explicit authorization.
  The next phase is analysis, visualization, and Markdown evidence notes;
  defer LaTeX manuscript drafting except figure captions.
- Paper figures must be publication-ready, use LaTeX fonts for all text,
  and have matching plotting scripts, saved figure data, provenance, and
  `.tex` captions. Each script must regenerate its PNG/PDF without rerunning
  fits or manual edits. Never silently fall back to non-LaTeX fonts or
  overwrite edited captions on a plotting rerun.
- Retain all qualifying centrals for Paper 1. Verify the original parent's
  peak-mass selection from source evidence; user recollection and downstream
  analysis masks alone do not certify the original query. Do not introduce
  stellar-mass, morphology, or star-formation cuts.
- (2026-09-27, user decision) Paper 1 concerns forward prediction of massive
  centrals' projected 1-D stellar mass distributions, primarily in TNG300 at
  z=0.4. Accurate individual-MAH recovery is not a project goal. A small
  reverse test is optional supporting evidence, never a completion gate.
- Anchor stellar mass at the native outer CoG aperture near 148 kpc, stating
  its exact stored radius. Test information beyond stellar mass and separately
  beyond mass plus size. Do not impose a figure or page limit.
- Paper 1 has an approved epoch-relevant sample exception: do not reject
  objects for unrelated multi-epoch history criteria. Predeclare applicable
  quality cuts and retain the old strict selection as a sensitivity. Other
  experiments' sample rules are unchanged.
- Xu's collaboration data are the source, with Xu's satellite-particle
  treatment, not Leidig's. Independent epoch fits may enter an appendix;
  joint evolution is not required. An N-body application is a possible
  forward-generation demonstration, not validation against stellar truth.
- Work on the paper plan only in the dedicated Codex feature worktree. Do not
  edit or switch Claude's main checkout, execute its code, or share writable
  caches. Read settled artifacts only for verified private snapshots.
- Keep this statistical-emulator paper plan distinct from Claude's physical
  model experiments and from unfinished Exp79. A paper-planning request does
  not adopt a statistical correction as the production physical framework.
- Claude retains integration ownership. The September 28 user request
  authorizes this planning branch's merge, but does not authorize changing
  Claude's active checkout: coordinate that integration first. No push or
  worktree deletion is included in this closeout.

## Working rules for two agents (user agreement, 2026-09-09)

- Claude uses the main `hongshao` directory on experiment feature branches
  started from current master; Claude alone handles integration and pushes.
- Codex uses isolated worktrees and feature branches only. Never edit files
  or switch branches in Claude's directory, or update master independently.
- Codex hands off verified commits, branch names and artifact locations to
  Claude. Delete a Codex worktree only after its commits and gitignored
  artifacts have been preserved, with copied artifacts hash-verified.
- Check existing worktrees, branches and experiment IDs before creation.
  Exp78 is reserved for Claude's size-aware objective work; do not duplicate it.
- Shared record additions must preserve both agents' contributions on merge.

Orientation for AI agents (and humans) working in this repo.

## What this is

HongShao is a **research project**, not a software product. The goal is the
"Ultimate SHMR": an assembly-resolved, profile-level extension of the
stellar–halo mass relation for massive central galaxies. See `README.md` for
the goal and `doc/` for the scientific source of truth:

- `doc/ultimate_shmr_context.md` — background, motivation, references.
- `doc/ultimate_shmr_possible_directions.md` — analysis directions + suggested sequence.

Read those two before proposing analyses.

## Scope — what HongShao is and isn't

HongShao predicts **central-galaxy stellar masses and profiles from halos** (the
Ultimate SHMR), and nothing downstream of that. In scope: the halo→galaxy map —
features (DiffMAH + `c_200c`), the N-target mean/scatter emulator
(`hongshao/emulator.py`) with generative sampling, the **profile/target layer**
(`hongshao/profile_emulator.py`) that graduates all four prediction modes (kpc
apertures, Re apertures, the cumulative CoG, and the 1-D density profile) through
that one core, the **multi-epoch layer** (`hongshao/multi_epoch.py`) — profiles
at any z in the fitted range by coefficient interpolation, with an AR(1)-in-epoch
latent for coherent multi-epoch draws (block-pinned product by default, log-CoG
product kept alongside; exp37) — and a thin, physically-labeled **deformation layer**
(`hongshao/forward.py`) — 5 knobs (`d0`, `d_slope`, `d_out`, `f_ab`, `s`) that
deform any of the four modes (target-agnostic for apertures; `forward_profile`
for the profile modes), with `Deform()` = the frozen baseline. The deformation
layer is the **hand-off boundary**: it outputs (deformed) stellar masses/profiles
for a halo catalog, full stop.

Supporting the above, two modules that fit rather than predict: **the fitting
objective** (`hongshao/objective.py`) — what a profile model is asked to
minimize, as six selectable axes with `Objective()` = the production objective
(exp48 measured the objective to be the strongest lever on the model's compact-
galaxy defect); and `hongshao/fitting.py`, currently one function guarding
against a scipy Nelder-Mead trap that has produced a false null here twice. An
objective is a fitting loss, not a likelihood — see the out-of-scope note below.

Out of scope (do NOT build here): weak-lensing or clustering predictions,
summary-statistic estimators, likelihoods, or samplers. Those need particle
data / pre-computed catalogs and a separate emulator each, and belong in a
distinct inference repo that *consumes* HongShao's predictions. Keep HongShao a
clean, portable SHMR library — don't let it grow into an inference framework.

## How to work here

- **Research mindset.** Expect exploration, false starts, and throwaway code.
  Favor fast, small-scale validation of an idea over polished infrastructure.
  Don't build frameworks for analyses that may not survive the week.
- **Measure, don't guess.** Never estimate numbers (scatter, variance
  explained, fit quality) — benchmark on the real data. Validate on a small
  subsample (sub-minute) before running on all 3388 halos.
- **Masses are h-free (Msun) by default.** All halo and stellar masses in this
  repo are log10(M / Msun), little-h divided out (`PICKLE_MASS_UNIT = 1e10/H`).
  External catalogs often differ: the official DiffMAH catalog (`diffmah_tng.h5`)
  is in **Msun/h** — add +log10(1/h) = +0.169 dex on ingest (see exp27). Always
  reconcile little-h before trusting a ~0.17 dex mass discrepancy.
- **Probabilistic framing.** Models are `P(theta_prof | M0, theta_MAH)`, not
  deterministic fits. Residual scatter is a result, not a bug.
- **Null models matter.** Compare against final-mass-only and shuffled-MAH
  controls before claiming an assembly signal.
- **Expect residual scatter; don't over-explain it — but test, don't assume.**
  Do not assume secondary halo properties will reduce it without evidence, and do
  not assume they *won't* either. We expected concentration to be redundant with
  the MAH; exp16 showed otherwise — `c_200c` is only ~25% MAH-determined and adds
  a real, independent +2.7% CRPS even on top of the full MAH-PCA(4) (+5% on the
  portable DiffMAH), and it is itself portable. So: measure each secondary
  property's incremental value (with a shuffle control), don't reason about it
  from "it correlates with formation time." Genuinely independent information
  (concentration, and possibly initial conditions / environment) can help. The stellar profiles are also single random 2-D projections of
  triaxial galaxies, adding noise we cannot remove from this dataset. Goal: a
  phenomenological model capturing the assembly-driven part, not zero scatter.

## Figures

- **Every experiment must produce at least one figure** demonstrating its
  result, whenever feasible. Use publication-quality styling via
  `hongshao.plotting.set_style()` (Okabe-Ito colorblind palette; `cividis` for
  sequential heatmaps; sequential, not diverging, when all values share a sign).
  Save PNG + PDF with `hongshao.plotting.save_fig()`. Follow the
  scientific-visualization skill for presentation.
- **Always surface new figures to the user for review** — don't just save them
  silently. In the SAME turn a figure is generated, explicitly list each new
  figure's full path (PNG, and note the PDF companion) AND a one-line description
  of its content, and display it. Do this every figure-generating turn, without
  being asked.
- **Always visualize fits and evaluations directly, not just summary metrics.**
  For any model fit or prediction check, produce an intuitive per-object
  visualization (e.g. measured-vs-model curves and residual profiles, by mass
  bin; truth-vs-predicted and residual-vs-truth) in addition to aggregate scores.
  Median RMS / CRPS hide problems that the eye catches instantly — e.g. the
  DiffMAH early-growth coverage issue (exp10) was obvious in the by-mass fit
  figure but invisible in the median RMS. Visual inspection is a first-class
  evaluation step, not an afterthought.
- **Promising or scientifically interesting models require the full standard QA
  battery.** Before treating such a model as a result, generate the same broad
  diagnostics used by the established experiments: aperture, annular, and
  outskirt masses; curves of growth and residuals binned by halo and stellar
  mass; density profiles; observational and size planes; cumulative residual
  distributions; and representative best, typical, and worst individual
  objects. Add cross-epoch stellar-growth diagnostics for multi-epoch models.
  Give every QA figure a self-contained caption or nearby summary that states
  what is plotted, the reference data, and the main interpretation.
- **Minimum visual gate for a promising direction.** Always inspect at least
  the average curves of growth in halo-mass bins and the stellar-mass planes
  before calling a model direction interesting or promising. Summary scores
  alone cannot support that judgment.

## Repo layout

```
hongshao/        # LIBRARY: stable, reusable, importable code (tng_data.py, …)
experiments/     # one self-contained folder per experiment (created on demand)
  expNN_slug/
    README.md    # question / method / inputs / key result / decision  (committed)
    run.py       # driver, written as # %% cell script                 (committed)
    figures/     # gitignored
    outputs/     # gitignored (tables, .npz/.fits) + manifest.json
data/
  external/      # vendored third-party inputs (committed, small)
  processed/     # shared derived datasets (gitignored, regenerable)
  raw/           # gitignored; raw drop lives outside the repo (see below)
scripts/         # cross-experiment tools (build dataset, QC)
doc/             # science context, data reference, todo.md, lessons.md
```

- **The experiment is the unit of organization.** Each gets a numbered,
  co-located folder; create it when the experiment starts, not upfront.
- **Library vs. experiment.** Reusable, validated code graduates into
  `hongshao/`; exploratory one-off analysis stays in the experiment folder.
- **Artifacts are regenerable, not committed.** Figures/datasets are gitignored
  and rebuilt from committed code + committed/vendored inputs. No DVC/Snakemake
  until a real shared pipeline needs it. Each `run.py` stamps the git SHA + key
  params into `outputs/manifest.json` for traceability.
- **Records:** per-experiment `README.md` = scientific source of truth;
  `doc/todo.md` = cross-experiment roadmap; the Obsidian journal = chronological
  session log. No overlap.

## Communication

- The user is a professional astronomer, not a software engineer or project
  manager. Explain plans, decisions, and trade-offs in plain language.
- Avoid software/PM jargon and tool names the user wouldn't know (e.g. YAGNI,
  CI/CD, DVC, Snakemake). If a term is genuinely important, define it in one
  plain sentence instead of assuming it.
- (2026-07-14, user rule) This explicitly includes development-methodology
  shorthand: say "write the check first, watch it fail, then code until it
  passes" instead of "TDD"; "one build-and-verify round" instead of "a TDD
  cycle"; describe what a change does rather than naming the methodology.
  Astronomy language needs NO simplification — the user is the domain expert;
  keep that side at full technical precision.
- (2026-07-14, user rule) **Result summaries must use plain, specific,
  self-contained language.** Concretely:
  - Every number quoted must say, in place: WHAT it measures (with units or
    definition), what the REFERENCE value is (the data, a previous model, a
    target), and whether it is better or worse — never a bare "(0.44/0.16)".
  - No project-internal shorthand labels without an in-place plain-language
    definition: not "P4" but "the known instability of the fitted
    star-formation-efficiency peak between single-epoch and joint fits"; not
    "the rail" but "the parameter pinned at its allowed bound"; not "the
    growth plane" but "the total stellar mass at z=2 versus at z=0.4, one
    point per galaxy". Experiment codenames and internal metric nicknames
    reset between sessions — the reader should never need the git history to
    decode a sentence.
  - When describing a new model component, first say in one sentence WHAT was
    done mechanically ("we fit a second, statistical model to the leftover
    errors of the physical model"), then the shorthand name may be introduced
    and used.
  - End each judged result with the interpretation: what this means for the
    model, in one plain sentence.

## Conventions

- (2026-09-09, user decision) Close Exp75/Exp77 as diagnostic experiments;
  do not advance their learned profile-correction maps as the production
  framework. The next direction is a compact physical prescription shared
  across epochs, tested against cumulative and annular masses together.
  Statistical direct/correction models remain diagnostic benchmarks. State
  full fitted complexity and which parameters observations would constrain
  before advocating a candidate for forward use.

- (2026-09-04, user clarification) Full halo MAHs, DiffMAH parameters,
  peak halo mass and final halo mass are legitimate prediction inputs, including
  halo history after the output epoch. The target galaxy's measured stellar
  mass at ANY epoch must never supply a prediction input or normalization.
  Stellar masses, their halo-mass relation and their evolution are outputs.
- (2026-09-04, user rule) Exp75 work must remain in its own worktree. Do not
  change Claude-owned branches, worktrees, environments, or artifacts. Read
  immutable source data only to make verified private input snapshots; never
  run another worktree's code or share writable output/cache directories.

- **Primary observable = CoG-derived masses.** Always model the
  aperture/annulus/outskirt stellar masses derived from the 1-D curve of growth
  (`logmstar_cog` / `logmstar_aper`, from X–Y isophote analysis). This is the
  observation-relevant quantity, and the end goal is to reproduce the CoG. The
  direct 2-D `logmstar_aper_proj` masses are physically more accurate (no
  perfect-ellipse assumption, matters during mergers) but are for
  **cross-checks only** — the difference is small. Do not switch the modeling
  target to `*_aper_proj`.
- English only — in code, comments, commits, and docs.
- Python: `snake_case` everywhere, never camelCase. Use `uv` for dependencies,
  `ruff` for lint/format.
- Exploration as `# %%` cell scripts (`.py`), not committed `.ipynb`.
- Raw data path via the `HONGSHAO_DATA_DIR` env var (defaults to the local drop).
- Never work on `master`/`main` directly — use feature branches; don't merge
  without permission.
- Record mistakes and rationale in `doc/lessons.md`; track the roadmap in
  `doc/todo.md`.
