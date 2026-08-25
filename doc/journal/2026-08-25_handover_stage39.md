# Session Handover — 2026-08-25 (Stage 3.9, open question B1)

## What this session did

**B1 is closed.** Genuine profile likelihoods for all seven parameters of
`gompertz_log-E2-S2`. The answer is that every identifiability number this
experiment has ever quoted overstated how well determined a parameter is, by
**2.3x to 17.6x in width**.

Branch **`exp54-stage39-profile-likelihood`**, two commits (`b71c7e4`,
`5c632a1`), **not merged and not pushed** — awaiting your review. Nothing is
running.

## The result

Fixing one parameter and re-optimising the other six, on eleven values per
parameter:

| parameter | what it sets | conditional rise surviving | width overstated by | may move |
|---|---|---|---|---|
| `a_Mz` | how the mass dependence changes with redshift | 0.32% | **17.6x** | −0.41 / +0.77 |
| `a_z` | how the efficiency changes with redshift | 0.42% | **15.4x** | −0.43 / +0.61 |
| `a_M` | how it changes with halo mass | 0.50% | **14.2x** | −0.81 / +0.44 |
| `a0` | mass kept by a `10^13.5 M_sun` halo at `z=0` | 0.69% | **12.1x** | −0.58 / +0.39 |
| `b` | how the deposit size fraction changes with redshift | 1.45% | **8.3x** | −1.17 / +1.07 |
| `log_f0` | the deposit size fraction at `z=0` | 1.88% | **7.3x** | −0.46 / +0.58 |
| `c` | the shape of one deposit's profile | 19.71% | **2.3x** | −0.25 / +0.51 |

"May move" is **how far the parameter can go, in each direction, before the best
achievable fit costs one percentage point of profile-shape error** — 13.79% to
14.79% typical error on the normalised curve of growth, `Δloss = 0.114`. **Not
a confidence interval**: the loss is a weighted sum of two normalised scores,
not a log-likelihood.

**The ordering is the model's own structure, and that is the part worth
keeping.** `a0`, `a_M`, `a_z`, `a_Mz` are the constant, two slopes and cross
term of one bilinear surface in halo mass and redshift, so displacing any one of
them is absorbed almost exactly by the other three tilting. The size pair trades
against itself and, through the mass inside 100 kpc, partly against the
efficiency law. `c` is the only parameter that changes an individual deposit's
radial **shape** rather than how much mass is laid down or how far out it goes,
and is the only one whose previously quoted determination was roughly honest.

**No conclusion changes.** The 13.787% profile-shape error, the representational
ceiling, the monotonicity limit of Section 8.10 and the four negative
enrichments are all statements about what the model class can reach, not about
where a coefficient sits. What changes is that `fit.py`'s rule — a parameter's
value is never quoted as a measurement unless it passes these checks — now binds
on all seven rather than on none.

## How it was done, and the two things that had to be checked first

- **The grid is geometric**, at `±1, 2, 4, 8, 16 × d0`, where `d0` is measured
  by bisection as the displacement at which the **conditional** rise reaches
  0.10 in loss units, separately in each direction (the loss is not symmetric:
  `c` needs `+5.02` to cost 2.0 and only `−0.40` for the same). Each grid
  therefore spans conditional rises from 0.10 to **25.6**. The conditional slice
  was measured at the identical points, one loss evaluation each.
- **The gate.** The centre point pins a parameter at its own fitted value and
  re-optimises the rest, so it must return the incumbent's loss. All seven
  returned **1.874253 to between 0 and 10⁻¹¹**.
- **A Hessian-sized grid was tried first and rejected.** Across
  finite-difference steps spanning a factor of ten, the five largest eigenvalues
  of the loss Hessian are stable to better than 0.1% while the two smallest move
  by **25x** and one goes **negative**. Cancellation, not non-convergence —
  ruled out by re-optimising all seven from the incumbent and recovering its
  loss exactly. Stored as a diagnostic; the instability is itself a result.

## Two side results, both needed before the headline could be quoted

- **The galaxy bootstrap, re-measured on the clean sample** (the stored one in
  `outputs/stage35_ident_PRE_ROW181/` predates the row-181 fix, so it could not
  be quoted beside anything). Spreads **0.9% to 8.0%** of each parameter's own
  value against 0.9% to 8.1% before — unchanged. In
  `outputs/stage39_ident_clean.npz`.
- **The bootstrap's own optimiser budget, checked and cleared.** A flat valley
  is exactly where a capped optimiser would fake a small spread. Four replicates
  refitted from the same start with an eight-fold budget (800 against 6400
  Nelder-Mead iterations) returned **bit-identical parameters** in the same wall
  clock. The spread is a measurement.

The profile widths are **20 to 37 times** the bootstrap spreads, because one
percentage point of shape error is a far larger degradation than sampling noise.
One bootstrap standard deviation costs 0.24% to 1.62% of that percentage point.
The two answer different questions and neither replaces the other.

## Where the bounds bite (open question C7)

`log_f0`'s grid reaches its upper bound of 0.0 and `c`'s reaches its lower bound
of 0.2, but **both one-percentage-point crossings lie inside those bounds**, so
neither reported width is a property of a bound. One free parameter does reach
one: at `b = −3.07`, far outside `b`'s determined range, the re-optimised
`log_f0` goes to exactly 0. Flagged; it touches no quoted number. Stage 3.7's
span solution is a different fit and is still owed the test.

## Files

New: `experiments/exp54_unpinned_amplitude/stage39_profile.py` (`--calibrate`,
`--only <name>`, `--extend <name>`, `--report`, `--smoke`),
`stage39_figures.py`, `doc/plans/2026-08-25-exp54-stage39-profile-likelihood.md`,
`doc/journal/2026-08-25_handover_stage39.md`.
Modified: `fit.py` (build `default_theta` only when no starts are supplied, so a
caller may pass any object that can report `bounds()` — that is how six of the
seven parameters are optimised through the same optimiser rather than a second
one), `README.md` (Stage 3.9), `doc/tech_note/deposition_model_2026-08.{md,pdf}`
(new Section 3.5.1, now 23 pp), `doc/open_questions.md`, `doc/todo.md`,
`doc/lessons.md`.
Outputs (gitignored): `outputs/stage39_grid.npz`,
`outputs/stage39_profiles/*.npz`, `outputs/stage39_profile.npz`,
`outputs/stage39_ident_clean.npz`, `figures/stage39_profile.{png,pdf}`.

## Cost

0.111 s per loss evaluation on 2397 galaxies, ~13000 evaluations per parameter,
about **30 minutes per parameter** with seven processes on ten cores. Set
`OMP_NUM_THREADS=1`: the evaluator is not limited by linear algebra, so threads
only contend. Full reproduction command in `README.md`, Stage 3.9.

## What is open, for you to choose

`doc/open_questions.md` is the register: A1 (partly), B2, B3, B4, C1–C10 live;
A2, A3, A4, B1, D1, D2 resolved. Two candidates worth naming:

- **B2 — the deposit truncation radius has never been scanned.** `C = 3` halo
  radii is a convention, and Stage 3.7's redshift-dependent deposit size
  interacts with a fixed truncation directly.
- **C10, newly raised by this session.** `c` is absorbed only 2.3-fold while
  everything else is absorbed 7 to 18-fold, because it is the only parameter
  that changes an individual deposit's radial shape. C1 asks whether the model
  has **any** freedom that moves the outskirts independently of the centre, and
  Stage 3.6 suggested it does not. Does that 2.3-fold number identify the
  deposit's radial shape as the one direction with genuine independent purchase,
  and so as where C1's extra shape parameter belongs? This is a suggestion from
  a degeneracy structure, not evidence, and would have to be tested by fitting a
  two-parameter deposit family.

The deferred expansion term (each deposit expands homologously after deposition,
`g = (t_k/t_j)^alpha`) is still deferred and still changes the model class; tech
note Section 10.2 item 3b.
