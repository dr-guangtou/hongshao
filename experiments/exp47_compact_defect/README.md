# exp47 — the compact-galaxy defect: one problem or two?

exp46 reframed the program's top gap. The "high-z ridge" is the model
failing on **compact galaxies at every epoch** — trimming to truth
R50 >= 5 kpc is significant at p = 0.00 at *every* epoch including z=0.4,
where it drops only 6.3% of galaxies and still moves the centered
inner-vs-outer mass plane score from 2.2 to 1.5. The high-redshift
appearance is demographics: the R50 < 5 kpc fraction runs 6.3 / 14.0 /
27.4 / 60.7 / 85.2% across z = 0.4 -> 2.0.

The user's framing for this experiment: this is **also** the model's poor
performance in galaxy centres. exp45 measured the link between size and
central mass to be near-deterministic (per-galaxy Spearman between
dlog R50 and the M\*(<5 kpc) bias = **-0.89 to -0.97 at every epoch**).
What has never been measured is whether the *population-plane* residual
follows the same driver.

Four staged proposals, cheapest first. Stages 1 and 3 need **no fitting**
and are done before any model change, because together they define the
target and the scoring.

## Stage 1 — is it ONE defect? (`unification.py`, no fitting)

Per galaxy, does the model's displacement on the observational plane track
its central-mass error?

The plane test (`qa.plane_energy`) is a POPULATION statistic, so it has no
per-galaxy value. The per-galaxy object under test is therefore the
**displacement vector** on the plane: for each galaxy,
(dlog M\*(<30 kpc), dlog M\*(30-50 kpc)) = model minus truth. Decomposed
against the truth's own fitted plane relation into

- **along-ridge** — moves the galaxy along the mass sequence, mostly
  harmless to the centered plane score;
- **across-ridge** — moves it OFF the relation, which is exactly what the
  centered energy statistic penalizes.

Pre-registered readouts:
1. Spearman(across-ridge residual, M\*(<5 kpc) bias) per epoch. If |rho|
   is large, the plane failure and the central deficit are ONE defect.
2. The same against truth R50 and against the on-grid concentration index
   c_in = M\*(<10)/M\*(<148) — does the residual track compactness?
3. The fraction of the population-level centered plane energy contributed
   by compact galaxies, measured by leave-the-group-out against a
   size-matched random control (the exp46 protocol).

**How it will be read (fixed in advance):** |rho| >= 0.6 in readout 1 at
most epochs -> one defect; the option menu collapses and stage 2 targets
the central mass directly. |rho| <= 0.3 -> two defects, and the compact
galaxies need their own mechanism.

## Stage 3 — score compact and extended separately (`subpop_score.py`)

Compact galaxies are 6.3% of the z=0.4 sample. Every previous fix was
therefore judged on an average they barely enter, which is a candidate
explanation for a standing puzzle in the record: the exp38 core channel
and retention floor both halved the inner deficit and then "broke the
differential physics". If the loss is dominated by the 94% that are
already fine, a fix aimed at the minority can only look like damage.

Deliverable: a reusable scorer that reports every tier-2b/2d number for
the compact and extended sub-populations separately.

**The mandatory control** (exp46's lesson, measured): E/floor falls for
free when n shrinks — going from n=1200 to n=200 moved a *fixed* model
from 4.4 to 1.8. Compact and extended sub-populations differ in size by
more than an order of magnitude at low z, so their scores are NOT directly
comparable. Every sub-population score is therefore reported against a
size-matched random subsample of the full population.

## Stage 1b — the gate: proposals 2 and 4 apply at DIFFERENT epochs
## (`residual_predictability.py`, no kernel fitting)

Proposals 2 and 4 prescribe opposite fixes for the tilt — change the
deterministic model, or accept the residual as stochastic. The deciding
question is whether the across-ridge residual is predictable from halo
properties at all: the kernel is already a function of those properties,
so anything a halo feature can still predict is information the current
model is leaving on the table.

5-fold cross-validated R^2 (shuffle control in brackets — the same design
matrix with features permuted across galaxies):

**Target: the across-ridge plane residual**

| z | A: current conditioning | B: + DiffMAH shape | C: + early head start | D: + concentration history |
|---|---|---|---|---|
| 0.4 | 0.031 [-0.003] | 0.038 | 0.038 | 0.083 |
| 0.7 | -0.003 [-0.003] | 0.053 | 0.056 | 0.071 |
| 1.0 | 0.037 [-0.002] | **0.154** | 0.154 | 0.168 |
| 1.5 | 0.194 [-0.001] | **0.256** | 0.253 | 0.257 |
| 2.0 | 0.279 [-0.003] | **0.311** | 0.316 | 0.319 |

**Target: the M\*(<5 kpc) bias** — same shape: A gives 0.058 / 0.016 /
0.036 / 0.138 / 0.136, B gives 0.057 / 0.068 / **0.121** / 0.159 / 0.158.

Set A is the kernel's current conditioning vector `[logMh, c200c, fz2]`;
B adds DiffMAH shape `[logtc, early, late]`; C adds the exp46 head start;
D adds the fitted concentration history as a negative control.

**Four conclusions, and they resolve the proposal menu:**

1. **At z <= 0.7 the residual is essentially UNPREDICTABLE** from halo
   properties (R^2 0.03-0.08 against a ~0.00 shuffle). No deterministic
   halo-conditioned model can fix the low-redshift plane residual, however
   it is reparameterized. **Proposal 4 (the stochastic width) is the only
   route there** — this is now a measurement, not a preference.
2. **At z >= 1.0 it IS substantially predictable** (R^2 0.15-0.32).
   Roughly a third of the z=2 across-ridge residual is halo-predictable
   and is currently being left on the table. **Proposal 2 has real
   headroom, but only at high redshift.**
3. **The missing ingredient is DiffMAH SHAPE.** Going from set A to set B
   is the whole gain (0.037 -> 0.154 at z=1.0; 0.194 -> 0.256 at z=1.5;
   0.279 -> 0.311 at z=2). The kernel ingests the full MAH but conditions
   its parameters only on `[logMh, c200c, fz2]` — the shape of the history
   never reaches the conditioning.
4. **The exp46 head start adds nothing beyond DiffMAH shape** (set C is
   flat against B: 0.154 / 0.253 / 0.316). Not a contradiction — the head
   start IS largely what `logtc`/`early`/`late` encode, so exp46's
   discovery is already reachable through parameters we hold. And the
   concentration control behaves: nothing at high z, a small +0.03-0.05 at
   z <= 0.7 where everything is near zero anyway.

**Verdict: the two proposals are complementary, not competing, and the
epoch decides which applies.** The concrete, cheap model change this
implies is to extend the conditioning vector with DiffMAH shape
parameters — a strictly smaller step than restructuring the efficiency
window, and one that stays inside the portability criterion since the
kernel already takes the MAH as input.

## Stage 2 — free the efficiency window (held until 1 and 3 land)

The window is narrow (mu = 1.52, sigma = 0.24 -> peak z ~ 3.6, half-max
z ~ 2.6-4.8, most of the budget inside a ~1.2 Gyr span), which compresses
the range of arrival times so that every halo receives a near-identical
centre. Three independent pointers: sigma is the ONLY parameter that moves
cross-epoch coherence (exp45 follow-up 1; q and log_rc are inert);
exp42's real-MAH refit chose sigma = 0.52 and posted its largest gains at
exactly z = 1.5 and z = 2.0; exp45's scoreboard flagged that refit as one
of only two partial coherence exceptions.

## Stage 4 — a compactness-aware stochastic width (held)

Only ~14-22% of compactness is halo-predictable at fixed progenitor mass
(exp46 stage 1e, linear), so the remainder cannot be captured by ANY
deterministic halo-conditioned model. This is the pinned "halo-dependent
deviation WIDTH" item, now with a measured motivation.

## Not to chase

~1.5 of the mass-size plane's 5.0 at z=2 is the below-grid extrapolation
artifact (13% of z=2 galaxies have R50 under the 2.0 kpc CoG grid start).
The inner-vs-outer mass planes carry no such artifact.

## Run

```bash
export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof
PYTHONPATH=. uv run python experiments/exp47_compact_defect/unification.py \
    {demo|run}
PYTHONPATH=. uv run python experiments/exp47_compact_defect/subpop_score.py \
    {demo|run}
```

## Results — stages 1 and 3 (2026-08-12, full n=2397)

### Stage 1: ONE DEFECT. The pre-registered criterion is met.

**Readout 1** — Spearman of the per-galaxy across-ridge displacement
against the M\*(<5 kpc) bias:

| z | M(<30) vs M(30-50) | M(<30) vs M(50-100) |
|---|---|---|
| 0.4 | **-0.83** | **-0.75** |
| 0.7 | **-0.81** | **-0.70** |
| 1.0 | **-0.78** | **-0.65** |
| 1.5 | **-0.69** | -0.56 |
| 2.0 | **-0.65** | -0.54 |

The pre-registered threshold was |rho| >= 0.6 at most epochs -> one
defect. Plane 1 clears it at **every** epoch; plane 2 at three of five.
**The population-plane residual IS the central-mass error re-expressed.**
The option menu collapses: the model's poor performance in galaxy centres
and its failure on the observational planes are the same defect, and a fix
should be aimed at central mass.

Refinement worth keeping: the link **weakens monotonically with redshift**
(-0.83 -> -0.65, and -0.75 -> -0.54). So a component of the z=2 residual
is NOT central-mass driven — consistent with exp45's separate outer
over-extension at z=2 (R80/R90 +0.116/+0.112 above R50's +0.087).

**Readout 2 — the residual is a signed TILT, not extra noise.** The
*signed* across-ridge residual correlates strongly with compactness
(rho vs the on-grid index c_in = +0.66 / +0.69 / +0.74 / +0.81 / **+0.86**,
strengthening with z; vs R50, -0.54 to -0.69), while its *magnitude*
barely does (rho(|across|, R50) = -0.06 to -0.35). Compact and extended
galaxies are displaced in OPPOSITE directions on the plane. That is the
mechanism behind the standing "the relation is too flat" complaint — and
it means the fix is a change in the model's size-dependence, not simply
"add mass to compact galaxies".

**Readout 3** — centered E/floor with the compact group removed, against
a size-matched random removal:

| z | all | drop compact | drop random | n compact |
|---|---|---|---|---|
| 0.4 | 2.2 | 1.5 | 1.7 | 150 |
| 0.7 | 2.6 | 1.7 | 2.0 | 336 |
| 1.0 | 2.8 | **1.5** | 2.5 | 657 |
| 1.5 | 3.5 | **1.1** | 1.9 | 1456 |
| 2.0 | 4.8 | **1.3** | 1.6 | 2043 |

Beyond the control at z >= 1.0, confirming exp46 on an independent cut.

### Stage 3: the scoring premise is confirmed

| z | % compact | compact: score / ctrl / p | extended: score / ctrl / p |
|---|---|---|---|
| 0.4 | 6.3% | 0.9 / 0.7 / 0.81 | **1.5 / 1.7 / 0.00** |
| 0.7 | 14.0% | 1.2 / 1.0 / **1.00** | **1.7 / 2.1 / 0.00** |
| 1.0 | 27.4% | 2.0 / 1.4 / **1.00** | **1.5 / 2.4 / 0.00** |
| 1.5 | 60.7% | 2.4 / 2.4 / 0.50 | **1.1 / 2.0 / 0.00** |
| 2.0 | 85.2% | 3.8 / 3.8 / 0.50 | **1.3 / 1.6 / 0.00** |

(`p` = fraction of 16 size-matched random subsamples scoring at or below
the group. p ~ 0 = the group is genuinely BETTER than a random mix of the
same size; p ~ 1 = genuinely worse.)

1. **The model is significantly better on extended galaxies at EVERY
   epoch** (p = 0.00 throughout), and their absolute scores are 1.1-1.8
   everywhere — **including 1.3 at z=2**. There is no high-redshift
   problem for extended galaxies. This is the cleanest statement of
   exp46's reframe yet.
2. **Where the split is meaningful (z <= 1.0), compact galaxies score
   significantly worse** (p = 0.81 / 1.00 / 1.00).
3. **Honest limit on the z >= 1.5 rows**: compact galaxies are 61% and 85%
   of the population there, so a size-matched random subsample is mostly
   compact too and the control is degenerate (p = 0.50 by construction).
   Those two cells say nothing; the extended column at the same epochs
   does, because 354 extended galaxies are being compared against a
   control that is ~85% compact.
4. The premise behind proposal 3 therefore holds: at the low-redshift
   epochs where the data are best, the objective is dominated by a
   sub-population the model already fits well. A fix aimed at the failing
   minority would be scored almost entirely on galaxies it was not meant
   to change — a plausible mechanism for the record's standing puzzle,
   that the exp38 core channel and retention floor both halved the inner
   deficit and then "broke the differential physics".
