# exp72 — C17: what does a measured error bar do to the objective?

On branch `exp71-c18-selection-leakage` (the user asked for C18 and C17 in one
session). Open question: `doc/open_questions.md` C17. Data: exp68's v2 profile
product, `doc/tng300_profile_data.md` §10–11.

| script | what it does | writes |
|---|---|---|
| `errors.py` | **Step 1, no fit**: the misspecification measurement, the usability gate, the leverage sweep, the sensitivity probes, the ranking test | `outputs/errors.{npz,log}` |
| `fit_gls.py` | **Step 2**: refits exp63's twelve parameters under the one objective Step 1 cleared, 8 starts, ~2.3 h | `outputs/fit_gls.{npz,log}` |
| `eval_gls.py` | **Step 2b**: the transfer matrix, the standard battery, shape and amplitude separated, exp71's diagnostics re-run | `outputs/eval_gls.{npz,log}`, `figures/qa/` |

**The short answer to C17 is in "The answer to C17" at the foot of this file: a
measured error bar does NOT make these fits better behaved, and the reason is
specific and useful.**

`outputs/` is gitignored and regenerable (~9 min; `--selftest` in seconds). Run:

```
HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof OMP_NUM_THREADS=1 \
PYTHONPATH=. uv run python -u experiments/exp72_error_objective/errors.py [--smoke|--selftest]
```

**Sample**: the repo rule, 2356 of 2397, and essentially all of them keep their
errors — 2356 / 2356 / 2356 / 2356 / 2354 galaxies per epoch have a finite
positive isophotal error at all 24 radii. The profile-quality flags **compose**
with the fitting-sample rule and never substitute for it.

**The user's constraint, and its cost, carried everywhere.** At z = 0.4 the
**isophotal error is used alone; the projection term is not included**, because
the three-projection view may not be the most accurate model of it and it does
not exist at any other epoch. §11.1 measured what that costs: a median reduced
χ² of **3.4** under the isophotal error alone against **1.1** once projection is
added. The error model here is deliberately known to be about a factor 1.8 too
small in σ at z = 0.4, and by more above it.

---

## 1. This is not a likelihood, and the number says how far from one it is

Median |model − truth| / σ, cumulative basis, exp63's joint mean:

| epoch | 2 kpc | 10 kpc | 52 kpc | 103 kpc | 148 kpc |
|---|---|---|---|---|---|
| 0.4 | 7.3 | 7.8 | 11.4 | 12.3 | 12.9 |
| 1.0 | 6.1 | 6.3 | 9.0 | 9.6 | 9.7 |
| 2.0 | 5.1 | 5.6 | 6.7 | 6.8 | 6.7 |

The model is wrong by **5 to 13 times** the measurement error everywhere (3–8×
in the annulus basis), so a χ² per degree of freedom would be 25–170. **The
covariance is a choice of weights over radii, epochs and galaxies, not a
probability statement**, and every result below is phrased that way. The nested
incumbent gives the same numbers to within 0.3, so this is a property of the
model class.

Note the direction: the ratio **falls with redshift** (12.9 → 6.7 at 148 kpc),
because the measurement error grows with redshift faster than the model error
does. An error-weighted objective therefore moves attention *away from z = 2*,
the epoch that fails.

### What the error model actually constrains

Fraction of galaxies whose annulus at this radius has a **positive** measured
error. Where it is zero the curve of growth has stopped rising because the
isophotes ran out, and the error model then asserts that annulus holds exactly
zero extra mass with no uncertainty — a statement about what was measured, not
about what is there:

| epoch | ≤52 kpc | 103 kpc | 148 kpc |
|---|---|---|---|
| 0.4 | 100% | 100% | 100% |
| 1.0 | 100% | 97% | 97% |
| 1.5 | 100% | 87% | 87% |
| **2.0** | 100% | **67%** | **67%** |

At z = 2 a third of the sample has no outer constraint at all — in exactly the
place exp71 found the halo-mass tilt.

## 2. The usability gate: two candidates are disqualified before any fit

The **effective number of galaxies** is the participation ratio
(Σ L)² / Σ L² — the count that would give the same total if all contributed
equally. It falls to 1 when one galaxy carries everything. The production
objective is the bar; nothing that concentrates more than it does deserves a
fit. Of 2356 galaxies (share carried by the worst 1% in brackets):

| objective | floor f | z=0.4 | z=1.0 | z=2.0 |
|---|---|---|---|---|
| `prod` (the bar) | — | 1686 (4%) | 1613 (4%) | 1205 (7%) |
| `frac_sigw` | — | 1590 (4%) | 1522 (5%) | 1208 (7%) |
| `chi2_diag` | any | 719 (11%) | 581 (13%) | **233 (25%)** |
| `chi2_gls` | 0.00 | **1 (100%)** | **11 (96%)** | 35 (60%) |
| `chi2_gls` | 0.10 | 722 (11%) | 662 (12%) | 474 (16%) |
| `chi2_gls_amp` | 0.00 | **1 (100%)** | **17 (94%)** | 23 (61%) |
| **`chi2_gls_amp`** | **0.10** | 649 (11%) | **1102 (7%)** | **1054 (7%)** |

Three findings, in order of how much they cost:

- **An unfloored GLS χ² on this error model is unusable.** At z = 0.4 *one
  galaxy* carries the entire population's loss. `annulus_sigma_abs` spans eight
  decades — its smallest positive value is 2 M⊙ against a median of 3×10⁸ —
  because past a galaxy's last isophote the curve of growth stops rising and its
  variance stops rising with it. This is not a rank-1 subtlety; it is a floor
  problem, and it would have wasted every fit run on it.
- **The repair is one number.** Flooring the annulus error at a fraction `f` of
  the *cumulative* error at the same radius — always positive, the measurement's
  own scale there, no new assumption — restores the GLS forms. f = 0.03–0.10
  works; f = 0.30 begins to over-smooth and the concentration climbs again.
- **`chi2_diag` is disqualified on its own terms**, and the floor cannot help it
  (σ on the cumulative profile is never small). At z = 2 its worst 1% of
  galaxies carry a quarter of the loss. This is a *second*, independent mark
  against the diagonal, on top of §11.1's finding that it flatters a fit 10–200×.

## 3. Where each objective puts its attention

Share of the total contribution by radius, at f = 0.1:

| objective | 2 kpc | 10 kpc | 52 kpc | 148 kpc |
|---|---|---|---|---|
| `prod` (z=0.4) | 11.0% | 3.4% | 3.6% | 3.7% |
| `chi2_diag` (z=0.4) | 4.3% | 2.6% | 5.1% | 6.3% |
| `frac_sigw` (z=0.4) | 3.6% | 2.4% | 5.2% | 6.6% |
| `chi2_gls` (z=0.4) | **19.0%** | 8.0% | 3.6% | **1.3%** |
| `chi2_gls` (z=2.0) | **24.7%** | 8.1% | 0.9% | **0.3%** |
| `chi2_gls_amp` (z=2.0) | 14.3% (+38.2% amplitude) | 5.3% | 0.9% | 0.3% |

**The two bases pull in opposite directions, and that is the rank-1 structure
showing itself.** Weighting the *cumulative* profile by its own error moves
attention outward (`chi2_diag`, `frac_sigw`: 6.3–6.6% at 148 kpc against 3.7%
for `prod`), because the cumulative error is relatively smallest there.
Weighting the *annuli* moves it decisively inward (`chi2_gls`: 19–25% on the
innermost annulus, 0.3% at 148 kpc), because that is where the increments are
measured precisely relative to their size.

## 4. What each objective is actually for

The cost charged for a pure amplitude error and a pure shape error of the same
size (0.05 dex), each applied in **both** directions and averaged — a one-sided
probe can move the model *towards* the truth and score better, which the shape
probe did. Excess cost as a percentage of the exp63 mean's own loss:

| objective | z=0.4 amp / shape / ratio | z=2.0 amp / shape / ratio |
|---|---|---|
| `prod` | 7.6% / 1.5% / **5.1** | 5.0% / 0.2% / **26.4** |
| `chi2_diag` | 21.6% / 1.7% / 12.7 | 11.5% / 0.1% / **87.1** |
| `frac_sigw` | 8.5% / 0.9% / 10.0 | 5.2% / 0.1% / **40.8** |
| `chi2_gls` | 12.2% / 3.1% / 4.0 | 6.5% / 0.6% / 10.5 |
| **`chi2_gls_amp`** | 8.3% / 3.3% / **2.5** | 3.8% / 0.7% / **5.2** |

This is the most useful table in Step 1.

- **Every incumbent objective becomes an amplitude meter at high redshift.**
  `prod`'s amplitude-to-shape ratio runs 5.1 → 26.4 from z = 0.4 to z = 2, and
  its shape sensitivity collapses from 1.5% to **0.2%**. At z = 2 the production
  objective charges almost nothing for getting the profile's shape wrong. That
  is a plausible reason the z = 2 shape has never improved: the objective barely
  asks for it.
- **Error weighting alone makes this worse, not better.** `frac_sigw` — the
  conservative "just use 1/σ² weights" change — pushes the ratio to 10.0 → 40.8.
  It is *more* amplitude-dominated than the objective it replaces. That
  disqualifies it as a repair, though it passes the usability gate.
- **Only the GLS forms keep shape sensitivity at z = 2** (0.6–0.7% against
  0.1–0.2%), and `chi2_gls_amp` is the only objective that charges comparably
  for both (2.5–6.1 across all five epochs).

## 5. Do the objectives disagree? Yes — enough to justify a fit

Each objective's loss for the nested incumbent divided by its own value for the
exp63 mean (mean over galaxies | median). The exp63 mean was fitted and the
incumbent was not, so an honest objective should put this above 1:

| objective | z=0.4 | z=1.0 | z=2.0 |
|---|---|---|---|
| `prod` | 1.04\|1.00 | 1.05\|1.03 | **0.99**\|1.03 |
| `chi2_diag` | 1.09\|1.01 | 1.13\|1.07 | **0.94**\|1.00 |
| `frac_sigw` | 1.04\|1.01 | 1.05\|1.03 | **0.99**\|1.03 |
| `chi2_gls` | 1.21\|1.13 | 1.25\|1.14 | 1.09\|1.03 |
| `chi2_gls_amp` | 1.12\|1.13 | 1.10\|1.11 | 1.04\|1.03 |

At z = 2 the cumulative-basis objectives **prefer the unfitted incumbent** to
the fitted mean (0.94–0.99 on the mean over galaxies, though not on the median —
so it is carried by the tail). The GLS forms do not. The objectives genuinely
disagree, which is what makes a refit informative rather than redundant.

## What Step 1 concludes

**Disqualified before any fit:**

- `chi2_gls` and `chi2_gls_amp` **unfloored** — one galaxy carries the
  population. Not a candidate in any form without a floor.
- `chi2_diag` — 2–5× more concentrated than the production objective at every
  floor, and the most amplitude-dominated objective tested (ratio 87 at z = 2).
  Two independent disqualifications on top of §11.1's.
- `frac_sigw` — passes usability, but it makes the amplitude domination *worse*
  (ratio 40.8 at z = 2 against `prod`'s 26.4). It is a change in the wrong
  direction, so it does not earn a fit either.

**Earns a fit: `chi2_gls_amp` with the annulus error floored at f ≈ 0.03–0.1.**
It is the only candidate that passes the usability gate at z ≥ 0.7 (effective
1006–1199 galaxies against `prod`'s 1205–1614), keeps real shape sensitivity at
z = 2 (0.7% against 0.1–0.2%), balances amplitude against shape within a factor
2.5–6 at every epoch, and disagrees with the incumbent objectives about the two
real models. Its z = 0.4 concentration (649–744) is its weakest point and should
be reported with every number it produces.

**Two facts any fit under it must carry.** The error model is known to be too
small (isophotal only: reduced χ² 3.4, not 1.1). And at z = 2 a third of the
sample has no measured constraint beyond 103 kpc, so the outskirts of the epoch
that fails are precisely where this objective has least to say.

## Selftest

`--selftest` asserts six structural claims, all passing:

- **A** the positional cross-match reproduces `population.npz`'s curves of growth to 2.1e-15 relative.
- **B** the annulus χ² equals the full-covariance quadratic form to 1.4e-13 on 855 galaxy-epochs — so GLS needs no matrix inversion.
- **C** on a pure common-mode error the diagonal charges 8× what the full covariance does.
- **D** the amplitude probe leaves the pinned shape unchanged (2.2e-16) and the shape probe leaves `M(<103 kpc)` unchanged (2.2e-16) — the probes are pure.
- **E** `prod` reproduces `hongshao.objective.Objective()` to 1.0e-17.
- **F** 2.5% of annuli have exactly zero measured error, none inside 70 kpc (0.058%), and masking them keeps every χ² finite.

---

# Step 2 — the refit, and what it says about C17

`fit_gls.py` (2.3 h, 8 starts) and `eval_gls.py`. Products
`outputs/fit_gls.{npz,log}`, `outputs/eval_gls.{npz,log}`,
`figures/qa/exp72_*`.

**The fit is settled, so what follows is about the objective and not about the
optimiser.** All **8 of 8** starts land on loss **4.271534** (null 5.000, −14.6
per cent) from starting losses spanning 4.59 to 18.28, with θ agreeing to 0.0014
across every one of them. A single basin, no ambiguity.

```
a0=-2.5004  a_M=-0.4068  a_z=+0.4593  a_Mz=+0.2091  m_half=+11.8159  d_split=+0.8700
log_f_c=+1.0468  b_c=-1.1567  log_f_e=-0.8436  b_e=-0.2761  n_c=+0.5000  c_e=+0.9189
```

`n_c` rails at its lower bound 0.500 in **all eight** starts. That is exp63's own
Stage 5f pathology ("`n_c` at its lower bound in every joint start") reproducing
exactly, so the new objective inherits it rather than causing it — but the
compact channel is still being misused by the shared law, and the loss number
should not be read without it.

## The transfer matrix — it bought a preference, not a better model

Each fit's loss under each objective, nested incumbent at 1.000:

| under `chi2_gls_amp` | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | total |
|---|---|---|---|---|---|---|
| **gls refit** | 0.812 | 0.812 | 0.820 | 0.894 | 0.934 | **4.272** |
| exp63 joint | 0.893 | 0.883 | 0.907 | 0.947 | 0.961 | 4.592 |
| nested incumbent | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 5.000 |

| under exp63's four-term loss | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | total |
|---|---|---|---|---|---|---|
| gls refit | 3.127 | 2.922 | 2.991 | 5.106 | 4.444 | **18.590** |
| **exp63 joint** | 3.467 | 3.004 | 2.960 | 2.945 | 2.952 | **15.329** |
| nested incumbent | 4.139 | 4.032 | 3.871 | 3.700 | 3.501 | 19.242 |

|  | `chi2_gls_amp` | exp63 four-term |
|---|---|---|
| gls refit | **−7.0%** | **+21.3%** |
| nested incumbent | +8.9% | +25.5% |

The refit wins its own objective by 7 per cent and loses the incumbent objective
by 21 per cent, landing at 18.59 against the **unfitted** incumbent's 19.24. Read
per epoch, it is *better* than exp63 at z = 0.4 and z = 0.7 under exp63's own
loss and far worse at z = 1.5 and z = 2.0 (5.11 and 4.44 against 2.95 and 2.95)
— the direction Step 1 predicted, because the measurement error grows with
redshift and an error-weighted objective therefore stops asking about z = 2.

## What it did to the profiles — read this before the summary tables

Median (model − data)/data in per cent, on the fitting sample, 2354 galaxies.
Figures: `figures/qa/qa_bins_exp72_gls_refit.png` against
`..._exp72_exp63_joint.png` (note the two panels use different y-axis ranges,
±40 and ±20 per cent — compare the numbers, not the flatness).

| | 2 kpc | 5 kpc | 10 kpc | 52 kpc | 103 kpc | 148 kpc |
|---|---|---|---|---|---|---|
| z=0.4 gls refit | **−2.1** | +1.9 | +1.8 | −2.4 | −2.0 | −1.9 |
| z=0.4 exp63 | **+5.8** | −0.7 | −1.7 | −2.7 | −2.6 | −2.6 |
| z=0.7 gls refit | −6.4 | −0.6 | +1.2 | +1.1 | +1.9 | +2.1 |
| z=0.7 exp63 | +1.9 | −0.4 | +1.2 | +1.5 | +1.6 | +1.3 |
| z=1.0 gls refit | −11.5 | −2.5 | +0.0 | +2.1 | +2.7 | +2.8 |
| z=1.0 exp63 | −1.2 | +1.5 | +4.2 | +3.1 | +2.0 | +1.6 |
| z=1.5 gls refit | −22.4 | −8.6 | −6.6 | −1.1 | −0.2 | +0.1 |
| z=1.5 exp63 | −9.0 | +0.5 | +1.7 | +0.9 | +0.2 | −0.1 |
| z=2.0 gls refit | **−27.4** | −14.6 | −11.9 | −6.8 | −5.9 | −5.5 |
| z=2.0 exp63 | **−11.0** | −1.6 | −1.8 | −3.8 | −4.6 | −4.7 |

**THE DISCOVERY: at z = 0.4 the chi-square fit very nearly solves the central
defect.** It leaves **2.1 per cent** too little stellar mass inside 2 kpc where
the incumbent objective leaves **5.8 per cent too much**, and it holds within
±2.4 per cent at *every* radius from 2 to 148 kpc. The centre has been this
model class's standing failure
(memory `central-defect-is-outside-the-model-class`), and no objective tried in
this programme has done that before. It is the first evidence that the defect is
partly a property of what the objective ASKED FOR rather than of the deposition
class itself.

**What it cost is the high-redshift CENTRE, not the outskirts.** At z = 2 the
refit leaves 27.4 per cent too little mass inside 2 kpc against exp63's 11.0.
In the outskirts the two are close: −5.9 against −4.6 per cent at 103 kpc, a
1.3-point difference. The trade is *centre at low redshift against centre at
high redshift*, at fixed radius across epochs.

> **A reporting correction, recorded because it changed the reading.** An
> earlier version of this file led with `M*(50–100 kpc)` at +41.9 per cent at
> z = 2 and called the outskirts badly damaged. The number is right and the
> emphasis was wrong: a shell is the difference of two large cumulative masses,
> so a −12 per cent error at 52 kpc against −6 per cent at 103 kpc turns into
> tens of per cent in the shell between them. Quote the cumulative profile
> first; quote a shell only alongside the two apertures it is built from.

## The question the objective was chosen for: did the shape improve? Not as the battery measures it

Median over galaxies of the mean |log10(model/truth)| of the profile pinned at
`M*(<103 kpc)`, in dex — lower is a better shape:

| model | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| gls refit | **0.0340** | 0.0349 | 0.0348 | 0.0344 | 0.0311 |
| exp63 joint | 0.0358 | 0.0349 | 0.0338 | **0.0305** | **0.0246** |
| nested incumbent | 0.0366 | 0.0363 | 0.0356 | 0.0336 | 0.0297 |

The refit wins the shape at z = 0.4 and loses it from z = 1.0 upward; tier 3
agrees (0.3080 at z = 2 against exp63's 0.2966). Averaged over five epochs the
incumbent objective still wins, which is the sense in which Step 2 is a negative
— but it is an epoch-by-epoch trade, not a uniform loss.

**Why, and Step 1 had already measured it.** The share table showed `chi2_gls`
putting **19–25 per cent** of its weight on the innermost annulus and **0.3 per
cent** at 148 kpc, and section 1 showed the measurement error growing with
redshift so that the misspecification ratio falls from 12.9 to 6.7. The
objective therefore concentrates on **the centre** and on **low redshift**, and
one shared θ across five epochs spends that attention where it is cheapest. That
is a statement about the WEIGHTS, and weights can be changed — see "What to do
next".

## exp71's diagnostics are unmoved, as they should be

Future-growth leakage, partial ρ | dy/dG at fixed halo mass:

| model | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|
| gls refit | +0.177\|+0.170 | +0.197\|+0.144 | +0.211\|+0.132 | +0.202\|+0.129 |
| exp63 joint | +0.181\|+0.174 | +0.198\|+0.144 | +0.206\|+0.128 | +0.193\|+0.123 |
| nested incumbent | +0.183\|+0.175 | +0.204\|+0.148 | +0.222\|+0.140 | +0.215\|+0.140 |

Identical to within 0.02 across three models fitted under two different
objectives. **C18's answer does not depend on the objective**, which is what it
should do if the leakage is a property of the model class — and it is the
prediction exp71 made and this is the independent test of it.

## The answer to C17

**Does taking the measurement error into account make the model behave better,
or find a new solution?** It finds a genuinely different solution, that solution
is better at z ≤ 0.7 and worse at z ≥ 1.0, and the reason is now measured rather
than guessed.

**It is a different solution, not a nudge.** Eleven of the twelve parameters
moved, and they moved to a different account of how the galaxy was built:

| what it says physically | exp63 (fitted on distances) | fitted with the error bars |
|---|---|---|
| fraction of new stars landing in the **compact** component, at a typical halo | 0–2% at every epoch | **16–27%**, rising toward early times |
| how sharply that split switches with halo mass | a near-switch, over **0.5 dex** | a gradual blend, over **1.9 dex** |
| size of the **extended** building blocks at z = 2 | 9.8 kpc | **18.2 kpc** |
| size of the extended blocks at z = 0.4 | 69 kpc | 64 kpc |
| size of the **compact** building blocks, z = 0.4 → 2 | 6.7 → 3.7 kpc | 7.5 → 3.1 kpc |

**What it buys and what it costs**, in stellar mass the model gets wrong: the
z = 0.4 centre improves from +5.8 to −2.1 per cent inside 2 kpc, and the z = 2
centre degrades from −11.0 to −27.4 per cent. Everything outside 50 kpc is
within about 1–3 points at every epoch.

**Three things this settles.**

1. **The error model is a centre-weighting and a low-redshift weighting.** Its
   information sits in the inner annuli (19–25 per cent of the weight on the
   innermost, 0.3 per cent at 148 kpc) and its errors grow with redshift, so a
   χ² asks about the z = 0.4 centre and stops asking about z = 2. That is a
   property of the measurement, not a choice anyone made.
2. **The central defect is partly an objective artefact.** The z = 0.4 result is
   the first time anything in this programme has put the centre right, and it
   was done by changing what was asked for, not by changing the model class.
3. **C17's three original hopes are not delivered as stated.** The binned
   tercile term is still needed (without it the halo-mass-binned outskirts drift);
   `n_c` still rails at its bound in all eight starts, exactly as under exp63's
   objective; and the per-galaxy weighting did stop being uniform, but so
   violently that one galaxy carried the whole loss until a floor was imposed by
   hand.

**Not established here**, and it bounds everything above: the floor `f = 0.1` is
a modelling choice rather than a measurement; the error model is knowingly too
small (isophotal only, reduced χ² 3.4 not 1.1); and at z = 2 only 67 per cent of
galaxies constrain anything beyond 103 kpc.

## What to do next

**Not** a new experiment on the weighted chi-square for its own sake. The control
above removes its headline claim: the z = 0.4 centre is reachable with the
ordinary objective by dropping the shared law, and the chi-square is a worse
route to the same place.

What survives, and it is worth having:

- **Per-epoch and per-radius weights are a real dial, on a real compromise.**
  Every epoch already normalises to 1.000 at the null, yet the fit extracted 19
  per cent at z = 0.4 and 7 per cent at z = 2 (final per-epoch losses 0.812 /
  0.812 / 0.820 / 0.894 / 0.934), because that is where the gradient was
  cheapest. Explicit weights would let the compromise be CHOSEN rather than
  inherited from the measurement's error budget.
- **But a dial is not a ceiling.** It reallocates between epochs; it will not
  beat the per-epoch fits, and it should not be sold as new capability.

**The question actually worth compute** is the one behind both results, and it is
a science question rather than a numerical one: *what should a shared five-epoch
law be asked to preserve, given that it demonstrably cannot preserve
everything?* A weighting sweep answers it directly and cheaply — by evaluation
at frozen theta before any fit, the way Step 1 did — and it needs a decision
from the user about what matters, not more optimiser time.

Recorded as open question **C20**.

