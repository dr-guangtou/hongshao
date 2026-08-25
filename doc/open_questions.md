# Open questions and unresolved issues

A running register. Started 2026-08-25 at the user's request, during exp54.
**Nothing is deleted from here** — items are marked resolved with the date and
the answer, so a question that comes back can be recognised as a repeat.

Ordered by how much they could change a conclusion, not by age.

---

## A. Live, and capable of changing a published number

### A1. Every number fitted between Stage 3.3 and 2026-08-24 was contaminated
**Status: fix applied; Stage 3.5 re-run and UNCHANGED in its conclusion;
Stage 3.4's ceiling still owed (A3).**
Row 181 is a broken cross-match whose measured `M*(<100 kpc)` falls 3.76 dex
from 10^11.72 at redshift 0.4 to a flat ~10^8 at every earlier epoch. It sat
inside the fitting sample from Stage 3.3's per-epoch run onward and carried
**17.8% of the total loss**. `selection.sane_history_mask` now removes it in the
fitting path.

- **Done**: Stage 3.5 re-fitted on the clean sample — all eight variants moved
  by under 0.025 percentage points and the seven-criterion judge returned the
  IDENTICAL ordering, so its negative conclusion stands. Stages 3.6 and 3.7
  were fitted clean from the start. Tech note Section 5.3 rewritten (A2).
- **Owed**: the technical note's Section 5.3 table and every claim resting on
  it; Section 8's ceiling numbers (Stage 3.4 used the contaminated sample);
  Stage 3.3's own reported scores.
- **Question not yet answered**: are there other galaxies that pass
  `MAX_BACKWARD_DEX` but are still unphysical in a way the shell decomposition
  would expose? Only the 100 kpc aperture is currently tested.

### A2. "The model is 36% worse than a regression at total stellar mass" is withdrawn
**Status: RESOLVED 2026-08-25. Tech note Section 5.3 rewritten, with a new
Section 5.3.1 recording the withdrawal; Sections 9 and 10.2 struck through.**
Section 5.3 called this "the strongest argument against the model as it
stands". Ratio of the model's error to the statistical benchmark, on the
completeness-controlled sample:

| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| with row 181 | 1.12 | 1.31 | 1.29 | 1.30 | 1.16 |
| without it | 1.12 | 1.06 | 1.03 | 1.04 | **0.94** |

At redshift 2 the model **beats** the benchmark. The mechanism the note read as
physics — "the regression barely notices the restriction, the model degrades
sharply" — is arithmetic: the completeness cut shrinks the sample from 2397 to
840, so a fixed contaminant grows as a fraction of a mean square.
**Open**: is there ANY residual degradation on complete samples once this is
removed, or is the effect entirely gone? The 1.06/1.03/1.04 above is close
enough to 1 that it may be noise.

### A3. Stage 3.4's ceiling was computed on the contaminated sample
**Status: RESOLVED 2026-08-25 — re-run, and the conclusions hold.**
All three Stage 3.4 scripts had two contaminations, not one: the sample was
built without `sane_history_mask`, AND the model parameters were read from the
Stage 3.3 fit, which also contained row 181. Both fixed; the whole stage re-run
on branch `exp54-recheck-contaminated-bounds`; pre-fix outputs kept as
`outputs/stage34_*_PRE_ROW181.npz`.

**Every rung moved by 0.01 to 0.03 percentage points.** At the precision the
note quotes, only two cells change: the free-deposit bound 4.46% -> 4.44%
(quoted 4.5% -> 4.4%) and the three-interval rung 10.46% -> 10.44% (10.5% ->
10.4%). The wrong-galaxy control is unchanged at 5.04%, the fitted model at
14.18%, the ceiling at 8.47%. The decline split is unchanged: 443 declining of
839, model -37.2% inside 2 kpc at z=2 for those against -11.5% for the rest.

**Why it is this small, and why that is not a contradiction of A2.** A ceiling
is a MEAN over 839 galaxies, so one object moves it by at most its own excess
over 839. The amplitude headline it destroyed (A2) was a mean over a sample the
completeness cut had already shrunk, where the same fixed contribution was a
much larger fraction. Same galaxy, opposite consequence, for a reason that is
arithmetic and worth remembering.

**A sequencing bug worth recording**: the basis scan was launched in parallel
with the ceiling it validates against, so it compared to the stale ceiling file
and its own gate stopped it (max difference 1.4e-3 against a 1e-9 tolerance).
Re-run after the ceiling finished, the gate passes at exactly 0.00e+00. The
gate was right; the sequencing was mine.

### A4. `c_z` was rejected on the contaminated sample
**Status: RESOLVED 2026-08-25 — the rejection holds on the clean sample.**
Re-scanned with `sane_history_mask` applied and the clean-sample parameters:
the best non-zero `c_z` makes the ceiling **WORSE** by **+0.0017** in mean
shape score on all 2397 galaxies and by **+0.0201** on the 839-galaxy cohort,
against +0.001 and +0.02 before. The optimum sits at `c_z = 0` on an interior
minimum in both samples.

**Still true and still worth not conflating**: a redshift-dependent deposit
SHAPE is a different object from an early/late mixture of two FAMILIES at two
sizes, and rejecting the first does not reject the second. That second object
was tested independently in Stage 3.8 and also rejected, for a different reason
(C8).

---

## B. Live, methodological

### B1. The identifiability claim is still not a profile likelihood
**Status: RESOLVED 2026-08-25 (Stage 3.9). The conditional slices overstated
EVERY parameter, by 2.3x to 17.6x in width. No conclusion changes; what may be
said about the parameter values does.**

Each of the seven was fixed on a geometric grid of eleven values and the other
six re-optimised at every point, two starts each. The centre point of every grid
returned the incumbent's loss of 1.874253 to between 0 and 1e-11, so the fit is
converged and every rise is measured against the right zero.

| | `a_Mz` | `a_z` | `a_M` | `a0` | `b` | `log_f0` | `c` |
|---|---|---|---|---|---|---|---|
| conditional rise that survives | 0.32% | 0.42% | 0.50% | 0.69% | 1.45% | 1.88% | 19.7% |
| width overstated by | 17.6x | 15.4x | 14.2x | 12.1x | 8.3x | 7.3x | **2.3x** |
| moves this far before the best fit costs 1 percentage point of shape error | −0.41/+0.77 | −0.43/+0.61 | −0.81/+0.44 | −0.58/+0.39 | −1.17/+1.07 | −0.46/+0.58 | −0.25/+0.51 |

The ordering is the model's own structure: the four efficiency-law parameters
are the constant, two slopes and cross term of ONE bilinear surface, so any one
of them is absorbed by the other three tilting; the size pair trades against
itself and partly against the efficiency law; `c` alone changes the shape of an
individual deposit, and is the only one of the seven whose previously quoted
determination was roughly honest. **These are not confidence intervals** — the
loss is not a log-likelihood.

**Newly raised by this, and NOT pursued** — see C9 and C10 below.

### B2. The deposit truncation radius has never been scanned
`C = 3` halo radii is a convention. Owed: refit at `C = 1, 2, 3, 5` reporting
amplitude, shape, mass beyond the last aperture, and the fitted efficiency.
**Newly relevant**: Stage 3.7 varies the deposit size with redshift, which
interacts with a fixed truncation directly.

### B3. Completeness-threshold sensitivity
Every Section 6 conclusion rests on a 60% completeness threshold, a particular
mass-bin width and a plateau rule. None has been varied.

### B4. The per-epoch benchmarks are not persisted
`SIGMA_A` is a hard-coded array in `fit.py`. The completeness-controlled
`score_A` table cannot be reproduced from one output file.

---

## C. Raised by the current work, not yet pursued

### C1. Does removing the loss's blindness ever help?
Stage 3.6 measured the production loss as 175,000x less sensitive at 132–148 kpc
than at 2 kpc, repaired it, and the outskirts got **worse** (median error beyond
52 kpc −0.001/−0.008 dex to −0.011/−0.033; outer slope +0.12/+0.21 to
+0.16/+0.24). The natural reading is that the model has no freedom that moves
the outskirts independently of the centre. **Not tested.** A direct test: give
the deposit profile family one extra shape parameter and see whether the
outskirts and centre can move independently at all.

### C2. The objective changes the central LEVEL by 0.1 dex at no cost
Five objectives shifted the central mass by up to +0.13 dex with the amplitude
score essentially unchanged (1.033 to 1.038) and the loss changing by 2%. If
0.1 dex of central mass is nearly free, what is holding the level where it is
in the production fit — and is the level therefore not a constraint at all?

### C3. exp55's exponential core, and what may and may not be imported
A parallel experiment finds an exponential core plus an extended Moffat
reproduces curves of growth as well as a three-component decomposition, and
connects to halo growth **at a single epoch**. This model must serve five, so
the conclusion cannot be carried over — only the hint that the inner component
is exponential. The promising form here is an **early exponential deposition
and a late Moffat-like deposition**, i.e. a mixture whose weight depends on
deposition time, not a two-component profile per deposit. Untested.

### C4. The predicted growth distribution is half as wide as the simulation's
0.145 dex against 0.284. A mean law cannot generate the population's diversity
by construction, so this is scope rather than a defect — but it bounds what the
model can ever be used for, and no stochastic layer exists in exp54.

### C5. A bound fitted for one quantity was read as a bound on another
**Raised and corrected within Stage 3.7, 2026-08-25 — but it is the third
instance.** The free size law was fitted under the production loss and its
central span reported as a ceiling; refitted to minimise the span directly it
reached 0.061 dex instead of 0.139. The same error appears in `doc/lessons.md`
twice already. **Question**: is there a cheap structural guard? A bound object
that carries the quantity it was optimised for, and refuses to be compared
against anything else, would have caught all three.

### C6. Every "free curve" bound spends its freedom where there is no data
`E8` (Stage 3.5) and `S6` (Stage 3.7) both put large excursions above redshift
7, where under 1% of the stellar mass is deposited
(`figures/stage35_time_law.png` panel b, `figures/stage37_size_epoch.png`
panel b). They are still valid bounds, but their fitted shapes are not
interpretable and their parameters are undetermined (smallest singular value
1.2e-05 for `S6f6`). **Question**: should free-curve bounds be fitted with the
basis weighted by the deposited mass, so the freedom is spent where the data
is? That would make the bound tighter and the shape readable.

### C11. Reach and the core-density target are in direct tension
**Raised 2026-08-26 (exp57), unresolved.** Every long-reach expansion law passes
G5(c) — the model's median core surface-density growth against the measurement —
and fails G5(b), opening the central-error span of the 58% of galaxies whose
centres never declined. The one short-reach law does the reverse. Long reach
lowers the core density everywhere, which fixes (c) and over-corrects the
galaxies that were already right; short reach protects them and cannot move (c).
**Question**: is this a genuine physical tension, or an artefact of applying one
global expansion strength to a population with 0.33x the observed spread (C4)?
`diagnose_decliners.py` says the two groups differ in FORMATION TIME
(`r = -0.317` with `t_half/t`) and not in halo mass (`r = +0.025`), so a
formation-time-conditioned strength is the one shared extension with something
to condition on.

### C12. The loss prefers the long-reach basin, and the gates reject it
**Raised 2026-08-26 (exp57).** The core-only expansion has two optima — a core
radius of 2.6 kpc and a near-homologous 92 kpc — and the loss prefers the second
by 0.12% while the mechanism gates pass only the first. This is the fourth time
this program has found the loss and the physics tests pointing opposite ways.
**Question**: is there a modification to the objective that would make it see
the reach at all? Stage 3.6 already established the production loss is 175,000x
less sensitive at 132-148 kpc than at 2 kpc, and repairing that made the
outskirts worse. The honest alternative is to keep ranking by the gates and stop
expecting the loss to agree.

### C9. A finite-difference Hessian cannot resolve this model's flat directions
**Raised 2026-08-25 (Stage 3.9), unresolved and possibly not worth resolving.**
Sizing the profile grids from the Gauss-Newton prediction `1/(H^-1)_jj` was
tried first and failed: across finite-difference steps spanning a factor of ten,
the five largest eigenvalues of the loss Hessian are stable to better than 0.1%
while the two smallest move by a factor of 25 and one goes NEGATIVE. Not
non-convergence — re-optimising all seven from the incumbent reproduces its loss
exactly — but cancellation, pulling a curvature of order 0.1 out of a matrix
whose diagonal is of order 100. **Question**: is there a cheap way to get the
small curvatures (a Gauss-Newton `J^T J` from the residual Jacobian is positive
semi-definite by construction and might work), or should any future
identifiability claim simply be profiled directly, as Stage 3.9 did? The
profile cost 30 minutes per parameter, so the answer may be that the cheap
route is not needed.

### C10. `c` is the only parameter the model does not absorb — is that a lever?
**Raised 2026-08-25 (Stage 3.9), untested.** Six of the seven parameters are
absorbed 7 to 18-fold by the others; the deposit shape `c` is absorbed only
2.3-fold, because it is the only one that changes the shape of an individual
deposit rather than how much mass is deposited or how far out it reaches. C1
asks whether the model has ANY freedom that moves the outskirts independently of
the centre, and Stage 3.6 suggested it does not. **Question**: does that
2.3-fold number identify the deposit's radial shape as the one direction with
genuine independent purchase, and therefore as where an extra shape parameter
(C1) should be added? This is a suggestion from a degeneracy structure, not
evidence; it would have to be tested by fitting a two-parameter deposit family.

### C7. The span solution sits at a parameter bound
`log_f0` reached −0.0002 against a bound of 0.0, i.e. a deposit half-mass
radius equal to its halo's radius at redshift 0. The reported 0.061 dex span is
therefore partly a property of that bound. **Owed**: re-run with the bound
relaxed, or establish that the bound is physically meant.

**Partly answered 2026-08-25 (Stage 3.9), for the PRODUCTION fit but not for
Stage 3.7's span solution.** Profiling `log_f0` runs its grid all the way to
that bound, but the point at which the best achievable fit costs one percentage
point of profile-shape error is `log_f0 = −0.26`, comfortably INSIDE it. So for
the incumbent the bound is not binding and the reported width is a property of
the model. Stage 3.7's span solution is a different fit under a different
objective and is still owed the test.

### C8. THE CENTRAL DEFECT IS MOSTLY OUTSIDE THE MODEL CLASS
**Measured 2026-08-25, before Stage 3.8's fits, and it largely dissolves that
stage's premise.** Splitting on Stage 3.4's preregistered flag — did the
MEASURED mass inside 4.92 kpc fall between redshift 2 and 0.4 —

| central error, median log10(model/truth) | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 | span |
|---|---|---|---|---|---|---|
| all admitted | +0.020 | −0.003 | −0.021 | −0.062 | −0.126 | 0.145 |
| centre DECLINED (42%) | +0.068 | +0.007 | −0.042 | −0.127 | −0.202 | **0.270** |
| centre did NOT decline (58%) | −0.008 | −0.008 | −0.012 | −0.034 | −0.053 | **0.045** |

The per-galaxy scatter is 0.16–0.20 dex in BOTH groups, so on the non-declining
majority the systematic epoch trend (0.045 dex) is about a quarter of the
galaxy-to-galaxy scatter, and a near-uniform offset besides — absorbable by the
efficiency normalisation. The declining galaxies really do lose central mass:
median −0.066 dex from redshift 2 to 0.4, a 14% loss, 10th–90th percentile
−0.129 to −0.019.

**A deposition-only model adds mass monotonically and cannot follow a decline
at any parameter value.** So the 0.145 dex defect that Stages 3.5, 3.6 and 3.7
all failed to move is not one defect: most of it is a failure mode the model
class forbids, and the part a compact channel could address is already small.

It also explains Stage 3.7's trade. To reduce the POPULATION swing the size law
had to make every early deposit compact — including for the galaxies whose
centres were already right — which is why it emptied the outskirts.

**THE QUESTION FOR THE USER, because it is a model-class decision and not a
parameterisation one**: does exp54 stay deposition-only, or does it gain a way
to REMOVE or REDISTRIBUTE central mass? Options, none costed yet:
  - stay deposition-only, and report the declining galaxies as out of scope;
  - fit and validate on the non-declining subset, quoting that scope honestly;
  - add a stripping/redistribution term, which breaks the Contract B guarantee
    that the model contains no stellar information and no per-object tuning —
    that guarantee is why the current results are trustworthy.

**Rescoping owed either way**: refit the incumbent on the non-declining galaxies
ALONE and see what is left. The current 0.045 dex is from a fit dragged by the
declining half.

## D. Resolved

### D1. Is exp48's density-plus-log objective the repair for the outer blindness?
**No. Resolved 2026-08-24.** It is structurally blind to the mass inside the
innermost radius: differencing a cumulative profile gives R−1 values from R
radii, and adding mass inside the first radius shifts every later cumulative
value by the same constant, leaving every difference unchanged. Verified to a
relative 1e-13 and asserted in `hongshao/objective.py`. That aperture holds 12%
of the stellar mass at redshift 0.4 and 32% at redshift 2. Fitted under it, the
model puts +0.149 dex too much mass inside 2 kpc at redshift 0.4.

### D2. Can a richer time dependence for the efficiency law reach 10–11%?
**No. Resolved 2026-08-24 (Stage 3.5).** Five nested forms reach 13.62% against
the incumbent's 13.80%; a free shared curve reaches 13.64%. The budget is 0.2
percentage points and two parameters collect it. The 10–11% target came from a
ladder of per-galaxy bounds that never constrained a shared law.
