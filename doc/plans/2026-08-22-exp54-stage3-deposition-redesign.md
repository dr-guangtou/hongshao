# exp54 Stage 3 — redesigning the deposition model

**Status: PLANNED.** This is the model-building half of exp54. Stages 0-2 are
complete and shipped (`experiments/exp54_unpinned_amplitude/`); this document
covers Stage 3, which **replaces the physical kernel** rather than adjusting it.

Companion documents: `2026-08-22-exp54-unpinned-amplitude.md` (the parent plan,
revision 2) and `2026-08-22-exp54-evidence-probe.py` (reproduces its evidence).

**REVISION 2 (2026-08-23)** responds to the independent review at
`hongshao_internal_review_2026_08_16/docs/internal/2026-08-22-exp54-stage3-redesign-review.md`.
Its two structural findings — an inconsistent mass-budget contract (§2.1) and
an unmeasured representational floor for deposition-only models (§2.2) — are
**confirmed by independent re-measurement** and have changed the plan. Three of
its quantitative claims were re-derived here and reproduce exactly; see §2.3.

---

## 1. Context — how we got here

exp53 closed with the adopted kernel classified as an **oracle-amplitude shape
model**: it pins `M_model(<500 kpc) == M_TNG(<500 kpc)` per galaxy per epoch and
normalizes its deposit weights to sum to 1, so it carries no absolute stellar
mass at all. exp54 was opened to remove that pin.

Removing it turned out to be the smaller half of the problem. Once the pin is
gone, every remaining ingredient has to justify itself as physics — and **four
of the five load-bearing ingredients cannot**. That is the motivation for a
redesign rather than a patch, and each item below is a measurement made in this
branch, not an opinion.

### 1.1 What the audit found

**(a) The efficiency window carries no mass, and its form is unmotivated.**
`walkthrough.py` scaled `w(z)` by 1000 and the predicted profile changed by
**3.3e-16** — the amplitude is destroyed twice over, by `dM = w/w.sum()` and
again by the pin. What `w` actually controls is the RADIAL distribution: moving
its centre `mu` from 1.0 to 2.0 runs the half-light radius from **41.8 to 2.0
kpc, a factor of 20**, while the pinned mass is exactly invariant. So `w` is a
radial mixing weight wearing the name of an efficiency.

Its lognormal form has no physical derivation. The record shows the original
(exp25/exp29) efficiency was a **broken power law in (1+z)** — monotone — which
was replaced around exp33/exp35. **exp36 already flagged the replacement as a
defect** ("the efficiency form is too rigid: the lognormal f(z) peak drifts
between z04-only and multi fits, peak z 2.7 vs 4.0 — the only ingredient that
does not land in one basin") **and already prescribed the fix** ("replace the
lognormal f(z) with a monotone low-order spline"). It was never done.

**(b) The deposit size law is arbitrary and badly posed.**
`R50_i = 10^log_R50 (t_i/t_obs)^g`. `t_obs` is a GLOBAL constant, so the
normalization changes nothing but the meaning of the intercept — which lands at
**1253 kpc**, extrapolated to a time when almost no deposits form, with the
regression pivot at the far EDGE of the deposit time range and `g` railed at
4.0. Measured over 42405 deposits, the fraction of the halo's own virial radius
`f = R50/R200c` runs **0.0013 at z=8-20 to 0.49 at z=0.5-1, spread 0.913 dex** —
about 3x steeper in log than virial growth. It places the earliest deposits at
11 pc.

**(c) The conditioning form fails its one physical check.**
`parameter = base + slopes . standardized halo properties` is a first-order
Taylor expansion, not a derivation. The only physical expectation available is
that galaxy size tracks virial radius, `R_vir ~ Mh^(1/3)`, so the `log R50` vs
`logMh` slope should be near **+0.333**. The adopted fit gives **-0.059 per
dex** — essentially zero, wrong sign. exp49 had already measured these six
slopes as the least-determined part of the model.

**(d) The profile family's tail is physically absurd.**
Stage 0 check 4b: the fitted Moffat has `gam = 1.3784`, an outer surface-density
slope of **-2.76**, so enclosed mass converges only as `R^-0.76`. **A deposit
with a 50 kpc half-mass radius needs 9.6 Mpc to enclose 99% of its own mass**
and 201 Mpc for 99.9%. This is why **18.0% of every galaxy's deposited stellar
mass lands beyond 500 kpc** (Stage 0 check 4) — silently discarded and restored
by the pin. Under the pin "the deposit's total mass" was never a well-posed
quantity, which is why nothing noticed.

**(e) The one ingredient that survives: transport does a real job.**
exp53's sharpest result is that deposition-only models carry a radial
distribution error that **flips sign with halo mass** (`M(<148)` -3% at low halo
mass, +6% at high, a 9-point tilt where the fiducial is flat), and concluded
that transport is what supplies the halo-mass-dependent concentration. The
redesign deletes transport, so **something else must supply that**, and §4.4
says what.

### 1.2 The quantitative case for a redesign

Stage 1's shuffled-MAH control is the decisive number. Permuting whole MAHs
within narrow final-mass bins and **refitting**, at z=0.4:

| model | real MAH | shuffled MAH | history information used |
|---|---|---|---|
| DiffMAH 4 params + linear regression | 0.1130 | 0.1228 | **0.0098 dex** |
| the 3-parameter deposition law | 0.1231 | 0.1247 | **0.0016 dex** |

**The physical law extracts about one sixth of the assembly-history information
that a plain linear regression on the same four numbers extracts.** A single
integral of a smooth efficiency against `dMh` cannot express what the MAH
supports. That is a statement about the functional form, and it is the
strongest single argument for rebuilding rather than tuning.

### 1.3 What Stage 2 says about the stakes

Re-baselining measured which conclusions actually depend on the pin.
**Population-level results are robust**: the tier 2b plane at z=0.4 moves
2.17 -> 2.20 under a realistic, cross-epoch-correlated amplitude error (the
parent plan predicted 2.1 -> 2.9; that estimate applied a paired operation to a
pairing-blind metric). **Per-galaxy results are not**: tier 3 profile max|rel|
degrades +38% at z=0.4 and +91% at z=2.0. And the pin's real hiding place was
the **cross-epoch growth tier**, where `Mtot` growth energy ratios sat at
0.27-0.33 — below the truth's own split-half floor, "indistinguishable from
reality" by construction — and are 1.22-1.60 once unpinned.

So: the redesign is not required to rescue the population-level science. It is
required to make the model **deployable** (it cannot run on an N-body halo at
all), to make its **mass growth** an honest prediction, and to replace four
ingredients that survive only because nothing ever scored them.

---

## 2. Two problems found in review, both confirmed

### 2.1 The mass-budget contract was inconsistent between stages

Stage 1 fitted `SUM_j eps_j dMh_j` **directly to the measured
`M*(<100 kpc)`**, with no radial profile and no factor for the fraction landing
inside 100 kpc. Stage 3 as first drafted then defined the SAME sum as the
deposit's total mass over all radii. Once a profile with a tail beyond 100 kpc
exists, the two definitions disagree:

```
M*(<100, t_k)  =  SUM_{j<=k}  eps_j * dMh_j * F_j(<100)      F_j(<100) < 1
```

Confirmed by reading the code: `scoreboard.py::law_predict` returns
`log10 SUM(eps dMh)` and it is scored against `y = log10 M*(<100 kpc)`. So

- **the Stage 1 parameters are STARTING VALUES, not a completed amplitude law**;
- **the Stage 1 scatter is an APERTURE benchmark**, not automatically the floor
  for a total-deposited-mass model;
- efficiency and radial reach are **coupled in the joint fit** even though the
  objective separates amplitude from shape.

**Resolution (see §2.4): adopt Contract B, made well-posed by truncation, and
rename `eps`.**

### 2.2 Deposition-only has a representational floor, and nobody had measured it

Any sum of **static, positive** deposits makes `M*(<R,t)` non-decreasing in time
at every radius. TNG violates this constantly. Independently re-measured here
(n=2397), fraction of galaxies whose enclosed mass DECREASES:

| interval | R=4.9 kpc | R=10.2 kpc | R=103.5 kpc |
|---|---|---|---|
| z=0.7 -> 0.4 | 57.9% | 46.6% | 15.6% |
| z=1.0 -> 0.7 | 55.1% | 47.7% | 19.9% |
| z=1.5 -> 1.0 | 57.3% | 42.2% | 14.8% |
| z=2.0 -> 1.5 | 50.1% | 42.3% | 16.1% |

Over the full z=2 -> 0.4 baseline, **66.5% decrease at 2 kpc** and 41.8% at
4.9 kpc. Projecting each galaxy's five-epoch sequence onto the closest
non-decreasing sequence (pool-adjacent-violators, per radius) gives a
**median all-epoch all-radius correction of 0.0227 dex RMS**, per-epoch maximum
corrections of 9.7-16.6%, and a median worst-case of ~30% somewhere in the grid.

**No static positive deposition model — one channel or two — can do better than
this.** Two static channels cannot repair it either; monotonicity is a property
of positivity, not of channel count.

**How much is real? A measurement the review did not have.** Among galaxies that
DO decrease, the median fractional loss at 4.9 kpc is **6.5 / 6.1 / 8.0 / 7.0%**
across the four adjacent-epoch intervals, against a stellar-mass-loss
expectation of only ~1-1.5% over the same 1-2 Gyr for an old population.
**Stellar mass loss therefore explains roughly one fifth of the effect.** The
remainder is some mixture of genuine radial redistribution and irreducible
measurement noise (projection of a triaxial galaxy, centring, merger
transients, particle assignment) — and **a deterministic model must not be
penalized for the noise part**. Separating the two is a Stage 3.0 task
(§6.0), because it decides whether a survival term is worth building.

### 2.3 Review claims independently re-derived

All three checked reproduce exactly, so the review's unchecked claims are
treated as credible:

| claim | review | re-derived here |
|---|---|---|
| monotonicity violation fractions | 57.9 / 46.6 / 15.6 %, ... | identical |
| isotonic floor | 0.0227 dex RMS | 0.0227 dex RMS |
| concentration coverage, stellar-weighted | 94.4% (z=0.4), 85.3% (z=2) | 94.4%, 85.3% |
| coverage by halo-mass tercile at z=0.4 | 90.8% -> 96.1% | 90.8% / 93.0% / 96.1% |

### 2.4 The contract, decided

**Adopt Contract B — total surviving central-galaxy deposition — because
truncation makes it well-posed.** The review's objection to Contract B was that
"the total is not directly measured by the current CoGs and needs a boundary or
a latent survival convention". **The truncation decision supplies exactly that
boundary**: the deposit's total is DEFINED as the mass inside
`R_trunc(t_j) = C * R200c(t_j)`, a halo-determined radius carrying no stellar
information. There is no appeal to a total at infinity.

**`eps` is renamed.** It is not a star-formation efficiency. Call it the
**effective surviving-central mass fraction**, `eps_surv`: the fraction of a
halo-mass increment that ends up as stars assigned to the central galaxy inside
`R_trunc`, as of the observation epoch. It absorbs baryon conversion, stellar
evolutionary mass loss, stripping, central-versus-satellite assignment, and
ex-situ delivery, and **those factors must not be interpreted separately** until
a later model decomposes them.

**Contract A is retained as the statistical benchmark.** The Stage 1 scoreboard
is a Contract-A measurement and stays the performance reference; it is not the
Contract-B floor.

## 3. The model

The user's specification (2026-08-22), which this plan implements:

> Given a MAH and halo properties, between two epochs a fraction of the halo
> mass increase converts into stellar mass. That fraction can depend on
> redshift or more. The new stellar mass follows a functional form for its
> radial distribution, with a few parameters, some or all depending on halo
> mass and redshift. The integral of that profile equals the stellar mass the
> efficiency predicts. To predict the distribution at a given epoch, sum the
> profiles from all past epochs.

Formally, for galaxy `i` observed at epoch `t_k`:

```
R_trunc,j      =  C * R200c(t_j)                             [0]  the boundary
Delta M*_j     =  eps_surv(H_j) * Delta Mh_j                 [1]  the budget
Sigma_j(R)     =  Delta M*_j * sigma( R ; theta_prof(H_j) )  [2]  the placement
M*(<R, t_k)    =  SUM_{t_j <= t_k}  Delta M*_j * F( R ; theta_prof(H_j) )  [3]
```

with `sigma` a projected surface density normalized so `F(R_trunc,j) = 1`. [0]
is what makes [1] well-posed: `Delta M*_j` is the mass inside a HALO-DEFINED
radius, never a total at infinity.

where `H_j` is the halo state **at the deposit's own formation time** `t_j` and
`F` is the unit-mass cumulative profile, so [2] conserves mass exactly.
`eps_surv` is the **effective surviving-central mass fraction** of §2.4, not a
star-formation efficiency. **No normalization, no rescaling,
no reference to any stellar quantity.** Stage 0's poisoned-data test is the
acceptance criterion: NaN every stellar array and the forward call must still
return finite masses in solar masses.

**Deleted from the incumbent**: the truth pin; `dM = w/w.sum()`; the transport
term (`q`, the core fraction `fc`); the `t_obs` normalization; the catalog
`logMh(z=0.4)` input; the 500 kpc reference radius.

---

## 4. Design axes and candidate forms

Each axis is built as a **switchable, nested family** in the exp53 `Spec`
pattern, so candidates are compared inside one harness and the simplest form is
always a special case of the richer one. Nesting matters: it makes "does this
term earn its place" a question the identifiability analysis can answer.

### 4.1 The efficiency `eps_surv(H)`

Base coordinate: `x = ln(1+z_j)`, `m = logMh(t_j) - 13.5`.

| id | form | rationale | status |
|---|---|---|---|
| **E1** | `log10 eps = a0 + a_M m + a_z x` | the measured base; 3 params, CV-ties the halo-mass baseline, bias <=0.004 dex | **measured** |
| **E2** | E1 `+ a_Mz m x` | the mass dependence tilts with redshift; measured to help at every epoch | **measured** |
| **E3** | E1/E2 `+ a_c * dlogc(t_j)` | concentration excess over Diemer19; measured to help | **measured** |
| **E4** | E1 with `a_z x` -> free cubic spline in `x`, 3-5 knots | exp36's prescription (D), WITHOUT its monotonicity constraint — let the data say whether a peak exists | new |
| **E5** | double power law in mass: `eps = 2 eps0(z) / [ (M/M1)^-beta + (M/M1)^gamma ]` | the standard SHMR form (Moster/Behroozi); contains a peak in MASS | **BUILD NOW** (user) |

**E1 is the base; E2/E3/E4 are switchable extensions; E5 IS BUILT AND TESTED
NOW** (user, 2026-08-22 — "build it now, test it anyway"). Scope statement to
carry, and the reason E5's outcome must be read carefully: this sample is
logMh = 13.01/13.29/14.36 (1/50/99th percentile), entirely above the SHMR peak
at ~10^12, so a single power law in mass is adequate **here**. **Pre-registered
consequence**: E5's peak mass `M1` and its low-mass slope `beta` should come out
UNIDENTIFIED — the sample never samples them. If the identifiability report says
otherwise, the fit is absorbing something else through them and E5 must not be
adopted on that evidence. E5 earns its place as the honest general form for
later application below the peak, not as a better fit here.

**Do NOT impose monotonicity in redshift.** The measurement says only that a
peak is *unconstrained* by the mass budget, not that it is excluded — the free
lognormal degenerated to the power law rather than rejecting it. E4 plus the
identifiability report is the honest way to settle it.

**PARAMETERIZATION TRAP, measured**: never stack the `a_Mz` cross-term on a free
lognormal window. That combination reaches the same loss at `a0 = 26.8,
mu = 92.5, sig = 7.93` — meaningless values a loss check cannot catch. On the
monotone base the same term is well-behaved (`a_Mz = +0.268`).

### 4.2 The deposit profile `F`

**A new hard requirement, from Stage 0 check 4b.** Any candidate family must
satisfy a **convergence constraint**: the radius enclosing 99% of a single
deposit's mass must not exceed a fixed multiple of the halo's virial radius at
that time,

```
R99( theta_prof )  <=  C * R200c(t_j),     C a fixed constant (start at C = 3)
```

This is a physical statement — stars are not deposited a hundred virial radii
out — and it replaces the arbitrary 300 kpc ceiling with something dimensionally
sensible. **It immediately excludes the incumbent Moffat at its fitted `gam`**
(R99 = 192 R50 ~ 9.6 Mpc against R200c ~ 0.5 Mpc).

**DECIDED (user, 2026-08-22): `C = 3`, ENFORCED BY TRUNCATION, and `C` is
scanned rather than fixed.** Revision 2 adds the review's qualification:
**`R99 <= 3 R200c` is NOT asserted as a physical law**. Both the 99% quantile
and the factor 3 are unmeasured, and a tiny analytic tail can dominate `R99`
while changing no observable. So:

- **Report before constraining.** For every candidate family report
  `R90/R200c`, `R95/R200c`, `R99/R200c`, and the mass fractions beyond
  `R200c`, `3 R200c`, 500 kpc and the 148 kpc measured horizon, each as a
  function of epoch and halo mass.
- **Then** use `C = 3` truncation as the working default and scan `C` over
  {1, 2, 3, 5}, deriving a preferred value from a declared central-galaxy /
  intracluster-light boundary rather than from taste.

*The TNG evidence for `C = 3`.* Measured on the truth:

| z | R200c (physical) | 148 kpc / R200c | 500 kpc / R200c | M(<148)/M(<500) |
|---|---|---|---|---|
| 0.4 | 492.9 kpc | 0.301 | 1.014 | 0.9285 |
| 0.7 | 394.5 | 0.376 | 1.267 | 0.9463 |
| 1.0 | 325.9 | 0.455 | 1.534 | 0.9597 |
| 1.5 | 234.7 | 0.631 | 2.130 | 0.9761 |
| 2.0 | 172.0 | 0.862 | 2.906 | 0.9898 |

Essentially all stellar mass sits inside **~1 R200c at z=0.4** and ~3 R200c at
z=2, so `C = 3` never excludes anything TNG has. **Caveat to carry: `C = 3` is
LOOSE at low z and TIGHT at high z** — 1479 kpc at z=0.4 against 516 kpc at
z=2. That asymmetry is defensible for a DEPOSIT constraint (early deposits form
in small haloes and should be compact) but it means `C` does less work at z=0.4
than the number suggests. Scan `C` over {1, 2, 3, 5} and report the sensitivity.

*Enforcement is by TRUNCATION, not by penalty.* For any family, define the
deposit's effective total mass inside the truncation radius and renormalize:

```
R_trunc(t_j)  =  C * R200c(t_j)
F_eff(R)      =  F(R) / F(R_trunc)          for R <= R_trunc,   1 beyond
```

so `Delta M*_j` is by construction the mass inside `R_trunc`. Three advantages
over a penalty: every family's total mass becomes well-defined and directly
comparable (the incumbent Moffat's was not — Stage 0 needed a 1e15 kpc
"infinity" to close it); mass conservation stays exact; and no fit has to
negotiate with a soft constraint, which is one fewer degenerate direction.

**This is NOT the truth pin.** `R_trunc` depends only on `R200c(t_j)`, a halo
quantity. Stage 0's poisoned-data test must still pass, and is the check.

**MANDATORY when truncating (review §4.2): re-solve for the half-mass radius.**
Truncating changes it — the advertised `R50` must be the ACTUAL post-truncation
half-mass radius, or the truncated family is no longer in the same coordinate
system as the untruncated ones and the shootout compares different quantities.
Assert `F(R50) = 0.5` numerically for every family at every fitted theta.

**MANDATORY for the sigmoid families**: verify the implied surface density
`dF/dR / (2 pi R)` is **non-negative over the whole fitting domain** and that
`F` integrates to 1. `gompertz_log`, `loglogistic` and `richards` are written
as cumulative forms, so a negative density is representable and would be
silently unphysical.

**Notation**: use projected surface density `Sigma(R)` throughout. The modelled
quantity is a projected curve of growth, and exp48's objective differences both
sides rather than evaluating the model analytically.

| id | form | rationale |
|---|---|---|
| **P1** | Sérsic in R50 coordinates, `n` free | finite well-behaved tail; exp53 measured `sersic-q0` putting only 0.48% beyond 500 kpc against Moffat's 6.61% |
| **P2** | generalized sigmoid (`gompertz_log` / `richards`) | exp53's best added-light kernel of eleven (0.072 vs the fiducial 0.081), winning 10/13 metrics; `richards` fitted `nu = 0.345` |
| **P3** | Moffat with `gam` bounded so the convergence constraint holds | keeps the heavy tail the incumbent liked, but forces it to converge |
| **P4** | any of the above times a truncation at `C * R200c(t_j)` | enforces the constraint by construction rather than by penalty |

**Recommendation: P1 and P2 as the primary shootout, P3 as the incumbent's
honest descendant, P4 as the fallback if the penalty proves hard to fit.** All
are already implemented in `exp53/families.py` in a common `R50`
parameterization, which is what makes the comparison meaningful.

Two conventions to state and keep: the deposit is a **projected surface
density** (the data is a projected curve of growth, and exp48's objective
differences both sides rather than evaluating the model analytically); and the
profile is evaluated in `x = R/R50_j`, so the size law alone carries the units.

### 4.3 The deposit size law

| id | form | rationale |
|---|---|---|
| **S1** | `R50_j = f0 * R200c(t_j)` | one dimensionless number; "deposits form at a fixed fraction of the virial radius"; builds in `Mh^(1/3)` |
| **S2** | S1 `* (1+z_j)^b` | `b` measures the DEVIATION from virial scaling; **`b = 0` is a testable null**; the incumbent implies `b ~ -3` |
| **S3** | S2 with `log f_j = log f0 + slopes . cond(t_j)` | the conditioning layer, now acting on a physically normalized quantity |
| **S4** | fallback: keep the time power law, re-pivot to the mass-weighted median deposit time (~2 Gyr) | cosmetic but decorrelates intercept from slope and makes the base readable |

**S2a — the alternative halo scale (review §4.3).** `R200c` is the halo
BOUNDARY; the NFW **scale radius** `R200c/c200c` is where the density profile
turns over, and the two encode different formation-time dependence. Both are
physically readable and the comparison is nearly free:

```
S2   R50_j = f0 * R200c(t_j)          * (1+z_j)^b       the boundary scale
S2a  R50_j = fs * R200c(t_j)/c200c(t_j) * (1+z_j)^b     the NFW scale radius
```

**Fit both and report both.** Caveat measured here: `R200c` is available at all
73 snapshots but `c200c` at only 16, so S2a's support is limited by the
concentration record (§4.4) and it inherits that missingness.

**Recommendation: S2 as the base, S2a as a mandatory comparison, S3 for
conditioning, S4 only if `R200c` turns out to be unusable.** `Group_R_Crit200` is on disk at **all 73 snapshots**
(`exp46_highz_ridge/outputs/so_history.npz`, 96.4% finite), in ckpc/h; convert
with `/h/(1+z)`.

The gain from S1/S2 is not only physical readability: with `R200c` carrying the
mass scaling, the conditioning slope on `logMh` **measures the deviation from
virial scaling** instead of having to reproduce it, which is a far better-posed
estimation problem than the one that produced -0.059 per dex.

### 4.4 Conditioning, and the job transport used to do

`cond(t_j)` = standardized, epoch-local, evaluated at the deposit's own
formation time:

```
cond(t_j) = [ logMh(t_j),  dlogc(t_j),  f_form(t_j) ]
dlogc(t_j)  =  log10[ c200c(t_j) / c_Diemer19(Mh(t_j), z_j) ]      the EXCESS
f_form(t_j) =  t_half(t_j) / t_j                                   the TIMING
```

**`f_form` in words**: `t_half(t_j)` is the cosmic time at which the halo first
reached **half the mass it has at `t_j`**, so `f_form` answers *"of the mass
this halo has right now, how long ago did it acquire half of it, as a fraction
of its current age?"* Near 0 means it assembled early; near 1 means it only
recently reached half its current mass.

**Why not the existing `fz2` or `t50`.** Both divide by the FINAL mass
(`exp32/dataset.py`: `fz2 = Mh(z=2)/Mh(final)`, `t50` = when the halo reached
half its final mass), so computing either for a halo AT z=2 requires knowing
what it becomes by z=0.4. `f_form` is computed from the MAH truncated at `t_j`,
is dimensionless and bounded in (0,1) so standardization behaves and it does not
drift with epoch, is always defined for any halo in any simulation, and reduces
to `t50/t_obs` at `t_j = t_obs`.

**Expect it to gain LESS than `fz2` did, and check that it does.** Stage 1
measured `+fz2` adding 0.0004 dex at z=0.4 but 0.0148 at z=1.5 and 0.0126 at
z=2.0 — but part of that high-z gain IS the descendant leak: dividing by the
final mass effectively hands the regression `Mh(z=2)/Mh(z=0.4)`, which helps pin
`Mh(z=2)`. An epoch-local `f_form` cannot do that. **If `f_form` matches `fz2`
at high z, look for a leak before celebrating.**

All three are available: `logMh(t_j)` from the official DiffMAH curve;
`dlogc` from `exp54/outputs/halo_structure_history.npz` (16 snapshots,
interpolated in `ln(1+z)`, **0 outside their support** — "assume typical for its
mass and redshift", recorded rather than silently imputed, covering 57.8% of
deposits); `f_form` computed on the fly. The concentration excess rather than
raw `c200c` because Diemer19 is deterministic in `(M,z)` and therefore
redundant with `logMh(t_j)`, while the excess carries 100% of the measured
signal (Stage 1: 0.1102 either way).

**Concentration missingness is MASS-DEPENDENT and must be reported
(review §4.4, re-measured here).** Weighted by the E1 predicted stellar
deposits, valid concentration covers **94.4%** of the accumulated budget at
z=0.4 falling to **85.3%** at z=2 — but per-galaxy coverage at z=0.4 rises from
**90.8%** in the lowest halo-mass tercile to **96.1%** in the highest (at z=2:
78.2% -> 88.0%). Setting a missing excess to zero therefore introduces a small
**mass-dependent** effect, not a random one.

Every `E3`/`S3` result must therefore report: deposit-count, `dMh`-weighted and
predicted-stellar-mass-weighted coverage; coverage by halo-mass bin and epoch;
a fit restricted to the well-supported redshift range or a complete-case
sensitivity test; and a comparison with concentration omitted entirely.
**The missingness flag must never become a predictor in the deployable model.**

**The transport job.** exp53 measured that transport supplies the
halo-mass-dependent concentration, and this design deletes transport. The
replacement is the `logMh(t_j)` slope in S3: the size law's mass dependence must
now generate the tilt that transport used to. **This is the redesign's principal
risk and it gets a dedicated test** — re-measure exp53's `M(<148)` tilt across
halo-mass terciles and require it to be flat, not merely small.

### 4.5 Optional: two channels (COMPACT and EXTENDED)

Massive galaxies are two-phase objects: a compact component formed early and an
extended envelope acquired later.

**Name the channels `compact` and `extended`, NOT `in-situ` and `ex-situ`
(review §4.5).** The model splits ONE host main-branch mass increment at ONE
time into two radial profiles. That does not represent the in-situ/ex-situ
distinction, because ex-situ stars form in secondary progenitors and arrive
after an orbital delay — information the smooth main-branch `dMh` does not
carry. Claiming the physical labels would attach a mechanism the data cannot
support.

```
Delta M*_j  ->  f_in(Mh_j, z_j) * Delta M*_j   at  R50_in  = f0_in  * R200c(t_j)
            +  (1 - f_in)       * Delta M*_j   at  R50_ex  = f0_ex  * R200c(t_j)
```

exp36 tried a two-channel split under the pin and it did not decide the
question; under an absolute amplitude it is a different experiment. **Build it,
but only after the single-channel version passes Stage 0 and the objective is
settled** — it doubles the profile parameters and is the most likely source of
degeneracy.

**Known limitation to state, not solve**: DiffMAH is a smooth fit, so it removes
the merger granularity that actually controls accreted mass. Residual scatter is
expected even in a correct model, and `f_compact` is a population-average
statement.

### 4.6 A delivery-DELAY kernel, before any return to transport

Deleting the old transport term does **not** establish that stellar delivery is
instantaneous — exp30 already found evidence of a lagged stellar response to
halo accretion. Before reintroducing post-deposition transport, test a delay:

```
dM*_dep(t)  =  INTEGRAL^t dt'  eps_surv[H(t')] * dMh(t')/dt' * K_tau(t - t')
```

with `K_tau` a one-parameter kernel (exponential or gamma). This is strictly
cheaper than transport (one parameter, no per-deposit radial bookkeeping),
it is physically motivated by infall and orbital decay timescales, and
`tau -> 0` nests the instantaneous model exactly. **It does NOT fix the
monotonicity floor of §2.2** — a delayed sum of positive deposits is still
monotone — so it is a separate question from the survival term.

---

## 5. The objective

Unchanged from the parent plan §4.3, restated because it now applies to a model
that supplies its own amplitude:

```
A        =  log10 M*(<100 kpc)                    the amplitude
F(<R)    =  M*(<R) / M*(<100 kpc)                 the shape
L        =  lambda_A * L_A  +  lambda_F * L_F
L_A      =  ( A_model - A_data )^2                dex^2, per galaxy per epoch
L_F      =  Objective() applied to F              the production loss, on shapes
```

Two things this plan adds.

**The amplitude floor is epoch-dependent, and it is now measured.** Use
`lambda_A(z_k) = 1/sigma_A,floor(z_k)^2` with the Stage 1 best halo-only row:
**0.1047 / 0.1359 / 0.1417 / 0.1456 / 0.1744** dex at z = 0.4 / 0.7 / 1.0 / 1.5
/ 2.0. Without this the high-z epochs dominate `L_A` merely because amplitude is
harder there.

**Translating the record.** `L_F` on `F` is the production loss with the pin
moved from 500 to 100 kpc, and Stage 0 measured that offset: **-0.027233
(-14.69%)**, production loss 0.185429 -> 0.158196. Every shape loss in
exp38/40/47/48/53 converts with it.

**The SHAPE term needs a measured scale too (review §5.1).** Normalizing only
`L_A` by its benchmark and leaving `L_F` raw is an implicit, undeclared weight.
The central choice must be one of:

- standardize BOTH terms by their cross-validated benchmark errors — for `L_F`
  that is the incumbent's own shape loss, measured in Stage 0 as **0.158196**
  at the 100 kpc anchor; or
- declare equal scientific priority AFTER normalizing each term to its
  baseline.

**Weight scan**: `lambda_A/lambda_F` over {1/4, 1, 4} around that choice,
reported. The weighting must be an evidenced choice.

**Amplitude and shape are separated in the OBJECTIVE but the PARAMETERS remain
coupled** — changing deposit size changes the fraction inside 100 kpc, hence
the predicted `A`. The separation prevents a coherent amplitude error from
masquerading as 24 shape errors; it does not make the fit block-diagonal, and
the identifiability report must not assume it does.

**Mass growth needs no extra LOSS term, but it IS a promotion metric**
(review §5.2). Scoring `L_A` at five epochs constrains the mean trajectory; it
does **not** guarantee the correct joint distribution of growth residuals. The
mass-growth planes and the cross-epoch residual correlation stay promotion
criteria. **If they fail while the five marginal scores pass, add a growth term
THEN, with the failure documented — do not assume it is redundant.** Stage 2
measured why this matters: `Mtot` growth energy ratio 0.27-0.33 pinned versus
1.22-1.60 honest.

---

## 6. Identifiability protocol — mandatory, every fit

Agreed with the user (2026-08-22): **do not cap the parameter count; measure
identifiability instead.** Overfitting is not the binding constraint —
2397 x 5 x 24 = 287,640 data points, and the 3-parameter law measured in-sample
0.1505 against CV 0.1506, a zero generalization gap. (Caveat: the 24 radii are
~93% correlated and the profile is 2-3 dimensional, so the effective constraint
count is nearer 36,000. Still far above any parameter count contemplated here.)

**But "the fit will decide" holds only for identifiable parameters, and the loss
cannot tell you which those are.** Three counterexamples from this project:
exp53's three families re-railing at a widened bound with losses converging to
within 1e-4 ("nothing chooses the answer"); exp49's six conditioning slopes
differing by up to 0.022 between fits agreeing to 0.015 on the base; and this
branch's cross-term reaching the *same loss* at `a0 = 26.8, mu = 92.5`.

Every fit ships:

1. **Hessian at the optimum**, eigendecomposed — but **only after parameter
   scaling or an unconstrained transform** (review §5.4). Finite-difference
   eigenvalues depend on units and are unreliable near bounds, and several of
   these parameters are bounded.
2. **An SVD of the scaled residual Jacobian** (equivalently the Gauss-Newton
   curvature) alongside the Hessian. It is cheaper, better conditioned, and does
   not require second differences.
3. **Effective parameter count** = singular values above a noise-set threshold.
4. **Profile likelihood** for every parameter whose value will be quoted: scan
   it, re-optimizing the rest. A flat profile means unidentified.
5. **Stability across starts, CV folds, and galaxy bootstraps.** A parameter
   whose local profile is sharp but whose bootstrap distribution is broad is not
   identified in any useful sense.
6. **Rail check by ESCAPE, not by loss** — exp53's lesson: one cell gained 0.37%
   and was flagged benign while still sitting on the new bound.
7. **Profiles of DERIVED quantities**, not only raw parameters:
   `R50/R200c`, `R90/R200c`, deposited mass by redshift bin, and aperture
   growth. These are often identified where their constituents are not, and
   they are what the science actually uses.
8. **Eigenvector composition**, so the report states which COMBINATION is
   determined rather than which parameter.

**Quote a physical parameter only if BOTH its local profile and its resampling
distribution are informative.** Otherwise quote the identified derived
combination and say so.

**The rule**: a parameter that fails may remain in the model, but **its value is
never quoted or interpreted**. Practical corollary: flat directions make
Nelder-Mead wander, which is why the sigmoid family's multi-start spread is
1.7e-3 against 1e-8 elsewhere. Multi-start stays mandatory.

---

## 7. Validation sequence

Restructured on the review's recommendation: **a pre-fit Stage 3.0 comes first**,
because two things that change the model's physical meaning must be settled
before any parameter is fitted.

### 7.0 Stage 3.0 — pre-fit contracts and floors (NO FITTING)

1. **Freeze the mass-budget contract in code.** Contract B, `eps_surv`,
   truncation at `C * R200c(t_j)`. Assert `F(R_trunc) = 1` and `F(R50) = 0.5`
   numerically for every family. Re-run Stage 0's poisoned-data test against
   the new forward call.
2. **Publish the monotonic-addition ceiling** (§2.2) as a named second ceiling
   alongside the oracle-amplitude ceiling, per epoch and per radius, in the
   objective's own units so it can be compared with a fitted loss.
3. **Split the floor into signal and noise.** The 0.0227 dex is an upper bound
   on what a deposition-only model must give up ONLY if all of it is real. Test
   how much is irreducible measurement fluctuation: does the violation
   correlate with merger activity in the MAH, with projected axis ratio
   (`q`, `s` are now in the halo-structure record), or is it consistent with
   the profile's own `cog_sigma_dex`? **This measurement decides whether a
   survival term is worth building** — stellar mass loss alone accounts for
   only ~1/5 of it (§2.2).
4. **Define the tail diagnostics** of §4.2 and run them on the incumbent
   Moffat, Sersic and the sigmoid family, before any constraint is imposed.
5. **Build endpoint-preserving MAH shuffles** (§7.3 below).

**Gate**: Stage 3.1 does not start until 1-3 are done and the floor is on the
record.

### 7.1 Stage 3.1 — the minimal absolute model, small and two-ended

E1 x one finite-tail profile x {S2, S2a}, on a **small mass-stratified sample**,
at **z=0.4 AND z=2 (never z=0.4 alone**, review §5.5). Must pass Stage 0's
contract, mass conservation, and the reparameterization assertions. Ships the
full identifiability report. This is the smallest thing that can be right.

### 7.2 Stage 3.2 — controlled factorial over the axes

E1/E2/E3/E4/E5 x P1/P2/P3 x S2/S2a/S3, still on the small two-epoch sample,
ranked **by the judge, never by loss**. Report which candidates the tail
diagnostics and the truncation bind. Carry a SHORT LIST forward; do not take
the full five-epoch fit until the small-scale factorial is stable.

### 7.3 Stage 3.3 — five epochs, with three null controls

The split objective with per-epoch amplitude floors and the weight scan.
Report amplitude, normalized shape, mass growth, cross-epoch residual
coherence, **distance to the monotonic floor**, and radial halo-mass trends.

**Three distinct controls (review §5.3), not one:**

1. epoch-local `logMh(z_k)` only;
2. the richest halo-only benchmark (Stage 1's best row);
3. an **endpoint-preserving** MAH-shape shuffle. The current shuffle matches on
   `Mh(z=0.4)` and therefore ALSO changes `Mh(z_k)` at higher redshift,
   conflating history shape with epoch-local mass — a caveat this plan already
   carried and the review sharpened. Match or residualize at **fixed
   `Mh(z_k)`**, and for tracked descendants at fixed `Mh(z=0.4)`.

**The transport-replacement test, corrected.** exp53's `M(<148)` halo-mass tilt
must be re-measured — but as a **held-out slope with a bootstrap interval, not
a point estimate required to equal zero** (review §5.5). And at **several
radii** — 5, 10, 50-100, and 100-148 kpc — since one outer aperture cannot
establish that the replacement for transport is radially correct.

### 7.4 Stage 3.4 — diagnose before adding anything

Classify the residual BEFORE choosing among: a second channel (§4.5), a
delivery-delay kernel (§4.6), a survival term, or constrained radial evolution.
Each addresses a different failure and only §2.2's floor decides between the
last two.

### 7.5 Stage 3.5 — deployment scoring, with the selection stated

Three problems scored separately: transfer to new haloes at a fitted epoch;
redshift extrapolation for tracked descendants; epoch-local prediction for an
independent halo catalogue.

**The third CANNOT be validated with this sample (review §5.6).** Epoch-local
inputs remove future information from each prediction, but they do not remove
the **sample selection**: every high-redshift object here is a progenitor of a
z=0.4-selected population. Validating deployment to an independently selected
z=2 halo catalogue needs a correspondingly selected training/validation sample
or an explicit selection correction. **Stage 3.5 must report the third use case
as UNVALIDATED unless that is done** — this is a limitation of the whole
program, not of this stage.

### 7.6 Figures — required, not cosmetic

The exp54 folder currently contains **no figures**, the main documentation gap
in otherwise careful Stages 0-2. Before Stage 3.1 fitting, produce at least:

1. Stage 1 truth-vs-predicted `M*(<100)` and residual-vs-truth, per epoch, for
   the best regression, E1/E2/E3, and the shuffled controls;
2. amplitude scatter and CRPS versus redshift with fold uncertainty;
3. Stage 2 measured vs modelled CoGs and radial residuals for `oracle-500`,
   `hybrid-mean` and a population draw;
4. the total-mass growth plane before and after unpinning;
5. tail-enclosed fraction versus radius for the incumbent and every candidate
   family;
6. **the monotonic-violation fraction as a radius-by-epoch heatmap.**

These are the fastest way to see mass-dependent regression to the mean, radial
compensation, high-redshift failure, and whether the redesigned family is
approaching the monotonic floor.

## 8. Pre-registered predictions

1. **The convergence constraint will bind on the Moffat and not on the
   Sérsic.** `moffat-q0` puts 6.61% of deposits beyond 500 kpc against
   `sersic-q0`'s 0.48%; the constraint formalizes that difference. If it binds
   on every family, `C = 3` is too tight and must be re-derived, not relaxed.
2. **`b` in S2 will be significantly negative but far smaller in magnitude than
   the incumbent's implied -3.** The incumbent had to use the time exponent to
   do a job that `R200c(t_j)` now does directly.
3. **The `logMh` conditioning slope in S3 will be small**, because `R200c`
   already carries `Mh^(1/3)`. A slope near zero here is SUCCESS, the opposite
   of its meaning in the incumbent.
4. **The efficiency spline (E4) will not find a significant peak** in
   `ln(1+z)`, and its identifiability report will show the peak's location as a
   flat direction. If it DOES find one, that is a genuine discovery and the
   monotone forms are wrong.
5. **Deleting transport will re-open exp53's halo-mass tilt**, and the S3 mass
   slope will be what closes it. If the tilt survives a free S3, transport was
   doing something the size law cannot express, and the two-channel split (§4.5)
   becomes necessary rather than optional.
6. **The redesign will use more MAH-shape information than the incumbent law's
   0.0016 dex** under the shuffle control, but will not reach the linear
   regression's 0.0098 dex. Reaching it would mean the physical form has become
   as expressive as an unconstrained fit, which would be surprising.
7. **E5's peak mass `M1` and low-mass slope `beta` will be UNIDENTIFIED** on
   this sample (logMh 13.0-14.4, entirely above the peak). If they are not, the
   fit is absorbing something else through them.
8. **`f_form` will gain less than `fz2` did at z >= 1.5**, because part of
   `fz2`'s high-z gain is the descendant leak. If they match, look for a leak.
9. **The best deposition-only model will land ABOVE the monotonic-addition
   floor** (§2.2) but within a factor of a few of it at small radii. If it lands
   AT the floor, the deposition-only family is exhausted and the remaining error
   is structural. If it lands far above, the forms are still the limitation and
   §2.2 is not yet binding.
10. **A delivery-delay kernel (§4.6) will NOT reduce the monotonic-floor
    distance**, because a delayed sum of positive deposits is still monotone. If
    it does, something in the implementation is not a pure delay.
11. **S2a (the NFW scale radius) will fit at least as well as S2 (the halo
    boundary) at low z and worse at high z**, tracking the concentration
    record's collapsing support above z ~ 5.

---

## 9. Decisions taken

1. **`C = 3`, enforced by TRUNCATION** (§4.2), with `C` scanned over {1,2,3,5}
   and the TNG evidence recorded. Truncation makes every family's total mass
   well-defined, which the incumbent Moffat's was not.
2. **`f_form = t_half(t_j)/t_j`** replaces `fz2`/`t50` (§4.4), which are
   descendant-referenced. Whether it earns its place is now a measurement in
   3b, with the leak caveat pre-registered.
3. **Build and test E5 now** (§4.1), with the pre-registered expectation that
   its peak mass and low-mass slope come out unidentified on this sample.

**NOTE — reviewer/user disagreement, resolved in favour of the user.** The
review recommends DEFERRING E5 "because the sample does not cross the SHMR
efficiency peak". That is the same reasoning behind prediction 7. The
difference is that building it turns an assumption into a measurement: the
identifiability report will DEMONSTRATE that `M1` and `beta` are unconstrained
rather than asserting it, at the cost of one extra factorial cell. If the
report shows them identified, that is a red flag for E5 absorbing something
else — which is information the deferral would not have produced.

### 9.1 Decisions taken in response to the review (2026-08-23)

4. **Contract B, made well-posed by truncation** (§2.4); `eps` renamed
   `eps_surv`, the **effective surviving-central mass fraction**, explicitly
   absorbing baryon conversion, mass loss, stripping, central assignment and
   delivery. Contract A (the Stage 1 scoreboard) is retained as the statistical
   benchmark and is **not** the Contract-B floor.
5. **A pre-fit Stage 3.0** (§7.0) that publishes the monotonic-addition ceiling
   and splits it into signal and noise before any parameter is fitted.
6. **`R99 <= 3 R200c` is a working default, not a physical law** — report the
   full tail diagnostics first, then derive `C` (§4.2).
7. **Truncated families must be reparameterized** so the advertised `R50` is
   the actual post-truncation half-mass radius, asserted numerically (§4.2).
8. **`S2a`, the NFW scale radius `R200c/c200c`, is a mandatory comparison**
   against the halo boundary `R200c` (§4.3).
9. **The two channels are named `compact` and `extended`**, not in-situ and
   ex-situ (§4.5), and a **delivery-delay kernel** is tested before any return
   to transport (§4.6).
10. **Concentration missingness is mass-dependent and must be reported**
    weighted three ways, by halo-mass bin and epoch, with a complete-case
    sensitivity test (§4.4).
11. **Identifiability broadens** beyond a local Hessian to scaled-Jacobian SVD,
    bootstrap stability, and derived-quantity profiles (§6).
12. **Screening happens at z=0.4 AND z=2 on a small mass-stratified sample**;
    the halo-mass tilt is a held-out slope with a bootstrap interval at several
    radii, not a point estimate required to be zero (§7.1-7.3).
13. **Deployment use case 3 is reported UNVALIDATED** until the progenitor
    selection is addressed (§7.5).
14. **Figures are required before Stage 3.1 fitting** (§7.6).

---

## 10. What success looks like

Under the agreed criterion — **deployability, not victory** — Stage 3 succeeds
if it produces a model that:

- passes Stage 0's poisoned-data test, so it runs on an N-body halo catalogue;
- is scored against **both** ceilings — the oracle-amplitude shape ceiling and
  the monotonic-addition floor of §2.2 — so its remaining error is attributed
  rather than merely reported;
- conserves mass exactly and keeps its deposits inside a physically sensible
  radius;
- predicts stellar mass with an amplitude error near the Stage 1 halo-only
  floor (0.1047 dex at z=0.4), with **honest** growth (`Mtot` growth energy
  ratio reported, not pinned to 0.3 by construction);
- has an identifiability report for every quoted parameter;
- and whose remaining defects are stated in physical terms rather than absorbed
  by a normalization.

It does **not** have to beat the statistical emulator. The emulator predicts
aperture masses; this predicts a galaxy, its profile, and its history, from a
halo — and can be wrong in ways that teach something.
