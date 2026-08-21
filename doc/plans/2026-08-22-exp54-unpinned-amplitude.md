# exp54 — remove the truth pin: predict the stellar mass, don't copy it

**Status: PLANNED.** Successor to exp53. This is the beginning of a **major
revision of hongshao** and, if it succeeds, it **changes the fiducial model**.
Proceed carefully and stage-gate it.

Sources: this session's investigation (exp53 branch), the user's formulation
(2026-08-22), and the independent review at
`hongshao_internal_review_2026_08_16/docs/internal/2026-08-22-truth-pinned-normalization-review.md`
(sections 7, 8, 10, and the addendum 11).

---

## 1. The problem, stated exactly

The adopted kernel normalizes every galaxy's profile so that

```
M_model(<500 kpc)  ==  M_TNG(<500 kpc)      per galaxy, per epoch
```

Verified to machine precision: the residual is **2.2e-16**. Two consequences.

1. **The Mh-M\*(<500) relation is inserted, not predicted** — with its scatter
   and its evolution. The deposit weights are normalized to sum to 1
   (`dM = w/w.sum()`), so the kernel carries **no absolute stellar mass at all**;
   100% of the mass and 100% of the mass growth come from the pinned value.
2. **The stated use case cannot be executed.** Given only an N-body halo there
   is no `M_TNG(<500)` to copy — that quantity is what the model is for. Supply
   an external SHMR and *that* relation, not hongshao, is assigning the mass.

The current model is therefore an **oracle-amplitude shape model**: *given a
halo and its already-known stellar mass, predict the radial distribution*. That
is a legitimate and useful experiment. It is not an Ultimate SHMR.

## 2. What is NOT the problem

Distinguish two operations that are easy to conflate (review 11.3):

- **Normalized representation** — writing a profile as amplitude x dimensionless
  shape. Statistically useful, keeps the scientific roles separate, and is
  **retained** in the recommended design:
  `log M*(<R) = A_predicted(H) + log F_predicted(<R | H)`.
- **Truth pinning** — setting that amplitude to the true mass of the very
  galaxy being predicted. This is what must go.

Removing the pin does not require abandoning normalized shapes.

## 3. Evidence gathered before planning (all measured this session)

| finding | number |
|---|---|
| the pin is an identity | residual 2.2e-16 |
| pinned quantity is extrapolated | 7.2% of `M(<500)` lies beyond the last measured radius at z=0.4 (16-84: 3.5-14.4%) |
| its form systematic | power-law vs exponential tail differ by 0.021 dex median at z=0.4, and that difference **tilts with halo mass by -0.021 dex** — comparable to exp53's decisive signal |
| over-reach is restored free | mass beyond 500 kpc, median per galaxy: moffat-qfree 4.00%, moffat-q0 6.61%, gompertz_log-q0 5.40%, sersic-q0 0.48% |
| mass assembly is unscored and wrong | fiducial's own accumulation grows x1.05 per galaxy where the truth grows x2.62 (**+0.398 dex** correction); deposition-only variants -0.15 dex |
| a global efficiency at the FITTED window fails | slope-one scatter **0.339 dex**, worse than the 0.280 dex do-nothing baseline |
| ... but recovers if the window adapts | best in scan mu=1.1, sig=0.5 -> **0.137 dex**; reviewer's optimum mu=1.188, sig=0.427 -> 0.134 dex |
| baselines it must beat | catalog z=0.4 logMh alone **0.140 dex**; epoch-matched DiffMAH logMh **0.120 dex** (reviewer); logMh+c200c+fz2 **0.119 dex**; adding `S` to that gains only 0.001 dex |
| amplitude would swamp the objective | a 0.137 dex amplitude error alone costs **2.05x the entire current loss** |
| removing the pin closes the horizon trap | un-normalized, mass beyond the aperture is simply absent and the fit pays for it |

**Two corrections to earlier claims in this session, both confirmed against the
reviewer**: (i) "the model already contains an SHMR" was wrong — `S` alone
(0.161 dex) is WORSE than halo mass alone (0.140 dex) and adds ~0.001 dex to a
proper baseline; (ii) the objective's RMS over radii does NOT multiply a
coherent amplitude error by 24 — it leaves it at full strength while diluting a
localized shape feature. The practical conclusion (amplitude dominates) is
unchanged.

**A defect nobody had noticed**: the conditioning vector is built ONCE from
z=0.4 descendant properties and applied at every epoch. Fine for tracking
progenitors of a selected sample; impossible for populating an arbitrary z=2
halo catalogue.

## 4. Design

### 4.1 Preserve the current model under an honest name

Keep the pinned kernel as `oracle_amplitude_shape_model`. It answers how well a
radial prescription can work when the reference mass is known perfectly, and it
is a legitimate ceiling. **Its absolute-mass QA values must never again be
quoted as forward-model scores.**

### 4.2 The reference aperture must be MEASURED

Use `M*(<100 kpc)` or `M*(<148 kpc)`, chosen to match the intended
observational surface-brightness limit — not the extrapolated `M(<500)`. If a
500 kpc or asymptotic quantity is scientifically wanted, its tail uncertainty
becomes part of the target model rather than being treated as exact.

### 4.3 Two experiments, in order

**exp54a — the predicted-amplitude hybrid (lower risk, do first).**
`log M*(<R) = A(H,z) + log F_physical(<R | H,z)`, with `A` supplied by the
existing statistical emulator, which already gives a portable halo-only
prediction with calibrated scatter. Pass a DRAW, not the TNG value. **Amplitude
and shape residuals must not be sampled independently** — preserve their
covariance via the existing annular-mass covariance or a joint residual model.

**exp54b — the absolute deposition law (more physical, separate).**
Delete `dM = w/w.sum()` and the final rescale. Then
`dM_i = eps_i * dMh_i` with an efficiency law that must be benchmarked, not
assumed; a reasonable massive-halo starting point is

```
log eps_i = a0 + a_M [log Mh_i - Mpiv] + a_z ln(1+z_i)
```

optionally times the existing temporal window. Consider separate in-situ and
accreted channels: late ex-situ mass is not determined by the smooth main-branch
`dMh` — it depends on progenitor SHMRs, merger ratios, stripping and survival,
and DiffMAH smooths away exactly that granularity. **Residual scatter is
expected even in a correct model.**

### 4.4 The objective must score amplitude and shape separately

```
L = lambda_A * L_A  +  lambda_F * L_shape
L_A     = [log M*_ref(model) - log M*_ref(data)]^2
L_shape = on normalized CoGs or differential annuli
```

Weights set by scientific priority and the measured scale of each target, NOT
by the number of radial samples. Equivalent decorrelated targets (one anchor
mass plus differential annuli, or profile coefficients) are acceptable.
`hongshao/objective.py` already has selectable axes; this needs a new amplitude
term, not a rewrite.

### 4.5 Make the conditioning epoch-local

Replace the z=0.4 descendant vector with quantities defined at or before the
epoch being predicted: current halo mass, current concentration, and a
formation fraction relative to the current halo mass. A progenitor-track mode
may deliberately condition on descendant properties, but must be labelled.

## 5. Validation sequence (stage-gated; do not skip ahead)

**Stage 0 — contract and bookkeeping.**
A forward call accepts halo quantities ONLY; assert that no array derived from
stellar data enters prediction. The profile must carry physical mass units
before comparison. Report mass inside the measured horizon, between the horizon
and the model boundary, and beyond the boundary — **without renormalizing any
of them away**. Verify mass conservation numerically.

**Stage 1 — amplitude baselines at ONE epoch (z=0.4).**
Predict the reference mass from: (1) final halo mass only; (2) DiffMAH only;
(3) DiffMAH + concentration; (4) the pre-normalization deposition sum `S`;
(5) combinations. Galaxy-level cross-validation; report the mean relation, RMS
or CRPS, residual scatter vs halo mass, and coverage. **Include shuffled-MAH
controls within narrow final-mass bins before claiming any assembly
information** — exp53 measured `S` adding only 0.001 dex over
logMh+c200c+fz2, so the burden of proof is real.

**Stage 2 — predicted amplitude + the existing physical shape.**
Replace the true anchor with out-of-fold predictions or draws. **Refit all
physical parameters.** Compare absolute aperture masses, normalized shape,
mass-size and two-aperture planes, population scatter and covariance, central
and outer residuals, and material beyond the horizon.

**Stage 3 — absolute deposition.** Fit efficiency and profile laws jointly with
no per-object rescaling. Start small, one epoch. Proceed only after the
absolute SHMR and mass conservation pass.

**Stage 4 — multi-epoch.** Score three DIFFERENT deployment problems
separately: transfer to new haloes at a fitted epoch; redshift extrapolation
for tracked descendants; epoch-local prediction for an independent halo
catalogue.

## 6. Predictions to check (pre-registered)

1. **The efficiency window will move.** Shape prefers mu=1.50, sig=0.22;
   amplitude prefers mu~1.19, sig~0.43. A joint fit must trade them off, so the
   fitted physics changes. If the window does NOT move, the amplitude term is
   not biting.
2. **Broad profiles will pay for over-reach.** `moffat-q0` puts 6.6% of its
   deposits beyond 500 kpc against `sersic-q0`'s 0.5%; un-normalized that is a
   direct cost, so **the profile-family ranking may change**.
3. **Transport and deposition can no longer trade against a free epoch-by-epoch
   amplitude.** exp52's redistribution finding must be re-derived, not assumed.
4. **The efficiency-outer-tail degeneracy will NOT vanish** (review 11.6).
   Removing the pin closes the per-object version, but a global or
   halo-dependent efficiency can still absorb the population-average fraction
   beyond the horizon. **Measure this explicitly rather than assuming it is
   gone** — it is the first thing exp54 should check after Stage 0.

## 7. Consequences to expect

Every absolute QA number in the program — exp38, exp40, exp47, exp48, exp53 —
is quoted under the pinned condition and must be **re-baselined**. Measured
example: the tier 2b plane at z=0.4 moves 2.1 -> 2.9 under a realistic 0.137 dex
amplitude error, and -> 3.6 at 0.20 dex. The profile-family ranking, the
efficiency window, the transport strength and the over-reach behaviour all
become open questions again.

**exp53 remains valid for the narrow question it answered**: with the correct
per-galaxy amplitude supplied, which radial kernel distributes it best. It does
not decide which family best predicts absolute stellar profiles from haloes.

## 8. Decision record (user + reviewer, 2026-08-22)

1. Freeze exp53 as a conditional-shape boundary study.
2. Reclassify the pinned kernel as an **oracle-amplitude shape model**, not the
   deployable fiducial Ultimate SHMR.
3. Removal of per-galaxy truth normalization is the **first structural goal**
   of the next physical-model experiment.
4. Start with the modular predicted-amplitude hybrid, using the existing
   absolute-mass emulator as both baseline and amplitude provider.
5. Treat the absolute deposition-efficiency model as a separate, more physical
   experiment, measured against final-halo-mass-only and shuffled-MAH controls.

> A reference stellar mass may be defined at any scientifically defensible
> aperture, but in a forward model it must be PREDICTED from halo information,
> with calibrated scatter, rather than copied from the galaxy being predicted.
