# exp54 — remove the truth pin: predict the stellar mass, don't copy it

**Status: PLANNED (revision 2, 2026-08-22).** Successor to exp53. The beginning
of a **major revision of hongshao**; if it succeeds it **changes the fiducial
model**. Stage-gate it.

Sources: the exp53 branch investigation; the user's formulation (2026-08-22);
the independent review at
`hongshao_internal_review_2026_08_16/docs/internal/2026-08-22-truth-pinned-normalization-review.md`
(sections 7, 8, 10, addendum 11); and **eight new measurements made while
refining this plan** (§3.2), which changed three of revision 1's structural
conclusions.

---

## 0. What changed in revision 2, and why

Revision 1 was written before anything was measured about the *unpinned*
amplitude. Four things it assumed are now known to be false or backwards.

1. **The absolute deposition law is not the risky, far-off Stage 3. Its
   amplitude already works.** The exact 3-parameter form revision 1 proposed
   reproduces the epoch-matched halo-mass baseline at every epoch, beats it at
   z >= 1.0, carries <= 0.004 dex bias, and cross-validates with no
   overfitting (§3.2.D). Revision 1 called this "more physical, separate,
   benchmark it, don't assume it". It has now been benchmarked and it holds.
2. **exp54a's "refit all physical parameters" is a no-op** under the split
   objective revision 1 also asks for. The two requirements are incompatible;
   §4.3 resolves it. Pre-registered prediction (i) — "the efficiency window
   will move" — **cannot be tested in exp54a at all.**
3. **Epoch-local conditioning is not a portability nicety, it is the single
   largest measured effect in this plan.** Descendant `logMh(z=0.4)` predicts
   `M*(<100)` at z=2 with 0.304 dex; epoch-matched DiffMAH `logMh(z_k)` gives
   0.182 dex — a **1.7x** improvement (§3.2.B).
4. **The reference-aperture choice is statistically free.** The amplitude
   baseline is 0.1209 / 0.1203 / 0.1202 dex at `M(<100)` / `M(<148)` /
   `M(<500)`. Revision 1 left the choice open as though it mattered for
   accuracy; it does not, so it can be decided on scientific meaning alone
   (§4.2).

Everything in revision 1 not contradicted here stands.

---

## 1. The problem, stated exactly

The adopted kernel normalizes every galaxy's profile so that

```
M_model(<500 kpc)  ==  M_TNG(<500 kpc)      per galaxy, per epoch
```

Verified to machine precision: residual **2.2e-16**
(`experiments/exp53_deposition_only/kernel.py:233`,
`out.append(m[:-1] * (g["m500"][k] / m[-1]))`). Two consequences.

1. **The Mh-M\*(<500) relation is inserted, not predicted** — with its scatter
   and its evolution. The deposit weights are normalized to sum to 1
   (`kernel.py:212`, `return w / w.sum()`), so the kernel carries **no absolute
   stellar mass at all**; 100% of the mass and 100% of the mass growth come
   from the pinned value.
2. **The stated use case cannot be executed.** Given only an N-body halo there
   is no `M_TNG(<500)` to copy — that quantity is what the model is for. Supply
   an external SHMR and *that* relation, not hongshao, assigns the mass.

The current model is an **oracle-amplitude shape model**: *given a halo and its
already-known stellar mass, predict the radial distribution.* Legitimate and
useful. Not an Ultimate SHMR.

## 2. What is NOT the problem

Two operations that are easy to conflate (review 11.3):

- **Normalized representation** — writing a profile as amplitude x
  dimensionless shape. Statistically useful, keeps the scientific roles
  separate, and is **retained**:
  `log M*(<R) = A_predicted(H,z) + log F_predicted(<R | H,z)`.
- **Truth pinning** — setting that amplitude to the true mass of the very
  galaxy being predicted. This is what must go.

Removing the pin does not require abandoning normalized shapes.

---

## 3. Evidence

### 3.1 Measured during exp53 (carried forward unchanged)

| finding | number |
|---|---|
| the pin is an identity | residual 2.2e-16 |
| pinned quantity is extrapolated | 7.2% of `M(<500)` lies beyond the last measured radius at z=0.4 (16-84: 3.5-14.4%) |
| its form systematic | power-law vs exponential tail differ by 0.021 dex median at z=0.4, tilting with halo mass by -0.021 dex — comparable to exp53's decisive signal |
| over-reach is restored free | mass beyond 500 kpc, median per galaxy: moffat-qfree 4.00%, moffat-q0 6.61%, gompertz_log-q0 5.40%, sersic-q0 0.48% |
| mass assembly is unscored and wrong | the fiducial's own accumulation grows x1.05 per galaxy where the truth grows x2.62 (**+0.398 dex** correction); deposition-only variants -0.15 dex |
| amplitude would swamp an unsplit objective | a 0.137 dex amplitude error alone costs **2.05x** the entire current loss |
| removing the pin closes the horizon trap | un-normalized, mass beyond the aperture is simply absent and the fit pays for it |

Two exp53 corrections that stand: `S` alone (0.161 dex) is WORSE than halo mass
alone and adds ~0.001 dex to a proper baseline; and the objective's RMS over
radii does NOT multiply a coherent amplitude error by 24 — it leaves it at full
strength while diluting a localized shape feature.

### 3.2 Measured while refining this plan (2026-08-22)

All on the production sample, n = 2397, target `log10 M*(<100 kpc)`, log-log
interpolated on the measured 24-point grid. Predictor scatters are 5-fold
galaxy-level CV on identical folds unless marked in-sample.

**Reproduce with** `doc/plans/2026-08-22-exp54-evidence-probe.py` (~4 min; a
planning probe, not exp54 — it writes nothing and touches nothing in
`hongshao/`). Do not trust the tables below without re-running it.

**A. The anchor aperture is statistically free.** CV scatter from `logMh(z_k)`
alone, per epoch:

| target | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| `M*(<100)` | 0.1209 | 0.1459 | 0.1488 | 0.1552 | 0.1820 |
| `M*(<148)` | 0.1203 | 0.1457 | 0.1478 | 0.1539 | 0.1809 |
| `M*(<500)` (extrapolated) | 0.1202 | 0.1458 | 0.1463 | 0.1518 | 0.1790 |

The spread across apertures never exceeds 0.0034 dex. `M(<100)/M(<148)` median
runs 0.958 (z=0.4) -> 0.993 (z=2.0). The apertures are near-interchangeable;
pick on meaning (§4.2).

**B. Epoch-local conditioning is worth 1.7x at z=2.**

| predictor | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| do-nothing (population sigma) | 0.2562 | 0.2754 | 0.2872 | 0.3074 | 0.3657 |
| descendant `logMh(z=0.4)` (catalog) | 0.1401 | 0.1815 | 0.2031 | 0.2392 | **0.3039** |
| epoch-matched DiffMAH `logMh(z_k)` | 0.1209 | 0.1459 | 0.1488 | 0.1552 | **0.1820** |
| + `c200c` + `fz2` | 0.1109 | 0.1366 | 0.1403 | 0.1437 | 0.1764 |

This reproduces the record's three quoted baselines (0.140 / 0.120 / 0.119) and
adds the epoch dependence they were missing. **Note the achievable amplitude
accuracy degrades by 50% from z=0.4 to z=2** — the objective must use an
epoch-dependent amplitude floor or the high-z epochs will dominate `L_A`.

**C. The shape-fitted window is a bad amplitude model, and it is bad in a
specific way.** At the adopted theta (mu = 1.5221, sig = 0.2352):

| epoch | `log S` slope-one | free-slope | fitted slope | `logMh(z_k)` |
|---|---|---|---|---|
| z=0.4 | 0.2173 | 0.1659 | **0.582** | 0.1209 |
| z=0.7 | 0.2090 | 0.1712 | 0.643 | 0.1459 |
| z=1.0 | 0.1867 | 0.1597 | 0.711 | 0.1488 |
| z=1.5 | 0.1650 | 0.1501 | 0.796 | 0.1552 |
| z=2.0 | 0.1814 | 0.1800 | 0.931 | 0.1820 |

A slope-one efficiency law is badly wrong at low z: `S` scales with `M*` at
slope 0.58. The best joint slope-one *pure-window* fit I could scan reaches
0.1631 pooled and railed at the scan edge (mu = 2.55, sig = 1.00) — still worse
than `logMh` alone at every epoch. **A window-only efficiency cannot be an
SHMR.** The missing ingredient is the halo-mass dependence, and the fitted
slope tells you its size: `a_M ~ -0.29` would bring 0.71 to 1.0.

**D. The plan's own efficiency law closes the gap — 3 parameters, no
per-galaxy freedom, no renormalization.** Fitting

```
log10 eps_i  =  a0  +  a_M [logMh(t_i) - 13.5]  +  a_z ln(1 + z_i)
M*_model(<100, z_k)  =  SUM_{i : t_i <= t_k}  eps_i * dMh_i
```

gives **a0 = -2.498, a_M = -0.190, a_z = 0.355**:

| epoch | law (5-fold CV) | law bias | `logMh(z_k)` | shuffled-MAH control |
|---|---|---|---|---|
| z=0.4 | 0.1231 | +0.0005 | 0.1209 | 0.1324 |
| z=0.7 | 0.1463 | -0.0015 | 0.1459 | 0.1741 |
| z=1.0 | 0.1473 | -0.0006 | 0.1488 | 0.2010 |
| z=1.5 | 0.1504 | +0.0035 | 0.1552 | 0.2530 |
| z=2.0 | 0.1802 | -0.0020 | 0.1820 | 0.3287 |

Pooled in-sample 0.1505, pooled CV **0.1506** — 3 parameters on ~12000 points,
so overfitting is nil. It **beats** the epoch-matched baseline at z >= 1.0. Its
bias is <= 0.004 dex at every epoch, against the pinned model's +0.398 dex
assembly correction: **the absolute law gets the mass growth right, which is
the single thing the pin was hiding.** Implied efficiency at logMh = 13.5:
0.0032 at z=0, 0.0078 at z=2 — physically sane.

The **shuffled-MAH control** (whole MAHs permuted within 60 narrow
final-halo-mass bins) is passed decisively and the margin grows with redshift:
0.180 vs 0.329 at z=2. Caveat to state when reporting it: swapping a whole MAH
also swaps `Mh(z_k)`, so at high z the control partly re-measures "epoch-local
halo mass matters". It is the right control, but it is not *only* an
assembly-information control.

**E. Two residual structures the 3-parameter law leaves behind. Both were
tested, and both name a term that works.**

- **Mass x redshift.** The CV residual-vs-`logMh(z_k)` slope runs **-0.09
  (z=0.4) -> +0.14 (z=2.0)**: one `a_M` cannot serve all epochs. Adding
  `a_Mz [logMh(t_i) - Mpiv] ln(1+z_i)` to the 3-parameter form (4 parameters
  total) takes pooled CV **0.1506 -> 0.1489** and z=0.4 to **0.1176 — better
  than `logMh(z_k)` alone (0.1209)** — while shrinking the offending slopes
  (-0.090 -> -0.035, +0.141 -> +0.076). Fitted:
  **`a0 = -2.763, a_M = -0.508, a_z = 0.634, a_Mz = +0.268`**.
  **But it trades bias for scatter**: its per-epoch bias runs +0.014 / -0.010
  dex where the 3-parameter law holds <= 0.004 everywhere. Under a
  deployability criterion (§5) that trade is not obviously worth taking —
  score both.
  **Parameterization warning.** Adding the cross-term *on top of the free
  lognormal window* (5 parameters) is degenerate: it reaches the same loss with
  `a0 = 26.8, mu = 92.5, sig = 7.93`, meaningless values that a loss check
  would not catch. On the monotone 3-parameter base the same term is
  well-behaved. **Never stack the cross-term on the free window.**
- **Concentration.** The `+c200c+fz2` baseline still beats the law by
  ~0.010-0.013 dex at every epoch, and the law's own CV residual **correlates
  with `c200c` at r = 0.244 (z=0.4), 0.204 (z=1.0), 0.119 (z=2.0)**. The law
  has no concentration term; adding `c200c(t_i)` is the obvious next lever and
  is exactly the deep conditioning of §4.4.

**F. The amplitude and the shape want opposite efficiency windows, and for the
amplitude the window is not a window at all.** The shape fit puts it at
mu = 1.522, sig = 0.235 — a sharp lognormal peaked at z ~ 3.6. The amplitude
wants `a_z = +0.355`: a smooth *monotone* rise toward high z with no peak.
Three measurements say the amplitude's window is degenerate rather than merely
different:

- A free lognormal window (4 parameters) reaches pooled CV **0.1506** — exactly
  what the 3-parameter monotone `a_z ln(1+z)` form reaches. **The window's two
  extra degrees of freedom buy nothing.**
- Its fitted values run off to mu = 5.18, sig = 2.19 — near exp53's own
  `SIG_FLAT = 3.0` "operationally flat" criterion, with mu unidentified.
- Over the sampled range `ln(1+z)` in [0.33, 2.4] that lognormal *is* a power
  law: `ln w ~ 1.08 ln(1+z)`, i.e. `a_z ~ 0.47` in log10 — recovered
  independently as 0.355 by the direct 3-parameter fit.

So one window must do two structurally incompatible jobs: set the mass budget
(monotone) and set the radial clock (sharply peaked). **This is the central
tension of the joint fit, quantified in advance.**

---

## 4. Design

### 4.1 Preserve the current model under an honest name

Keep the pinned kernel as `oracle_amplitude_shape_model`. It answers how well a
radial prescription can work when the reference mass is known perfectly, and it
is a legitimate ceiling — and now a useful one, because it is exactly the
`lambda_A -> infinity` limit of §4.3, so it is a free reference row on every
scoreboard. **Its absolute-mass QA values must never again be quoted as
forward-model scores.**

### 4.2 The reference aperture: `M*(<100 kpc)` — DECIDED

Not `M(<500)`: that is extrapolated (7.2% of it at z=0.4) and its form
systematic tilts with halo mass by the same amount as exp53's decisive signal.
Between the two measured options the statistics are indistinguishable (§3.2.A),
so the choice is made on meaning:

- `M*(<100 kpc)` matches the observational convention for massive galaxies
  (HSC/Huang-style aperture masses), and sits inside the measured grid rather
  than on its edge.
- `M*(<148 kpc)` is reported as a **free out-of-model prediction**, as is
  `M*(<500)` (against the extrapolated truth, with its 0.021 dex systematic
  stated). Neither is fitted.

Implementation note: the anchor is obtained by log-log interpolation at
`log10(100)` on the 24-point grid. **`np.interp` CLAMPS** — that is safe at
100 kpc (interior) and must never be used to reach 500 kpc.

### 4.3 The objective: a decoupling, not a weighting trick

Reparameterize each CoG losslessly into amplitude and shape:

```
A       =  log10 M*(<100 kpc)
F(<R)   =  M*(<R) / M*(<100 kpc)          F(<100) == 1
L       =  lambda_A * L_A  +  lambda_F * L_F
L_A     =  ( A_model - A_data )^2                      [dex^2, per galaxy per epoch]
L_F     =  Objective() applied to F                    [the production loss, on shapes]
```

Three properties worth stating explicitly.

1. **`L_F` on `F` is numerically the current production loss with the pin moved
   from 500 to 100 kpc.** So the entire exp38-exp53 record translates with a
   single measured offset, which Stage 0 must measure and record.
2. **The split makes the shape fit independent of the amplitude provider.**
   This is the real justification, stronger than "amplitude would swamp shape".
   Under an *unsplit* absolute objective (`frac` residual on un-normalized
   CoGs), a galaxy whose predicted `A` is too high drags the whole fitted
   profile down to compensate — the amplitude error is laundered into a shape
   distortion. The split forbids that. It is the mechanism behind revision 1's
   "amplitude and shape residuals must not be sampled independently".
3. **Its corollary is that exp54a cannot refit anything.** If `A` comes from
   the emulator and `L_F` depends only on `F`, then `L_A` contains no kernel
   parameter and `argmin L = argmin L_F` = exp53's fit at a different pin
   radius. **This is the design working as intended, not a bug** — but it means
   exp54a is a *measurement* (how much does realistic, correlated amplitude
   error cost every QA number?) and not a model revision. Any refit with
   content requires the kernel to supply its own amplitude (§4.5, product C).

**Setting the weights.** Not by the number of radial samples. Put each term on
its own achievable floor, in the program's existing plane-energy idiom:

```
lambda_A(z_k) = 1 / sigma_A,floor(z_k)^2      sigma_A,floor = the epoch-matched
                                              halo-only baseline of §3.2.B
                                              (0.121 dex at z=0.4 ... 0.182 at z=2)
lambda_F      = 1 / L_F,incumbent^2
```

Both terms then read ~1 at "as good as we currently know how to do", and the
epoch-dependent `sigma_A,floor` stops z=2 from dominating `L_A` merely because
amplitude is harder there. **Then scan `lambda_A/lambda_F` over
{1/4, 1, 4} and report what the trade-off buys** — the weighting must be an
evidenced choice, not a taste.

### 4.4 Epoch-local conditioning: condition each deposit at its own formation time

The current conditioning vector is built ONCE from z=0.4 descendant properties
and applied at every epoch (`stage2_multiepoch.py:_w_init`, `g["cond"]` from
`pop["logmh"], pop["c200c"], pop["fz2"]`). Replace it with

```
log R50_i  =  log R50_0  +  slopes @ cond(t_i)
cond(t_i)  =  standardized [ logMh(t_i),  c200c(t_i),  f_form(t_i) ]
```

where `f_form(t_i)` is a formation fraction relative to the halo mass **at
`t_i`** (e.g. `t_half` of the MAH truncated at `t_i`), not relative to the
descendant. Three reasons this is the deep version rather than the cheap
epoch-k swap:

- It is measured to be worth 1.7x at z=2 (§3.2.B).
- It replaces the phenomenological `(t_i/t_obs)^g` clock with the physical
  quantity that clock was standing in for. `t_obs` is a **global constant**
  (9.389 Gyr, snapshot 72) and not per-galaxy, so `(t_i/t_obs)^g` is just
  `const * t_i^g` — a bare time power law with no halo content. Whether `g`
  survives once `logMh(t_i)` is available is a real, scored question.
- It gives the shape a lever that does **not** run through the efficiency
  window, which is the only structural hope of relieving the §3.2.F tension.

A progenitor-track mode may deliberately condition on descendant properties,
but must be labelled as such and never used for catalogue population.

Bookkeeping to fix explicitly: `dMh` is `diff(10**logMh_full)`, so deposit `i`
spans `[t_i, t_{i+1}]` and the code aligns it with `logMh_full[1:]` — the
**post-deposit** halo mass. State the convention in the module docstring and
assert it; an off-by-one here is a silent 0.01-0.03 dex systematic in `a_M`.

### 4.5 Three products, and what each can and cannot test

| | amplitude from | shape from | can test | cost |
|---|---|---|---|---|
| **A. oracle** (= exp53 incumbent) | TNG truth | kernel | nothing new; it is the `lambda_A -> inf` ceiling and a free reference row | zero |
| **B. hybrid** | statistical emulator, out-of-fold | kernel | the **cost of realistic, correlated amplitude error** on every QA tier | cheap — no refit (§4.3.3) |
| **C. absolute** | the kernel's own `SUM eps_i dMh_i` | kernel | predictions (i)-(iv); the §3.2.F window tension; the real fiducial | a full joint fit campaign |

---

## 5. Scope of exp54 — and one decision reopened

**Agreed with the user (2026-08-22):** exp54 = Stages 0-2 (contract, amplitude
scoreboard, hybrid + re-baselining); the absolute deposition law becomes
**exp55**. Success criterion: **deployability, not victory.** The deliverable is
an honest unpinned forward model that predicts mass and profile jointly with
its amplitude error measured and reported; the halo-only baselines are reported
context, not a gate. It is adoptable at ~0.15 dex because it runs on an N-body
catalogue where the pinned model cannot.

**That decision was taken before §3.2.D and §4.3.3 were known, and both cut
against it.** Stated plainly so the next session can re-take it:

- exp54a turned out to be *much cheaper* than budgeted — it is a measurement,
  not a fit campaign — so exp54 as scoped will not fill a branch.
- exp55's amplitude is *already done*: 3 parameters, CV'd, shuffle-controlled,
  competitive, near-zero bias — and with the mass x redshift cross-term it
  **beats the epoch-matched halo-mass baseline at z=0.4 too** (0.1176 vs
  0.1209). What remains for exp55 is the joint fit, i.e. the §3.2.F window
  tension.
- None of the pre-registered predictions 1-4 can be tested under the current
  exp54 scope; predictions 5-7 are Stage 1/3 questions.

The counter-argument, which is why this is a question and not a correction:
the §3.2.F tension is real and structural, and the joint fit is where it bites.
Keeping exp55 separate keeps a clean boundary between "we measured what the pin
was hiding" and "we built its replacement".

**Recommendation: fold product C into exp54** as Stage 3, keeping Stages 0-2
exactly as agreed and as the deliverable that ships first. The split still
protects the staging — Stage 3 cannot corrupt Stages 0-2, because §4.3.2
guarantees the shape fit is independent of the amplitude provider. If the
answer is no, exp54 stands as scoped and this section becomes exp55's opening
argument.

---

## 6. Validation sequence (stage-gated; do not skip ahead)

**Stage 0 — contract and bookkeeping.**

- **The poisoned-data test, the single most important test in exp54.** Set
  `g["m500"]` and `g["data"]` to NaN and assert a forward call still returns
  finite profiles in physical `Msun`. Two lines; it is the only test that can
  *prove* no stellar-derived array enters prediction.
- Report mass **inside** the anchor, **between** anchor and 148 kpc, between
  148 and 500, and **beyond** 500 — four numbers, none renormalized away.
- Verify mass conservation numerically: `SUM_R dM(R) == SUM_i eps_i dMh_i` to
  machine precision.
- **Measure and record the `L_F` offset** between the 500-pinned and
  100-pinned loss, so every exp38-exp53 number can be translated.
- **Equivalence bridge**: the unpinned kernel post-hoc renormalized to
  `M(<500)` must reproduce exp53's CoGs to ~1e-12, proving the change is
  *only* in the amplitude. Mirror `exp53/kernel.py::demo` check (1).

**Stage 1 — the amplitude scoreboard, all epochs, 5-fold galaxy-level CV.**
One table, target `log M*(<100)`, every row on identical folds:

1. do-nothing; 2. descendant `logMh(z=0.4)`; 3. epoch-matched
`logMh(z_k)`; 4. `+ c200c(z_k) + f_form(z_k)`; 5. DiffMAH 4 params;
6. `S` at the shape-fitted window, slope one; 7. same, free slope;
8. **the 3-parameter absolute law**; 9. the law `+ c200c(t_i)` and
`+ a_Mz` cross-term (§3.2.E); 10. the multi-epoch statistical emulator
(the hybrid's amplitude provider).

Report mean relation, RMS **and** CRPS, residual scatter vs halo mass, and
interval coverage — the exp07 suite, not RMS alone. **Shuffled-MAH controls
within narrow final-mass bins for rows 5, 8, 9**, reported with the §3.2.D
caveat.

**Stage 2 — the hybrid, and the re-baselining.** `A` from **out-of-fold**
emulator predictions; pass a DRAW, not the TNG value, and preserve the
amplitude-shape residual covariance. Then run `hongshao.qa.evaluate` on
products A and B and publish a **translation table for every absolute QA
number in exp38 / exp40 / exp47 / exp48 / exp53.** Known anchor: the tier 2b
z=0.4 plane moves 2.1 -> 2.9 under a 0.137 dex amplitude error and -> 3.6 at
0.20 dex. `qa.evaluate` needs no change — it already computes an
amplitude-pinned residual curve (`qa.py::_tercile_curves`) which becomes the
shape diagnostic, and its tier 1+2 bias/scatter cells become real forward
scores for the first time.

**Stage 3 (pending §5) — the absolute law, jointly fitted.** No per-object
rescaling. Start at one epoch, then multi-epoch. Proceed only after Stages 0-1
pass. Consider separate in-situ and accreted channels: late ex-situ mass is not
determined by the smooth main-branch `dMh` — it depends on progenitor SHMRs,
merger ratios, stripping and survival, and DiffMAH smooths away exactly that
granularity. **Residual scatter is expected even in a correct model**; §3.2.D's
0.15 dex is a plausible floor, not a failure.

**Stage 4 — deployment scoring.** Score three DIFFERENT problems separately:
transfer to new haloes at a fitted epoch; redshift extrapolation for tracked
descendants; epoch-local prediction for an independent halo catalogue. Only the
third is the stated use case.

---

## 7. Pre-registered predictions (revised)

1. **The efficiency window will move — CONFIRMED IN ADVANCE, and quantified.**
   Shape wants mu = 1.522, sig = 0.235 (sharp, peaked at z ~ 3.6); amplitude
   wants a monotone `(1+z)^0.355` rise with no peak. The *new* prediction is
   about the price: **the joint fit lands between them and `L_F` degrades
   measurably.** That degradation is exp55's headline number. Testable only in
   product C.
2. **Over-reach will cost, but not the way revision 1 said.** `moffat-q0` puts
   6.61% of deposits beyond 500 kpc against `sersic-q0`'s 0.48%. Un-normalized,
   a global `a0` absorbs the *population mean* over-reach for free — so the
   cost is carried only by the **galaxy-to-galaxy scatter** in over-reach
   fraction. Measure that scatter per family; it, not the median, decides
   whether the family ranking changes.
3. **exp52's redistribution finding must be re-derived, not assumed** —
   transport and deposition can no longer trade against a free epoch-by-epoch
   amplitude.
4. **The efficiency-outer-tail degeneracy will NOT vanish**, and it is now
   nameable: it becomes a degeneracy between `a0` and `log_R50`, broken only by
   radial shape. Test it exactly as exp49 tested the `g` rail — profile the
   loss along `(a0, log_R50)` and report the conditional curvature.
5. **The concentration term will work — already confirmed, so what is
   pre-registered is its SIZE.** The law's CV amplitude residual correlates
   with `c200c` at r = 0.244 (z=0.4) falling to 0.119 (z=2.0), and the
   `+c200c+fz2` baseline is 0.010-0.013 dex better than the law. Prediction:
   adding `c200c(t_i)` to the efficiency recovers **most but not all** of that
   gap, and recovers **more at low z than at high z**, tracking the correlation.
6. **The mass x redshift cross-term works and is identifiable — both now
   confirmed.** Pooled CV 0.1506 -> 0.1489, z=0.4 to 0.1176, and on the
   monotone base the fitted `a_Mz = +0.268` is positive as predicted (the
   efficiency's halo-mass dependence flattens toward high z) with no runaway
   parameters. What remains pre-registered is the **trade**: the cross-term
   costs bias (+0.014 dex at z=0.4 against <= 0.004 for the 3-parameter law).
   Prediction: once `c200c(t_i)` is added, the cross-term's scatter advantage
   will shrink — both are absorbing the same low-z structure — and the
   3-parameter law plus concentration will beat the 4-parameter law without
   concentration.
7. **NEW: `g` will lose significance once `logMh(t_i)` conditions the deposit
   scale.** `(t_i/t_obs)^g` is a bare time power law with no halo content
   (§4.4). Prediction: with deposit-time conditioning in place, `g` either
   shrinks toward zero or its conditioning slopes absorb it. If `g` is
   undisturbed, the deep conditioning is not reaching the deposit scale and
   the implementation is wrong.

---

## 8. Consequences, risks, warnings

**Consequences.** Every absolute QA number in exp38, exp40, exp47, exp48 and
exp53 is quoted under the pinned condition and must be re-baselined. The
profile-family ranking, the efficiency window, the transport strength and the
over-reach behaviour all become open questions again. **exp53 remains valid for
the narrow question it answered**: with the correct per-galaxy amplitude
supplied, which radial kernel distributes it best.

**Risks.**

- The §3.2.F window tension may be unresolvable in the current one-window
  architecture. §4.4's deep conditioning is the designed mitigation; if it
  fails, the honest outcome is two windows (one for the mass budget, one for
  the radial clock), which is a real architecture change and needs its own
  approval.
- Stage 1 rows 8-9 are 3-5 parameter fits; Stage 3 is a full multi-start
  campaign. Budget accordingly and **validate at small scale first**.
- exp53's fits were IN-SAMPLE (10-fold CV over 17 cells x 3 starts was
  unaffordable). Do not compare an exp54 CV number against an exp53 in-sample
  number without saying so.

**Warnings carried forward (all cost real time in past sessions).**

- Multi-start is **mandatory** for the sigmoid family (spread 1.7e-3 to 2.5e-3
  vs 1e-8 for the others; one start in three lands in a worse basin).
- **`np.interp` CLAMPS** — `interp(0.0, R, cog)` is not zero, and the 100 kpc
  anchor is safe only because it is interior.
- State the **quantity AND the binning variable** when quoting a trend with
  mass; the apparent sign depends on both.
- A **SLICE is not a PROFILE**.
- Rank candidates **by the judge, never by loss**.
- usetex eats `%` — use `hongshao.qa._pct()`.
- Do **not** name a new experiment module `run.py` (it shadows exp30/exp35's).
- Fits SERIAL and sharded across processes; watch the PYTHON worker, not the
  `uv`/`caffeinate` wrappers. `caffeinate -i` does NOT prevent lid-close sleep.
- Always `export HONGSHAO_DATA_DIR=/Users/shuang/Desktop/tng300_mah_mprof`.

---

## 9. Decision record

**From 2026-08-22, user + reviewer (revision 1):**

1. Freeze exp53 as a conditional-shape boundary study.
2. Reclassify the pinned kernel as an **oracle-amplitude shape model**, not the
   deployable fiducial Ultimate SHMR.
3. Removal of per-galaxy truth normalization is the **first structural goal**
   of the next physical-model experiment.
4. Start with the modular predicted-amplitude hybrid, using the existing
   absolute-mass emulator as both baseline and amplitude provider.
5. Treat the absolute deposition-efficiency model as a separate, more physical
   experiment, measured against final-halo-mass-only and shuffled-MAH controls.

**From 2026-08-22, user (revision 2):**

6. Success criterion is **deployability, not victory** (§5).
7. exp54 = Stages 0-2; the absolute law becomes exp55 — **reopened in §5** on
   evidence that arrived after the decision.
8. The amplitude anchor is **`M*(<100 kpc)`**; 148 and 500 kpc are reported as
   free out-of-model predictions.
9. Epoch-local conditioning goes **deep** — each deposit conditioned at its own
   formation time (§4.4).

> A reference stellar mass may be defined at any scientifically defensible
> aperture, but in a forward model it must be PREDICTED from halo information,
> with calibrated scatter, rather than copied from the galaxy being predicted.
