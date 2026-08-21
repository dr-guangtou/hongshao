# exp53 — the deposition-only boundary test

**Status: COMPLETE.** Branch `exp53-deposition-only`.
Date 2026-08-21. Sample n = 2397 (TNG300 centrals), five epochs
z = 0.4 / 0.7 / 1.0 / 1.5 / 2.0, fitted on z <= 1.5 and scored at all five.

**Verdict (user, 2026-08-21): no deposition-only model is significantly better
than the fiducial. The generalized-sigmoid family is not bad but still has
problems at the centre and at high redshift; it cannot solve the problem alone,
but it is worth carrying into the next phase.**

---

## 1. Where the model stood before exp53

The adopted product is the **`1ch-mof` kernel** (exp38) fitted at the exp40
`z <= 1.5` scope, plus the exp41 stochastic layer. It builds a galaxy by
depositing stellar mass along its halo's accretion history and then
transporting some of it outward:

- a **lognormal efficiency window** in `ln(1+z)` decides *when* halo mass
  accretion turns into stars — parameters `mu`, `sig`;
- each deposit is laid down with a **Moffat profile** whose scale grows as a
  power of formation time, `rc(t_i) = 10^log_rc (t_i/t_obs)^g`, with outer
  slope `gamma`;
- **transport**: a fraction `1 - fc` of each deposit later migrates outward to
  `rc (t_k/t_i)^q`, with `fc = exp(-(t_k-t_i)/(alpha t_i))`. **`q` is fitted;
  `alpha` is hardcoded to 1** (exp29 measured alpha ~ 1.04). So transport is a
  two-parameter mechanism with one parameter frozen;
- everything is **normalized to `M(<500 kpc)`**, and a halo-conditioning layer
  puts the deposit scale and the window width on the standardized
  `[logMh, c200c, fz2]` vector.

Two prior results set up this experiment.

**exp52** decomposed the adopted kernel's own growth and found it grows
galaxies by **redistribution, not accretion**: 96.5% of its `M*[50,100]` growth
over z = 0.7 -> 0.4 is material it already had, moved outward, and its own
deposition supplies only ~11% of the z = 2 -> 0.4 mass growth. It also framed
the defect as **"right source, ~10x too much reach"** — 93% of the drained mass
leaves 0-5 kpc (physically reasonable for AGN-driven expansion) but 58.5% is
delivered beyond 50 kpc and 12% beyond 500 kpc, where the normalization
discards it.

**exp47/exp48** established a standing failure the kernel family has never
fixed: on **compact** galaxies the model under-fills the centre by 30-44% at
every epoch, and the adopted product does not pass the mass-size plane.
exp48 also concluded "the objective is the lever and the profile is not" — but
provisionally, since every profile family there was fitted against a wall.

## 2. Why exp53

Three motivations, in order of weight.

1. **The transport magnitude is not physical.** Moving most of a massive
   galaxy's late outskirt growth from the centre outward contradicts the
   established picture, in which the extended stellar halo is built by dry
   minor mergers depositing new stars at large radii. A model can reproduce the
   summed profile while getting the mechanism wrong, and the objective cannot
   see the difference.
2. **A boundary condition is the cleanest way to size the effect.** Setting
   `q = 0` is an exact nested special case — `rcw = rc0` makes `fc`
   algebraically irrelevant — so "deposition only" is a well-defined limit of
   the same model rather than a different one.
3. **Profile families deserve a fair comparison in a simple setting.** With
   transport removed there is no compensating mechanism, so the deposit profile
   has to do the work alone and the families separate cleanly.

The pre-registered falsifiable prediction: exp52 implied transport drains the
centre, so **removing it should improve central masses**.

## 3. What was built

All in `experiments/exp53_deposition_only/`, each module self-checked on real
galaxies.

| module | what it does |
|---|---|
| `families.py` | seven deposit families, every one parameterized by its **half-mass radius** so `log_R50` means the same physical thing across the shootout |
| `kernel.py` | the production kernel with switchable family and switchable `q`; reproduces exp38's **adopted** CoGs to **1.3e-15** and its loss to 1e-12 |
| `score.py` | exp38 stage 0.2's added-light-kernel operator applied to model CoGs, exp52's slopes, transport share, deposit reach, growth ratios |
| `fit.py` | resumable, shardable driver (multi-start, real bounds, L-BFGS-B) |
| `rail_check.py` | refits any cell that converged at a bound, with the bound widened; BENIGN / LOAD-BEARING verdict |
| `outskirt.py` | the stellar-halo test: annuli and apertures, 2000 galaxy bootstraps, **paired** per-galaxy comparison, `--by logmh\|logms` |
| `headtohead.py` | every judged metric scored by **distance to target** |
| `report.py` | verdict tables, kernel figures, and the standard QA battery per family |

**The R50 reparameterization.** exp38 rejected Sersic for an "n-scale
degeneracy". Measured here: at a fixed raw scale `a = 1`, running `n` from 0.5
to 8 moves the profile's half-mass radius by **1.2e6x** — scale and index were
competing to set the same physical radius. (Prior art: exp48 step C had already
built `sersic_r50` and re-run it; exp53's contribution is applying the
coordinate to *every* family, not reinstating Sersic.)

## 4. Tests run

- **The SLICE** — adopted theta, transport switched off, nothing re-optimized.
- **The REFIT** — `q = 0` with all other parameters re-optimized, 3 starts.
- **The family shootout** — 7 families at `q = 0`; moffat/sersic/expo/gauss
  also at `q` free. 3 starts each, full sample.
- **Rail checks** — every cell that landed on a bound.
- **The standard QA battery** — `hongshao.qa.evaluate` for all 8 headline
  cells: tier 1+2 mass tables, 2b planes, 2c growth, 2d size planes, 2e CDFs,
  profile visual QA, and binned CoG profiles.
- **The outskirt test** — the branch's actual target, bootstrapped and paired.
- *(Out of scope, retained as an appendix: a profiled `q` grid. The user scoped
  it out on 2026-08-21 as answering a transport-tuning question, not this one.)*

## 5. Results

### 5.1 The prediction fails under refitting — and that gap is the finding

| compact-galaxy `M*(<5 kpc)` bias | value |
|---|---|
| fiducial (`q` = 0.908) | **-43.4%** |
| transport off, **nothing refitted** (slice) | **+13.6%** |
| transport off, **refitted** | **-39.4%** |

The slice says the prediction passes spectacularly; the refit says it buys four
points out of forty-three. The model does not respond to losing transport by
depositing more centrally — it flattens its efficiency window
(`sig` 0.22 -> 2.68) and shrinks its deposits (`log R50` 3.27 -> 1.73), buying
the outskirt fit back a different way. **A mechanism that is real in a model's
internal bookkeeping is not thereby the CAUSE of that model's error.** exp52's
claim survives in its narrow form and fails in its causal form; the
exp47/exp48 compact defect is unexplained again.

### 5.2 The family ranking (deposition-only, ranked by the measured added light)

| cell | loss | par | kernel [dex] | kern n mod/dat | M(<10) growth | wins vs fiducial |
|---|---|---|---|---|---|---|
| **gompertz_log-q0** | 0.159687 | 11 | **0.072** | **2.67/2.67** | **1.004** | **10/13** |
| richards-q0 | **0.158948** | 12 | 0.076 | 2.47/2.67 | 1.009 | 9/13 |
| *moffat-qfree (fiducial)* | *0.153577* | *12* | *0.081* | *2.67/2.67* | *0.722* | *—* |
| moffat-q0 | 0.160001 | 11 | 0.086 | 2.37/2.67 | 0.879 | 9/13 |
| loglogistic-q0 | 0.161177 | 11 | 0.098 | 1.88/2.67 | 0.950 | 7/13 |
| sersic-q0 | 0.162079 | 11 | 0.102 | 2.27/2.67 | 1.015 | 6/13 |
| expo-q0 | 0.175218 | 10 | 0.286 | 1.88/2.67 | 1.875 | 6/13 |
| gauss-q0 | 0.177770 | 10 | 0.339 | 1.68/2.67 | 2.227 | 7/13 |

- **`richards` fits nu = 0.345.** Since `nu` -> 0 recovers `gompertz_log` and
  `nu` = 1 recovers `loglogistic`, the data want an asymmetric sigmoid closer
  to Gompertz than to log-logistic but neither exactly. It beats both families
  it nests, as an interior optimum must.
- `gompertz_log-q0` beats `moffat-q0` at **equal parameter count**, and
  confirms exp48's 2nd-of-six ranking across a different objective and harness.
- **Wings are essential.** Without transport the wingless families collapse at
  high z: `gauss-q0`'s `M[50,148]` reaches **-6.2 dex** at z = 2 and its
  mass-mass plane slope is +4.11 against a measured +1.43. The falsification
  control fails exactly as designed.
- **The Moffat needs transport to reach.** `moffat-q0` carries a ~12% halo
  deficit at z <= 1 and builds only **0.885x** the truth's halo growth over the
  full baseline (fiducial 1.062x); at z = 2.0 -> 1.5 for massive galaxies it
  manages **0.702x** where the fiducial is 0.992x.
- **`sersic-q0` matches the incumbent on the halo** — within +/-0.03 dex at
  every epoch, growth 1.073x vs 1.062x, and per galaxy significantly CLOSER at
  z >= 1.0 — despite losing on both loss and kernel. Deposit reach explains it:
  `moffat-q0` throws 6.8% of deposited mass beyond 500 kpc where the
  normalization discards it, `sersic-q0` only 0.7%.

### 5.3 The defect that decides the verdict

`M(<148)` median dlog(model/truth), binned by **halo mass**:

| halo-mass tercile | fiducial | gompertz_log-q0 | richards-q0 | moffat-q0 |
|---|---|---|---|---|
| T1 low | **+3%** | **-3%** | -2% | -3% |
| T3 high | **+3%** | **+6%** | +7% | +6% |

**A 9-percentage-point tilt with halo mass, where the fiducial is flat.** And
it is not a mass-budget error — the `M[50,148]` annulus tilts the OTHER way
(+17% low, -9% high):

| halo mass | inner `M(<50)` | outer `M[50,148]` | total `M(<148)` |
|---|---|---|---|
| low | too little | +17% | **-3%** |
| high | too much | -9% | **+6%** |

**The deposition-only models carry a RADIAL DISTRIBUTION error that flips sign
with halo mass** — profiles too extended in low-mass haloes, too concentrated
in high-mass ones. The model's concentration does not vary with halo mass the
way the truth's does. **The fiducial's transport term is what supplies that
halo-mass dependence** — the sharpest statement exp53 produced about what
transport is actually *for*. It also explains the strained outskirt-overshoot
gate (+0.074 to +0.090 against a +/-0.06 band), which is measured on the
low-mass tercile, exactly where the over-extension sits.

### 5.4 Rails and identifiability

- **`mu` at its bound on every deposition-only cell: BENIGN** (0.02-0.05% of
  the loss). Design constraint 2 held — the efficiency window was effectively
  free.
- **The deposit scale is UNIDENTIFIED for every family except the Moffat.**
  sersic/expo/gauss escape `log_R50` = 3.5 and immediately rail at 5.0;
  widened, the three converge to within 1e-4 of each other. "Every deposit at
  the ceiling" is the same model whatever shape is hung on it — **a mechanism
  for exp48's "the profile does not matter"**, which is therefore conditional
  on an unpinned scale.
- **The sigmoid loss surface is much rougher**: start-to-start spreads of
  1.7e-3 to 2.5e-3 against 1e-8 to 1e-10 for the other families, with one start
  in three landing in a worse basin. Multi-start is mandatory for this family.

### 5.5 A correction to the record

`exp52/decompose.py` divided the model's APERTURE growth by the truth's ANNULUS
growth (`np.interp(0.0, R, cog)` clamps to the CoG at 2 kpc rather than
returning zero). **`M(<10)` model/truth growth is 0.726, not 0.573.** exp52's
qualitative finding survives at 27% rather than 43%; its transport shares are
unaffected. Fixed, re-run, and locked out by assertions.

## 6. Where the record lives

| what | where |
|---|---|
| full results, all tables | `experiments/exp53_deposition_only/README.md` |
| design and constraints | `doc/plans/2026-08-20-deposition-only-test.md` |
| roadmap entry | `doc/todo.md` (exp53) |
| lessons | `doc/lessons.md` (5 entries) |
| fitted thetas | `experiments/exp53_deposition_only/outputs/fits.npz` |
| verdict tables (JSON) | `outputs/verdict_shootout.json`, `verdict_slice.json` |
| run logs | `experiments/exp53_deposition_only/logs/` |
| **QA figures, per family** | `figures/<family>/` — 10+ per cell, PNG + PDF |
| cross-family figures | `figures/_overview/` |

QA figure set per cell: `qa_planes`, `qa_size`, `qa_growth`, `qa_cdf`,
`qa_bins` (halo-mass bins), **`qa_bins_ms` (total-stellar-mass bins, new
standard as of 2026-08-21)**, `qa_cases`, `qa_mass_kpc_aper/diff`,
`qa_mass_Re_aper/diff`.

## 7. Next steps

1. **Make the profile shape redshift-dependent.** exp53 inherited a design
   asymmetry it did not test: the deposit SCALE is controlled by five numbers
   (`log_R50`, the time exponent `g`, three halo-conditioning slopes) and
   varies with halo and epoch, while the SHAPE is **one population constant**
   with no halo or epoch dependence. Sketch: `shape(t_i) = shape_0 + shape_1
   log10(t_i/t_obs)` plus a shape conditioning row; `shape_1 = 0` nests exp53
   exactly, so the fitted `shape_1` IS the measurement. The measured kernel
   cannot currently resolve the trend (early-minus-late `n` = +0.20 against a
   0.49 within-epoch spread and a 0.197 grid), so pair it with a finer
   re-measurement of the exp38 stage 0.2 kernel. **This is a branch of its own,
   not a footnote** — exp53's earlier lean toward "probably not worth it" is
   withdrawn.
2. **Give the two-channel model another try.** exp53's decisive defect is that
   a single deposit profile cannot make the concentration vary with halo mass
   the way the data do. A second channel is the natural way to buy that freedom
   without reintroducing large-scale transport, and the 2ch architecture
   (exp36/exp38 `2ch-exp`) has never been tested with the R50 coordinate, the
   deposit ceiling, or the sigmoid families.
3. **Re-introduce a "transport-lite" mechanism confined to the inner region.**
   exp52's diagnosis was *right source, too much reach*: the drained mass comes
   from 0-5 kpc, which is physically sound, but is delivered far too far out.
   A transport term with a bounded reach — capped migration radius, or a
   `q` that decays with radius — would keep the adiabatic-expansion physics for
   the core while leaving the stellar halo to deposition. exp53 shows the halo
   does not need transport (`sersic-q0` matches the incumbent there), so the
   two roles can be separated.
4. **Carry the sigmoid family forward** (user's instruction). `gompertz_log`
   and `richards` are live candidates for any of the above; `richards` is the
   efficient entry point since its `nu` measures which sigmoid the data want.
5. **Cross-validate before adopting anything.** exp53's fits are IN-SAMPLE;
   exp38 stage 2 used 10-fold CV. Differences of a fraction of a per cent —
   which includes `gompertz_log-q0` vs `moffat-q0` — are inside the margin.
6. **Pin the deposit scale** before any further family shootout, since exp53
   showed the comparison collapses when the scale is free to rail.

## 8. Open caveats

- All exp53 fits use the **production objective** (CoG, fractional residuals).
  exp48 measured that the objective REVERSES the profile ranking, so this
  ordering is conditional on that choice.
- In-sample, no CV (see next step 5).
- `gauss-qfree` / `sersic-qfree` hit the rail-check evaluation cap on one start
  each; both starts agree to <= 2e-6.
- The `gausswing` family (exp38 stage 1 co-winner) was not tested.
- `q` free was not run for the sigmoid families (out of this branch's scope).
