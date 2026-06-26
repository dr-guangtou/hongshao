# exp25 — Build a massive galaxy from its MAH with a smooth deposition kernel

## Question
A physics-inspired forward toy (Lackner-Ostriker / El-Badry spirit, context doc
§6–7), independent of the data-driven emulator and **not labeling stars in-situ
vs ex-situ** — only "stellar mass grown" per epoch. Is it mathematically feasible
to turn a halo's *actual* MAH directly into a 1-D surface-density profile, and can
it reproduce a real galaxy? Phase 1: the single most massive galaxy in TNG300.

## The model
Between two snapshots the halo gains `dM_h`. A fraction `ε(z)` of it becomes new
stellar mass `dM* = ε·dM_h`, deposited as one **centred 2-D Gaussian**. A
mass-normalized Gaussian has **no free amplitude** (set by mass + width), so each
epoch contributes a single number — its width. The galaxy is the sum; the curve
of growth is closed-form:

```
Σ(R)   = Σ_i dM*_i / (2π σ_i²) · exp(−R²/2σ_i²)
M*(<R) = Σ_i dM*_i · (1 − exp(−R²/2σ_i²))
```

Two smooth ingredients:

- **Width** `σ(t) = σ_0 · (t/t_obs)^g` — a direct function of cosmic time.
  *(We tested tying σ to `R_200c` and dropped it — see below.)*
- **Efficiency**, two-epoch "quenching", continuous at a transition redshift `z_c`:
  ```
  ε(z) ∝ (1+z)^b_early                              for z ≥ z_c
  ε(z) ∝ (1+z_c)^(b_early−b_late) · (1+z)^b_late    for z <  z_c
  ```
  Before `z_c`: rapid early growth (steep `b_early`). After `z_c` ("quenched"):
  shallower (`b_late`). `z_c` is the halo's quenching redshift; in a *population*
  model it should scale with halo mass (massive halos quench earlier). The
  normalization is fixed by the galaxy's total stellar mass.

## Key result — feasible, and the physical structure matters
TNG300's most massive galaxy (`logM* = 12.36`, `logM_h = 14.96`, 62-snapshot MAH):

| efficiency model | params | CoG RMS |
|---|---|---|
| single power-law `ε∝(1+z)^β`, `σ(t)` | 3 | 0.028 dex |
| **two-epoch quenching, `σ(t)`** | 5 | **0.008 dex** |

- **Mathematically clean** (closed-form CoG; no per-snapshot fitting).
- **The two-epoch quenching law reproduces the BCG to 0.008 dex** with a physical
  story: rapid early growth (`b_early≈8.7`) builds the compact core until
  `z_c≈5`, then a much shallower regime (`b_late≈1.1`) adds the extended envelope
  — exactly the "early rapid growth → quench → slow growth" picture, *without*
  invoking in-situ/ex-situ. The figure shows it: early epochs lay down narrow
  central Gaussians, late epochs wide ones.
- **Width: `σ(t)`, not `R_200c`.** Tying `σ` to the virial radius was tested
  (`σ=f·R_200c`: 0.055 dex; `σ=f·R_200c^p`: 0.027 dex) against a direct
  `σ(t)=σ_0(t/t_obs)^g` (0.028 dex) and `σ(z)` (0.030 dex). They fit *equally
  well*, so `R_200c` was dropped — it added the assumption `M_peak = M_200c` and
  baked cosmology into the width for no gain. (R_200c was estimated as
  `[3 M_peak / (4π·200·ρ_crit(z))]^{1/3}`.)

## Caveats
- **ε > 1 at the 2 earliest epochs** (z≳8), reaching 1.73 — unphysical (you can't
  convert >100% of added halo mass to stars). But those epochs carry only **2.8%
  of M***, and capping ε at the cosmic baryon fraction (`f_b=0.157`) keeps RMS at
  0.007 — so it is an overfit of the compact core at the highest z, not a
  structural flaw. A deployed model should bound ε ≤ f_b.
- **5 parameters for one galaxy / 23 CoG points** reaches 0.008 dex — risk of
  overfitting. `z_c` and `b_early` are partly degenerate on a single object (a
  high `z_c` + steep `b_early` ≈ a concentrated early burst); the `z_c(M_h)`
  quenching scaling is a *population* statement and is where it becomes
  identifiable and testable.
- `M_peak` is used as the halo mass; the centred-Gaussian kernel is the simplest
  possible deposit shape.

## Decision / next
- **Proof of concept succeeds:** a halo MAH + a two-epoch quenching efficiency
  + a smooth `σ(t)` build a real BCG's 1-D profile to ~0.01 dex, with
  interpretable, physical parameters and no in-situ/ex-situ assumption. Naturally
  supports the "stop at each redshift" dream (integrate the MAH to `t(z_obs)`).
- Independent experiment; library untouched (reuses `tng_data`,
  `profile_emulator.density_from_cog`).

---

# Phase 2 — the population fit (n=2540)

`population_fit.py` fits the same 5-parameter model to every clean galaxy (the
`use` cut, 2540 with a usable MAH) and asks the population questions. Two cached
fit passes (`outputs/population_{full,reduced}.fits`); `--refit` recomputes.

## The forward map generalizes — at ~0.005 dex
| model | free params / galaxy | CoG RMS (median, 90th) |
|---|---|---|
| **full** (σ₀, g, b_early, b_late, z_c) | 5 | **0.0045**, 0.011 dex |
| **reduced** (σ₀, z_c; shape fixed at medians) | 2 | 0.022, 0.055 dex |
| emulator halo→profile reconstruction (exp22) | — | 0.116 dex |

The toy reconstructs *every* galaxy's curve of growth from its MAH, not just the
BCG. Even a **reduced model** with the shape parameters frozen at the population
median (`g=1.67, b_early=3.62, b_late=1.15`) and only the size `σ₀` and quenching
`z_c` free per galaxy reaches 0.022 dex. **Caveat:** the toy is *given* the true
`M*_tot`, so this is a forward MAH→profile-*shape* map, **not** a halo→profile
prediction — it is not a fair head-to-head with the emulator's 0.116 dex (which
predicts the mass too). The takeaway is that the MAH + a low-D smooth kernel
captures the profile *shape* essentially exactly.

## The shape parameters are ~universal
None of the five parameters correlates with halo mass (|r(logMₕ)| ≤ 0.08):

| param | median | [16–84] | r(logMₕ) |
|---|---|---|---|
| σ₀ [kpc] | 74 | 55–135 | −0.01 |
| g | 1.67 | 1.40–2.31 | +0.03 |
| b_early | 3.62 | 0.0–5.9 | −0.00 |
| b_late | 1.15 | −3.5–4.0 | −0.07 |
| z_c | 2.41 | 1.11–4.34 | −0.08 |

The two-epoch structure (`b_early > b_late`, steep early growth then shallow)
holds in the median. So a single *universal* kernel — `σ(t)=σ₀(t/t_obs)^1.67`,
steep-then-shallow efficiency with a break near `z_c≈2.4` — describes the whole
massive population, with per-galaxy scatter but no mass trend.

## Headline: z_c does **not** scale with halo mass (prediction refuted)
The quenching picture predicted `z_c` rises with `logMₕ` (massive halos quench
earlier). It doesn't:

- **reduced model:** `z_c = 2.66 − 0.55·(logMₕ−13.5)`, **r = −0.09** (p=9e-6).
- cross-check (full-fit identifiable subset, n=1790): slope −0.39, r=−0.08.
- `z_c` vs formation redshift `z50`: r=+0.07; vs `logM*`: r=+0.01.

Both fits give a *negative* slope (opposite the prediction), but `r²<0.01` — the
trend is statistically detectable only because n=2540, and is physically
negligible. **`z_c` is essentially independent of halo mass, stellar mass, and
assembly time** — a broadly-distributed (~2.4) parameter that absorbs per-galaxy
profile shape, not a clean quenching clock. The phase-1 single-galaxy `z_c≈5.0`
was a degenerate steep-early-burst solution; under the `ε≤1` cap the population
fit relaxes the most massive galaxy to `z_c≈1.8`.

## Method notes / caveats
- **ε cap at 1, not f_b.** Per-epoch efficiency `ε=dM*/dM_h` is bounded to ≤1
  (hard limit) via a soft hinge in the loss (`penw=5`, pins `ε_max=1`). We
  deliberately do **not** cap at `f_b=0.157`: that inverts `b_early<b_late` and
  destroys the two-epoch structure, and because the SHMR makes low-mass halos
  ε-richer it would distort them *differentially* and could **fake** a `z_c(Mₕ)`
  trend. Per-epoch `ε>f_b` is physical (stars form from a gas reservoir); ~3.7%
  of `M*` is deposited above `f_b`, reported as a diagnostic only.
- **z_c is fragile per galaxy** (z_c/b_early degeneracy): identifiable in
  1790/2540 full fits, and `b_late` rails at its bounds for a fraction. This is
  why the **reduced model** (universal shape, only σ₀ & z_c free, both
  identifiable) is the honest population statement, and it agrees with the
  full-fit cross-check.
- **σ depends only on cosmic time — a structural limitation (important caveat).**
  `σ(t)=σ₀(t/t_obs)^g` bakes in a strict time→radius ordering (early growth →
  narrow/inner, late → wide/outer), so the *inner* profile is driven almost
  entirely by high-z accretion. That is not generally true: assembly is
  event-dependent, and a **major merger even at low z** drives violent relaxation
  + dynamical friction that **add and redistribute stellar mass in the inner
  region** — late accretion does not go only to the outskirts (the central density
  cannot keep rising, but it is not frozen either). The toy cannot capture a late
  merger rebuilding the core. Physically the width should be a function of the
  accretion *event* (merger mass ratio, orbit, smooth vs clumpy), learnable from
  the simulation. **Concrete future test:** the dataset already has z-resolved
  profiles at z=0.4/0.7/1.0/1.5/2.0 (`tng_data.ANCHORS`); integrating the MAH only
  to `t(z)` and comparing to the *measured* profile at that z directly tests the
  time→radius rule across cosmic time (the "stop at each redshift" idea) and would
  localize where late mergers reshape the inner profile.
- **Phase 2 is 2540 *independent* per-galaxy fits — exploration, not yet a true
  population fit.** They show which params are ~universal, but the deposition
  parameters should be *shared* across galaxies. The genuine population fit (one
  shared kernel) is **Phase 3 below**.

## Verdict
The deposition-kernel toy is a **good interpretable forward model**: one
universal low-D kernel maps any massive halo's MAH to its z=0.4 stellar profile
shape to ~0.005 dex. But the **population hypothesis fails** — `z_c` carries no
halo-mass (or assembly) quenching signal in TNG300. The toy's value is exactly
what the handover anticipated: *interpretability + a forward MAH→profile map*,
not a new halo→profile predictor and not a quenching-mass clock.

Run: `PYTHONPATH=. uv run python experiments/exp25_deposition_kernel/population_fit.py [N] [--refit]`

---

# Phase 3 — the *true* population fit (one shared kernel)

Phase 2 fit every galaxy independently. A real population model shares the
deposition physics: the **same** parameters map any halo's MAH to its profile.
We minimize the mean per-galaxy CoG RMS over the *whole* sample at once
(vectorized; the per-galaxy fits supply the starting point and the upper bound on
accuracy). Three nested global models:

| global model | shared params | CoG RMS (median) |
|---|---|---|
| per-galaxy free (Phase 2, *reference*) | — (5/galaxy) | 0.0045 |
| reduced (Phase 2, *reference*) | shape only; σ₀,z_c free/galaxy | 0.022 |
| **B′** `g,b_early,b_late` + `σ₀(R50)` + `z_c(logMₕ)` | 7 total | 0.0805 |
| **B** `g,b_early,b_late,z_c` + `σ₀=f(R50)` | 6 total | **0.0802** |
| **A** `g,b_early,b_late,z_c,σ₀` all constant | 5 total | 0.0831 |

**A single universal kernel reconstructs all 2540 profiles to ~0.08 dex** — below
the data-driven emulator's 0.116 (modulo the same M\*_tot caveat). The
*cost of universality* is steep but bounded: removing per-galaxy freedom takes the
median RMS 0.0045 → 0.022 → 0.080, roughly ×4 per step.

Three results from the global fit:
- **z_c does not scale with halo mass — confirmed at the population level.** Adding
  a `z_c = z_c0 + s·(logMₕ−13.5)` slope (model B′) gives `s=+0.19` and changes the
  median RMS by **−0.0003 dex** (i.e. nothing). The strongest form of the test:
  the data do not want a quenching-mass trend.
- **σ₀ is only weakly tied to galaxy size.** The best relation is
  `log σ₀ = 1.81 + 0.33·log R50` — a shallow slope, and tying σ₀ to R50 improves
  the global fit by only 0.083 → 0.080 dex. The deposition width is set more by
  cosmic time (`g`) than by the galaxy's final size.
- **The efficiency-slope *interpretation* is not robust (a model degeneracy).**
  The population-optimal shared shape **inverts** the Phase-1/per-galaxy "steep
  early growth": it prefers `b_early≈0.2 < b_late≈4.0` with a steeper `g≈2.4`
  (deposit mass *late* with fast-growing widths). This optimum is real, not a
  fluke — 3 of 4 diverse Nelder–Mead starts (including a steep-early one) converge
  to it (loss 0.096 vs 0.102 for the steep-early basin). `(g, b_early, b_late)`
  trade off: "centrally concentrated early" can be built by *small early widths*
  **or** by *slow width-growth + late efficiency*. So the toy's reconstruction
  accuracy and its null z_c–mass result are robust, but the **sign of the
  two-epoch asymmetry is not** — a reminder that this is a phenomenological map,
  not a measured star-formation history.

**Phase-3 verdict.** A genuinely universal MAH→profile kernel exists and works to
~0.08 dex with 5–6 shared parameters; the quenching redshift carries no halo-mass
signal even in the strongest population-level test; and the detailed efficiency
shape is degenerate, so the toy should be read as a compact forward *map*, not as
an inferred assembly history.

## Best-fit summary — model B (the true population model), n=2540

Forward map (per galaxy, **zero per-galaxy fit knobs**): from the halo MAH
`{dM_h,i, z_i, t_i}`, the total stellar mass `M*_tot` (sets the normalization),
and the half-mass radius `R50` (sets σ₀):

```
dM*_i  = M*_tot · ε(z_i) dM_h,i / Σ_j ε(z_j) dM_h,j        # stellar mass per epoch
M*(<R) = Σ_i dM*_i · [1 − exp(−R²/2 σ(t_i)²)]             # closed-form curve of growth
σ(t)   = σ₀ (t/t_obs)^g ,   log₁₀ σ₀ = a + b·log₁₀ R50     # deposition width
ε(z) ∝ (1+z)^b_early                            (z ≥ z_c)  # two-epoch efficiency,
ε(z) ∝ (1+z_c)^(b_early−b_late) (1+z)^b_late    (z <  z_c) #   continuous at z_c
```

| param | best-fit ± boot | role / phenomenology |
|---|---|---|
| `g` | **2.41 ± 0.04** | width-growth exponent — σ grows steeply with cosmic time (early clumps compact, late clumps wide) |
| `a` | **1.81 ± 0.05** | σ₀ intercept [log₁₀ kpc] |
| `b` | **0.33 ± 0.02** | σ₀–size slope — only *sub-linear*: the width is set by cosmic *time*, not the galaxy's final size |
| `b_early` | **−0.33 ± 0.12** | efficiency slope at z≥z_c — a near-flat *plateau* |
| `b_late` | **4.12 ± 0.19** | efficiency slope at z<z_c — a *steep* decline |
| `z_c` | **2.69 ± 0.14** | efficiency transition redshift |
| →`σ₀(R50=10 kpc)` | **138 ± 16 kpc** | derived: latest-epoch clump width for a 10-kpc galaxy |

Errors = sampling scatter over **N=40 galaxy bootstraps** (`bootstrap_global.py`),
warm-started, so they do **not** include the `g`/`b_early`/`b_late` basin
degeneracy — the dominant systematic. Median CoG RMS **0.080 dex**.

**Reads as:** stars form from halo accretion with a conversion efficiency that
**plateaus at z≳2.7 and then drops ~200× toward z=0** (a late quench, ε ∝
(1+z)^4.1 below z_c); each epoch's mass lands in a clump whose width **grows
steeply with cosmic time** (g≈2.4), so the rapid high-z halo growth (high ε, small
σ) builds a compact core and the slow, low-efficiency late growth (small ε, large
σ) adds a thin extended envelope. One such law fits all 2540 galaxies to ~0.08
dex. The *sign* of the early/late efficiency asymmetry is degenerate with `g` (the
per-galaxy fits prefer the mirror solution with the efficiency peak at the highest
z), so treat ε(z)'s exact shape as phenomenology, not a measured SFH; the robust
outputs are the ~0.08-dex reconstruction and the null z_c–mass trend.
