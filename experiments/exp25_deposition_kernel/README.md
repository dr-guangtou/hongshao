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
- **Next (parked):** fit the 5 parameters across the population — are `σ_0, g,
  b_early, b_late` universal, and does `z_c` scale with halo mass as the quenching
  picture predicts? That turns this toy into a forward population model.
- Independent experiment; library untouched (reuses `tng_data`,
  `profile_emulator.density_from_cog`).
