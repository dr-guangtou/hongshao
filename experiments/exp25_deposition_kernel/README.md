# exp25 — Build a massive galaxy from its MAH with a smooth deposition kernel

## Question
A physics-inspired forward toy (Lackner-Ostriker / El-Badry spirit, context doc
§6–7), independent of the data-driven emulator and **ignoring the in-situ/ex-situ
distinction**. Is it mathematically feasible to turn a halo's *actual* mass
accretion history directly into a 1-D surface-density profile, and can it
reproduce a real galaxy? Phase 1: the single most massive galaxy in TNG300.

## The model
Between two snapshots the halo gains `dM_h`. A fraction `eps(z)` of it becomes
new stellar mass `dM* = eps·dM_h`, deposited as one **centred 2-D Gaussian**.
A mass-normalized Gaussian has **no free amplitude** (it is set by the mass and
the width), so each epoch contributes a single number — its width:

```
Sigma_i(R) = dM*_i / (2π σ_i²) · exp(−R²/2σ_i²)
Σ(R) = Σ_i Σ_i(R)          M*(<R) = Σ_i dM*_i·(1 − exp(−R²/2σ_i²))   [closed form]
```

Widths are **not** fit per snapshot — they follow the halo size at deposition,
`σ_i = f·R_200c(M_h,i, z_i)^p`, so recently accreted stars (low z, big halo)
land at large radius automatically. Efficiency evolves smoothly,
`eps ∝ (1+z)^β`, with its normalization fixed by the galaxy's total stellar mass.

Three nested models (1–3 parameters):

| model | efficiency | width | params |
|---|---|---|---|
| 0 | constant | `f·R_200c` | `f` |
| 1 | `(1+z)^β` | `f·R_200c` | `f, β` |
| 2 | `(1+z)^β` | `f·R_200c^p` | `f, β, p` |

## Key result
**Yes — feasible, and 2 parameters reproduce the BCG.** For TNG300's most massive
galaxy (`logM* = 12.36`, `logM_h = 14.96`, 62-snapshot MAH):

| model | CoG fit RMS | fitted parameters |
|---|---|---|
| 0 — const eps | 0.234 dex | `f=0.017` |
| **1 — eps∝(1+z)^β** | **0.055 dex** | `f=0.055, β=2.48` |
| 2 — + width slope | 0.027 dex | `f=0.003, β=1.73, p=1.40` |

- **Mathematically clean:** the curve of growth is a closed-form sum of
  `(1 − Gaussian)` terms; no per-snapshot fitting, no integration.
- **Constant efficiency fails (0.234 dex): too extended.** It overweights late,
  dry halo growth (when `R_200c ~ 1 Mpc`), dumping stars far out. This is the
  signature of ignoring that late halo growth adds little *new in-situ* mass.
- **A single physical knob fixes it:** `eps ∝ (1+z)^2.5` — efficiency ~higher
  early — and the profile snaps to **0.055 dex** with just `(f, β)`. The width
  fraction `f≈0.05` means stars deposit at ~5% of the virial radius at each
  epoch; `β≈2.5` echoes the known rise of star-formation efficiency toward high z.
- The picture (`exp25_build_galaxy.png`): early epochs deposit narrow central
  Gaussians (the compact core), late epochs deposit wide ones (the envelope);
  their sum is the BCG. The ingredients (`exp25_ingredients.png`): the MAH sets
  *how much* stellar mass forms *when*, and the halo's growing size sets *where*
  it lands.

## Decision / interpretation
- **The toy works** as a proof of concept: a halo MAH + two interpretable knobs
  (a deposition-radius fraction and an efficiency-evolution slope) build a real
  massive galaxy's 1-D profile to ~0.05 dex. This is a genuine *forward,
  generative* alternative to the data-driven emulator, and it naturally supports
  the "stop at each redshift" dream (integrate the MAH only up to `t(z_obs)`).
- **Caveats:** (1) one galaxy — `(f, β)` are fit, not yet learned as
  population/redshift functions; (2) `M_peak` is used as the halo mass (not
  strictly `M_200c`); (3) the centred-Gaussian kernel is the simplest possible
  shape — real deposits are not Gaussian, and the in-situ/ex-situ split is
  ignored by construction.
- **Next (if pursued):** fit `(f, β)` across many galaxies → are they universal,
  or halo-mass/assembly-dependent? That would turn this into a population forward
  model. Parked for now (this was a "build a galaxy for fun" proof of concept).
- Independent experiment; library untouched (reuses `tng_data`,
  `profile_emulator.density_from_cog`).
