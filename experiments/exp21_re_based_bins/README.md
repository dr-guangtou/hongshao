# exp21 — An Re-based emulator: aperture/outskirt masses in half-mass-radius units

## Question
The graduated emulator predicts stellar masses in *fixed physical* apertures
(<10, 10–30, 30–50, 50–100 kpc). But a fixed kpc aperture mixes physically
different regions across the size–mass range — "<10 kpc" is the whole inner
galaxy for a compact system and a sub-core slice for an extended one. Does an
**effective-radius (Re) coordinate** — bins defined as multiples of each
galaxy's own half-mass radius — give an emulator that is as predictable from
halo assembly, and does the secondary (`c_200c`) signal change?

## Method
- **Re** = half-mass radius *within 120 kpc*, read off each galaxy's 1-D curve of
  growth (`logmstar_cog`, 24 radii): the R where M(<R) = ½·M(<120 kpc). 120 kpc
  is the practical observational "total" (light beyond is hard to measure).
- **Five Re-based bins tiling 0→120 kpc:** `<0.5Re`, `0.5–1Re`, `1–2Re`, `2–4Re`,
  `4Re–120kpc`. Measured Re distribution: **median 11.1 kpc** (5/95th 4.5/22.5),
  so 120 kpc ≈ 11 Re — the envelope bin (4Re→120kpc) is one coarse, far-out
  region by necessity. `4Re < 120 kpc` for >99% of galaxies, so the cap is safe.
- Same probabilistic emulator as exp19 (linear mean on `DiffMAH + c_200c` +
  heteroscedastic full covariance), 5-fold CV, exp07 suite. A **local copy** of
  the exp19 CV machinery generalized to n bins — `hongshao/emulator.py` is left
  untouched. The kpc 4-bin emulator is rerun through the same code as a reference
  (it reproduces exp19: CRPS 0.0831, NLL −3.452 — sanity check passed).

## Key result
**Re-normalized masses are at least as predictable as kpc masses, the `c_200c`
gain is comparable (slightly stronger, +5.6% vs +4.7%), and Re-binning cleanly
isolates the stochastic accreted envelope beyond ~4 Re.**

| Re bin (median galaxy) | CV CRPS [dex] | R²(mean) | `c_200c` CRPS gain |
|---|---|---|---|
| <0.5Re (<5.5 kpc) | 0.058 | 0.86 | +0.0060 |
| 0.5–1Re (5.5–11 kpc) | 0.069 | 0.75 | +0.0042 |
| 1–2Re (11–22 kpc) | 0.074 | 0.78 | +0.0044 |
| 2–4Re (22–44 kpc) | 0.074 | 0.81 | +0.0044 |
| **4Re–120kpc (44–120 kpc)** | **0.101** | **0.37** | +0.0034 |

- **The emulator works and is well-calibrated** (marginal coverage 0.54/0.71/
  0.91/0.95 vs nominal 0.50/0.68/0.90/0.95): mean CRPS **0.0751**, joint NLL
  **−5.252**, conditional-coverage gap 0.037. (NLL is more negative than the kpc
  emulator's −3.45, but the two are different-dimension targets — 5 vs 4 bins —
  so the joint NLL is not directly comparable; CRPS-per-bin and R² are.)
- **The "predictable galaxy" lives inside ~4 Re** (R² 0.75–0.86), and the
  **envelope beyond 4 Re collapses to R²=0.37** — the irreducible, accreted/ICL
  scatter, more sharply isolated than in kpc bins (where the noisiest 50–100 kpc
  bin still held R²=0.83 because of its large dynamic range). Re-normalization
  separates the deterministic in-situ body from the stochastic envelope.
- **Concentration helps most in the *core*** (`<0.5Re`: +0.0060 dex), fading
  outward — consistent with exp16 ("`c_200c` partial-corr strongest in the
  core"). The overall `c_200c` gain (+5.6%) slightly exceeds the kpc value.

## Decision
- **Re-based binning is a viable, arguably cleaner target** for the Ultimate
  SHMR: same halo-assembly predictability, a comparable concentration signal, and
  a physically meaningful split between the in-situ body (<4 Re) and the accreted
  envelope (4 Re→120 kpc). Keep it as an alternative target definition, not a
  replacement — the graduated kpc emulator stays the default (directly
  observable apertures; no per-galaxy Re needed downstream).
- The envelope bin (4 Re→120 kpc) is the natural place to focus future
  outskirt/ICL modeling — it carries essentially all the Re-frame scatter.
- Independent experiment; `hongshao/emulator.py` and `forward.py` unchanged.
