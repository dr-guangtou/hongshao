# exp26 — differential stellar-density profiles: is the "Gaussian deposition" real?

## Question
The exp25 deposition kernel assumes each epoch's *added* stellar mass has a
**centred 2-D Gaussian** surface density. The TNG300 drop stores measured
profiles at **5 redshifts** (z = 0.4, 0.7, 1.0, 1.5, 2.0 = snaps 72/59/50/40/33),
so we can test that assumption directly. For each galaxy we build the
**differential surface-density profile** between adjacent epochs

```
ΔΣ(R) = Σ(R, z_low) − Σ(R, z_high)        # later − earlier = mass grown in the interval
```

for the four pairs `[0.4,0.7], [0.7,1.0], [1.0,1.5], [1.5,2.0]`, store them, and
ask what functional form describes them. **Focus is the density profile Σ(R), not
the curve of growth.**

## Method
- `intensity` from the isophote `prof` tables is the surface mass density
  [M⊙/kpc²]; we interpolate `log Σ` onto a common log-radial grid (2–150 kpc) at
  each epoch and difference. We look at both the **linear** `ΔΣ` (what a deposition
  kernel adds) and the **fractional** `Δlog Σ` (robust to the steep, marginally
  resolved core that dominates the linear difference).
- Diagnostic: fit `Δlog Σ(R) = a + b·log R` over the resolved range (6–100 kpc).
  `b > 0` ⇒ the outer density grows faster than the inner ⇒ **inside-out**, and
  the profile evolves *multiplicatively* as `Σ_low/Σ_high ∝ R^b`. `b = 0` is
  self-similar growth. We also fit a Sérsic index to the positive `ΔΣ` (n = 0.5 is
  a Gaussian, n = 1 exponential).
- Adjacent pairs have small Δt → noisy per galaxy, so the headline numbers come
  from the **stacked** (per-radius median) profile and the long baseline.

## Sample
Every galaxy with a **valid z=0.4 profile** (`flag` & finite CoG) = **3380** of
3388. exp26 only differences *measured profiles*, so it deliberately drops the
MAH-quality cuts in the exp25 `use` set (`mah_declined`, `logmh≥13`,
`latest_snap`) — those concern the halo accretion history, which this experiment
never touches. (The 695 declining-MAH galaxies, re-included here, are analysed
separately in `declining_mah.py`; see below.)

## Key result — the added mass is NOT a centred Gaussian
**Differential growth is inside-out and multiplicative; the profile flattens and
extends with time.** (n = 3380)

| epoch pair | `b_stack` | Δlog Σ(8 kpc) | Δlog Σ(60 kpc) |
|---|---|---|---|
| [0.4, 0.7] | +0.15 | +0.04 | +0.18 |
| [0.7, 1.0] | +0.16 | +0.06 | +0.20 |
| [1.0, 1.5] | +0.28 | +0.15 | +0.42 |
| [1.5, 2.0] | +0.34 | +0.19 | +0.52 |
| **z=2 → 0.4 (long baseline)** | **+0.95** | **+0.62** (×4.2) | **+1.47** (×29) |

- **`Δlog Σ` rises ~linearly with `log R`** at every epoch → `Σ_low/Σ_high ∝ R^b`.
  Over the long baseline the inner density (8 kpc) grows ~**4×** while the outer
  (60 kpc) grows ~**29×**. The profile is amplified by a radius-dependent power-law,
  flattening as it grows. (On the narrower exp25 `use` sample, n=2545, b=+0.85 —
  the result is robust to the sample.)
- **A centred Gaussian is the wrong shape.** A Gaussian deposit adds the *most*
  mass at R = 0 and declines; the data adds the most (fractionally) at large R.
  The per-galaxy Sérsic index of the positive `ΔΣ` scatters widely (median n ≈ 2–3,
  not 0.5) and rails — no single positive centred form fits.
- **Inside-out, not core rebuild.** Only **~3 %** of galaxies show an inner-density
  *drop* over the long baseline (the most massive BCGs — e.g. the central galaxy's
  3-kpc Σ falls as it puffs up). For the rest the centre grows, just far less than
  the envelope. The growth gets **more inside-out toward higher z** (`b`: 0.14 →
  0.34 from the low-z to the high-z adjacent pair).

## Implication for the deposition kernel
The exp25 idea that *late mass lands at large radii* (the growing width `σ(t)`) is
**directionally right**, but the **centred-Gaussian shape is wrong**. The data say
the right primitive is either (a) **multiplicative power-law amplification** of the
existing profile (`Σ → Σ·(R/R₀)^b`, which flattens it), or equivalently (b) an
**outer-weighted / shell-like deposit** rather than a centred one. A forward model
should deposit new mass with a profile that *peaks well outside the centre* and
leaves the core nearly fixed — not a centred Gaussian that piles mass at R = 0.

## Caveats
- **Sparse time sampling** (5 epochs) and **z ≤ 2** only (massive galaxies are
  compact ~exponential there). Adjacent-pair `ΔΣ` is **noisy** (small Δt, lower Σ),
  worst at high z — robust trends use the stacked median / long baseline.
- `intensity` is the isophote surface density; integrating `2πR dR` overshoots the
  CoG by the ellipticity factor `1/(1−e) ≈ 1.3–1.5`, so *shapes* are reliable but
  absolute differential masses are not corrected for ellipticity here.
- Inner R ≲ 6 kpc is marginally resolved (softening); the linear `ΔΣ` there is
  dominated by the steep-core shift, so we restrict shape fits to R ≥ 6 kpc.
- This is the differential of measured surface density at *fixed physical radius*;
  it mixes genuine mass growth with apparent redistribution (size growth). A clean
  separation needs 3-D / Lagrangian tracking we do not have.

## Side analysis — the declining-MAH halos (`declining_mah.py`)
The 695 `mah_declined` galaxies (main-branch mass >5% below peak by z=0.4) are
re-included here. They are **not** preferentially low-mass (median logMh 13.2),
and the declines are modest (median 10%). Grouped by depth / timing / smoothness
into four archetypes that map to the likely causes:

| archetype | n | median decline | what it is |
|---|---|---|---|
| `recent_dip` | 289 | 9% | smooth, peaked <1.5 Gyr ago — benign recent downturn |
| `fluctuating` | 177 | 9% | noisy post-peak (mono 0.57) — **merger-tree / halo-finder artifacts** |
| `sustained` | 170 | 12% | smooth, peaked ≥1.5 Gyr ago — genuine turnover / slow stripping |
| `deep` | 59 | 23% | strong loss — major stripping / disruption |

Most are benign (recent smooth dips) or numerical (fluctuating); genuine sustained
stripping is a minority. See `figures/exp26_declining_mah.*` (top: full MAHs;
bottom: peak-aligned decline shape) and `outputs/declining_mah_classes.fits`.

**Central/satellite:** the sample has no central/satellite flag, but `catgrp_id`
(TNG FoF GroupID) is **unique for all 3388** and they were selected as central
galaxies → **all are centrals, 0 satellites at z=0.4** (so the declining MAH is
*not* current-satellite stripping; it's tree/finder fluctuation or backsplash).
Definitive infall history needs the `diffmah_tng.h5` catalog (`upid`, `t_infall`)
matched by SubhaloID, or the TNG snap-72 group catalog — neither is in our drop.

**Do they still grow stars? Yes (`declining_growth.py`).** Tracking total M*(z)
and the [0.4,0.7] differential density per group vs a non-declined control:

| group | dlogM*_late [0.7→0.4] | dlogM*_total [z2→0.4] | dlogΣ(8kpc) | dlogΣ(60kpc) |
|---|---|---|---|---|
| deep | +0.12 | +0.53 | +0.19 | +0.33 |
| sustained | +0.13 | +0.51 | +0.14 | +0.27 |
| fluctuating | +0.07 | +0.56 | +0.09 | +0.21 |
| recent_dip | +0.10 | +0.48 | +0.09 | +0.28 |
| **control (non-declined)** | **+0.05** | **+0.39** | **+0.03** | **+0.14** |

- **93% still grow M\*** in the late interval; *all* groups grow inside-out after
  the halo turns over. Stellar mass is robust to the halo-mass decline.
- **They grow MORE than the control**, not less — because a declining/​fluctuating
  halo mass tracks **merger activity**, and mergers *deliver* stars. `deep` and
  `sustained` (the strongest events) grow most, with **elevated inner growth**
  (dlogΣ at 8 kpc = 0.14–0.19 vs 0.03 for the control). This is the user's
  point-1 caveat made visible: **major mergers add central stellar mass**, exactly
  where the centred-Gaussian/σ(t) picture is weakest.

## Files
- `differential_profiles.py` — build (cached `outputs/differential_profiles.npz`:
  `Σ` and `ΔΣ` per galaxy/epoch on a common grid) + analysis + figures.
- `declining_mah.py` — classify & visualise the 695 declining-MAH histories.
- `declining_growth.py` — post-turnover stellar growth of the declining groups vs control.
- `outputs/differential_fits.fits` — per (galaxy, pair) descriptors (`dlog_b`,
  `dlog_in/out`, `R_peak`, Sérsic `n`, class).
- `figures/exp26_dlog_growth.*` — the headline (fractional growth + power-law fit).
- `figures/exp26_examples.*`, `exp26_classes.*` — examples + `b` distribution.
- `figures/exp26_declining_mah.*` — the declining-MAH archetypes.
- `figures/exp26_declining_growth.*` — post-turnover stellar growth by archetype.

Run: `PYTHONPATH=. uv run python experiments/exp26_differential_profiles/differential_profiles.py [N] [--refit]`
