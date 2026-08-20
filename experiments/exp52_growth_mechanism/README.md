# exp52 — the adopted kernel grows massive galaxies by redistribution, not accretion

**Status: DIAGNOSIS. A structural incompatibility between the adopted kernel
and the established physical picture, quantified. No fix proposed here.**

## The physical expectation

For massive central galaxies, late-time growth is ex-situ dominated: continued
dry, mostly minor mergers deposit NEW stellar material in the outskirts. It is
not the redistribution of stars already assembled in the centre. If the model's
late-time outskirt growth is dominated by transport, the model is telling a
story incompatible with that understanding — and it would do so invisibly,
because the objective only ever sees the summed profile.

## The measurement

The kernel deposits mass at time `t_i` with scale `rc0`, and a fraction
`(1 - fc)` later expands to `rc0*(t_k/t_i)^q`. So between epochs an annulus can
grow two ways. Splitting the deposits into those already present at the earlier
epoch (`old`) and those arriving between the two (`new`), the change in a
normalized aperture mass is exactly

    dM_hat = N_k0*(A_old_k0 - A_old_k1)   TRANSPORT       same material, moved
           + (N_k0 - N_k1)*A_old_k1       RENORMALIZATION the M(<500) pinning
           + N_k0*A_new_k0                NEW DEPOSITION

with `N_k = m500_k / M_k(500)`. The transport share below is
transport/(transport + new deposition) — **the model's own physics with the
pinning removed.** Full sample, n = 2397, adopted 1ch-mof theta.

## Result 1 — the outskirts are grown by moving old stars, increasingly so

Transport share of the model's own growth:

| epoch pair | M\*[50,100] | M\*[100,148] |
|---|---|---|
| z=2.0 → 1.5 | 18.0% | 17.4% |
| z=1.5 → 1.0 | 38.4% | 32.8% |
| z=1.0 → 0.7 | 76.9% | 63.3% |
| **z=0.7 → 0.4** | **96.5%** | **91.7%** |
| **z=1.0 → 0.4** | **87.5%** | 77.9% |

At z=2 the model is deposition-dominated and behaves sensibly. **The crossover
happens near z ≈ 1.2 — precisely where the observational picture says
minor-merger accretion should be taking over.** By the final interval, 96.5% of
the model's outskirt growth is redistribution of stars it already had.

## Result 2 — the kernel's own deposition supplies ~11% of the mass growth

M(<500 kpc) growth per interval:

| epoch pair | measured | the kernel's OWN deposition | pinning supplies |
|---|---|---|---|
| z=2.0 → 1.5 | x1.340 | x1.094 | x1.225 |
| z=1.5 → 1.0 | x1.366 | x1.027 | x1.330 |
| z=1.0 → 0.7 | x1.203 | x1.000 | x1.204 |
| z=0.7 → 0.4 | x1.201 | **x0.992** | x1.211 |

Cumulatively from z=2 to z=0.4 the galaxies grow **x2.645** while the kernel's
own deposition supplies **x1.115** — **11% of the log mass growth. The other 89%
is manufactured by the M(<500) normalization**, which pins the model to the
measured total.

After z ≈ 1.5 the deposition machine is effectively off, and in the last
interval the un-normalized mass inside 500 kpc *falls* (x0.992): expansion
carries material out through the aperture faster than new deposits arrive.

Cause: the fitted efficiency window (mu = 1.522, sig = 0.235 in ln(1+z)) peaks
at z = 3.58 with a 1-sigma span of z = 2.6–4.8, and by z = 1 has fallen to
~0.2% of peak. The model deposits **1.5%** of its stellar mass at z < 1 while
the halo assembles **47%** of its mass there.

## Result 3 — the mechanism explains the known central-mass deficit

Transport is strongly NEGATIVE in the centre: the model grows the outskirts by
draining the middle. z=1.0 → 0.4, per aperture, model growth against truth:

| aperture | model dM / truth dM | transport term |
|---|---|---|
| **M(<10)** | **57.3%** | **−8.77e13** (drains) |
| M(<30) | 93.5% | −7.49e13 (drains) |
| M[50,100] | 98.9% | +1.89e13 (fills) |
| M[100,148] | 101.7% | +9.39e12 (fills) |

The model captures only **57% of the central mass growth** while matching the
outskirts to ~1%. **This is a mechanism for the compact-galaxy central-mass
deficit that exp47 and exp48 spent two experiments characterising**: with the
deposition machine off, the only way the kernel can grow an outskirt is to move
material out of the centre, so the centre is systematically under-filled.

## Result 4 — the transport has the RIGHT SOURCE and a REACH ~10x too long

Adiabatic expansion is physically well grounded for massive galaxies: strong,
continued AGN/SMBH feedback offsets baryons outward, the inner potential
re-configures, and the stellar distribution responds. The established result
from theory and hydro simulations is that this **flattens the inner stellar
profile**. It does not build the stellar halo at 50-150 kpc. So some
transformation that lowers the central density with time IS legitimately
needed; large-scale transportation is not.

Measured (`reach.py`, n=2397), for material already assembled at the earlier
epoch — where transport takes mass FROM and delivers it TO:

| shell (kpc) | z=1.0 -> 0.4 | z=2.0 -> 1.0 |
|---|---|---|
| 0-5 | **-93.1%** (drained) | **-100.0%** (drained) |
| 5-10 | -6.9% (drained) | +16.2% |
| 10-30 | +24.8% | +39.1% |
| 30-50 | +16.7% | +13.6% |
| 50-100 | +20.1% | +12.5% |
| 100-148 | +9.0% | +4.8% |
| 148-300 | +11.8% | +5.7% |
| 300-500 | +5.6% | +2.6% |
| beyond 500 | **+12.0%** | +5.5% |
| **delivered inside 50 kpc** | **41.5%** | 68.9% |
| **delivered beyond 50 kpc** | **58.5%** | 31.1% |

**The source is exactly right.** 100% of the drained mass comes from inside
10 kpc, 93% from inside 5 kpc — precisely the region where AGN-driven expansion
should operate. Between z=1 and z=0.4 this moves **12.24% of all deposited
stellar mass** out of the inner 10 kpc, and the half-mass radius of the
assembled stars grows **11.27 -> 19.04 kpc (+69%)**.

**The destination is wrong, and gets worse with time.** By z=1 -> 0.4, 58.5% of
the drained mass is delivered beyond 50 kpc, **29.4% beyond 148 kpc — outside
the measured profile entirely** — and **12.0% beyond 500 kpc, where the
normalization then discards it**. At the earlier pair the reach is far more
defensible (68.9% stays inside 50 kpc); it lengthens as the deposition machine
switches off and transport becomes the only way to grow an outskirt.

**So the defect is specific and quantified: the right physical process, applied
with roughly an order of magnitude too much reach.** This is a sharper target
than "the transport term is wrong" — the term is needed, its radial extent is
not.

## What this does NOT say

- It does not say the model fits badly. It reproduces the profiles at all five
  epochs to ~10-15%. The concern is that it may be right for structurally wrong
  reasons.
- It does not say the model mispredicts total mass growth. **The model does not
  predict mass growth at all** — M(<500) is pinned to the measured value, so
  the deposition history controls only the radial SHAPE.
- **The transport picture has never been tested against TNG's own in-situ /
  ex-situ decomposition.** That is the direct falsification test and is unrun.
  **Checked 2026-08-20: the split is NOT in the local drop.** The historical
  profiles (`save_tng300_072_hist_prof`) carry only isophote quantities —
  `r_kpc, intensity, intens_err, ellipticity, ellip_err, pa, pa_err` plus an
  ellipse-fit diagnostics table. `map_tng100_hist_stellar.hdf5` is **852 MB of
  zeros** (no HDF5 signature, zero bytes at every sampled offset) — a failed or
  sparse download, not data. TNG's public release does provide Stellar Assembly
  supplementary catalogs with in-situ / ex-situ stellar mass per subhalo; the
  RADIAL split we would need here is a further step and its availability has
  not been verified.

## Files

- `decompose.py` — the exact three-term decomposition, all six epoch pairs.
- `reach.py` — where transport takes mass from and delivers it to.
- `figure.py` — `figures/exp52_growth_mechanism.png`.
