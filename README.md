# HongShao: A Recipe to Make a Massive Galaxy

**HongShao** develops the **"Ultimate SHMR"** — an assembly-resolved,
profile-level extension of the stellar–halo mass relation (SHMR) for massive
quenched central galaxies. Instead of `P(M_star | M_halo)`, it predicts the
*full stellar-mass profile* of a central galaxy from *portable, N-body-available
halo properties*, as a calibrated probability distribution.

## What it does

Given a halo's **mass accretion history** (compressed to 4 DiffMAH parameters)
and its **concentration** `c_200c` — five numbers, all measurable in any
gravity-only or hydrodynamic simulation — the emulator predicts the central
galaxy's stellar-mass profile and its scatter:

```python
from hongshao.emulator import fit
from hongshao.profile_emulator import aperture_targets

X = ...                          # (N, 5) halo features [logmp, logtc, early, late, c200c]
Y = aperture_targets(cog, radii, edges_kpc=[10, 30, 50, 100])   # (N, 4) target masses

emu = fit(X, Y)                  # mean + heteroscedastic full covariance
mu, sigma, cov = emu.predict(X_new)        # predictive Gaussian per halo
draws = emu.sample(X_new, size=500)        # generative: correlated mock profiles
```

The model is **probabilistic and generative**: it returns a full covariance, and
`sample()` draws correlated mock profiles that reproduce the real galaxy
population — including its scatter and tails. (Use it this way; the mean alone is
under-dispersed — see the manual.)

### Four prediction modes, one core

The same heteroscedastic core (`hongshao/emulator.py`) drives four targets via
thin builders in `hongshao/profile_emulator.py`:

| mode | target | builder |
|---|---|---|
| 1 | stellar masses in fixed **kpc** apertures/annuli | `aperture_targets` |
| 2 | masses in **effective-radius (Re)** bins | `re_targets` |
| 3 | the cumulative **curve of growth** | `fit_profile` (CoG) |
| 4 | the 1-D **surface-density profile** Σ(R) | `fit_profile` (density) |

A thin **deformation layer** (`hongshao/forward.py`) exposes five physically
labeled knobs (`d0`, `d_slope`, `d_out`, `f_ab`, `s`) for tuning the relation in
external inference, with the defaults reproducing the calibrated model exactly.

## Key results

- **Assembly history + concentration beat a plain `M_halo` SHMR by +27–31% in the
  outskirts** (50–100 kpc), removing ~half the residual scatter a current-mass
  relation leaves there.
- Calibrated 5-fold-CV performance (CoG-derived masses, TNG300 z=0.4, n≈2539):
  **CRPS ≈ 0.083 dex, conditional-coverage gap ≈ 0.01**.
- The predictable signal lives in the **total mass + a concentration-driven shape
  mode**; the rest of the profile-shape scatter is intrinsic (projection / ICL /
  triaxiality) and no feature set or parameterization we tried extracts more.

## Install & run

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
PYTHONPATH=. uv run python -m hongshao.emulator           # self-check (reproduces the headline numbers)
PYTHONPATH=. uv run python -m hongshao.profile_emulator   # all four modes
PYTHONPATH=. uv run python -m hongshao.forward            # the deformation layer
```

Each library module has a runnable `__main__` self-check.

## Documentation

- **[`doc/emulator_manual.md`](doc/emulator_manual.md)** — the user manual: how to
  **train on your own simulation**, **predict each observable**, and the
  **caveats & gotchas**. Start here to use the emulator.
- [`doc/ultimate_shmr_context.md`](doc/ultimate_shmr_context.md) — scientific background, motivation, references.
- [`doc/ultimate_shmr_possible_directions.md`](doc/ultimate_shmr_possible_directions.md) — analysis directions and project sequence.
- `experiments/expNN_*/README.md` — the experiment log (each result, with its decision).

## Data

- TNG300-1 at `z = 0.4`; massive central halos selected with `M_peak(z=0.4) >
  10^13 M_sun` (≈2539 in the working sample).
- **Features (portable):** DiffMAH parameters of the main-branch MAH (`logmp`,
  `logtc`, `early`, `late`) + the NFW concentration `c_200c`.
- **Target:** the curve-of-growth-derived projected stellar masses (the
  observation-relevant quantity).

## Scope

HongShao predicts **stellar masses and profiles from halos** — and stops there.
Weak-lensing / clustering predictions, summary statistics, likelihoods, and
samplers are deliberately *out of scope*; they belong in a separate inference
repository that *consumes* HongShao's predictions. The deformation layer is the
hand-off boundary.

## Status

A working, validated emulator with a stable public API. Still a **research**
project — expect the experiment log to keep growing — but the core library
(`emulator` → `profile_emulator` → `forward`) is feature- and target-complete for
this dataset.
