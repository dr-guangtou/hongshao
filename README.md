# HongShao: A Recipe to Make a Massive Galaxy

**HongShao** is a research project to develop the **"Ultimate SHMR"** — an
assembly-resolved, profile-level extension of the stellar–halo mass relation
(SHMR) for massive quenched central galaxies.

## Goal

The classical SHMR compresses two complex objects into two scalars:
`P(M_star | M_halo)`. HongShao aims to replace those scalars with the full
objects on each side:

- **Halo side:** the main-branch mass assembly history (MAH), `M_peak(z)`.
- **Galaxy side:** the full projected stellar mass profile / curve of growth (CoG).

The target is the conditional distribution

```
P(theta_prof(z) | theta_MAH, M_h(z), z)
```

connecting *how a halo was assembled* to *how stellar mass is distributed* in
its central galaxy. The long-term dream is a forward model that paints
realistic, redshift-dependent stellar profiles onto halos in N-body
simulations.

## Immediate question

> At fixed `M_peak(z=0.4)`, does the main-branch halo assembly history improve
> the prediction of central-galaxy stellar mass profiles?

A controlled, falsifiable first step — even a null result is informative.

## Data

- 3388 massive halos at `z = 0.4`, selected with `M_peak(z=0.4) > 10^13 Msun`.
- Halo side: main-branch MAHs as `M_peak(z)` across snapshots.
- Galaxy side: projected stellar mass density profiles of the centrals.

## Status

Early, exploratory, high-risk. This is a **research** project — expect parallel
directions, throwaway analyses, and dead ends. The scientific context and
candidate directions live in `doc/`:

- [`doc/ultimate_shmr_context.md`](doc/ultimate_shmr_context.md) — background, motivation, references.
- [`doc/ultimate_shmr_possible_directions.md`](doc/ultimate_shmr_possible_directions.md) — concrete analysis directions and a suggested project sequence.
