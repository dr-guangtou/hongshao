# exp53 — the deposition-only boundary test

Plan agreed 2026-08-20, after exp52 found the adopted kernel grows galaxies by
REDISTRIBUTION rather than by accretion. Branch `exp53-deposition-only`.

## The question

The adopted `1ch-mof` kernel deposits mass at time `t_i` with a scale
`rc(t_i)`, then TRANSPORTS a fraction of it outward to `rc*(t_k/t_i)**q`. exp52
measured that this transport term, not the deposition term, supplies the late
outskirt growth: 96.5% of the model's own `M*[50,100]` growth over z=0.7->0.4,
and 87.5% over z=1.0->0.4. Its own deposition supplies ~11% of the z=2->0.4
mass growth; the other 89% is the `M(<500)` renormalization. exp52 also showed
this explains the exp47/exp48 compact-galaxy central deficit — with deposition
effectively off after z~1.5, filling an outskirt requires emptying the centre,
and the model's `M(<10)` growth over z=1.0->0.4 is 57.3% of the truth's.

So: **how much of the fit actually needs transport, and what does the model
look like without it?** `q = 0` is a clean nested special case — `rcw = rc0`
makes the core fraction `fc` irrelevant and the kernel becomes pure deposition.

## Design constraints (all measured last session; none are negotiable)

1. **GRADED, not binary.** `q` is fitted at 0.908 inside a `[0, 3]` box,
   comfortably interior, so under the present setup the data actively WANTS
   transport. Profile the loss over `q` with `q = 0` as the endpoint. And
   PROFILE it: exp49 showed a frozen-parameter SLICE and a profile differ in
   SIGN in this model. Re-optimize every other parameter at each grid point.
2. **The efficiency window stays FREE.** Transport and the window are
   entangled: at `sig = 0.237` the late deposits are nearly massless, so a
   no-transport model with a frozen window would have no way to change shape at
   late times and its failure would be uninterpretable. `mu`, `sig` and the
   `sig` conditioning row are free in every fit here.
3. **Judge against the MEASURED added-light kernel, not the loss.**
   `exp38/outputs/stage0_kernels.npz` holds it (3 logM* terciles x 4 adjacent
   epoch pairs x 23 radii), measured in exp38 stage 0.2 as centrally peaked at
   3-5 kpc, Sersic `n` = 2.1-3.1, kernel half-mass radius 15-47 kpc. exp53
   applies the IDENTICAL operator to model CoGs, cell by cell.
4. **Sersic is back in.** exp38 stage 1 rejected `sersic (n free)` for a
   "LOWER rail 3/5 epochs (n-scale degeneracy; needs an R50 reparameterization)"
   — a fixable numerical artifact, and the family the measured kernel points
   at. exp53 supplies the reparameterization for EVERY family.

## The reparameterization (the enabling change)

Every family's deposit scale is expressed as its own **half-mass radius**
`R50`, not its family-specific scale parameter:

| family  | Sigma(R)                     | scale from R50                                  |
|---------|------------------------------|-------------------------------------------------|
| moffat  | `(1+(R/rc)^2)^-gam`          | `rc = R50 / sqrt(2^(1/(gam-1)) - 1)`            |
| sersic  | `exp(-(R/a)^(1/n))`          | `a  = R50 / b_n^n`, `b_n = gammaincinv(2n, 1/2)`|
| expo    | sersic with `n = 1` fixed    | same                                            |
| gauss   | sersic with `n = 0.5` fixed  | same (`= s*sqrt(2 ln 2)` — the incumbent)       |

Two things this buys. (a) The scale and shape parameters stop competing to set
the same physical radius, which is the degeneracy that railed Sersic's `n`.
(b) `log_R50` means the SAME physical quantity in every family, so the family
comparison, the `300 kpc` deposit ceiling and the conditioning slopes are all
on common ground. exp49 measured that a 300 kpc ceiling costs 1.5e-5 refitted
— free — and it matters more here precisely because `rc` is not comparable
across families.

## Theta layout

```
[log_R50, g, q, mu, sig, <shape>] + [R50:logMh, R50:c200c, R50:fz2]
                                  + [sig:logMh, sig:c200c, sig:fz2]
```
`<shape>` is `gam` (moffat) or `n` (sersic) and is absent for expo/gauss:
12 parameters with a free shape, 11 without; one fewer again when `q` is fixed.
Everything else is inherited unchanged from the exp38/exp40 production harness
— lognormal efficiency in `ln(1+z)`, `M(<500)` normalization, joint fit on
`KS = [0, 1, 2, 3]` (the adopted exp40 `z <= 1.5` scope), the production
`Objective()`.

## Stages

**0. Equivalence.** The R50-parameterized moffat kernel must reproduce
`exp38/stage2_multiepoch.py::model_cogs(..., "1ch-mof")` to machine precision
on real galaxies at the adopted theta, and `q = 0` must make `fc` provably
irrelevant. Nothing downstream is meaningful without this.

**A. The `q` profile (moffat).** `q` on a grid `0.00 ... 1.20`, the other 11
free, L-BFGS-B with real bounds, warm-started outward from the free-`q`
optimum; the free-`q` fit and the `q = 0` endpoint each get independent
multi-start. Report the loss curve AND, at every point, the judged observables
— this is where a slice and a profile were measured to disagree in sign.

**B. The family shootout.** `{moffat, sersic, expo, gauss}` x `{q = 0, q free}`,
three starts each, full sample. `gauss` is the falsification control: it is the
wingless incumbent-of-record and should lose.

**C. The verdict.** Every fitted theta passes ONE scorer: exp47's
`judge.verdict` + `physics_gates`, plus exp53's kernel scorecard (stage 0.2
operator vs the measured kernel; exp52's `dlogSigma` at 2-3 kpc and the
inside-out slope `b`). Ranked by the kernel scorecard and the gates, NEVER by
the loss.

## The falsifiable prediction

exp52: transport drains `M(<10 kpc)` to 57.3% of the truth's z=1.0->0.4 growth.
**Removing transport should IMPROVE central masses.** If `M(<10)` and the
compact-galaxy `M(<5)` bias do not improve as `q -> 0`, exp52's mechanism story
is wrong and this branch has falsified it.

## Warnings carried in

- A SLICE is not a PROFILE — always re-optimize the others.
- The dev100 subsample has inverted the full-sample answer on `g` twice. Every
  reported number here is full-sample (n = 2397); dev is for wiring only.
- Multi-start is not optional: this loss has at least two basins.
- Fits run SERIAL, sharded across independent processes (the in-loss pool has
  hung before). Stall alarms must watch the PYTHON worker, not the `uv` /
  `caffeinate` wrappers, which read 0% CPU during a healthy run.
- `scipy` Nelder-Mead freezes zero-initialized parameters — use
  `hongshao.fitting.initial_simplex` if Nelder-Mead is ever used here.
