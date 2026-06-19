# exp04 — First halo→profile model: P(profile | M0, assembly history)

## Question

The payoff of exp01–exp03: can we **predict a galaxy's full light profile from
its halo's mass and growth history**, and how much does the history add over mass
alone — across the whole profile?

## Method

Cross-validated (5-fold) multi-output **linear** regression predicting the
24-point curve of growth from halo features:

- baseline `X = [logM0]`
- with history `X = [logM0, Mpeak(z=0.7/1/1.5/2), z50, z75, z90]`
- shuffle control: history features permuted among same-M0 halos.

Done for the absolute curve of growth and for the mass-normalized **shape**.
Plus a "profile painting" demo. Driver: `run.py`. (n = 2538.)

## Key results

**1. Adding assembly history predicts the full profile ~23% better.**

| target | RMS (M0) | RMS (M0+history) | shuffled | improvement |
|---|---|---|---|---|
| full curve of growth | 0.152 dex | 0.118 dex | 0.152 dex | **22.6%** |
| shape (mass divided out) | 0.082 dex | 0.076 dex | 0.082 dex | **7.3%** |

Shuffling gives ~0%, so the gain is real. This generalizes exp01's ~20%
aperture result to the entire profile.

**2. Where history helps depends on what you ask** (Panel C):
- For the **absolute** profile, the improvement *grows with radius* (13% in the
  center to ~30% at 150 kpc): history strongly improves the prediction of total
  and outer mass.
- For pure **shape**, the gain is smaller (7.3%) and concentrated at small radii:
  history mostly refines the inner concentration.

So most of the absolute-profile gain is better prediction of *how much* stellar
mass there is (early-forming halos are more massive at fixed M0); a smaller,
real part is better prediction of *how it is distributed*.

**3. Profile painting works** (Panel D). Same-M0 halos with different histories
get visibly different predicted profiles. The model learns: at fixed final halo
mass, **early-forming halos are more spatially extended**, late-forming halos
more concentrated — consistent with Pillepich et al. (2014), where older halos
have shallower / more extended stellar halos (more time to build an accreted
envelope).

## Interpretation & caveats

- This is a working first-generation **Ultimate-SHMR prototype**: a model that
  paints assembly-dependent profiles onto halos of given mass + history.
- The model is deliberately **linear** (the minimal emulator). It already
  captures the signal; non-linear models (GP / normalizing flow) and explicit
  scatter modeling are later refinements.
- The shape gain (7.3%) is modest, as expected — MAH is partial information and
  single-projection triaxiality adds irreducible noise (`AGENTS.md`).

## Decision

The Phase 1–4 arc is complete with a **positive, physically-coherent result**:
halo assembly history measurably and sensibly improves profile prediction. Good
point to consolidate and choose among next directions:

- predict the **radial-DiffMAH parameters** instead of the raw CoG (generative,
  guarantees monotonic profiles);
- add **secondary halo properties** (concentration / tidal field) — the user
  noted MAH is not the whole story;
- move from a mean predictor to a **probabilistic** emulator (scatter / flows);
- **redshift evolution**, once profiles at other epochs are available.
