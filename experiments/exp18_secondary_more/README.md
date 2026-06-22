# exp18 — More secondary properties: accretion rate and 3D halo shape

## Question
exp16 found halo concentration `c_200c` adds real, MAH-independent information.
Do the other secondary halo properties in the aperture table — accretion rate
`acc_rate` and 3D halo shape (`c_to_a_3d`, `b_to_a_3d`) — help too? And what is
the combined limit of all secondary properties?

## Method
Same test as exp16, on the four CoG-derived annulus masses (5-fold CV). For each
property P: CRPS gain of `DiffMAH(4)+P` over DiffMAH(4); gain of `MAH-PCA(4)+P`
over MAH-PCA(4) (does it help beyond the *full* MAH?); a shuffle control;
`R²(P | DiffMAH)` (how MAH-determined); and the partial correlation of P with the
50–100 kpc annulus at fixed DiffMAH (leftover stellar-mass info). Plus the
combined `DiffMAH + all 4`. n=2481 (slightly below 2545 because `acc_rate` is
missing for ~59 galaxies).

## Key result
**Only concentration helps. Accretion rate is MAH-redundant; 3D halo shape is
independent of the MAH but carries no stellar-mass information.**

| property | gain / DiffMAH | gain / MAH-PCA(4) | shuffle | R²(P\|MAH) | partial corr (50–100) |
|---|---|---|---|---|---|
| **c_200c** | **+4.4%** | **+2.3%** | −0.0% | 0.24 | **+0.27** |
| acc_rate | +0.9% | −0.0% | −0.1% | 0.35 | −0.11 |
| c_to_a_3d | +0.5% | +0.1% | −0.0% | 0.04 | +0.07 |
| b_to_a_3d | +0.7% | +0.3% | −0.0% | 0.03 | +0.08 |
| **all 4** | +4.8% | +2.4% | — | — | — |

- **`c_200c` is the only useful one** — and "all 4 together" (+2.4% on MAH-PCA)
  is essentially `c_200c` alone (+2.3%); accretion rate and shape add nothing on
  top of it.
- **`acc_rate` is MAH-redundant.** It helps a little over the *smoothed* DiffMAH
  (+0.9%) but **nothing over MAH-PCA(4)** (−0.0%), and it is the most
  MAH-determined property (R²=0.35) — unsurprising, since accretion rate *is* a
  MAH quantity. Its small DiffMAH gain is just MAH detail the 4-param smoothing
  dropped, recovered by the richer MAH-PCA.
- **3D halo shape is independent but irrelevant.** `c_to_a_3d`/`b_to_a_3d` are
  *not* MAH-determined (R²≈0.03–0.04 — genuinely independent halo information),
  but they carry almost no information about the central stellar mass (gains
  ≲0.3% on MAH-PCA, partial corr ≈0.07). Independence of the MAH does not imply
  relevance to the galaxy.
- All shuffle controls collapse to ≈0, confirming the (null and non-null)
  results are real.

Panel B makes the distinction explicit: concentration has *both* substantial
leftover stellar-mass info (partial corr +0.27) *and* moderate MAH-independence;
accretion rate is MAH-determined (high R²) with no leftover info; shape is
MAH-independent (R²≈0) but with no leftover info.

## Decision
- **`c_200c` is the one secondary property worth adding** to the emulator (it is
  also portable). `acc_rate` and 3D shape are not — drop them.
- Revises the roadmap: among MAH-derived properties, accretion rate behaves as
  exp06 expected (redundant), but concentration does not (exp16/17). The
  distinguishing feature is whether the property carries *structural* information
  (inner density) the MAH does not fully set.
- The secondary-property axis is now exhausted for this dataset: the achievable
  ceiling with halo features is DiffMAH + c_200c (≈ MAH-PCA + c_200c). Remaining
  scatter is intrinsic (exp13/15) — genuinely new info would need environment /
  initial conditions, not these halo summaries.
