# exp34 — asymptotic total stellar mass from CoG extrapolation

> **RESULT (2026-07-12, n=2397 x 5 epochs).** The power-law density tail
> extrapolates the CoG cleanly (truncation-validated at 0.0026 dex over the
> last factor 1.5 in radius, 0% failures) and the beyond-aperture mass is
> REAL and large where it matters: median f_out = 12% at z=0.4, rising to
> **26% for the massive quartile** (and ~40% at the very top), falling to 1%
> by z=2. The exp33 "unphysical" transport basin (which parked ~15% of stars
> beyond the aperture) was a caricature of a true effect; the physical box
> (full visibility) over-corrected — exactly matching their +3-4 point
> held-out penalty. The differential-deposition measurement gives the
> transport model its first direct empirical constraint on where new mass
> lands.

## Method
Tail forms fitted to the outer CoG (R in [50, 148] kpc, linear mass):
power `M(<R) = M_tot - A R^-a` (density Sigma ~ R^-(a+2)) and exponential.
Validation is internal and honest: fit only R <= 80/100 kpc, predict the
MEASURED M(<148) — power wins at 0.0026 dex (expo 0.0035; 0% fit failures).
Form disagreement -> per-galaxy systematic (median 0.046 dex at z=0.4,
0.005 at z=2). Terminology: "beyond-aperture fraction" f_out, not "ICL"
(user: observational ICL definitions overlap heavily with R < 150 kpc light).

## Results
| | z=0.4 | z=0.7 | z=1.0 | z=1.5 | z=2.0 |
|---|---|---|---|---|---|
| median f_out | 0.123 | 0.089 | 0.064 | 0.039 | 0.014 |
| 16–84% | 0.05–0.44 | 0.03–0.35 | 0.02–0.37 | 0.01–0.32 | 0.00–0.17 |

f_out vs mass (z=0.4): 9%/8%/12% for the lower quartiles -> **26% for
logM* 11.5–12.4** (running median reaches ~0.4 at logM* > 12).

**Differential deposition** (median fraction of inter-snapshot mass growth
landing at R>50 / R>100 kpc): grows toward low z and high mass — massive
tercile: 23%/6% (z=2->1.5) up to **37%/11% (z=0.7->0.4)**. Late-time growth
of massive centrals is strongly outskirt-weighted (the ex-situ channel seen
directly); this is the model-free constraint the transport kernel's width law
must reproduce.

## Caveats
- f_out is an R->infinity integral; galaxies whose tail exponent rails at
  a ~ 0 (panel B spike) have slowly-converging tails and soft (high) f_out.
  The validated statement is the FORM over the last factor 1.5 in radius.
- **Recommendation for the transport refit**: normalize to a finite
  extrapolated mass, M_tot(<500 kpc) (small extrapolation lever, contains the
  budget that matters), with the aperture fraction M(<148)/M(<500) as the
  fitted datum per epoch — rather than the formally infinite total.

## Files
- `run.py` (demo: exact synthetic recovery of both forms; truncation test
  separates true from wrong form). Output: `outputs/asymptotic_total.npz`
  (M_tot per form, f_out, tail slopes, form systematics).
