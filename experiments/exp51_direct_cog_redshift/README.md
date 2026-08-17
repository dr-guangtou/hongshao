# exp51 — Redshift evolution of the direct DiffMAH-to-CoG map

Status: COMPLETE.

## Question

Can the direct halo-to-curve-of-growth conversion from exp50 be extended from
z=0.4 to z=2 without reverting to opaque profile principal components? Can it
use only halo information available at the prediction epoch, and can it make
temporally coherent stochastic galaxy histories?

## Sample and profile representation

The analysis uses 2,395 central galaxies with valid measured progenitor curves
of growth at z=0.4, 0.7, 1.0, 1.5, and 2.0. Two known broken progenitor matches
are removed with the mass-drop criterion established in exp37. Analyses that
also require fitted concentration histories use their 2,365-galaxy overlap.

The five readable profile coordinates are

\[
\theta_{\rm CoG} =
(\log M_{*,148},\operatorname{logit}f_{*,2},\log R_{20,\rm out},
 \log\Delta R_{20-50,\rm out},\log\Delta R_{50-80,\rm out}).
\]

Here, \(f_{*,2}\) is the stellar-mass fraction inside 2 kpc and the three radii
describe quantiles of the remaining stellar mass outside 2 kpc. This differs
from exp50 because one valid z=2 galaxy has 81.1% of its measured mass inside
2 kpc: its ordinary total-mass R20 and R50 are both unresolved below the
0.3-kpc grid limit. The outer-mass coordinates retain this genuinely compact
galaxy rather than discard it or invent sub-grid radii.

The representation itself remains accurate. Its median error over the full
measured curve is 0.00369, 0.00367, 0.00392, 0.00415, and 0.00408 dex from
z=0.4 through z=2, respectively. The matched three-component PCA errors are
0.00334, 0.00339, 0.00374, 0.00385, and 0.00349 dex. Thus the readable
coordinates cost at most 0.00059 dex in median reconstruction error while
remaining physically interpretable.

## Two prediction questions

The experiment keeps two scientifically different uses separate.

1. The **descendant-conditioned map** uses the complete DiffMAH parameters and
   z=0.4 concentration of the final descendant to predict all five progenitor
   profiles. It is appropriate for generating the history of a population
   selected at z=0.4, but it uses halo growth that occurs after a high-redshift
   prediction epoch.
2. The **epoch-local map** refits DiffMAH using only the halo history at cosmic
   times up to the prediction epoch and uses concentration at that epoch. It is
   the portable model for painting a halo catalog selected independently at
   that redshift.

All quoted scores are five-fold held-out measurements. Profile CRPS is the
mean continuous ranked probability score over the 24 measured radii, in dex;
lower values are better.

## Result 1 — the readable map remains competitive with PCA at every epoch

For the descendant-conditioned degree-2 map, readable-coordinate profile CRPS
is 0.06253, 0.06651, 0.07236, 0.08067, and 0.09354 dex from z=0.4 through
z=2. The like-for-like PCA reference gives 0.06223, 0.06618, 0.07167,
0.08050, and 0.09294 dex. The readable map is worse by only 0.00018--0.00070
dex, smaller at every epoch than the PCA score's measured 0.00117--0.00333 dex
fold-to-fold standard deviation. It therefore passes the predeclared parity
test at all five epochs.

The nonlinear relation matters increasingly toward high redshift. Relative to
the same readable linear model, the degree-2 relation lowers profile CRPS by
3.42% at z=0.4, 8.33% at z=0.7, 16.40% at z=1, 21.63% at z=1.5, and 21.47%
at z=2. A universal linear conversion is therefore inadequate even though the
five profile coordinates themselves remain stable.

## Result 2 — descendant-conditioned coefficients evolve smoothly

At each interior snapshot, the fit was repeated after excluding that snapshot
and interpolating the readable model coefficients from the other redshifts.
At z=0.7, z=1, and z=1.5, the interpolated profile CRPS values are 0.06733,
0.07267, and 0.08160 dex, respectively. Direct fits at those epochs give
0.06651, 0.07236, and 0.08067 dex, so interpolation is worse by only 1.23%,
0.43%, and 1.15%. All are within the predeclared 5% tolerance and substantially
better than copying the nearest available epoch, which gives 0.07530, 0.08198,
and 0.10665 dex. The descendant-conditioned direct map can therefore be made
continuous in redshift by interpolating its coefficients.

This test does not yet establish a continuous-redshift epoch-local map. The
epoch-local DiffMAH parameters and concentration change with the prediction
epoch, so that model needs common physical scaling of its evolving inputs
before the same closure test is meaningful.

## Result 3 — an epoch-local DiffMAH conversion is feasible and stronger at high z

The official TNG DiffMAH history was truncated at each prediction epoch and
refitted for every galaxy, with the catalog's Msun/h masses converted to Msun.
All 2,365 galaxies received valid fits. The full 11,825-fit calculation took
37.86 seconds. Median DiffMAH reconstruction errors are 0.0707, 0.0681,
0.0651, 0.0588, and 0.0505 dex from z=0.4 through z=2; the smaller high-z
values partly reflect the shorter fitted time interval and should not be read
as stronger parameter identification.

Adding epoch-local DiffMAH and concurrent concentration to current halo mass
lowers profile CRPS by 23.60%, 20.79%, 23.46%, 25.41%, and 24.55% relative to
current halo mass alone across the five epochs. At z=1.5 and z=2, the
epoch-local model gives 0.07197 and 0.08363 dex, respectively, compared with
0.07847 and 0.09107 dex for the complete-descendant-DiffMAH model with the
same concurrent concentration. The portable model is better by 8.28% and
8.17%. The high-redshift result therefore does not depend on information from
the halo's future; describing the assembly history in the correct local time
frame is more useful.

Mass-conditioned shuffle controls show that the pairing between a galaxy and
its real assembly shape becomes more important with redshift. Compared with
current halo mass alone, real epoch-local DiffMAH plus concentration lowers
profile CRPS by 23.60% at z=0.4 and 24.55% at z=2, while shuffling DiffMAH
shape between halos in narrow current-mass bins lowers it by only 19.29% and
10.85%, respectively. The real assembly pairing therefore contributes an
additional 4.31 percentage points at z=0.4 and 13.69 percentage points at z=2.

## Result 4 — concentration should be evaluated at the prediction epoch

With the complete descendant DiffMAH features held fixed, replacing z=0.4
concentration with concentration at the prediction epoch lowers profile CRPS
by 3.33% at z=0.7, 4.66% at z=1, 2.93% at z=1.5, and 2.23% at z=2. Shuffling
the concurrent concentrations within halo-mass bins removes the improvement
and makes the score 0.10--2.30% worse than the z=0.4-concentration reference.
This is independent evidence that concentration carries information not
contained in DiffMAH and that its time label matters.

For a linear model, adding all earlier concentration measurements available by
the prediction epoch improves on concurrent concentration alone by 3.40% at
z=0.4 and 2.06% at z=0.7, but by at most 0.17% at z>=1. This says that extended
concentration history is useful mainly when a substantial history is actually
available; it is not a uniform extra predictor at every epoch.

## Result 5 — coherent stochastic histories are necessary and approximately work

The held-out readable-coordinate residuals have an epoch-to-epoch AR(1)
coefficient of 0.660. Conditional-mean predictions are too persistent: the
Spearman rank correlation of galaxy half-mass radius between z=0.4 and z=2 is
0.591 in the mean predictions, compared with 0.330 in the TNG profiles.
Correlated stochastic draws lower this to 0.228. They also reproduce adjacent-
epoch size correlations reasonably well: the truth values are 0.816, 0.819,
0.727, and 0.706, while the mean values across draws are 0.787, 0.779, 0.712,
and 0.630. The stochastic layer substantially repairs the conditional mean's
over-coherence, although it slightly underestimates persistence over the full
z=0.4-to-2 baseline.

## Result 6 — symbolic regression does not yet provide a useful universal law

At z=2, a nested-fold symbolic search was applied only to the residuals of the
full epoch-local linear relation. It lowers profile CRPS from 0.08804 dex for
the linear model to 0.08717 dex, a 1.00% improvement. A dense degree-2 model
reaches 0.08363 dex, a 5.01% improvement over linear, so the discovered terms
recover only about 20% of the available nonlinear gain. The three fixed terms
found at z=0.4 transfer even less strongly, reaching 0.08758 dex, only 0.52%
better than linear.

No nonlinear term recurs in four of five folds for total stellar mass, the
2-kpc fraction, or the outermost radial gap. One squared-halo-mass term recurs
for the first outer radius, while four coupled mass/transition-time/slope terms
recur for the middle radial gap. This is too target-specific and too weak in
end-to-end prediction to present as a compact physical conversion law. The
data support a direct phenomenological map, but not yet a small set of
redshift-independent symbolic equations.

## Decision

The central idea is supported. A five-coordinate, probabilistic conversion
from DiffMAH and concentration to the full stellar curve of growth works from
z=0.4 to z=2, remains statistically indistinguishable from the PCA target
representation at the present precision, and does not require knowledge of a
halo's future when constructed in epoch-local coordinates.

Retain the readable degree-2 model as the working direct-conversion candidate.
Do not graduate it to the library yet. The next decisive test is a held-epoch
closure test for the epoch-local model using one common scaling convention for
the evolving DiffMAH inputs. After that, the remaining scientific work is to
test whether an explicit low-order redshift dependence can replace independent
snapshot fits without losing the z>=1 nonlinear gain, and to refine the
cross-epoch stochastic process so that both adjacent and endpoint rank
correlations agree with TNG.

## Files and commands

- `model.py`: high-redshift-safe readable profile coordinates and decoder.
- `run.py`: descendant-conditioned five-epoch fit, PCA comparison, coefficient
  closure, controls, coherence model, and primary figures.
- `concentration.py`: concurrent and historical concentration tests.
- `epoch_local.py`: truncated-history DiffMAH fits and portable prediction test.
- `symbolic_z2.py`: nested-fold z=2 residual symbolic search.

Small validations run in less than one minute:

```bash
uv run python experiments/exp51_direct_cog_redshift/run.py demo
uv run python experiments/exp51_direct_cog_redshift/concentration.py demo
EXP51_LOCAL_NMAX=100 uv run python \
  experiments/exp51_direct_cog_redshift/epoch_local.py demo
```

Full runs:

```bash
uv run python experiments/exp51_direct_cog_redshift/run.py fit
uv run python experiments/exp51_direct_cog_redshift/concentration.py fit
uv run python experiments/exp51_direct_cog_redshift/epoch_local.py fit
uv run python experiments/exp51_direct_cog_redshift/symbolic_z2.py
```

Regenerable numerical products and manifests are in `outputs/`; all figures
are in `figures/`. Both directories are gitignored.
