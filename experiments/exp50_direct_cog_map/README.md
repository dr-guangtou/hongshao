# exp50 — Direct DiffMAH-to-CoG conversion

Status: COMPLETE.

## Question

Can the statistical connection between a halo's DiffMAH parameters and a
massive central galaxy's curve of growth (CoG) be written as a compact,
phenomenologically readable map without losing the predictive quality of the
PCA profile emulator?

The model is explicitly a conditional distribution, not a deterministic law:

\[
P(\theta_{\rm CoG}\mid \theta_{\rm DiffMAH},c_{200c}).
\]

## Method

- Single epoch: z=0.4.
- Sample: 2,539 TNG300 massive central galaxies with valid DiffMAH fits,
  concentration, and CoG-derived stellar masses.
- Galaxy target: the cumulative CoG on the standard 2.0--148.2 kpc grid.
- Halo inputs: `dmah_logmp`, `dmah_logtc`, `dmah_early`, `dmah_late`, and
  `c_200c`.
- Interpretable CoG coordinates: total stellar mass within 148.2 kpc, the
  stellar-mass fraction within 2 kpc, R20, R50, and R80. The radius intervals
  are modeled in logarithmic form so that their order is valid by construction.
- Decoder: monotone PCHIP interpolation of enclosed stellar-mass fraction
  against logarithmic radius.
- Mean relations: a linear map and the established seven-term degree-2 map.
- Five shared held-out folds. Every fit, PCA basis, residual covariance, and
  symbolic search is trained without the held-out fold.
- Controls: final halo mass only, DiffMAH without concentration, DiffMAH-shape
  parameters shuffled within final-mass bins, and concentration shuffled within
  final-mass bins.
- Reference: the three-component PCA CoG emulator on the identical folds.
- Generative predictions: 32 draws per halo from the fitted correlated residual
  distribution.
- Symbolic regression: searched only for compact corrections to the complete
  linear map, independently inside each training fold.
- Gated extension: repeated the fit for the 2,366 galaxies with five-epoch
  concentration histories from z=2.0 to z=0.4, including a shuffled-history
  control.

The 2-kpc fraction was added after a small preflight test. Quantile radii alone
reconstructed the CoG beyond 5 kpc accurately but did not fix the innermost
measured point. Adding the measured central fraction reduced the median
within-galaxy representation error from 0.027 dex to about 0.004 dex.

## Results

### 1. The readable coordinates preserve the CoG accurately

Using total mass, the 2-kpc fraction, and R20/R50/R80 reconstructs each measured
CoG with a median root-mean-square error of 0.00437 dex. The three-component PCA
representation reaches 0.00337 dex, so the readable representation is 0.00100
dex worse but remains far below the error of the halo-to-galaxy prediction.
Adding R90 improves representation error to 0.00316 dex, 0.00021 dex better than
PCA, but does not improve held-out halo prediction. The R20/R50/R80 version is
therefore the adopted compact representation.

### 2. The direct halo-to-CoG map matches the PCA emulator

The selected degree-2 map to the readable coordinates obtains a held-out
profile CRPS of 0.06327 dex. The matched degree-2 PCA emulator obtains 0.06317
dex, so the readable map is worse by only 0.00010 dex; that difference is
smaller than the measured 0.00024 dex fold-to-fold standard deviation of the
PCA score. It therefore passes the predeclared predictive-parity test.

The selected readable map has a median within-galaxy CoG root-mean-square error
of 0.08560 dex, compared with 0.08377 dex for the PCA emulator, so PCA remains
better by 0.00182 dex on the conditional mean. The median largest fractional
stellar-mass error beyond 5 kpc is 24.64% for the readable map, compared with
24.25% for PCA, making the readable map worse by 0.39 percentage points.

For stellar mass within 10 kpc and in the 10--30, 30--50, and 50--100 kpc
annuli, the readable map's held-out root-mean-square errors are 0.1115, 0.1524,
0.1670, and 0.1665 dex. The PCA reference gives 0.1106, 0.1513, 0.1637, and
0.1667 dex, respectively. The two coordinate systems are therefore practically
equivalent for the intended observable, although neither predicts an individual
galaxy exactly.

### 3. Assembly history and concentration both contain real information

A final-halo-mass-only map has a median CoG root-mean-square error of 0.10890
dex. Adding the four DiffMAH parameters reduces this to 0.09362 dex, an
improvement of 0.01527 dex relative to final mass alone. Shuffling the three
DiffMAH shape parameters within final-mass bins degrades the error to 0.09472
dex. This modest but repeatable loss shows that the map uses the pairing between
each halo's history and its galaxy rather than only the population distribution.

Adding the correctly paired z=0.4 concentration to DiffMAH reduces the linear
map's median CoG error from 0.09362 to 0.08693 dex. Shuffling concentration
instead gives 0.09338 dex, nearly returning to the DiffMAH-only reference. The
incremental concentration information is therefore real and is not explained
by its marginal distribution at fixed halo mass.

### 4. Most individual shape coordinates remain only partly predictable

For the adopted degree-2 map, the held-out fraction of coordinate variance
explained is about 0.87 for total stellar mass, 0.31 for the 2-kpc central
fraction, 0.36 for R20, 0.31 for the logarithmic R20-to-R50 interval, and 0.15
for the R50-to-R80 interval. This is why the appropriate result is a conditional
distribution: the halo variables constrain the mean CoG, but substantial and
radially correlated galaxy-to-galaxy scatter remains.

### 5. Correlated draws restore the population width

The conditional mean is too narrow as a galaxy population. In the total-mass
versus R50 plane its energy distance from the TNG population is 3.94 times the
finite-sample floor; one correlated generative draw reduces this to 1.35 times
the floor, which is better by a factor of 2.9. In the inner-versus-outer
stellar-mass plane, the corresponding ratios are 3.46 for the conditional mean
and 1.21 for a draw, an improvement by a factor of 2.9. The residual covariance
is therefore an essential part of the conversion rather than optional noise.

With 32 draws per halo, the nominal central 68% and 90% pointwise intervals
contain the measured CoG values 65.2% and 85.6% of the time, respectively. The
PCA reference gives similarly low values of 64.8% and 84.9%, so this comparison
does not identify a coordinate-specific calibration failure. More draws or an
analytic interval calculation are needed before interpreting the absolute
coverage shortfall.

### 6. Symbolic regression finds small, stable corrections—not a new law

Adding only terms selected in at least four of five training folds reduces the
held-out profile CRPS from 0.06577 dex for the linear map to 0.06421 dex, a 2.38%
improvement relative to the linear reference. The dense degree-2 map reaches
0.06327 dex, a 3.80% improvement, so the sparse symbolic corrections recover
about 62% of the available nonlinear gain.

The recurring corrections are concentration squared for total stellar mass,
the product of DiffMAH transition time and early growth rate for the 2-kpc
fraction, and the product of final halo mass and early growth rate for the
R20-to-R50 interval. No nonlinear term recurs in at least four folds for R20 or
the R50-to-R80 interval. These terms are compact empirical corrections in
standardized variables; they should not be interpreted as fundamental physical
laws.

### 7. Concentration history adds independent predictive information

For the 2,366 galaxies with complete five-epoch concentration histories, adding
the history to the current-concentration degree-2 model reduces held-out profile
CRPS from 0.06318 to 0.06135 dex, a 2.89% improvement relative to the
current-concentration model. Shuffling the histories within final-mass bins
instead changes CRPS to 0.06325 dex, which is 0.12% worse than the reference.
The real-history improvement is positive in all five held-out folds, whereas
the shuffled history is worse in all five. Concentration history therefore
contains a small but genuine profile-wide signal beyond DiffMAH and present-day
concentration.

## Decision

The proposed direct conversion is well-founded and feasible as an interpretable,
probabilistic phenomenological model. It reaches the predictive quality of the
PCA emulator while exposing total mass, central fraction, and enclosed-mass
radii as readable intermediate quantities. It is not a deterministic physical
conversion: only total stellar mass is tightly predicted, and the correlated
residual distribution remains essential.

Keep the R20/R50/R80 coordinate model as a successful experimental alternative
to PCA, but do not replace the library emulator yet. The concentration-history
gain warrants a separate multi-epoch experiment because its input is more
expensive and its portability beyond this TNG300 sample has not been tested.
Treat the sparse symbolic terms as hypotheses for physical interpretation, not
as evidence for unique equations.

## Limitations

- The result is restricted to TNG300 massive central galaxies at z=0.4.
- Stellar profiles are single projected, CoG-derived measurements.
- The experiment tests DiffMAH and concentration features already present in
  the repository; it does not establish that these are sufficient halo state
  variables.
- The symbolic equations depend on the chosen standardized variables and CoG
  coordinates.
- The concentration-history extension uses a smaller overlap sample and should
  be repeated across epochs before becoming a model input.

## Run

```bash
# sub-minute mechanics and small-sample validation
PYTHONPATH=. uv run python experiments/exp50_direct_cog_map/run.py demo
EXP50_NMAX=400 EXP50_N_DRAW=16 PYTHONPATH=. uv run python \
  experiments/exp50_direct_cog_map/run.py fit

# full five-fold comparison
PYTHONPATH=. uv run python experiments/exp50_direct_cog_map/run.py fit

# nested-fold symbolic regression
PYTHONPATH=. uv run python experiments/exp50_direct_cog_map/symbolic.py

# gated concentration-history test
PYTHONPATH=. uv run python \
  experiments/exp50_direct_cog_map/concentration_history.py
```

Figures and numerical outputs are regenerable and gitignored. The committed
experiment consists of this record and the scripts needed to reproduce them.
