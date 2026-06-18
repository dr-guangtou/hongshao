# Ultimate SHMR: Possible Directions with the Current Simulation Data

**Current dataset:** 3388 massive halos at `z = 0.4`, selected with `Mpeak(z=0.4) > 10^13 Msun`, with main-branch `Mpeak(z)` across snapshots and stellar mass density profiles of the corresponding central galaxies.

**Immediate scientific objective:** determine whether the main-branch halo MAH contains statistically useful information about the stellar mass density profile or CoG of massive central galaxies beyond what is already contained in final halo mass.

**Long-term objective:** build a forward, data-driven model that can populate massive halos in N-body simulations with realistic projected stellar mass profiles as a function of redshift.

---

## 0. Recommended philosophy

The first project should not attempt to prove that halo MAH fully determines galaxy profiles. It should ask a more controlled and publishable question:

> At fixed `Mpeak(z=0.4)`, does the main-branch halo assembly history improve the prediction of central-galaxy stellar mass profiles?

The answer can be quantified through reductions in scatter, increases in explained variance, better recovery of aperture masses, and improved prediction of profile-shape parameters.

The core model should be probabilistic:

\[
P(\theta_{\rm prof}\mid M_{\rm peak}(z=0.4),\theta_{\rm MAH},z=0.4),
\]

not deterministic:

\[
\theta_{\rm prof}=f(\theta_{\rm MAH}).
\]

The residual scatter is scientifically meaningful. It represents merger granularity, orbital stochasticity, baryonic physics, satellite stellar mass scatter, and any information not captured by the smooth main-branch MAH.

---

## 1. Data preparation and quality-control layer

### 1.1 Standardize the radial profiles

Before testing any assembly connection, define a consistent representation of the stellar mass profiles.

Recommended inputs:

- projected stellar mass density profile: `Sigma_star(R)`;
- projected CoG:

\[
M_{\star,\rm 2D}(<R)=2\pi\int_0^R \Sigma_\star(R')R'dR';
\]

- aperture masses in physically interpretable bins:
  - `Mstar(<10 kpc)`;
  - `Mstar(10--30 kpc)`;
  - `Mstar(30--50 kpc)`;
  - `Mstar(50--100 kpc)`;
  - `Mstar(<100 kpc)`;
  - if available, `Mstar(100--300 kpc)` or central+ICL mass.

Use the same radial grid for all galaxies, preferably logarithmic in physical kpc. A practical first choice is something like:

\[
R = 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200\ {\rm kpc},
\]

but the exact range should match the reliable simulation measurement range.

### 1.2 Work primarily in projected CoG space

Use the projected CoG as the main profile observable:

\[
\log M_\star(<R_i).
\]

Reasons:

- monotonic and stable;
- directly comparable to observations;
- naturally yields aperture masses;
- less noisy than differentiating `Sigma_star(R)`;
- avoids 3D deprojection assumptions.

Use the differential profile `Sigma_star(R)` mainly for diagnostic plots and slope calculations.

### 1.3 Normalize profiles in multiple ways

Use several profile representations, because each isolates different information:

1. **Absolute CoG:**

\[
\log M_\star(<R).
\]

This includes total stellar mass and profile shape.

2. **Normalized CoG:**

\[
\log \frac{M_\star(<R)}{M_\star(<R_{\rm max})}.
\]

This isolates profile shape at fixed aperture-defined total mass.

3. **Residual CoG at fixed total mass:**

Fit and subtract the mean relation with `Mstar(<Rmax)` or `Mpeak(z=0.4)`.

This is useful for finding assembly-history effects in profile shape rather than total mass.

### 1.4 Define halo-side quantities

From raw `Mpeak(z)` compute both direct summaries and, later, DiffMAH fits.

Recommended raw summaries:

- `logM0 = log Mpeak(z=0.4)`;
- `Mpeak(z=0.7)`, `Mpeak(z=1)`, `Mpeak(z=1.5)`, `Mpeak(z=2)`, if snapshots allow;
- fractional mass assembled by redshift:

\[
f(z) = \frac{M_{\rm peak}(z)}{M_{\rm peak}(z=0.4)};
\]

- mass growth between epochs:

\[
\Delta \log M(z_1,z_2) = \log M_{\rm peak}(z_2)-\log M_{\rm peak}(z_1),
\]

where `z2 < z1` means later cosmic time;

- formation redshifts:
  - `z50`: redshift where halo reached 50% of `Mpeak(z=0.4)`;
  - `z75`;
  - `z90`;

- recent growth fraction:

\[
f_{\rm late}=1-\frac{M_{\rm peak}(z=1)}{M_{\rm peak}(z=0.4)};
\]

- early mass fraction:

\[
f_{\rm early}=\frac{M_{\rm peak}(z=2)}{M_{\rm peak}(z=0.4)}.
\]

These raw summaries are important because they are interpretable and can be compared directly with DiffMAH parameters.

---

## 2. Direction A: aperture-mass assembly mapping

This is the lowest-risk first analysis and should be done before fitting complex profile models.

### 2.1 Scientific question

Which radial stellar mass apertures correlate most strongly with which epochs of halo assembly?

Examples:

\[
M_\star(<10\ {\rm kpc}) \leftrightarrow M_{\rm peak}(z\gtrsim 2),
\]

\[
M_\star(50-100\ {\rm kpc}) \leftrightarrow M_{\rm peak}(z\sim 1)\ \text{or late mass growth}.
\]

### 2.2 Suggested analysis

For each aperture mass `Y`, fit nested models:

**Model A0: final halo mass only**

\[
Y = a_0 + a_1\log M_{\rm peak}(z=0.4)+\epsilon.
\]

**Model A1: final halo mass + one MAH summary**

\[
Y = a_0 + a_1\log M_0 + a_2 X_{\rm MAH}+\epsilon.
\]

where `X_MAH` can be `z50`, `f_early`, `f_late`, `Mpeak(z=1)`, `Mpeak(z=2)`, or recent growth rate.

**Model A2: final halo mass + multiple MAH summaries**

\[
Y = a_0 + a_1\log M_0 + \sum_k a_k X_k+\epsilon.
\]

Compare residual scatter and cross-validated predictive accuracy.

### 2.3 Diagnostics

For each radial aperture:

- plot residual aperture mass at fixed `M0` versus each MAH summary;
- compute Spearman/Pearson correlations;
- compute partial correlations controlling for `M0`;
- use k-fold cross-validation to avoid overinterpreting noise;
- identify the redshift `z` where `Mpeak(z)` best predicts each aperture mass at fixed final mass.

A useful visualization:

\[
\rho_{\rm partial}\left[M_\star(R_i<R<R_j), M_{\rm peak}(z) \mid M_{\rm peak}(z=0.4)\right]
\]

as a 2D heatmap over radial aperture and redshift.

### 2.4 Expected outcomes

This direction will tell you whether the basic physical picture is visible in the data:

- inner mass linked to earlier halo assembly;
- outer mass linked to late/intermediate halo growth;
- total mass dominated by final halo mass;
- profile-shape residuals linked to assembly history.

Even a null result is useful: it would show that main-branch MAH alone is insufficient for aperture-level information and motivate merger-tree/secondary-branch extensions.

---

## 3. Direction B: fit DiffMAH and compare to raw MAH summaries

### 3.1 Scientific question

Does the DiffMAH parameterization provide a compact and useful representation of the halo-side information for predicting stellar profiles?

### 3.2 Fit DiffMAH to the 3388 MAHs

Use `Mpeak(z)` across snapshots and fit the DiffMAH model to each halo. Store:

\[
\theta_{\rm MAH}=(\log M_0,\log t_c,\alpha_{\rm early},\alpha_{\rm late},t_{\rm peak}),
\]

plus fit-quality metrics:

- RMS error in `log Mpeak`;
- maximum residual;
- whether the fit is physically plausible;
- whether late-time behavior near `z=0.4` is well captured.

Because the sample is at `z=0.4`, define the reference time carefully. It may be better to fit the history up to `z=0.4` and define an observation-time mass normalization at `t_obs = t(z=0.4)`, rather than blindly using a `z=0` convention.

### 3.3 Compare DiffMAH to raw features

Build predictive models for aperture masses or profile parameters using:

1. raw MAH summaries;
2. DiffMAH parameters;
3. principal components of the raw MAH curves;
4. combinations of the above.

The key comparison:

\[
\theta_{\rm prof} \sim M_0 + \theta_{\rm MAH}^{\rm DiffMAH}
\]

versus

\[
\theta_{\rm prof} \sim M_0 + \{z_{50},z_{75},f_{\rm late},M(z=1),M(z=2)\}.
\]

If DiffMAH performs similarly to raw features with fewer parameters, it is a good basis for the Ultimate SHMR. If raw features perform better, DiffMAH may still be useful as a smooth prior but not as the full predictor.

### 3.4 Important caution

DiffMAH is designed to describe halo growth histories, not necessarily to isolate the aspects of halo assembly most relevant for stellar deposition. It should be treated as a basis, not as a sacred parameterization. The project should compare it against simpler formation-time and mass-growth summaries.

---

## 4. Direction C: radial-DiffMAH profile fitting

This is the most direct realization of the current conceptual idea.

### 4.1 Scientific question

Can the stellar CoGs of massive central galaxies be represented by a low-dimensional, differentiable, DiffMAH-like radial function?

### 4.2 Model the CoG slope

Define:

\[
\beta(R)=\frac{d\ln M_\star(<R)}{d\ln R}.
\]

Use a sigmoid in log-radius:

\[
\beta(R)=\beta_{\rm out}+\frac{\beta_{\rm in}-\beta_{\rm out}}
{1+\exp[(\ln R-\ln R_c)/\Delta_R]}.
\]

Then recover the CoG by integrating:

\[
\ln M_\star(<R)=\ln M_{\star,0}+\int_{\ln R_0}^{\ln R}\beta(R')d\ln R'.
\]

Fit parameters:

\[
\theta_{\rm prof}=(\log M_{\star,0},\beta_{\rm in},\beta_{\rm out},\log R_c,\Delta_R).
\]

### 4.3 Practical parameter constraints

To keep the model physical:

- enforce `Mstar(<R)` monotonic by requiring `beta(R) >= 0`;
- use `beta_out >= 0` but allow values close to zero for outer saturation;
- require `beta_in > beta_out` for standard profiles, unless data show otherwise;
- constrain `R_c` to lie within or near the fitted radial range;
- constrain `Delta_R > 0`.

A convenient reparameterization:

\[
\beta_{\rm out}=\mathrm{softplus}(b_{\rm out}),
\]

\[
\beta_{\rm in}=\beta_{\rm out}+\mathrm{softplus}(\Delta b),
\]

\[
\Delta_R=\mathrm{softplus}(d_R).
\]

### 4.4 Fit quality tests

For each galaxy:

- fit radial-DiffMAH to the CoG;
- compute residuals in `log Mstar(<R)`;
- compute residuals in aperture masses;
- test whether residuals are systematic at small or large radii;
- identify galaxies that require more than one transition.

Compare to alternatives:

- single projected Einasto-like CoG;
- double-component model;
- PCA basis;
- spline basis.

### 4.5 Scientific output

If radial-DiffMAH fits most CoGs to acceptable accuracy, then profile diversity can be compressed into a small parameter vector. The next step is to model:

\[
P(\theta_{\rm prof}\mid M_0,\theta_{\rm MAH}).
\]

This becomes the first concrete version of the Ultimate SHMR.

---

## 5. Direction D: PCA / low-rank CoG emulator

This is the most flexible data-driven baseline.

### 5.1 Scientific question

How many independent modes are needed to describe the CoGs of massive central galaxies?

### 5.2 Method

Construct a matrix:

\[
X_{ij}=\log M_{\star,i}(<R_j)
\]

or normalized residuals:

\[
X_{ij}=\log\frac{M_{\star,i}(<R_j)}{M_{\star,i}(<R_{\rm max})}.
\]

Subtract the mean relation, optionally at fixed stellar mass or halo mass, and perform PCA.

Keep the first few coefficients:

\[
\theta_{\rm prof}^{\rm PCA}=(c_1,c_2,c_3,...).
\]

Then regress:

\[
c_k \sim M_0 + \theta_{\rm MAH}.
\]

### 5.3 Why this is useful

PCA does not impose an analytic profile form. It provides:

- a flexible benchmark for radial-DiffMAH;
- a way to measure intrinsic dimensionality;
- an interpretable set of modes after inspection;
- a check on whether one or two radial transitions are enough.

### 5.4 Key outputs

- fraction of CoG variance captured by first 1, 2, 3 modes;
- correlation of each mode with `M0`, formation time, late-growth fraction, DiffMAH parameters;
- reconstruction accuracy compared with radial-DiffMAH.

A useful possible result:

- PC1 ≈ total mass / normalization;
- PC2 ≈ compact versus extended structure;
- PC3 ≈ outer envelope curvature or transition-radius variation.

If PC2 or PC3 correlates with MAH at fixed `M0`, that is direct evidence for an assembly-resolved profile SHMR.

---

## 6. Direction E: two-component profile model

### 6.1 Scientific question

Does a two-component compact-plus-accreted basis improve fit quality and physical interpretability relative to a single radial-DiffMAH model?

### 6.2 Model structure

Use:

\[
M_\star(<R)=M_{\rm comp}(<R)+M_{\rm ext}(<R).
\]

The components can be represented by:

- two radial-DiffMAH CoGs;
- two projected Einasto-like profiles;
- one compact component plus one flexible outer kernel.

The goal is not necessarily to recover true in-situ/ex-situ masses, but to create a physically interpretable latent basis.

### 6.3 Possible parameterization

\[
M_{\rm comp}(<R)=M_{\rm comp,tot}G_{\rm comp}(R\mid R_{\rm comp},s_{\rm comp}),
\]

\[
M_{\rm ext}(<R)=M_{\rm ext,tot}G_{\rm ext}(R\mid R_{\rm ext},s_{\rm ext}).
\]

Then connect the component masses to MAH summaries:

\[
\log M_{\rm comp,tot} \sim M_{\rm peak}(z\sim2),
\]

\[
\log M_{\rm ext,tot} \sim \Delta M_{\rm peak}(z\sim1\rightarrow0.4).
\]

And connect scale radii to formation/deposition epochs:

\[
R_{\rm comp} \sim \text{early compactness},
\]

\[
R_{\rm ext} \sim \text{late accretion / deposition radius}.
\]

### 6.4 When this is worth doing

Use this direction if:

- single radial-DiffMAH leaves coherent residuals;
- PCA suggests more than one shape mode beyond normalization;
- high-redshift extension becomes a priority;
- simulation data include true in-situ/ex-situ decompositions for validation.

---

## 7. Direction F: profile--MAH cross-correlation maps

This is a powerful exploratory analysis that requires no profile fitting.

### 7.1 Construct residual profile vectors

At each radius, define residuals after removing final halo mass dependence:

\[
\Delta \log M_\star(<R_i)
=\log M_\star(<R_i)-\langle\log M_\star(<R_i)\mid \log M_0\rangle.
\]

Similarly, define MAH residuals:

\[
\Delta \log M_{\rm peak}(z_j)
=\log M_{\rm peak}(z_j)-\langle\log M_{\rm peak}(z_j)\mid \log M_0\rangle.
\]

Then compute:

\[
C(R_i,z_j)=\mathrm{corr}\left[\Delta \log M_\star(<R_i),\Delta \log M_{\rm peak}(z_j)\right].
\]

Or use differential aperture masses instead of CoG values:

\[
M_\star(R_i<R<R_{i+1}).
\]

### 7.2 Expected insight

This map directly asks:

> Which radial parts of the galaxy remember which epochs of halo growth?

Possible outcomes:

- inner CoG residuals correlate with early MAH residuals;
- outer apertures correlate with late/intermediate growth;
- all radii correlate only with final halo mass;
- no significant MAH correlation remains after controlling for final mass.

This should be one of the first analyses, because it will inform whether radial-DiffMAH or two-component models are likely to succeed.

---

## 8. Direction G: null models and shuffled controls

Inspired by the globular-cluster semi-analytic work, build null models to test whether profile--halo relations are merely hierarchical averaging.

### 8.1 Null model 1: final-mass-only model

Predict profiles using only `Mpeak(z=0.4)`:

\[
P(\theta_{\rm prof}\mid M_0).
\]

This is the baseline classical/profile SHMR.

### 8.2 Null model 2: shuffled MAHs at fixed final mass

Within narrow bins of `M0`, randomly shuffle MAHs among halos. Refit the MAH-conditioned model. If predictive power remains unchanged, the MAH information is not truly being used.

### 8.3 Null model 3: randomized radial profiles at fixed stellar mass and halo mass

Shuffle profile shapes at fixed `Mstar(<Rmax)` and `M0`. This tests whether any detected relation is driven only by total stellar mass.

### 8.4 Null model 4: PCA randomization

Shuffle PCA coefficients independently at fixed `M0` to test whether covariance among radial bins contains assembly information.

### 8.5 Success criterion

The MAH-conditioned model should outperform all null models in cross-validation, especially for profile-shape quantities and not just total stellar mass.

---

## 9. Direction H: build the first probabilistic Ultimate SHMR emulator

Once a profile representation is chosen, build the conditional distribution:

\[
P(\theta_{\rm prof}\mid M_0,\theta_{\rm MAH}).
\]

### 9.1 Minimal emulator

Use a multivariate Gaussian model:

\[
\theta_{\rm prof} = A x_{\rm halo}+b+\epsilon,
\]

where

\[
x_{\rm halo}=(\log M_0,z_{50},f_{\rm late},f_{\rm early},...)
\]

or

\[
x_{\rm halo}=(\log M_0,\log t_c,\alpha_{\rm early},\alpha_{\rm late},t_{\rm peak}).
\]

and

\[
\epsilon\sim\mathcal{N}(0,\Sigma_{\rm prof}).
\]

### 9.2 More flexible emulator

If the linear model is insufficient:

- polynomial regression;
- Gaussian process regression;
- mixture density network;
- normalizing flow;
- small JAX/NumPyro neural network with uncertainty.

Do not start with the most flexible model. First establish the information content using linear and low-order models.

### 9.3 Output

Given a new N-body halo with `Mpeak(z)`:

1. compute or fit MAH features;
2. sample `theta_prof` from the conditional distribution;
3. generate projected CoG and/or `Sigma_star(R)`;
4. optionally assign ellipticity/orientation for mock images.

This is the first working version of a profile-painting model for massive halos.

---

## 10. Direction I: redshift-evolution extension

The long-term dream is a model that explains profile evolution across redshift. With only `z=0.4` profiles, this cannot yet be fully calibrated, but the architecture can be designed now.

### 10.1 Time-dependent version

At observation redshift `z_obs`, use the MAH only up to that epoch:

\[
M_{\rm h}(t\le t_{\rm obs}).
\]

Generate:

\[
P\left[M_\star(<R,z_{\rm obs})\mid M_{\rm h}(t\le t_{\rm obs}),z_{\rm obs}\right].
\]

### 10.2 Integral growth model

A more physical version:

\[
M_\star(<R,z_{\rm obs})
=
\int_0^{t_{\rm obs}}
\dot M_{\rm h}(t)
\epsilon_\star(t)
K(<R\mid t,z_{\rm obs})dt.
\]

For two components:

\[
M_\star(<R,z_{\rm obs})=
\int_0^{t_{\rm obs}}
\left[\dot M_{\rm in}(t)K_{\rm in}(<R\mid t,z_{\rm obs})+
\dot M_{\rm acc}(t)K_{\rm acc}(<R\mid t,z_{\rm obs})\right]dt.
\]

### 10.3 What can be tested now

Even with only `z=0.4` profiles, you can test whether profile radii correspond to different MAH epochs using cross-correlation maps. If inner and outer profiles respond to different MAH epochs, it supports the redshift-evolution model design.

---

## 11. Suggested first project sequence

### Phase 1: exploratory diagnostics

1. Clean and standardize stellar profiles.
2. Compute CoGs and aperture masses.
3. Compute raw MAH summaries.
4. Plot profile--MAH cross-correlation maps at fixed final mass.
5. Identify which radial ranges are most sensitive to which MAH epochs.

Deliverable: evidence for or against assembly information in profiles.

### Phase 2: profile compression

1. Fit radial-DiffMAH CoGs.
2. Perform PCA on normalized CoGs.
3. Compare profile reconstruction accuracy.
4. Decide whether one-component radial-DiffMAH is enough or whether two-component/double-sigmoid models are needed.

Deliverable: a low-dimensional `theta_prof` representation.

### Phase 3: MAH compression

1. Fit DiffMAH to all MAHs.
2. Compare DiffMAH parameters with raw MAH summaries.
3. Test which representation better predicts `theta_prof`.

Deliverable: a low-dimensional `theta_MAH` representation.

### Phase 4: conditional emulator

1. Fit `P(theta_prof | M0)` baseline.
2. Fit `P(theta_prof | M0, theta_MAH)` assembly model.
3. Use cross-validation and null models.
4. Quantify information gain.

Deliverable: first version of the Ultimate SHMR emulator.

### Phase 5: mock profile painting

1. Apply the emulator to halos in an N-body catalog.
2. Generate CoGs and surface-density profiles.
3. Compare mock stellar profile distributions and aperture-mass functions with the simulation calibration sample.
4. Prepare for comparison with observations/lensing/clustering.

Deliverable: profile-painting prototype.

---

## 12. Recommended first figures

1. **Example MAHs and profiles:** show a few halos with similar `M0` but different MAHs and their corresponding CoGs.
2. **Profile diversity at fixed halo mass:** normalized CoGs in narrow `M0` bins.
3. **MAH diversity at fixed halo mass:** raw `Mpeak(z)/M0` curves.
4. **Aperture mass versus halo mass:** reproduce classical/profile SHMR behavior.
5. **Residual aperture mass versus formation time:** inner/outer aperture comparison.
6. **Radius--redshift correlation map:** `corr(profile residual at R, MAH residual at z | M0)`.
7. **Radial-DiffMAH fit examples:** best, median, worst residuals.
8. **PCA modes of CoGs:** first 3 modes and their correlations with MAH.
9. **Variance explained:** final mass only versus final mass + MAH.
10. **Null model comparison:** true MAH versus shuffled MAH.

---

## 13. First technical implementation checklist

- [ ] Convert snapshot redshifts to cosmic time.
- [ ] Interpolate all MAHs onto a common redshift/time grid.
- [ ] Ensure `Mpeak(z)` is monotonic along the main branch; handle numerical artifacts.
- [ ] Define `M0 = Mpeak(z=0.4)`.
- [ ] Compute `Mpeak(z)/M0` and growth-rate summaries.
- [ ] Compute CoGs from `Sigma_star(R)`.
- [ ] Define radial fitting range.
- [ ] Compute aperture masses.
- [ ] Fit simple regressions: aperture mass versus `M0` and MAH summaries.
- [ ] Build correlation heatmaps.
- [ ] Fit radial-DiffMAH profiles.
- [ ] Run PCA on CoGs.
- [ ] Fit DiffMAH to MAHs.
- [ ] Compare raw MAH, PCA-MAH, and DiffMAH as halo-side predictors.
- [ ] Construct null/shuffled controls.
- [ ] Build first conditional emulator.

---

## 14. Decision points after first analysis

### If MAH strongly predicts profile residuals

Proceed with radial-DiffMAH or PCA-based profile emulator. Focus on deriving a compact, interpretable Ultimate SHMR and testing redshift-evolution predictions when additional snapshots are available.

### If MAH weakly predicts profile residuals

Test whether merger-tree granularity is required. Add secondary variables such as number of major mergers, largest progenitor mass ratio, satellite accretion history, or ex-situ fraction if available.

### If radial-DiffMAH fits poorly

Use PCA or a spline-based CoG representation as the profile basis. Radial-DiffMAH can remain an interpretive toy model.

### If radial-DiffMAH fits well but MAH does not predict its parameters

The stellar profiles are low-dimensional, but their variation is not captured by main-branch MAH. This would motivate adding merger-tree or baryonic latent variables.

### If final mass dominates everything

This is still useful: it would show that the profile-level SHMR is mostly a higher-dimensional mass proxy at this halo-mass/redshift range. The next test would focus on residuals, covariance, and redshift evolution.

---

## 15. Most promising immediate path

The most efficient sequence is:

1. **Do aperture-level diagnostics first.**  
   They are easy, interpretable, and directly tied to existing HSC results.

2. **Construct the radius--redshift correlation map.**  
   This will reveal whether different radial zones correspond to different MAH epochs.

3. **Fit radial-DiffMAH and PCA in parallel.**  
   Radial-DiffMAH gives interpretability; PCA gives a flexible benchmark.

4. **Fit DiffMAH only after raw MAH diagnostics.**  
   This avoids overcommitting to a parameterization before confirming the signal.

5. **Build the first emulator only after profile compression is validated.**  
   The emulator should model the distribution of profile parameters, not raw noisy profiles.

A compact first scientific claim could be:

> In massive halos at `z=0.4`, the stellar mass profiles of central galaxies occupy a low-dimensional space. At fixed final halo mass, part of this profile diversity is correlated with main-branch halo assembly history. This motivates an assembly-resolved, profile-level extension of the SHMR.

That would be the first concrete step toward the Ultimate SHMR.
