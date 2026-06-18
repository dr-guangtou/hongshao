# Ultimate SHMR: Scientific Context and Background

**Project concept:** develop an assembly-resolved, profile-level extension of the stellar--halo mass relation (SHMR) for massive quenched central galaxies. The long-term goal is a forward model that maps dark-matter halo assembly histories to statistically representative stellar mass density profiles or curves of growth (CoGs), enabling rapid population of massive halos in N-body simulations with extended stellar mass distributions.

**Current data available:**

- Sample: 3388 massive halos at `z = 0.4`.
- Halo selection: `Mpeak(z=0.4) > 10^13 Msun`.
- Halo-side data: main-branch mass assembly histories, expressed as `Mpeak(z)` across all simulation snapshots.
- Galaxy-side data: stellar mass density profiles of the central galaxies.
- Optional next step: fit DiffMAH parameters to each halo MAH.

---

## 1. The conceptual target: from SHMR to Ultimate SHMR

The classical SHMR relates one scalar galaxy quantity to one scalar halo quantity:

\[
P(M_\star \mid M_{\rm h}, z).
\]

This has been foundational for galaxy--halo connection modeling, but it compresses two complex objects into two numbers: the galaxy's stellar mass and the halo's mass. Both quantities have definition-dependent ambiguities: stellar masses depend on aperture, profile fitting, mass-to-light assumptions, treatment of low-surface-brightness outskirts, and satellite/ICL definitions; halo masses depend on overdensity definition, halo finder, pseudo-evolution, and whether `Mpeak`, `M200c`, `Mvir`, or another measure is used.

The proposed **Ultimate SHMR** treats the full stellar mass distribution as the galaxy-side object and the halo assembly history as the halo-side object:

\[
P\left[\Sigma_\star(R,z)\ \middle|\ M_{\rm h}(t), z\right]
\]

or, in terms of a compact profile parameterization,

\[
P\left(\theta_{\rm prof}(z) \mid \theta_{\rm MAH}, M_{\rm h}(z), z\right).
\]

Here:

- `theta_MAH` is a low-dimensional representation of the halo mass assembly history, e.g. DiffMAH parameters or derived formation-time / accretion-rate summaries.
- `theta_prof` is a low-dimensional representation of the stellar mass profile, e.g. radial-DiffMAH parameters, projected Einasto-like parameters, double-component parameters, or PCA coefficients of CoGs.

The classical SHMR becomes a marginal projection of this deeper relation:

\[
P(M_{\star,\rm tot} \mid M_{\rm h})
= \int P(\theta_{\rm prof}\mid M_{\rm h})\,
\delta\left[M_{\star,\rm tot} - M_{\star,\rm tot}(\theta_{\rm prof})\right]d\theta_{\rm prof}.
\]

Aperture-based relations, such as `Mstar(<10 kpc)` or `Mstar(50--100 kpc)` versus halo mass, are also projections of the same parent relation.

---

## 2. Why massive quenched central galaxies are the right regime

The idea is most plausible for massive central galaxies, especially at low and intermediate redshift, because their stellar mass growth is dominated by the two-phase assembly process:

1. **Early compact/in-situ growth:** high-redshift halo growth fuels gas cooling and star formation, building a dense central stellar body.
2. **Late accretion/ex-situ growth:** after quenching, further stellar mass growth is dominated by stars accreted through mergers, tidal stripping, and satellite disruption, building an extended stellar halo and potentially an ICL component.

Rodriguez-Gomez et al. (2016) used Illustris to show that the two-phase picture is most applicable to the most massive galaxies. In their analysis, ex-situ fractions rise strongly with stellar mass, reaching very large values for the most massive systems; in-situ stars dominate the inner regions, while ex-situ stars are deposited at larger galactocentric distances, with merger mass ratio affecting the spatial distribution of accreted stars.

This spatial segregation provides the central physical intuition behind the Ultimate SHMR:

\[
\text{early halo growth} \rightarrow \text{inner stellar mass concentration},
\]

\[
\text{late halo growth/mergers} \rightarrow \text{extended stellar envelope}.
\]

The proposal does **not** assume a deterministic mapping for individual galaxies. Instead, it assumes that the main-branch MAH contains useful low-order information about the statistical distribution of stellar profiles, with real stochasticity left as intrinsic scatter.

---

## 3. Observational motivation from profile-dependent halo mass

A key empirical motivation comes from work showing that the stellar mass distribution of massive galaxies contains more halo information than total stellar mass alone.

### 3.1 HSC massive-galaxy profile work

Huang et al. (2018) used deep HSC imaging to measure individual stellar mass density profiles of massive galaxies to approximately 100 kpc, showing substantial diversity in outer envelopes and a dependence of profile structure on environment/halo mass.

Huang et al. (2020) used HSC weak lensing to show a tight connection between stellar mass distribution and halo mass. At fixed total stellar mass, massive galaxies with more extended stellar mass distributions live in more massive halos. A two-parameter description involving total/large-aperture stellar mass and inner stellar mass provides a better high-mass galaxy--halo connection than the scalar SHMR.

Huang et al. (2022) further showed that outer stellar mass, especially the stellar mass in the 50--100 kpc radial range, is an effective halo-mass proxy for massive galaxies at `0.2 < z < 0.5`, with scatter competitive with richness-based cluster mass proxies and potentially reduced sensitivity to projection/mis-centering effects.

These works motivate a hierarchy:

1. classical scalar SHMR: `Mstar` versus `Mhalo`;
2. aperture-level SHMR: `Mstar(<10 kpc)`, `Mstar(10--100 kpc)`, `Mstar(50--100 kpc)` versus `Mhalo`;
3. profile-level SHMR: full `Sigma_star(R)` or `Mstar(<R)` versus `Mhalo`;
4. assembly-resolved profile SHMR: full profile versus halo MAH.

The current project aims at level 4.

---

## 4. Simulation/theory motivation: stellar halos as assembly-history tracers

### 4.1 Illustris stellar halo slopes and assembly history

Pillepich et al. (2014) analyzed stellar halos in Illustris and found that the logarithmic slope of the outer stellar density profile is strongly related to halo mass: more massive halos have shallower stellar halos. Importantly, at fixed halo mass, recently formed halos or halos with larger accreted stellar fractions have shallower stellar halos than older analogues.

This is a direct precedent for the Ultimate SHMR. It implies that the **shape** of the stellar halo, not just its total mass, carries information about halo assembly history.

### 4.2 In-situ/ex-situ spatial decomposition

Rodriguez-Gomez et al. (2016) showed in Illustris that accreted stars become increasingly important at high stellar mass and that in-situ and ex-situ components are spatially segregated. This supports two-component representations of massive-galaxy profiles:

\[
\rho_\star(r) = \rho_{\rm in}(r) + \rho_{\rm acc}(r),
\]

or in projection,

\[
\Sigma_\star(R) = \Sigma_{\rm in}(R) + \Sigma_{\rm acc}(R).
\]

The decomposition need not be literal in observations. It can instead be used as a latent modeling basis: a compact component associated with early assembly and an extended component associated with late assembly.

---

## 5. DiffMAH and the low-dimensional halo-side representation

DiffMAH provides a compact, differentiable model for halo mass assembly histories. The model approximates halo growth as a power law in cosmic time with a time-dependent slope that transitions smoothly from an early fast-accretion regime to a late slow-accretion regime:

\[
M_{\rm peak}(t) \sim M_0 \left(\frac{t}{t_0}\right)^{\alpha(t)}.
\]

The time-dependent logarithmic slope is represented by a sigmoid-like transition between early and late slopes:

\[
\alpha(t) \rightarrow \{\alpha_{\rm early},\alpha_{\rm late},t_c\}.
\]

The public implementation includes parameters such as:

\[
\theta_{\rm MAH}=(\log M_0,\log t_c,\alpha_{\rm early},\alpha_{\rm late},t_{\rm peak}).
\]

DiffMAH is valuable here because it turns noisy merger-tree histories into a smooth, low-dimensional, differentiable representation. The Ultimate SHMR can then be framed as a conditional distribution:

\[
P(\theta_{\rm prof}\mid \theta_{\rm MAH},M_{\rm h}(z),z).
\]

This is analogous to the logic of Diffstar, which uses DiffMAH as the foundation for modeling galaxy star-formation histories. The proposed project is the spatial analogue: using halo assembly histories to predict stellar mass deposition profiles rather than only star-formation histories or total stellar masses.

---

## 6. Lessons from Lackner & Ostriker (2010)

Lackner & Ostriker (2010), *Dissipational versus Dissipationless Galaxy Formation and the Dark Matter Content of Galaxies*, developed a deliberately simplified analytic framework for two extreme modes of elliptical galaxy formation:

1. **Dissipational formation:** gas loses orbital/thermal energy, sinks to the center, forms compact stars, and induces dark-matter contraction.
2. **Dissipationless formation:** stellar clumps sink through dynamical friction, transfer orbital energy to the dark matter, and build the galaxy through dry accretion.

The work is not a direct model for the Ultimate SHMR because it assumes a final stellar profile and studies the dark-matter response. However, it offers a useful principle:

\[
\text{depositing dry-accreted stellar mass at small radius has an energy cost.}
\]

This motivates a physically inspired prior or kernel for accreted stellar deposition:

\[
dM_{\star,\rm acc}(R)
= \int dt\,\epsilon_{\rm acc}(t)\,\dot M_{\rm h}(t-\tau)\,
K_{\rm acc}(R\mid M_{\rm h}(t),R_{\rm vir}(t),c(t),\eta).
\]

The key borrowable idea is not their exact shell calculation, but the idea that the radial scale of the accreted component should be connected to halo binding energy, accretion time, and dry merger energetics. In a data-driven model, this should enter as a weak physical regularizer, not as a hard deterministic law.

---

## 7. Lessons from El-Badry et al. (2018/2019) on globular-cluster assembly

El-Badry et al. built a semi-analytic model for globular-cluster populations on dark-matter merger trees. The model attaches simple formation rules to progenitor halos, propagates the resulting GC populations through hierarchical assembly, and predicts a wide range of observables.

The most important lesson for the Ultimate SHMR is methodological: global present-day scaling relations can arise through hierarchical summation even if the underlying formation rules are not tightly tied to halo mass. In their GC model, a near-linear total GC mass--halo mass relation at high mass can emerge largely because massive halos assemble from many progenitors; the central limit theorem reduces scatter.

For the Ultimate SHMR, this is a warning:

- a tight relation between outer stellar mass and halo mass may partly reflect hierarchical averaging;
- the deeper assembly information may live in profile shape, covariance across radial bins, redshift evolution, and residuals at fixed halo mass.

The useful architecture is:

\[
\text{history} \rightarrow \text{simple local/event rule} \rightarrow \text{propagation} \rightarrow \text{multi-observable validation}.
\]

For stellar profiles, an analogous model could be:

\[
M_\star(<R,z_{\rm obs})
= \int_0^{t_{\rm obs}} dt\,
\left[\dot M_{\rm in}(t)G_{\rm in}(<R\mid t,z_{\rm obs})
+ \dot M_{\rm acc}(t)G_{\rm acc}(<R\mid t,z_{\rm obs})\right].
\]

This kind of model naturally supports the “stop at each redshift” dream: at any `z_obs`, integrate only up to `t(z_obs)` and generate the statistically representative CoG at that redshift.

---

## 8. 2D versus 3D profile modeling

The project can be formulated in either 3D density space or projected 2D profile space.

### 8.1 2D-first approach

The directly observed quantities are projected stellar mass density profiles and projected CoGs:

\[
\Sigma_\star(R),
\]

\[
M_{\star,\rm 2D}(<R)=2\pi\int_0^R \Sigma_\star(R')R' dR'.
\]

A 2D-first model is attractive because it avoids deprojection assumptions, ellipticity/inclination degeneracies, and unnecessary geometric restrictions. It is also the natural space for comparison with observed stellar profiles, weak lensing, clustering, and survey selection.

### 8.2 3D latent layer

A 3D model can be useful as a physical regularizer or simulation interface:

\[
\rho_\star(r,z) = \rho_{\rm in}(r,z)+\rho_{\rm acc}(r,z),
\]

followed by projection:

\[
\Sigma_\star(R)=2\int_R^\infty \frac{\rho_\star(r)rdr}{\sqrt{r^2-R^2}}.
\]

This is useful for connecting to deposition radii, halo binding energy, and hydro-simulation particle data. But it also imposes stronger assumptions about geometry and profile form.

### 8.3 Recommended compromise

For the first-generation Ultimate SHMR, the recommended core model is projected CoG-first:

\[
P\left[M_{\star,\rm 2D}(<R,z)\mid \theta_{\rm MAH},M_{\rm h}(z),z\right].
\]

A 3D layer can be added later as an optional latent model, especially for comparison to simulations or particle-level mock generation.

---

## 9. Radial DiffMAH as a profile parameterization

A promising profile model is to reuse the DiffMAH mathematical philosophy in radius. Define the CoG logarithmic slope:

\[
\beta(R) \equiv \frac{d\ln M_\star(<R)}{d\ln R}.
\]

Then model it as a smooth transition between inner and outer slopes:

\[
\beta(R)=\beta_{\rm out}+
\frac{\beta_{\rm in}-\beta_{\rm out}}
{1+\exp[(\ln R-\ln R_c)/\Delta_R]}.
\]

The cumulative profile is then reconstructed by integration:

\[
\ln M_\star(<R)=\ln M_{\star,0}+\int_{\ln R_0}^{\ln R}\beta(R')d\ln R'.
\]

This parameterization is attractive because:

- it is monotonic by construction if `beta(R) >= 0`;
- it is differentiable;
- it has interpretable parameters: inner slope, outer slope, transition radius, transition width, normalization;
- it mirrors the early/late slope transition in DiffMAH;
- it can approximate Einasto-like or smooth double-component CoGs over finite radial ranges.

For more complex profiles, a double-sigmoid radial DiffMAH or a two-component radial-DiffMAH model can be used.

---

## 10. Main scientific questions for the current data

With 3388 halos and central-galaxy stellar profiles at `z = 0.4`, the immediate questions are:

1. **Can a low-dimensional profile representation fit the simulated CoGs accurately?**
   - radial DiffMAH;
   - projected Einasto-like model;
   - double-component model;
   - PCA / basis expansion.

2. **How much profile variance is explained by present-day halo mass alone?**
   - baseline model: `theta_prof ~ Mpeak(z=0.4)`.

3. **How much additional variance is explained by MAH summaries?**
   - e.g. `Mpeak(z=1)`, `Mpeak(z=2)`, formation redshifts, recent accretion fraction, DiffMAH parameters.

4. **Is fitting DiffMAH necessary?**
   - compare raw MAH summary statistics versus fitted DiffMAH parameters.

5. **Which profile regions are most connected to which MAH epochs?**
   - inner radii versus early halo mass;
   - intermediate/outer radii versus `Mpeak(z~1)` or late mass growth;
   - 50--100 kpc aperture mass versus final mass and late assembly.

6. **Does main-branch MAH contain useful information beyond halo mass and concentration-like proxies?**
   - compare to null/shuffled models.

7. **Can the same framework be evolved across redshift?**
   - if profiles at other redshifts become available, test whether the model can be stopped at each redshift and still predict representative CoGs.

---

## References and relevant works

- Hearin, A. P., Chaves-Montero, J., Becker, M. R., & Alarcon, A. (2021), *A Differentiable Model of the Assembly of Individual and Populations of Dark Matter Halos*, arXiv:2105.05859. https://arxiv.org/abs/2105.05859
- Alarcon, A., Hearin, A. P., Becker, M. R., & Chaves-Montero, J. (2022), *Diffstar: A Fully Parametric Physical Model for Galaxy Assembly History*, arXiv:2205.04273. https://arxiv.org/abs/2205.04273
- Huang, S. et al. (2018), *A Detection of the Environmental Dependence of the Sizes and Stellar Haloes of Massive Central Galaxies*, arXiv:1803.02824. https://arxiv.org/abs/1803.02824
- Huang, S. et al. (2020), *Weak Lensing Reveals a Tight Connection Between Dark Matter Halo Mass and the Distribution of Stellar Mass in Massive Galaxies*, MNRAS. https://academic.oup.com/mnras/article/492/3/3685/5658706
- Huang, S. et al. (2022), *The Outer Stellar Mass of Massive Galaxies: A Simple Tracer of Halo Mass with Scatter Comparable to Richness and Reduced Projection Effects*, MNRAS. https://academic.oup.com/mnras/article/515/4/4722/6640421
- Pillepich, A. et al. (2014), *Halo Mass and Assembly History Exposed in the Faint Outskirts: the Stellar and Dark Matter Haloes of Illustris Galaxies*, arXiv:1406.1174. https://arxiv.org/abs/1406.1174
- Rodriguez-Gomez, V. et al. (2016), *The stellar mass assembly of galaxies in the Illustris simulation: growth by mergers and the spatial distribution of accreted stars*, MNRAS. https://academic.oup.com/mnras/article/458/3/2371/2589235
- Lackner, C. N. & Ostriker, J. P. (2010), *Dissipational versus Dissipationless Galaxy Formation and the Dark Matter Content of Galaxies*, arXiv:1002.0585. https://arxiv.org/abs/1002.0585
- El-Badry, K. et al. (2019), *The formation and hierarchical assembly of globular cluster populations*, MNRAS. https://arxiv.org/abs/1805.03652
- Behroozi, P., Wechsler, R. H., & Conroy, C. (2019), *UniverseMachine: The correlation between galaxy growth and dark matter halo assembly from z = 0--10*, MNRAS. https://academic.oup.com/mnras/article/488/3/3143/5484868
- Baes, M. (2022), *Analytical properties of Einasto models*, arXiv:2209.03639. https://arxiv.org/abs/2209.03639
