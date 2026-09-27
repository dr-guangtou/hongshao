# Literature positioning and novelty boundaries

September 27 user clarification: Xu and Leidig are collaboration precursors;
Xu supplied this data with Xu's satellite-particle treatment. Position Paper 1
as a step toward a forward galaxy–halo connection for massive centrals using
projected 1-D stellar profiles over an individually measurable radial range
and transparent low-complexity methods. The distinction is not a blanket
claim that all earlier samples are broad or all machine-learning methods
are unusable observationally. Earlier requests below for user confirmation
of the data lineage have now been answered.

Primary-source check, September 26, 2026. This is a targeted positioning
review, not an exhaustive systematic literature search. No priority claim
such as “first” is justified by it. Entries distinguish published results
from our proposed contribution. Numerical performance comparisons across
these different samples, mass definitions, and targets would be misleading.

## Closest precedents

### Stellar outskirts already trace assembly

[Pillepich et al. (2014), *Halo Mass and Assembly History Exposed in the
Faint Outskirts*](https://arxiv.org/abs/1406.1174), studies 3-D stellar density
slopes in Illustris and their dependence on halo history at fixed halo mass.
Recently assembled halos have shallower stellar halos than older halos of
similar mass. This is direct precedent for the broad motivation. Exp04's
attribution of the opposite halo-age trend should not be carried into the
paper; stellar population age and halo formation time are distinct.

**Our proposed addition:** projected, observation-motivated mass profiles;
explicit separation of mass, size, and additional radial information;
out-of-sample tests against matched statistical references.

### Much of the average profile is already determined by mass

[Pillepich et al. (2018), *The stellar mass content of groups and clusters
of galaxies*](https://arxiv.org/abs/1707.03406), demonstrates strong relations
between halo mass, stellar mass, size, and average stellar profiles in TNG.
It also makes clear that stellar-component boundaries and apertures matter.

**Our proposed addition:** quantify the residual dependence on assembly
after a sufficiently flexible mass-dependent profile reference. Showing
that a mean profile exists, or that profiles are compressible, is not enough.

### Stellar mass itself is already an assembly proxy

[Lim et al. (2016), *An observational proxy of halo assembly time and its
correlation with galaxy properties*](https://academic.oup.com/mnras/article/455/1/499/984431),
uses the central stellar-to-halo mass ratio as an assembly proxy. This
motivates the crucial stellar-mass control: a profile analysis should show
what is gained beyond that scalar ratio, not merely rediscover it.

**Our proposed addition:** the incremental information from radial structure
at the same halo and stellar mass. Bibliographic metadata should be exported
from the publisher when the manuscript bibliography is assembled.

### Observational motivation for going beyond one stellar mass

[Huang et al. (2020), *Weak lensing reveals a tight connection between dark
matter halo mass and the distribution of stellar mass in massive
galaxies*](https://academic.oup.com/mnras/article/492/3/3685/5658706), shows that
inner and extended stellar masses together describe halo mass better than
a single stellar mass. It motivates a richer galaxy–halo connection, but
does not by itself establish recovery of MAHs.

[Huang et al. (2022), *The outer stellar mass of massive galaxies: a simple
tracer of halo mass with scatter comparable to richness and reduced
projection effects*](https://academic.oup.com/mnras/article/515/4/4722/6640421),
establishes the observational value of outer stellar mass as a halo-mass
proxy. Our paper should address assembly information **at fixed halo mass**,
not reframe an improved mass proxy as a new assembly measurement.

### Particularly close simulation and measurement precedents

[Xu et al., *The Outskirt Stellar Mass of Low-Redshift Massive Galaxies is
an Excellent Halo Mass Proxy in Illustris/IllustrisTNG
Simulations*](https://arxiv.org/html/2412.03406v2), compares projected aperture
and outskirt masses across Illustris, TNG100, and TNG300. It uses isophotal
geometry and projected maps with bound satellite particles removed. It
distinguishes direct aperture photometry from integrating isophote profiles.

**Our proposed addition:** move from a mass-proxy comparison to a
conditional profile–assembly relation. Confirm whether HongShao uses this
exact measurement lineage, why parent sample counts differ, and which
previous catalog/methods deserve explicit attribution. Do not silently
equate its CoG definition with every similarly named repository array.

[Leidig et al., *Reaching for the Edge II: Stellar Halos out to Large Radii
as a Tracer of Dark Matter Halo Mass*](https://arxiv.org/abs/2511.10723), uses
HSC-like mock measurements in TNG, explores outer annuli, and provides
halo-mass-dependent mean profile models. Its much larger radial reach is
also relevant to what a finite HongShao aperture omits.

**Our proposed addition:** history-dependent residual structure and its
incremental value, not another search for the best halo-mass annulus.
The mock-observation machinery might support a later coordinated check;
its availability for this project must be confirmed with the user.

### A direct competitor to an overly broad inverse claim

[Srivastava et al. (2025), *Predicting Halo Formation Time Using Machine
Learning*](https://arxiv.org/html/2504.14426v1), A&A 700, A87, predicts halo
half-mass assembly time in The Three Hundred/Gizmo-Simba simulations from
halo and baryonic properties. Its inputs include BCG/ICL properties and
radial maps; it also tests simpler relations involving BCG–satellite mass
gaps and mass ratios.

**Our proposed addition:** a deliberately restricted central stellar-profile
information test, controlling for mass and size, rather than maximizing
accuracy with a broad baryonic feature inventory. Recovering a formation
time from galaxy properties is not new by itself. A profile-only comparison
should be fair to simple mass/size/aperture-ratio predictors.

### MAH coordinates are a method, not the discovery

[Hearin et al., *A Differentiable Model of the Assembly of Individual and
Populations of Dark Matter Halos*](https://arxiv.org/abs/2105.05859), supplies
a compact model for halo histories. For HongShao, measured-history PCs and
DiffMAH are alternative ways to describe the inputs. Neither makes a
TNG-trained stellar relation automatically universal. The paper should
separate representation accuracy from the scientific information test.

## Recommended novelty statement

> We quantify which parts of the projected stellar mass distribution of
> massive central galaxies carry information about halo assembly beyond
> present halo mass, total stellar mass, and galaxy size, and construct a
> compact probabilistic description of that relation in TNG300.

This is a proposed contribution, conditional on the confirmatory results.
If the size-controlled test is null, replace the first clause with a result
that locates the information principally in amplitude and size. Do not
promise a universal mapping or a recovered detailed MAH.

## Checks before submission

- Ask the user how this paper should relate to the Xu and Leidig projects,
  including data provenance, overlap, and coordination with collaborators.
- Refresh the literature search at manuscript freeze; inspect citing work
  on profile-based formation-time prediction and on BCG/ICL assembly proxies.
- Verify all citation metadata and exact measurement definitions from the
  published versions. This planning note is not a ready-to-submit bibliography.
- Do not interpret spherical-overdensity mass growth as purely physical
  accretion: changing reference density can contribute. If the paper interprets
  timing physically, add a dedicated mass-definition/pseudo-evolution check
  and the appropriate primary literature, rather than assuming the distinction
  is negligible for these halos.
