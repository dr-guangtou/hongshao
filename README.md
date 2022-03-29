# HongShao: Recipe to Make a Massive Galaxy

----

- The ultimate goal here is to build a model that can connect the assembly of massive dark matter halos to the growth of massive galaxies in them. 
    - i.e., Given a halo mass assembly history (MAH), can we (statistically) predict the 1-D or 2-D distribution of stellar masses of their central galaxies (above certain mass range) at different redshift range.
    - We already have a (relatively) mature physical scenario: the two-phase formation scenario (e.g., Oser et al. 2010; 2012) in place. And there are many existing hydro-dynamic simulations or semi-analytic/empirical models that can help us. 

- In practice, it will be based on:
    1. Deep photometry and accurate lensing capability of HSC survey.
    2. Large footprint and high-quality photometry of the DECaLS survey, and its synergy with DESI spectroscopic survey. 
    - There are other datasets, such as the DES, KiDS, and CFIS imaging data to consider as well.

- These data will provide us:
    - Stellar mass density profiles of massive galaxies at $0 < z < 0.5$ and different definitions of aperture stellar masses. 
    - Geometric information such as the 1-D profile of ellipticity and position angles. 
    - Stellar mass functions using different definitions of aperture masses. 
    - Galaxy-galaxy lensing measurements. 
    - Two-point correlation measurements. 
    - Richness estimate around the sample or richness-based clusters samples.
- These observations become the foundation of the modeling. 

## Reference 

- This project has a close connection to the [`ASAP`](https://github.com/dr-guangtou/asap) repo that models the dark matter halo properties across the M100kpc-M10kpc plane.

- It is also inspired by the Top_N_ tests presented using the [`jianbing`](https://github.com/dr-guangtou/jianbing) repo.
