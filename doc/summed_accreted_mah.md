# The summed-accreted-mass MAH — definition & prior work

*Literature review (web research + adversarial verification), 2026-06-27, for
exp28. Companion to `doc/mah_definitions_research.md` (which covers the
declining-MAH problem). All masses h-free Msun by repo convention.*

## TL;DR
- The MAH we are building — **`M_sum(z)` = sum the bound mass of *all* progenitor
  subhalos at each epoch** — is **not a standard named quantity**. Its theoretical
  root is the EPS *conditional progenitor mass function* (Bond+1991; Lacey & Cole
  1993) and its operational cousin is the *unevolved subhalo mass function* (USMF;
  Giocoli+2008; Jiang & van den Bosch 2014, 2016), but no surveyed paper publishes
  `M_sum(z)` as an explicit time series to compare against the main branch. **This
  is a genuine gap — opportunity (novel) and caution (no established baseline).**
- `M_sum(z)` is the **merger-accreted ("clumpy") component only**. It excludes
  *smooth* accretion — mass in particles never part of a resolved halo — which is
  **~40% of total halo growth** (Genel+2010, particle-level; ~60% merger-accreted).
  So `M_sum(z)` is **not** a drop-in replacement for `M_200c(z)`; it is the
  resolved-collapsed fraction of the final mass, and it is **resolution-dependent**
  (what is "smooth" at TNG300-1 resolution would be "clumpy" at higher resolution).
- Our implementation (sum exclusive `SubhaloMass` over all tree nodes per snapshot)
  is exactly the review's **recommended** SubLink approach: SUBFIND masses partition
  the bound particles, so the sum never double-counts, and it is immune to
  central–satellite switching (both the ex-central and the satellite are summed).

## Definitions (with canonical references)
| flavour | definition | canonical refs |
|---|---|---|
| **main-branch MMP** | follow the *instantaneous* most-massive progenitor; `M(z)=M0·exp(−α_c z)` | van den Bosch 2002 (astro-ph/0105158); Wechsler+2002 (astro-ph/0108151); Correa+2015 |
| **main-branch MMH** | follow the *most-massive-history* (branch-mass) progenitor — **what SubLink/our trees use** | Springel+2005 (origin, astro-ph/0504097); De Lucia & Blaizot 2007; Rodriguez-Gomez+2015 (SubLink, 1502.01339) |
| **Mpeak** | running max of the main-branch mass (handles satellite stripping) | Behroozi+2019 UniverseMachine (1806.07893) |
| **DiffMAH** | smooth monotonic parametric fit to the main branch (≈ Mpeak) | Hearin+2021 (2105.05859) |
| **summed progenitor `M_sum(z)`** | Σ mass of **all** progenitors at each epoch (our exp28) | EPS foundation: Bond+1991 (DOI 10.1086/170520), Lacey & Cole 1993 (DOI 10.1093/mnras/262.3.627); **no explicit time-series in the literature** |
| **infall/peak-mass sum (USMF)** | Σ each accreted subhalo's mass at infall / pre-infall peak (merger-accreted only) | Giocoli+2008 (0712.1563); Jiang & van den Bosch 2014/2016 (1403.6827, 1509.02175) |

USMF universal form: `n(m_acc/M0) ∝ (m_acc/M0)^−0.8`, normalisation ≈0.21.

## Which prior work used a summed/total-progenitor MAH?
**None as a named MAH time series.** Across van den Bosch 2002, Wechsler+2002,
Correa+2015, De Lucia & Blaizot 2007, UniverseMachine, and DiffMAH → all use the
**main branch** (MMP or MMH). The Sussing-Merger-Trees project (Srisawat+2013,
Avila+2014, Wang+2016) documents the main-branch *ambiguity* but proposes no summed
alternative. The summed concept appears only (a) probabilistically in EPS, and (b)
as the *infall-mass* USMF, which is the merger-accreted component — not the full
`M_sum(z)`. **Corrected a wrong attribution along the way:** the MMH "most-massive
history" criterion originates with Springel+2005, not De Lucia & Blaizot 2007.

## Implications for HongShao
1. **Frame `M_sum(z)` correctly:** it is the *summed resolved-progenitor mass*
   (merger-accreted/collapsed fraction), normalised to the central's z=0.4
   `SubhaloMass` (our tree is rooted at snap 72, so `M_sum(72)` = central mass, not
   the FoF/`M200c`). It de-biases switching/dropouts but **redefines the mass** — it
   is not `M200c`. State this whenever it is used as a model feature.
2. **Novelty + caution:** since no published baseline exists, validate against the
   main branch (exp28) and check resolution sensitivity (a mass-threshold variant).
3. **If we want a monotonic de-biased MAH**, the **infall-peak sum** (USMF-style,
   each progenitor's pre-infall peak) is the literature-grounded choice and is
   monotonic by construction — preferable to the raw bound-mass `M_sum(z)` (which
   dips ≤0.12 dex from pre-merger stripping, exp28).
4. **Never call a DiffMAH-form fit to `M_sum(z)` "the DiffMAH fit"** (cf.
   `doc/mah_definitions_research.md`).

## Key sources
Bond+1991 (10.1086/170520) · Lacey & Cole 1993 (10.1093/mnras/262.3.627) ·
van den Bosch 2002 (astro-ph/0105158) · Wechsler+2002 (astro-ph/0108151) ·
Springel+2005 (astro-ph/0504097) · De Lucia & Blaizot 2007 (astro-ph/0601466) ·
Giocoli+2008 (0712.1563) · Genel+2010 (1005.4058) · Fakhouri & Ma 2010 (1001.2304) ·
Srisawat+2013 (1307.3577) · Jiang & van den Bosch 2014 (1403.6827) ·
Rodriguez-Gomez+2015 (1502.01339, SubLink) · Jiang & van den Bosch 2016 (1509.02175) ·
Behroozi+2019 (1806.07893) · Hearin+2021 (2105.05859).
