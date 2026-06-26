# Research note — halo MAH definitions & the "declining MAH" problem

*Investigation (deep-research harness + manual vetting), 2026-06-27. Triggered by
the ~20% of TNG300 massive centrals whose tracked main-branch halo mass sits below
its peak by z=0.4 (exp26). The automated adversarial-verification step was
interrupted by a spend limit, so the 25 gathered claims below were vetted by hand
against the cited primary sources and established knowledge; a few specific numbers
(flagged) still warrant independent confirmation.*

## TL;DR for our pipeline
1. **We already apply the literature's #1 fix.** `peak_history()` builds the MAH
   with `np.maximum.accumulate` → the kernel input is the **running-max `Mpeak`**,
   which *cannot* decline. So the "unphysical decline" never enters exp25/26; it is
   a *quality flag* (`mah_decline` is computed from the *raw* history).
2. The residual issue is subtler and **DiffMAH does NOT fix it.** The running-max
   Mpeak (and the DiffMAH fit to it) reflects the *transient merger fluctuation*,
   not a clean accretion history: central–satellite switching briefly inflates the
   central mass (a spike), the running-max grabs the spike, and the later "decline"
   is the settling. Measured: declining-MAH galaxies have a *higher* DiffMAH
   late-growth index (`dmah_late` median 0.93 vs 0.34) — steeper late growth driven
   by the artifact, not a frozen plateau. So the kernel's late-time deposition for
   these galaxies is **biased in an uncontrolled direction**; smoothing cannot
   recover mass the tracer corrupted.
3. **DiffMAH only makes them *runnable* (monotonic input), it does not de-bias
   them.** The genuine fix needs a better mass tracer — a **summed-accreted-mass**
   history from the public SubLink merger trees (sum each progenitor's peak mass
   before infall; immune to switching), or `SubLink_gal` (baryonic tree), or a
   phase-space / history-based finder. The cheap alternative is to **drop the
   merger-active tail with the scope caveat** (per the property analysis).
4. Switching halo finder (Rockstar/HBT) or boundary (splashback) would be *more
   physical* but needs reprocessing particle data and is overkill for a
   phenomenological forward model.

---

## Q1 — Why it happens & the TNG-recommended handling

**Cause (well established).** TNG runs FoF+SUBFIND **independently per snapshot**;
SUBFIND uses a *configuration-space, exclusive (self-bound)* mass. During a merger
the "central" identity **switches between substructures** ("central–satellite
switching") — one illustrative system undergoes **seven** switches in a single
merger — producing large *transient* fluctuations in the instantaneous main-branch
mass that are **not physical mass loss** [MNRAS 472,3659]. Across finders, **~15–30%
of main-branch halos show a declining mass at any given step even without tidal
stripping** [Avila+2014] — consistent with our ~20%.

**Recommended fixes (in rough order of effort):**
- **`Mpeak` / cumulative-maximum mass** — the standard mitigation; *what we already
  use*. Monotonic by construction.
- **Smooth parametric MAH — DiffMAH** [Hearin+2021, arXiv:2105.05859]: a
  5-parameter (`logm0, logtc, early_index, late_index, t_peak`) differentiable
  model, **monotonic non-decreasing by construction**, fit to the *full* `log_mah(t)`
  so it sidesteps per-snapshot fluctuations; validated on gravity-only **and TNG**
  (ships a `load_tng_data` loader). *We already have this as `dmah_*`.*
- **`SubLink_gal`** — the baryonic-particle-tracking SubLink variant, built to
  follow galaxies whose DM subhalo is lost during/after a merger; **not in the
  public release but available on request** [Nelson+2019; TNG docs].
- TNG ships **supplementary "Subhalo Mass Assembly"-type catalogs** (max-past-mass
  etc.) with specific papers, though not as one systematic project-wide MAH product
  [TNG docs].
- Note SubLink's main branch = "the progenitor with the most massive *history*
  behind it" (not the most massive at the previous snapshot), which affects which
  branch is followed through a merger [Nelson+2019].

## Q2 — Does a different halo finder give cleaner MAHs?

- **Halo-finder choice dominates over tree-builder choice** for MAH quality —
  swapping the *tree* code while keeping SUBFIND will not cure it [Avila+2014].
- **Phase-space (Rockstar) and modern history-based (HBT+/HBT-HERONS) finders give
  more monotonic MAHs than SUBFIND.** Rockstar/AHF "cluster at high growth fraction
  with low scatter"; SUBFIND shows broader merger-time fluctuations [Avila+2014].
  A 2025 history-space finder, **HBT-HERONS**, reportedly cuts mass-swapping /
  sudden-formation artifacts from **~50–80% (SUBFIND, massive objects) to ~5–10%**
  [arXiv:2501.07677 — *specific numbers worth independent confirmation*].
- **Caveats:** (i) phase-space/SO finders may *truncate* (lose) subhalos near host
  centres earlier than SUBFIND even while giving smoother mass when tracked
  [Avila+2014] — truncation ≠ fluctuation; (ii) the "HBThalo" tested in 2014 is an
  *older* code than HBT-HERONS (2025), so don't conflate them; (iii) **a Rockstar /
  VELOCIraptor catalog for TNG was *planned* but the public release is SUBFIND +
  SubLink/LHaloTree** [TNG docs] — using Rockstar means reprocessing snapshots.

## Q3 — Better halo-boundary / mass definitions

- **Spherical-overdensity masses (M200c, M200m, Mvir) "pseudo-evolve"**: the
  reference density drops with cosmic time, so the measured radius/mass **grows even
  for a static profile** — up to **~×2 since z=1** [Diemer, More & Kravtsov 2013].
  A large part of apparent late SO "growth" is non-physical. (M200m < affected than
  M200c; Mvir intermediate.)
- **Splashback radius/mass** [Diemer & Kravtsov 2014; More, Diemer & Kravtsov 2015;
  SPARTA/MORIA, Diemer 2017/2020] and the **orbiting-vs-infalling / "dynamical" mass
  decomposition** [Diemer 2024; Garcia et al. 2021/2023] are physically motivated and
  more monotonic, but heavier to compute and absent from standard TNG catalogs.

## Sources
- TNG data docs — https://www.tng-project.org/data/docs/background/
- Nelson et al. 2019 (TNG public data release) — arXiv:1812.05609
- Avila et al. 2014 ("Sussing Merger Trees: the influence of the halo finder") — arXiv:1402.2381
- central–satellite switching — MNRAS 472, 3659 (academic.oup.com/mnras/article/472/3/3659)
- Hearin et al. 2021 (DiffMAH) — arXiv:2105.05859
- HBT-HERONS (2025) — arXiv:2501.07677
- Diemer, More & Kravtsov 2013 (pseudo-evolution) — benediktdiemer.com/research/pseudoevolution/

*(deep-research run wf_da0f8c11-483: 112 agents, ~731k tokens, 25 claims / 7 sources;
verification incomplete due to spend limit — manually vetted.)*
