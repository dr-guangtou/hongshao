# TNG300-1 Snapshot-72 (z=0.4) Data: Format, Loading, and Caveats

This documents the raw data drop used for the first Ultimate-SHMR experiments
and how `hongshao/tng_data.py` turns it into an analysis-ready table. It is the
English companion to the original `save_tng300_072_file_structure.md` (Chinese)
that ships with the data, extended with the file-format quirks found while
probing the actual files.

**Sample:** central galaxies of ~3,388 massive dark-matter halos
(`Mhalo > 10^13 Msun`) from IllustrisTNG **TNG300-1**, observed at **snapshot 72
(z = 0.4)**. Each galaxy has a main-branch halo assembly history (MAH) and a set
of 1-D stellar mass profiles.

**Raw drop location (external, not in the repo):**
`/Users/mac/Desktop/tng300_mah_mprof`

---

## 1. Files

| File / dir | What it is |
|---|---|
| `galaxies_tng300_072_mah_hmc.txt` | **Pickle** (not text) — list of 3,388 main-branch MAHs. |
| `galaxies_tng300_072_hmc_13_aperture_mass.txt` | **Pickle** (QTable, not text) — 3,388 galaxies × 3 projections: exact z=0.4 halo mass, secondary halo properties, per-projection aperture masses + galaxy shapes. See §3b. |
| `save_tng300_072_hist_aper_dir/` | 3,388 `.npy` — aperture masses + curves of growth (CoG). |
| `save_tng300_072_hist_prof/` | 3,388 `.npy` — isophote profile tables + quality flags. |
| `diffmah/diffmah_tng.h5` | Public DiffMAH fits for the full TNG halo set (288,405 halos). |
| `diffmah/tng_diffstar_fits_default.h5` | Public Diffstar SFH fits (288,405 halos). |
| `save_tng300_072_file_structure.md` | Original (Chinese) format note. |
| `map_tng100_hist_stellar.hdf5` | **Corrupted — 852 MB of all-zero bytes. Unusable.** |

Files in the two `save_tng300_072_*` directories share the same `index`
(`0…3387`); the same index is the **same central galaxy**. The MAH pickle is a
flat list indexed identically.

---

## 2. MAH pickle — `galaxies_tng300_072_mah_hmc.txt`

```python
import pickle
with open(".../galaxies_tng300_072_mah_hmc.txt", "rb") as f:
    mah = pickle.load(f)          # list, len 3388
entry = mah[0]                    # numpy array, shape (2, N)
snaps = entry[0]                  # snapshot numbers, descending (71, 70, …)
mass  = entry[1] * 1e10 / 0.6774  # -> Msun  (raw units are 1e10 Msun/h)
```

Properties found by probing all 3,388 entries:

- **Already a peak-mass history.** Mass is monotonically non-decreasing with
  time, so the running maximum equals the stored value. `peak_history()` sorts
  ascending in snapshot and enforces the running max only as a numerical guard.
- **Snapshot gaps exist.** Some main branches skip snapshots (e.g. `…,69,58,…`),
  so always interpolate against the actual `snaps` array, never assume a dense
  grid.
- **Variable length / truncation.** The latest recorded snapshot is **71 for
  3,249 galaxies** but `< 71` for **132 galaxies** (down to snap 30 in extreme
  cases). Histories also start at different early snapshots.
- **7 galaxies have an empty MAH** and are dropped.

### M0 caveat (important)
The observation epoch is snapshot **72** (z=0.4), but the pickle reaches at most
snapshot **71**. The original recipe prepended the snap-72 mass from a separate
table (`tab2['mass_halo']`) **which is not in this drop**. We therefore define

> `logm0_halo` = peak mass at the **latest available snapshot** (~z=0.42).

For the 3,249 galaxies reaching snap 71 this is an excellent approximation. For
the 132 truncated ones (especially ~92 that fall below `10^13 Msun`) it
underestimates the true z=0.4 mass — these are excluded by the `use` cut.

> **Resolved (2026-06-22).** The aperture-mass table (§3b) supplies the exact
> snapshot-72 halo mass `mass_halo`. The table now carries `logmh_z0p4 =
> log10(mass_halo)`; over the `use` sample it differs from the `logm0_halo`
> proxy by only **+0.004 dex (median), 0.020 dex scatter** — confirming the
> proxy was excellent, and providing the exact value going forward. Prefer
> `logmh_z0p4` as M0.

---

## 3. Aperture / CoG `.npy` — `save_tng300_072_hist_aper_dir/`

```python
aper = np.load(".../galaxies_tng300_072_0_hist_aper.npy", allow_pickle=True).item()
a = aper["z0p4"]["aper"]   # shape (7,)  integrated stellar mass, Msun
c = aper["z0p4"]["cog"]    # shape (24,) curve of growth, Msun
```

Five redshift keys: `z0p4, z0p7, z1, z1p5, z2`. Each holds:

- `aper` (7,): stellar mass within fixed semi-major-axis apertures
  `SMA_KPC = [10, 30, 50, 75, 100, 120, 150]` kpc.
- `cog` (24,): cumulative stellar mass on a log radial grid
  `COG_RAD_KPC = np.arange(2**0.25, 150**0.25, 0.1)**4` (2.0 → 148.2 kpc).

**Quirk:** a small number of CoG/aper arrays are `object` dtype containing
`None` (failed bins) — roughly 1 in 484 sampled. `tng_data.py` coerces these to
`NaN`; the `use` cut drops galaxies with any non-finite CoG bin.

These aperture masses (especially the 50–100 kpc range) are the core
galaxy-side observable for the first experiment.

---

## 3b. Per-projection aperture-mass table — `galaxies_tng300_072_hmc_13_aperture_mass.txt`

A pickled astropy **QTable**, **10,164 rows = 3,388 galaxies × 3 sky
projections** (`proj` ∈ {`xy`, `xz`, `yz`}). Keyed by `gal_num` (**== our
`index`**, verified: its `xy` aperture masses reproduce the §3 npy masses
exactly). Loaded and merged by `tng_data.load_aperture_extras()`.

```python
import pickle
with open(".../galaxies_tng300_072_hmc_13_aperture_mass.txt", "rb") as f:
    t = pickle.load(f)          # QTable, 10164 rows
```

Columns (masses are **linear Msun**):

| column | meaning |
|---|---|
| `gal_num` | drop index 0…3387 (== our `index`) |
| `proj` | sky projection: `xy` / `xz` / `yz` |
| `catgrp_id` | TNG FoF **GroupID** (constant over a galaxy's 3 rows) — see §5 |
| `mass_halo` | **exact z=0.4 halo mass** (Msun) — resolves the §2 M0 gap |
| `aper_{10,30,50,75,100,150}_gal` | aperture stellar mass (Msun), 6 radii — **per projection** |
| `c_200c` | halo concentration (3D, projection-independent) |
| `c_to_a_3d`, `b_to_a_3d` | 3D halo axis ratios |
| `v_sigma_3d` | 3D velocity dispersion |
| `acc_rate` | halo accretion rate |
| `ellipticity`, `PA` | projected galaxy shape (per projection) |
| `r_20_gal`, `r_50_gal`, `r_80_gal` | projected 20/50/80% light radii (per projection) |

**What it adds.** (1) the exact M0; (2) **secondary halo properties**
(`c_200c`, 3D shape, `acc_rate`) to *test* whether they reduce residual scatter
beyond the MAH; (3) **3 projections**, which directly measure the projection
contribution to scatter — found to be small (≈0.007 dex at the 100 kpc
cumulative aperture, ≈0.037 dex for the 50–100 kpc annulus, vs the ≈0.17 dex
total outskirt residual), i.e. the outskirt scatter is **mostly genuine
halo-to-halo variation, not projection** (caveat: 3 orthogonal axes undersample
orientation). It does **not** carry the DiffMAH parameters (§5).

In the assembled table these become: `mass_halo`/`logmh_z0p4`, `catgrp_id`, the
five 3D halo properties, `logmstar_aper_proj` (n, 3, 6) on
`projections=(xy,xz,yz)` × `aper6_sma_kpc=[10,30,50,75,100,150]`, and the
per-projection `ellipticity_proj`/`pa_proj`/`r20_proj`/`r50_proj`/`r80_proj`.

> **Which masses to model.** The 2-D `logmstar_aper_proj` are physically more
> accurate (no perfect-ellipse assumption — relevant during mergers), but the
> **CoG-derived `logmstar_cog`/`logmstar_aper` remain the modeling target**:
> they are what an observer measures, the difference is small, and the end goal
> is to reproduce the CoG. Use `*_aper_proj` for cross-checks only (see AGENTS.md).

---

## 4. Profile `.npy` — `save_tng300_072_hist_prof/`

```python
prof = np.load(".../galaxies_tng300_072_0_hist.npy", allow_pickle=True).item()
z04 = prof["map_hist_z0p4"]
z04["prof"]   # astropy QTable: r_kpc, intensity, intens_err, ellipticity, pa, …
z04["flag"]   # bool — profile measurement is valid
z04["test"]   # bool — shape measurement hit a default lower bound
z04["other"]  # QTable of isophote-fit diagnostics (x0/y0, ndata, stop_code, …)
```

Same five redshift keys (prefixed `map_hist_`). `prof` is the 2-D isophote fit
(intensity is a stellar **surface mass density**, not the integrated CoG). For
experiment 1 we mainly use `flag`/`test`; the integrated CoG from §3 is the
preferred profile observable (monotonic, less noisy than differential
intensity).

---

## 5. DiffMAH / Diffstar HDF5 (`diffmah/`)

Public products from the DiffMAH (arXiv:2105.05859) and Diffstar
(arXiv:2205.04273) projects, downloaded from the HACC NERSC portal.

- `diffmah_tng.h5` (288,405 halos): per-halo DiffMAH parameters
  (`early_index`, `late_index`, `mah_logtc`, `mah_k`, `logmp_fit`), the
  simulated and fitted log-MAH on a 100-point grid (`log_mah_sim`,
  `log_mah_fit`), and bookkeeping (`halo_id`, `upid`, `t0`, …).
- `tng_diffstar_fits_default.h5` (288,405): Diffstar SFH parameters in
  unbounded form (`u_lgmcrit`, `u_lgy_at_mcrit`, `u_qt`, …), keyed by `halo_id`.

**Cross-match gap (id-system mismatch).** Both files are keyed by `halo_id` =
the global TNG **SubhaloID** (0…288404, sorted = row index; `halo_id=0` is the
most massive group's *central*, `halo_id=1` its first *satellite*, …). The
aperture table's `catgrp_id` is the TNG FoF **GroupID** (mass-ranked, 0…4831).
These are **different numbering systems**, so equating the integers
(`catgrp_id == halo_id`) cross-matches the *wrong* objects (it lands on
satellites — e.g. our `catgrp_id=1` central is at `halo_id=29051`, not
`halo_id=1`). A by-mass match of our galaxies to the catalog's centrals
(`upid==-1`) is only approximate (r≈0.96; ranks shift because our 3,388 are a
subset of ~3,727 massive centrals). **An exact match needs the TNG
`GroupFirstSub` array (GroupID → central SubhaloID), the SubhaloID of each
galaxy, or 3D positions** (the catalog has `x,y,z`). Until then we use our own
DiffMAH fits (`dmah_*`, exp10) and keep `catgrp_id` in the table for the
eventual match. (`log_mah_sim` in the catalog sits ~0.2 dex below the pickle
peak history for the same halo — a mass-*definition* difference, same shape.)

---

## 6. Snapshot ↔ redshift / cosmic time

The pickle snapshot integers are **0-based TNG-native numbering** (snapshot `N`
is TNG snapshot `N`); no index shift is needed. Verified facts:

- Pickle snapshots span **1…71**; snapshot 0 and 72 never appear. The
  observation epoch (snap 72 = z=0.4) is *not* in the MAH — the pickle holds
  only progenitor snapshots.
- The 5 profile redshifts map to integer TNG snapshots (official spec):

  | z | 0.4 | 0.7 | 1.0 | 1.5 | 2.0 |
  |---|---|---|---|---|---|
  | snapshot | 72 | 59 | 50 | 40 | 33 |

**Cosmic time** comes from the vendored file `data/external/tng_cosmic_time.txt`
(100 values, Gyr, from the diffmah NERSC portal). Its human label "Snapshot 1"
= TNG snapshot 0, so loaded as a numpy array `t`, **`t[snap]` gives the cosmic
time of 0-based snapshot `snap` directly** — no offset. This was confirmed by
matching `t` at snaps 72/59/50/40/33/99/0 to `astropy` ages for the TNG (Planck
2015) cosmology to <1 Myr.

Anchor halo masses `Mpeak(z)` are interpolated at the integer anchor snapshots.
Formation times/redshifts (z50/z75/z90) are computed from `t[snap]` plus a
`t → z` inversion of the TNG cosmology (`TNG_COSMO` in `tng_data.py`).

---

## 7. The assembled table

`hongshao/tng_data.py` builds one row per galaxy. Build it with:

```bash
uv run python -m hongshao.tng_data --selftest        # quick checks (60 galaxies)
uv run python -m hongshao.tng_data --full \
    --out data/processed/tng300_072_z0p4.fits         # all 3388 -> FITS
```

Columns (stellar & halo masses are `log10(Msun)`):

| column | meaning |
|---|---|
| `index` | drop index 0…3387 |
| `logm0_halo` | peak halo mass at latest snapshot (~z=0.4); see §2 caveat |
| `logmpeak_z0p4 … z2` | `Mpeak` interpolated at snaps 72/59/50/40/33 |
| `f_early`, `f_late` | `Mpeak(z=2)/M0`, `1 − Mpeak(z=1)/M0` |
| `dlogm_z2_z1`, `dlogm_z1_z0p4` | inter-epoch halo growth |
| `t50/t75/t90` | cosmic time (Gyr) peak mass first reached 50/75/90% of M0 |
| `z50/z75/z90` | formation redshifts (from `t*` via TNG cosmology) |
| `logmstar_aper` (7) | stellar mass in `SMA_KPC` apertures, z=0.4 (= `xy` projection) |
| `logmstar_cog` (24) | curve of growth at `COG_RAD_KPC`, z=0.4 |
| `mass_halo`, `logmh_z0p4` | **exact** z=0.4 halo mass (Msun; and its log10) — §3b. Prefer `logmh_z0p4` as M0 |
| `catgrp_id` | TNG FoF GroupID (int, -1 if missing); for the future DiffMAH match — §5 |
| `c_200c`, `c_to_a_3d`, `b_to_a_3d`, `v_sigma_3d`, `acc_rate` | secondary 3D halo properties (projection-independent) — §3b |
| `logmstar_aper_proj` (3, 6) | log10 aperture M* on `projections × aper6_sma_kpc` — §3b |
| `ellipticity_proj`, `pa_proj`, `r20_proj`, `r50_proj`, `r80_proj` (3) | projected galaxy shape/size per projection — §3b |
| `rdm_*` | radial-DiffMAH profile fit (`logMstar0`, `beta_in`, `beta_out`, `R_c`, `Delta`, `rms`); see `hongshao/profiles.py`. Fit over `R ≥ COG_FIT_RMIN_KPC` (5 kpc; inner core marginally resolved — exp07), so `rdm_logMstar0` is the normalization at the innermost fitted radius and reconstructions must use the same inner cut. Skip with `build_dataset(fit_profiles=False)` |
| `flag`, `test` | profile quality flags |
| `valid_mah`, `latest_snap`, `n_mah_pts` | MAH availability |
| `mah_decline`, `mah_declined` | `1 − M(z=0.4)/M_peak`, and whether it exceeds 5% |
| `use` | clean cut: good profile, not declined, `latest_snap ≥ 70`, `M0 ≥ 1e13`, finite CoG |

**Decline cut.** ~99% of halos show small (few-%) snapshot-to-snapshot mass
fluctuations, which are normal. We exclude only halos that have *turned over* —
whose z=0.4 mass is more than `MAH_DECLINE_TOL = 5%` below their historical peak
(stripping / backsplash / tracking loss). This removes ~700 halos (20.6%); the
threshold is a tunable module constant.

`COG_RAD_KPC`, `SMA_KPC`, units and the M0 caveat are stored in `tbl.meta`.

---

## 8. Known issues / TODO before / during experiment 1

1. ~~**Missing snap-72 halo mass (`mass_halo`)**~~ — **resolved (§3b).** The
   aperture table supplies the exact z=0.4 mass (`logmh_z0p4`); it matches the
   snap-71 peak proxy to +0.004 dex (median) over the `use` sample.
2. **DiffMAH/Diffstar cross-match is an id-system mismatch, not a missing id**
   (§5): `catgrp_id` (GroupID) ≠ `halo_id` (SubhaloID). Need `GroupFirstSub`,
   per-galaxy SubhaloID, or 3D positions for an exact link. We use our own
   `dmah_*` fits meanwhile.
3. **`map_tng100_hist_stellar.hdf5` is corrupted** (all zeros) — re-fetch if it
   is actually needed; not required for experiment 1.
4. ~~No verified full snapshot→redshift table~~ — **resolved.** Cosmic times are
   vendored in `data/external/tng_cosmic_time.txt` and verified; formation-time
   summaries are now in the table.
