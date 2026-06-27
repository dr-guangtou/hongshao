# exp27 — TNG-API cross-match: attach the official DiffMAH/DiffStar fits

## Question
Our 3388 z=0.4 (snap-72) massive galaxies are keyed by **SubhaloID**. The local
DiffMAH/DiffStar value-added catalogs (`diffmah_tng.h5`,
`tng_diffstar_fits_default.h5`) are keyed by a **z=0 contiguous row index**
(`halo_id` 0–288404), with no SubhaloID. A direct mass-history match is non-unique
(exp26 handover: RMS 0.11). **How do we attach the official fits to our galaxies?**

## Bridge (exact, validated)
The DiffMAH catalog stores, for every z=0 halo, the **position of its
main-progenitor branch at each TNG snapshot**, with the **array column index ==
snapshot number** (verified empirically: galaxy 0's snap-72 position matches
`diffmah` row 0 column 72 to 5×10⁻⁶ cMpc/h). Column 72 is therefore the z=0.4
main-progenitor position, in cMpc/h, drawn from the *same SUBFIND catalog* as our
galaxies. So:

```
our snap-72 SubhaloPos (ckpc/h → /1000)  ==  diffmah[:, x/y/z, col 72]   (cMpc/h)
```

matches **exactly** (float precision) whenever our snap-72 subhalo is the main
progenitor of some z=0 halo. The two catalogs share the same `halo_id` row order,
so one matched row attaches both at once. Panel (b) of the QA figure confirms the
bridge end-to-end: input `logMh(z=0.4)` ≡ official `log M₂₀₀c` at snap 72 to
**+0.000 dex** (identical SUBFIND definition).

## Method
1. `fetch_mpb.py` — pull `…/snapshots/72/subhalos/{id}/sublink/mpb.hdf5` for all
   3388 (concurrent, resumable, stdlib `urllib`). Gives the snap-72 `SubhaloPos`
   **and** the official main-branch history (`Group_M_Crit200`, `SubhaloMass`,
   `SubhaloMassInRadType`, …) in one request. Cached to `outputs/mpb_cache/`.
2. `crossmatch.py` — build a `cKDTree` from the DiffMAH snap-72 positions, query
   each galaxy, accept matches within **1 ckpc/h** (= 10⁻³ cMpc/h), attach the
   DiffMAH + DiffStar fit parameters, and snapshot-align the official main-branch
   MAH onto the 0–99 grid.
3. `qa.py` — three-panel QA figure (match quality, mass consistency, example MAHs).

## Results
- **3154 / 3388 matched (93.1 %)**, median match distance **1.5×10⁻⁵ cMpc/h**
  (exact). DiffStar `success` is **100 %** among matched.
- **234 (6.9 %) flagged `matched=False`.** Their nearest DiffMAH main branch sits
  **16–122 ckpc/h** away — a clean *second* mode (nothing falls between 10⁻³ and
  10⁻² cMpc/h), so the tolerance is unambiguous. These are snap-72 subhalos that
  are **not** the main progenitor of any z=0 halo: destroyed / absorbed by z=0, so
  no DiffMAH row exists for them (the catalog only contains z=0 main branches).
  This is the non-uniqueness the exp26 handover anticipated; they span the full
  mass range, so dropping them is a small, definitional incompleteness, not a mass
  cut.
- Fetch: 3372 pulls in **30 min** at 1.9 gal/s (10 workers), **zero failures**.

## Outputs (git-ignored)
- `outputs/crossmatch.fits` — 3388 rows × 31 cols: `index`, `subhalo_id_snap72`,
  `matched`, `halo_id` (DiffMAH/DiffStar row, −1 if none), `match_dist_cmpc`,
  `pos72_{x,y,z}`, the DiffMAH fit params (`logmp_fit`, `mah_logtc`, `early_index`,
  `late_index`, `mah_k`, `t_infall`, `upid`, `diffmah_loss`), and the DiffStar
  params (`u_*`, `diffstar_success`, `diffstar_loss`). Unmatched → NaN / −1.
- `outputs/official_mah.npz` — snapshot-aligned (3388×100) main-branch histories:
  `log_m200c`, `log_mstar_inrad`, `log_msubhalo` (official SUBFIND, log₁₀ M⊙),
  plus `diffmah_log_mah_{sim,fit}` for matched rows and `cosmic_time_gyr`.
- `outputs/diffmah_combined.fits` — both DiffMAH flavours per galaxy (`own_*`,
  `official_*`) + `diffmah_source` + recommended official/own-fallback columns.
- `outputs/mpb_cache/{sid}.hdf5` — the raw MPB pulls (≈ 440 MB).
- `figures/crossmatch_qa.{png,pdf}`, `figures/diffmah_own_vs_official.{png,pdf}`.

## Two DiffMAH flavours & the combined catalog
We now hold **two** DiffMAH fits per galaxy, same model (Hearin+2021 rolling power
law, `k = 3.5` fixed) but different anchor `t0` — **do not confuse them**:

| flavour | source | anchor `t0` | `logmp` means | coverage |
|---|---|---|---|---|
| **own** | `hongshao.diffmah`, `data/processed/tng300_072_z0p4.fits` (`dmah_*`) | t(snap 72) = 9.39 Gyr | peak mass **at z=0.4** | 3375/3388 (13 too few MAH pts) |
| **official** | `diffmah_tng.h5` via the cross-match (`logmp_fit`, …) | 13.80 Gyr | peak mass **at z=0** | 3154 matched |

`diffmah_consistency.py` reconciles them and writes **`outputs/diffmah_combined.fits`**
(`own_*`, `official_*`, a `diffmah_source` ∈ {official, own, none}, and recommended
columns = official-preferred / own-fallback). Sources: **official 3154, own 221,
none 13**.

**Unit gotcha (fixed):** the DiffMAH catalog stores log masses in **Msun/h**; our
M200c is in Msun. The raw mismatch was a flat −0.169 dex (= log₁₀ h) — once
corrected, `diffmah log_mah_sim ≡ M200c` to −0.008 dex, i.e. **the official fit is
to the same M200c we use**, just h-ful. `crossmatch.py` now adds +log₁₀(1/h) so all
masses in `crossmatch.fits` / `official_mah.npz` / `diffmah_combined.fits` are Msun.

**Consistency (3154 common):**
- Reconstructed MAH **curves agree to 0.078 dex** (median, over z ≥ 0.4; 69 % < 0.10
  dex). z=0.4 **normalization agrees to −0.032 dex** (median, scatter 0.115).
- **Raw shape params are unbiased but individually degenerate** (`early` scatter
  1.1, `logtc` 0.5, `late` 0.7; our z=0.4-limited fit even rails `logtc` against its
  bounds). Different `(logtc, early, late)` triples reproduce the same curve.

**Practical rule:** for anything needing **M(t) or the normalization** (profile
prediction, halo mass), official and own are interchangeable → use official by
default, own for the 221 fallback (combined catalog already does this). But do
**not** feed mixed-flavour raw `(logtc, early, late)` into one ML model — pick a
single flavour per model, since the two are degenerate re-parametrisations.

## Next
Build the **summed-accreted-mass MAH** from `…/sublink/full.hdf5` (the de-biased
MAH that replaces the running-max Mpeak DiffMAH was fit to). Heavy: the full tree
for the biggest cluster is ~300 MB / 290k rows, so this is a separate, careful
pull — likely the matched subset and a streaming tree-walk rather than caching
every full tree to disk.

## Reproduce
```
uv run python experiments/exp27_tng_api_crossmatch/fetch_mpb.py --workers 10
uv run python experiments/exp27_tng_api_crossmatch/crossmatch.py
PYTHONPATH=. uv run python experiments/exp27_tng_api_crossmatch/qa.py
PYTHONPATH=. uv run python experiments/exp27_tng_api_crossmatch/diffmah_consistency.py
```
Needs the TNG API key at `~/.tng_api_key` (line `TNG_API_KEY=…`, outside the repo).
