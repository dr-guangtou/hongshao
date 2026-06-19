# Lessons

Mistakes, gotchas, and decisions worth remembering. Review at session start.

## Data handling (TNG300 drop)

- **Measure before encoding a data cut.** "Exclude halos whose mass decreases"
  sounded simple, but 98.6% of halos have *some* per-step decrease (median worst
  step ~4%) — normal fluctuation. The meaningful cut is net post-peak decline
  (z=0.4 mass >5% below historical peak → 695 halos). Always probe the
  distribution before choosing a threshold.
- **Verify snapshot indexing empirically, don't assume.** The MAH pickle is
  0-based TNG-native (snaps 1–71; snap 72 = z=0.4 is the unstored observation
  epoch). Confirmed by matching the cosmic-time file `t[snap]` to astropy ages
  at the anchors to <1 Myr. `t` indexes directly with no offset.
- **Don't trust file extensions.** `map_tng100_hist_stellar.hdf5` was 852 MB of
  all-zero bytes, not HDF5. Check magic bytes / load before relying on a file.
- **CoG / aperture arrays can be object-dtype with `None` bins** (~1/484). Coerce
  to NaN and mask; don't assume clean float arrays.

## Workflow

- Background `uv run` commands buffer stdout through pipes; redirect to a file
  and read that, or block on the process, rather than polling a pipe.
