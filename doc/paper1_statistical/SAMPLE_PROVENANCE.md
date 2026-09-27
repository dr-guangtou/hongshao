# Parent-sample provenance — September 28, 2026

## Finding

Peak-mass selection is **supported by the project record but not yet certified
from the original selection query**. Preserve the original parent IDs; do not
silently substitute a present-day or snapshot-72 halo-mass cut. Retain all
qualifying centrals under the user-approved epoch-relevant measurement rules.

## Evidence reviewed

| Source | What it establishes | What it does not establish |
|---|---|---|
| User recollection, September 28 | Xu's original 3,388 objects were probably peak selected | Exact field, threshold convention, and selection implementation |
| `doc/ultimate_shmr_context.md`, lines 7–9; `doc/ultimate_shmr_possible_directions.md`, opening; `README.md`, data section | Project explicitly describes `Mpeak(z=0.4) > 10^13 Msun` | These are project descriptions, not independent copies of the original query |
| Original `save_tng300_072_file_structure.md`, sections 1–3 | Two matching sets indexed 0–3387, one central per index, five stellar epochs | Explicit catalog selection, completeness, or a documented central-finder criterion |
| Same supplied note, section 4 | Appends snapshot 72 and `tab2['mass_halo']`, converts history masses by `1e10/0.6774`, then takes a running maximum | This history construction does not prove that parent membership was selected by that maximum |
| `hongshao/tng_data.py`, `peak_history()` and legacy `use` | Later derived peak-history coordinates and analysis exclusions | The query that selected the original parent; this mask must not define the paper by default |
| `doc/tng300_data.md` | Older summary calls the parent massive centrals with `Mhalo > 10^13 Msun` | It is less specific than the motivation notes and cannot resolve peak versus instantaneous mass |

The original format note was read in full at:

`/Users/shuang/Desktop/tng300_mah_mprof/save_tng300_072_file_structure.md`

SHA-256:
`3374bad5ca11023c030c1540b06034bd02c52d2e7f54c81a290b489d7f3d725f`.

The bounded search found no original selection script among the raw drop's
Markdown/Python/notebook files. This is not a claim that such a script does
not exist elsewhere. No data-provider contact, new data query, or sample fit
was performed. No final paper-sample count has been measured.

## Important distinctions for the input audit

1. Parent membership, present halo-mass control, and history representation
   are three separate choices. Peak-selected membership does not force the
   forward reference to use peak mass alone.
2. The source recipe includes the exact endpoint; old history pickles can
   stop earlier. Verify the table-row/galaxy join before appending an endpoint.
   Do not assume similarly named mass fields share a halo definition.
3. Retain raw histories and any running-max histories separately. The recipe
   is inconsistent with treating every stored raw history as automatically
   monotonic; measure declines instead of assuming they are absent.
4. A match to the z=0 official DiffMAH catalog is not a neutral sample cut.
   The later Exp28 record identifies its history as SubhaloMass peak mass,
   not M200c. Prefer fitting the paper's matched history source.
5. Source `flag`/`test` descriptions are not sufficient substitutes for the
   later profile-v2 measurement audit. Certify relevant quality masks per
   measurement, and never discard an entire usable CoG just because an
   unrelated density bin or high-redshift profile is unavailable.

## Certification request and fallback

Seek the original query/script, or a written confirmation from Xu specifying:

- snapshot and central definition;
- the underlying mass field and its units, including little-h;
- peak over which branch and time interval, and the exact threshold/operator;
- any additional selection or measurement exclusions before delivery;
- whether the supplied IDs constitute all qualifying centrals or a subset.

Independently compare delivered masses/IDs with the claimed selection once
definitions are certified. Catalog agreement is a consistency test, not proof
of completeness without a reference parent catalog.

Until then, describe the input as **the supplied 3,388-central parent catalog**,
with the peak-selection statement explicitly marked provisional. Preparation,
data validation, and synthetic checks can proceed in `ushmr1`; pause before
freezing confirmatory science fits if the remaining ambiguity changes the
eligible population or the halo/history mass controls. No replacement cut
or completeness claim may be invented to avoid that pause.
