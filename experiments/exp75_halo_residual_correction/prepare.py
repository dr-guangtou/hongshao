# %%
"""Extract hash-verified discovery-only private inputs; no source writes."""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from hongshao.profile_data import load_profiles  # noqa: E402
from hongshao.tng_data import COG_RAD_KPC  # noqa: E402


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def prepare(source_root, archive_root):
    output = HERE / "outputs" / "inputs.npz"
    if output.exists():
        raise FileExistsError("private inputs already exist; do not overwrite")
    paths = {
        "profiles": source_root / "data/processed/tng300_072_profiles_v2.npz",
        "population": source_root
        / "experiments/exp32_full_population/outputs/population.npz",
        "diffmah": source_root
        / "experiments/exp27_tng_api_crossmatch/outputs/diffmah_combined.fits",
        "structure": source_root
        / "experiments/exp54_unpinned_amplitude/outputs/halo_structure_history.npz",
        "atlas_index": archive_root
        / "experiments/exp62_cog_fit_atlas/outputs/full/atlas/predictions.npz",
        "discovery_index": archive_root
        / "experiments/exp67_squared_slope_symbolic_density/outputs/discovery_round_a/expressions/arctangent_w0p35.npz",
    }
    hashes = {key: file_hash(path) for key, path in paths.items()}
    with (
        np.load(paths["discovery_index"]) as archive,
        np.load(paths["atlas_index"]) as atlas,
    ):
        discovery = np.unique(atlas["rows"][archive["indices"]])
    if len(discovery) != 1200:
        raise ValueError("Exp67 discovery membership changed")
    with np.load(paths["population"]) as population:
        rows = np.flatnonzero(np.isin(population["index"], discovery))
        indices = population["index"][rows]
        reference = population["data"][rows]
        epoch_mass = population["logmh_zk_diffmah"][rows]
        concentration = population["c200c"][rows]
    profiles = load_profiles(paths["profiles"])
    target = profiles["cog_provided"][indices]
    del profiles
    np.testing.assert_allclose(target, reference, rtol=1e-12, atol=0)
    selection = load_module(
        "exp75_selection", ROOT / "experiments/exp54_unpinned_amplitude/selection.py"
    )
    keep = selection.fitting_sample_mask(target, epoch_mass)
    target, indices, epoch_mass, concentration = (
        value[keep] for value in (target, indices, epoch_mass, concentration)
    )
    table = Table.read(paths["diffmah"])
    lookup = {int(index): row for row, index in enumerate(table["index"])}
    table = table[[lookup[int(index)] for index in indices]]
    if not np.all(np.asarray(table["diffmah_source"]).astype(str) == "official"):
        raise ValueError("non-official DiffMAH input")
    halo = np.column_stack(
        [
            table[key]
            for key in (
                "official_logmp_z0",
                "official_logtc",
                "official_early",
                "official_late",
            )
        ]
    ).astype(float)
    with np.load(paths["structure"]) as structure:
        lookup = {int(index): row for row, index in enumerate(structure["index"])}
        structure_rows = [lookup[int(index)] for index in indices]
        epochs = [list(structure["snaps"]).index(snap) for snap in (72, 59, 50, 40, 33)]
        history = structure["c200c"][structure_rows][:, epochs]
    core = np.column_stack(
        [halo, np.log10(np.where(concentration > 0, concentration, np.nan))]
    )
    history = np.column_stack([halo, np.log10(np.where(history > 0, history, np.nan))])
    if np.any(np.diff(target, axis=-1) / target[..., :-1] < -1e-12):
        raise ValueError("provided CoG is not nondecreasing")
    after = {key: file_hash(path) for key, path in paths.items()}
    if hashes != after:
        raise RuntimeError("source changed during extraction; refusing to save")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        indices=indices,
        discovery_indices=discovery,
        halo=halo,
        core=core,
        history=history,
        target=target,
        epoch_mass=epoch_mass,
        radii=np.asarray(COG_RAD_KPC),
    )
    record = {
        "sources": {
            key: {"path": str(paths[key]), "sha256": hashes[key]} for key in paths
        },
        "admitted_galaxies": len(indices),
        "discovery_population_overlap": len(rows),
        "snapshot_sha256": file_hash(output),
    }
    output.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source_root.resolve(), args.archive_root.resolve())
