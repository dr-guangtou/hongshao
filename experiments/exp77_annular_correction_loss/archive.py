# %%
"""Final Exp77 inventory and exclusive, hash-verified artifact archival."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / "experiments/exp75_halo_residual_correction"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs():
    record = json.loads((HERE / "outputs/inputs.json").read_text())
    for name, expected in record["files_sha256"].items():
        if digest(REFERENCE / name) != expected:
            raise ValueError(f"Private reference changed: {name}")
    return record


def inventory():
    inputs = verify_inputs()
    destination = HERE / "outputs/manifest.json"
    paths = sorted(
        path
        for path in HERE.rglob("*")
        if path.is_file() and path != destination and "__pycache__" not in path.parts
    )
    record = {
        "experiment": HERE.name,
        "status": "complete_diagnostic_not_production",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "files_sha256": {str(path.relative_to(HERE)): digest(path) for path in paths},
        "reference_inputs": inputs,
        "note": "Frozen input snapshots are archived under Exp75; fitting checkpoints retain their own SHA and code hashes. No protected selection/validation data were used.",
    }
    with destination.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"manifest": str(destination), "inventoried_files": len(paths)}))


def copy_artifacts():
    inputs = verify_inputs()
    master = ROOT.parent / "hongshao_master"
    destination = master / "experiments" / HERE.name
    for name, expected in inputs["files_sha256"].items():
        if (
            digest(master / "experiments/exp75_halo_residual_correction" / name)
            != expected
        ):
            raise ValueError(f"Master reference input differs: {name}")
    paths = sorted(
        path
        for folder in (HERE / "outputs", HERE / "figures")
        for path in folder.rglob("*")
        if path.is_file()
    )
    for source in paths:
        target = destination / source.relative_to(HERE)
        if source.is_symlink() or target.is_symlink():
            raise ValueError("Artifact symlinks are not supported")
        if target.exists() and digest(source) != digest(target):
            raise FileExistsError(f"Different existing artifact: {target}")
    copied = 0
    for source in paths:
        target = destination / source.relative_to(HERE)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            copied += 1
        if digest(source) != digest(target):
            raise ValueError(f"Copy verification failed: {target}")
    print(
        json.dumps(
            {
                "destination": str(destination),
                "copied": copied,
                "verified": len(paths),
                "reference_inputs_verified": len(inputs["files_sha256"]),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-to-master", action="store_true")
    arguments = parser.parse_args()
    if arguments.copy_to_master:
        copy_artifacts()
    else:
        inventory()
