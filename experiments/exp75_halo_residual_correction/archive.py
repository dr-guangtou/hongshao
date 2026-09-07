# %%
"""Inventory Exp75 artifacts and copy them without replacing existing files."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_original():
    original = json.loads((HERE / "outputs/manifest.json").read_text())
    files = {
        name: value
        for name, value in original["files_sha256"].items()
        if name.startswith(("outputs/", "figures/"))
    }
    for name, value in files.items():
        if digest(HERE / name) != value:
            raise ValueError(f"Original artifact changed: {name}")
    return len(files)


def inventory():
    destination = HERE / "outputs/continuation/manifest.json"
    original_count = verify_original()
    paths = sorted(
        path
        for path in HERE.rglob("*")
        if path.is_file()
        and path != destination
        and "__pycache__" not in path.parts
        and path.suffix in (".py", ".md", ".json", ".npz", ".png", ".pdf")
    )
    record = {
        "experiment": HERE.name,
        "stage": "measured_history_discovery_complete_not_production",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "original_artifacts_verified": original_count,
        "protected_selection_validation_role": "none",
        "files_sha256": {str(path.relative_to(HERE)): digest(path) for path in paths},
        "note": "Fold records preserve fitting-time code hashes and SHA; this inventories final reporting code and artifacts. Original manifest is unchanged.",
    }
    with destination.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"manifest": str(destination), "files": len(paths)}))


def copy_artifacts(destination):
    destination = destination.resolve()
    expected = ROOT.parent / "hongshao_master/experiments" / HERE.name
    if destination != expected:
        raise ValueError(
            f"Only the declared archive destination is allowed: {expected}"
        )
    verify_original()
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
            {"destination": str(destination), "copied": copied, "verified": len(paths)}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy-to", type=Path)
    arguments = parser.parse_args()
    if arguments.copy_to:
        copy_artifacts(arguments.copy_to)
    else:
        inventory()
