"""Record which code version + parameters produced an experiment's outputs.

Each experiment driver calls ``write_manifest(outdir, params)`` so that any
figure or table can be traced back to a git commit, even though outputs are not
committed to the repo.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def git_sha(short: bool = True) -> str:
    return _git("rev-parse", "--short", "HEAD") if short else _git("rev-parse", "HEAD")


def is_dirty() -> bool:
    """True if the working tree has uncommitted changes."""
    return bool(_git("status", "--porcelain"))


def write_manifest(outdir, params: dict | None = None) -> dict:
    """Write outdir/manifest.json with git SHA, dirty flag, time, and params."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git_sha": git_sha(),
        "git_dirty": is_dirty(),
        "created": datetime.now().isoformat(timespec="seconds"),
        "params": params or {},
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
