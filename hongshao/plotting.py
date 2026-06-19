"""Shared publication-quality plotting defaults for HongShao experiments.

Repo rule: every experiment should produce at least one figure. Call
``set_style()`` at the top of a driver and ``save_fig()`` to export PNG + PDF.
Colors follow the Okabe-Ito colorblind-safe palette; heatmaps default to the
perceptually-uniform, grayscale-safe ``cividis``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# Okabe-Ito colorblind-safe qualitative palette
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=OKABE_ITO),
        "image.cmap": "cividis",
    })


def save_fig(fig, path_stem, formats=("png", "pdf")):
    """Save a figure as <stem>.<fmt> for each format. Returns the paths."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        fig.savefig(out)
        paths.append(out)
    return paths
