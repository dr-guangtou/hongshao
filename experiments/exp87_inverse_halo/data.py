"""exp87 — the samples, the features read from a curve of growth, the frozen
lockbox and folds, and the box's halo-mass prior.

Two samples (the user's decision, 2026-10-02):

  parent   z = 0.4 only. Every TNG300 central with M200c(z = 0.4) >= 10^13 Msun
           that has a finite curve of growth (3380 of the 3388; the parent IS
           the box above the cut). Targets: halo mass, concentration,
           formation times, the z = 0.4-anchored DiffMAH parameters.
  curated  five epochs. The 2397 galaxies with a measured main-progenitor
           history, minus the INPUT-ONLY sanity cuts (the 3 dex backward rule,
           the > 1 dex stellar-mass jump, the > 0.3 dex inner drop). The repo's
           2356 fitting sample (which also drops galaxies > 0.5 dex off the
           M*-Mh relation, a cut that needs the truth) is carried as the flag
           `fit2356` for the sensitivity rows.

Inputs are log10 M*(<R) on the 24-point grid (2-148 kpc). Lockbox and folds
are assigned once per GALAXY on the parent and inherited by the curated
sample, so a galaxy sits in the same fold at every epoch and in both samples.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C                                       # noqa: E402
from hongshao.qa import enclosed_radius                  # noqa: E402

Z0P4_FITS = ROOT / "data/processed/tng300_072_z0p4.fits"
POP_NPZ = ROOT / "experiments/exp32_full_population/outputs/population.npz"
HS_NPZ = ROOT / "experiments/exp54_unpinned_amplitude/outputs/halo_structure_history.npz"
SPLIT_NPZ = C.OUTDIR / "splits.npz"
PRIOR_NPZ = C.OUTDIR / "box_prior.npz"
SIZE_FRACTIONS = (0.2, 0.5, 0.8)
#: a shell's mass is floored here before the log (non-monotone curves of growth, 0.8% of annuli at z = 2)
SHELL_FLOOR_MSUN = 1e6


def _selection_module():
    """exp54's `selection.py`, loaded by path (it inserts its own import paths)."""
    spec = importlib.util.spec_from_file_location(
        "exp54_selection", ROOT / "experiments/exp54_unpinned_amplitude/selection.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Sample:
    """One (population, epoch): inputs, targets and the frozen split."""
    name: str
    epoch: int
    index: np.ndarray                 # (n,) galaxy index in the raw drop
    logcog: np.ndarray                # (n, 24) log10 M*(<R) [Msun]
    targets: dict                     # name -> (n,) array (NaN where missing)
    radii: np.ndarray                 # (24,) kpc
    lockbox: np.ndarray               # (n,) bool
    fold: np.ndarray                  # (n,) int in 0..N_FOLDS-1 (-1 in the lockbox)
    flags: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.index)

    def subset(self, keep, name=None):
        keep = np.asarray(keep)
        return Sample(name or self.name, self.epoch, self.index[keep], self.logcog[keep],
                      {k: v[keep] for k, v in self.targets.items()}, self.radii, self.lockbox[keep], self.fold[keep],
                      {k: v[keep] for k, v in self.flags.items()})


# --------------------------------------------------------------------------- #
# features read from a curve of growth                                          #
# --------------------------------------------------------------------------- #
def aperture_logmass(logcog, radii, lo, hi):
    """log10 of the stellar mass between `lo` and `hi` kpc (hi None = the grid's
    end), by linear interpolation of the linear cumulative mass, as `qa.measure` does."""
    cog = 10.0 ** np.asarray(logcog, float)
    radii = np.asarray(radii, float)

    def cum(r):
        if r <= 0.0:
            return np.zeros(len(cog))
        return np.array([np.interp(r, radii, row) for row in cog])
    upper = cog[:, -1] if hi is None else cum(hi)
    return np.log10(np.clip(upper - cum(lo), 1.0, None))


def sizes(logcog, radii, fractions=SIZE_FRACTIONS):
    """log10 of the radii enclosing the given fractions of the grid's total mass, (n, n_frac)."""
    cog = 10.0 ** np.asarray(logcog, float)
    return np.log10(np.array([[enclosed_radius(row, radii, f) for f in fractions] for row in cog]))


def feature_set(sample, name):
    """(n, p) design of one named feature set, and the column labels.

    mtot        log M*(<148 kpc): the scalar stellar-mass relation
    <aperture>  one entry of `config.APERTURES`: a single aperture or annulus
    mass_size   total mass + log R20, R50, R80
    raw24       the 24 log cumulative masses (the whole curve of growth)
    shape23     total mass + the 23 log mass fractions M(<R_i)/M(<148): the same
                span as raw24 with the normalisation separated from the shape
    """
    x, radii = sample.logcog, sample.radii
    if name == "none":                                   # no stellar input: for the oracle baselines (extra columns only)
        return np.zeros((len(x), 0)), []
    if name == "mtot":
        return x[:, [-1]], ["logM148"]
    if name in C.APERTURES:
        lo, hi = C.APERTURES[name]
        return aperture_logmass(x, radii, lo, hi)[:, None], [name]
    if name in ("outer_shell", "mtot+outer_shell"):      # the mass between the grid's last two points (132-148 kpc)
        shell = aperture_logmass(x, radii, float(radii[-2]), None)[:, None]
        return (shell, ["logM(132-148)"]) if name == "outer_shell" else (np.column_stack([x[:, -1], shell]), ["logM148", "logM(132-148)"])
    if name == "shells24":                               # the DIFFERENTIAL profile: log mass inside 2 kpc, then of each shell
        cog = 10.0 ** x
        shells = np.column_stack([cog[:, 0], np.diff(cog, axis=1)])
        return np.log10(np.clip(shells, SHELL_FLOOR_MSUN, None)), ["logM(<2)"] + [f"sh{a:.0f}-{b:.0f}" for a, b in zip(radii[:-1], radii[1:])]
    if name == "mass_size":
        return np.column_stack([x[:, -1], sizes(x, radii)]), ["logM148", "logR20", "logR50", "logR80"]
    if name == "raw24":
        return x.copy(), [f"logM{r:.0f}" for r in radii]
    if name == "shape23":
        return np.column_stack([x[:, -1], x[:, :-1] - x[:, [-1]]]), ["logM148"] + [f"f{r:.0f}" for r in radii[:-1]]
    raise KeyError(f"unknown feature set {name!r}")


# --------------------------------------------------------------------------- #
# the frozen split                                                              #
# --------------------------------------------------------------------------- #
def _blocked_assignment(order_key, n_groups, rng):
    """Sort by `order_key`, cut into consecutive blocks of `n_groups`, and give
    each block a random permutation of 0..n_groups-1 (exp75's fold pattern):
    every group spans the whole range of the key."""
    order = np.argsort(order_key, kind="stable")
    out = np.empty(len(order_key), int)
    for start in range(0, len(order), n_groups):
        block = order[start:start + n_groups]
        out[block] = rng.permutation(n_groups)[:len(block)]
    return out


def make_splits(index, logmh, force=False):
    """Assign every parent galaxy to the lockbox (1 in 5) or to one of the five
    development folds, stratified in halo mass. Written once; later calls read it."""
    if SPLIT_NPZ.exists() and not force:
        z = np.load(SPLIT_NPZ)
        return z["index"], z["lockbox"], z["fold"]
    rng = np.random.default_rng(C.SEED)
    group = _blocked_assignment(logmh, int(round(1.0 / C.LOCKBOX_FRACTION)), rng)
    lockbox = group == 0
    fold = np.full(len(index), -1)
    dev = np.flatnonzero(~lockbox)
    fold[dev] = _blocked_assignment(logmh[dev], C.N_FOLDS, rng)
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(SPLIT_NPZ, index=index, lockbox=lockbox, fold=fold, seed=C.SEED)
    return index, lockbox, fold


def _split_for(index):
    z = np.load(SPLIT_NPZ)
    pos = {int(g): i for i, g in enumerate(z["index"])}
    rows = np.array([pos[int(g)] for g in index])
    return z["lockbox"][rows], z["fold"][rows]


# --------------------------------------------------------------------------- #
# the samples                                                                   #
# --------------------------------------------------------------------------- #
def load_parent(verbose=True):
    """The complete z = 0.4 population above the cut, with a finite curve of growth."""
    t = Table.read(Z0P4_FITS)
    radii = np.asarray(t.meta["cog_rad_kpc"], float)
    logcog = np.asarray(t["logmstar_cog"], float)
    keep = np.isfinite(logcog).all(1)
    index = np.asarray(t["index"], int)[keep]
    mh = np.asarray(t["logmh_z0p4"], float)[keep]
    make_splits(index, mh)
    lockbox, fold = _split_for(index)
    c200 = np.asarray(t["c_200c"], float)[keep]
    targets = dict(mh=mh, logc=np.log10(np.where(c200 > 0, c200, np.nan)))
    for key in ("t50", "t75", "t90", "dmah_logtc", "dmah_early", "dmah_late", "dmah_logmp", "f_early", "f_late"):
        targets[key] = np.asarray(t[key], float)[keep]
    flags = dict(use=np.asarray(t["use"], bool)[keep], mah_declined=np.asarray(t["mah_declined"], bool)[keep],
                 aper_proj=np.asarray(t["logmstar_aper_proj"], float)[keep])
    if verbose:
        print(f"  PARENT (z = 0.4): {keep.sum()} of {len(t)} centrals with M200c >= 10^{C.PARENT_CUT:g} have a finite "
              f"curve of growth; lowest halo mass {mh.min():.4f}; lockbox {lockbox.sum()}, development {(~lockbox).sum()}")
    return Sample("parent", 0, index, logcog[keep], targets, radii, lockbox, fold, flags)


def load_curated(epoch, verbose=True):
    """The main-progenitor sample at one epoch (0..4 = z 0.4..2), input-only cuts."""
    sel = _selection_module()
    pop = np.load(POP_NPZ)
    hs = np.load(HS_NPZ, allow_pickle=True)
    data = np.asarray(pop["data"], float)
    index = np.asarray(pop["index"], int)
    finite = np.isfinite(data).all(axis=(1, 2)) & (data > 0).all(axis=(1, 2))
    backward = sel.sane_history_mask(data).all(1)
    flag_a, flag_b, flag_c, _ = sel.stellar_history_flags(data, pop["logmh_zk_diffmah"])
    keep = finite & backward & ~flag_b.any(1) & ~flag_c.any(1)
    fit2356 = sel.fitting_sample_mask(data, pop["logmh_zk_diffmah"], verbose=False)
    mh = sel.sample_masses(hs)
    snaps = list(hs["snaps"])
    cols = [snaps.index(s) for s in C.ANCHOR_SNAP]
    ok = (np.asarray(hs["GroupFlag"])[:, cols] == 1)
    c200 = np.where(ok, np.asarray(hs["c200c"], float)[:, cols], np.nan)
    future_growth = mh[:, [0]] - mh
    if not SPLIT_NPZ.exists():
        load_parent(verbose=False)
    lockbox, fold = _split_for(index)
    logcog = np.log10(np.clip(data[:, epoch], 1.0, None))
    radii = np.asarray(Table.read(Z0P4_FITS).meta["cog_rad_kpc"], float)
    targets = dict(mh=mh[:, epoch], logc=np.log10(np.where(c200[:, epoch] > 0, c200[:, epoch], np.nan)),
                   future_growth=future_growth[:, epoch], mh_z0p4=mh[:, 0])
    flags = dict(fit2356=fit2356, complete=np.isfinite(mh[:, epoch]) & (mh[:, epoch] >= C.COMPLETE_CUTS[epoch]),
                 population_row=np.arange(len(index)))
    sample = Sample(f"curated_{C.EPOCH_TAG[epoch]}", epoch, index, logcog, targets, radii, lockbox, fold, flags)
    sample = sample.subset(keep)
    if verbose:
        print(f"  CURATED (z = {C.ANCHOR_Z[epoch]}): {keep.sum()} of {len(index)} main progenitors pass the input-only cuts "
              f"(dropped {(~finite).sum()} non-finite, {(finite & ~backward).sum()} by the 3 dex rule, "
              f"{(finite & backward & (flag_b.any(1) | flag_c.any(1))).sum()} jumps or drops); "
              f"{int(np.isfinite(sample.targets['mh']).sum())} have a halo mass at this epoch, "
              f"{int(sample.flags['complete'].sum())} sit above the completeness cut {C.COMPLETE_CUTS[epoch]}, "
              f"{int(sample.flags['fit2356'].sum())} are in the 2356 fitting sample")
    return sample


# --------------------------------------------------------------------------- #
# the box's halo-mass prior                                                     #
# --------------------------------------------------------------------------- #
def box_prior(force=False):
    """Smooth log-density of TNG300 centrals in log10 M200c at the five epochs.

    Read once from the raw halo-structure catalogue (`GroupFlag == 1`), binned
    in 0.1 dex, and smoothed by a cubic in log-count weighted by the counts
    (the mass function is smooth and steep; a kernel estimate would round the
    hard selection edge). Returns dict(centres (5, B), counts (5, B),
    coeff (5, 4), n_above_cut (5,)).
    """
    if PRIOR_NPZ.exists() and not force:
        return dict(np.load(PRIOR_NPZ))
    import h5py
    raw = Path(os.environ.get("HONGSHAO_HALO_STRUCTURE_DIR", C.HALO_STRUCTURE_DIR_DEFAULT))
    edges = np.arange(11.0, 15.61, 0.1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts, coeff, n_above = [], [], []
    for snap, cut in zip(C.ANCHOR_SNAP, [C.PARENT_CUT, *C.COMPLETE_CUTS[1:]]):
        with h5py.File(raw / f"halo_structure.{snap}.hdf5", "r") as f:
            flag, mass = f["GroupFlag"][:], f["M200c"][:]
        mass = np.asarray(mass[(flag == 1) & np.isfinite(mass) & (mass > 0)], float)
        hist = np.histogram(mass, edges)[0].astype(float)
        good = hist >= 5
        coeff.append(np.polyfit(centres[good], np.log(hist[good] / 0.1), 3, w=np.sqrt(hist[good])))
        counts.append(hist)
        n_above.append(int((mass >= cut).sum()))
    out = dict(centres=np.tile(centres, (5, 1)), counts=np.array(counts), coeff=np.array(coeff),
               n_above_cut=np.array(n_above))
    C.OUTDIR.mkdir(parents=True, exist_ok=True)
    np.savez(PRIOR_NPZ, **out)
    return out


def log_prior_density(y, epoch, prior=None):
    """log of the box's number density per dex at log halo mass `y` (unnormalised)."""
    prior = prior or box_prior()
    return np.polyval(prior["coeff"][epoch], np.asarray(y, float))
