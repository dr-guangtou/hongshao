# %%
"""Report invalid saved pre-epoch inputs without changing or excluding them."""

from continuation import BASE, OUTPUT, build_inputs, load_sample, write_record

import matplotlib.pyplot as plt
import numpy as np
from hongshao.plotting import save_fig, set_style


def run():
    sample = load_sample()
    curves, _ = build_inputs(sample, "pre_epoch")
    measured, _ = build_inputs(sample, "measured")
    with np.load(OUTPUT / "frozen_swap.npz") as archive:
        predictions = {
            name: archive[name] for name in ("official", "measured", "pre_epoch")
        }
    records = []
    for epoch in range(5):
        times = np.linspace(
            np.log10(BASE.PHYSICAL.E.T_START),
            np.log10(BASE.PHYSICAL.E.T_ANCHOR[epoch]),
            1000,
        )
        slopes = np.array(
            [
                np.min(
                    BASE.PHYSICAL.E.dm_dlnt(times, curve)
                    / 10 ** BASE.PHYSICAL.E.log_mah(times, curve)
                )
                for curve in curves[epoch]
            ]
        )
        value = predictions["pre_epoch"][:, epoch]
        invalid = np.any(~np.isfinite(value) | (value <= 0), axis=-1)
        records.append(
            {
                "redshift": BASE.REDSHIFTS[epoch],
                "negative_growth_ids": sample["indices"][slopes < -1e-12].tolist(),
                "negative_growth_count": int(np.sum(slopes < -1e-12)),
                "minimum_logarithmic_growth": float(slopes.min()),
                "invalid_stellar_prediction_ids": sample["indices"][invalid].tolist(),
                "invalid_stellar_prediction_count": int(invalid.sum()),
            }
        )
    set_style()
    plt.rcParams["text.usetex"] = False
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for column, galaxy in enumerate((2582, 1256)):
        row = int(np.flatnonzero(sample["indices"] == galaxy)[0])
        epoch = (0, 4)[column]
        times = np.linspace(
            np.log10(BASE.PHYSICAL.E.T_START),
            np.log10(BASE.PHYSICAL.E.T_ANCHOR[epoch]),
            500,
        )
        for label, curve, color in (
            ("Saved pre-epoch DiffMAH", curves[epoch][row], "#D55E00"),
            ("Measured-history interpolation", measured[0][row], "#0072B2"),
        ):
            mass = BASE.PHYSICAL.E.log_mah(times, curve)
            slope = BASE.PHYSICAL.E.dm_dlnt(times, curve) / 10**mass
            axes[0, column].plot(10**times, mass, color=color, label=label)
            axes[1, column].plot(10**times, slope, color=color)
        valid = (sample["GroupFlag"][row] == 1) & np.isfinite(sample["M200c"][row])
        knot_times = np.array(
            [BASE.PHYSICAL.E.T_SNAP[int(snap)] for snap in sample["snaps"]]
        )
        valid &= knot_times <= BASE.PHYSICAL.E.T_ANCHOR[epoch]
        axes[0, column].scatter(
            knot_times[valid],
            sample["M200c"][row, valid],
            c="k",
            s=14,
            label="Measured M200c",
        )
        axes[0, column].set_title(f"Galaxy {galaxy}; output z={BASE.REDSHIFTS[epoch]}")
        axes[0, column].set_ylabel(r"$\log_{10}(M_{200c}/M_\odot)$")
        axes[1, column].set_ylabel(r"$d\log M/d\log t$")
        axes[1, column].set_xlabel("Cosmic time (Gyr)")
        axes[1, column].axhline(0, color="0.5", lw=0.7)
        for axis in axes[:, column]:
            axis.set_xscale("log")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Some saved pre-epoch fits decrease between measured times\nNegative accretion is not a valid input to the positive-deposition model",
        fontsize=11,
    )
    fig.tight_layout()
    paths = save_fig(fig, BASE.HERE / "figures/continuation/pre_epoch_input_audit")
    plt.close(fig)
    record = {
        "status": "pre_epoch_input_invalid_as_supplied_no_refit",
        "epochs": records,
        "decision": "Do not clip negative accretion, remove galaxies, or call this a science null. Continue the valid measured-input comparison.",
        "frozen_valid_input_rms_dex_by_epoch": {
            name: np.sqrt(
                np.mean(np.log10(predictions[name] / sample["target"]) ** 2, axis=-1)
            )
            .mean(0)
            .tolist()
            for name in ("official", "measured")
        },
        "figures": [str(path) for path in paths],
    }
    write_record(OUTPUT / "history_audit.json", record)
    print(record, flush=True)


if __name__ == "__main__":
    run()
