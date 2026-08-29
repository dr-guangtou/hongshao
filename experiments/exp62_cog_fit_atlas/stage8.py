# %%
"""Exp62 Stage 8: cubic-logit mathematical reference and library promotion.

Commands:

    uv run python experiments/exp62_cog_fit_atlas/stage8.py mechanics
    uv run python experiments/exp62_cog_fit_atlas/stage8.py demo
    uv run python experiments/exp62_cog_fit_atlas/stage8.py full
    uv run python experiments/exp62_cog_fit_atlas/stage8.py figures demo|full
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from hongshao.cubic_logit import (  # noqa: E402
    cubic_logit_aperture_cog,
    cubic_logit_cumulative_fraction,
    cubic_logit_density_stationary_radii,
    cubic_logit_enclosed_radius,
    cubic_logit_fourier_transform,
    cubic_logit_surface_density,
    cubic_logit_surface_density_log_slope,
    cubic_logit_warp,
    cubic_logit_warp_derivative,
    gaussian_psf_fourier_transform,
    moffat_psf_fourier_transform,
    render_cubic_logit_image,
)
from hongshao.plotting import OKABE_ITO, save_fig, set_style  # noqa: E402
from hongshao.provenance import write_manifest  # noqa: E402
from hongshao.qa import enclosed_radius  # noqa: E402
from run import OUTDIR, RADII, REDSHIFTS  # noqa: E402
from stage5 import STAGE5_OUTDIR  # noqa: E402

set_style()

STAGE8_OUTDIR = OUTDIR / "stage8"
STAGE8_FIGDIR = HERE / "figures" / "stage8"
PROFILE_NAME = "cubic-logit"
PSF_FWHM_KPC = 2.0
MOFFAT_BETA = 4.765


def _load(mode: str) -> dict[str, np.ndarray]:
    source = np.load(STAGE5_OUTDIR / mode / "predictions.npz")
    return {name: np.asarray(source[name]) for name in source.files}


def _physical_shape(parameters: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(10.0 ** parameters[1]),
        float(np.exp(parameters[2])),
        float(parameters[3]),
        float(np.exp(parameters[4])),
    )


def _promoted_predictions(parameters: np.ndarray) -> np.ndarray:
    predictions = np.empty((len(parameters), len(RADII)))
    for index, values in enumerate(parameters):
        shape = _physical_shape(values)
        predictions[index] = np.log10(
            np.clip(
                cubic_logit_aperture_cog(
                    RADII,
                    10.0 ** values[0],
                    RADII[-1],
                    *shape,
                ),
                1.0e-300,
                None,
            )
        )
    return predictions


def _stable_log_density_at_coordinates(
    coordinate: np.ndarray,
    derivative_margin: float,
    asymmetry: float,
    cubic_curvature: float,
) -> np.ndarray:
    warp = cubic_logit_warp(coordinate, derivative_margin, asymmetry, cubic_curvature)
    derivative = cubic_logit_warp_derivative(
        coordinate, derivative_margin, asymmetry, cubic_curvature
    )
    return (
        -np.logaddexp(0.0, -warp)
        - np.logaddexp(0.0, warp)
        + np.log(derivative)
        - 2.0 * coordinate
    )


def _profile_properties(parameters: np.ndarray) -> dict[str, np.ndarray]:
    n_profile = len(parameters)
    enclosed_radii = np.empty((n_profile, 3))
    density_peak_radius = np.empty(n_profile)
    density_peak_fraction = np.empty(n_profile)
    density_maximum_count = np.empty(n_profile, int)
    density_stationary_count = np.empty(n_profile, int)
    density_slope_at_2_kpc = np.empty(n_profile)
    measured_density_is_decreasing = np.empty(n_profile, bool)
    aperture_fraction = np.empty(n_profile)

    for index, values in enumerate(parameters):
        shape = _physical_shape(values)
        enclosed_radii[index] = cubic_logit_enclosed_radius(
            np.asarray([0.2, 0.5, 0.8]), *shape
        )
        stationary = cubic_logit_density_stationary_radii(*shape, grid_size=1024)
        if len(stationary) % 2 == 0:
            stationary = cubic_logit_density_stationary_radii(*shape, grid_size=4096)
        density_stationary_count[index] = len(stationary)
        density_maximum_count[index] = (len(stationary) + 1) // 2
        coordinate = np.log(stationary / shape[0])
        log_density = _stable_log_density_at_coordinates(coordinate, *shape[1:])
        density_peak_radius[index] = stationary[np.argmax(log_density)]
        density_peak_fraction[index] = cubic_logit_cumulative_fraction(
            density_peak_radius[index], *shape
        )
        density_slope_at_2_kpc[index] = cubic_logit_surface_density_log_slope(
            2.0, *shape
        )
        measured_slopes = cubic_logit_surface_density_log_slope(RADII, *shape)
        measured_density_is_decreasing[index] = np.all(measured_slopes <= 0.0)
        aperture_fraction[index] = cubic_logit_cumulative_fraction(RADII[-1], *shape)

    return {
        "enclosed_radii": enclosed_radii,
        "density_peak_radius": density_peak_radius,
        "density_peak_fraction": density_peak_fraction,
        "density_maximum_count": density_maximum_count,
        "density_stationary_count": density_stationary_count,
        "density_slope_at_2_kpc": density_slope_at_2_kpc,
        "measured_density_is_decreasing": measured_density_is_decreasing,
        "aperture_fraction": aperture_fraction,
    }


def _representative_index(parameters: np.ndarray) -> int:
    shape_coordinates = parameters[:, 1:]
    center = np.median(shape_coordinates, axis=0)
    scale = np.clip(np.median(np.abs(shape_coordinates - center), axis=0), 1.0e-6, None)
    distance = np.sum(((shape_coordinates - center) / scale) ** 2, axis=1)
    return int(np.argmin(distance))


def _radial_profile_from_image(
    image: np.ndarray,
    radius: np.ndarray,
    pixel_area: float,
    radial_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bin_index = np.searchsorted(radial_edges, radius.ravel(), side="right") - 1
    valid = (bin_index >= 0) & (bin_index < len(radial_edges) - 1)
    shell_mass = np.bincount(
        bin_index[valid],
        weights=image.ravel()[valid] * pixel_area,
        minlength=len(radial_edges) - 1,
    )
    shell_area = np.bincount(
        bin_index[valid],
        weights=np.full(np.count_nonzero(valid), pixel_area),
        minlength=len(radial_edges) - 1,
    )
    density = np.full_like(shell_mass, np.nan)
    np.divide(shell_mass, shell_area, out=density, where=shell_area > 0.0)
    cumulative = np.cumsum(shell_mass)
    return density, cumulative


def _representative_products(
    parameters: np.ndarray, mode: str
) -> dict[str, np.ndarray | float | int]:
    index = _representative_index(parameters)
    values = parameters[index]
    shape = _physical_shape(values)
    half_mass_radius = shape[0]

    scaled_frequency = np.linspace(0.0, 30.0, 241)
    frequency = scaled_frequency / half_mass_radius
    transform_1024 = cubic_logit_fourier_transform(
        frequency, *shape, quadrature_order=1024
    )
    transform_4096 = cubic_logit_fourier_transform(
        frequency, *shape, quadrature_order=4096
    )
    gaussian_sigma = PSF_FWHM_KPC / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    moffat_alpha = PSF_FWHM_KPC / (2.0 * np.sqrt(2.0 ** (1.0 / MOFFAT_BETA) - 1.0))
    gaussian_transfer = gaussian_psf_fourier_transform(frequency, gaussian_sigma)
    moffat_transfer = moffat_psf_fourier_transform(frequency, moffat_alpha, MOFFAT_BETA)

    npixel = 384 if mode == "demo" else 768
    extent = 10.0 * half_mass_radius
    pixel_scale = 2.0 * extent / npixel
    coordinate = (np.arange(npixel) + 0.5) * pixel_scale - extent
    x_coordinate, y_coordinate = np.meshgrid(coordinate, coordinate)
    radius = np.hypot(x_coordinate, y_coordinate)
    intrinsic = render_cubic_logit_image(x_coordinate, y_coordinate, 1.0, *shape)
    elliptical = render_cubic_logit_image(
        x_coordinate,
        y_coordinate,
        1.0,
        *shape,
        axis_ratio=0.6,
        position_angle=np.deg2rad(30.0),
    )
    gaussian_psf = np.exp(-0.5 * (radius / gaussian_sigma) ** 2) / (
        2.0 * np.pi * gaussian_sigma**2
    )
    moffat_psf = (
        (MOFFAT_BETA - 1.0)
        / (np.pi * moffat_alpha**2)
        * (1.0 + (radius / moffat_alpha) ** 2) ** (-MOFFAT_BETA)
    )
    pixel_area = pixel_scale**2
    gaussian_psf /= np.sum(gaussian_psf) * pixel_area
    moffat_psf /= np.sum(moffat_psf) * pixel_area
    gaussian_convolved = fftconvolve(intrinsic, gaussian_psf, mode="same") * pixel_area
    moffat_convolved = fftconvolve(intrinsic, moffat_psf, mode="same") * pixel_area

    radial_edges = np.geomspace(max(pixel_scale, 0.01), extent, 110)
    radial_centers = np.sqrt(radial_edges[:-1] * radial_edges[1:])
    radial_density = []
    radial_cumulative = []
    for image in (intrinsic, gaussian_convolved, moffat_convolved):
        density, cumulative = _radial_profile_from_image(
            image, radius, pixel_area, radial_edges
        )
        radial_density.append(density)
        radial_cumulative.append(cumulative)

    intrinsic_flux = float(np.sum(intrinsic) * pixel_area)
    gaussian_flux = float(np.sum(gaussian_convolved) * pixel_area)
    moffat_flux = float(np.sum(moffat_convolved) * pixel_area)
    cumulative_array = np.asarray(radial_cumulative)
    convolved_r50 = np.asarray(
        [
            np.interp(0.5 * cumulative[-1], cumulative, radial_centers)
            for cumulative in cumulative_array
        ]
    )
    return {
        "representative_index": index,
        "representative_parameters": values,
        "representative_shape": np.asarray(shape),
        "scaled_frequency": scaled_frequency,
        "transform_1024": transform_1024,
        "transform_4096": transform_4096,
        "gaussian_transfer": gaussian_transfer,
        "moffat_transfer": moffat_transfer,
        "image_extent": extent,
        "intrinsic_image": intrinsic,
        "elliptical_image": elliptical,
        "gaussian_convolved_image": gaussian_convolved,
        "moffat_convolved_image": moffat_convolved,
        "radial_centers": radial_centers,
        "radial_density": np.asarray(radial_density),
        "radial_cumulative": cumulative_array,
        "intrinsic_grid_flux": intrinsic_flux,
        "gaussian_same_grid_flux": gaussian_flux,
        "moffat_same_grid_flux": moffat_flux,
        "convolved_r50": convolved_r50,
        "pixel_scale": pixel_scale,
        "gaussian_sigma": gaussian_sigma,
        "moffat_alpha": moffat_alpha,
    }


def _anatomy_figure(
    mode: str, representative: dict[str, np.ndarray | float | int]
) -> None:
    shape = tuple(np.asarray(representative["representative_shape"]))
    half_mass_radius = shape[0]
    radius = half_mass_radius * np.geomspace(1.0e-4, 20.0, 1000)
    fraction = cubic_logit_cumulative_fraction(radius, *shape)
    density = cubic_logit_surface_density(radius, 1.0, *shape)
    slope = cubic_logit_surface_density_log_slope(radius, *shape)
    coordinate = np.log(radius / half_mass_radius)
    mass_per_log_radius = (
        fraction
        * (1.0 - fraction)
        * cubic_logit_warp_derivative(coordinate, *shape[1:])
    )
    characteristic = cubic_logit_enclosed_radius(np.asarray([0.2, 0.5, 0.8]), *shape)
    stationary = cubic_logit_density_stationary_radii(*shape)

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True)
    axes[0, 0].plot(radius / half_mass_radius, fraction, color=OKABE_ITO[4])
    axes[0, 0].set_ylabel(r"$M(<R)/M_{\rm tot}$")
    axes[0, 1].plot(
        radius / half_mass_radius,
        density * half_mass_radius**2,
        color=OKABE_ITO[1],
    )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel(r"$\Sigma R_{50}^2/M_{\rm tot}$")
    axes[1, 0].plot(radius / half_mass_radius, mass_per_log_radius, color=OKABE_ITO[2])
    axes[1, 0].set_ylabel(r"$d[M(<R)/M_{\rm tot}]/d\ln R$")
    axes[1, 1].plot(radius / half_mass_radius, slope, color=OKABE_ITO[5])
    axes[1, 1].axhline(0.0, color="0.5", linewidth=0.7)
    axes[1, 1].set_ylabel(r"$d\ln\Sigma/d\ln R$")
    for axis in axes.ravel():
        axis.set_xscale("log")
        axis.set_xlabel(r"$R/R_{50}$")
        for value, label in zip(characteristic, ("R20", "R50", "R80")):
            axis.axvline(value / half_mass_radius, color="0.65", linestyle=":")
            if axis is axes[0, 0]:
                axis.text(
                    value / half_mass_radius,
                    0.04,
                    label,
                    rotation=90,
                    ha="right",
                    va="bottom",
                    fontsize=7,
                )
        for value in stationary:
            axis.axvline(value / half_mass_radius, color="black", linestyle="--")
    figure.suptitle(
        "Cubic-logit profile anatomy: dotted lines are R20/R50/R80; dashed lines are density stationary points"
    )
    figure.tight_layout()
    save_fig(figure, STAGE8_FIGDIR / mode / "exp62_stage8_cubic_logit_anatomy")
    plt.close(figure)


def _parameter_response_figure(mode: str, data: dict[str, np.ndarray]) -> None:
    parameters = data["parameters_logit_cubic5"]
    physical_parameters = np.column_stack(
        (
            np.exp(parameters[:, 2]),
            parameters[:, 3],
            np.exp(parameters[:, 4]),
        )
    )
    centers = np.median(physical_parameters, axis=0)
    variations = np.quantile(physical_parameters, [0.16, 0.5, 0.84], axis=0)
    names = ("Derivative margin m", "Asymmetry b", "Cubic curvature c")
    symbols = ("m", "b", "c")
    radius_ratio = np.geomspace(0.01, 100.0, 500)
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), sharex=True)
    for column, (name, symbol) in enumerate(zip(names, symbols)):
        for quantile_index, (quantile, color) in enumerate(
            zip((0.16, 0.50, 0.84), (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4]))
        ):
            shape = centers.copy()
            shape[column] = variations[quantile_index, column]
            fraction = cubic_logit_cumulative_fraction(radius_ratio, 1.0, *shape)
            coordinate = np.log(radius_ratio)
            mass_per_log_radius = (
                fraction
                * (1.0 - fraction)
                * cubic_logit_warp_derivative(coordinate, *shape)
            )
            percentile = int(round(100.0 * quantile))
            label = rf"{symbol}={shape[column]:.3g} ({percentile}th percentile)"
            axes[0, column].plot(radius_ratio, fraction, color=color, label=label)
            axes[1, column].plot(radius_ratio, mass_per_log_radius, color=color)
        axes[0, column].set_title(name)
        axes[0, column].legend(frameon=False, fontsize=7)
        axes[1, column].set_xlabel(r"$R/R_{50}$")
        for axis in axes[:, column]:
            axis.set_xscale("log")
            axis.axvline(1.0, color="0.55", linestyle=":", linewidth=0.8)
    axes[0, 0].set_ylabel(r"$F(R)=M(<R)/M_{\rm tot}$")
    axes[1, 0].set_ylabel(r"$dF/d\ln R$")
    figure.text(
        0.5,
        0.975,
        r"Cubic-logit: $F=\sigma(ax+bx^2+cx^3)$, "
        r"$x=\ln(R/R_{50})$, $a=m+b^2/(3c)$, "
        r"$P'=m+3c[x+b/(3c)]^2>0$",
        ha="center",
        va="top",
        fontsize=12.0,
    )
    figure.text(
        0.5,
        0.935,
        "One coordinate varies through its fitted-population 16th, 50th, and 84th percentiles; the other two remain at their medians",
        ha="center",
        va="top",
        fontsize=8.0,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    save_fig(
        figure,
        STAGE8_FIGDIR / mode / "exp62_stage8_cubic_logit_parameter_responses",
    )
    plt.close(figure)


def _population_figure(
    mode: str, data: dict[str, np.ndarray], properties: dict[str, np.ndarray]
) -> None:
    parameters = data["parameters_logit_cubic5"]
    epochs = data["epochs"].astype(int)
    r20, r50, r80 = properties["enclosed_radii"].T
    values = (
        np.log10(properties["density_peak_radius"]),
        np.log10(properties["density_peak_radius"] / r50),
        np.log10(r80 / r20),
        -np.log10(properties["aperture_fraction"]),
        properties["density_slope_at_2_kpc"],
    )
    labels = (
        r"$\log_{10}(R_{\Sigma,\max}/{\rm kpc})$",
        r"$\log_{10}(R_{\Sigma,\max}/R_{50})$",
        r"$\log_{10}(R_{80}/R_{20})$",
        r"$\log_{10}(M_{\rm tot}/M_{148})$",
        r"$d\ln\Sigma/d\ln R$ at 2 kpc",
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.5))
    positions = np.arange(len(REDSHIFTS))
    for axis, quantity, label in zip(axes.ravel()[:5], values, labels):
        distributions = [quantity[epochs == epoch] for epoch in positions]
        plot = axis.boxplot(
            distributions,
            positions=positions,
            widths=0.6,
            showfliers=False,
            patch_artist=True,
        )
        for patch, color in zip(plot["boxes"], OKABE_ITO[:5]):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        axis.set_xticks(positions, [f"{redshift:g}" for redshift in REDSHIFTS])
        axis.set_xlabel("Redshift")
        axis.set_ylabel(label)
    fraction_peak_below_2 = [
        np.mean(properties["density_peak_radius"][epochs == epoch] < 2.0)
        for epoch in positions
    ]
    fraction_rising_at_2 = [
        np.mean(properties["density_slope_at_2_kpc"][epochs == epoch] > 0.0)
        for epoch in positions
    ]
    fraction_multimodal = [
        np.mean(properties["density_maximum_count"][epochs == epoch] > 1)
        for epoch in positions
    ]
    width = 0.24
    axes[1, 2].bar(
        positions - width,
        fraction_peak_below_2,
        width,
        color=OKABE_ITO[4],
        label="Global density maximum below 2 kpc",
    )
    axes[1, 2].bar(
        positions,
        fraction_rising_at_2,
        width,
        color=OKABE_ITO[1],
        label="Density rising at 2 kpc",
    )
    axes[1, 2].bar(
        positions + width,
        fraction_multimodal,
        width,
        color=OKABE_ITO[5],
        label="More than one density maximum",
    )
    axes[1, 2].set_xticks(positions, [f"{redshift:g}" for redshift in REDSHIFTS])
    axes[1, 2].set(xlabel="Redshift", ylabel="Fraction of fitted profiles", ylim=(0, 1))
    axes[1, 2].legend(frameon=False, fontsize=7)
    figure.suptitle(
        f"Exp62 Stage 8 {mode}: cubic-logit mathematical properties across {len(parameters):,} fitted galaxy--epoch profiles"
    )
    figure.tight_layout()
    save_fig(figure, STAGE8_FIGDIR / mode / "exp62_stage8_cubic_logit_population")
    plt.close(figure)


def _rendering_figure(
    mode: str, representative: dict[str, np.ndarray | float | int]
) -> None:
    images = (
        representative["intrinsic_image"],
        representative["elliptical_image"],
        representative["gaussian_convolved_image"],
        representative["moffat_convolved_image"],
    )
    titles = (
        "Circular intrinsic",
        "Elliptical intrinsic (q=0.6)",
        f"Gaussian PSF (FWHM={PSF_FWHM_KPC:g} kpc)",
        f"Moffat PSF (FWHM={PSF_FWHM_KPC:g} kpc, beta={MOFFAT_BETA:g})",
    )
    extent = float(representative["image_extent"])
    display_extent = min(extent, 4.0 * representative["representative_shape"][0])
    npixel = images[0].shape[0]
    coordinate = np.linspace(-extent, extent, npixel, endpoint=False)
    selected = np.abs(coordinate) <= display_extent
    figure = plt.figure(figsize=(14.0, 7.5))
    grid = figure.add_gridspec(2, 4, height_ratios=(1.0, 0.75))
    for column, (image, title) in enumerate(zip(images, titles)):
        axis = figure.add_subplot(grid[0, column])
        cutout = image[np.ix_(selected, selected)]
        positive = cutout[cutout > 0.0]
        floor = np.quantile(positive, 0.01)
        axis.imshow(
            np.log10(np.clip(cutout, floor, None)),
            origin="lower",
            extent=(-display_extent, display_extent, -display_extent, display_extent),
            cmap="cividis",
        )
        axis.set_title(title)
        axis.set_xlabel("x (kpc)")
        if column == 0:
            axis.set_ylabel("y (kpc)")
    radial_centers = representative["radial_centers"]
    radial_density = representative["radial_density"]
    radial_cumulative = representative["radial_cumulative"]
    density_axis = figure.add_subplot(grid[1, :2])
    cog_axis = figure.add_subplot(grid[1, 2:])
    for index, (label, color, linestyle) in enumerate(
        (
            ("Intrinsic", "black", "-"),
            ("Gaussian", OKABE_ITO[1], "--"),
            ("Moffat", OKABE_ITO[5], ":"),
        )
    ):
        density_axis.plot(
            radial_centers,
            radial_density[index],
            color=color,
            linestyle=linestyle,
            label=label,
        )
        cog_axis.plot(
            radial_centers,
            radial_cumulative[index] / radial_cumulative[index, -1],
            color=color,
            linestyle=linestyle,
            label=label,
        )
    density_axis.set(
        xscale="log",
        yscale="log",
        xlabel="R (kpc)",
        ylabel="Surface density (arbitrary normalization)",
    )
    cog_axis.set(xscale="log", xlabel="R (kpc)", ylabel="Enclosed grid-mass fraction")
    density_axis.legend(frameon=False)
    cog_axis.legend(frameon=False)
    figure.suptitle(
        "Cubic-logit 2-D rendering and numerical PSF convolution; convolution conserves flux before finite-grid cropping"
    )
    figure.tight_layout()
    save_fig(figure, STAGE8_FIGDIR / mode / "exp62_stage8_cubic_logit_rendering_psf")
    plt.close(figure)


def _fourier_figure(
    mode: str, representative: dict[str, np.ndarray | float | int]
) -> None:
    frequency = representative["scaled_frequency"]
    transform = representative["transform_4096"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(frequency, transform, color="black", label="Intrinsic")
    axes[0].plot(
        frequency,
        transform * representative["gaussian_transfer"],
        color=OKABE_ITO[1],
        linestyle="--",
        label="With Gaussian PSF",
    )
    axes[0].plot(
        frequency,
        transform * representative["moffat_transfer"],
        color=OKABE_ITO[5],
        linestyle=":",
        label="With Moffat PSF",
    )
    axes[0].axhline(0.0, color="0.6", linewidth=0.7)
    axes[0].set(xlabel=r"$kR_{50}$", ylabel="Normalized 2-D Fourier transform")
    axes[0].legend(frameon=False)
    axes[1].semilogy(
        frequency[1:],
        np.abs(
            representative["transform_4096"][1:] - representative["transform_1024"][1:]
        ),
        color=OKABE_ITO[2],
    )
    axes[1].set(
        xlabel=r"$kR_{50}$",
        ylabel="Absolute quadrature difference (4096 - 1024)",
    )
    figure.suptitle(
        "Cubic-logit Fourier behavior: numerical Hankel transform and PSF multiplication"
    )
    figure.tight_layout()
    save_fig(figure, STAGE8_FIGDIR / mode / "exp62_stage8_cubic_logit_fourier")
    plt.close(figure)


def _resolved_qa_figure(
    mode: str,
    data: dict[str, np.ndarray],
    promoted_prediction: np.ndarray,
) -> None:
    truth = data["truth_log"]
    stored = data["prediction_logit_cubic5"]
    epochs = data["epochs"].astype(int)
    halo_mass = data["halo_mass"]
    figure, axes = plt.subplots(2, len(REDSHIFTS), figsize=(17.0, 7.0))
    colors = (OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[4])
    for epoch, redshift in enumerate(REDSHIFTS):
        selected_epoch = epochs == epoch
        selected_halo = selected_epoch & np.isfinite(halo_mass)
        mass = halo_mass[selected_halo]
        edges = np.quantile(mass, [0.0, 1 / 3, 2 / 3, 1.0])
        for mass_bin, color in enumerate(colors):
            selected = (
                selected_halo
                & (halo_mass >= edges[mass_bin])
                & (halo_mass <= edges[mass_bin + 1])
            )
            axes[0, epoch].plot(
                RADII,
                np.median(truth[selected], axis=0),
                color=color,
                linestyle=":",
            )
            axes[0, epoch].plot(
                RADII,
                np.median(stored[selected], axis=0),
                color=color,
                linestyle="--",
            )
            axes[0, epoch].plot(
                RADII,
                np.median(promoted_prediction[selected], axis=0),
                color=color,
                label=rf"${edges[mass_bin]:.2f}<\log M_h<{edges[mass_bin + 1]:.2f}$",
            )
        epoch_truth = truth[selected_epoch]
        epoch_model = promoted_prediction[selected_epoch]
        stellar_mass = epoch_truth[:, -1]
        truth_r50 = np.asarray(
            [enclosed_radius(10.0**profile, RADII, 0.5) for profile in epoch_truth]
        )
        model_r50 = np.asarray(
            [enclosed_radius(10.0**profile, RADII, 0.5) for profile in epoch_model]
        )
        axes[1, epoch].scatter(
            stellar_mass,
            np.log10(truth_r50),
            s=4,
            color="0.65",
            alpha=0.18,
            rasterized=True,
        )
        mass_edges = np.quantile(stellar_mass, np.linspace(0.0, 1.0, 9))
        centers = []
        truth_median = []
        model_median = []
        for lower, upper in zip(mass_edges[:-1], mass_edges[1:]):
            in_bin = (stellar_mass >= lower) & (stellar_mass <= upper)
            centers.append(np.median(stellar_mass[in_bin]))
            truth_median.append(np.median(np.log10(truth_r50[in_bin])))
            model_median.append(np.median(np.log10(model_r50[in_bin])))
        axes[1, epoch].plot(
            centers, truth_median, "o-", color="black", label="TNG median"
        )
        axes[1, epoch].plot(
            centers,
            model_median,
            "s--",
            color=OKABE_ITO[5],
            label="Cubic-logit median",
        )
        axes[0, epoch].set(xscale="log", title=rf"$z={redshift:g}$", xlabel="R (kpc)")
        axes[1, epoch].set(
            xlabel=r"$\log_{10}M_{*,148}/M_\odot$",
            ylabel=r"$\log_{10}R_{50}/{\rm kpc}$",
        )
    axes[0, 0].set_ylabel(r"$\log_{10}M_*(<R)/M_\odot$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.0,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
    )
    figure.text(
        0.5,
        0.905,
        "Solid: promoted implementation; dashed: stored Stage 5 prediction; dotted: TNG",
        ha="center",
        fontsize=8.0,
    )
    axes[1, 0].legend(frameon=False, fontsize=7)
    figure.text(
        0.5,
        0.985,
        f"Exp62 Stage 8 {mode}: promoted cubic-logit implementation reproduces the stored resolved-range model",
        ha="center",
        va="top",
        fontsize=13.0,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    save_fig(figure, STAGE8_FIGDIR / mode / "exp62_stage8_cubic_logit_resolved_qa")
    plt.close(figure)


def make_figures(
    mode: str,
    data: dict[str, np.ndarray],
    properties: dict[str, np.ndarray],
    promoted_prediction: np.ndarray,
    representative: dict[str, np.ndarray | float | int],
) -> None:
    _anatomy_figure(mode, representative)
    _parameter_response_figure(mode, data)
    _population_figure(mode, data, properties)
    _rendering_figure(mode, representative)
    _fourier_figure(mode, representative)
    _resolved_qa_figure(mode, data, promoted_prediction)


def _epoch_values(values: np.ndarray, epochs: np.ndarray) -> list[list[float]]:
    return [
        np.quantile(values[epochs == epoch], [0.16, 0.5, 0.84]).tolist()
        for epoch in range(len(REDSHIFTS))
    ]


def run(mode: str) -> dict[str, object]:
    start = time.perf_counter()
    data = _load(mode)
    parameters = data["parameters_logit_cubic5"]
    promoted_prediction = _promoted_predictions(parameters)
    properties = _profile_properties(parameters)
    representative = _representative_products(parameters, mode)
    make_figures(mode, data, properties, promoted_prediction, representative)

    output_dir = STAGE8_OUTDIR / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "mathematical_properties.npz",
        rows=data["rows"],
        epochs=data["epochs"],
        redshifts=REDSHIFTS,
        parameters=parameters,
        promoted_prediction=promoted_prediction,
        **properties,
        representative_index=representative["representative_index"],
        representative_shape=representative["representative_shape"],
        scaled_frequency=representative["scaled_frequency"],
        fourier_transform=representative["transform_4096"],
        gaussian_transfer=representative["gaussian_transfer"],
        moffat_transfer=representative["moffat_transfer"],
    )

    epochs = data["epochs"].astype(int)
    r20, r50, r80 = properties["enclosed_radii"].T
    total_correction_dex = -np.log10(properties["aperture_fraction"])
    elapsed = time.perf_counter() - start
    summary = {
        "mode": mode,
        "profile_name": PROFILE_NAME,
        "n_profile": len(parameters),
        "n_galaxy": int(len(np.unique(data["rows"]))),
        "maximum_promoted_minus_stored_cog_difference_dex": float(
            np.max(np.abs(promoted_prediction - data["prediction_logit_cubic5"]))
        ),
        "density_peak_radius_kpc_16_50_84_by_epoch": _epoch_values(
            properties["density_peak_radius"], epochs
        ),
        "density_peak_over_r50_16_50_84_by_epoch": _epoch_values(
            properties["density_peak_radius"] / r50, epochs
        ),
        "density_peak_enclosed_fraction_16_50_84_by_epoch": _epoch_values(
            properties["density_peak_fraction"], epochs
        ),
        "log10_r80_over_r20_16_50_84_by_epoch": _epoch_values(
            np.log10(r80 / r20), epochs
        ),
        "log10_total_over_m148_16_50_84_by_epoch": _epoch_values(
            total_correction_dex, epochs
        ),
        "fraction_global_density_peak_below_2_kpc_by_epoch": [
            float(np.mean(properties["density_peak_radius"][epochs == epoch] < 2.0))
            for epoch in range(len(REDSHIFTS))
        ],
        "fraction_density_peak_encloses_less_than_1e_6": float(
            np.mean(properties["density_peak_fraction"] < 1.0e-6)
        ),
        "fraction_density_rising_at_2_kpc_by_epoch": [
            float(np.mean(properties["density_slope_at_2_kpc"][epochs == epoch] > 0.0))
            for epoch in range(len(REDSHIFTS))
        ],
        "fraction_density_decreasing_over_2_148_kpc_by_epoch": [
            float(
                np.mean(properties["measured_density_is_decreasing"][epochs == epoch])
            )
            for epoch in range(len(REDSHIFTS))
        ],
        "fraction_multiple_density_maxima_by_epoch": [
            float(np.mean(properties["density_maximum_count"][epochs == epoch] > 1))
            for epoch in range(len(REDSHIFTS))
        ],
        "maximum_stationary_count": int(np.max(properties["density_stationary_count"])),
        "log10_total_over_m148_global_quantiles_50_84_95_99_99p9_max": np.quantile(
            total_correction_dex, [0.5, 0.84, 0.95, 0.99, 0.999, 1.0]
        ).tolist(),
        "fourier_maximum_1024_vs_4096_absolute_difference_k_r50_0_30": float(
            np.max(
                np.abs(
                    representative["transform_1024"] - representative["transform_4096"]
                )
            )
        ),
        "representative_intrinsic_flux_inside_10_r50_grid": float(
            representative["intrinsic_grid_flux"]
        ),
        "representative_gaussian_convolved_flux_on_same_grid": float(
            representative["gaussian_same_grid_flux"]
        ),
        "representative_moffat_convolved_flux_on_same_grid": float(
            representative["moffat_same_grid_flux"]
        ),
        "representative_intrinsic_gaussian_moffat_r50_kpc": np.asarray(
            representative["convolved_r50"]
        ).tolist(),
        "wall_seconds": elapsed,
        "subminute_gate_passed": bool(elapsed < 60.0) if mode == "demo" else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_manifest(
        output_dir,
        {
            "experiment": "exp62_cog_fit_atlas",
            "stage": "8_cubic_logit_mathematical_reference",
            "mode": mode,
            "profile_name": PROFILE_NAME,
            "n_profile": len(parameters),
            "psf_fwhm_kpc": PSF_FWHM_KPC,
            "moffat_beta": MOFFAT_BETA,
        },
    )
    print(json.dumps(summary, indent=2))
    if mode == "demo" and elapsed >= 60.0:
        raise RuntimeError(
            f"Stage 8 demo took {elapsed:.2f} seconds; full run forbidden"
        )
    return summary


def figures(mode: str) -> None:
    data = _load(mode)
    stored = np.load(STAGE8_OUTDIR / mode / "mathematical_properties.npz")
    properties = {
        name: np.asarray(stored[name])
        for name in (
            "enclosed_radii",
            "density_peak_radius",
            "density_peak_fraction",
            "density_maximum_count",
            "density_stationary_count",
            "density_slope_at_2_kpc",
            "measured_density_is_decreasing",
            "aperture_fraction",
        )
    }
    representative = _representative_products(data["parameters_logit_cubic5"], mode)
    make_figures(
        mode,
        data,
        properties,
        np.asarray(stored["promoted_prediction"]),
        representative,
    )


def mechanics() -> None:
    data = _load("demo")
    parameters = data["parameters_logit_cubic5"][:3]
    promoted = _promoted_predictions(parameters)
    assert promoted.shape == (3, len(RADII))
    assert np.all(np.diff(promoted, axis=1) > 0.0)
    assert np.max(np.abs(promoted - data["prediction_logit_cubic5"][:3])) < 1.0e-12
    properties = _profile_properties(parameters)
    assert np.all(properties["density_peak_radius"] > 0.0)
    assert np.all(properties["density_stationary_count"] % 2 == 1)
    print("exp62 Stage 8 mechanics OK")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "mechanics"
    if command == "mechanics":
        mechanics()
    elif command in ("demo", "full"):
        run(command)
    elif command == "figures":
        figure_mode = sys.argv[2] if len(sys.argv) > 2 else "full"
        figures(figure_mode)
    else:
        raise SystemExit(__doc__)
