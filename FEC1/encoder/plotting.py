"""Diagnostic plots for the common measurement raster."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .common_raster import CommonRaster


DEFAULT_PLOT_CHANNELS = (
    "active_psi_centered_norm",
    "active_b_r_asinh",
    "active_b_z_asinh",
    "active_field_spatial_valid",
    "thomson_core_log1p_te_eV",
    "thomson_core_log1p_ne_1e18_m3",
    "thomson_core_valid_support",
    "thomson_core_age_log1p",
    "thomson_edge_log1p_te_eV",
    "thomson_edge_log1p_ne_1e18_m3",
    "thomson_edge_valid_support",
    "thomson_edge_age_log1p",
)

_DIVERGING_CHANNELS = {
    "active_psi_centered_norm",
    "active_b_r_asinh",
    "active_b_z_asinh",
    "canonical_r",
    "canonical_z",
    "cylindrical_z_over_length",
}


def plot_common_raster(
    raster: CommonRaster,
    *,
    frame_index: int = 0,
    channels: Sequence[str] = DEFAULT_PLOT_CHANNELS,
    columns: int = 4,
    coordinates: str = "canonical",
    figure_title: str | None = None,
):
    """Plot selected channels and return ``(figure, axes)``.

    Parameters
    ----------
    coordinates:
        ``"canonical"`` uses the shared ``[-1, 1]`` canvas and is best for
        comparing machines. ``"physical"`` labels the same pixels in metres.
    """

    import matplotlib.pyplot as plt

    if frame_index < 0:
        frame_index += len(raster.times_ms)
    if frame_index < 0 or frame_index >= len(raster.times_ms):
        raise IndexError("frame_index is outside the raster time axis")
    selected = tuple(channels)
    if not selected:
        raise ValueError("at least one channel is required")
    if columns < 1:
        raise ValueError("columns must be positive")
    unknown = [name for name in selected if name not in raster.channel_names]
    if unknown:
        raise KeyError(f"unknown common-raster channels: {unknown}")
    if coordinates == "canonical":
        extent = [
            float(raster.canonical_r[0]),
            float(raster.canonical_r[-1]),
            float(raster.canonical_z[0]),
            float(raster.canonical_z[-1]),
        ]
        x_label, y_label = "canonical R", "canonical Z"
    elif coordinates == "physical":
        extent = [
            float(raster.grid_r_m[0]),
            float(raster.grid_r_m[-1]),
            float(raster.grid_z_m[0]),
            float(raster.grid_z_m[-1]),
        ]
        x_label, y_label = "R [m]", "Z [m]"
    else:
        raise ValueError("coordinates must be 'canonical' or 'physical'")

    rows = (len(selected) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.8 * columns, 3.25 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, name in zip(axes.flat, selected):
        values = np.asarray(raster.channel(name)[frame_index], dtype=np.float64)
        image_kwargs: dict[str, object] = {
            "origin": "lower",
            "extent": extent,
            "aspect": "auto",
        }
        if name in _DIVERGING_CHANNELS:
            limit = max(float(np.max(np.abs(values))), 1.0e-12)
            image_kwargs.update(cmap="coolwarm", vmin=-limit, vmax=limit)
        elif name.endswith(("_valid", "_support", "_coverage", "_fresh")):
            image_kwargs.update(cmap="viridis", vmin=0.0, vmax=1.0)
        else:
            image_kwargs.update(cmap="viridis")
        image = axis.imshow(values, **image_kwargs)
        axis.set_title(name.replace("_", " "), fontsize=9)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        figure.colorbar(image, ax=axis, shrink=0.82)
    for axis in axes.flat[len(selected) :]:
        axis.set_visible(False)
    if figure_title is None:
        figure_title = (
            f"{raster.source} common raster — "
            f"t={float(raster.times_ms[frame_index]):.3f} ms"
        )
    figure.suptitle(figure_title)
    return figure, axes


def plot_common_raster_comparison(
    rasters: Sequence[CommonRaster],
    *,
    frame_indices: Sequence[int] | None = None,
    channels: Sequence[str] = (
        "active_psi_centered_norm",
        "active_b_r_asinh",
        "active_b_z_asinh",
        "thomson_core_valid_support",
    ),
    labels: Sequence[str] | None = None,
    figure_title: str = "Common raster comparison",
):
    """Plot machines on shared canonical axes and shared channel color scales."""

    import matplotlib.pyplot as plt

    items = tuple(rasters)
    selected = tuple(channels)
    if not items:
        raise ValueError("at least one raster is required")
    if not selected:
        raise ValueError("at least one channel is required")
    if frame_indices is None:
        indices = (0,) * len(items)
    else:
        indices = tuple(frame_indices)
        if len(indices) != len(items):
            raise ValueError("frame_indices must have one entry per raster")
    if labels is None:
        row_labels = tuple(item.source for item in items)
    else:
        row_labels = tuple(labels)
        if len(row_labels) != len(items):
            raise ValueError("labels must have one entry per raster")
    for item, frame_index in zip(items, indices):
        if frame_index < 0 or frame_index >= len(item.times_ms):
            raise IndexError("frame index is outside one raster time axis")
        unknown = [name for name in selected if name not in item.channel_names]
        if unknown:
            raise KeyError(f"unknown common-raster channels: {unknown}")

    figure, axes = plt.subplots(
        len(items),
        len(selected),
        figsize=(3.65 * len(selected), 3.15 * len(items)),
        squeeze=False,
        constrained_layout=True,
    )
    extent = [-1.0, 1.0, -1.0, 1.0]
    for column, name in enumerate(selected):
        channel_values = [
            np.asarray(item.channel(name)[frame_index], dtype=np.float64)
            for item, frame_index in zip(items, indices)
        ]
        image_kwargs: dict[str, object] = {
            "origin": "lower",
            "extent": extent,
            "aspect": "auto",
        }
        if name in _DIVERGING_CHANNELS:
            limit = max(
                max(float(np.max(np.abs(values))) for values in channel_values),
                1.0e-12,
            )
            image_kwargs.update(cmap="coolwarm", vmin=-limit, vmax=limit)
        elif name.endswith(("_valid", "_support", "_coverage", "_fresh")):
            image_kwargs.update(cmap="viridis", vmin=0.0, vmax=1.0)
        else:
            lower = min(float(np.min(values)) for values in channel_values)
            upper = max(float(np.max(values)) for values in channel_values)
            if upper <= lower:
                upper = lower + 1.0e-12
            image_kwargs.update(cmap="viridis", vmin=lower, vmax=upper)
        column_images = []
        for row, (item, frame_index, values) in enumerate(
            zip(items, indices, channel_values)
        ):
            image = axes[row, column].imshow(values, **image_kwargs)
            column_images.append(image)
            if row == 0:
                axes[row, column].set_title(name.replace("_", " "), fontsize=9)
            axes[row, column].set_xlabel("canonical R")
            if column == 0:
                axes[row, column].set_ylabel(
                    f"{row_labels[row]}\ncanonical Z\n"
                    f"t={float(item.times_ms[frame_index]):.3f} ms"
                )
        figure.colorbar(
            column_images[0],
            ax=axes[:, column].tolist(),
            shrink=0.82,
        )
    figure.suptitle(figure_title)
    return figure, axes
