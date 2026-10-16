"""Explicit projection of permitted public-input parquet columns."""
from __future__ import annotations


_SAFE_FIXED_COLUMNS = {
    "source",
    "efit_times",
    "efit_grid_R",
    "efit_grid_Z",
    "magnetics_time",
    "magnetics_plasma_current_times",
}


_SAFE_PREFIXES = ("coil_", "thomson_")


_FORBIDDEN_EXACT = {
    "efit_psirz",
    "efit_beta_n",
    "efit_li",
    "efit_q95",
    "efit_r_axis",
    "efit_z_axis",
    "efit_lcfs_n",
    "efit_lcfs_r",
    "efit_lcfs_z",
    "magnetics_dsep",
    "magnetics_dsep_times",
}


def safe_input_columns(column_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Return the explicit target-free parquet projection used by the cache."""

    selected = []
    for name in column_names:
        if name in _FORBIDDEN_EXACT:
            continue
        if (
            name in _SAFE_FIXED_COLUMNS
            or name.startswith(_SAFE_PREFIXES)
            or name.startswith("magnetics_")
        ):
            selected.append(name)
    forbidden = _FORBIDDEN_EXACT.intersection(selected)
    if forbidden or any(
        name.startswith("efit_") and name not in _SAFE_FIXED_COLUMNS for name in selected
    ):
        raise RuntimeError(f"target columns crossed the input firewall: {forbidden}")
    required = {
        "source",
        "efit_times",
        "efit_grid_R",
        "efit_grid_Z",
        "magnetics_plasma_current",
        "coil_input_column",
    }
    missing = required.difference(selected)
    if missing:
        raise RuntimeError(f"raw parquet is missing required inputs: {sorted(missing)}")
    return tuple(selected)
