"""Challenge-row preprocessing and shot-sharded training data."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from data_fixes import align_mast_thomson_core
from .config import NeuralGSConfig
from .geometry import render_magnetic_inputs

ELECTRON_VOLT_J = 1.602176634e-19
TARGET_AXIS_SIGN = {"DIII-D": -1.0, "MAST": 1.0}


def as_profiles(raw) -> np.ndarray:
    try:
        array = np.asarray(raw, dtype=np.float64)
        if array.ndim == 2:
            return array
    except (TypeError, ValueError):
        pass
    return np.stack([np.asarray(values, dtype=np.float64) for values in raw])


def as_psirz(raw) -> np.ndarray:
    try:
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim == 3:
            return array
    except (TypeError, ValueError):
        pass
    return np.stack(
        [np.stack([np.asarray(line, dtype=np.float32) for line in frame]) for frame in raw]
    )


def _system_coordinates(row: dict, system: str, channels: int) -> np.ndarray:
    names = np.asarray(row.get("thomson_chord_name", []), dtype=str)
    chord_r = np.asarray(row.get("thomson_chord_R", []), dtype=np.float64)
    chord_z = np.asarray(row.get("thomson_chord_Z", []), dtype=np.float64)
    n = min(len(names), len(chord_r), len(chord_z))
    lower = np.char.lower(names[:n]) if n else np.asarray([], dtype=str)
    if system == "core":
        selected = np.char.find(lower, "core") >= 0
    else:
        selected = (np.char.find(lower, "edge") >= 0) | (np.char.find(lower, "tangential") >= 0)
    coords = np.column_stack([chord_r[:n][selected], chord_z[:n][selected]]) if selected.any() else np.empty((0, 2))
    if len(coords) == channels:
        return coords

    source = str(row.get("source", "DIII-D"))
    if system == "core":
        radii = np.asarray(row.get("thomson_core_R", []), dtype=np.float64)
        if len(radii) >= channels:
            radii = radii[-channels:]
        else:
            fill = float(np.nanmean(radii)) if len(radii) else float(np.mean(row["efit_grid_R"]))
            radii = np.pad(radii, (channels - len(radii), 0), constant_values=fill)
        return np.column_stack([radii, np.zeros(channels)])
    spatial = np.asarray(row.get("thomson_edge_spatial", []), dtype=np.float64)
    if len(spatial) >= channels:
        spatial = spatial[-channels:]
    else:
        spatial = np.pad(spatial, (channels - len(spatial), 0), constant_values=0.0)
    if source == "MAST":
        return np.column_stack([spatial, np.zeros(channels)])
    return np.column_stack([np.full(channels, np.mean(row["efit_grid_R"])), spatial])


def _nearest_profiles(row: dict, system: str, query: np.ndarray):
    time_key = f"thomson_{system}_times"
    if time_key not in row:
        return None
    times = np.asarray(row[time_key], dtype=np.float64)
    if system == "core":
        _, te, ne = align_mast_thomson_core(row)
        te, ne = np.asarray(te, dtype=np.float64), np.asarray(ne, dtype=np.float64)
    else:
        te = as_profiles(row[f"thomson_{system}_Te"])
        ne = as_profiles(row[f"thomson_{system}_ne"])
    nt = min(len(times), len(te), len(ne))
    times, te, ne = times[:nt], te[:nt], ne[:nt]
    finite_times = np.isfinite(times)
    if not finite_times.any():
        return None
    valid_indices = np.flatnonzero(finite_times)
    nearest_local = np.abs(query[:, None] - times[finite_times][None, :]).argmin(axis=1)
    nearest = valid_indices[nearest_local]
    age_s = (query - times[nearest]) / 1000.0
    return te[nearest], ne[nearest], age_s


def evenly_spaced_frame_indices(n_frames: int, max_frames: int) -> np.ndarray:
    """Select deterministic bin centers while retaining every shot."""

    n_frames = max(int(n_frames), 0)
    max_frames = int(max_frames)
    if max_frames <= 0 or n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)
    # Bin centers avoid over-representing startup and shutdown while covering
    # the complete shot timeline.  The floor operation is unique when k <= n.
    return np.floor((np.arange(max_frames) + 0.5) * n_frames / max_frames).astype(np.int64)


def _frame_indices(row: dict, frame_indices: np.ndarray | None) -> np.ndarray:
    n_frames = len(np.asarray(row["efit_times"]).ravel())
    if frame_indices is None:
        return np.arange(n_frames, dtype=np.int64)
    selected = np.asarray(frame_indices, dtype=np.int64).ravel()
    if len(selected) and (selected.min() < 0 or selected.max() >= n_frames):
        raise IndexError("frame index is outside efit_times")
    return selected


def build_thomson_tokens(
    row: dict,
    config: NeuralGSConfig,
    frame_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return physical coordinates, token values, and validity masks.

    Token features are ``log Te``, ``log ne``, ``log pe``, signed age in
    seconds, and a core/edge identifier.  R/Z remain separate because they are
    used only to sample the current iterate; they are not fed to attention.
    """

    selected = _frame_indices(row, frame_indices)
    query = np.asarray(row["efit_times"], dtype=np.float64)[selected]
    pieces = []
    for system_index, system in enumerate(("core", "edge")):
        nearest = _nearest_profiles(row, system, query)
        if nearest is None:
            continue
        te, ne, age = nearest
        channels = min(te.shape[1], ne.shape[1])
        te, ne = te[:, :channels], ne[:, :channels]
        coords = _system_coordinates(row, system, channels)
        channels = min(channels, len(coords))
        te, ne, coords = te[:, :channels], ne[:, :channels], coords[:channels]
        pressure = np.maximum(te, 0.0) * np.maximum(ne, 0.0) * ELECTRON_VOLT_J
        features = np.stack(
            [
                np.log1p(np.maximum(te, 0.0) / 100.0),
                np.log1p(np.maximum(ne, 0.0) / 1.0e18),
                np.log1p(pressure / 1.0e3),
                np.broadcast_to(np.clip(age, -2.0, 2.0)[:, None], te.shape),
                np.full(te.shape, -1.0 if system_index == 0 else 1.0),
            ],
            axis=-1,
        )
        mask = (
            np.isfinite(te) & np.isfinite(ne) & (te > 0.0) & (ne > 0.0)
            & np.isfinite(coords[:, 0])[None, :] & np.isfinite(coords[:, 1])[None, :]
        )
        # Attention masks do not make NaN token payloads harmless: the input
        # projection runs before masking and NaNs can still contaminate attention.
        # Replace payloads only after preserving their invalidity in ``mask``.
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        pieces.append((coords, features, mask))

    total = sum(len(coords) for coords, _, _ in pieces)
    keep = min(total, config.max_tokens)
    coords_out = np.zeros((len(query), config.max_tokens, 2), dtype=np.float32)
    values_out = np.zeros((len(query), config.max_tokens, config.token_features), dtype=np.float32)
    mask_out = np.zeros((len(query), config.max_tokens), dtype=bool)
    if not pieces or keep == 0:
        return coords_out, values_out, mask_out
    coords = np.concatenate([part[0] for part in pieces], axis=0)
    values = np.concatenate([part[1] for part in pieces], axis=1)
    mask = np.concatenate([part[2] for part in pieces], axis=1)
    if total > config.max_tokens:
        selected = np.unique(np.linspace(0, total - 1, config.max_tokens).round().astype(int))
        coords, values, mask = coords[selected], values[:, selected], mask[:, selected]
        keep = len(selected)
    coords_out[:, :keep] = coords[None, :keep]
    values_out[:, :keep] = values[:, :keep]
    mask_out[:, :keep] = mask[:, :keep]
    return coords_out, values_out, mask_out


def _bilinear_numpy(field: np.ndarray, r: np.ndarray, z: np.ndarray, rq, zq) -> np.ndarray:
    rq = np.asarray(rq, dtype=np.float64)
    zq = np.asarray(zq, dtype=np.float64)
    fr = np.interp(rq, r, np.arange(len(r)))
    fz = np.interp(zq, z, np.arange(len(z)))
    r0 = np.clip(np.floor(fr).astype(int), 0, len(r) - 2)
    z0 = np.clip(np.floor(fz).astype(int), 0, len(z) - 2)
    tr, tz = fr - r0, fz - z0
    return (
        field[z0, r0] * (1 - tr) * (1 - tz)
        + field[z0, r0 + 1] * tr * (1 - tz)
        + field[z0 + 1, r0] * (1 - tr) * tz
        + field[z0 + 1, r0 + 1] * tr * tz
    )


def target_normalized_flux(
    row: dict,
    psi: np.ndarray,
    frame_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Build teacher psi_N from EFIT axis and LCFS labels when available."""

    r = np.asarray(row["efit_grid_R"], dtype=np.float64)
    z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    result = np.empty_like(psi, dtype=np.float32)
    raxis = np.asarray(row.get("efit_r_axis", np.full(len(psi), np.nan)), dtype=np.float64)
    zaxis = np.asarray(row.get("efit_z_axis", np.full(len(psi), np.nan)), dtype=np.float64)
    counts = np.asarray(row.get("efit_lcfs_n", np.zeros(len(psi))), dtype=np.int64)
    lcfs_r = row.get("efit_lcfs_r")
    lcfs_z = row.get("efit_lcfs_z")
    selected = (
        np.arange(len(psi), dtype=np.int64)
        if frame_indices is None
        else np.asarray(frame_indices, dtype=np.int64).ravel()
    )
    if len(selected) != len(psi):
        raise ValueError("frame_indices must align with psi")
    for output_index, (source_index, frame) in enumerate(zip(selected, psi)):
        if (
            source_index < len(raxis)
            and np.isfinite(raxis[source_index])
            and np.isfinite(zaxis[source_index])
        ):
            axis = float(
                _bilinear_numpy(
                    frame, r, z, [raxis[source_index]], [zaxis[source_index]]
                )[0]
            )
        else:
            central = frame[4:-4, 4:-4]
            axis = float(np.nanmax(central))
        boundary = np.nan
        if (
            lcfs_r is not None
            and lcfs_z is not None
            and source_index < len(counts)
            and counts[source_index] >= 8
        ):
            n = int(counts[source_index])
            cr = np.asarray(lcfs_r[source_index], dtype=np.float64)[:n]
            cz = np.asarray(lcfs_z[source_index], dtype=np.float64)[:n]
            good = np.isfinite(cr) & np.isfinite(cz)
            if good.sum() >= 8:
                boundary = float(np.nanmedian(_bilinear_numpy(frame, r, z, cr[good], cz[good])))
        if not np.isfinite(boundary):
            perimeter = np.concatenate([frame[0], frame[-1], frame[1:-1, 0], frame[1:-1, -1]])
            boundary = float(np.nanpercentile(perimeter, 90.0))
        if boundary >= axis:
            # Incomplete startup labels can put the supplied axis outside the
            # closed-flux region.  Keep the teacher finite and weakly scaled;
            # scalar-label NaNs will still mask those frames where applicable.
            boundary = axis - max(0.10 * float(np.nanstd(frame)), 1.0e-4)
        denominator = max(axis - boundary, 1.0e-6)
        result[output_index] = (axis - frame) / denominator
    return result


def preprocess_row(
    row: dict,
    config: NeuralGSConfig,
    include_targets: bool = True,
    frame_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Convert one raw challenge shot into the NeuralGS tensor contract."""

    grid_r = np.asarray(row["efit_grid_R"], dtype=np.float32)
    grid_z = np.asarray(row["efit_grid_Z"], dtype=np.float32)
    if len(grid_r) != config.grid_size or len(grid_z) != config.grid_size:
        raise ValueError(f"expected {config.grid_size}x{config.grid_size} grid")
    selected = _frame_indices(row, frame_indices)
    magnetic = render_magnetic_inputs(row, selected)
    coords, values, mask = build_thomson_tokens(row, config, selected)
    n = len(magnetic.ip_abs)
    out: dict[str, np.ndarray] = {
        "vacuum_psi": magnetic.vacuum_psi[:n],
        "ip_abs": magnetic.ip_abs[:n],
        "global_features": magnetic.global_features[:n],
        "canonical_global_features": magnetic.canonical_global_features[:n],
        "tf_ampere": magnetic.tf_ampere[:n],
        "tf_turns": magnetic.tf_turns[:n],
        "token_coords": coords[:n],
        "token_values": values[:n],
        "token_mask": mask[:n],
        "grid_r": grid_r,
        "grid_z": grid_z,
        "canonical_current_sign": np.asarray([magnetic.canonical_sign], dtype=np.float32),
    }
    if include_targets and "efit_psirz" in row:
        target = as_psirz(row["efit_psirz"])[selected]
        target_factor = TARGET_AXIS_SIGN.get(str(row.get("source", "DIII-D")), 1.0)
        canonical = target_factor * target
        out["target_psi"] = canonical.astype(np.float32)
        out["target_psi_n"] = target_normalized_flux(
            row, canonical, selected
        ).astype(np.float32)
        for output, column in (("target_q95", "efit_q95"), ("target_betaN", "efit_beta_n")):
            all_frames = len(np.asarray(row["efit_times"]).ravel())
            values_target = np.asarray(
                row.get(column, np.full(all_frames, np.nan)), dtype=np.float32
            ).ravel()
            padded = np.full(all_frames, np.nan, dtype=np.float32)
            padded[: min(all_frames, len(values_target))] = values_target[:all_frames]
            out[output] = padded[selected]
    return out


def save_prepared_shot(
    row: dict,
    path: Path,
    config: NeuralGSConfig,
    frame_indices: np.ndarray | None = None,
    compressed: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prepared = preprocess_row(
        row, config, include_targets=True, frame_indices=frame_indices
    )
    writer = np.savez_compressed if compressed else np.savez
    writer(path, **prepared)


PER_FRAME_KEYS = {
    "vacuum_psi", "ip_abs", "global_features", "canonical_global_features",
    "tf_ampere", "tf_turns", "token_coords", "token_values", "token_mask", "target_psi",
    "target_psi_n", "target_q95", "target_betaN",
    "oracle_current", "oracle_source_coefficients", "oracle_target_quality",
}


class PreparedFrameDataset:
    """Frame-indexed view over compressed, shot-sharded ``.npz`` files."""

    def __init__(
        self,
        paths: list[Path],
        frame_stride: int = 1,
        oracle_dir: Path | None = None,
    ):
        self.paths = [Path(path) for path in paths]
        self.oracle_dir = Path(oracle_dir) if oracle_dir is not None else None
        self.index: list[tuple[int, int]] = []
        frame_stride = max(int(frame_stride), 1)
        for shot_index, path in enumerate(self.paths):
            with np.load(path, allow_pickle=False) as shot:
                length = len(shot["ip_abs"])
            self.index.extend((shot_index, frame) for frame in range(0, length, frame_stride))

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    @lru_cache(maxsize=8)
    def _load(path: str) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as shot:
            return {key: shot[key] for key in shot.files}

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_oracle(path: str) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as oracle:
            return {key: oracle[key] for key in oracle.files}

    def __getitem__(self, item: int) -> dict:
        import torch

        shot_index, frame = self.index[item]
        shot = self._load(str(self.paths[shot_index].resolve()))
        oracle = None
        if self.oracle_dir is not None:
            oracle_path = self.oracle_dir / self.paths[shot_index].name
            if not oracle_path.exists():
                raise FileNotFoundError(f"oracle target is missing: {oracle_path}")
            oracle = self._load_oracle(str(oracle_path.resolve()))
        output = {}
        for key, value in shot.items():
            array = value[frame] if key in PER_FRAME_KEYS else value
            output[key] = torch.from_numpy(np.asarray(array))
        if oracle is not None:
            for key, value in oracle.items():
                if key in PER_FRAME_KEYS:
                    output[key] = torch.from_numpy(np.asarray(value[frame]))
        return output


class PreparedShotDataset:
    """Shot-indexed view that reads each archive once per epoch.

    The frame-level dataset is convenient for small experiments, but globally
    shuffling millions of frame indices makes the eight-entry archive cache
    thrash.  Compute-light training shuffles shots and concatenates their
    already-sampled frames in ``collate_prepared_shots`` instead.
    """

    def __init__(
        self,
        paths: list[Path],
        frame_stride: int = 1,
        oracle_dir: Path | None = None,
    ):
        self.paths = [Path(path) for path in paths]
        self.oracle_dir = Path(oracle_dir) if oracle_dir is not None else None
        self.frame_stride = max(int(frame_stride), 1)
        self.lengths: list[int] = []
        for path in self.paths:
            with np.load(path, allow_pickle=False) as shot:
                n_frames = len(shot["ip_abs"])
            self.lengths.append(len(range(0, n_frames, self.frame_stride)))

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def max_frames(self) -> int:
        return max(self.lengths, default=1)

    @property
    def n_frames(self) -> int:
        return sum(self.lengths)

    def __getitem__(self, item: int) -> dict:
        import torch

        with np.load(self.paths[item], allow_pickle=False) as shot:
            output = {}
            for key in shot.files:
                array = shot[key]
                if key in PER_FRAME_KEYS:
                    array = array[:: self.frame_stride]
                # Own the data before closing the archive.  This also avoids
                # non-writable-array warnings from torch in worker processes.
                output[key] = torch.from_numpy(np.array(array, copy=True))
        if self.oracle_dir is not None:
            oracle_path = self.oracle_dir / self.paths[item].name
            if not oracle_path.exists():
                raise FileNotFoundError(f"oracle target is missing: {oracle_path}")
            with np.load(oracle_path, allow_pickle=False) as oracle:
                for key in oracle.files:
                    if key in PER_FRAME_KEYS:
                        output[key] = torch.from_numpy(
                            np.array(oracle[key][:: self.frame_stride], copy=True)
                        )
        return output


def collate_prepared_shots(shots: list[dict]) -> dict:
    """Concatenate per-frame tensors while retaining one shared grid."""

    import torch

    if not shots:
        raise ValueError("cannot collate an empty list of shots")
    output = {}
    for key in shots[0]:
        if key in PER_FRAME_KEYS:
            output[key] = torch.cat([shot[key] for shot in shots], dim=0)
        else:
            output[key] = shots[0][key]
    return output
