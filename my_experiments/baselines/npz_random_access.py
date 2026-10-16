"""Read slices from uncompressed NumPy ``.npz`` members without full extraction.

``numpy.load(...)[key][indices]`` materializes the complete member before
applying ``indices``.  Prepared NeuralGS shot caches are deliberately written
with ``np.savez`` (ZIP_STORED), so their embedded ``.npy`` payloads can instead
be memory-mapped directly at the byte offset recorded in the ZIP local header.
"""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_SIGNATURE = 0x04034B50


@dataclass(frozen=True)
class StoredArrayInfo:
    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool
    data_offset: int


def stored_array_info(path: Path, key: str) -> StoredArrayInfo:
    """Return direct-mapping metadata for one ZIP_STORED NPY member."""

    path = Path(path)
    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f"{path}:{member} is compressed and cannot be directly mapped"
            )
        header_offset = int(info.header_offset)
    with path.open("rb") as stream:
        stream.seek(header_offset)
        raw = stream.read(LOCAL_FILE_HEADER.size)
        if len(raw) != LOCAL_FILE_HEADER.size:
            raise ValueError(f"truncated ZIP local header in {path}:{member}")
        values = LOCAL_FILE_HEADER.unpack(raw)
        if values[0] != LOCAL_FILE_SIGNATURE:
            raise ValueError(f"invalid ZIP local header in {path}:{member}")
        name_length, extra_length = values[-2:]
        stream.seek(name_length + extra_length, 1)
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            shape, fortran_order, dtype = np.lib.format._read_array_header(
                stream, version
            )
        dtype = np.dtype(dtype)
        if dtype.hasobject:
            raise ValueError(f"object array cannot be memory mapped: {path}:{member}")
        return StoredArrayInfo(
            shape=tuple(shape),
            dtype=dtype,
            fortran_order=bool(fortran_order),
            data_offset=stream.tell(),
        )


def load_stored_array(
    path: Path,
    key: str,
    indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Copy a complete member or selected leading-axis records via memmap."""

    info = stored_array_info(path, key)
    mapped = np.memmap(
        path,
        dtype=info.dtype,
        mode="r",
        offset=info.data_offset,
        shape=info.shape,
        order="F" if info.fortran_order else "C",
    )
    try:
        selected = mapped if indices is None else mapped[np.asarray(indices, dtype=np.int64)]
        return np.array(selected, copy=True)
    finally:
        del mapped
