"""Lossless, hash-checked model packaging; no model deserialization is needed."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import zstandard
from pathlib import Path, PurePosixPath
import shutil
import tempfile

FORMAT = "fec1-lossless-zstd-parts-v1"
BLOCK = 4 << 20
PART_BYTES = 1 << 30


def _relative(name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError(f"unsafe artifact path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError(f"unsafe artifact path: {name!r}")
    return Path(*path.parts)


def _digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(BLOCK), b""):
            result.update(data)
    return result.hexdigest()


def _pack_one(record, models, destination, part_bytes, level):
    relative = _relative(record["path"])
    source = models / relative
    if source.stat().st_size != record["bytes"]:
        raise ValueError(f"original artifact size mismatch: {source}")
    digest = hashlib.sha256()
    parts, total = [], 0
    with source.open("rb") as stream:
        while total < record["bytes"]:
            part_name = relative.as_posix() + f".part{len(parts):03d}.zst"
            part = destination / part_name
            part.parent.mkdir(parents=True, exist_ok=True)
            temporary = part.with_suffix(".zst.tmp")
            raw_bytes = 0
            with temporary.open("wb") as raw, zstandard.ZstdCompressor(
                    level=level, write_checksum=True).stream_writer(raw, closefd=False) as compressed:
                while raw_bytes < min(part_bytes, record["bytes"] - total):
                    data = stream.read(min(BLOCK, part_bytes - raw_bytes,
                                           record["bytes"] - total - raw_bytes))
                    if not data:
                        raise ValueError(f"original artifact truncated: {source}")
                    digest.update(data)
                    compressed.write(data)
                    raw_bytes += len(data)
            temporary.replace(part)
            parts.append(dict(path=part_name, raw_bytes=raw_bytes,
                              bytes=part.stat().st_size, sha256=_digest(part)))
            total += raw_bytes
            print(json.dumps(dict(stage="compressed_part", artifact=record["path"],
                                  part=len(parts), raw_bytes=raw_bytes,
                                  compressed_bytes=parts[-1]["bytes"])), flush=True)
        if stream.read(1):
            raise ValueError(f"original artifact grew while packing: {source}")
    if digest.hexdigest() != record["sha256"]:
        raise ValueError(f"original artifact checksum mismatch: {source}")
    return dict(path=record["path"], bytes=total, sha256=digest.hexdigest(), parts=parts)


def pack_artifacts(models, destination, records, *, workers=5,
                   part_bytes=PART_BYTES, level=6):
    if workers < 1 or not 1 <= part_bytes <= PART_BYTES or not 1 <= level <= 22:
        raise ValueError("invalid compression resource/format parameters")
    if len({r["path"] for r in records}) != len(records) or not records:
        raise ValueError("artifact identities must be unique and nonempty")
    for record in records:
        _relative(record["path"])
    destination, models = Path(destination), Path(models)
    destination.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        artifacts = list(pool.map(lambda record: _pack_one(
            record, models, destination, part_bytes, level), records))
    result = dict(format=FORMAT, codec="zstd", level=level, part_bytes=part_bytes,
                  raw_bytes=sum(r["bytes"] for r in artifacts),
                  compressed_bytes=sum(p["bytes"] for r in artifacts for p in r["parts"]),
                  artifacts=artifacts)
    # A failed pack never publishes a manifest that could be installed.
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _validate_package(package, expected):
    if package.get("format") != FORMAT or package.get("codec") != "zstd":
        raise ValueError("unsupported compressed artifact format")
    records = package["artifacts"]
    expected = {r["path"]: (r["bytes"], r["sha256"]) for r in expected}
    actual = {r["path"]: (r["bytes"], r["sha256"]) for r in records}
    if actual != expected or len(actual) != len(records):
        raise ValueError("compressed package does not match the frozen incumbent")
    if (package["raw_bytes"] != sum(r["bytes"] for r in records)
            or package["compressed_bytes"] != sum(p["bytes"] for r in records for p in r["parts"])):
        raise ValueError("compressed package size totals are inconsistent")
    seen = set()
    for record in records:
        _relative(record["path"])
        if not record["parts"] or sum(p["raw_bytes"] for p in record["parts"]) != record["bytes"]:
            raise ValueError("compressed part sizes do not match the original artifact")
        for part in record["parts"]:
            _relative(part["path"])
            if part["path"] in seen or not 0 < part["raw_bytes"] <= PART_BYTES or part["bytes"] <= 0:
                raise ValueError("invalid or duplicate compressed part")
            seen.add(part["path"])
    return records


def verify_package(source, expected):
    source = Path(source)
    package = json.loads((source / "manifest.json").read_text())
    records = _validate_package(package, expected)
    for record in records:
        for part in record["parts"]:
            path = source / part["path"]
            with path.open("rb") as stream:
                if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ValueError("model files are Git LFS pointers; run git lfs pull first")
            if path.stat().st_size != part["bytes"] or _digest(path) != part["sha256"]:
                raise ValueError(f"compressed part checksum mismatch: {path}")
    return package


def unpack_artifacts(source, destination, expected):
    """Stage and verify every original byte before exposing an installation."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"use a fresh artifact directory: {destination}")
    package = verify_package(source, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fec1-unpack-", dir=destination.parent))
    try:
        for record in package["artifacts"]:
            path = staging / _relative(record["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            digest, total = hashlib.sha256(), 0
            with path.open("wb") as target:
                for part in record["parts"]:
                    count = 0
                    with (source / part["path"]).open("rb") as raw, \
                            zstandard.ZstdDecompressor().stream_reader(raw) as compressed:
                        while True:
                            data = compressed.read(min(BLOCK, part["raw_bytes"] - count + 1))
                            if not data:
                                break
                            count += len(data)
                            if count > part["raw_bytes"]:
                                raise ValueError("compressed part expands beyond its declared size")
                            target.write(data)
                            digest.update(data)
                    if count != part["raw_bytes"]:
                        raise ValueError("compressed part is truncated")
                    total += count
            if total != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise ValueError(f"restored artifact checksum mismatch: {record['path']}")
            print(json.dumps(dict(stage="restored_artifact", artifact=record["path"],
                                  bytes=total, sha256=digest.hexdigest())), flush=True)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return dict(passed=True, artifacts=len(package["artifacts"]),
                raw_bytes=package["raw_bytes"], compressed_bytes=package["compressed_bytes"])
