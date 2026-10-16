"""Content identities for raw data, generated models and run records."""
from pathlib import Path
import hashlib
import json
import os


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_records(root, paths):
    root = Path(root).resolve()
    return [
        dict(
            path=str(Path(p).resolve().relative_to(root)),
            bytes=Path(p).stat().st_size,
            sha256=sha256(p),
        )
        for p in sorted(paths)
    ]


def verify_records(root, records):
    root = Path(root).resolve()
    for record in records:
        path = root / record["path"]
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"artifact escapes its root: {path}")
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"missing or changed file: {path}")


def source_identity(root):
    root = Path(root)
    paths = sorted((root / "fec1").rglob("*.py")) + sorted((root / "config").glob("*.json"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()
