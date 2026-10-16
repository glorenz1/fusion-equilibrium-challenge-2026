"""Download the official DIII-D parquet files at an immutable revision."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import re
import urllib.parse
import urllib.request

from fec1.paths import ROOT
from fec1.provenance import sha256, write_json


def get_json(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.load(response)


def download_file(task):
    url, destination, expected_size, expected_hash = task
    destination = Path(destination)
    if destination.is_file():
        if destination.stat().st_size == expected_size and sha256(destination) == expected_hash:
            return destination
        raise ValueError(f"existing file differs from the pinned release: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.part")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=300) as response, temporary.open("wb") as output:
            for block in iter(lambda: response.read(8 << 20), b""):
                digest.update(block)
                output.write(block)
        if temporary.stat().st_size != expected_size or digest.hexdigest() != expected_hash:
            raise ValueError(f"download checksum mismatch: {destination}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def download(output, splits, workers=4, revision=None):
    specification = json.loads((ROOT / "config/dataset.json").read_text())
    revision = revision or specification["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be an immutable 40-character commit SHA")
    if workers < 1:
        raise ValueError("workers must be positive")
    repo = specification["repo_id"]
    metadata = get_json(
        f"https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true"
    )
    if metadata["sha"] != revision:
        raise ValueError("server resolved a different dataset revision")
    output = Path(output).resolve()
    for split in splits:
        if split not in specification["splits"]:
            raise ValueError(f"unsupported split: {split}")
        prefix = f"data/{split}/"
        files = sorted(
            (
                p
                for p in metadata["siblings"]
                if p["rfilename"].startswith(prefix) and p["rfilename"].endswith(".parquet")
            ),
            key=lambda p: p["rfilename"],
        )
        if len(files) != specification["splits"][split]:
            raise ValueError(f"unexpected shot count for {split}: {len(files)}")
        tasks, records = [], []
        for item in files:
            relative = item["rfilename"]
            name = Path(relative).name
            checksum = item["lfs"]["sha256"]
            size = item["lfs"]["size"]
            url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(relative)}"
            tasks.append((url, output / split / name, size, checksum))
            records.append(dict(path=name, bytes=size, sha256=checksum))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for number, path in enumerate(pool.map(download_file, tasks), 1):
                if number == 1 or number % 100 == 0 or number == len(files):
                    print(f"{split}: verified {number}/{len(files)} ({path.name})", flush=True)
        write_json(
            output / split / "download.json",
            dict(
                repo_id=repo,
                revision=revision,
                split=split,
                license=specification["license"],
                files=records,
            ),
        )
