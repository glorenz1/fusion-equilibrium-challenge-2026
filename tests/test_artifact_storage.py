import hashlib
import json
from pathlib import Path

import pytest
import zstandard

from artifact_storage import BLOCK, pack_artifacts, unpack_artifacts, verify_package


def fixture_package(tmp_path, data=b"model bytes\x00" * 30, part_bytes=97):
    original = tmp_path / "original"
    (original / "base_fold0").mkdir(parents=True)
    (original / "base_fold0/base.joblib").write_bytes(data)
    records = [dict(path="base_fold0/base.joblib", bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest())]
    package = tmp_path / "weights"
    metadata = pack_artifacts(original, package, records, part_bytes=part_bytes, workers=1)
    return original, package, records, metadata


def save_manifest(package, metadata):
    metadata["compressed_bytes"] = sum(p["bytes"] for r in metadata["artifacts"] for p in r["parts"])
    (package / "manifest.json").write_text(json.dumps(metadata))


def test_multipart_roundtrip_restores_every_byte_and_is_independent(tmp_path):
    data = bytes(range(256)) * (BLOCK // 128) + b"tail\x00\xff\x01"
    original, package, records, metadata = fixture_package(tmp_path, data, part_bytes=BLOCK+17)
    assert len(metadata["artifacts"][0]["parts"]) > 1
    restored = tmp_path / "restored"
    assert unpack_artifacts(package, restored, records)["passed"]
    original_file = original / records[0]["path"]
    original_file.unlink()
    assert (restored / records[0]["path"]).read_bytes() == data
    assert not (restored / records[0]["path"]).is_symlink()


def test_compressed_package_is_deterministic(tmp_path):
    original, first, records, metadata = fixture_package(tmp_path)
    second = tmp_path / "weights2"
    assert pack_artifacts(original, second, records, part_bytes=97, workers=1) == metadata
    for item in metadata["artifacts"][0]["parts"]:
        assert (first / item["path"]).read_bytes() == (second / item["path"]).read_bytes()


def test_corruption_and_missing_parts_are_rejected_before_install(tmp_path):
    _, package, records, metadata = fixture_package(tmp_path)
    part = package / metadata["artifacts"][0]["parts"][0]["path"]
    data = bytearray(part.read_bytes())
    data[len(data)//2] ^= 1
    part.write_bytes(data)
    destination = tmp_path / "restored"
    with pytest.raises(ValueError, match="checksum"):
        unpack_artifacts(package, destination, records)
    assert not destination.exists()
    part.unlink()
    with pytest.raises(FileNotFoundError):
        unpack_artifacts(package, destination, records)


def test_git_lfs_pointer_has_actionable_error(tmp_path):
    _, package, records, metadata = fixture_package(tmp_path)
    (package / metadata["artifacts"][0]["parts"][0]["path"]).write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:" + "0"*64 + "\nsize 1\n")
    with pytest.raises(ValueError, match="git lfs pull"):
        verify_package(package, records)


@pytest.mark.parametrize("path", ["../escape.zst", "/tmp/escape.zst", "a/../../escape", "a\\b", "a//b"])
def test_manifest_cannot_escape_package(tmp_path, path):
    _, package, records, metadata = fixture_package(tmp_path)
    metadata["artifacts"][0]["parts"][0]["path"] = path
    save_manifest(package, metadata)
    with pytest.raises(ValueError, match="unsafe"):
        unpack_artifacts(package, tmp_path / "restored", records)


def test_tampered_model_identity_is_rejected(tmp_path):
    _, package, records, metadata = fixture_package(tmp_path)
    metadata["artifacts"][0]["sha256"] = "0"*64
    save_manifest(package, metadata)
    with pytest.raises(ValueError, match="frozen incumbent"):
        verify_package(package, records)


def test_valid_compressed_wrong_bytes_fail_original_hash_and_remove_staging(tmp_path):
    _, package, records, metadata = fixture_package(tmp_path, data=b"a"*97, part_bytes=97)
    part = metadata["artifacts"][0]["parts"][0]
    wrong = zstandard.ZstdCompressor(level=6).compress(b"b"*97)
    (package / part["path"]).write_bytes(wrong)
    part.update(bytes=len(wrong), sha256=hashlib.sha256(wrong).hexdigest())
    save_manifest(package, metadata)
    with pytest.raises(ValueError, match="restored artifact checksum"):
        unpack_artifacts(package, tmp_path / "restored", records)
    assert not (tmp_path / "restored").exists()
    assert not list(tmp_path.glob(".fec1-unpack-*"))


def test_existing_installation_is_never_overwritten(tmp_path):
    _, package, records, _ = fixture_package(tmp_path)
    destination = tmp_path / "restored"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        unpack_artifacts(package, destination, records)
    assert sentinel.read_bytes() == b"keep"


def test_expansion_beyond_declared_part_size_is_rejected(tmp_path):
    _, package, records, metadata = fixture_package(tmp_path, data=b"a"*194, part_bytes=97)
    parts = metadata["artifacts"][0]["parts"]
    parts[0]["raw_bytes"] -= 1
    parts[1]["raw_bytes"] += 1
    save_manifest(package, metadata)
    with pytest.raises(ValueError, match="expands beyond"):
        unpack_artifacts(package, tmp_path / "restored", records)
    assert not (tmp_path / "restored").exists()


def test_source_hash_failure_does_not_publish_manifest(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "bad.pt").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="original artifact checksum"):
        pack_artifacts(original, tmp_path / "weights",
                       [dict(path="bad.pt", bytes=5, sha256="0"*64)], workers=1)
    assert not (tmp_path / "weights/manifest.json").exists()
