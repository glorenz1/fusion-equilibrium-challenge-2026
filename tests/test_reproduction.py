import json
from pathlib import Path

import numpy as np
import pytest

import fec1


def test_frozen_source_is_complete_and_unchanged():
    assert fec1.verify_source() >= 90


def test_incumbent_manifest_has_all_members_and_calibrations():
    m = fec1.manifest()
    assert len(m["artifacts"]) == 10
    assert [r["selected_epoch"] for r in m["folds"]] == [36, 30, 30, 36, 36]
    assert m["artifacts"][6]["sha256"] == "89b61ab3f11d38916bad2c8e149f8dd97aff93d2f698688b63916634b00c0331"
    for fold in range(5):
        report = json.loads((fec1.ROOT / f"repro/history/fold{fold}/result.json").read_text())
        assert report["correction_strengths"] == m["folds"][fold]["correction_strengths"]
        assert not report["smoke_only"] and report["nested_shot_crossfit"]
        assert not report["development_used"] and not report["frozen_audit_used"]


def test_frozen_split_roles_are_disjoint_and_oof_covers_training_once():
    from my_experiments.fit_fec1_full_frame_pca import _split_records
    training, development, audit = _split_records()
    assert (len(training), len(development), len(audit)) == (5677, 689, 675)
    outer = []
    for fold in range(5):
        m = json.loads((fec1.ROOT / f"repro/splits/fold{fold}.json").read_text())
        roles = [set(m["roles"][str(i)]) for i in range(3)]
        assert len(roles[1]) == 64
        assert not roles[0] & roles[1] and not roles[0] & roles[2] and not roles[1] & roles[2]
        assert set.union(*roles) == set(training)
        assert not set.union(*roles) & (development | audit)
        outer.extend(roles[2])
    assert len(outer) == len(set(outer)) == 5677


def test_checksum_check_rejects_corruption_and_missing_file(tmp_path):
    p = tmp_path / "test.pt"
    p.write_bytes(b"trusted")
    record = dict(path=p.name, bytes=7, sha256=fec1.digest(p))
    fec1.verify_files(tmp_path, [record])
    p.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        fec1.verify_files(tmp_path, [record])
    p.unlink()
    with pytest.raises(FileNotFoundError):
        fec1.verify_files(tmp_path, [record])


def test_archive_comparison_rejects_changed_arrays_and_nonfinite_values(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    x = np.arange(12, dtype=np.float32)
    np.savez(a, shot_0000_q95=x)
    np.savez(b, shot_0000_q95=x, shot_0001_q95=x)
    assert fec1.compare_archives(a, b)["passed"]
    np.savez(a, shot_0000_q95=x+1)
    with pytest.raises(AssertionError):
        fec1.compare_archives(a, b)
    np.savez(a, shot_0000_q95=x*np.nan)
    with pytest.raises(ValueError, match="nonfinite"):
        fec1.compare_archives(a, b)


def test_context_preserves_all_four_stride_phases():
    from my_experiments.export_fec1_dct24_public import _context
    x = np.arange(33, dtype=np.float32).reshape(11, 3)
    context = _context(x)
    np.testing.assert_array_equal(context[:, :3], x)
    for phase in range(4):
        rows = np.arange(phase, len(x), 4)
        np.testing.assert_array_equal(context[rows[0], 3:6], 0)
        np.testing.assert_array_equal(context[rows[1:], 3:6], x[rows[1:]]-x[rows[:-1]])
        np.testing.assert_allclose(context[rows, 6:], np.broadcast_to(x[rows].mean(0), (len(rows), 3)))


def test_public_reader_excludes_target_columns():
    from my_experiments.causal.cache_causal_hybrid_encoder_inputs import safe_input_columns
    names = ("source", "efit_times", "efit_grid_R", "efit_grid_Z",
             "magnetics_plasma_current", "coil_input_column",
             "efit_psirz", "efit_q95", "efit_betaN", "equilibrium")
    safe = safe_input_columns(names)
    assert not set(safe) & {"efit_psirz", "efit_q95", "efit_betaN", "equilibrium"}


@pytest.mark.parametrize("copy", [False, True])
def test_artifact_import_verifies_then_installs_without_overwrite(tmp_path, monkeypatch, copy):
    source = tmp_path / "retained"
    source.mkdir()
    original = source / "original.pt"
    original.write_bytes(b"checkpoint")
    record = dict(path="dct_fold0/checkpoint.pt", source="original.pt",
                  bytes=10, sha256=fec1.digest(original))
    monkeypatch.setattr(fec1, "manifest", lambda: {"artifacts": [record]})
    destination = tmp_path / "installed"
    fec1.import_artifacts(source, destination, copy=copy)
    installed = destination / record["path"]
    assert installed.is_symlink() is (not copy)
    assert installed.read_bytes() == b"checkpoint"
    with pytest.raises(FileExistsError):
        fec1.import_artifacts(source, destination, copy=copy)
    original.write_bytes(b"corruption")
    with pytest.raises(ValueError):
        fec1.import_artifacts(source, tmp_path / "bad", copy=copy)
    assert not (tmp_path / "bad").exists()


def test_initial_pca_helper_preserves_historical_sampling_and_fit():
    from scripts.prepare_pca import fit_pca
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(19)
    sample = rng.normal(size=(80, 5, 5)).astype(np.float32)
    selected = np.random.default_rng(260901).choice(len(sample), 40, replace=False)
    expected = PCA(n_components=4, svd_solver="randomized", iterated_power=3,
                   random_state=260902).fit(sample[selected].reshape(40, -1))
    actual = fit_pca(sample, components=4, frames=40)
    np.testing.assert_array_equal(actual.components_, expected.components_)
    np.testing.assert_array_equal(actual.mean_, expected.mean_)
