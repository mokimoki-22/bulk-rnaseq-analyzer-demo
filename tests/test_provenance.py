"""Unit checks for offline, modality-independent provenance."""

from datetime import datetime
import hashlib
import io
import json
from unittest.mock import patch

import pandas as pd
import pytest

import brim_provenance as provenance
from rna_support import small_counts


@pytest.mark.parametrize("position", [0, 1, 3])
def test_checksum_reads_complete_stream_without_consuming_it(position):
    stream = io.BytesIO(b"abc")
    stream.seek(position)
    assert provenance.file_checksum(stream) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert stream.tell() == position


def test_checksum_bytes_changes_with_input():
    assert provenance.file_checksum(b"abc") != provenance.file_checksum(b"abd")


def test_checksum_rejects_text_and_restores_position():
    stream = io.StringIO("abc")
    stream.seek(1)
    with pytest.raises(TypeError, match="binary"):
        provenance.file_checksum(stream)
    assert stream.tell() == 1


def test_manifest_is_detached_json_and_markdown_preserves_every_field():
    inputs = {"rna": {"source_files": [{"file_name": "日本語.csv", "sha256": "abc"}]}}
    settings = {"app_version": "1.1.0", "rna": {"padj_threshold": 0.03}}
    services = {"external_services_used": ["mygene.info"]}
    manifest = provenance.build_manifest(inputs, settings, {"rna": {"result_genes": 4}}, services)
    assert set(manifest) == {"environment", "inputs", "settings", "counts", "services"}
    assert manifest["environment"]["app_version"] == "1.1.0"
    assert datetime.fromisoformat(manifest["environment"]["timestamp"]).utcoffset() is not None
    assert manifest["environment"]["timezone"]
    assert manifest["environment"]["packages"]["pandas"]
    for category in ("inputs", "settings", "counts"):
        assert manifest[category]["atac"] is None
    inputs["rna"]["source_files"].clear()
    services["external_services_used"].clear()
    assert manifest["inputs"]["rna"]["source_files"]
    assert manifest["services"]["external_services_used"] == ["mygene.info"]
    markdown = provenance.render_manifest_markdown(manifest)
    for category, value in manifest.items():
        assert f"## {category}" in markdown
        assert json.dumps(value, indent=2, ensure_ascii=False) in markdown
    assert json.loads(json.dumps(manifest, allow_nan=False)) == manifest


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object()])
def test_manifest_rejects_non_json_values(invalid):
    with pytest.raises(ValueError, match="settings"):
        provenance.build_manifest({}, {"invalid": invalid}, {}, {})


def test_environment_missing_optional_package_is_explicit():
    with patch.object(provenance.metadata, "version", side_effect=provenance.metadata.PackageNotFoundError):
        environment = provenance.collect_environment()
    assert all(version is None for version in environment["packages"].values())


def test_rna_description_hashes_export_bytes_and_counts_flags():
    counts, metadata = small_counts()
    raw = counts.to_csv(lineterminator="\n")
    results = pd.DataFrame({"padj_is_na": [True, False, True, False],
                            "lfc_is_na": [False, True, True, False]})
    matrix, described = provenance.describe_rna_data(raw, results, metadata)
    assert matrix["sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert described["na_counts"] == {"padj_is_na": 2, "lfc_is_na": 2}
    assert described["samples_by_condition"] == {"control": 3, "treated": 3}
    assert described["input_genes"] == 8
    assert described["input_samples"] == 6
    assert described["result_genes"] == 4


def test_legacy_missing_flags_are_unknown_not_inferred_from_padded_values():
    counts, _ = small_counts()
    _, described = provenance.describe_rna_data(counts.to_csv(), pd.DataFrame({"padj": [1.0]}), None)
    assert described["na_counts"] == {"padj_is_na": None, "lfc_is_na": None}


@pytest.mark.parametrize("flags", [[True, None], ["True", "False"], [1, 0]])
def test_invalid_flags_are_rejected(flags):
    counts, metadata = small_counts()
    with pytest.raises(ValueError, match="boolean"):
        provenance.describe_rna_data(counts.to_csv(), pd.DataFrame({"padj_is_na": flags}), metadata)


def test_provenance_has_no_streamlit_dependency_or_network_access():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(provenance.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.split(".")[0] in {"streamlit", "requests", "urllib", "socket"} for name in imports)
