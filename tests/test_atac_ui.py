"""Phase 2 UI coverage for the ATAC-only workflow."""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import brim_atac
from rna_support import capture_downloads


ROOT = Path(__file__).resolve().parents[1]


def _uploaded(text: str, name: str) -> io.BytesIO:
    file_obj = io.BytesIO(text.encode("utf-8"))
    file_obj.name = name
    return file_obj


def _install_atac_uploader(monkeypatch: pytest.MonkeyPatch, file_obj: io.BytesIO) -> None:
    original_uploader = st.file_uploader

    def upload(label, *args, **kwargs):
        if label in {"ATAC input file", "ATAC入力ファイル"}:
            return file_obj
        return original_uploader(label, *args, **kwargs)

    monkeypatch.setattr(st, "file_uploader", upload)


def _fake_dar(counts, metadata, ref_condition, test_condition, normalization, **kwargs):
    result = pd.DataFrame({
        "peak_id": counts["peak_id"], "chrom": counts["chrom"], "start": counts["start"], "end": counts["end"],
        "log2FoldChange": [1.5, -1.5], "padj": [0.01, 0.02],
        "padj_is_na": [False, False], "lfc_is_na": [False, False],
        "is_significant": [True, True], "accessibility_direction": ["opening", "closing"],
    })
    result.attrs["warnings"] = []
    result.attrs["normalization"] = normalization
    result.attrs["transforms"] = list(counts.attrs.get("transforms", []))
    return result


def test_count_matrix_atac_ui_runs_and_mode_change_invalidates_results(monkeypatch):
    uploaded = _uploaded(
        "peak,control_0,control_1,control_2,treated_0,treated_1,treated_2\n"
        "chr1:100-200,20,21,22,30,31,32\n"
        "chr1:300-400,40,41,42,50,51,52\n",
        "atac_counts.csv",
    )
    _install_atac_uploader(monkeypatch, uploaded)
    monkeypatch.setattr(brim_atac, "run_dar", _fake_dar)
    app = AppTest.from_file(str(ROOT / "Bulk_RNAseq_Analyzer.py"), default_timeout=60)
    app.session_state["lang_display"] = "English"
    app.session_state["language_selector"] = "English"
    app.run()
    assert not app.exception
    assert any("Multi-omics" in tab.label for tab in app.tabs)
    app.button(key="atac_load_counts").click().run()
    assert not app.exception
    assert app.session_state["atac_counts_df"] is not None
    app.selectbox(key="atac_normalization").set_value("deseq2_median_of_ratios").run()
    app.button(key="atac_run_dar").click().run()
    assert not app.exception
    assert app.session_state["atac_results"] is not None
    app.radio(key="atac_input_mode").set_value("dar_table").run()
    assert app.session_state["atac_results"] is None
    assert app.session_state["atac_counts_df"] is None


def test_same_name_and_size_different_atac_bytes_invalidate_all_downstream_state(monkeypatch):
    first = (
        "peak,control_0,control_1,control_2,treated_0,treated_1,treated_2\n"
        "chr1:100-200,20,21,22,30,31,32\nchr1:300-400,40,41,42,50,51,52\n"
    )
    replacement = first.replace("20,21,22", "23,21,22")
    assert len(first) == len(replacement)
    uploaded = _uploaded(first, "atac_counts.csv")
    _install_atac_uploader(monkeypatch, uploaded)
    monkeypatch.setattr(brim_atac, "run_dar", _fake_dar)
    app = AppTest.from_file(str(ROOT / "Bulk_RNAseq_Analyzer.py"), default_timeout=60)
    app.session_state["lang_display"] = "English"
    app.session_state["language_selector"] = "English"
    app.run()
    app.button(key="atac_load_counts").click().run()
    app.selectbox(key="atac_normalization").set_value("deseq2_median_of_ratios").run()
    app.button(key="atac_run_dar").click().run()
    app.session_state["integration_edge_results"] = pd.DataFrame({"old": [1]})
    uploaded.seek(0)
    uploaded.write(replacement.encode("utf-8"))
    uploaded.seek(0)
    app.run()
    assert app.session_state["atac_results"] is None
    assert app.session_state["atac_counts_df"] is None
    assert app.session_state["integration_edge_results"] is None


def test_atac_only_export_uses_shared_manifest_and_conditional_artifacts():
    result = _fake_dar(
        pd.DataFrame({"peak_id": ["chr1:100-200", "chr1:300-400"], "chrom": ["chr1", "chr1"],
                      "start": [100, 300], "end": [200, 400]}),
        None, None, None, "deseq2_median_of_ratios",
    )
    result.attrs["normalization"] = None
    app = AppTest.from_file(str(ROOT / "Bulk_RNAseq_Analyzer.py"), default_timeout=60)
    state = {
        "lang_display": "English", "language_selector": "English", "atac_results": result,
        "atac_input_provenance": {"file_name": "atac.csv", "byte_size": 42, "sha256": "a" * 64,
                                  "source_mode": "dar_table"},
        "atac_validation_report": {"source_mode": "dar_table", "coordinate_system": "0-based",
                                    "column_map": {}, "transforms": [],
                                    "thresholds": {"padj": 0.05, "log2FoldChange": 1.0}},
        "atac_peak_gene_edges": pd.DataFrame({"peak_id": ["chr1:100-200"], "mapping_method": ["promoter"]}),
        "atac_unmapped_peaks": pd.DataFrame({"peak_id": ["chr1:300-400"]}),
    }
    for key, value in state.items():
        app.session_state[key] = value
    with capture_downloads() as downloads:
        app.run()
    assert not app.exception
    assert "results.zip" in downloads and "manifest.json" in downloads
    with zipfile.ZipFile(io.BytesIO(downloads["results.zip"])) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("Provenance/manifest.json"))
    assert {"ATAC/dar_standardized.csv", "ATAC/dar_significant.csv", "ATAC/peak_gene_edges.csv",
            "ATAC/unmapped_peaks.csv", "ATAC/atac_validation.json", "Provenance/manifest.json"}.issubset(names)
    assert manifest["inputs"]["rna"] is None
    assert manifest["inputs"]["atac"]["source_file"]["sha256"] == "a" * 64
    assert manifest["settings"]["atac"]["normalization"] is None


def test_count_matrix_export_keeps_input_and_reference_artifacts():
    counts = pd.DataFrame({"peak_id": ["chr1:100-200", "chr1:300-400"], "chrom": ["chr1", "chr1"],
                           "start": [100, 300], "end": [200, 400], "control": [20, 21], "treated": [30, 31]})
    result = _fake_dar(counts, None, None, None, "deseq2_median_of_ratios")
    app = AppTest.from_file(str(ROOT / "Bulk_RNAseq_Analyzer.py"), default_timeout=60)
    state = {
        "lang_display": "English", "language_selector": "English", "atac_results": result,
        "atac_counts_df": counts, "atac_input_provenance": {"file_name": "counts.csv", "byte_size": 32,
                                                                "sha256": "b" * 64, "source_mode": "count_matrix"},
        "atac_validation_report": {"source_mode": "count_matrix", "coordinate_system": "0-based",
                                    "column_map": None, "transforms": [],
                                    "thresholds": {"padj": 0.05, "log2FoldChange": 1.0}},
        "atac_reference_metadata": {"build": "GRCh38", "release": "48", "sha256": "c" * 64},
    }
    for key, value in state.items():
        app.session_state[key] = value
    with capture_downloads() as downloads:
        app.run()
    with zipfile.ZipFile(io.BytesIO(downloads["results.zip"])) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("Provenance/manifest.json"))
    assert {"ATAC/peak_counts.csv", "Provenance/reference_manifest.json"}.issubset(names)
    assert manifest["counts"]["atac"]["input_peaks"] == 2
    assert manifest["settings"]["atac"]["reference"]["release"] == "48"


def test_dar_table_atac_ui_validates_independently_and_renders_japanese(monkeypatch):
    uploaded = _uploaded(
        "chrom,start,end,log2FoldChange,padj\nchr1,100,200,1.5,0.01\nchr1,300,400,-1.5,0.02\n",
        "atac_dar.csv",
    )
    _install_atac_uploader(monkeypatch, uploaded)
    app = AppTest.from_file(str(ROOT / "Bulk_RNAseq_Analyzer.py"), default_timeout=60)
    app.session_state["lang_display"] = "日本語"
    app.session_state["language_selector"] = "日本語"
    app.session_state["atac_input_mode"] = "dar_table"
    app.run()
    assert not app.exception
    app.button(key="atac_validate_dar").click().run()
    assert not app.exception
    result = app.session_state["atac_results"]
    assert result is not None
    assert result["accessibility_direction"].tolist() == ["opening", "closing"]
    app.button(key="atac_annotate").click().run()
    assert not app.exception
    assert app.session_state["atac_qc_summary"] is not None
    assert any("ATAC-seq解析" in header.value for header in app.header)
