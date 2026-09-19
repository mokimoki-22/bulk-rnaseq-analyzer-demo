"""Phase 2 UI coverage for the ATAC-only workflow."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import brim_atac


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
