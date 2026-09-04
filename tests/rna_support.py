"""Shared inputs and UI harness for RNA compatibility tests."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "phase05_before"
LEGACY_COLUMNS = ["baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]


def small_counts():
    """Return the original baseline's eight-gene, six-sample input."""
    counts = pd.DataFrame({
        "control_1": [100, 250, 500, 70, 320, 90, 140, 200],
        "control_2": [120, 230, 470, 90, 280, 110, 130, 180],
        "control_3": [90, 260, 520, 80, 310, 100, 150, 220],
        "treated_1": [250, 120, 980, 180, 155, 220, 300, 95],
        "treated_2": [230, 110, 1020, 170, 165, 210, 280, 105],
        "treated_3": [270, 130, 960, 190, 145, 230, 320, 100],
    }, index=[f"Gene_{number}" for number in range(1, 9)])
    metadata = pd.DataFrame({"condition": ["control"] * 3 + ["treated"] * 3}, index=counts.columns)
    return counts, metadata


def result_app(result, language="English"):
    """Create an AppTest with a real, populated RNA analysis state."""
    counts, metadata = small_counts()
    app = AppTest.from_file(str(ROOT / "Bulk_RNAseq_Analyzer.py"), default_timeout=60)
    state = {
        "counts_df": counts, "qc_filtered_df": counts.copy(), "metadata": metadata,
        "conditions": ["control", "treated"], "deg_results": result.copy(),
        "last_contrast": "treated vs control", "sp": {"org": "mmu", "string_id": 10090},
        "lang_display": language, "language_selector": language,
        "lfc_t": 0.7, "padj_t": 0.1, "deg_t": (0.7, 0.1),
        "norm_method": "log1p", "filter_enable": True,
        "filter_min_count": 12, "filter_min_samples": 3,
        "analysis_log": [{"time": "12:34:56", "action": "DEG", "details": "treated vs control"}],
    }
    for key, value in state.items():
        app.session_state[key] = value
    return app


@contextmanager
def capture_downloads():
    """Capture real download payloads; isolate only static image conversion."""
    downloads = {}
    original = st.download_button

    def capture(label, data, file_name=None, mime=None, **kwargs):
        downloads[file_name] = data
        return original(label, data, file_name=file_name, mime=mime, **kwargs)

    with patch("streamlit.download_button", side_effect=capture), patch(
        "plotly.basedatatypes.BaseFigure.to_image", return_value=b"test-image"
    ):
        yield downloads
