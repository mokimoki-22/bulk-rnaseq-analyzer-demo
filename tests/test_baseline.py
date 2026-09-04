"""Regression baseline for the pre-ATAC BRIM RNA workflow."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPOSITORY_ROOT / "Bulk_RNAseq_Analyzer.py"


def test_existing_modules_importable() -> None:
    """Every module shipped in the v1.1.0 baseline imports successfully."""
    for module_name in (
        "i18n",
        "brim_enrichment",
        "brim_tf_networks",
        "Bulk_RNAseq_Analyzer",
    ):
        importlib.import_module(module_name)


def test_run_deg_returns_standard_results_dataframe() -> None:
    """A small, valid count matrix produces the expected DEG result schema."""
    app = importlib.import_module("Bulk_RNAseq_Analyzer")
    counts = pd.DataFrame(
        {
            "control_1": [100, 250, 500, 70, 320, 90, 140, 200],
            "control_2": [120, 230, 470, 90, 280, 110, 130, 180],
            "control_3": [90, 260, 520, 80, 310, 100, 150, 220],
            "treated_1": [250, 120, 980, 180, 155, 220, 300, 95],
            "treated_2": [230, 110, 1_020, 170, 165, 210, 280, 105],
            "treated_3": [270, 130, 960, 190, 145, 230, 320, 100],
        },
        index=[f"Gene_{number}" for number in range(1, 9)],
    )
    metadata = pd.DataFrame(
        {
            "condition": ["control"] * 3 + ["treated"] * 3,
        },
        index=counts.columns,
    )

    result = app.run_deg(counts, metadata, "control", "treated", n_cpus=1)

    expected_columns = {
        "baseMean",
        "log2FoldChange",
        "lfcSE",
        "stat",
        "pvalue",
        "padj",
    }
    assert isinstance(result, pd.DataFrame)
    assert result.index.tolist()
    assert expected_columns.issubset(result.columns)
    assert result.index.isin(counts.index).all()
    assert np.isfinite(result["padj"]).all()
    assert result["padj"].is_monotonic_increasing


def test_app_starts_and_renders_existing_primary_tabs() -> None:
    """The Streamlit app starts and renders every primary v1.1.0 tab."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=60)
    app.run(timeout=60)

    assert not app.exception
    labels = {tab.label for tab in app.tabs}
    for expected_section in (
        "Upload",
        "DEG",
        "Visualization",
        "Network",
        "Meta",
        "Export",
        "Info",
    ):
        assert any(expected_section in label for label in labels)
