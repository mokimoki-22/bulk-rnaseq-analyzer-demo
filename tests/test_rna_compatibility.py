"""Numerical compatibility with the recorded pre-Phase-0.5 application."""

import importlib

import pandas as pd

from rna_support import FIXTURES, LEGACY_COLUMNS, small_counts


def test_existing_six_deg_columns_match_recorded_output():
    """Retain every gene and the original six numerical columns."""
    counts, metadata = small_counts()
    app = importlib.import_module("Bulk_RNAseq_Analyzer")
    result = app.run_deg(counts, metadata, "control", "treated", n_cpus=1)
    expected = pd.read_csv(FIXTURES / "deg_results.csv", index_col=0)
    assert result.index.is_unique
    assert set(result.index) == set(counts.index)
    pd.testing.assert_frame_equal(result[LEGACY_COLUMNS], expected, rtol=1e-6, atol=1e-10)
    # Absolute tolerance must not hide replacement of very small p-values by 0.
    for column in ("pvalue", "padj"):
        pd.testing.assert_series_equal(result[column], expected[column], rtol=1e-6, atol=0)


def test_actual_deseq_result_includes_boolean_na_flags():
    """Verify the real statistical engine path, in addition to boundary mocks."""
    counts, metadata = small_counts()
    result = importlib.import_module("Bulk_RNAseq_Analyzer").run_deg(
        counts, metadata, "control", "treated", n_cpus=1,
    )
    for flag in ("padj_is_na", "lfc_is_na"):
        assert result[flag].dtype == bool
        assert not result[flag].any()
