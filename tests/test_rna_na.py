"""Verify NA provenance at the DESeq2 boundary without stochastic NA causes."""

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rna_support import LEGACY_COLUMNS, small_counts


@pytest.mark.parametrize("padj_na,lfc_na", [(True, False), (False, True), (True, True), (False, False)])
def test_run_deg_preserves_na_flags_and_legacy_values(monkeypatch, padj_na, lfc_na):
    app = importlib.import_module("Bulk_RNAseq_Analyzer")
    from pydeseq2 import dds, ds
    raw = pd.DataFrame({
        "baseMean": [80.0], "log2FoldChange": [np.nan if lfc_na else -2.5],
        "lfcSE": [0.25], "stat": [np.nan], "pvalue": [np.nan],
        "padj": [np.nan if padj_na else 0.01],
    }, index=["Gene_1"])
    untouched = raw.copy(deep=True)
    monkeypatch.setattr(dds, "DeseqDataSet", lambda **kwargs: SimpleNamespace(deseq2=lambda: None))
    monkeypatch.setattr(ds, "DeseqStats", lambda *args, **kwargs: SimpleNamespace(
        results_df=raw, summary=lambda: None))
    counts, metadata = small_counts()
    result = app.run_deg(counts, metadata, "control", "treated", n_cpus=1)
    expected = pd.DataFrame({
        "baseMean": [80.0], "log2FoldChange": [0.0 if lfc_na else -2.5],
        "lfcSE": [0.25], "stat": [0.0], "pvalue": [np.nan],
        "padj": [1.0 if padj_na else 0.01],
    }, index=["Gene_1"])
    pd.testing.assert_frame_equal(result[LEGACY_COLUMNS], expected)
    pd.testing.assert_frame_equal(raw, untouched)
    assert result["padj_is_na"].dtype == bool
    assert result["lfc_is_na"].dtype == bool
    assert bool(result.at["Gene_1", "padj_is_na"]) == padj_na
    assert bool(result.at["Gene_1", "lfc_is_na"]) == lfc_na
