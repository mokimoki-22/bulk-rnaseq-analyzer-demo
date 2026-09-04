"""Exercise actual DESeq2 NA production, without injecting statistical results."""

import importlib

import numpy as np
import pandas as pd
from pydeseq2 import ds

import brim_provenance


def test_real_independent_filtering_and_zero_counts_preserve_na_flags(monkeypatch):
    """Capture real engine output and isolate filtering with a no-filter control.

    The fixed seed and mixture reproduce the Phase 0.5 audit: 2,000 low-count
    genes, 1,000 expressed genes (300 up, 300 down), and five all-zero genes.
    Do not assert an exact number of independently filtered genes: numerical
    libraries may alter the selected cutoff, but filtering must actually occur.
    """
    rng = np.random.default_rng(20260904)
    means = np.concatenate([np.full(2000, 1.0), np.full(1000, 100.0), np.zeros(5)])
    mu = np.repeat(means[:, None], 6, axis=1)
    mu[2000:2300, 3:] *= 2
    mu[2300:2600, 3:] *= 0.5
    # Negative-binomial shape 5 supplies replicate variation without an engine mock.
    counts = pd.DataFrame(
        rng.negative_binomial(5, 5 / (5 + mu)),
        index=[f"g{i}" for i in range(len(means))],
        columns=[f"s{i}" for i in range(6)],
    )
    metadata = pd.DataFrame(
        {"condition": ["control"] * 3 + ["treated"] * 3}, index=counts.columns,
    )
    real_stats = ds.DeseqStats
    captured = []

    def capture_real_stats(*args, **kwargs):
        stats = real_stats(*args, **kwargs)
        captured.append(stats)
        return stats

    # Spy on construction only; fitting, testing, and filtering remain real.
    monkeypatch.setattr(ds, "DeseqStats", capture_real_stats)
    app = importlib.import_module("Bulk_RNAseq_Analyzer")
    result = app.run_deg(counts, metadata, "control", "treated", n_cpus=1)
    assert len(captured) == 1
    raw = captured[0].results_df.copy(deep=True)
    assert result.index.is_unique
    assert set(result.index) == set(counts.index)

    filtered = raw["pvalue"].notna() & raw["padj"].isna()
    assert filtered.any(), "Input must trigger real independent filtering, not only all-zero NAs"
    all_zero = counts.sum(axis=1).eq(0)
    assert all_zero.any()
    assert raw.loc[all_zero, ["padj", "log2FoldChange"]].isna().all().all()
    for flag, column in (("padj_is_na", "padj"), ("lfc_is_na", "log2FoldChange")):
        assert result[flag].dtype == bool
        expected = raw[column].isna().reindex(result.index).rename(flag)
        pd.testing.assert_series_equal(result[flag], expected)
    assert result.loc[result["padj_is_na"], "padj"].eq(1.0).all()
    assert result.loc[result["lfc_is_na"], "log2FoldChange"].eq(0.0).all()
    assert result["padj"].is_monotonic_increasing

    unfiltered = real_stats(
        captured[0].dds, contrast=["condition", "treated", "control"],
        n_cpus=1, independent_filter=False,
    )
    unfiltered.summary()
    # Same Wald p-values, different padj availability: identifies the NA cause.
    pd.testing.assert_series_equal(raw["pvalue"], unfiltered.results_df["pvalue"])
    assert unfiltered.results_df.loc[filtered, "padj"].notna().all()
    assert unfiltered.results_df.loc[all_zero, "padj"].isna().all()

    # The real NA counts also survive the shared provenance path.
    matrix, described = brim_provenance.describe_rna_data(counts.to_csv(), result, metadata)
    manifest = brim_provenance.build_manifest(
        {"rna": {"count_matrix": matrix}}, {}, {"rna": described}, {},
    )
    assert manifest["counts"]["rna"]["na_counts"] == {
        "padj_is_na": int(raw["padj"].isna().sum()),
        "lfc_is_na": int(raw["log2FoldChange"].isna().sum()),
    }
