"""Phase 1 tests for the Streamlit-free ATAC core."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import brim_atac


ROOT = Path(__file__).resolve().parents[1]


def _counts_and_metadata(n_per_group: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = [f"control_{i}" for i in range(n_per_group)] + [f"treated_{i}" for i in range(n_per_group)]
    counts = pd.DataFrame({
        "peak_id": ["chr1:100-200", "chr1:300-400"],
        "chrom": ["chr1", "chr1"], "start": [100, 300], "end": [200, 400],
        **{sample: [20 + i, 40 + i] for i, sample in enumerate(samples)},
    })
    metadata = pd.DataFrame(
        {"condition": ["control"] * n_per_group + ["treated"] * n_per_group},
        index=samples,
    )
    return counts, metadata


def _install_fake_deseq(monkeypatch, raw: pd.DataFrame):
    from pydeseq2 import dds, ds

    created = []

    class FakeDDS:
        def __init__(self, counts, metadata, **kwargs):
            self.counts = counts
            self.X = counts.to_numpy(dtype=float)
            self.obs_names = counts.index
            self.obs = pd.DataFrame(index=self.obs_names)
            self.var = pd.DataFrame(index=counts.columns)
            self.layers = {}
            self.refit_cooks = False
            self.calls = []
            created.append(self)

        def deseq2(self):
            self.calls.append("deseq2")
            self.obs["size_factors"] = 1.0

        def __getattr__(self, name):
            if name.startswith("fit_") or name in {"calculate_cooks", "refit", "cooks_outlier"}:
                def call():
                    self.calls.append(name)
                return call
            raise AttributeError(name)

    class FakeStats:
        def __init__(self, fitted, **kwargs):
            self.results_df = raw.reindex(fitted.counts.columns)

        def summary(self):
            return None

    monkeypatch.setattr(dds, "DeseqDataSet", FakeDDS)
    monkeypatch.setattr(ds, "DeseqStats", FakeStats)
    return created


def test_module_is_streamlit_free_and_thresholds_are_explicit_arguments():
    source = (ROOT / "brim_atac.py").read_text(encoding="utf-8")
    assert "import streamlit" not in source
    assert "st.session_state" not in source
    for function in (brim_atac.run_dar, brim_atac.read_dar_table, brim_atac.validate_dar_table):
        signature = inspect.signature(function)
        assert "padj_threshold" in signature.parameters
        assert "lfc_threshold" in signature.parameters


def test_peak_count_matrix_index_and_columns_converge():
    index_csv = "peak,one,two\nchr1:100-200,3,4\nchr2:300-350,5,6\n"
    columns_csv = "chrom,start,end,one,two\nchr1,100,200,3,4\nchr2,300,350,5,6\n"
    by_index = brim_atac.read_peak_count_matrix(io.StringIO(index_csv), ",", "index", "0-based")
    by_columns = brim_atac.read_peak_count_matrix(io.StringIO(columns_csv), ",", "columns", "0-based")
    pd.testing.assert_frame_equal(by_index, by_columns)


def test_one_based_count_coordinates_are_explicitly_converted_and_logged():
    raw = "chrom,start,end,s1\nchr1,101,200,4\n"
    result = brim_atac.read_peak_count_matrix(io.StringIO(raw), ",", "columns", "1-based")
    assert result.loc[0, ["start", "end", "peak_id"]].tolist() == [100, 200, "chr1:100-200"]
    assert result.attrs["transforms"][0]["name"] == "one_based_closed_to_zero_based_half_open"
    assert result.attrs["transforms"][0]["success_rate"] == 1.0
    with pytest.raises(brim_atac.CoordinateSystemError, match="start >= 1"):
        brim_atac.read_peak_count_matrix(
            io.StringIO("chrom,start,end,s1\nchr1,0,10,4\n"), ",", "columns", "1-based",
        )


@pytest.mark.parametrize(
    "text,message",
    [
        ("chrom,start,end,s1\nchr1,200,100,1\n", "end > start"),
        ("chrom,start,end,s1\nchr1,-1,100,1\n", "start >= 0"),
        ("chrom,start,end,s1\nchr1,x,100,1\n", "start must contain integers"),
        ("chrom,start,end,s1\nchr1,1,100,1.2\n", "Counts must be non-negative integers"),
    ],
)
def test_peak_count_matrix_rejects_invalid_coordinates_and_counts(text, message):
    with pytest.raises(brim_atac.ATACError, match=message):
        brim_atac.read_peak_count_matrix(io.StringIO(text), ",", "columns", "0-based")


def test_dar_schema_preserves_na_and_classifies_with_explicit_thresholds():
    raw = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"], "start": [100, 300, 500], "end": [200, 400, 600],
        "log2FoldChange": [1.5, -1.2, 2.0], "padj": [0.01, 0.2, np.nan],
    })
    validation = brim_atac.validate_dar_table(raw, "0-based", 0.05, 1.0)
    assert validation.valid
    result = validation.data
    assert result["padj_is_na"].tolist() == [False, False, True]
    assert result["lfc_is_na"].tolist() == [False, False, False]
    assert result["padj"].tolist() == [0.01, 0.2, 1.0]
    assert result["accessibility_direction"].tolist() == ["opening", "not_significant", "not_tested"]


@pytest.mark.parametrize(
    "column,value",
    [
        ("start", -1), ("end", 50), ("start", "bad"), ("padj", 1.1),
        ("log2FoldChange", np.nan), ("log2FoldChange", np.inf),
        ("log2FoldChange", "bad"),
    ],
)
def test_dar_schema_rejects_invalid_values(column, value):
    raw = pd.DataFrame({
        "chrom": ["chr1"], "start": [100], "end": [200],
        "log2FoldChange": [1.2], "padj": [0.02],
    })
    if isinstance(value, str):
        raw[column] = raw[column].astype(object)
    raw.loc[0, column] = value
    validation = brim_atac.validate_dar_table(raw, "0-based", 0.05, 1.0)
    assert not validation.valid


def test_dar_schema_rejects_text_in_na_permitted_numeric_column():
    raw = pd.DataFrame({
        "chrom": ["chr1"], "start": [100], "end": [200],
        "log2FoldChange": [1.2], "padj": ["not-a-number"],
    })
    validation = brim_atac.validate_dar_table(raw, "0-based", 0.05, 1.0)
    assert not validation.valid
    assert "padj must be numeric when provided." in validation.errors


def test_read_dar_aliases_and_sample_dar_table():
    aliases = "chr,chromStart,stop,log2FC,FDR\nchr1,100,200,1.5,0.01\n"
    result = brim_atac.read_dar_table(io.StringIO(aliases), ",", 0.05, 1.0, "0-based")
    assert result.loc[0, "peak_id"] == "chr1:100-200"
    sample = brim_atac.read_dar_table(
        ROOT / "sample_data" / "BRIM_ATAC_DAR.csv", ",", 0.05, 1.0, "0-based",
    )
    assert len(sample) == 5
    assert sample["padj_is_na"].sum() == 1


def test_run_dar_retains_engine_na_flags_and_prefilter_provenance(monkeypatch):
    counts, metadata = _counts_and_metadata(2)
    raw = pd.DataFrame({
        "baseMean": [30.0, 40.0], "log2FoldChange": [np.nan, 1.5], "pvalue": [np.nan, 0.01],
        "padj": [np.nan, 0.02], "stat": [np.nan, 2.1],
    }, index=counts["peak_id"])
    created = _install_fake_deseq(monkeypatch, raw)
    result = brim_atac.run_dar(
        counts, metadata, "control", "treated", "deseq2_median_of_ratios", 1,
        0.05, 1.0, prefilter_enabled=True, prefilter_total_count=10,
    )
    assert created[0].calls == ["deseq2"]
    first = result.set_index("peak_id").loc["chr1:100-200"]
    assert first["padj_is_na"] and first["lfc_is_na"]
    assert first["padj"] == 1.0 and first["log2FoldChange"] == 0.0
    assert first["accessibility_direction"] == "not_tested"
    assert result.attrs["warnings"]
    assert result.attrs["sample_size_warning"] is True
    assert result.attrs["minimum_group_size"] == 2
    assert result.attrs["prefilter"] == {
        "enabled": True, "total_count_threshold": 10, "input_peaks": 2,
        "excluded_peaks": 0, "analyzed_peaks": 2,
    }
    dar_mode = brim_atac.validate_dar_table(
        pd.DataFrame({
            "chrom": ["chr1"], "start": [100], "end": [200],
            "log2FoldChange": [1.5], "padj": [0.02],
        }),
        "0-based", 0.05, 1.0,
    ).data
    assert result.columns.tolist() == dar_mode.columns.tolist()


def test_run_dar_distinguishes_all_engine_na_combinations(monkeypatch):
    counts, metadata = _counts_and_metadata(3)
    extra = counts.copy()
    extra["peak_id"] = ["chr1:100-200", "chr1:300-400"]
    extra2 = extra.copy()
    extra2["start"] += 400
    extra2["end"] += 400
    extra2["peak_id"] = ["chr1:500-600", "chr1:700-800"]
    counts = pd.concat([extra, extra2], ignore_index=True)
    raw = pd.DataFrame({
        "baseMean": [20.0] * 4,
        "log2FoldChange": [1.1, np.nan, 1.1, np.nan],
        "pvalue": [0.01, 0.01, np.nan, np.nan],
        "padj": [0.02, 0.02, np.nan, np.nan],
        "stat": [2.0, np.nan, 2.0, np.nan],
    }, index=counts["peak_id"])
    _install_fake_deseq(monkeypatch, raw)
    result = brim_atac.run_dar(
        counts, metadata, "control", "treated", "deseq2_median_of_ratios", 1, 0.05, 1.0,
    ).set_index("peak_id").loc[counts["peak_id"]]
    assert list(zip(result["padj_is_na"], result["lfc_is_na"])) == [
        (False, False), (False, True), (True, False), (True, True),
    ]


def test_prefilter_records_exclusion_and_preserves_original_input_row(monkeypatch):
    counts, metadata = _counts_and_metadata(3)
    counts.loc[0, metadata.index] = 1
    raw = pd.DataFrame({
        "baseMean": [40.0], "log2FoldChange": [0.3], "pvalue": [0.5],
        "padj": [0.6], "stat": [0.2],
    }, index=["chr1:300-400"])
    _install_fake_deseq(monkeypatch, raw)
    result = brim_atac.run_dar(
        counts, metadata, "control", "treated", "deseq2_median_of_ratios", 1,
        0.05, 1.0, prefilter_enabled=True, prefilter_total_count=10,
    )
    assert result["peak_id"].tolist() == ["chr1:300-400"]
    assert result["input_row"].tolist() == [1]
    assert result.attrs["prefilter"]["excluded_peaks"] == 1


@pytest.mark.parametrize("normalization", ["total_reads_in_peaks", "user_supplied_size_factors"])
def test_run_dar_applies_explicit_nondefault_size_factors(monkeypatch, normalization):
    counts, metadata = _counts_and_metadata(3)
    raw = pd.DataFrame({
        "baseMean": [30.0, 40.0], "log2FoldChange": [0.2, -0.3], "pvalue": [0.8, 0.7],
        "padj": [0.9, 0.9], "stat": [0.1, -0.1],
    }, index=counts["peak_id"])
    created = _install_fake_deseq(monkeypatch, raw)
    supplied = ({sample: index + 1 for index, sample in enumerate(metadata.index)}
                if normalization == "user_supplied_size_factors" else None)
    result = brim_atac.run_dar(
        counts, metadata, "control", "treated", normalization, 1, 0.05, 1.0,
        size_factors=supplied,
    )
    assert "deseq2" not in created[0].calls
    assert created[0].calls == [
        "fit_genewise_dispersions", "fit_dispersion_trend", "fit_dispersion_prior",
        "fit_MAP_dispersions", "fit_LFC", "calculate_cooks", "cooks_outlier",
    ]
    assert np.isclose(np.exp(np.log(list(result.attrs["size_factors"].values())).mean()), 1.0)
    assert result.attrs["normalization"] == normalization


def test_run_dar_rejects_n1_before_engine_execution():
    counts, metadata = _counts_and_metadata(1)
    with pytest.raises(brim_atac.InsufficientSampleError):
        brim_atac.run_dar(
            counts, metadata, "control", "treated", "deseq2_median_of_ratios", 1, 0.05, 1.0,
        )


@pytest.mark.parametrize(
    "normalization",
    ["deseq2_median_of_ratios", "total_reads_in_peaks", "user_supplied_size_factors"],
)
def test_real_run_dar_estimates_results_and_retains_all_zero_na(normalization):
    rng = np.random.default_rng(20260906)
    means = np.concatenate([np.full(120, 40.0), np.zeros(1)])
    mu = np.repeat(means[:, None], 6, axis=1)
    mu[:30, 3:] *= 2.0
    values = rng.negative_binomial(8, 8 / (8 + mu))
    samples = [f"s{i}" for i in range(6)]
    counts = pd.DataFrame(values, columns=samples)
    counts.insert(0, "end", np.arange(len(counts)) * 200 + 150)
    counts.insert(0, "start", np.arange(len(counts)) * 200 + 100)
    counts.insert(0, "chrom", "chr1")
    counts.insert(0, "peak_id", counts["chrom"] + ":" + counts["start"].astype(str) + "-" + counts["end"].astype(str))
    metadata = pd.DataFrame({"condition": ["control"] * 3 + ["treated"] * 3}, index=samples)
    supplied = ({sample: 1.0 for sample in samples}
                if normalization == "user_supplied_size_factors" else None)
    result = brim_atac.run_dar(
        counts, metadata, "control", "treated", normalization, 1, 0.05, 1.0,
        size_factors=supplied,
    )
    assert len(result) == len(counts)
    all_zero_id = counts.iloc[-1]["peak_id"]
    all_zero = result.set_index("peak_id").loc[all_zero_id]
    assert all_zero["padj_is_na"] and all_zero["lfc_is_na"]
    assert all_zero["padj"] == 1.0 and all_zero["log2FoldChange"] == 0.0
    changed_ids = counts.iloc[:30]["peak_id"]
    assert result.set_index("peak_id").loc[changed_ids, "log2FoldChange"].median() > 0.5


def _genes() -> pd.DataFrame:
    return pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr1"],
        "tss": [1000, 2000, 3000, 3000], "strand": ["+", "-", "+", "-"],
        "gene_id": ["g_plus", "g_minus", "g_tie_1", "g_tie_2"],
        "gene_symbol": ["PLUS", "MINUS", "TIE1", "TIE2"],
        "gene_type": ["protein_coding"] * 4,
        "reference_build": ["GRCh38"] * 4, "reference_release": ["48"] * 4,
    })


def test_promoter_mapping_is_strand_aware_and_preserves_one_to_many_edges():
    peaks = pd.DataFrame({
        "peak_id": ["plus", "minus", "both"], "chrom": ["chr1"] * 3,
        "start": [810, 2160, 2950], "end": [830, 2180, 3050],
    })
    edges = brim_atac.map_peaks_to_promoters(peaks, _genes(), upstream=200, downstream=50)
    pairs = set(zip(edges["peak_id"], edges["gene_id"]))
    assert ("plus", "g_plus") in pairs
    assert ("minus", "g_minus") in pairs
    assert {gene for peak, gene in pairs if peak == "both"} == {"g_tie_1", "g_tie_2"}
    assert edges["reference_release"].eq("48").all()


def test_nearest_tss_retains_all_ties_and_respects_maximum_distance():
    genes = pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"], "tss": [100, 199, 199], "strand": ["+", "+", "-"],
        "gene_id": ["left", "right1", "right2"], "gene_symbol": ["L", "R1", "R2"],
    })
    peak = pd.DataFrame({"peak_id": ["p"], "chrom": ["chr1"], "start": [140], "end": [160]})
    tied = brim_atac.map_peaks_to_nearest_tss(peak, genes, max_distance=40)
    assert set(tied["gene_id"]) == {"left", "right1", "right2"}
    assert brim_atac.map_peaks_to_nearest_tss(peak, genes, max_distance=39).empty


def test_merge_evidence_only_collapses_same_peak_gene_pair():
    peaks = pd.DataFrame({"peak_id": ["p"], "chrom": ["chr1"], "start": [140], "end": [160]})
    genes = pd.DataFrame({
        "chrom": ["chr1", "chr1"], "tss": [150, 150], "strand": ["+", "-"],
        "gene_id": ["g1", "g2"], "gene_symbol": ["G1", "G2"],
    })
    promoter = brim_atac.map_peaks_to_promoters(peaks, genes, 100, 100)
    nearest = brim_atac.map_peaks_to_nearest_tss(peaks, genes, 100)
    merged = brim_atac.merge_peak_gene_evidence(promoter, nearest)
    assert len(merged) == 2
    assert all(methods == ["promoter", "nearest_tss"] for methods in merged["mapping_methods"])
    assert all(types == ["promoter_overlap", "distance_based_candidate"]
               for types in merged["mapping_evidence_types"])


def test_reference_manifest_and_both_pinned_builds_are_loadable():
    manifest_path = ROOT / "references" / "genome_annotations" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"]["hg38"]["release"] == "48"
    assert manifest["files"]["mm10"]["release"] == "M25"
    for build, expected_build in (("hg38", "GRCh38"), ("mm10", "GRCm38")):
        record = manifest["files"][build]
        assert record["source"] == "GENCODE"
        assert record["download_url"].startswith("https://ftp.ebi.ac.uk/")
        path = manifest_path.parent / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        script = ROOT / record["generation_script"]
        assert hashlib.sha256(script.read_bytes()).hexdigest() == record["generation_script_sha256"]
        annotation = brim_atac.load_gene_annotation(build)
        assert not annotation.empty
        assert annotation["reference_build"].eq(expected_build).all()


def test_generation_script_checksum_is_checkout_stable():
    """Keep the manifest's byte-level script checksum reproducible on Windows."""
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "scripts/build_gene_annotations.py text eol=lf" in attributes


def test_sample_peak_count_matrix_and_metadata_are_usable():
    counts = brim_atac.read_peak_count_matrix(
        ROOT / "sample_data" / "BRIM_ATAC_peak_counts.csv", ",", "index", "0-based",
    )
    metadata = pd.read_csv(ROOT / "sample_data" / "BRIM_ATAC_metadata.csv", index_col="sample")
    assert counts.shape == (120, 10)
    assert list(metadata["condition"].value_counts().sort_index()) == [3, 3]


def test_sample_peak_count_matrix_runs_through_real_dar_engine():
    counts = brim_atac.read_peak_count_matrix(
        ROOT / "sample_data" / "BRIM_ATAC_peak_counts.csv", ",", "index", "0-based",
    )
    metadata = pd.read_csv(ROOT / "sample_data" / "BRIM_ATAC_metadata.csv", index_col="sample")
    result = brim_atac.run_dar(
        counts, metadata, "Control", "Treated", "deseq2_median_of_ratios", 1, 0.05, 1.0,
    )
    assert len(result) == len(counts)
    assert set(result["peak_id"]) == set(counts["peak_id"])
    ordered = result.set_index("peak_id").loc[counts["peak_id"]]
    assert ordered.iloc[:30]["log2FoldChange"].median() > 0.5
    assert ordered.iloc[30:60]["log2FoldChange"].median() < -0.5


def test_bed_export_excludes_not_tested_peaks_and_uses_explicit_thresholds():
    dar = pd.DataFrame({
        "peak_id": ["open", "closed", "na"], "chrom": ["chr1"] * 3,
        "start": [10, 30, 50], "end": [20, 40, 60],
        "log2FoldChange": [2.0, -2.0, 3.0], "padj": [0.01, 0.02, 1.0],
        "padj_is_na": [False, False, True], "lfc_is_na": [False, False, False],
    })
    thresholds = {"padj": 0.05, "log2FoldChange": 1.0}
    assert "open" in brim_atac.export_peaks_as_bed(dar, "opening", thresholds)
    assert "closed" in brim_atac.export_peaks_as_bed(dar, "closing", thresholds)
    assert "na" in brim_atac.export_peaks_as_bed(dar, "all", thresholds)


def test_chromosome_conversion_updates_peak_id_and_records_original():
    peaks = pd.DataFrame({
        "peak_id": ["1:10-20", "MT:30-40"], "chrom": ["1", "MT"],
        "start": [10, 30], "end": [20, 40],
    })
    converted, log = brim_atac.standardize_chromosomes(peaks, "GRCh38")
    assert converted["chrom"].tolist() == ["chr1", "chrM"]
    assert converted["peak_id"].tolist() == ["chr1:10-20", "chrM:30-40"]
    assert converted["original_peak_id"].tolist() == ["1:10-20", "MT:30-40"]
    assert log.affected_rows == 2 and log.success_rate == 1.0
