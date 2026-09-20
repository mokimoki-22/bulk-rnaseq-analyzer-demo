"""Phase 3 Level 1 integration contracts and negative controls."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import importlib

import brim_multiomics as multi


T = {"rna_padj": .05, "rna_lfc": 1., "atac_padj": .05, "atac_lfc": 1.}


def rna(rows):
    return multi.standardize_rna_results(pd.DataFrame(rows, columns=["gene", "log2FoldChange", "padj", "padj_is_na", "lfc_is_na"]).set_index("gene"), "gene_symbol")


def edges(rows):
    result = pd.DataFrame(rows, columns=["peak_id", "gene_symbol", "atac_log2FoldChange", "atac_padj", "atac_padj_is_na", "atac_lfc_is_na"])
    return result.assign(gene_id=result["gene_symbol"], edge_id=[f"e{i}" for i in range(len(result))], mapping_method="promoter", distance_to_tss=0)


def classified(r, e):
    return multi.classify_integration_edges(multi.integrate_peak_gene_edges(r, e, T), T)


def test_all_edge_classes_and_na_flags_are_preserved():
    r = rna([("ca", 2, .01, 0, 0), ("cr", -2, .01, 0, 0), ("do", -2, .01, 0, 0), ("dc", 2, .01, 0, 0),
             ("ao", 0, .5, 0, 0), ("ro", 2, .01, 0, 0), ("rn", 0, 1, 1, 0), ("an", 2, .01, 0, 0),
             ("bn", 0, 1, 1, 0), ("ns", 0, .5, 0, 0)])
    e = edges([("p1", "ca", 2, .01, 0, 0), ("p2", "cr", -2, .01, 0, 0), ("p3", "do", 2, .01, 0, 0),
               ("p4", "dc", -2, .01, 0, 0), ("p5", "ao", 2, .01, 0, 0), ("p6", "ro", .2, .5, 0, 0),
               ("p7", "rn", 2, .01, 0, 0), ("p8", "an", 0, 1, 1, 0), ("p9", "bn", 0, 1, 1, 1),
               ("p10", "ns", .2, .5, 0, 0)])
    got = dict(zip(classified(r, e).gene_key, classified(r, e).integration_class))
    assert got == {"ca": "concordant_activation", "cr": "concordant_repression", "do": "discordant_open_down", "dc": "discordant_closed_up", "ao": "atac_only", "ro": "rna_only_on_mapped_peak", "rn": "rna_not_tested", "an": "atac_not_tested", "bn": "both_not_tested", "ns": "not_significant"}
    assert {"rna_padj_is_na", "rna_lfc_is_na", "atac_padj_is_na", "atac_lfc_is_na"}.issubset(classified(r, e))


def test_each_lfc_na_flag_is_not_tested_and_zero_effect_is_not_directional():
    r = rna([("rna_lfc", 0, 1, 0, 1), ("atac_lfc", 2, .01, 0, 0), ("zero", 0, .01, 0, 0), ("zero_unmapped", 0, .01, 0, 0)])
    e = edges([("p1", "rna_lfc", 2, .01, 0, 0), ("p2", "atac_lfc", 0, 1, 0, 1), ("p3", "zero", 0, .01, 0, 0)])
    result = classified(r, e).set_index("gene_key")
    assert result.loc["rna_lfc", "integration_class"] == "rna_not_tested"
    assert result.loc["atac_lfc", "integration_class"] == "atac_not_tested"
    zero_thresholds = {**T, "rna_lfc": 0., "atac_lfc": 0.}
    zero = multi.classify_integration_edges(multi.integrate_peak_gene_edges(r, e, zero_thresholds), zero_thresholds).set_index("gene_key")
    assert not zero.loc["zero", "rna_significant"] and not zero.loc["zero", "atac_significant"]
    assert multi.summarize_integration_by_gene(zero.reset_index(), zero_thresholds).set_index("gene_key").loc["zero_unmapped", "integration_class"] == "not_significant"
    with pytest.raises(multi.IntegrationError):
        multi.classify_integration_edges(zero.reset_index(), {**zero_thresholds, "rna_lfc": float("inf")})


def test_boundary_mixed_unique_counts_and_rna_only_summary():
    r = rna([("G", 1, .05, 0, 0), ("RNA_ONLY", 2, .01, 0, 0)])
    e = edges([("open", "G", 1, .05, 0, 0), ("close", "G", -2, .01, 0, 0), ("na", "G", 0, 1, 1, 0)])
    result = classified(r, e)
    assert set(result.integration_class) == {"concordant_activation", "discordant_closed_up", "atac_not_tested"}
    summary = multi.summarize_integration_by_gene(result, T).set_index("gene_key")
    assert summary.loc["G", "integration_class"] == "mixed_accessibility"
    assert summary.loc["G", "n_mapped_peaks"] == 3 and summary.loc["G", "n_atac_not_tested_peaks"] == 1
    assert summary.loc["G", "source_edge_ids"] == ["e0", "e1", "e2"]
    assert summary.loc["RNA_ONLY", "integration_class"] == "rna_only_no_mapped_peak"


def test_ensembl_transform_duplicates_and_compatibility_contract():
    frame = pd.DataFrame({"gene_id": ["ENSG000001.7"], "log2FoldChange": [2.], "padj": [.01], "padj_is_na": [False], "lfc_is_na": [False]})
    normalized = multi.standardize_rna_results(frame, "gene_id")
    assert normalized.loc[0, "gene_key"] == "ENSG000001" and normalized.attrs["transforms"][0]["success_rate"] == 1.0
    with pytest.raises(multi.GeneIdentifierError):
        multi.standardize_rna_results(pd.concat([frame, frame.assign(gene_id="ENSG000001")]), "gene_id")
    base = {"species": "Human", "genome_build": "not_applicable", "contrast": {"reference": "C", "test": "T"}, "gene_keys": ["A", "B", "C", "D", "E"]}
    atac = {"species": "human", "genome_build": "hg38", "contrast": {"reference": "C", "test": "T"}, "gene_keys": ["A"]}
    compatible = multi.check_integration_compatibility(base, atac)
    assert compatible.compatible and compatible.warnings and compatible.counts["n_shared_genes"] == 1
    assert not multi.check_integration_compatibility(base, {**atac, "gene_keys": ["Z"]}).compatible
    assert not multi.check_integration_compatibility(base, {**atac, "contrast": {"reference": "T", "test": "C"}}).compatible
    for missing in (None, {}, {"reference": "C"}, {"reference": "C", "test": ""}, "T vs C"):
        assert not multi.check_integration_compatibility(base, {**atac, "contrast": missing}).compatible
        assert not multi.check_integration_compatibility({**base, "contrast": missing}, atac).compatible
    assert not multi.check_integration_compatibility(base, {k: v for k, v in atac.items() if k != "contrast"}).compatible


def _signal():
    genes = [f"G{i}" for i in range(12)]
    signs = np.array([1, -1] * 6)
    return (rna([(g, float(2 * s), .01, 0, 0) for g, s in zip(genes, signs)]),
            edges([(f"p{i}", g, float(2 * s), .01, 0, 0) for i, (g, s) in enumerate(zip(genes, signs))]))


def _concordant(r, e):
    return int(classified(r, e).integration_class.isin(["concordant_activation", "concordant_repression"]).sum())


def test_fixed_seed_edge_and_rna_label_shuffles_reduce_concordance():
    r, e = _signal()
    shuffled_edges = e.copy()
    shuffled_edges["gene_symbol"] = np.random.default_rng(11).permutation(shuffled_edges.gene_symbol.to_numpy())
    shuffled_edges["gene_id"] = shuffled_edges["gene_symbol"]
    counts = pd.DataFrame({"c1": [4, 40, 4, 40], "c2": [4, 40, 4, 40], "t1": [40, 4, 40, 4], "t2": [40, 4, 40, 4]}, index=["G0", "G1", "G2", "G3"])
    metadata = pd.Series(["control", "control", "treated", "treated"], index=counts.columns)
    def results_from_labels(labels):
        control = counts.loc[:, labels[labels == "control"].index].mean(axis=1)
        treated = counts.loc[:, labels[labels == "treated"].index].mean(axis=1)
        return rna([(gene, float(np.log2(treated[gene] / control[gene])), .01, 0, 0) for gene in counts.index])
    shuffled_labels = metadata.copy()
    shuffled_labels.iloc[np.random.default_rng(19).permutation(len(shuffled_labels))] = metadata.to_numpy()
    shuffled_rna = results_from_labels(shuffled_labels)
    subset_edges = e.loc[e.gene_symbol.isin(counts.index)].copy()
    assert _concordant(r, shuffled_edges) < _concordant(r, e)
    assert _concordant(shuffled_rna, subset_edges) < _concordant(results_from_labels(metadata), subset_edges)


def test_rna_resets_clear_integration_results_but_keep_atac_input():
    app = importlib.import_module("Bulk_RNAseq_Analyzer")
    import streamlit as st
    st.session_state["atac_counts_df"] = pd.DataFrame({"peak": [1]})
    for reset in (app.reset_data_results, app.reset_contrast_results, app.reset_threshold_dependent_results):
        st.session_state["integration_edge_results"] = pd.DataFrame({"old": [1]})
        st.session_state["integration_gene_results"] = pd.DataFrame({"old": [1]})
        st.session_state["rna_contrast"] = {"reference": "control", "test": "treated"}
        reset()
        assert st.session_state["integration_edge_results"] is None
        assert st.session_state["integration_gene_results"] is None
        assert st.session_state["atac_counts_df"] is not None
        if reset is app.reset_threshold_dependent_results:
            assert st.session_state["rna_contrast"] == {"reference": "control", "test": "treated"}
        else:
            assert st.session_state["rna_contrast"] is None
