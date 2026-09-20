"""Phase 4 local ORA policy tests without Streamlit."""

from __future__ import annotations

import pandas as pd
import pytest

import brim_integration_enrichment as enrichment
from brim_multiomics import IntegrationError


def _summary():
    return pd.DataFrame({
        "gene_symbol": ["CA", "NS", "MIX", "RNA_NA", "ATAC_NA", "UNMAPPED"],
        "integration_class": ["concordant_activation", "not_significant", "mixed_accessibility", "rna_not_tested", "atac_not_tested", "rna_only_no_mapped_peak"],
        "rna_padj_is_na": [False, False, False, True, False, False],
        "rna_lfc_is_na": [False, False, False, False, False, False],
        "n_atac_tested_peaks": [1, 2, 1, 1, 0, 0],
    })


def test_background_keeps_tested_mapped_genes_and_not_significant():
    assert enrichment.build_ora_background(_summary()) == ["CA", "MIX", "NS"]
    genes, background, warnings = enrichment.prepare_class_ora(_summary(), "concordant_activation")
    assert genes == ["CA"] and background == ["CA", "MIX", "NS"] and warnings
    with pytest.raises(IntegrationError):
        enrichment.prepare_class_ora(_summary(), "rna_not_tested")


def test_mouse_go_is_recorded_as_unavailable_without_calling_runner():
    calls = []

    def runner(genes, library, background):
        calls.append(library)
        return pd.DataFrame({"Term": ["pathway"]})

    result = enrichment.run_class_ora(_summary(), "concordant_activation", "Mouse", runner)
    assert calls == ["KEGG_2019_Mouse"]
    assert result["libraries"]["GO_BP"]["status"] == "unavailable"
    assert "human gene symbols" in result["libraries"]["GO_BP"]["reason"]


def test_ora_result_reports_background_definition_size_and_mixed_direction_note():
    def runner(genes, library, background):
        return pd.DataFrame({"Term": ["pathway"]})

    result = enrichment.run_class_ora(_summary(), "concordant_activation", "Human", runner)
    assert result["background_size"] == len(result["background_genes"]) == 3
    assert "not-significant genes are included" in result["background_definition"]
    assert result["background_definition_ja"] and result["direction_note"] is None
    mixed = enrichment.run_class_ora(_summary(), "mixed_accessibility", "Human", runner)
    assert "does not assume a single direction" in mixed["direction_note"] and mixed["direction_note_ja"]


def test_zero_overlap_is_retained_as_an_executed_local_attempt():
    def no_overlap(genes, library, background):
        raise ValueError("No pathway overlaps were found for the selected genes.")

    result = enrichment.run_class_ora(_summary(), "concordant_activation", "Human", no_overlap)
    assert {record["status"] for record in result["history"]} == {"zero_overlap"}
