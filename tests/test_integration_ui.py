"""Phase 4 Level 1 integration UI and export regression tests."""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import pandas as pd
from streamlit.testing.v1 import AppTest

import brim_integration_enrichment
from rna_support import capture_downloads


ROOT = Path(__file__).resolve().parents[1]


def _app(atac_contrast=None):
    deg = pd.DataFrame({
        "baseMean": [15.0, 9.0, 15.0], "log2FoldChange": [2.0, -2.0, 0.1],
        "pvalue": [0.005, 0.005, 0.6], "stat": [3.0, -3.0, 0.2], "padj": [0.01, 0.01, 0.7],
        "padj_is_na": [False, False, False], "lfc_is_na": [False, False, False],
    }, index=["G1", "G2", "G3"])
    counts = pd.DataFrame({"control_1": [10, 12, 15], "control_2": [11, 13, 16],
                           "treated_1": [20, 6, 15], "treated_2": [21, 5, 14]}, index=deg.index)
    metadata = pd.DataFrame({"condition": ["Control", "Control", "Treatment", "Treatment"]}, index=counts.columns)
    edges = pd.DataFrame({
        "peak_id": ["p1", "p2", "p3"], "gene_id": ["G1", "G2", "G3"],
        "gene_symbol": ["G1", "G2", "G3"], "edge_id": ["e1", "e2", "e3"],
        "atac_log2FoldChange": [2.0, 2.0, 0.1], "atac_padj": [0.01, 0.01, 0.7],
        "atac_padj_is_na": [False, False, False], "atac_lfc_is_na": [False, False, False],
        "mapping_method": ["promoter"] * 3, "distance_to_tss": [0] * 3,
    })
    atac = pd.DataFrame({
        "peak_id": ["p1", "p2", "p3"], "chrom": ["chr1"] * 3,
        "start": [1, 11, 21], "end": [5, 15, 25], "log2FoldChange": [2.0, 2.0, 0.1],
        "padj": [0.01, 0.01, 0.7], "padj_is_na": [False] * 3, "lfc_is_na": [False] * 3,
        "is_significant": [True, True, False], "accessibility_direction": ["opening", "opening", "none"],
    })
    app = AppTest.from_file(str(ROOT / "Bulk_RNAseq_Analyzer.py"), default_timeout=60)
    state = {
        "lang_display": "English", "language_selector": "English", "deg_results": deg,
        "counts_df": counts, "qc_filtered_df": counts.copy(), "metadata": metadata,
        "conditions": ["Control", "Treatment"],
        "rna_contrast": {"reference": "Control", "test": "Treatment"},
        "last_contrast": "Treatment vs Control", "sp": {"org": "hsa", "string_id": 9606},
        "lfc_t": 1.0, "padj_t": 0.05, "deg_t": (1.0, 0.05),
        "atac_results": atac, "atac_peak_gene_edges": edges,
        "atac_contrast": atac_contrast or {"reference": "Control", "test": "Treatment"},
        "atac_species": "Human", "atac_genome_build": "hg38",
        "atac_validation_report": {"thresholds": {"padj": 0.05, "log2FoldChange": 1.0}, "source_mode": "dar_table"},
        "atac_input_provenance": {"file_name": "atac.csv", "byte_size": 1, "sha256": "a" * 64, "source_mode": "dar_table"},
    }
    for key, value in state.items():
        app.session_state[key] = value
    return app


def test_level1_integration_runs_and_shared_export_contains_results():
    app = _app()
    app.run()
    assert not app.exception
    app.button(key="integration_run").click().run()
    assert not app.exception
    assert app.session_state["integration_edge_results"] is not None
    assert set(app.session_state["integration_gene_results"]["integration_class"]) == {
        "concordant_activation", "discordant_open_down", "not_significant"
    }
    with capture_downloads() as downloads:
        app.run()
    with zipfile.ZipFile(io.BytesIO(downloads["results.zip"])) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("Provenance/manifest.json"))
    assert {"Integration/integration_edges.csv", "Integration/gene_summary.csv", "Integration/summary.json",
            "Integration/analysis_notebook.md", "Integration/ORA/history.json"}.issubset(names)
    assert manifest["settings"]["integration"]["rna_contrast"] == {"reference": "Control", "test": "Treatment"}
    quadrant = manifest["settings"]["integration"]["quadrant"]
    assert quadrant["unit"] == "gene_summary" and quadrant["plotted_genes"] == 3
    assert set(quadrant["exclusions"]) == {
        "both_not_tested", "rna_not_tested", "atac_not_tested", "mixed_accessibility",
        "rna_only_no_mapped_peak", "missing_atac_coordinates",
    }


def test_contrast_mismatch_blocks_integration_without_log2fc_inversion():
    app = _app({"reference": "Treatment", "test": "Control"})
    app.run()
    assert not app.exception
    assert app.button(key="integration_run").disabled
    assert app.session_state["integration_edge_results"] is None


def test_ora_export_keeps_each_same_class_execution_with_its_class_name(monkeypatch):
    def fake_ora(summary, integration_class, species):
        assert species == "Human"
        return {
            "integration_class": integration_class, "species": species, "input_genes": ["G1"],
            "background_genes": ["G1", "G2", "G3"], "warnings": [],
            "libraries": {"KEGG": {"status": "executed", "library": "KEGG_2021_Human",
                                    "result_count": 1, "results": pd.DataFrame({"Term": ["pathway"]})}},
            "history": [{"library_type": "KEGG", "status": "executed", "library": "KEGG_2021_Human", "result_count": 1}],
            "independent_test_notice": "ORA adjusted p-values are new, independent tests and are not combined with RNA or ATAC adjusted p-values.",
        }

    monkeypatch.setattr(brim_integration_enrichment, "run_class_ora", fake_ora)
    app = _app()
    app.run()
    app.button(key="integration_run").click().run()
    app.button(key="integration_run_ora").click().run()
    app.button(key="integration_run_ora").click().run()
    history = app.session_state["integration_provenance"]["ora_history"]
    assert len(history) == 2 and all(item["integration_class"] == "concordant_activation" for item in history)
    with capture_downloads() as downloads:
        app.run()
    with zipfile.ZipFile(io.BytesIO(downloads["results.zip"])) as archive:
        export_history = json.loads(archive.read("Integration/ORA/history.json"))
    assert len(export_history) == 2 and all(item["integration_class"] == "concordant_activation" for item in export_history)
