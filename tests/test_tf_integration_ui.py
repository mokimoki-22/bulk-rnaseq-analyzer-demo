"""Phase 5 Level 2 session-state, invalidation and UI regression tests.

Step 4: TF-only invalidation, the upstream reset chain (design section 15.1), the Level 1 rerun reset and the
``tf_collectri_meta`` bookkeeping.  Later steps add the Level 2 UI and export tests to this file.
"""

from __future__ import annotations

import ast
import copy
import functools
import io
import json
import re
import types
import zipfile
from pathlib import Path

import pytest

import re

import numpy as np
import pandas as pd
from rna_support import capture_downloads
from test_integration_ui import _app
from tf_support import synthetic_tf_level1_state


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "Bulk_RNAseq_Analyzer.py").read_text(encoding="utf-8")
_RESET_FUNCTIONS = {
    "invalidate_tf_level2_results", "reset_tf_integration_results", "reset_integration_results",
    "reset_peak_mapping_results", "reset_atac_results", "reset_atac_input", "reset_atac_species_mapping",
    "reset_contrast_results", "reset_threshold_dependent_results", "reset_data_results",
}


def _load_reset_functions(session):
    """Execute the app's real reset functions against a plain dict standing in for session_state."""
    tree = ast.parse(APP_SOURCE)
    body = [node for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in _RESET_FUNCTIONS)
            or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_DATA_RESULT_DEFAULTS" for t in node.targets))]
    namespace = {"st": types.SimpleNamespace(session_state=session)}
    exec(compile(ast.Module(body=body, type_ignores=[]), "app_reset_functions", "exec"), namespace)
    return namespace


def _populated_session():
    return {
        "integration_edge_results": "edges", "integration_gene_results": "genes", "integration_settings": {"x": 1},
        "integration_summary": {"x": 1}, "integration_enrichment": {"concordant_activation": {"ora": True}},
        "integration_tf_results": {"concordant_activation": {"status": "executed"}},
        "integration_motif_results": None, "integration_motif_source": None,
        "integration_provenance": {"ora_history": [{"integration_class": "concordant_activation"}],
                                   "tf_level2": {"status": "executed"}, "species": "Mouse"},
        "tf_collectri": "activity", "tf_collectri_meta": {"organism": "mouse"}, "tf_dorothea": "activity",
        "atac_species": "Mouse", "atac_genome_build": "mm10", "atac_validated_df": "x", "atac_validation_report": "x",
        "atac_results": "x", "atac_contrast": "x", "atac_peak_gene_edges": "x", "atac_unmapped_peaks": "x",
        "atac_qc_summary": "x", "atac_applied_user_mapping": "x", "atac_reference_metadata": "x",
        "rna_contrast": {"reference": "a", "test": "b"}, "analysis_log": [{"x": 1}],
    }


def test_tf_only_invalidation_keeps_ora_results_and_history():
    session = _populated_session()
    _load_reset_functions(session)["invalidate_tf_level2_results"]()
    assert session["integration_tf_results"] is None
    assert "tf_level2" not in session["integration_provenance"]
    assert session["integration_enrichment"] == {"concordant_activation": {"ora": True}}
    assert session["integration_provenance"]["ora_history"] == [{"integration_class": "concordant_activation"}]
    assert session["integration_provenance"]["species"] == "Mouse"
    assert session["integration_edge_results"] == "edges"          # Level 1 is untouched


def test_tf_only_invalidation_tolerates_missing_provenance():
    session = {"integration_tf_results": {"x": 1}, "integration_provenance": None}
    _load_reset_functions(session)["invalidate_tf_level2_results"]()
    assert session["integration_tf_results"] is None and session["integration_provenance"] is None


def test_reset_tf_integration_results_clears_ora_tf_motif_and_the_provenance_copy():
    session = _populated_session()
    _load_reset_functions(session)["reset_tf_integration_results"]()
    assert session["integration_enrichment"] is None and session["integration_tf_results"] is None
    assert "tf_level2" not in session["integration_provenance"]


@pytest.mark.parametrize("reset_name", [
    "reset_data_results", "reset_contrast_results", "reset_threshold_dependent_results",
    "reset_peak_mapping_results", "reset_atac_results", "reset_atac_input", "reset_atac_species_mapping",
    "reset_integration_results",
])
def test_every_upstream_change_clears_level2_results_and_their_provenance(reset_name):
    """Design section 15.1: RNA re-run, thresholds, contrast, species, mapping and input changes."""
    session = _populated_session()
    _load_reset_functions(session)[reset_name]()
    assert session["integration_tf_results"] is None, reset_name
    assert session["integration_enrichment"] is None, reset_name
    provenance = session["integration_provenance"]
    assert provenance is None or "tf_level2" not in provenance, reset_name


@pytest.mark.parametrize("reset_name", ["reset_data_results", "reset_contrast_results", "reset_threshold_dependent_results"])
def test_activity_metadata_is_cleared_together_with_the_activity_result(reset_name):
    session = _populated_session()
    _load_reset_functions(session)[reset_name]()
    assert session["tf_collectri"] is None and session["tf_collectri_meta"] is None


def test_every_place_that_clears_tf_collectri_also_clears_its_metadata():
    lines = APP_SOURCE.splitlines()
    clears = [i for i, line in enumerate(lines) if line.strip() == 'st.session_state["tf_collectri"] = None']
    assert len(clears) == 2                                          # normalization change and gene-length load
    for index in clears:
        assert 'st.session_state["tf_collectri_meta"] = None' in "\n".join(lines[index:index + 3])
    assert '"tf_collectri_meta": None,' in APP_SOURCE                  # _DATA_RESULT_DEFAULTS
    assert '"tf_results", "tf_collectri", "tf_collectri_meta", "tf_dorothea"' in APP_SOURCE   # reset_contrast_results
    assert 'if "tf_collectri_meta" not in st.session_state' in APP_SOURCE


def test_activity_metadata_is_saved_with_the_activity_result_using_run_time_values():
    saved = APP_SOURCE.index('st.session_state["tf_collectri"] = acts_c')
    block = APP_SOURCE[saved:saved + 500]
    assert 'st.session_state["tf_collectri_meta"] = {' in block
    for key in ('"method_requested": method_sel', '"method_used": method_c', '"tmin": int(min_targets)',
                '"normalization": _norm_method_tf', '"organism": organism', '"network": "collectri"'):
        assert key in block


def test_level1_rerun_clears_stale_ora_tf_and_motif_results_and_keeps_the_new_level1_outputs():
    app = _app()
    app.default_timeout = APPTEST_TIMEOUT_SECONDS
    app.session_state["integration_enrichment"] = {"concordant_activation": {"stale": True}}
    app.session_state["integration_tf_results"] = {"concordant_activation": {"status": "executed"}}
    app.session_state["integration_provenance"] = {"ora_history": [{"stale": True}], "tf_level2": {"stale": True}}
    app.run()
    assert not app.exception
    app.button(key="integration_run").click().run()
    assert not app.exception
    assert app.session_state["integration_enrichment"] is None
    assert app.session_state["integration_tf_results"] is None
    assert app.session_state["integration_motif_results"] is None
    provenance = app.session_state["integration_provenance"]
    assert provenance["ora_history"] == [] and "tf_level2" not in provenance
    # The reset happens before the new results are stored, so the new Level 1 outputs survive it.
    assert app.session_state["integration_edge_results"] is not None
    assert app.session_state["integration_gene_results"] is not None
    assert app.session_state["integration_summary"] is not None
    assert app.session_state["integration_settings"]["rna_contrast"] == {"reference": "Control", "test": "Treatment"}


# ----------------------------------------------------------------------------------------------
# Step 5: Level 2 UI (AppTest with the synthetic Mouse fixture)
# ----------------------------------------------------------------------------------------------

_EDGE_COLUMNS = ["peak_id", "gene_id", "gene_symbol", "edge_id", "atac_log2FoldChange", "atac_padj",
                 "atac_padj_is_na", "atac_lfc_is_na", "mapping_method", "distance_to_tss"]
_CAUSAL = re.compile(r"\b(regulates|drives|causes|caused|causal)\b", re.IGNORECASE)
_NEGATED = "show no evidence that a TF regulates these genes or drives a phenotype"
_FIXTURE = synthetic_tf_level1_state()
APPTEST_TIMEOUT_SECONDS = 300
_LEVEL1_KEYS = ("integration_edge_results", "integration_gene_results", "integration_settings",
                "integration_summary", "integration_provenance")


def _tf_app(with_activity=True, language="English"):
    state = _FIXTURE
    app = _app()
    app.default_timeout = APPTEST_TIMEOUT_SECONDS        # CI runners (especially Windows) are much slower than 60 s allows
    deg = state["deg_results"].assign(baseMean=10.0, pvalue=lambda d: d["padj"], stat=lambda d: d["log2FoldChange"])
    samples = list(state["metadata"].index)
    counts = pd.DataFrame(np.random.default_rng(0).integers(5, 50, size=(len(deg), len(samples))),
                          index=deg.index, columns=samples)
    values = {
        "lang_display": language, "language_selector": language, "deg_results": deg, "counts_df": counts,
        "qc_filtered_df": counts.copy(), "metadata": state["metadata"], "conditions": ["control", "treated"],
        "rna_contrast": dict(state["contrast"]), "atac_contrast": dict(state["contrast"]),
        "last_contrast": "treated vs control", "sp": {"org": "mmu", "string_id": 10090},
        "atac_species": "Mouse", "atac_genome_build": "mm10",
        "atac_peak_gene_edges": state["edges"].loc[:, _EDGE_COLUMNS].copy(),
    }
    for key, value in values.items():
        app.session_state[key] = value
    if with_activity:
        app.session_state["tf_collectri"] = state["activity_scores"].copy()
        # The existing TF tab reads both activity tables, so a DoRothEA result is stored as it would be after a run.
        app.session_state["tf_dorothea"] = state["activity_scores"].iloc[:, :5].copy()
        app.session_state["tf_collectri_meta"] = {
            "method_requested": "ULM", "method_used": "ULM", "tmin": 5, "normalization": "log1p",
            "organism": "mouse", "network": "collectri",
        }
    return app


def _level1_real(app):
    """Run Level 1 through the real button."""
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    app.button(key="integration_run").click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["integration_gene_results"] is not None
    return app


@functools.lru_cache(maxsize=1)
def _level1_snapshot():
    """The stored Level 1 outputs of one real run, so most tests do not repeat the slow full-app Level 1 run."""
    app = _level1_real(_tf_app())
    return {key: copy.deepcopy(app.session_state[key]) for key in _LEVEL1_KEYS}


def _level1(app):
    """Put the (real-run) Level 1 outputs in place and render once."""
    for key, value in copy.deepcopy(_level1_snapshot()).items():
        app.session_state[key] = value
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _level2(app, set_name="concordant_activation"):
    app.selectbox(key="tf_level2_set").set_value(set_name)
    app.button(key="tf_level2_run").click().run()
    assert not app.exception, [e.value for e in app.exception]
    return app

def _texts(app):
    elements = list(app.markdown) + list(app.caption) + list(app.warning) + list(app.info) + list(app.error)
    return [element.value for element in elements]


def _table(app):
    frames = [element.value for element in app.dataframe if "supported / evaluable axes" in element.value.columns]
    return frames[0] if frames else None


def test_level2_controls_are_hidden_before_level1_and_shown_after_it():
    app = _tf_app()
    app.run()
    assert not app.exception
    assert not [b for b in app.button if b.key == "tf_level2_run"]
    assert any("Level 3 (motif) is not available in this version" in text for text in _texts(app))
    _level1_real(app)
    button = [b for b in app.button if b.key == "tf_level2_run"]
    assert len(button) == 1 and not button[0].disabled
    texts = " ".join(_texts(app))
    assert "Background size:" in texts and "neither all genes nor only the significant genes" in texts
    assert "literal count" in texts and "Level 2 has not been run for this gene set." in texts
    assert not [b for b in app.button if "motif" in (b.key or "").lower()]


def test_level2_run_shows_separate_axes_not_run_motif_and_permanent_limitations():
    app = _level2(_level1(_tf_app()))
    table = _table(app)
    assert table is not None and "motif_status" in table.columns
    planted = table.set_index("tf_symbol").loc[_FIXTURE["planted_tf"]]
    assert planted["supported / evaluable axes"] == "3 / 3" and planted["tf_activity_status"] == "separated_up"
    assert planted["tf_expression_status"] == "supported_up" and set(table["motif_status"]) == {"not run"}
    assert not [c for c in table.columns if any(w in c.lower() for w in ("confidence", "combined", "weighted"))]
    texts = " ".join(_texts(app))
    assert "Limitations" in texts and "not independent evidence" in texts and "about 10% of null TFs" in texts
    assert "Motif enrichment: not run" in texts and "sorting aid, not a statistic" in texts
    assert "How to read the table" in texts and "not_tested = DESeq2 gave NA" in texts and "not \"not significant\"" in texts
    assert "corrected within the selected gene set across the TFs that were tested only" in texts
    runs = app.session_state["integration_tf_results"]
    assert set(runs) == {"concordant_activation"} and runs["concordant_activation"]["status"] == "executed"
    assert app.session_state["integration_provenance"]["tf_level2"]["status"] == "executed"
    assert any(element.key == "tf_level2_drill_tf" for element in app.selectbox)


def test_activity_column_shows_not_run_and_points_to_the_tf_tab_when_activity_was_not_run():
    app = _level2(_level1(_tf_app(with_activity=False)))
    table = _table(app)
    planted = table.set_index("tf_symbol").loc[_FIXTURE["planted_tf"]]
    assert planted["tf_activity_status"] == "not run" and planted["supported / evaluable axes"] == "2 / 2"
    assert any("TF Activity has not been run" in text for text in _texts(app))


def test_level2_is_blocked_for_gene_id_level1_runs_and_for_unsupported_species():
    app = _level1(_tf_app())
    app.session_state["integration_settings"]["rna_gene_id_type"] = "gene_id"
    app.run()
    button = [b for b in app.button if b.key == "tf_level2_run"][0]
    assert button.disabled and any("gene symbol identifier" in text for text in _texts(app))
    app = _level1(_tf_app())
    app.session_state["integration_settings"]["species"] = "Rat"
    app.run()
    assert [b for b in app.button if b.key == "tf_level2_run"][0].disabled


def test_level2_stops_when_activity_was_estimated_for_another_species():
    app = _level1(_tf_app())
    app.session_state["tf_collectri_meta"] = {**app.session_state["tf_collectri_meta"], "organism": "human"}
    app.run()
    app.button(key="tf_level2_run").click().run()
    assert not app.exception
    assert app.session_state["integration_tf_results"] is None
    assert any("different species" in text for text in _texts(app))


def test_activity_without_recorded_parameters_is_treated_as_not_run():
    app = _tf_app()
    app.session_state["tf_collectri_meta"] = None
    table = _table(_level2(_level1(app)))
    assert set(table["tf_activity_status"]) == {"not run"}


def test_empty_gene_set_is_reported_and_not_tested():
    app = _level2(_level1(_tf_app()), "mixed_accessibility")
    run = app.session_state["integration_tf_results"]["mixed_accessibility"]
    assert run["status"] == "empty_gene_set" and _table(app) is None
    assert any("not tested" in text for text in _texts(app))


def test_small_gene_sets_show_the_exploratory_warning_when_flagged():
    app = _level1(_tf_app())
    executed = 0
    for set_name in ("discordant_open_down", "rna_only_on_mapped_peak", "concordant_repression"):
        _level2(app, set_name)
        run = app.session_state["integration_tf_results"][set_name]
        if run["status"] != "executed":
            continue
        executed += 1
        shown = any("fewer than 20 genes" in text for text in _texts(app))
        assert shown == run["gene_set"]["small_gene_set_warning"], set_name
    assert executed >= 1

def test_reactivity_change_clears_only_level2_and_keeps_ora_results():
    app = _level2(_level1(_tf_app()))
    app.session_state["integration_enrichment"] = {"concordant_activation": {"ora": True}}
    app.session_state["integration_provenance"] = {**app.session_state["integration_provenance"],
                                                   "ora_history": [{"integration_class": "concordant_activation"}]}
    changed = app.session_state["tf_collectri"].copy()
    changed.iloc[0, 0] += 1.0                                    # the TF Activity result was estimated again
    app.session_state["tf_collectri"] = changed
    app.run()
    assert not app.exception
    assert app.session_state["integration_tf_results"] is None
    assert "tf_level2" not in app.session_state["integration_provenance"]
    assert app.session_state["integration_enrichment"] == {"concordant_activation": {"ora": True}}
    assert app.session_state["integration_provenance"]["ora_history"]
    assert any("TF Activity result changed" in text for text in _texts(app))
    assert app.session_state["integration_gene_results"] is not None          # Level 1 is untouched


def test_relabelled_samples_with_the_same_contrast_labels_clear_level2():
    app = _level2(_level1(_tf_app()))
    relabelled = app.session_state["metadata"].copy()
    relabelled.iloc[0, relabelled.columns.get_loc("condition")] = "treated"
    app.session_state["metadata"] = relabelled
    app.run()
    assert app.session_state["integration_tf_results"] is None


def test_clearing_the_activity_result_as_the_normalization_change_does_clears_level2_results():
    """The normalization and gene-length paths clear tf_collectri without calling the reset chain.

    The render-time fingerprint check must notice "was present, now absent" (the clearing sites themselves are
    checked in test_every_place_that_clears_tf_collectri_also_clears_its_metadata).
    """
    app = _level2(_level1(_tf_app()))
    assert app.session_state["integration_tf_results"] is not None
    app.session_state["tf_collectri"] = None
    app.session_state["tf_collectri_meta"] = None
    app.session_state["tf_dorothea"] = None
    app.run()
    assert not app.exception
    assert app.session_state["integration_tf_results"] is None
    assert "tf_level2" not in app.session_state["integration_provenance"]
    assert any("TF Activity result changed" in text for text in _texts(app))
    assert app.session_state["integration_gene_results"] is not None

def test_level1_result_change_clears_all_downstream_results_with_a_notice():
    app = _level2(_level1(_tf_app()))
    stored = dict(app.session_state["integration_tf_results"])
    stale = {**stored["concordant_activation"], "fingerprints": {"input_fingerprint": "old", "activity_fingerprint": None}}
    app.session_state["integration_tf_results"] = {"concordant_activation": stale}
    app.session_state["integration_enrichment"] = {"concordant_activation": {"ora": True}}
    app.run()
    assert app.session_state["integration_tf_results"] is None
    assert app.session_state["integration_enrichment"] is None
    assert any("Level 1 result changed" in text for text in _texts(app))


def test_level2_text_has_no_causal_wording_in_english_and_shows_japanese_limitations():
    app = _level2(_level1(_tf_app()))
    for text in _texts(app):
        assert not _CAUSAL.search(text.replace(_NEGATED, "")), text
    japanese = _level2(_level1(_tf_app(language="日本語")))
    joined = " ".join(_texts(japanese))
    assert "限界" in joined and "レベル3（motif）はこの版では利用できません" in joined
    assert "背景遺伝子数" in joined and "TF Activity" in joined or "支持軸数" in joined
    assert "制御する" not in joined.replace("これらの遺伝子を制御する、あるいは表現型を引き起こすことを示す根拠ではありません", "")


def test_drill_down_lists_the_selected_tfs_targets_with_every_peak_gene_edge():
    app = _level2(_level1(_tf_app()))
    app.selectbox(key="tf_level2_drill_tf").set_value(_FIXTURE["planted_tf"]).run()
    assert not app.exception
    drill = [e.value for e in app.dataframe if "tf_target_weight" in e.value.columns]
    assert drill and set(drill[0]["gene_symbol"]) == set(_FIXTURE["planted_targets"][:25])
    assert len(drill[0]) >= 25

# ----------------------------------------------------------------------------------------------
# Step 6: Level 2 export, manifest and provenance
# ----------------------------------------------------------------------------------------------

def _export(app):
    with capture_downloads() as downloads:
        app.run()
    assert not app.exception, [e.value for e in app.exception]
    archive = zipfile.ZipFile(io.BytesIO(downloads["results.zip"]))
    return archive, json.loads(archive.read("Provenance/manifest.json"))


def test_export_has_no_level2_files_or_manifest_block_before_level2_is_run():
    archive, manifest = _export(_level1(_tf_app()))
    names = set(archive.namelist())
    assert "Integration/gene_summary.csv" in names
    assert "Integration/tf_candidates.csv" not in names and "Integration/tf_summary.json" not in names
    assert "tf_level2" not in manifest["settings"]["integration"]


def test_export_contains_level2_candidates_summary_and_full_manifest_provenance():
    app = _level2(_level1(_tf_app()))
    app = _level2(app, "atac_only")
    archive, manifest = _export(app)
    names = set(archive.namelist())
    assert {"Integration/tf_candidates.csv", "Integration/tf_summary.json", "Integration/gene_summary.csv",
            "Integration/ORA/history.json"}.issubset(names)
    candidates = pd.read_csv(io.BytesIO(archive.read("Integration/tf_candidates.csv")))
    assert set(candidates["gene_set"]) == {"concordant_activation", "atac_only"}
    assert set(candidates["motif_status"]) == {"not_run"}
    assert candidates["motif_enrichment_padj"].isna().all() and candidates["motif_enrichment_score"].isna().all()
    assert candidates["motif_source"].isna().all()
    assert candidates["targets_in_set"].map(lambda value: isinstance(value, str) or pd.isna(value)).all()
    assert {"n_axes_supported", "n_axes_evaluable", "tf_activity_status", "tf_expression_status",
            "target_enrichment_padj", "fisher_alternative"} <= set(candidates.columns)
    block = manifest["settings"]["integration"]["tf_level2"]
    assert json.loads(archive.read("Integration/tf_summary.json")) == block
    assert block["status"] == "executed" and block["external_services_used"] == []
    assert block["network"] == {"source": "collectri", "organism": "mouse", "n_edges": len(_FIXTURE["network"]),
                                "file": "references/tf_networks/collectri_mouse.csv.gz"}
    assert block["fisher_alternative"] == "greater" and block["min_targets"] == 10 and block["alpha"] == 0.05
    assert block["bh_scope"] == "within_gene_set_across_tested_tfs" and block["motif_axis"] == "not_run"
    assert block["universe"]["universe_size"] > 0 and block["universe"]["universe_definition_ja"]
    assert {gene_set["name"] for gene_set in block["gene_sets"]} == {"concordant_activation", "atac_only"}
    assert block["symbol_matching"]["matching"] == "strip+casefold"
    assert block["expression_rule"]["thresholds"] == {"rna_lfc": 1.0, "rna_padj": 0.05}
    assert block["activity_parameters"]["organism"] == "mouse" and block["activity_parameters"]["tmin"] == 5
    assert block["fingerprints"]["input_fingerprint"] and block["fingerprints"]["activity_fingerprint"]
    assert block["limitations_text"] and block["limitations_text_ja"] and block["contrast"] == {
        "reference": "control", "test": "treated"}
    assert len(block["run_history"]) == 2
    assert manifest["services"]["external_services_used"] == []


def test_export_omits_level2_when_its_inputs_are_no_longer_current():
    app = _level2(_level1(_tf_app()))
    changed = app.session_state["tf_collectri"].copy()
    changed.iloc[0, 0] += 1.0
    app.session_state["tf_collectri"] = changed
    archive, manifest = _export(app)
    assert "Integration/tf_candidates.csv" not in archive.namelist()
    assert "tf_level2" not in manifest["settings"]["integration"]


def test_export_builds_level2_only_from_current_results_and_never_from_the_stored_copy():
    start = APP_SOURCE.index("exported_provenance = {key: value")
    block = APP_SOURCE[start:start + 900]
    assert "_current_tf_level2_runs(integration_genes)" in block and 'if key != "tf_level2"' in block
    assert "settings[\"integration\"] = integration_provenance" not in APP_SOURCE


def test_level1_provenance_and_ora_history_are_unchanged_by_the_level2_export():
    app = _level2(_level1(_tf_app()))
    archive, manifest = _export(app)
    integration = manifest["settings"]["integration"]
    assert integration["rna_contrast"] == {"reference": "control", "test": "treated"}
    assert integration["ora_history"] == [] and "quadrant" in integration
    assert json.loads(archive.read("Integration/ORA/history.json")) == []
    notebook = archive.read("Integration/analysis_notebook.md").decode("utf-8")
    assert "tf_level2" in notebook

def test_export_records_each_gene_sets_own_min_targets_and_alpha_when_the_sliders_changed_between_runs():
    app = _level1(_tf_app())
    _level2(app, "concordant_activation")
    app.slider(key="tf_level2_min_targets").set_value(20).run()
    app.selectbox(key="tf_level2_alpha").set_value(0.1).run()
    _level2(app, "atac_only")
    archive, manifest = _export(app)
    block = manifest["settings"]["integration"]["tf_level2"]
    assert block["settings_vary_between_gene_sets"] is True and block["min_targets"] is None and block["alpha"] is None
    per_set = {entry["name"]: (entry["min_targets"], entry["alpha"]) for entry in block["gene_sets"]}
    assert per_set == {"concordant_activation": (10, 0.05), "atac_only": (20, 0.1)}
    candidates = pd.read_csv(io.BytesIO(archive.read("Integration/tf_candidates.csv")))
    rows = candidates.groupby("gene_set")[["min_targets", "alpha"]].first().to_dict("index")
    assert rows == {"concordant_activation": {"min_targets": 10, "alpha": 0.05},
                    "atac_only": {"min_targets": 20, "alpha": 0.1}}