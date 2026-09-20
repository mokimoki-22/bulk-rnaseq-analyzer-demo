"""Phase 6 Level 3 session-state, invalidation and (later) UI regression tests.

Step 4: the Level 3 clearing rules and the stale check, run against the app's real functions (extracted from the app
source, so no heavy full-app run is needed).  Step 5: two AppTests for the Level 3 UI (at most three in total, plan D14;
the third covers the export in step 6).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import types
import zipfile
from pathlib import Path

import pytest

import brim_motif_import as mi
from motif_support import THRESHOLDS, generic_motif_csv, homer_known_text, synthetic_dar_table
from rna_support import capture_downloads
from test_tf_integration_ui import _FIXTURE, _level1, _level2, _table, _texts, _tf_app


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "Bulk_RNAseq_Analyzer.py").read_text(encoding="utf-8")
_FUNCTIONS = {
    "invalidate_tf_level2_results", "reset_tf_integration_results", "reset_motif_results", "reset_integration_results",
    "reset_peak_mapping_results", "reset_atac_results", "reset_atac_input", "reset_atac_species_mapping",
    "reset_contrast_results", "reset_threshold_dependent_results", "reset_data_results",
}
UPSTREAM_RESETS = ["reset_data_results", "reset_contrast_results", "reset_threshold_dependent_results",
                   "reset_peak_mapping_results", "reset_atac_results", "reset_atac_input", "reset_atac_species_mapping",
                   "reset_integration_results", "reset_tf_integration_results", "invalidate_tf_level2_results"]


def _load(session, names=None, extra=None):
    """Execute the app's real functions (and _DATA_RESULT_DEFAULTS) against a dict standing in for session_state."""
    wanted = _FUNCTIONS if names is None else set(names)
    tree = ast.parse(APP_SOURCE)
    body = [node for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in wanted)
            or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "_DATA_RESULT_DEFAULTS" for t in node.targets))]
    namespace = {"st": types.SimpleNamespace(session_state=session), "brim_motif_import": mi, **(extra or {})}
    exec(compile(ast.Module(body=body, type_ignores=[]), "app_functions", "exec"), namespace)
    return namespace


def _populated():
    return {
        "integration_edge_results": "edges", "integration_gene_results": "genes", "integration_settings": {"x": 1},
        "integration_summary": {"x": 1}, "integration_enrichment": {"concordant_activation": {"ora": True}},
        "integration_tf_results": {"concordant_activation": {"status": "executed"}},
        "integration_motif_source": {"peakset_fingerprint": "f"}, "integration_motif_results": {"imports": {}, "history": []},
        "integration_provenance": {"ora_history": [{"integration_class": "concordant_activation"}], "species": "Mouse",
                                   "tf_level2": {"status": "executed"}, "tf_level3": {"status": "imported"}},
        "tf_collectri": "activity", "tf_collectri_meta": {"organism": "mouse"}, "tf_dorothea": "activity",
        "atac_species": "Mouse", "atac_genome_build": "mm10", "atac_validated_df": "x", "atac_validation_report": "x",
        "atac_results": "x", "atac_contrast": "x", "atac_peak_gene_edges": "x", "atac_unmapped_peaks": "x",
        "atac_qc_summary": "x", "atac_applied_user_mapping": "x", "atac_reference_metadata": "x",
        "rna_contrast": {"reference": "a", "test": "b"}, "analysis_log": [{"x": 1}],
    }


@pytest.mark.parametrize("reset_name", UPSTREAM_RESETS)
def test_every_upstream_change_and_a_level2_invalidation_clear_the_motif_state_and_its_provenance(reset_name):
    session = _populated()
    _load(session)[reset_name]()
    assert session["integration_motif_source"] is None and session["integration_motif_results"] is None, reset_name
    assert session["integration_tf_results"] is None, reset_name
    provenance = session["integration_provenance"]
    assert provenance is None or not {"tf_level2", "tf_level3"} & set(provenance), reset_name


def test_a_level2_invalidation_leaves_ora_and_level1_and_the_other_provenance_untouched():
    session = _populated()
    _load(session)["invalidate_tf_level2_results"]()
    assert session["integration_enrichment"] == {"concordant_activation": {"ora": True}}
    assert session["integration_provenance"]["ora_history"] == [{"integration_class": "concordant_activation"}]
    assert session["integration_provenance"]["species"] == "Mouse"
    assert session["integration_edge_results"] == "edges" and session["integration_settings"] == {"x": 1}


def test_the_level2_invalidation_is_self_contained_it_needs_no_other_app_function():
    session = _populated()
    namespace = _load(session, names={"invalidate_tf_level2_results"})            # nothing else is defined: a call would be a NameError
    namespace["invalidate_tf_level2_results"]()
    assert session["integration_motif_results"] is None and "tf_level3" not in session["integration_provenance"]
    body = ast.get_source_segment(APP_SOURCE, next(n for n in ast.parse(APP_SOURCE).body
                                                   if isinstance(n, ast.FunctionDef) and n.name == "invalidate_tf_level2_results"))
    assert "reset_motif_results" not in body.split('"""')[2]                       # not called in the code (the docstring may mention it)


def test_the_general_reset_also_clears_ora_and_needs_only_the_level2_invalidation():
    session = _populated()
    _load(session, names={"invalidate_tf_level2_results", "reset_tf_integration_results"})["reset_tf_integration_results"]()
    assert session["integration_enrichment"] is None and session["integration_motif_results"] is None
    assert "tf_level3" not in session["integration_provenance"] and session["integration_provenance"]["species"] == "Mouse"


def test_the_motif_only_reset_clears_level3_and_nothing_else():
    session = _populated()
    _load(session, names={"reset_motif_results"})["reset_motif_results"]()
    assert session["integration_motif_source"] is None and session["integration_motif_results"] is None
    assert session["integration_provenance"] == {"ora_history": [{"integration_class": "concordant_activation"}],
                                                 "species": "Mouse", "tf_level2": {"status": "executed"}}
    assert session["integration_tf_results"] == {"concordant_activation": {"status": "executed"}}
    assert session["integration_enrichment"] == {"concordant_activation": {"ora": True}}
    session["integration_provenance"] = None
    _load(session, names={"reset_motif_results"})["reset_motif_results"]()        # tolerates a missing provenance
    assert session["integration_provenance"] is None


def test_the_motif_only_reset_is_called_only_by_the_render_time_stale_check():
    callers = [n.name for n in ast.walk(ast.parse(APP_SOURCE)) if isinstance(n, ast.FunctionDef) and n.name != "reset_motif_results"
               and "reset_motif_results(" in (ast.get_source_segment(APP_SOURCE, n) or "").replace('"""', "", 2)]
    assert set(callers) <= {"_render_tf_level3_ui"}                                # step 5 adds this single caller


def _bound_state(peak_sets, fingerprint=None):
    bundle = mi.build_motif_bundle(peak_sets, "9.9.9", "t")
    source = mi.prepared_source_record(peak_sets, bundle, "t", "9.9.9")
    if fingerprint:
        source["peakset_fingerprint"] = fingerprint
    return source


def _current_state_env(dar, level2_runs="runs"):
    settings = {"thresholds": THRESHOLDS, "genome_build": "hg38", "species": "Human"}
    session = {"atac_results": dar, "integration_settings": settings, "integration_motif_source": None,
               "integration_motif_results": None}
    namespace = _load(session, names={"_current_peakset_fingerprint", "_current_motif_state"},
                      extra={"_current_tf_level2_runs": lambda genes: (level2_runs, None)})
    return session, namespace


def test_the_stale_check_treats_the_state_as_absent_without_modifying_it():
    dar = synthetic_dar_table()
    peak_sets = mi.build_peak_sets(dar, THRESHOLDS, "hg38", "Human")
    session, namespace = _current_state_env(dar)
    assert namespace["_current_motif_state"]("genes") == (None, None, None)                   # nothing prepared yet
    source = _bound_state(peak_sets)
    state = {"imports": {"opening": {"record": {"peakset_fingerprint": peak_sets.peakset_fingerprint}},
                         "closing": {"record": {"peakset_fingerprint": "stale"}}}, "history": [{"h": 1}]}
    session.update(integration_motif_source=source, integration_motif_results=state)
    before = copy.deepcopy(session)
    current_source, current_state, reason = namespace["_current_motif_state"]("genes")
    assert reason is None and current_source["peakset_fingerprint"] == peak_sets.peakset_fingerprint
    assert set(current_state["imports"]) == {"opening"} and current_state["history"] == [{"h": 1}]        # the stale import is left out
    plain = lambda s: {k: v for k, v in s.items() if k != "atac_results"}                        # a DataFrame cannot be compared with ==
    assert plain(session) == plain(before) and session["atac_results"] is dar                    # session state was not modified


def test_the_stale_check_reports_a_changed_peak_set_or_a_missing_level2_result():
    dar = synthetic_dar_table()
    peak_sets = mi.build_peak_sets(dar, THRESHOLDS, "hg38", "Human")
    source = _bound_state(peak_sets)
    session, namespace = _current_state_env(dar)
    session.update(integration_motif_source=source, integration_motif_results={"imports": {}, "history": []})
    assert namespace["_current_motif_state"]("genes")[2] is None
    for change in ({"thresholds": {"atac_padj": 0.01, "atac_lfc": 1.0}}, {"genome_build": "mm10"}, {"species": "Mouse"}):
        session["integration_settings"] = {**session["integration_settings"], **change}
        assert namespace["_current_motif_state"]("genes") == (None, None, "peakset"), change
    session["integration_settings"] = {"thresholds": THRESHOLDS, "genome_build": "hg38", "species": "Human"}
    session["atac_results"] = dar.assign(padj=dar["padj"] * 0.5)
    assert namespace["_current_motif_state"]("genes")[2] == "peakset"                          # the DAR values changed
    session["atac_results"] = None
    assert namespace["_current_motif_state"]("genes")[2] == "peakset"
    session["atac_results"] = dar
    session2 = {"atac_results": dar, "integration_settings": session["integration_settings"],
                "integration_motif_source": source, "integration_motif_results": {"imports": {}, "history": []}}
    namespace2 = _load(session2, names={"_current_peakset_fingerprint", "_current_motif_state"},
                       extra={"_current_tf_level2_runs": lambda genes: (None, None)})
    assert namespace2["_current_motif_state"]("genes") == (None, None, "level2")               # Level 3 needs a current Level 2 result


def test_the_peak_set_fingerprint_helper_returns_none_when_it_cannot_be_computed():
    session, namespace = _current_state_env(synthetic_dar_table())
    assert namespace["_current_peakset_fingerprint"](None) is None
    session["atac_results"] = None
    assert namespace["_current_peakset_fingerprint"](session["integration_settings"]) is None
    session["atac_results"] = synthetic_dar_table().drop(columns=["padj_is_na"])
    assert namespace["_current_peakset_fingerprint"](session["integration_settings"]) is None      # invalid DAR: no crash
    session["atac_results"] = synthetic_dar_table()
    assert namespace["_current_peakset_fingerprint"](session["integration_settings"]) == \
        mi.build_peak_sets(synthetic_dar_table(), THRESHOLDS, "hg38", "Human").peakset_fingerprint


# ----------------------------------------------------------------------------------------------
# Step 5: the Level 3 UI (AppTest; the uploader itself cannot be driven by AppTest, so the import is set up through the
# module's own functions and session state, and the upload handler's core is unit-tested in test_motif_import.py)
# ----------------------------------------------------------------------------------------------

def _level3_app():
    """Level 1 and Level 2 done on the synthetic Mouse fixture, with a DAR table that has enough peaks."""
    app = _tf_app()
    app.session_state["atac_results"] = synthetic_dar_table()
    _level1(app)
    return _level2(app)


def _mouse_peak_sets():
    return mi.build_peak_sets(synthetic_dar_table(), THRESHOLDS, "mm10", "Mouse")


def test_level3_waits_for_level2_then_prepares_the_files_and_shows_the_command_without_touching_the_axis_counts():
    app = _tf_app()
    app.session_state["atac_results"] = synthetic_dar_table()
    _level1(app)
    assert not [b for b in app.button if (b.key or "").startswith("tf_level3_")]              # Level 2 has not been run yet
    _level2(app)
    assert [b for b in app.button if b.key == "tf_level3_prepare"] and not [b for b in app.button if b.key == "tf_level3_import"]
    before = _table(app)
    assert set(before["motif_status"]) == {"not run"}
    with capture_downloads() as downloads:
        app.button(key="tf_level3_prepare").click().run()
    assert not app.exception, [e.value for e in app.exception]
    source = app.session_state["integration_motif_source"]
    assert source["peakset_fingerprint"] == _mouse_peak_sets().peakset_fingerprint and app.session_state["integration_motif_results"] is None
    assert app.session_state["integration_provenance"]["tf_level3"]["status"] == "bed_prepared_no_import"
    archive = zipfile.ZipFile(io.BytesIO(downloads["MotifAnalysis.zip"]))
    assert set(archive.namelist()) == {"MotifAnalysis/opened_peaks_padj0.05_lfc1.bed", "MotifAnalysis/closed_peaks_padj0.05_lfc1.bed",
                                       "MotifAnalysis/all_peaks_background.bed", "MotifAnalysis/motif_analysis_README.txt"}
    for name in archive.namelist():                                                           # the ZIP matches the recorded checksums
        assert hashlib.sha256(archive.read(name)).hexdigest() == source["exported_files_sha256"][name]
    codes = [c.value for c in app.code]
    assert any("findMotifsGenome.pl opened_peaks_padj0.05_lfc1.bed mm10 homer_opening/" in code for code in codes)
    assert any("configureHomer.pl -install mm10" in code for code in codes) and not any("hg38" in code for code in codes)
    texts = " ".join(_texts(app))
    assert "BRIM does not scan sequences or run any tool" in texts and "Motif enrichment: not run" in texts
    after = _table(app)
    assert "motif_status" not in after.columns and set(after["motif_opening_status"]) == {"not_imported"}
    assert set(after["motif_closing_status"]) == {"not_imported"}
    assert list(after["supported / evaluable axes"]) == list(before["supported / evaluable axes"])       # the Level 2 counts are unchanged
    assert [b for b in app.button if b.key == "tf_level3_import"]                              # the import controls appear once prepared


def test_an_imported_result_fills_the_motif_columns_with_badges_and_a_changed_peak_set_clears_only_level3():
    app = _level3_app()
    with capture_downloads():
        app.button(key="tf_level3_prepare").click().run()
    planted = _FIXTURE["planted_tf"]
    peak_sets = _mouse_peak_sets()
    rows = [(f"{planted}(Zf)/Fixture/Homer", "N", "1e-12", "-27", "0.0002", "50", "40%", "800", "17%"),
            ("Oct4:Sox17(POU,Homeobox/HMG)/Fixture/Homer", "N", "1e-4", "-9", "0.3", "20", "16%", "500", "11%"),
            ("NotATf(Zf)/Fixture/Homer", "N", "0.1", "-2", "0.9", "5", "4%", "400", "9%")]
    state = mi.import_uploaded_result(
        None, peak_sets, homer_known_text(rows).encode("utf-8"), "known.txt", "homer_known", None, "opening",
        {"analysis_source": "other_file", "declared_thresholds": {"atac_padj": 0.01, "atac_lfc": 1.0}, "genome_attested": True,
         "background_choice": "tool_default"}, {planted, "Oct4", "Sox17"}, "2026-09-21T00:00:00")
    app.session_state["integration_motif_results"] = state
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    table = _table(app)
    row = table.set_index("tf_symbol").loc[planted]
    assert row["motif_opening_status"] == mi.STATUS_LE and row["motif_opening_padj"] == 0.0002
    assert row["motif_opening_motif_name"].startswith(f"{planted}(Zf)") and row["motif_opening_match_status"] == "matched"
    assert row["motif_opening_note"] == "threshold mismatch, other background" and row["motif_closing_status"] == mi.STATUS_NOT_IMPORTED
    assert row["supported / evaluable axes"].endswith("/ 3") and "motif_status" not in table.columns
    texts = " ".join(_texts(app))
    assert "The motif columns come from the result you imported" in texts and "Motif enrichment: not run" not in texts
    assert "differ from the current BRIM ATAC thresholds" in texts and "background other than BRIM" in texts
    assert "⚠ threshold mismatch, other background" in texts and "TFs found only in the motif result" in texts
    frames = [e.value for e in app.dataframe]
    assert any("unmatched_symbols" in f.columns for f in frames) and any(set(f.columns) >= {"tf_symbol", "peak_set", "status"} and
                                                                        set(f["tf_symbol"]) == {"Oct4", "Sox17"} for f in frames)
    # A changed ATAC result changes the peak set: Level 3 is cleared with a notice while Level 2 stays.
    app.session_state["atac_results"] = synthetic_dar_table().assign(padj=lambda d: d["padj"] * 0.5)
    app.run()
    assert not app.exception
    assert app.session_state["integration_motif_source"] is None and app.session_state["integration_motif_results"] is None
    assert app.session_state["integration_tf_results"] is not None
    assert any("Level 3 results were cleared" in text for text in _texts(app))
    assert "tf_level3" not in (app.session_state["integration_provenance"] or {})