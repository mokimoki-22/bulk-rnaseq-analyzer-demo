"""Result-bearing UI, semantic Export compatibility, and shared-generator wiring."""

from datetime import datetime
import hashlib
import io
import json
import zipfile

import pandas as pd
import pytest

import brim_provenance
from rna_support import FIXTURES, LEGACY_COLUMNS, capture_downloads, result_app


def flagged_results():
    """Attach tested flags to the recorded non-missing DEG fixture."""
    result = pd.read_csv(FIXTURES / "deg_results.csv", index_col=0)
    result["padj_is_na"] = False
    result["lfc_is_na"] = False
    return result


@pytest.fixture
def exported():
    app = result_app(flagged_results())
    with capture_downloads() as downloads:
        app.run()
    assert not app.exception
    assert not app.error
    with zipfile.ZipFile(io.BytesIO(downloads["results.zip"])) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    return app, downloads, contents


def test_export_keeps_original_zip_members_and_six_deg_columns(exported):
    _, _, contents = exported
    with zipfile.ZipFile(FIXTURES / "results.zip") as original:
        assert set(contents) == set(original.namelist()) | {"Provenance/manifest.json", "Provenance/manifest.md"}
        for name in original.namelist():
            if name.endswith(".csv"):
                actual = pd.read_csv(io.BytesIO(contents[name]), index_col=0)
                expected = pd.read_csv(io.BytesIO(original.read(name)), index_col=0)
                if name == "1.1_DEG_Results.csv":
                    assert set(actual) == set(LEGACY_COLUMNS) | {"padj_is_na", "lfc_is_na"}
                    assert not actual[["padj_is_na", "lfc_is_na"]].any().any()
                    actual = actual[LEGACY_COLUMNS]
                pd.testing.assert_frame_equal(actual, expected, rtol=1e-6, atol=1e-10)
            else:
                assert contents[name].decode().replace("\r\n", "\n") == original.read(name).decode().replace("\r\n", "\n")


def test_manifest_preserves_all_legacy_report_information(exported):
    _, _, contents = exported
    manifest = json.loads(contents["Provenance/manifest.json"])
    old = json.loads((FIXTURES / "reproducibility_report.json").read_text())
    assert set(old) == {"timestamp", "app_version", "species", "deg_parameters", "contrasts"}
    assert datetime.fromisoformat(manifest["environment"]["timestamp"]).utcoffset() is not None
    assert manifest["environment"]["app_version"] == old["app_version"]
    assert manifest["settings"]["species"] == old["species"]
    for name, value in old["deg_parameters"].items():
        assert manifest["settings"]["rna"][name] == value
    assert manifest["settings"]["rna"]["analysis_log"] == old["contrasts"]
    assert manifest["settings"]["rna"]["contrast"] == "treated vs control"
    assert manifest["inputs"]["rna"]["count_matrix"]["sha256"] == hashlib.sha256(contents["0.1_Raw_Counts.csv"]).hexdigest()
    assert manifest["counts"]["rna"]["na_counts"] == {"padj_is_na": 0, "lfc_is_na": 0}
    assert manifest["counts"]["rna"]["samples_by_condition"] == {"control": 3, "treated": 3}


def test_standalone_downloads_are_identical_to_zip_documents(exported):
    _, downloads, contents = exported
    for name in ("manifest.json", "manifest.md"):
        assert downloads[name].encode("utf-8") == contents[f"Provenance/{name}"]
    assert "reproducibility_report.json" not in downloads


def test_rna_export_uses_shared_generator_and_records_true_na_counts(monkeypatch):
    original = brim_provenance.build_manifest
    calls = []

    def tracked(**kwargs):
        manifest = original(**kwargs)
        manifest["environment"]["test_connection_marker"] = "shared generator result"
        calls.append(manifest)
        return manifest

    monkeypatch.setattr(brim_provenance, "build_manifest", tracked)
    result = flagged_results()
    result.loc["Gene_1", ["padj", "padj_is_na"]] = [1.0, True]
    result.loc["Gene_2", ["log2FoldChange", "lfc_is_na"]] = [0.0, True]
    result.loc["Gene_3", ["padj", "log2FoldChange", "padj_is_na", "lfc_is_na"]] = [1.0, 0.0, True, True]
    app = result_app(result)
    with capture_downloads() as downloads:
        app.run()
    assert not app.exception
    manifest = json.loads(downloads["manifest.json"])
    assert manifest in calls
    assert manifest["environment"]["test_connection_marker"] == "shared generator result"
    assert manifest["counts"]["rna"]["na_counts"] == {"padj_is_na": 2, "lfc_is_na": 2}


@pytest.mark.parametrize("section", ["DEG", "Visualization", "Export"])
def test_populated_primary_sections_render_their_contents(exported, section):
    app, downloads, _ = exported
    tab = next(tab for tab in app.tabs if section in tab.label)
    assert not tab.exception
    if section == "DEG":
        assert any(set(LEGACY_COLUMNS).issubset(frame.value.columns) for frame in tab.dataframe)
        values = {metric.label: int(metric.value) for metric in tab.metric if metric.label in ("Up", "Down")}
        assert values == {"Up": 0, "Down": 3}
    elif section == "Visualization":
        assert len(tab.get("plotly_chart")) >= 2
        assert {"Volcano", "MA Plot"}.issubset({child.label for child in tab.tabs})
    else:
        assert "results.zip" in downloads
        assert "manifest.json" in downloads


@pytest.mark.parametrize("language", ["English", "日本語"])
def test_populated_ui_reruns_without_losing_na_flags(language):
    app = result_app(flagged_results(), language)
    with capture_downloads():
        app.run()
        app.run()
    assert not app.exception
    pd.testing.assert_frame_equal(app.session_state["deg_results"], flagged_results())
