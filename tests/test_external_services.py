"""Exercise user actions, notices, and service provenance with mocked HTTP."""

import base64
import hashlib
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from rna_support import ROOT, capture_downloads, result_app, small_counts
from test_rna_export import flagged_results


@pytest.fixture(autouse=True)
def clear_service_caches():
    # AppTest executes the script as __main__, whose cache is distinct from a
    # regular import of Bulk_RNAseq_Analyzer. Clear both via the public API.
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def test_string_notice_precedes_explicit_action_and_manifest_records_cached_use(monkeypatch):
    image = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII=")
    post = Mock(return_value=SimpleNamespace(status_code=200, content=image))
    monkeypatch.setattr(requests, "post", post)
    app = result_app(flagged_results())
    with capture_downloads() as downloads:
        app.run()
        assert not app.exception
        assert any("gene list" in item.value and "string-db.org" in item.value for item in app.info)
        post.assert_not_called()
        button = next(item for item in app.button if "Run STRING" in item.label)
        button.click().run()
        assert not app.exception
        assert not app.error
        post.assert_called_once()
        assert post.call_args.args[0] == "https://string-db.org/api/image/network"
        manifest = json.loads(downloads["manifest.json"])
        assert manifest["services"]["external_services_used"] == ["string-db.org"]
        assert manifest["services"]["events"][0]["data_type"] == "gene list"
        assert downloads["string_network.png"] == image
        # A rerun is not another service action; an explicit cached lookup is.
        app.run()
        assert len(app.session_state["external_service_events"]) == 1
        next(item for item in app.button if "Run STRING" in item.label).click().run()
        assert not app.exception
        assert len(app.session_state["external_service_events"]) == 2
        post.assert_called_once()


def test_single_upload_mapping_notice_original_sha_and_export_record(monkeypatch):
    counts, metadata = small_counts()
    source = counts.to_csv(lineterminator="\r\n").encode()
    uploaded = io.BytesIO(source)
    uploaded.name = "original.csv"
    original_uploader = st.file_uploader

    def upload(label, *args, **kwargs):
        if label == "Count Matrix":
            return uploaded
        return original_uploader(label, *args, **kwargs)

    monkeypatch.setattr(st, "file_uploader", upload)
    post = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: [
        {"query": gene, "symbol": gene} for gene in counts.index
    ]))
    monkeypatch.setattr(requests, "post", post)
    app = AppTest.from_file(str(ROOT / "Bulk_RNAseq_Analyzer.py"), default_timeout=60)
    for key, value in {"upload_mode": "single", "lang_display": "English", "language_selector": "English",
                       "gn_0": "control", "gn_1": "treated"}.items():
        app.session_state[key] = value
    for sample in counts.columns:
        app.session_state[f"gs_{sample}"] = metadata.loc[sample, "condition"]
    with capture_downloads() as downloads:
        app.run()
        assert not app.exception
        id_radio = next(item for item in app.radio if "identifiers mode" in item.label)
        id_radio.set_value("Gene IDs (convert to symbol)").run()
        assert any("gene IDs" in item.value and "mygene.info" in item.value for item in app.info)
        post.assert_not_called()
        next(item for item in app.button if item.label == "Load").click().run()
        assert not app.exception
        post.assert_called_once()
        assert post.call_args.args[0] == "https://mygene.info/v3/query"
        assert app.session_state["rna_input_files"] == [
            {"file_name": "original.csv", "sha256": hashlib.sha256(source).hexdigest()}
        ]
        assert app.session_state["rna_id_mapping"][0]["success_rate"] == 1.0
        pd.testing.assert_frame_equal(app.session_state["counts_df"], counts)
        app.session_state["deg_results"] = flagged_results()
        app.session_state["last_contrast"] = "treated vs control"
        app.run()
        assert not app.exception
        manifest = json.loads(downloads["manifest.json"])
        assert manifest["services"]["external_services_used"] == ["mygene.info"]
        assert manifest["inputs"]["rna"]["source_files"][0]["sha256"] == hashlib.sha256(source).hexdigest()
        assert manifest["settings"]["rna"]["gene_id_mapping"][0]["matched_ids"] == 8
        # Loading a new symbol matrix must not inherit the previous service use.
        next(item for item in app.radio if "identifiers mode" in item.label).set_value("Gene symbol (Actb, GAPDH)").run()
        next(item for item in app.button if item.label == "Load").click().run()
        assert not app.exception
        assert app.session_state["external_service_events"] == []
        assert app.session_state["rna_id_mapping"] == []
        post.assert_called_once()


def test_multistudy_mapping_notice_and_sources_survive_successful_load(monkeypatch):
    counts, _ = small_counts()
    files = []
    for name in ("study_A.csv", "study_B.csv"):
        stream = io.BytesIO(counts.to_csv().encode())
        stream.name = name
        files.append(stream)
    original_uploader = st.file_uploader

    def upload(label, *args, **kwargs):
        if kwargs.get("accept_multiple_files"):
            return files
        return original_uploader(label, *args, **kwargs)

    monkeypatch.setattr(st, "file_uploader", upload)
    post = Mock(return_value=SimpleNamespace(status_code=200, json=lambda: [
        {"query": gene, "symbol": gene} for gene in counts.index
    ]))
    monkeypatch.setattr(requests, "post", post)
    app = AppTest.from_file(str(ROOT / "Bulk_RNAseq_Analyzer.py"), default_timeout=60)
    app.session_state["upload_mode"] = "multi"
    app.session_state["lang_display"] = "English"
    app.session_state["language_selector"] = "English"
    for number in range(2):
        app.session_state[f"study_idmode_{number}"] = "ensembl"
        for sample in counts.columns:
            app.session_state[f"study_gs_{number}_{sample}"] = "Control" if sample.startswith("control") else "Disease"
    with capture_downloads() as downloads:
        app.run()
        assert not app.exception
        assert any("mygene.info" in item.value for item in app.info)
        post.assert_not_called()
        app.button(key="multi_load_btn").click().run()
        assert not app.exception
        assert len(app.session_state["rna_input_files"]) == 2
        assert len(app.session_state["rna_id_mapping"]) == 2
        assert len(app.session_state["external_service_events"]) == 2
        # Identical IDs/species use the original cache on the second study.
        post.assert_called_once()
        app.session_state["deg_results"] = flagged_results()
        app.session_state["last_contrast"] = "[study_A] Disease vs [study_A] Control"
        app.run()
        assert not app.exception
        manifest = json.loads(downloads["manifest.json"])
        assert manifest["services"]["external_services_used"] == ["mygene.info"]
        assert len(manifest["inputs"]["rna"]["source_files"]) == 2
