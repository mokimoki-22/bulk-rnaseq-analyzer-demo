"""Retain real request attempts independently of input acceptance and caches."""

import importlib
import io
import json
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import requests
import streamlit as st

from rna_support import capture_downloads, result_app, small_counts
from test_rna_export import flagged_results


@pytest.fixture(autouse=True)
def isolate_services():
    st.cache_data.clear()
    st.session_state["external_service_history"] = []
    yield
    st.cache_data.clear()


def mapping_response(genes, status=200):
    return SimpleNamespace(status_code=status, json=lambda: [
        {"query": gene, "symbol": gene} for gene in genes
    ])


@pytest.mark.parametrize("failure", ["timeout", "http", "json", "object", "entry", "symbol"])
def test_mapping_failure_is_journaled_before_request_and_on_cache_hit(monkeypatch, failure):
    module = importlib.import_module("Bulk_RNAseq_Analyzer")
    response = mapping_response(["gene"])
    if failure == "http":
        response.status_code = 503
    elif failure == "json":
        response.json = Mock(side_effect=ValueError("invalid JSON"))
    elif failure == "object":
        response.json = lambda: {"error": "not a result list"}
    elif failure == "entry":
        response.json = lambda: [None]
    elif failure == "symbol":
        response.json = lambda: [{"query": "gene", "symbol": None}]

    def post(*args, **kwargs):
        event = st.session_state["external_service_history"][-1]
        assert event["requests"][0]["outcome"] == "pending"
        assert kwargs["timeout"] == 30
        if failure == "timeout":
            raise requests.Timeout("do not store request data in exception text")
        return response

    mocked = Mock(side_effect=post)
    monkeypatch.setattr(requests, "post", mocked)
    assert module.run_online_mapping(["gene"], "mouse") == {}
    event = st.session_state["external_service_history"][0]
    assert event["lookup_outcome"] == "failed"
    assert event["access"] == "network"
    assert event["requests"][0]["outcome"] == "failed"
    assert event["requests"][0]["http_status"] == (None if failure == "timeout" else response.status_code)
    assert "do not store" not in json.dumps(event)
    assert module.run_online_mapping(["gene"], "mouse") == {}
    cached = st.session_state["external_service_history"][1]
    assert cached["lookup_outcome"] == "failed"
    assert cached["access"] == "cache"
    assert cached["requests"] == []
    mocked.assert_called_once()


def test_partial_mapping_and_failed_later_chunk_keep_all_attempts(monkeypatch):
    module = importlib.import_module("Bulk_RNAseq_Analyzer")
    genes = [f"id_{i}" for i in range(1001)]
    post = Mock(side_effect=[mapping_response(genes[:2]), requests.Timeout()])
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    assert module.run_online_mapping(genes, "mouse") == {gene: gene for gene in genes[:2]}
    event = st.session_state["external_service_history"][0]
    assert event["lookup_outcome"] == "partial"
    assert [request["outcome"] for request in event["requests"]] == ["success", "failed"]
    assert event["requests"][1]["error_type"] == "Timeout"
    assert post.call_count == 2


@pytest.mark.parametrize("failure", ["timeout", "http"])
def test_string_failure_remains_in_export_history_not_successful_services(monkeypatch, failure):
    post = Mock(side_effect=requests.Timeout()) if failure == "timeout" else Mock(
        return_value=SimpleNamespace(status_code=503, content=b"unavailable"))
    monkeypatch.setattr(requests, "post", post)
    app = result_app(flagged_results())
    with capture_downloads() as downloads:
        app.run()
        assert any("string-db.org" in item.value for item in app.info)
        post.assert_not_called()
        next(button for button in app.button if "Run STRING" in button.label).click().run()
        assert not app.exception
        assert app.error
        services = json.loads(downloads["manifest.json"])["services"]
    assert services["external_services_used"] == []
    assert services["events"][0]["lookup_outcome"] == "failed"
    assert services["external_service_events"] == services["events"]
    assert services["external_service_events"][0]["requests"][0]["outcome"] == "failed"


@pytest.mark.parametrize("case", ["invalid_count", "timeout", "http", "partial", "malformed"])
def test_single_input_outcome_and_service_outcome_are_independent(monkeypatch, case):
    counts, metadata = small_counts()
    incoming = counts.copy()
    if case == "invalid_count":
        incoming.iloc[0, 0] = -1
    stream = io.BytesIO(incoming.to_csv().encode())
    stream.name = "incoming.csv"
    uploader = st.file_uploader

    def upload(label, *args, **kwargs):
        if label == "Count Matrix":
            stream.seek(0)
            return stream
        return uploader(label, *args, **kwargs)

    monkeypatch.setattr(st, "file_uploader", upload)
    response = mapping_response(counts.index[:2] if case == "partial" else counts.index)
    if case == "http":
        response.status_code = 503
    if case == "malformed":
        response.json = lambda: {"error": "bad payload"}
    post = Mock(side_effect=requests.Timeout()) if case == "timeout" else Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)
    app = result_app(flagged_results())
    app.session_state["upload_mode"] = "single"
    app.session_state["gn_0"] = "control"
    app.session_state["gn_1"] = "treated"
    for sample in counts.columns:
        app.session_state[f"gs_{sample}"] = metadata.loc[sample, "condition"]
    with capture_downloads() as downloads:
        app.run()
        next(radio for radio in app.radio if "identifiers mode" in radio.label).set_value(
            "Gene IDs (convert to symbol)").run()
        assert any("mygene.info" in item.value for item in app.info)
        post.assert_not_called()
        next(button for button in app.button if button.label == "Load").click().run()
        assert not app.exception
        if case == "invalid_count":
            assert any("negative count" in error.value for error in app.error)
            pd.testing.assert_frame_equal(app.session_state["deg_results"], flagged_results())
            assert app.session_state["external_service_events"] == []
        else:
            assert not app.error
            app.session_state["deg_results"] = flagged_results()
            app.session_state["last_contrast"] = "treated vs control"
            app.run()
        pd.testing.assert_frame_equal(app.session_state["counts_df"], counts)
        manifest = json.loads(downloads["manifest.json"])
        event = manifest["services"]["external_service_events"][0]
        assert event["source"]["file_name"] == "incoming.csv"
        assert event["input_outcome"] == ("rejected" if case == "invalid_count" else "accepted")
        expected = "success" if case == "invalid_count" else "partial" if case == "partial" else "failed"
        assert event["lookup_outcome"] == expected
        if case == "partial":
            mapping = manifest["settings"]["rna"]["gene_id_mapping"][0]
            assert mapping["matched_ids"] == 2
            assert mapping["success_rate"] == 0.25
        assert manifest["services"]["external_services_used"] == (["mygene.info"] if case == "partial" else [])
        with zipfile.ZipFile(io.BytesIO(downloads["results.zip"])) as archive:
            assert archive.read("Provenance/manifest.json") == downloads["manifest.json"].encode()
        # A later successful input replacement resets current associations, not history.
        next(radio for radio in app.radio if "identifiers mode" in radio.label).set_value(
            "Gene symbol (Actb, GAPDH)").run()
        stream = io.BytesIO(counts.to_csv().encode())
        stream.name = "replacement.csv"
        next(button for button in app.button if button.label == "Load").click().run()
        assert not app.exception
        assert app.session_state["external_service_events"] == []
        assert app.session_state["external_service_history"] == [event]
        app.session_state["deg_results"] = flagged_results()
        app.run()
        assert json.loads(downloads["manifest.json"])["services"]["external_service_events"] == [event]
    post.assert_called_once()


@pytest.mark.parametrize("reuse_cache", [True, False])
def test_multistudy_rejection_keeps_prior_and_failed_study_lookups(monkeypatch, reuse_cache):
    counts, _ = small_counts()
    streams = []
    for number in range(2):
        incoming = counts.copy()
        if number == 1:
            incoming.iloc[0, 0] = -1
            if not reuse_cache:
                incoming.index = [f"second_{gene}" for gene in counts.index]
        stream = io.BytesIO(incoming.to_csv().encode())
        stream.name = f"study_{number}.csv"
        streams.append(stream)
    uploader = st.file_uploader

    def upload(label, *args, **kwargs):
        if kwargs.get("accept_multiple_files"):
            for stream in streams:
                stream.seek(0)
            return streams
        return uploader(label, *args, **kwargs)

    monkeypatch.setattr(st, "file_uploader", upload)
    post = Mock(side_effect=lambda *args, **kwargs: mapping_response(kwargs["data"]["q"].split(",")))
    monkeypatch.setattr(requests, "post", post)
    app = result_app(flagged_results())
    app.session_state["upload_mode"] = "multi"
    for number in range(2):
        app.session_state[f"study_idmode_{number}"] = "ensembl"
        for sample in counts.columns:
            app.session_state[f"study_gs_{number}_{sample}"] = "Control" if sample.startswith("control") else "Disease"
    with capture_downloads() as downloads:
        app.run()
        assert any("mygene.info" in item.value for item in app.info)
        post.assert_not_called()
        app.button(key="multi_load_btn").click().run()
        assert not app.exception
        assert any("negative count" in error.value for error in app.error)
        events = app.session_state["external_service_history"]
        assert len(events) == 2
        assert [event["input_outcome"] for event in events] == ["rejected", "rejected"]
        assert [event["access"] for event in events] == ["network", "cache" if reuse_cache else "network"]
        assert [event["source"]["file_name"] for event in events] == ["study_0.csv", "study_1.csv"]
        assert all(event["lookup_outcome"] == "success" for event in events)
        pd.testing.assert_frame_equal(app.session_state["counts_df"], counts)
        pd.testing.assert_frame_equal(app.session_state["deg_results"], flagged_results())
        # The failing action calls st.stop; a normal rerun reaches Export again.
        app.run()
        manifest = json.loads(downloads["manifest.json"])
        assert manifest["services"]["external_service_events"] == events
        assert manifest["services"]["external_services_used"] == []
        assert manifest["services"]["events"] == []
        with zipfile.ZipFile(io.BytesIO(downloads["results.zip"])) as archive:
            assert archive.read("Provenance/manifest.json") == downloads["manifest.json"].encode()
    assert post.call_count == (1 if reuse_cache else 2)
