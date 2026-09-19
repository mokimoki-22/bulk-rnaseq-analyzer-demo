"""Regression coverage for populated legacy Meta, TF, and interaction views."""

import pandas as pd

from rna_support import result_app
from test_rna_export import flagged_results


def _interaction_inputs():
    samples = [f"{condition}_{number}" for condition in ("control", "treated") for number in range(1, 5)]
    counts = pd.DataFrame(
        {
            sample: [80 + number * 4, 140 - number * 3]
            for number, sample in enumerate(samples, start=1)
        },
        index=["Gene_1", "Gene_2"],
    )
    metadata = pd.DataFrame(
        {
            "condition": ["control"] * 4 + ["treated"] * 4,
            "age": [20.0, 30.0, 40.0, 50.0] * 2,
        },
        index=samples,
    )
    result = pd.DataFrame(
        {
            "baseMean": [100.0, 120.0],
            "log2FoldChange": [1.2, -0.2],
            "lfcSE": [0.2, 0.3],
            "stat": [4.0, -0.7],
            "pvalue": [0.001, 0.4],
            "padj": [0.01, 0.8],
            "padj_is_na": [False, False],
            "lfc_is_na": [False, False],
        },
        index=counts.index,
    )
    return counts, metadata, result


def _interaction_app():
    from streamlit.testing.v1 import AppTest

    from rna_support import ROOT

    counts, metadata, result = _interaction_inputs()
    app = AppTest.from_file(str(ROOT / "Bulk_RNAseq_Analyzer.py"), default_timeout=60)
    state = {
        "counts_df": counts,
        "qc_filtered_df": counts.copy(),
        "metadata": metadata,
        "conditions": ["control", "treated"],
        "deg_results": result.copy(),
        "last_contrast": "treated vs control",
        "sp": {"org": "mmu", "string_id": 10090},
        "lang_display": "English",
        "language_selector": "English",
        "lfc_t": 0.7,
        "padj_t": 0.1,
        "deg_t": (0.7, 0.1),
        "norm_method": "log1p",
        "filter_enable": True,
        "filter_min_count": 1,
        "filter_min_samples": 1,
        "analysis_log": [],
        "ia_var_sel": "age",
        "ia_var_type": "Continuous (numeric)",
        "ia_ref_sel": "control",
        "ia_test_sel": "treated",
    }
    for key, value in state.items():
        app.session_state[key] = value
    return app, result


def test_meta_view_renders_populated_independent_studies():
    first = flagged_results().copy()
    second = flagged_results().copy()
    first.loc[:, "pvalue"] = [0.001] * len(first)
    second.loc[:, "pvalue"] = [0.002] * len(second)
    first.loc[:, "padj"] = [0.01] * len(first)
    second.loc[:, "padj"] = [0.02] * len(second)
    first.loc[:, "log2FoldChange"] = [1.5] * len(first)
    second.loc[:, "log2FoldChange"] = [1.3] * len(second)

    app = result_app(first)
    app.session_state["batch_deg_results"] = {"study_a": first, "study_b": second}
    app.session_state["batch_deg_provenance"] = {"study_a": "study-a", "study_b": "study-b"}
    app.session_state["multi_study_names"] = ["study_a", "study_b"]

    app.run()
    app.button(key="meta_plot_btn").click().run()

    assert not app.exception
    matrix = app.session_state["lfc_meta_matrix"]
    assert list(matrix.columns) == ["study_a", "study_b", "meta_padj"]
    assert matrix["meta_padj"].notna().all()


def test_tf_view_renders_populated_collectri_and_dorothea_results():
    app = result_app(flagged_results())
    samples = app.session_state["counts_df"].columns
    app.session_state["tf_collectri"] = pd.DataFrame(
        {"Stat3": [0.4] * len(samples), "Fos": [-0.2] * len(samples), "Myc": [0.1] * len(samples)},
        index=samples,
    )
    app.session_state["tf_dorothea"] = pd.DataFrame(
        {"Stat3": [0.3] * len(samples), "Fos": [-0.1] * len(samples), "Myc": [0.2] * len(samples)},
        index=samples,
    )

    app.run()

    assert not app.exception
    tab_labels = [tab.label for tab in app.tabs]
    assert "CollecTRI" in tab_labels
    assert "DoRothEA" in tab_labels
    assert "Consensus" in tab_labels


def test_interaction_run_preserves_na_flags_and_renders_results(monkeypatch):
    from pydeseq2 import dds, ds

    raw = pd.DataFrame(
        {
            "baseMean": [100.0, 120.0],
            "log2FoldChange": [float("nan"), -0.8],
            "lfcSE": [0.2, 0.3],
            "stat": [float("nan"), -2.0],
            "pvalue": [float("nan"), 0.01],
            "padj": [float("nan"), 0.02],
        },
        index=["Gene_1", "Gene_2"],
    )

    class FakeDeseqDataSet:
        def __init__(self, **kwargs):
            self.obsm = {"design_matrix": pd.DataFrame(columns=[
                "Intercept", "age", "condition[T.treated]", "age:condition[T.treated]",
            ])}

        def deseq2(self):
            return None

    class FakeDeseqStats:
        def __init__(self, *args, **kwargs):
            self.results_df = raw.copy()

        def summary(self):
            return None

    monkeypatch.setattr(dds, "DeseqDataSet", FakeDeseqDataSet)
    monkeypatch.setattr(ds, "DeseqStats", FakeDeseqStats)
    app, _ = _interaction_app()

    app.run()
    assert not app.exception
    app.button(key="ia_run_btn").click().run()

    assert not app.exception
    result = app.session_state["ia_results"]
    assert result["padj_is_na"].tolist() == [True, False]
    assert result["lfc_is_na"].tolist() == [True, False]
    assert result.loc["Gene_1", "padj"] == 1.0
    assert result.loc["Gene_1", "log2FoldChange"] == 0.0
