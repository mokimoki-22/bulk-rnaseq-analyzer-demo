"""Phase 5 Level 2 core tests (Streamlit-free), built up step by step.

Step 1: universe, gene sets, target sets, Fisher, BH and the synthetic positive/negative controls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact

import brim_integration_enrichment as ora
import brim_tf_integration as tf
from tests.tf_support import permuted_summary, synthetic_tf_level1_state


def _summary():
    return pd.DataFrame({
        "gene_symbol": ["CA", "CR", "NS", "MIX", "RNA_NA", "ATAC_NA", "UNMAPPED", "UNMAPPED_NS"],
        "integration_class": ["concordant_activation", "concordant_repression", "not_significant",
                              "mixed_accessibility", "rna_not_tested", "atac_not_tested",
                              "rna_only_no_mapped_peak", "not_significant"],
        "rna_padj_is_na": [False, False, False, False, True, False, False, False],
        "rna_lfc_is_na": [False] * 8,
        "n_mapped_peaks": [1, 1, 2, 1, 1, 1, 0, 0],
        "n_atac_tested_peaks": [1, 1, 2, 1, 1, 0, 0, 0],
    })


def _network(rows):
    return pd.DataFrame(rows, columns=["source", "target"])


@pytest.fixture(scope="module")
def state():
    return synthetic_tf_level1_state()


def test_universe_equals_ora_background_and_excludes_unclassifiable_genes():
    universe = tf.build_universe(_summary())
    assert universe == ora.build_ora_background(_summary()) == ["CA", "CR", "MIX", "NS"]
    for excluded in ("RNA_NA", "ATAC_NA", "UNMAPPED", "UNMAPPED_NS"):
        assert excluded not in universe
    assert "NS" in universe


def test_universe_description_reports_size_and_literal_definition_counts():
    described = tf.describe_universe(_summary())
    assert described["universe_size"] == 4
    assert described["n_mapped_rna_tested"] == 5          # CA, CR, NS, MIX, ATAC_NA
    assert described["n_excluded_atac_not_tested"] == 1   # ATAC_NA
    assert described["n_rna_only_no_mapped_peak"] == 1 and described["n_rna_not_tested"] == 1
    assert described["universe_definition"] and described["universe_definition_ja"]


def test_gene_sets_are_unions_restricted_to_the_universe_and_count_removed_genes():
    summary = _summary()
    universe = tf.build_universe(summary)
    assert tf.resolve_gene_set(summary, "concordant_all", universe) == (["CA", "CR"], 0)
    assert tf.resolve_gene_set(summary, "mixed_accessibility", universe) == (["MIX"], 0)
    assert tf.resolve_gene_set(summary, "concordant_activation", ["CR"]) == ([], 1)
    with pytest.raises(tf.TFIntegrationError):
        tf.resolve_gene_set(summary, "not_a_set", universe)
    assert set(tf.SET_DEFINITIONS) >= {"concordant_all", "discordant_all", "mixed_accessibility"}


def test_fisher_one_sided_known_table_and_bh_known_vector():
    universe = [f"G{i}" for i in range(8)]
    gene_set = ["G0", "G1", "G2", "G3"]
    network = _network([("T", "G0"), ("T", "G1"), ("T", "G2"), ("T", "G4")])
    result = tf.test_target_enrichment(gene_set, universe, network, min_targets=1, network_source="collectri")
    row = result.iloc[0]
    assert (row.n_targets_in_set, row.n_targets_in_universe) == (3, 4)
    assert row.target_enrichment_p == pytest.approx(17 / 70)
    assert fisher_exact([[3, 1], [1, 3]], alternative="two-sided")[1] == pytest.approx(0.485714, abs=1e-5)
    assert row.target_enrichment_p != pytest.approx(0.485714, abs=1e-3)
    assert row.fisher_alternative == "greater"
    assert tf.benjamini_hochberg([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    p_values = np.random.default_rng(1).random(50)
    adjusted = tf.benjamini_hochberg(p_values)
    assert (adjusted >= p_values).all() and (adjusted <= 1).all()
    assert (np.diff(adjusted[np.argsort(p_values)]) >= 0).all()
    with pytest.raises(tf.TFIntegrationError):
        tf.benjamini_hochberg([0.5, 1.5])


def test_bh_is_monotone_in_p_and_family_is_only_the_tested_tfs():
    universe = [f"G{i}" for i in range(20)]
    gene_set = [f"G{i}" for i in range(5)]
    network = _network(
        [("BIG", f"G{i}") for i in range(6)] + [("MID", f"G{i}") for i in range(3, 9)]
        + [("ZERO", f"G{i}") for i in range(10, 16)] + [("TINY", "G1"), ("TINY", "G2")]
    )
    result = tf.test_target_enrichment(gene_set, universe, network, min_targets=3, network_source="collectri")
    assert set(result.tf_symbol) == {"BIG", "MID", "ZERO"}          # TINY has fewer than min_targets
    assert result.attrs["n_tests"] == 3 == int(result.n_tests.iloc[0])
    assert result.attrs["n_tfs_below_min_targets"] == 1
    zero = result.set_index("tf_symbol").loc["ZERO"]
    assert zero.n_targets_in_set == 0 and zero.target_enrichment_p == pytest.approx(1.0)
    ordered = result.sort_values("target_enrichment_p")
    assert ordered["target_enrichment_padj"].is_monotonic_increasing
    assert "BH" in result.attrs["bh_scope_note"] or "Benjamini" in result.attrs["bh_scope_note"]


def test_results_for_different_gene_sets_are_independent():
    universe = [f"G{i}" for i in range(20)]
    network = _network([("T1", f"G{i}") for i in range(6)] + [("T2", f"G{i}") for i in range(10, 16)])
    a = tf.test_target_enrichment([f"G{i}" for i in range(5)], universe, network, 3, "collectri")
    b = tf.test_target_enrichment([f"G{i}" for i in range(10, 15)], universe, network, 3, "collectri")
    a2 = tf.test_target_enrichment([f"G{i}" for i in range(5)], universe, network, 3, "collectri")
    pd.testing.assert_frame_equal(a, a2)
    assert a.set_index("tf_symbol").loc["T1", "target_enrichment_p"] != b.set_index("tf_symbol").loc["T1", "target_enrichment_p"]


def test_targets_and_gene_set_are_restricted_to_the_universe():
    universe = ["A", "B", "C", "D"]
    network = _network([("T", "A"), ("T", "B"), ("T", "C"), ("T", "OUTSIDE1"), ("T", "OUTSIDE2")])
    result = tf.test_target_enrichment(["A", "B"], universe, network, 1, "collectri")
    assert result.iloc[0].n_targets_in_universe == 3
    with pytest.raises(tf.TFIntegrationError):
        tf.test_target_enrichment(["A", "OUTSIDE1"], universe, network, 1, "collectri")


def test_symbols_are_matched_with_strip_and_casefold_and_the_match_is_reported():
    network = _network([("Tf1", "Myc"), ("Tf1", "Trp53"), ("Tf1", "Other"), ("Tf2", " trp53 "), ("Tf2", "TRP53")])
    targets, report = tf.build_target_sets(network, ["MYC", "TRP53", "NOT_IN_NET"], min_targets=1)
    assert targets["Tf1"] == frozenset({"MYC", "TRP53"})
    assert report["matching"] == "strip+casefold"
    assert report["n_universe_symbols"] == 3 and report["n_universe_symbols_in_network"] == 2
    assert report["n_network_targets_matched"] == 2
    assert report["n_casefold_collisions_network"] == 1     # " trp53 " and "TRP53" and "Trp53" share a folded key
    assert report["n_casefold_collisions_universe"] == 0
    _, collision = tf.build_target_sets(network, ["Myc", "MYC"], min_targets=1)
    assert collision["n_casefold_collisions_universe"] == 1
    assert tf.build_target_sets(network, ["Myc", "MYC"])[0]["Tf1"] == frozenset({"Myc", "MYC"})


def test_zero_symbol_match_stops_with_an_actionable_error():
    with pytest.raises(tf.TFIntegrationError, match="species"):
        tf.build_target_sets(_network([("T", "A")]), ["X", "Y"], min_targets=1)
    with pytest.raises(tf.TFIntegrationError):
        tf.build_target_sets(_network([("T", "A")]), ["A"], min_targets=0)
    with pytest.raises(tf.TFIntegrationError):
        tf.build_target_sets(pd.DataFrame({"source": ["T"]}), ["A"], min_targets=1)


@pytest.mark.parametrize("size, expected", [(1, True), (19, True), (20, False)])
def test_small_gene_sets_are_flagged_and_empty_sets_produce_no_rows(size, expected):
    universe = [f"G{i}" for i in range(40)]
    network = _network([("T", f"G{i}") for i in range(12)])
    result = tf.test_target_enrichment(universe[:size], universe, network, 3, "collectri")
    assert bool(result.small_gene_set_warning.iloc[0]) is expected
    empty = tf.test_target_enrichment([], universe, network, 3, "collectri")
    assert len(empty) == 0 and list(empty.columns)[0] == "tf_symbol"


def test_synthetic_state_is_production_shaped_and_deterministic(state):
    again = synthetic_tf_level1_state()
    pd.testing.assert_frame_equal(state["summary"], again["summary"])
    summary = state["summary"]
    assert len(summary) == 401      # 400 mapped genes plus the planted TF's own unmapped RNA row
    assert (summary["integration_class"] == "concordant_activation").sum() == 40
    assert {"rna_padj_is_na", "rna_lfc_is_na", "n_atac_tested_peaks", "n_mapped_peaks"} <= set(summary.columns)
    assert state["planted_tf"] not in set(summary.loc[summary["n_mapped_peaks"] > 0, "gene_symbol"])
    universe = tf.build_universe(summary)
    assert 0 < len(universe) < 400 and set(state["planted_targets"][:25]) <= set(universe)


def test_planted_tf_is_enriched_in_the_concordant_activation_set(state):
    summary = state["summary"]
    universe = tf.build_universe(summary)
    genes, removed = tf.resolve_gene_set(summary, "concordant_activation", universe)
    assert removed == 0 and len(genes) == 40
    result = tf.test_target_enrichment(genes, universe, state["network"], 10, "collectri",
                                       set_name="concordant_activation")
    planted = result.set_index("tf_symbol").loc[state["planted_tf"]]
    assert planted.target_enrichment_padj < 0.05 and planted.n_targets_in_set >= 25


def test_negative_control_has_no_enriched_tf_for_the_fixed_seed(state):
    """Regression check for seed 0 only; not a general false-positive guarantee (plan section 5.6)."""
    summary = permuted_summary(state["summary"])
    universe = tf.build_universe(summary)
    assert universe == tf.build_universe(state["summary"])
    for set_name in ("concordant_activation", "concordant_all"):
        genes, _ = tf.resolve_gene_set(summary, set_name, universe)
        result = tf.test_target_enrichment(genes, universe, state["network"], 10, "collectri", set_name=set_name)
        assert (result["target_enrichment_padj"] <= 0.05).sum() == 0, set_name


# ----------------------------------------------------------------------------------------------
# Step 2: evidence axes and display-only axis counts
# ----------------------------------------------------------------------------------------------

RNA_T = {"rna_padj": 0.05, "rna_lfc": 1.0}


def _rna(rows):
    import brim_multiomics as multi

    frame = pd.DataFrame(rows, columns=["gene", "log2FoldChange", "padj", "padj_is_na", "lfc_is_na"]).set_index("gene")
    return multi.standardize_rna_results(frame, "gene_symbol")


def _tfs(*symbols):
    return pd.DataFrame({"tf_symbol": list(symbols)})


def test_expression_axis_keeps_na_as_not_tested_and_applies_level1_rules():
    rna = _rna([("UP", 2.0, 0.01, False, False), ("DOWN", -2.0, 0.01, False, False),
                ("EDGE", 1.0, 0.05, False, False), ("ONE", 1.0, 1.0, False, False),
                ("NAPADJ", 3.0, 1.0, True, False), ("NALFC", 0.0, 0.001, False, True),
                ("ZERO", 0.0, 0.001, False, False), ("Dup", 2.0, 0.01, False, False), ("DUP", 2.0, 0.01, False, False)])
    table = tf.attach_tf_expression(_tfs("up", "DOWN", "EDGE", "ONE", "NAPADJ", "NALFC", "ZERO", "GONE", "dup"), rna, RNA_T)
    status = dict(zip(table.tf_symbol, table.tf_expression_status))
    assert status == {"up": "supported_up", "DOWN": "supported_down", "EDGE": "supported_up",
                      "ONE": "not_significant", "NAPADJ": "not_tested", "NALFC": "not_tested",
                      "ZERO": "not_significant", "GONE": "not_in_rna_results", "dup": "ambiguous_symbol"}
    by_tf = table.set_index("tf_symbol")
    na = by_tf.loc["NAPADJ"]
    assert pd.isna(na.tf_rna_padj) and pd.isna(na.tf_rna_log2FoldChange) and bool(na.tf_rna_padj_is_na)
    assert by_tf.loc["ONE", "tf_rna_padj"] == 1.0 and not bool(by_tf.loc["ONE", "tf_rna_padj_is_na"])
    assert pd.isna(by_tf.loc["GONE", "tf_rna_padj"]) and pd.isna(by_tf.loc["GONE", "tf_rna_padj_is_na"])
    zero_threshold = tf.attach_tf_expression(_tfs("ZERO"), rna, {"rna_padj": 0.05, "rna_lfc": 0.0})
    assert zero_threshold.tf_expression_status.iloc[0] == "not_significant"       # lfc == 0 has no direction


def test_expression_axis_validates_inputs():
    rna = _rna([("A", 2.0, 0.01, False, False)])
    with pytest.raises(tf.TFIntegrationError):
        tf.attach_tf_expression(_tfs("A"), rna.drop(columns=["gene_key"]), RNA_T)
    with pytest.raises(tf.TFIntegrationError):
        tf.attach_tf_expression(_tfs("A"), rna, {"rna_padj": 0.05})
    with pytest.raises(tf.TFIntegrationError):
        tf.attach_tf_expression(pd.DataFrame({"x": [1]}), rna, RNA_T)


def _activity(**columns):
    samples = ["c1", "c2", "c3", "t1", "t2", "t3"]
    return pd.DataFrame(columns, index=samples)


CONDITIONS = pd.Series(["ctl", "ctl", "ctl", "trt", "trt", "trt"], index=["c1", "c2", "c3", "t1", "t2", "t3"])


def test_activity_axis_is_a_descriptive_separation_rule_without_p_values():
    scores = _activity(UPTF=[0, 0.1, 0.2, 1, 1.1, 1.2], DOWNTF=[1, 1.1, 1.2, 0, 0.1, 0.2],
                       OVERLAP=[0, 1, 2, 1, 2, 3], TIE=[0, 1, 2, 2, 3, 4], WITHNAN=[0, 1, 2, np.nan, 3, 4])
    table = tf.attach_tf_activity(_tfs("UPTF", "DOWNTF", "OVERLAP", "TIE", "WITHNAN", "ABSENT"), scores,
                                  CONDITIONS, "ctl", "trt")
    status = dict(zip(table.tf_symbol, table.tf_activity_status))
    assert status == {"UPTF": "separated_up", "DOWNTF": "separated_down", "OVERLAP": "not_separated",
                      "TIE": "not_separated", "WITHNAN": "not_estimated", "ABSENT": "not_estimated"}
    by_tf = table.set_index("tf_symbol")
    assert by_tf.loc["UPTF", "tf_activity_score"] == pytest.approx(1.0)
    assert pd.isna(by_tf.loc["WITHNAN", "tf_activity_score"])
    assert by_tf.loc["UPTF", "tf_activity_n_ref"] == 3 and by_tf.loc["UPTF", "tf_activity_n_test"] == 3
    assert by_tf.loc["UPTF", "tf_activity_source"] == "collectri"
    assert not [c for c in table.columns if "pval" in c.lower() or c.lower().endswith("_p") or "padj" in c.lower()]


def test_activity_axis_not_run_insufficient_samples_and_ignored_samples():
    scores = _activity(TF=[0, 0.1, 0.2, 1, 1.1, 1.2])
    assert set(tf.attach_tf_activity(_tfs("TF"), None, CONDITIONS, "ctl", "trt").tf_activity_status) == {"not_run"}
    assert set(tf.attach_tf_activity(_tfs("TF"), scores, None, "ctl", "trt").tf_activity_status) == {"not_run"}
    missing = scores.drop(index=["t3"])                      # a contrast sample is absent from the matrix
    assert set(tf.attach_tf_activity(_tfs("TF"), missing, CONDITIONS, "ctl", "trt").tf_activity_status) == {"not_run"}
    assert set(tf.attach_tf_activity(_tfs("TF"), scores, CONDITIONS, "ctl", "other").tf_activity_status) == {"not_run"}
    two_vs_two = pd.Series(["ctl", "ctl", "x", "trt", "trt", "x"], index=CONDITIONS.index)
    small = tf.attach_tf_activity(_tfs("TF"), scores, two_vs_two, "ctl", "trt")
    assert set(small.tf_activity_status) == {"insufficient_samples"} and small.tf_activity_n_samples_ignored.iloc[0] == 2
    padded = scores.copy()
    padded.loc["extra"] = [9.0]
    extra = tf.attach_tf_activity(_tfs("TF"), padded, CONDITIONS, "ctl", "trt")
    assert extra.tf_activity_status.iloc[0] == "separated_up" and extra.tf_activity_n_samples_ignored.iloc[0] == 1


def test_motif_axis_is_an_explicit_not_run_marker():
    table = tf.add_motif_placeholder(_tfs("A", "B"))
    assert set(table.motif_status) == {"not_run"} and table.motif_enrichment_padj.isna().all()
    assert table.motif_enrichment_score.isna().all() and set(table.motif_source) == {""}


def _axis_table(**overrides):
    base = {"tf_symbol": ["A"], "target_enrichment_padj": [0.5], "n_targets_in_set": [3],
            "tf_expression_status": ["not_significant"], "tf_activity_status": ["not_separated"]}
    base.update({key: [value] for key, value in overrides.items()})
    return pd.DataFrame(base)


def test_axis_count_changes_by_exactly_one_when_a_single_axis_flips():
    off = tf.count_supported_axes(_axis_table())
    assert off.n_axes_supported.iloc[0] == 0 and off.n_axes_evaluable.iloc[0] == 3
    flips = [{"target_enrichment_padj": 0.01}, {"tf_expression_status": "supported_up"},
             {"tf_expression_status": "supported_down"}, {"tf_activity_status": "separated_up"},
             {"tf_activity_status": "separated_down"}]
    for flip in flips:
        assert tf.count_supported_axes(_axis_table(**flip)).n_axes_supported.iloc[0] == 1, flip
    everything = tf.count_supported_axes(_axis_table(target_enrichment_padj=0.01, tf_expression_status="supported_up",
                                                     tf_activity_status="separated_up"))
    assert everything.n_axes_supported.iloc[0] == 3
    no_hit = tf.count_supported_axes(_axis_table(target_enrichment_padj=0.01, n_targets_in_set=0))
    assert no_hit.n_axes_supported.iloc[0] == 0


def test_not_run_and_not_estimable_axes_are_neither_supported_nor_evaluable():
    for status in ("not_tested", "not_in_rna_results", "ambiguous_symbol"):
        row = tf.count_supported_axes(_axis_table(tf_expression_status=status)).iloc[0]
        assert row.n_axes_supported == 0 and row.n_axes_evaluable == 2, status
    for status in ("not_run", "not_estimated", "insufficient_samples"):
        row = tf.count_supported_axes(_axis_table(tf_activity_status=status)).iloc[0]
        assert row.n_axes_supported == 0 and row.n_axes_evaluable == 2, status
    with_motif = tf.count_supported_axes(tf.add_motif_placeholder(_axis_table()))
    assert with_motif.n_axes_evaluable.iloc[0] == 3           # the motif marker never counts


def test_axes_are_display_only_no_combined_score_columns_and_deterministic_sort():
    frame = pd.concat([
        _axis_table(tf_symbol="B", target_enrichment_padj=0.02, tf_expression_status="supported_up"),
        _axis_table(tf_symbol="A", target_enrichment_padj=0.02, tf_expression_status="supported_up"),
        _axis_table(tf_symbol="C", target_enrichment_padj=0.001),
        _axis_table(tf_symbol="D", target_enrichment_padj=0.5),
    ], ignore_index=True)
    result = tf.count_supported_axes(tf.add_motif_placeholder(frame))
    assert list(result.tf_symbol) == ["A", "B", "C", "D"]      # 2, 2 (tie by padj then symbol), 1, 0
    assert list(result.n_axes_supported) == [2, 2, 1, 0]
    forbidden = ("score", "confidence", "combined", "weighted")
    offenders = [c for c in result.columns if any(word in c.lower() for word in forbidden)
                 and c not in ("tf_activity_score", "motif_enrichment_score")]
    assert offenders == []
    counted = ["n_axes_supported", "n_axes_evaluable", "axis_target_enrichment_supported",
               "axis_expression_supported", "axis_activity_supported"]
    plain = tf.add_motif_placeholder(frame).sort_values("tf_symbol").reset_index(drop=True)
    pd.testing.assert_frame_equal(result.drop(columns=counted).sort_values("tf_symbol").reset_index(drop=True), plain)
    with pytest.raises(tf.TFIntegrationError):
        tf.count_supported_axes(frame.drop(columns=["tf_activity_status"]))
    with pytest.raises(tf.TFIntegrationError):
        tf.count_supported_axes(frame, alpha=2)

# ----------------------------------------------------------------------------------------------
# Step 3: fingerprints, one Level 2 run, manifest block, drill-down and module boundaries
# ----------------------------------------------------------------------------------------------

def _run(state, summary=None, set_name="concordant_activation", activity=True, **overrides):
    arguments = dict(
        summary=state["summary"] if summary is None else summary, set_name=set_name, network=state["network"],
        rna_results=state["rna_results"], thresholds=state["thresholds"], contrasts={"reference": "control", "test": "treated"},
        activity_scores=state["activity_scores"] if activity else None,
        sample_conditions=state["metadata"]["condition"] if activity else None, min_targets=10, alpha=0.05,
    )
    arguments.update(overrides)
    return tf.run_level2(**arguments)


def test_planted_tf_is_a_candidate_on_at_least_two_axes_end_to_end(state):
    result = _run(state)
    assert result["status"] == "executed" and result["motif_axis"] == "not_run"
    table = result["table"]
    planted = table.set_index("tf_symbol").loc[state["planted_tf"]]
    assert planted.target_enrichment_padj < 0.05 and planted.n_axes_supported >= 2
    assert planted.tf_activity_status == "separated_up" and planted.tf_expression_status == "supported_up"
    assert planted.motif_status == "not_run" and planted.n_axes_evaluable == 3
    assert table.n_axes_supported.max() == planted.n_axes_supported == 3      # display order puts a 3-axis TF first
    assert result["universe"]["universe_size"] == len(tf.build_universe(state["summary"]))
    assert result["n_tests"] == len(table) == int(table.n_tests.iloc[0])
    assert result["settings"]["fisher_alternative"] == "greater" and result["fingerprints"]["activity_fingerprint"]


def test_negative_control_run_has_no_target_enriched_tf_for_the_fixed_seed(state):
    """Regression check for seed 0 only (plan section 5.6); not a general false-positive guarantee."""
    result = _run(state, summary=permuted_summary(state["summary"]))
    assert (result["table"]["target_enrichment_padj"] <= 0.05).sum() == 0
    assert not result["table"]["axis_target_enrichment_supported"].any()


def test_run_without_activity_marks_the_axis_not_run_and_excludes_it_from_evaluable(state):
    result = _run(state, activity=False)
    planted = result["table"].set_index("tf_symbol").loc[state["planted_tf"]]
    assert planted.tf_activity_status == "not_run" and planted.n_axes_evaluable == 2
    assert result["fingerprints"]["activity_fingerprint"] is None


def test_empty_gene_set_is_not_tested_and_recorded(state):
    empty = state["summary"].assign(integration_class="not_significant")
    result = _run(state, summary=empty)
    assert result["status"] == "empty_gene_set" and result["table"].empty and result["n_tests"] == 0
    assert result["message"] and result["message_ja"]
    block = tf.build_tf_summary({"concordant_activation": result}, {"source": "collectri"})
    assert block["status"] == "no_executed_gene_set" and block["gene_sets"][0]["status"] == "empty_gene_set"


def test_run_requires_a_contrast_and_explicit_thresholds(state):
    with pytest.raises(tf.TFIntegrationError):
        _run(state, contrasts={"reference": "control"})
    with pytest.raises(tf.TFIntegrationError):
        _run(state, thresholds={"rna_padj": 0.05})
    with pytest.raises(tf.TFIntegrationError):
        _run(state, set_name="unknown")


def test_fingerprints_change_when_their_inputs_change(state):
    contrasts = {"reference": "control", "test": "treated"}
    base = tf.compute_fingerprints(state["summary"], state["thresholds"], contrasts, state["activity_scores"],
                                   {"organism": "mouse"}, state["metadata"]["condition"])
    again = tf.compute_fingerprints(state["summary"], state["thresholds"], contrasts, state["activity_scores"],
                                    {"organism": "mouse"}, state["metadata"]["condition"])
    assert base == again and all(base.values())
    changed_summary = state["summary"].copy()
    changed_summary.loc[0, "integration_class"] = "atac_only"
    assert tf.compute_fingerprints(changed_summary, state["thresholds"], contrasts, None)["input_fingerprint"] != base["input_fingerprint"]
    looser = {**state["thresholds"], "rna_padj": 0.1}
    assert tf.compute_fingerprints(state["summary"], looser, contrasts, None)["input_fingerprint"] != base["input_fingerprint"]
    swapped = {"reference": "treated", "test": "control"}
    assert tf.compute_fingerprints(state["summary"], state["thresholds"], swapped, None)["input_fingerprint"] != base["input_fingerprint"]
    edited = state["activity_scores"].copy()
    edited.iloc[0, 0] += 1.0
    assert tf.compute_fingerprints(state["summary"], state["thresholds"], contrasts, edited, {"organism": "mouse"},
                                   state["metadata"]["condition"])["activity_fingerprint"] != base["activity_fingerprint"]
    assert tf.compute_fingerprints(state["summary"], state["thresholds"], contrasts, state["activity_scores"],
                                   {"organism": "human"}, state["metadata"]["condition"])["activity_fingerprint"] != base["activity_fingerprint"]
    relabeled = state["metadata"]["condition"].copy()
    relabeled.iloc[0] = "treated"                             # same contrast labels, different group membership
    assert tf.compute_fingerprints(state["summary"], state["thresholds"], contrasts, state["activity_scores"],
                                   {"organism": "mouse"}, relabeled)["activity_fingerprint"] != base["activity_fingerprint"]
    assert tf.compute_fingerprints(state["summary"], state["thresholds"], contrasts, None)["activity_fingerprint"] is None


def test_tf_summary_block_records_the_full_provenance(state):
    run = _run(state, activity_meta={"organism": "mouse", "tmin": 5, "method_used": "ULM"})
    block = tf.build_tf_summary({"concordant_activation": run},
                                {"source": "collectri", "organism": "mouse", "n_edges": len(state["network"])})
    assert block["status"] == "executed" and block["external_services_used"] == [] and block["motif_axis"] == "not_run"
    assert block["fisher_alternative"] == "greater" and block["bh_scope"] == "within_gene_set_across_tested_tfs"
    assert block["universe"]["universe_size"] > 0 and block["universe"]["universe_definition_ja"]
    assert block["symbol_matching"]["matching"] == "strip+casefold"
    assert block["gene_sets"][0]["n_genes"] == 40 and block["activity_parameters"]["tmin"] == 5
    assert block["expression_rule"]["thresholds"] == {"rna_lfc": 1.0, "rna_padj": 0.05}
    assert block["limitations_text"] and block["limitations_text_ja"] and block["activity_rule"]
    assert "not a statistic" in block["n_axes_supported_note"]
    import json
    json.dumps(block, allow_nan=False)                        # the block must be JSON-serializable as is
    with pytest.raises(tf.TFIntegrationError):
        tf.build_tf_summary({}, {})


def test_combined_table_joins_hits_and_skips_empty_sets(state):
    runs = {"concordant_activation": _run(state), "atac_only": _run(state, set_name="atac_only")}
    combined = tf.combine_tf_tables(runs)
    assert set(combined.gene_set) == {"concordant_activation", "atac_only"}
    assert combined["targets_in_set"].map(lambda value: isinstance(value, str)).all()
    assert tf.combine_tf_tables({}).empty


def test_drill_down_keeps_every_peak_gene_edge(state):
    summary = state["summary"]
    genes, _ = tf.resolve_gene_set(summary, "concordant_activation", tf.build_universe(summary))
    drill = tf.get_tf_targets_in_set(state["planted_tf"], genes, state["network"], state["edges"])
    hits = set(state["planted_targets"][:25])
    assert set(drill.gene_symbol) == hits and (drill.tf_symbol == state["planted_tf"]).all()
    expected = state["edges"].loc[state["edges"].gene_symbol.isin(hits)]
    assert len(drill) == len(expected) >= len(hits)           # one-to-many edges are not deduplicated
    assert {"tf_target_weight", "peak_id", "edge_id"} <= set(drill.columns)
    with pytest.raises(tf.TFIntegrationError):
        tf.get_tf_targets_in_set("T", genes, state["network"], state["edges"].drop(columns=["gene_symbol"]))


def test_module_is_streamlit_free_and_makes_no_network_calls():
    import ast
    import pathlib

    source = pathlib.Path(tf.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source and "st.session_state" not in source
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "streamlit" not in imported
    assert not imported & {"requests", "urllib", "urllib3", "socket", "http", "httpx", "aiohttp", "ftplib"}


def test_result_columns_carry_no_combined_score_and_no_causal_wording(state):
    table = _run(state)["table"]
    assert not [c for c in table.columns if any(w in c.lower() for w in ("confidence", "combined", "weighted"))]
    import re
    causal = re.compile(r"\b(regulates|drives|causes|caused|causal)\b", re.IGNORECASE)
    for sentence in tf.LIMITATIONS_EN:
        stripped = sentence.replace("show no evidence that a TF regulates these genes or drives a phenotype", "")
        assert not causal.search(stripped), sentence