"""Level 2 TF-candidate contracts for RNA--ATAC integration.

This module is Streamlit-free (AGENTS.md I-3.1): it takes DataFrames plus explicit
settings and returns DataFrames or plain dictionaries.  The three evidence axes
(target enrichment, TF expression, TF activity) stay separate columns and are never
merged into one score (I-1.3), and no p-values from different modalities are combined
(I-1.2).  See ``docs/phase5_implementation_plan.md`` for the decisions behind each rule.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

import brim_integration_enrichment
from brim_multiomics import IntegrationError, extract_gene_set


class TFIntegrationError(IntegrationError):
    """Raised when Level 2 cannot run with the supplied inputs."""


SMALL_GENE_SET_THRESHOLD = 20
FISHER_ALTERNATIVE = "greater"

# Level 1 classes (or unions of them) that can be tested per TF.  Unions add no
# direction resolution (I-1.4): each member class is kept as it was classified.
SET_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "concordant_activation": ("concordant_activation",),
    "concordant_repression": ("concordant_repression",),
    "discordant_open_down": ("discordant_open_down",),
    "discordant_closed_up": ("discordant_closed_up",),
    "atac_only": ("atac_only",),
    "rna_only_on_mapped_peak": ("rna_only_on_mapped_peak",),
    "mixed_accessibility": ("mixed_accessibility",),
    "concordant_all": ("concordant_activation", "concordant_repression"),
    "discordant_all": ("discordant_open_down", "discordant_closed_up"),
}

UNIVERSE_DEFINITION = (
    "Level 2 background: genes with RNA padj and log2FC both tested (not NA) and at least one tested ATAC peak "
    "mapped (classifiable mapped genes); not-significant genes are included. It is neither all genes nor only "
    "the significant genes."
)
UNIVERSE_DEFINITION_JA = (
    "レベル2の背景遺伝子: RNAのpadjとlog2FCがともに検定済み（NAでない）で、検定済みATAC peakが1つ以上対応付いた遺伝子"
    "（分類可能な対応付き遺伝子）。有意でない遺伝子も含みます。全遺伝子でも、有意な遺伝子のみでもありません。"
)
BH_SCOPE_NOTE = (
    "padj is Benjamini-Hochberg corrected within the selected gene set across the TFs that were tested only. "
    "Looking at several gene sets is exploratory multiplicity and is not corrected."
)
BH_SCOPE_NOTE_JA = (
    "padjは選択した遺伝子集合の内側で、検定したTFのみを対象にBH補正しています。複数の遺伝子集合を見ることは"
    "探索的な多重性であり、補正していません。"
)

_ENRICHMENT_COLUMNS = [
    "tf_symbol", "gene_set", "n_gene_set", "n_universe", "n_targets_in_universe", "n_targets_in_set",
    "odds_ratio", "fold_enrichment", "target_enrichment_p", "target_enrichment_padj", "n_tests",
    "fisher_alternative", "target_enrichment_source", "small_gene_set_warning", "targets_in_set",
]


def fold_symbol(symbol: Any) -> str:
    """Return the comparison form of a gene symbol (strip + casefold); inputs are never rewritten."""
    return str(symbol).strip().casefold()


def build_universe(summary: pd.DataFrame) -> list[str]:
    """Return the Level 2 background, identical to the Phase 4 ORA background by construction."""
    return brim_integration_enrichment.build_ora_background(summary)


def describe_universe(summary: pd.DataFrame) -> dict[str, Any]:
    """Return the universe size and the counts needed to state its definition (I-1.5)."""
    universe = build_universe(summary)
    rna_not_tested = summary["rna_padj_is_na"].astype(bool) | summary["rna_lfc_is_na"].astype(bool)
    mapped = pd.to_numeric(summary["n_mapped_peaks"], errors="coerce").fillna(0).gt(0)
    tested_peak = pd.to_numeric(summary["n_atac_tested_peaks"], errors="coerce").fillna(0).gt(0)
    return {
        "universe_definition": UNIVERSE_DEFINITION,
        "universe_definition_ja": UNIVERSE_DEFINITION_JA,
        "universe_size": len(universe),
        "n_mapped_rna_tested": int((mapped & ~rna_not_tested).sum()),
        "n_excluded_atac_not_tested": int((mapped & ~rna_not_tested & ~tested_peak).sum()),
        "n_rna_only_no_mapped_peak": int((summary["integration_class"] == "rna_only_no_mapped_peak").sum()),
        "n_rna_not_tested": int(rna_not_tested.sum()),
    }


def resolve_gene_set(summary: pd.DataFrame, set_name: str, universe: Iterable[str]) -> tuple[list[str], int]:
    """Return the set's genes inside the universe and how many were removed for being outside it."""
    if set_name not in SET_DEFINITIONS:
        raise TFIntegrationError("Unknown Level 2 gene set: " + str(set_name))
    genes: set[str] = set()
    for integration_class in SET_DEFINITIONS[set_name]:
        genes.update(extract_gene_set(summary, integration_class))
    inside = genes.intersection(set(universe))
    return sorted(inside), len(genes) - len(inside)


def build_target_sets(
    network: pd.DataFrame, universe: Iterable[str], min_targets: int = 1,
) -> tuple[dict[str, frozenset[str]], dict[str, Any]]:
    """Map each TF to its targets inside the universe and report how symbols were matched.

    Symbols are compared with strip + casefold, which is a recorded conversion (I-2.2).
    ``gene_key`` and network symbols are never modified, and there is no alias expansion.
    """
    missing = {"source", "target"}.difference(network.columns)
    if missing:
        raise TFIntegrationError("TF network requires columns: " + ", ".join(sorted(missing)))
    if int(min_targets) < 1:
        raise TFIntegrationError("min_targets must be at least 1.")
    universe_list = sorted(set(map(str, universe)))
    by_fold: dict[str, list[str]] = {}
    for symbol in universe_list:
        by_fold.setdefault(fold_symbol(symbol), []).append(symbol)
    edges = network.loc[:, ["source", "target"]].dropna().astype(str)
    edges = edges.assign(_fold=edges["target"].map(fold_symbol))
    matched_folds = set(edges["_fold"]).intersection(by_fold)
    if not matched_folds:
        raise TFIntegrationError(
            "None of the universe gene symbols match the TF network targets. Check that the species and the "
            "gene identifier type (gene symbol) are correct."
        )
    matched = edges.loc[edges["_fold"].isin(matched_folds)]
    target_sets: dict[str, frozenset[str]] = {}
    for tf_symbol, group in matched.groupby("source", sort=True):
        symbols: set[str] = set()
        for fold in group["_fold"].unique():
            symbols.update(by_fold[fold])
        target_sets[str(tf_symbol)] = frozenset(symbols)
    network_fold_spellings = edges.loc[edges["_fold"].isin(matched_folds)].groupby("_fold")["target"].nunique()
    universe_in_network = {symbol for fold in matched_folds for symbol in by_fold[fold]}
    report = {
        "matching": "strip+casefold",
        "n_universe_symbols": len(universe_list),
        "n_universe_symbols_in_network": len(universe_in_network),
        "n_network_targets_matched": len(matched_folds),
        "n_casefold_collisions_universe": int(sum(len(symbols) > 1 for symbols in by_fold.values())),
        "n_casefold_collisions_network": int((network_fold_spellings > 1).sum()),
        "min_targets": int(min_targets),
    }
    return target_sets, report


def benjamini_hochberg(pvalues: Iterable[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(list(pvalues), dtype=float)
    if p.size == 0:
        return p
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise TFIntegrationError("p-values must be finite values between 0 and 1.")
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def test_target_enrichment(
    gene_set: Iterable[str], universe: Iterable[str], network: pd.DataFrame, min_targets: int,
    network_source: str, set_name: str = "",
) -> pd.DataFrame:
    """One-sided Fisher target enrichment of each TF in one gene set (a new, independent test; I-1.6).

    A TF is tested only if at least ``min_targets`` of its targets are in the universe.  BH correction
    is applied within this gene set across the tested TFs; TFs with no hit stay in the family with p=1
    or above.  Result attributes carry ``n_tests``, ``n_tfs_below_min_targets`` and the match report.
    """
    universe_set = set(map(str, universe))
    genes = set(map(str, gene_set))
    if not genes.issubset(universe_set):
        raise TFIntegrationError("The gene set must be a subset of the universe.")
    target_sets, report = build_target_sets(network, universe_set, min_targets)
    n_universe, n_set = len(universe_set), len(genes)
    rows: list[dict[str, Any]] = []
    below = 0
    if not genes:
        # An empty gene set is not tested (plan D7); the caller reports it as `empty_gene_set`.
        target_sets = {}
    for tf_symbol, targets in target_sets.items():
        k = len(targets)
        if k < int(min_targets):
            below += 1
            continue
        hits = sorted(targets.intersection(genes))
        a = len(hits)
        table = [[a, n_set - a], [k - a, n_universe - n_set - (k - a)]]
        odds_ratio, p_value = fisher_exact(table, alternative=FISHER_ALTERNATIVE)
        expected = k / n_universe if n_universe else float("nan")
        rows.append({
            "tf_symbol": tf_symbol, "gene_set": set_name, "n_gene_set": n_set, "n_universe": n_universe,
            "n_targets_in_universe": k, "n_targets_in_set": a, "odds_ratio": float(odds_ratio),
            "fold_enrichment": (a / n_set) / expected if n_set and expected else float("nan"),
            "target_enrichment_p": float(p_value), "fisher_alternative": FISHER_ALTERNATIVE,
            "target_enrichment_source": network_source,
            "small_gene_set_warning": n_set < SMALL_GENE_SET_THRESHOLD, "targets_in_set": hits,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=_ENRICHMENT_COLUMNS)
    else:
        result["target_enrichment_padj"] = benjamini_hochberg(result["target_enrichment_p"])
        result["n_tests"] = len(result)
        result = result.loc[:, _ENRICHMENT_COLUMNS]
        result = result.sort_values(["target_enrichment_padj", "target_enrichment_p", "tf_symbol"],
                                    kind="mergesort").reset_index(drop=True)
    result.attrs.update({
        "n_tests": len(result), "n_tfs_below_min_targets": below, "match_report": report,
        "bh_scope_note": BH_SCOPE_NOTE, "bh_scope_note_ja": BH_SCOPE_NOTE_JA,
    })
    return result


# --------------------------------------------------------------------------------------------------
# Evidence axes (plan D4, D5, D6).  Each axis stays a separate set of columns (I-1.3).
# --------------------------------------------------------------------------------------------------

MIN_ACTIVITY_SAMPLES_PER_GROUP = 3
EXPRESSION_SUPPORTED = ("supported_up", "supported_down")
ACTIVITY_SEPARATED = ("separated_up", "separated_down")
_EXPRESSION_DEFINITE = EXPRESSION_SUPPORTED + ("not_significant",)
_ACTIVITY_DEFINITE = ACTIVITY_SEPARATED + ("not_separated",)


def _require_symbol_column(tf_table: pd.DataFrame) -> None:
    if "tf_symbol" not in tf_table.columns:
        raise TFIntegrationError("The TF table requires a tf_symbol column.")


def _rna_thresholds(thresholds: Mapping[str, Any]) -> tuple[float, float]:
    try:
        padj, lfc = float(thresholds["rna_padj"]), float(thresholds["rna_lfc"])
    except (KeyError, TypeError, ValueError) as error:
        raise TFIntegrationError("Thresholds require numeric rna_padj and rna_lfc.") from error
    if not (np.isfinite(padj) and np.isfinite(lfc)) or not 0 <= padj <= 1 or lfc < 0:
        raise TFIntegrationError("rna_padj must be within [0, 1] and rna_lfc must be non-negative.")
    return padj, lfc


def attach_tf_expression(tf_table: pd.DataFrame, rna_results: pd.DataFrame,
                         thresholds: Mapping[str, Any]) -> pd.DataFrame:
    """Add the TF expression axis from the standardized RNA table (NA is "not tested", I-1.1).

    ``rna_results`` must come from ``brim_multiomics.standardize_rna_results``.  A TF whose RNA padj or
    log2FC was NA in DESeq2 is ``not_tested`` and its values stay blank; it is never "not significant".
    """
    _require_symbol_column(tf_table)
    required = {"gene_key", "log2FoldChange", "padj", "padj_is_na", "lfc_is_na"}
    if required.difference(rna_results.columns):
        raise TFIntegrationError("RNA results must be standardized (gene_key and NA flags are required).")
    padj_limit, lfc_limit = _rna_thresholds(thresholds)
    by_fold: dict[str, list[int]] = {}
    for position, key in enumerate(rna_results["gene_key"].astype(str)):
        by_fold.setdefault(fold_symbol(key), []).append(position)
    records: list[dict[str, Any]] = []
    for tf_symbol in tf_table["tf_symbol"].astype(str):
        positions = by_fold.get(fold_symbol(tf_symbol), [])
        blank = {"tf_rna_log2FoldChange": np.nan, "tf_rna_padj": np.nan,
                 "tf_rna_padj_is_na": pd.NA, "tf_rna_lfc_is_na": pd.NA}
        if not positions:
            records.append({**blank, "tf_expression_status": "not_in_rna_results"})
            continue
        if len(positions) > 1:
            records.append({**blank, "tf_expression_status": "ambiguous_symbol"})
            continue
        row = rna_results.iloc[positions[0]]
        padj_na, lfc_na = bool(row["padj_is_na"]), bool(row["lfc_is_na"])
        lfc, padj = float(row["log2FoldChange"]), float(row["padj"])
        record = {"tf_rna_padj_is_na": padj_na, "tf_rna_lfc_is_na": lfc_na}
        if padj_na or lfc_na or not (np.isfinite(lfc) and np.isfinite(padj)):
            records.append({**blank, **record, "tf_expression_status": "not_tested"})
            continue
        supported = padj <= padj_limit and abs(lfc) >= lfc_limit and lfc != 0
        status = ("supported_up" if lfc > 0 else "supported_down") if supported else "not_significant"
        records.append({**record, "tf_rna_log2FoldChange": lfc, "tf_rna_padj": padj, "tf_expression_status": status})
    added = pd.DataFrame(records, index=tf_table.index)
    result = pd.concat([tf_table.drop(columns=[c for c in added.columns if c in tf_table.columns]), added], axis=1)
    result.attrs = dict(tf_table.attrs)
    return result


def attach_tf_activity(
    tf_table: pd.DataFrame, activity_scores: pd.DataFrame | None, sample_conditions: pd.Series | None,
    reference: str, test: str, source: str = "collectri",
) -> pd.DataFrame:
    """Add the TF activity axis as a descriptive group-separation rule on existing per-sample scores.

    No statistical test is run and no p-value is created.  ``separated_up`` means every test sample scores
    above every reference sample.  With 3 vs 3 samples about 10% of null TFs pass this rule, so it is
    exploratory.  Non-finite scores give ``not_estimated`` and are never averaged over.
    """
    _require_symbol_column(tf_table)
    columns = {"tf_activity_score": np.nan, "tf_activity_source": source, "tf_activity_n_ref": pd.NA,
               "tf_activity_n_test": pd.NA, "tf_activity_n_samples_ignored": pd.NA}
    reference_samples: list[str] = []
    test_samples: list[str] = []
    ignored = 0
    global_status: str | None = None
    if activity_scores is None or activity_scores.empty or sample_conditions is None:
        global_status = "not_run"
    else:
        conditions = sample_conditions.astype(str)
        reference_samples = [str(s) for s in conditions.index[conditions == str(reference)]]
        test_samples = [str(s) for s in conditions.index[conditions == str(test)]]
        present = set(map(str, activity_scores.index))
        if (not reference_samples or not test_samples
                or not set(reference_samples).issubset(present) or not set(test_samples).issubset(present)):
            global_status = "not_run"
        else:
            ignored = len(present - set(reference_samples) - set(test_samples))
            if min(len(reference_samples), len(test_samples)) < MIN_ACTIVITY_SAMPLES_PER_GROUP:
                global_status = "insufficient_samples"
    records: list[dict[str, Any]] = []
    scores = None
    if global_status in (None, "insufficient_samples") and activity_scores is not None:
        scores = activity_scores.copy()
        scores.index = scores.index.map(str)
    for tf_symbol in tf_table["tf_symbol"].astype(str):
        record = dict(columns)
        if global_status == "not_run":
            record["tf_activity_status"] = "not_run"
        else:
            record.update(tf_activity_n_ref=len(reference_samples), tf_activity_n_test=len(test_samples),
                          tf_activity_n_samples_ignored=ignored)
            if global_status == "insufficient_samples":
                record["tf_activity_status"] = "insufficient_samples"
            elif tf_symbol not in scores.columns:
                record["tf_activity_status"] = "not_estimated"
            else:
                column = pd.to_numeric(scores[tf_symbol], errors="coerce")
                if not np.isfinite(column.to_numpy(dtype=float)).all():
                    record["tf_activity_status"] = "not_estimated"
                else:
                    ref_values = column.loc[reference_samples].to_numpy(dtype=float)
                    test_values = column.loc[test_samples].to_numpy(dtype=float)
                    record["tf_activity_score"] = float(test_values.mean() - ref_values.mean())
                    if test_values.min() > ref_values.max():
                        record["tf_activity_status"] = "separated_up"
                    elif test_values.max() < ref_values.min():
                        record["tf_activity_status"] = "separated_down"
                    else:
                        record["tf_activity_status"] = "not_separated"
        records.append(record)
    added = pd.DataFrame(records, index=tf_table.index)
    result = pd.concat([tf_table.drop(columns=[c for c in added.columns if c in tf_table.columns]), added], axis=1)
    result.attrs = dict(tf_table.attrs)
    return result


def add_motif_placeholder(tf_table: pd.DataFrame) -> pd.DataFrame:
    """Add the motif axis as an explicit "not run" marker (Level 3 is not implemented; I-5.1)."""
    result = tf_table.copy()
    result["motif_enrichment_padj"] = np.nan
    result["motif_enrichment_score"] = np.nan
    result["motif_status"] = "not_run"
    result["motif_source"] = ""
    result.attrs = dict(tf_table.attrs)
    return result


def count_supported_axes(tf_table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add the per-axis flags and the display-only axis counts, then sort for display.

    ``n_axes_supported`` is a count for ordering only: it is not a statistic, not a filter, and it feeds
    no test (I-1.3).  Axes that were not run, not tested or not estimable are neither supported nor
    evaluable.  The tie-break uses the single target-enrichment axis.
    """
    needed = {"target_enrichment_padj", "n_targets_in_set", "tf_expression_status", "tf_activity_status", "tf_symbol"}
    if needed.difference(tf_table.columns):
        raise TFIntegrationError("The TF table is missing axis columns: " + ", ".join(sorted(needed.difference(tf_table.columns))))
    if not np.isfinite(float(alpha)) or not 0 <= float(alpha) <= 1:
        raise TFIntegrationError("alpha must be within [0, 1].")
    result = tf_table.copy()
    padj = pd.to_numeric(result["target_enrichment_padj"], errors="coerce")
    target_supported = padj.le(float(alpha)) & pd.to_numeric(result["n_targets_in_set"], errors="coerce").ge(1)
    expression_supported = result["tf_expression_status"].isin(EXPRESSION_SUPPORTED)
    activity_supported = result["tf_activity_status"].isin(ACTIVITY_SEPARATED)
    result["axis_target_enrichment_supported"] = target_supported.astype(bool)
    result["axis_expression_supported"] = expression_supported.astype(bool)
    result["axis_activity_supported"] = activity_supported.astype(bool)
    result["n_axes_supported"] = (target_supported.astype(int) + expression_supported.astype(int)
                                  + activity_supported.astype(int))
    result["n_axes_evaluable"] = (padj.notna().astype(int)
                                  + result["tf_expression_status"].isin(_EXPRESSION_DEFINITE).astype(int)
                                  + result["tf_activity_status"].isin(_ACTIVITY_DEFINITE).astype(int))
    result = result.sort_values(["n_axes_supported", "target_enrichment_padj", "tf_symbol"],
                                ascending=[False, True, True], kind="mergesort").reset_index(drop=True)
    result.attrs = dict(tf_table.attrs)
    return result

# --------------------------------------------------------------------------------------------------
# Limitations text, fingerprints, one Level 2 run, manifest block and drill-down (plan D4, D8).
# --------------------------------------------------------------------------------------------------

LIMITATIONS_EN = (
    "All three Level 2 columns come from RNA-seq and curated regulatory databases; ATAC only narrows the input "
    "gene set.",
    "TFs regulated by nuclear translocation or post-translational modification (for example NF-kB, STAT, SMAD, "
    "HIF-1a) may show weak mRNA change and weak target response, and may not be detected.",
    "Curated databases reflect research volume, so TFs with few reports are structurally hard to detect.",
    "Target-enrichment padj is a new test computed from the Level 1 classification; it is independent of RNA and "
    "ATAC padj and is corrected within one gene set only.",
    "Candidates are hypotheses and show no evidence that a TF regulates these genes or drives a phenotype.",
    "Motif enrichment: not run. Number of supported axes is a sorting aid, not a statistic.",
    "Activity 'separated' is a descriptive group-separation rule without a p-value; about 10% of null TFs pass it "
    "with 3 vs 3 samples.",
    "Target enrichment ignores the sign of regulation (activation vs repression) in the network.",
    "Padj is BH-corrected within one gene set across tested TFs; TFs with overlapping target sets are not "
    "independent tests.",
    "The three axes are all RNA-derived (target enrichment uses the RNA-derived Level 1 classes; expression and "
    "activity use RNA) and are not independent evidence.",
    "Some CollecTRI entries are complexes and may appear as not_in_rna_results.",
)
LIMITATIONS_JA = (
    "レベル2の3列はいずれもRNA-seqとキュレーション済み制御データベースに由来し、ATACは入力遺伝子集合の絞り込みにのみ寄与します。",
    "核移行や翻訳後修飾で活性化するTF（NF-κB、STAT、SMAD、HIF-1αなど）は、mRNA発現も既知標的の応答も弱く、"
    "検出されないことがあります。",
    "キュレーションDBは研究の蓄積量に依存するため、報告の少ないTFは構造的に検出されにくくなります。",
    "標的濃縮のpadjはレベル1の分類を入力とする新規の検定で、RNA・ATACのpadjとは独立であり、1つの遺伝子集合内でのみ補正されています。",
    "候補は仮説であり、TFがこれらの遺伝子を制御する、あるいは表現型を引き起こすことを示す根拠ではありません。",
    "motif濃縮は未実行です。支持軸数は並べ替えの補助であり統計量ではありません。",
    "activityの「separated」はp値を伴わない記述的な群分離規則で、3 vs 3では帰無のTFの約10%が通過します。",
    "標的濃縮はネットワークの制御の符号（活性化・抑制）を考慮しません。",
    "padjは1つの遺伝子集合内の検定したTFでBH補正されており、標的集合が重なるTFは独立な検定ではありません。",
    "3つの軸はいずれもRNA由来であり（標的濃縮はRNA由来のレベル1分類を使用）、独立な根拠ではありません。",
    "CollecTRIの複合体エントリはnot_in_rna_resultsになることがあります。",
)
ACTIVITY_RULE = (
    "Descriptive rule on the existing per-sample TF activity scores: score = mean(test) - mean(reference); "
    "'separated_up' if every test sample is above every reference sample ('separated_down' for the reverse); "
    "fewer than 3 samples in a group gives 'insufficient_samples'; non-finite scores give 'not_estimated'. "
    "No p-value is computed."
)
EMPTY_GENE_SET_MESSAGE = "The selected gene set has no genes inside the universe, so it was not tested."
EMPTY_GENE_SET_MESSAGE_JA = "選択した遺伝子集合には背景遺伝子内の遺伝子がないため、検定しませんでした。"

_SUMMARY_FINGERPRINT_COLUMNS = ("gene_key", "gene_symbol", "integration_class", "rna_log2FoldChange", "rna_padj",
                                "rna_padj_is_na", "rna_lfc_is_na", "n_mapped_peaks", "n_atac_tested_peaks")


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def compute_fingerprints(
    summary: pd.DataFrame, thresholds: Mapping[str, Any], contrasts: Mapping[str, Any],
    activity_scores: pd.DataFrame | None, activity_meta: Mapping[str, Any] | None = None,
    sample_conditions: pd.Series | None = None,
) -> dict[str, str | None]:
    """Fingerprint the inputs a Level 2 result depends on, so stale results can be detected (I-6.2).

    ``input_fingerprint`` covers the Level 1 gene summary, thresholds and contrast.  ``activity_fingerprint``
    covers the activity matrix, its recorded parameters and the contrast's reference/test sample names; it
    is ``None`` when no activity result exists.
    """
    columns = [c for c in _SUMMARY_FINGERPRINT_COLUMNS if c in summary.columns]
    body = summary.loc[:, columns].sort_values(columns[0] if columns else summary.columns[0], kind="mergesort")
    input_fingerprint = _digest({
        "summary": body.astype(object).where(body.notna(), None).values.tolist(), "columns": columns,
        "thresholds": {key: float(thresholds[key]) for key in sorted(thresholds)}, "contrast": dict(contrasts),
    })
    activity_fingerprint = None
    if activity_scores is not None and not activity_scores.empty:
        reference = str(contrasts.get("reference"))
        test = str(contrasts.get("test"))
        groups: dict[str, list[str]] = {"reference": [], "test": []}
        if sample_conditions is not None:
            conditions = sample_conditions.astype(str)
            groups = {"reference": sorted(map(str, conditions.index[conditions == reference])),
                      "test": sorted(map(str, conditions.index[conditions == test]))}
        values = np.round(activity_scores.to_numpy(dtype=float), 8)
        activity_fingerprint = _digest({
            "index": list(map(str, activity_scores.index)), "columns": list(map(str, activity_scores.columns)),
            "values": np.where(np.isfinite(values), values, None).tolist(), "meta": dict(activity_meta or {}),
            "groups": groups,
        })
    return {"input_fingerprint": input_fingerprint, "activity_fingerprint": activity_fingerprint}


def run_level2(
    summary: pd.DataFrame, set_name: str, network: pd.DataFrame, rna_results: pd.DataFrame,
    thresholds: Mapping[str, Any], contrasts: Mapping[str, Any], activity_scores: pd.DataFrame | None,
    sample_conditions: pd.Series | None, min_targets: int, alpha: float, network_source: str = "collectri",
    activity_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Level 2 for one gene set and return the table plus everything needed for provenance.

    All settings are explicit arguments (I-3.2): thresholds, contrast and network come from the Level 1 run
    settings, never from live widgets.  The result of an empty gene set is ``status="empty_gene_set"`` with no
    rows.
    """
    if not {"reference", "test"}.issubset(contrasts) or not contrasts["reference"] or not contrasts["test"]:
        raise TFIntegrationError("Level 2 requires the Level 1 contrast (reference and test).")
    universe = build_universe(summary)
    description = describe_universe(summary)
    genes, removed = resolve_gene_set(summary, set_name, universe)
    fingerprints = compute_fingerprints(summary, thresholds, contrasts, activity_scores, activity_meta, sample_conditions)
    gene_set_record = {"name": set_name, "n_genes": len(genes), "n_removed_outside_universe": removed,
                       "small_gene_set_warning": len(genes) < SMALL_GENE_SET_THRESHOLD}
    result: dict[str, Any] = {
        "set_name": set_name, "gene_set": gene_set_record, "universe": description, "genes": genes,
        "settings": {"min_targets": int(min_targets), "alpha": float(alpha), "fisher_alternative": FISHER_ALTERNATIVE,
                     "bh_scope": "within_gene_set_across_tested_tfs", "network_source": network_source,
                     "thresholds": {key: float(thresholds[key]) for key in sorted(thresholds)},
                     "contrast": {"reference": str(contrasts["reference"]), "test": str(contrasts["test"])}},
        "fingerprints": fingerprints, "motif_axis": "not_run", "activity_meta": dict(activity_meta or {}),
        "bh_scope_note": BH_SCOPE_NOTE, "bh_scope_note_ja": BH_SCOPE_NOTE_JA,
        "limitations": list(LIMITATIONS_EN), "limitations_ja": list(LIMITATIONS_JA), "activity_rule": ACTIVITY_RULE,
    }
    if not genes:
        result.update(status="empty_gene_set", message=EMPTY_GENE_SET_MESSAGE, message_ja=EMPTY_GENE_SET_MESSAGE_JA,
                      table=pd.DataFrame(columns=_ENRICHMENT_COLUMNS), n_tests=0, n_tfs_below_min_targets=0,
                      match_report=None)
        return result
    enrichment = test_target_enrichment(genes, universe, network, min_targets, network_source, set_name=set_name)
    table = attach_tf_expression(enrichment, rna_results, thresholds)
    table = attach_tf_activity(table, activity_scores, sample_conditions, str(contrasts["reference"]),
                               str(contrasts["test"]), source=network_source)
    table = count_supported_axes(add_motif_placeholder(table), alpha)
    # The settings differ between gene sets when the user changes the sliders between runs, so each row keeps
    # the values that produced it.
    table["min_targets"] = int(min_targets)
    table["alpha"] = float(alpha)
    result.update(status="executed", table=table, n_tests=int(enrichment.attrs["n_tests"]),
                  n_tfs_below_min_targets=int(enrichment.attrs["n_tfs_below_min_targets"]),
                  match_report=enrichment.attrs["match_report"])
    return result


def combine_tf_tables(runs: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Concatenate the executed gene-set tables for ``tf_candidates.csv`` (list columns are joined)."""
    frames = []
    for set_name in sorted(runs):
        run = runs[set_name]
        if run.get("status") != "executed" or run["table"].empty:
            continue
        frame = run["table"].copy()
        frame["targets_in_set"] = frame["targets_in_set"].map(lambda hits: ";".join(hits))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_ENRICHMENT_COLUMNS)


def build_tf_summary(runs: Mapping[str, Mapping[str, Any]], network_info: Mapping[str, Any]) -> dict[str, Any]:
    """Build the ``tf_level2`` manifest/summary block from the stored runs (their single source, plan D8)."""
    if not runs:
        raise TFIntegrationError("There are no Level 2 runs to summarize.")
    ordered = [runs[name] for name in sorted(runs)]
    first = ordered[0]
    executed = [run for run in ordered if run.get("status") == "executed"]
    min_targets_values = {run["settings"]["min_targets"] for run in ordered}
    alpha_values = {run["settings"]["alpha"] for run in ordered}
    matching = next((dict(run["match_report"]) for run in executed), None)
    if matching is not None:
        matching.pop("min_targets", None)      # a per-gene-set setting, recorded in gene_sets instead
    return {
        "status": "executed" if executed else "no_executed_gene_set",
        "network": dict(network_info),
        # Stated once only when every gene set used the same value; otherwise see each entry of gene_sets.
        "min_targets": min_targets_values.pop() if len(min_targets_values) == 1 else None,
        "alpha": alpha_values.pop() if len(alpha_values) == 1 else None,
        "settings_vary_between_gene_sets": len({run["settings"]["min_targets"] for run in ordered}) > 1
        or len({run["settings"]["alpha"] for run in ordered}) > 1,
        "fisher_alternative": FISHER_ALTERNATIVE, "bh_scope": first["settings"]["bh_scope"],
        "bh_scope_note": BH_SCOPE_NOTE, "bh_scope_note_ja": BH_SCOPE_NOTE_JA,
        "universe": first["universe"],
        "gene_sets": [{**run["gene_set"], "status": run["status"], "n_tests": run["n_tests"],
                       "n_tfs_below_min_targets": run["n_tfs_below_min_targets"],
                       "min_targets": run["settings"]["min_targets"], "alpha": run["settings"]["alpha"]}
                      for run in ordered],
        "symbol_matching": matching,
        "expression_rule": {"thresholds": {k: v for k, v in first["settings"]["thresholds"].items() if k.startswith("rna_")},
                            "supported_requires": "not NA, padj <= rna_padj, |log2FC| >= rna_lfc, log2FC != 0"},
        "activity_rule": ACTIVITY_RULE, "activity_parameters": first["activity_meta"] or "not recorded",
        "contrast": first["settings"]["contrast"],
        "fingerprints": first["fingerprints"], "motif_axis": "not_run",
        "n_axes_supported_note": "display-only sorting aid, not a statistic",
        "external_services_used": [],
        "run_history": [{"gene_set": run["set_name"], "status": run["status"], "n_genes": run["gene_set"]["n_genes"],
                         "n_tests": run["n_tests"], "min_targets": run["settings"]["min_targets"],
                         "alpha": run["settings"]["alpha"], "fingerprints": run["fingerprints"]} for run in ordered],
        "limitations_text": list(LIMITATIONS_EN), "limitations_text_ja": list(LIMITATIONS_JA),
    }


def get_tf_targets_in_set(tf_symbol: str, gene_set: Iterable[str], network: pd.DataFrame,
                          edges: pd.DataFrame) -> pd.DataFrame:
    """Drill-down: the gene-set targets of one TF with every peak--gene edge preserved (I-2.1)."""
    if not {"source", "target"}.issubset(network.columns) or "gene_symbol" not in edges.columns:
        raise TFIntegrationError("Drill-down requires a TF network and peak-gene edges with gene_symbol.")
    genes = sorted(set(map(str, gene_set)))
    links = network.loc[network["source"].astype(str) == str(tf_symbol)].copy()
    links["_fold"] = links["target"].map(fold_symbol)
    folds = {fold_symbol(gene): gene for gene in genes}
    links = links.loc[links["_fold"].isin(folds)]
    hits = sorted({folds[fold] for fold in links["_fold"]})
    selected = edges.loc[edges["gene_symbol"].astype(str).isin(hits)].copy()
    weights = links.assign(gene_symbol=links["_fold"].map(folds)).drop_duplicates("gene_symbol")
    keep = [c for c in ("gene_symbol", "weight", "sign_decision") if c in weights.columns]
    selected = selected.merge(weights.loc[:, keep].rename(columns={"weight": "tf_target_weight",
                                                                    "sign_decision": "tf_target_sign_decision"}),
                              on="gene_symbol", how="left")
    selected.insert(0, "tf_symbol", str(tf_symbol))
    return selected.reset_index(drop=True)

# The public name required by the plan starts with `test_`; keep pytest from collecting it when imported.
test_target_enrichment.__test__ = False
