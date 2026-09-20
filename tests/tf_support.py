"""Deterministic synthetic RNA+ATAC state for the Phase 5 (Level 2) tests.

The parameters below are fixed by ``docs/phase5_implementation_plan.md`` section 5.6.  They must not be
tuned, and no seed may be searched for: the negative-control assertions are regression checks for this
fixed seed, not a general false-positive guarantee.  This is a synthetic positive/negative control for the
pipeline, not real data and not evidence of biological validity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import brim_multiomics as multi
from brim_tf_networks import load_collectri_network


THRESHOLDS = {"rna_padj": 0.05, "rna_lfc": 1.0, "atac_padj": 0.05, "atac_lfc": 1.0}
CONTRAST = {"reference": "control", "test": "treated"}
SEED = 0
N_PLANTED_TARGETS = 60
N_DECOYS = 340
N_PLANTED_CONCORDANT = 25
N_DECOY_CONCORDANT = 15
RNA_NA_FRACTION = 0.05
ATAC_NA_FRACTION = 0.05
MULTI_PEAK_FRACTION = 0.05
N_OTHER_TFS = 49
REFERENCE_SAMPLES = ["control_1", "control_2", "control_3"]
TEST_SAMPLES = ["treated_1", "treated_2", "treated_3"]
PLANTED_ACTIVITY_REFERENCE = [-0.5, 0.0, 0.5]
PLANTED_ACTIVITY_TEST = [2.5, 3.0, 3.5]
PLANTED_TF_RNA = {"log2FoldChange": 1.5, "padj": 0.001}

# (rna_lfc, rna_padj, atac_lfc, atac_padj) for each Level 1 edge class.
_CLASS_VALUES = {
    "concordant_activation": (2.0, 0.001, 2.0, 0.001),
    "concordant_repression": (-2.0, 0.001, -2.0, 0.001),
    "discordant_open_down": (-2.0, 0.001, 2.0, 0.001),
    "discordant_closed_up": (2.0, 0.001, -2.0, 0.001),
    "atac_only": (0.1, 0.5, 2.0, 0.001),
    "rna_only_on_mapped_peak": (2.0, 0.001, 0.1, 0.5),
    "not_significant": (0.1, 0.5, 0.1, 0.5),
}
_OTHER_CLASSES = ["not_significant", "atac_only", "rna_only_on_mapped_peak", "concordant_repression",
                  "discordant_open_down", "discordant_closed_up"]
_OTHER_CLASS_PROBABILITIES = [0.55, 0.12, 0.12, 0.07, 0.07, 0.07]


def _pick_planted_tf(network: pd.DataFrame) -> tuple[str, list[str]]:
    counts = network.groupby("source")["target"].nunique().sort_index(kind="mergesort")
    planted_tf = str(counts.sort_values(ascending=False, kind="mergesort").index[0])
    targets = sorted(set(network.loc[network["source"] == planted_tf, "target"].astype(str)) - {planted_tf})
    assert len(targets) >= N_PLANTED_TARGETS
    return planted_tf, targets


def synthetic_tf_level1_state(seed: int = SEED, species: str = "Mouse") -> dict[str, Any]:
    """Build a production-shaped Level 1 result with one planted TF (see module docstring)."""
    assert species == "Mouse", "The synthetic fixture uses the bundled mouse CollecTRI network."
    rng = np.random.default_rng(seed)
    network = load_collectri_network("mouse")
    planted_tf, planted_target_pool = _pick_planted_tf(network)
    planted_targets = planted_target_pool[:N_PLANTED_TARGETS]
    excluded = set(planted_target_pool) | {planted_tf}
    decoy_pool = sorted(set(network["target"].astype(str)) - excluded)
    decoys = sorted(rng.choice(decoy_pool, size=N_DECOYS, replace=False).tolist())
    genes = planted_targets + decoys

    decoy_order = rng.permutation(decoys).tolist()
    concordant_decoys = set(decoy_order[:N_DECOY_CONCORDANT])
    classes: dict[str, str] = {}
    for gene in planted_targets[:N_PLANTED_CONCORDANT]:
        classes[gene] = "concordant_activation"
    for gene in concordant_decoys:
        classes[gene] = "concordant_activation"
    remaining = [gene for gene in genes if gene not in classes]
    for gene, chosen in zip(remaining, rng.choice(_OTHER_CLASSES, size=len(remaining), p=_OTHER_CLASS_PROBABILITIES)):
        classes[gene] = str(chosen)

    na_candidates = sorted(set(decoys) - concordant_decoys)
    n_na = int(round(RNA_NA_FRACTION * len(genes)))
    rna_na = set(rng.choice(na_candidates, size=n_na, replace=False).tolist())
    atac_na = set(rng.choice(na_candidates, size=int(round(ATAC_NA_FRACTION * len(genes))), replace=False).tolist())
    multi_candidates = sorted(set(genes) - atac_na)
    multi_peak = set(rng.choice(multi_candidates, size=int(round(MULTI_PEAK_FRACTION * len(genes))),
                                replace=False).tolist())

    rna_rows, edge_rows = [], []
    for gene in genes:
        rna_lfc, rna_padj, atac_lfc, atac_padj = _CLASS_VALUES[classes[gene]]
        rna_rows.append({"gene": gene, "log2FoldChange": rna_lfc, "padj": 1.0 if gene in rna_na else rna_padj,
                         "padj_is_na": gene in rna_na, "lfc_is_na": False})
        edge_rows.append({"peak_id": f"peak_{gene}", "gene_symbol": gene, "atac_log2FoldChange": atac_lfc,
                          "atac_padj": 1.0 if gene in atac_na else atac_padj,
                          "atac_padj_is_na": gene in atac_na, "atac_lfc_is_na": False})
        if gene in multi_peak:
            edge_rows.append({"peak_id": f"peak_{gene}_extra", "gene_symbol": gene, "atac_log2FoldChange": 0.1,
                              "atac_padj": 0.5, "atac_padj_is_na": False, "atac_lfc_is_na": False})
    rna_rows.append({"gene": planted_tf, **PLANTED_TF_RNA, "padj_is_na": False, "lfc_is_na": False})

    deg_results = pd.DataFrame(rna_rows).set_index("gene")
    edges = pd.DataFrame(edge_rows)
    edges = edges.assign(gene_id=edges["gene_symbol"], edge_id=[f"e{i}" for i in range(len(edges))],
                         mapping_method="promoter", distance_to_tss=0)
    rna_results = multi.standardize_rna_results(deg_results, "gene_symbol")
    integrated = multi.integrate_peak_gene_edges(rna_results, edges, THRESHOLDS)
    classified = multi.classify_integration_edges(integrated, THRESHOLDS)
    summary = multi.summarize_integration_by_gene(classified, THRESHOLDS)

    other_tfs = sorted(set(network["source"].astype(str)) - {planted_tf})
    chosen_tfs = sorted(rng.choice(other_tfs, size=N_OTHER_TFS, replace=False).tolist())
    samples = REFERENCE_SAMPLES + TEST_SAMPLES
    activity = pd.DataFrame(rng.normal(size=(len(samples), len(chosen_tfs))), index=samples, columns=chosen_tfs)
    activity[planted_tf] = PLANTED_ACTIVITY_REFERENCE + PLANTED_ACTIVITY_TEST
    metadata = pd.DataFrame({"condition": ["control"] * 3 + ["treated"] * 3}, index=samples)
    return {
        "seed": seed, "species": species, "network": network, "planted_tf": planted_tf,
        "planted_targets": planted_targets, "deg_results": deg_results, "rna_results": rna_results,
        "edges": classified, "summary": summary, "thresholds": dict(THRESHOLDS), "contrast": dict(CONTRAST),
        "activity_scores": activity, "metadata": metadata,
    }


def permuted_summary(summary: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Negative control: shuffle class labels among the universe genes, keeping the universe unchanged."""
    from brim_tf_integration import build_universe

    rng = np.random.default_rng(seed)
    result = summary.copy()
    inside = result["gene_symbol"].isin(set(build_universe(summary)))
    labels = result.loc[inside, "integration_class"].to_numpy()
    result.loc[inside, "integration_class"] = rng.permutation(labels)
    return result
