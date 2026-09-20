"""Local, class-specific Level 1 ORA contracts for RNA--ATAC integration."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import pandas as pd

from brim_enrichment import run_overrepresentation
from brim_multiomics import IntegrationError, extract_gene_set


ORA_CLASSES = frozenset({
    "concordant_activation", "concordant_repression", "discordant_open_down",
    "discordant_closed_up", "atac_only", "rna_only_on_mapped_peak",
    "mixed_accessibility",
})

_LIBRARIES = {
    "Human": {
        "KEGG": {"status": "available", "library": "KEGG_2021_Human"},
        "GO_BP": {"status": "available", "library": "GO_Biological_Process_2021"},
    },
    "Mouse": {
        "KEGG": {"status": "available", "library": "KEGG_2019_Mouse"},
        "GO_BP": {
            "status": "unavailable",
            "reason": "Mouse GO Biological Process is unavailable because the bundled GO library contains human gene symbols.",
            "reason_ja": "同梱GOライブラリがヒト遺伝子シンボル用のため、Mouse GO Biological Processは利用できません。",
        },
    },
}


BACKGROUND_DEFINITION = (
    "ORA background: genes with RNA padj and log2FC both tested (not NA) and at least one tested ATAC peak mapped; "
    "not-significant genes are included. It is neither all genes nor only the significant genes."
)
BACKGROUND_DEFINITION_JA = (
    "ORAの背景遺伝子: RNAのpadjとlog2FCがともに検定済み（NAでない）で、検定済みATAC peakが1つ以上対応付いた遺伝子。"
    "有意でない遺伝子も含みます。全遺伝子でも、有意な遺伝子のみでもありません。"
)
DIRECTION_AGNOSTIC_NOTE = (
    "mixed_accessibility contains genes with both opening and closing peaks; this ORA does not assume a single direction."
)
DIRECTION_AGNOSTIC_NOTE_JA = (
    "mixed_accessibilityは開くpeakと閉じるpeakの両方を持つ遺伝子です。このORAは単一の方向を仮定しません。"
)


def _species_libraries(species: str) -> Mapping[str, Mapping[str, str]]:
    if species not in _LIBRARIES:
        raise IntegrationError("Integration enrichment species must be Human or Mouse.")
    return _LIBRARIES[species]


def _unique_symbols(values: pd.Series) -> list[str]:
    return sorted({str(value).strip() for value in values.dropna() if str(value).strip()})


def build_ora_background(summary: pd.DataFrame) -> list[str]:
    """Return the tested, mapped gene-summary universe for new ORA tests."""
    required = {"gene_symbol", "rna_padj_is_na", "rna_lfc_is_na", "n_atac_tested_peaks"}
    missing = required.difference(summary.columns)
    if missing:
        raise IntegrationError("Gene summary missing ORA background columns: " + ", ".join(sorted(missing)))
    eligible = (
        ~summary["rna_padj_is_na"].astype(bool)
        & ~summary["rna_lfc_is_na"].astype(bool)
        & pd.to_numeric(summary["n_atac_tested_peaks"], errors="coerce").fillna(0).gt(0)
    )
    return _unique_symbols(summary.loc[eligible, "gene_symbol"])


def prepare_class_ora(summary: pd.DataFrame, integration_class: str) -> tuple[list[str], list[str], list[str]]:
    """Validate one allowed class and return its genes, background, and warnings."""
    if integration_class not in ORA_CLASSES:
        raise IntegrationError("ORA is unavailable for integration class: " + integration_class)
    genes = extract_gene_set(summary, integration_class)
    background = build_ora_background(summary)
    if not set(genes).issubset(background):
        # A mixed or significance class may be present but untested in a separate modality;
        # those genes are deliberately excluded rather than silently altering the universe.
        genes = sorted(set(genes).intersection(background))
    warnings = []
    if len(genes) < 20:
        warnings.append("The selected gene set has fewer than 20 eligible genes; ORA is exploratory.")
    return genes, background, warnings


def run_class_ora(
    summary: pd.DataFrame, integration_class: str, species: str,
    runner: Callable[[list[str], str, list[str]], pd.DataFrame] = run_overrepresentation,
) -> dict[str, Any]:
    """Run only available local libraries and preserve unavailable/zero-overlap evidence."""
    genes, background, warnings = prepare_class_ora(summary, integration_class)
    if not genes:
        raise IntegrationError("The selected integration class has no eligible tested, mapped genes for ORA.")
    results: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    for label, spec in _species_libraries(species).items():
        record = {"library_type": label, **dict(spec)}
        if record["status"] == "unavailable":
            results[label] = record
            history.append(record)
            continue
        try:
            frame = runner(genes, record["library"], background)
        except ValueError as error:
            if "No pathway overlaps" not in str(error):
                raise
            record.update(status="zero_overlap", result_count=0)
        else:
            record.update(status="executed", result_count=int(len(frame)), results=frame)
        results[label] = record
        history.append({key: value for key, value in record.items() if key != "results"})
    return {
        "integration_class": integration_class,
        "species": species,
        "input_genes": genes,
        "background_genes": background,
        "background_size": len(background),
        "background_definition": BACKGROUND_DEFINITION,
        "background_definition_ja": BACKGROUND_DEFINITION_JA,
        "direction_note": DIRECTION_AGNOSTIC_NOTE if integration_class == "mixed_accessibility" else None,
        "direction_note_ja": DIRECTION_AGNOSTIC_NOTE_JA if integration_class == "mixed_accessibility" else None,
        "warnings": warnings,
        "libraries": results,
        "history": history,
        "independent_test_notice": "ORA adjusted p-values are new, independent tests and are not combined with RNA or ATAC adjusted p-values.",
    }
