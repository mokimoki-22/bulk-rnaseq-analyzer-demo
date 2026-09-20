"""Streamlit-free Level 1 RNA--ATAC integration logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


class IntegrationError(ValueError):
    """Base error for a non-integrable RNA--ATAC input."""


class GeneIdentifierError(IntegrationError):
    """Raised when RNA gene identifiers are absent, ambiguous, or unsupported."""


class CompatibilityError(IntegrationError):
    """Raised when explicit RNA and ATAC metadata cannot be integrated."""


@dataclass(frozen=True)
class CompatibilityResult:
    """Explicit compatibility evidence; callers must not infer missing metadata."""

    compatible: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)
    rates: Mapping[str, float] = field(default_factory=dict)


_RNA_REQUIRED = {"log2FoldChange", "padj", "padj_is_na", "lfc_is_na"}
_EDGE_REQUIRED = {
    "peak_id", "gene_id", "gene_symbol", "atac_log2FoldChange", "atac_padj",
    "atac_padj_is_na", "atac_lfc_is_na",
}


def _as_bool(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise IntegrationError(f"{name} must not contain missing values.")
    if not series.isin([True, False, 0, 1]).all():
        raise IntegrationError(f"{name} must contain boolean values.")
    return series.astype(bool)


def _validate_thresholds(thresholds: Mapping[str, Any]) -> tuple[float, float, float, float]:
    keys = ("rna_padj", "rna_lfc", "atac_padj", "atac_lfc")
    if set(keys).difference(thresholds):
        raise IntegrationError("Thresholds require rna_padj, rna_lfc, atac_padj, and atac_lfc.")
    values = tuple(float(thresholds[key]) for key in keys)
    if not all(np.isfinite(values)):
        raise IntegrationError("Thresholds must be finite numeric values.")
    if not 0 <= values[0] <= 1 or not 0 <= values[2] <= 1:
        raise IntegrationError("padj thresholds must be between 0 and 1.")
    if values[1] < 0 or values[3] < 0:
        raise IntegrationError("Absolute log2FC thresholds must be non-negative.")
    return values


def standardize_rna_results(deg_results: pd.DataFrame, gene_id_type: str) -> pd.DataFrame:
    """Validate explicitly declared RNA identifiers and preserve NA provenance.

    `gene_id` accepts only explicit Ensembl version-suffix removal; gene symbols
    are never case-normalized or alias-expanded.
    """
    if gene_id_type not in {"gene_id", "gene_symbol"}:
        raise GeneIdentifierError("gene_id_type must be 'gene_id' or 'gene_symbol'.")
    missing = _RNA_REQUIRED.difference(deg_results.columns)
    if missing:
        raise GeneIdentifierError("RNA results missing required columns: " + ", ".join(sorted(missing)))
    result = deg_results.copy()
    source = result[gene_id_type] if gene_id_type in result else pd.Series(result.index, index=result.index)
    if source.isna().any() or source.astype(str).str.strip().eq("").any():
        raise GeneIdentifierError(f"RNA {gene_id_type} values must be non-empty.")
    result["gene_key"] = source.astype(str).str.strip()
    transforms: list[dict[str, Any]] = []
    if gene_id_type == "gene_id":
        stripped = result["gene_key"].str.replace(r"^(ENS[A-Z]*[GTP]\d+)\.\d+$", r"\1", regex=True)
        changed = int((stripped != result["gene_key"]).sum())
        result["gene_key"] = stripped
        transforms.append({"name": "ensembl_version_suffix_removed", "applied": changed > 0,
                           "affected_rows": changed, "total_rows": len(result), "success_rate": 1.0})
    if result["gene_key"].duplicated().any():
        duplicates = result.loc[result["gene_key"].duplicated(), "gene_key"].head(5).tolist()
        raise GeneIdentifierError("RNA gene identifiers are not unique: " + ", ".join(duplicates))
    for column in ("log2FoldChange", "padj"):
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise GeneIdentifierError(f"RNA {column} must be finite numeric values after DEG NA handling.")
        result[column] = values.astype(float)
    if not result["padj"].between(0, 1).all():
        raise GeneIdentifierError("RNA padj must be between 0 and 1.")
    result["padj_is_na"] = _as_bool(result["padj_is_na"], "RNA padj_is_na")
    result["lfc_is_na"] = _as_bool(result["lfc_is_na"], "RNA lfc_is_na")
    result.attrs["gene_id_type"] = gene_id_type
    result.attrs["transforms"] = transforms
    return result


def _normal_species(value: Any) -> str | None:
    aliases = {"human": "human", "homo sapiens": "human", "mouse": "mouse", "mus musculus": "mouse"}
    return aliases.get(str(value).strip().lower())


def _contrast(meta: Mapping[str, Any], label: str, errors: list[str]) -> tuple[str, str] | None:
    value = meta.get("contrast")
    if not isinstance(value, Mapping) or not value.get("reference") or not value.get("test"):
        errors.append(f"{label} requires structured contrast metadata with reference and test.")
        return None
    return str(value["reference"]), str(value["test"])


def check_integration_compatibility(rna_meta: Mapping[str, Any], atac_meta: Mapping[str, Any]) -> CompatibilityResult:
    """Return explicit compatibility evidence without parsing display labels."""
    errors: list[str] = []
    warnings: list[str] = []
    rna_species, atac_species = _normal_species(rna_meta.get("species")), _normal_species(atac_meta.get("species"))
    if rna_species is None or atac_species is None:
        errors.append("RNA and ATAC species must be explicitly declared as Human or Mouse.")
    elif rna_species != atac_species:
        errors.append("RNA and ATAC species do not match.")
    build = atac_meta.get("genome_build")
    if not isinstance(build, str) or not build.strip():
        errors.append("ATAC genome build must be explicitly declared.")
    rna_build = rna_meta.get("genome_build", "not_applicable")
    if rna_build != "not_applicable":
        errors.append("RNA count-based DEG metadata must declare genome_build as not_applicable.")
    rna_contrast = _contrast(rna_meta, "RNA", errors)
    atac_contrast = _contrast(atac_meta, "ATAC", errors)
    if rna_contrast and atac_contrast and rna_contrast != atac_contrast:
        errors.append("RNA and ATAC contrast direction does not match.")
    rna_genes = set(map(str, rna_meta.get("gene_keys", ())))
    atac_genes = set(map(str, atac_meta.get("gene_keys", ())))
    if not rna_genes or not atac_genes:
        errors.append("RNA and ATAC gene keys are required for compatibility checking.")
    shared = rna_genes & atac_genes
    if rna_genes and atac_genes and not shared:
        errors.append("RNA and ATAC have zero shared gene identifiers.")
    rna_rate = len(shared) / len(rna_genes) if rna_genes else 0.0
    atac_rate = len(shared) / len(atac_genes) if atac_genes else 0.0
    if shared and (rna_rate < 0.8 or atac_rate < 0.8):
        warnings.append("Integration uses shared IDs only; low matching may reflect identifier, species, annotation, or input mismatch and is not biological absence.")
    return CompatibilityResult(not errors, tuple(errors), tuple(warnings),
                               {"n_rna_unique_genes": len(rna_genes), "n_atac_unique_mapped_genes": len(atac_genes),
                                "n_shared_genes": len(shared)},
                               {"rna_gene_match_rate": rna_rate, "atac_gene_match_rate": atac_rate})


def integrate_peak_gene_edges(rna: pd.DataFrame, edges: pd.DataFrame, thresholds: Mapping[str, Any]) -> pd.DataFrame:
    """Join shared RNA--ATAC evidence without collapsing one-to-many edges."""
    _validate_thresholds(thresholds)
    if "gene_key" not in rna or rna["gene_key"].duplicated().any():
        raise IntegrationError("RNA results must be standardized with unique gene_key values.")
    missing = _EDGE_REQUIRED.difference(edges.columns)
    if missing:
        raise IntegrationError("Peak--gene edges missing required columns: " + ", ".join(sorted(missing)))
    key_type = rna.attrs.get("gene_id_type")
    if key_type not in {"gene_id", "gene_symbol"}:
        raise IntegrationError("RNA gene identifier type is missing.")
    edge_key = key_type
    source = edges.copy()
    source["gene_key"] = source[edge_key].astype(str).str.strip()
    if source["gene_key"].eq("").any():
        raise IntegrationError("Peak--gene edge identifiers must be non-empty.")
    for column in ("gene_id", "gene_symbol", "edge_id"):
        if column not in source or source[column].isna().any() or source[column].astype(str).str.strip().eq("").any():
            raise IntegrationError(f"Peak--gene {column} values must be non-empty.")
    if source["edge_id"].duplicated().any():
        raise IntegrationError("Peak--gene edge_id values must be unique and non-missing.")
    for column in ("atac_log2FoldChange", "atac_padj"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
        if source[column].isna().any() or not np.isfinite(source[column]).all():
            raise IntegrationError(f"{column} must be finite numeric values.")
    if not source["atac_padj"].between(0, 1).all():
        raise IntegrationError("ATAC padj must be between 0 and 1.")
    source["atac_padj_is_na"] = _as_bool(source["atac_padj_is_na"], "ATAC padj_is_na")
    source["atac_lfc_is_na"] = _as_bool(source["atac_lfc_is_na"], "ATAC lfc_is_na")
    rna_columns = ["gene_key", "log2FoldChange", "padj", "padj_is_na", "lfc_is_na"]
    joined = source.merge(rna.loc[:, rna_columns], on="gene_key", how="inner", validate="many_to_one")
    joined = joined.rename(columns={"log2FoldChange": "rna_log2FoldChange", "padj": "rna_padj",
                                    "padj_is_na": "rna_padj_is_na", "lfc_is_na": "rna_lfc_is_na"})
    joined.attrs["rna_results"] = rna.copy()
    joined.attrs["thresholds"] = dict(thresholds)
    return joined


def classify_integration_edges(integrated: pd.DataFrame, thresholds: Mapping[str, Any]) -> pd.DataFrame:
    """Classify each preserved edge using independent modality thresholds."""
    rpadj, rlfc, apadj, alfc = _validate_thresholds(thresholds)
    needed = {"rna_log2FoldChange", "rna_padj", "rna_padj_is_na", "rna_lfc_is_na",
              "atac_log2FoldChange", "atac_padj", "atac_padj_is_na", "atac_lfc_is_na"}
    if needed.difference(integrated.columns):
        raise IntegrationError("Integrated edges do not contain the complete NA-preserving contract.")
    result = integrated.copy()
    for column in ("rna_padj_is_na", "rna_lfc_is_na", "atac_padj_is_na", "atac_lfc_is_na"):
        result[column] = _as_bool(result[column], column)
    rna_not = result["rna_padj_is_na"] | result["rna_lfc_is_na"]
    atac_not = result["atac_padj_is_na"] | result["atac_lfc_is_na"]
    rna_sig = ~rna_not & result["rna_padj"].le(rpadj) & result["rna_log2FoldChange"].abs().ge(rlfc) & result["rna_log2FoldChange"].ne(0)
    atac_sig = ~atac_not & result["atac_padj"].le(apadj) & result["atac_log2FoldChange"].abs().ge(alfc) & result["atac_log2FoldChange"].ne(0)
    classes = pd.Series("not_significant", index=result.index, dtype="object")
    classes.loc[rna_not & atac_not] = "both_not_tested"
    classes.loc[rna_not & ~atac_not] = "rna_not_tested"
    classes.loc[~rna_not & atac_not] = "atac_not_tested"
    normal = ~rna_not & ~atac_not
    classes.loc[normal & ~rna_sig & atac_sig] = "atac_only"
    classes.loc[normal & rna_sig & ~atac_sig] = "rna_only_on_mapped_peak"
    both = normal & rna_sig & atac_sig
    classes.loc[both & result["rna_log2FoldChange"].gt(0) & result["atac_log2FoldChange"].gt(0)] = "concordant_activation"
    classes.loc[both & result["rna_log2FoldChange"].lt(0) & result["atac_log2FoldChange"].lt(0)] = "concordant_repression"
    classes.loc[both & result["rna_log2FoldChange"].lt(0) & result["atac_log2FoldChange"].gt(0)] = "discordant_open_down"
    classes.loc[both & result["rna_log2FoldChange"].gt(0) & result["atac_log2FoldChange"].lt(0)] = "discordant_closed_up"
    result["rna_significant"] = rna_sig
    result["atac_significant"] = atac_sig
    result["integration_class"] = classes
    result.attrs = integrated.attrs.copy()
    return result


def summarize_integration_by_gene(integrated: pd.DataFrame, thresholds: Mapping[str, Any]) -> pd.DataFrame:
    """Summarize preserved edges without resolving mixed accessibility by vote."""
    _validate_thresholds(thresholds)
    if "integration_class" not in integrated:
        integrated = classify_integration_edges(integrated, thresholds)
    rna = integrated.attrs.get("rna_results")
    rows: list[dict[str, Any]] = []
    for gene_key, group in integrated.groupby("gene_key", sort=True):
        first = group.iloc[0]
        rna_not = bool(first.rna_padj_is_na or first.rna_lfc_is_na)
        peak_states = group.drop_duplicates("peak_id").copy()
        atac_not = peak_states["atac_padj_is_na"].astype(bool) | peak_states["atac_lfc_is_na"].astype(bool)
        tested = peak_states.loc[~atac_not]
        significant = tested.loc[tested["atac_significant"].astype(bool)]
        opening = significant.loc[significant["atac_log2FoldChange"].gt(0)]
        closing = significant.loc[significant["atac_log2FoldChange"].lt(0)]
        pattern = "mixed_accessibility" if not opening.empty and not closing.empty else (
            "opening" if not opening.empty else "closing" if not closing.empty else "no_significant_peak")
        if rna_not and tested.empty:
            summary_class = "both_not_tested"
        elif rna_not:
            summary_class = "rna_not_tested"
        elif tested.empty:
            summary_class = "atac_not_tested"
        elif pattern == "mixed_accessibility":
            summary_class = "mixed_accessibility"
        elif significant.empty:
            summary_class = "rna_only_on_mapped_peak" if bool(first.rna_significant) else "not_significant"
        elif not bool(first.rna_significant):
            summary_class = "atac_only"
        elif pattern == "opening" and first.rna_log2FoldChange > 0:
            summary_class = "concordant_activation"
        elif pattern == "closing" and first.rna_log2FoldChange < 0:
            summary_class = "concordant_repression"
        elif pattern == "opening":
            summary_class = "discordant_open_down"
        else:
            summary_class = "discordant_closed_up"
        representative = significant if not significant.empty else tested
        representative_id: str | None = None
        if not representative.empty:
            representative = representative.assign(_abs=representative["atac_log2FoldChange"].abs())
            representative = representative.sort_values(["atac_padj", "_abs", "peak_id"], ascending=[True, False, True])
            representative_id = str(representative.iloc[0]["peak_id"])
        rows.append({"gene_key": gene_key, "gene_symbol": str(first.gene_symbol),
                     "rna_log2FoldChange": float(first.rna_log2FoldChange), "rna_padj": float(first.rna_padj),
                     "rna_padj_is_na": bool(first.rna_padj_is_na), "rna_lfc_is_na": bool(first.rna_lfc_is_na),
                     "n_mapped_peaks": int(len(peak_states)), "n_opening_peaks": int(len(opening)),
                     "n_closing_peaks": int(len(closing)), "n_significant_peaks": int(len(significant)),
                     "n_atac_padj_na_peaks": int(peak_states["atac_padj_is_na"].astype(bool).sum()),
                     "n_atac_lfc_na_peaks": int(peak_states["atac_lfc_is_na"].astype(bool).sum()),
                     "n_atac_not_tested_peaks": int(atac_not.sum()), "n_atac_tested_peaks": int((~atac_not).sum()),
                     "accessibility_pattern": pattern, "representative_peak_id": representative_id,
                     "representative_peak_rule": "smallest_atac_padj_then_largest_abs_lfc",
                     "source_edge_ids": sorted(group["edge_id"].astype(str).unique()), "integration_class": summary_class})
    summary = pd.DataFrame(rows)
    if isinstance(rna, pd.DataFrame):
        mapped = set(integrated["gene_key"])
        for record in rna.loc[~rna["gene_key"].isin(mapped)].itertuples(index=False):
            rna_not = bool(record.padj_is_na or record.lfc_is_na)
            rna_sig = (not rna_not and record.padj <= float(thresholds["rna_padj"])
                       and abs(record.log2FoldChange) >= float(thresholds["rna_lfc"])
                       and record.log2FoldChange != 0)
            summary.loc[len(summary)] = {"gene_key": record.gene_key, "gene_symbol": record.gene_key,
                "rna_log2FoldChange": record.log2FoldChange, "rna_padj": record.padj,
                "rna_padj_is_na": record.padj_is_na, "rna_lfc_is_na": record.lfc_is_na,
                "n_mapped_peaks": 0, "n_opening_peaks": 0, "n_closing_peaks": 0, "n_significant_peaks": 0,
                "n_atac_padj_na_peaks": 0, "n_atac_lfc_na_peaks": 0, "n_atac_not_tested_peaks": 0,
                "n_atac_tested_peaks": 0, "accessibility_pattern": "no_significant_peak",
                "representative_peak_id": None, "representative_peak_rule": "not_applicable",
                "source_edge_ids": [], "integration_class": "rna_not_tested" if rna_not else ("rna_only_no_mapped_peak" if rna_sig else "not_significant")}
    return summary.sort_values("gene_key", kind="stable").reset_index(drop=True)


def extract_gene_set(summary: pd.DataFrame, integration_class: str) -> list[str]:
    """Return sorted unique gene symbols for one explicitly selected class."""
    if not {"gene_symbol", "integration_class"}.issubset(summary):
        raise IntegrationError("Gene summary requires gene_symbol and integration_class.")
    return sorted(summary.loc[summary["integration_class"] == integration_class, "gene_symbol"].dropna().astype(str).unique())


def build_integration_summary(edges: pd.DataFrame, genes: pd.DataFrame, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return descriptive counts; no enrichment or p-value combination occurs here."""
    return {"edge_count": int(len(edges)), "gene_count": int(len(genes)),
            "edge_classes": {str(key): int(value) for key, value in edges["integration_class"].value_counts().items()},
            "gene_classes": {str(key): int(value) for key, value in genes["integration_class"].value_counts().items()},
            "settings": dict(settings)}
