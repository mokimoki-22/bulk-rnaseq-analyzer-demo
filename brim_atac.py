"""Streamlit-free ATAC-seq input, DAR, annotation, mapping, and export logic."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any, BinaryIO, Mapping, Sequence, TextIO

import numpy as np
import pandas as pd


REFERENCE_DIR = Path(__file__).resolve().parent / "references" / "genome_annotations"
PEAK_PATTERN = re.compile(r"^([^:\s]+):(\d+)-(\d+)$")
NORMALIZATIONS = {
    "deseq2_median_of_ratios",
    "total_reads_in_peaks",
    "user_supplied_size_factors",
}
DAR_ALIASES = {
    "chrom": ("chrom", "chr", "chromosome"),
    "start": ("start", "chromstart"),
    "end": ("end", "chromend", "stop"),
    "log2FoldChange": ("log2foldchange", "log2fc", "logfc"),
    "padj": ("padj", "fdr", "qvalue", "adj.p.val"),
    "pvalue": ("pvalue", "p.value", "pval"),
    "baseMean": ("basemean", "mean", "base_mean"),
    "stat": ("stat", "waldstat", "wald_stat"),
    "peak_id": ("peak_id", "peak", "region"),
}


class ATACError(ValueError):
    """Base class for stable ATAC validation errors."""

    code = "ATAC_ERROR"


class PeakCountMatrixError(ATACError):
    """Raised for an invalid peak count matrix."""

    code = "PEAK_COUNT_MATRIX_INVALID"


class DARSchemaError(ATACError):
    """Raised for an invalid DAR table."""

    code = "DAR_SCHEMA_INVALID"


class CoordinateSystemError(ATACError):
    """Raised for unsupported or invalid coordinates."""

    code = "COORDINATE_SYSTEM_INVALID"


class GenomeBuildError(ATACError):
    """Raised for an unsupported genome build."""

    code = "GENOME_BUILD_UNSUPPORTED"


class ReferenceAnnotationError(ATACError):
    """Raised when a pinned annotation is unavailable or invalid."""

    code = "REFERENCE_ANNOTATION_INVALID"


class InsufficientSampleError(ATACError):
    """Raised when a contrast has fewer than two samples in either group."""

    code = "INSUFFICIENT_SAMPLES"


@dataclass(frozen=True)
class ValidationResult:
    """Schema-validation details with the standardized table when valid."""

    valid: bool
    data: pd.DataFrame | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    transforms: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TransformLog:
    """Record an explicit coordinate/chromosome conversion."""

    name: str
    applied: bool
    affected_rows: int
    total_rows: int
    success_rate: float
    details: Mapping[str, Any] = field(default_factory=dict)


def _transform_record(name: str, affected_rows: int, total_rows: int,
                      details: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "applied": affected_rows > 0,
        "affected_rows": affected_rows,
        "total_rows": total_rows,
        "success_rate": 1.0,
        "details": dict(details),
    }


def _read_frame(file_obj: BinaryIO | TextIO | bytes | str, sep: str) -> pd.DataFrame:
    if sep not in {",", "\t"}:
        raise ValueError("Separator must be comma or tab.")
    if isinstance(file_obj, bytes):
        file_obj = io.BytesIO(file_obj)
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    return pd.read_csv(file_obj, sep=sep)


def parse_peak_coordinates(index_or_columns: pd.Index | pd.Series | pd.DataFrame | Sequence[str]) -> pd.DataFrame:
    """Parse index-style peaks or validate explicit chrom/start/end columns."""
    if isinstance(index_or_columns, pd.DataFrame):
        missing = {"chrom", "start", "end"}.difference(index_or_columns.columns)
        if missing:
            raise CoordinateSystemError("Missing coordinate columns: " + ", ".join(sorted(missing)))
        result = index_or_columns[["chrom", "start", "end"]].copy()
    else:
        values = pd.Index(index_or_columns).astype(str)
        parsed = values.to_series(index=range(len(values))).str.extract(PEAK_PATTERN)
        if parsed.isna().any().any():
            bad = values[parsed.isna().any(axis=1).to_numpy()].tolist()[:5]
            raise CoordinateSystemError("Invalid peak coordinate(s): " + ", ".join(bad))
        parsed.columns = ["chrom", "start", "end"]
        result = parsed
    result["chrom"] = result["chrom"].astype(str).str.strip()
    for column in ("start", "end"):
        numeric = pd.to_numeric(result[column], errors="coerce")
        if (numeric.isna().any() or not np.isfinite(numeric).all()
                or not np.equal(numeric, np.floor(numeric)).all()):
            raise CoordinateSystemError(f"{column} must contain integers.")
        result[column] = numeric.astype(np.int64)
    if (result["chrom"] == "").any() or (result["start"] < 0).any() or (result["end"] <= result["start"]).any():
        raise CoordinateSystemError("Coordinates require non-empty chrom, start >= 0, and end > start.")
    result["peak_id"] = result["chrom"] + ":" + result["start"].astype(str) + "-" + result["end"].astype(str)
    return result.reset_index(drop=True)


def read_peak_count_matrix(file_obj: BinaryIO | TextIO | bytes | str, sep: str,
                           coordinate_column_mode: str, coordinate_system: str) -> pd.DataFrame:
    """Read and validate an integer peak-by-sample count matrix."""
    raw = _read_frame(file_obj, sep)
    if coordinate_column_mode == "index":
        if raw.shape[1] < 2:
            raise PeakCountMatrixError("Index mode requires a coordinate column and samples.")
        coordinates = parse_peak_coordinates(raw.iloc[:, 0].astype(str))
        values = raw.iloc[:, 1:].copy()
    elif coordinate_column_mode == "columns":
        lowered = {str(column).strip().lower(): column for column in raw.columns}
        if not all(name in lowered for name in ("chrom", "start", "end")):
            raise PeakCountMatrixError("Column mode requires chrom, start, and end.")
        source_columns = [lowered[name] for name in ("chrom", "start", "end")]
        coordinates = parse_peak_coordinates(raw[source_columns].set_axis(["chrom", "start", "end"], axis=1))
        values = raw.drop(columns=source_columns)
    else:
        raise PeakCountMatrixError("coordinate_column_mode must be 'index' or 'columns'.")
    if coordinate_system not in {"0-based", "1-based"}:
        raise CoordinateSystemError("coordinate_system must be '0-based' or '1-based'.")
    transforms: list[dict[str, Any]] = []
    if coordinate_system == "1-based":
        coordinates["start"] -= 1
        if (coordinates["start"] < 0).any():
            raise CoordinateSystemError("1-based coordinates require start >= 1.")
        coordinates["peak_id"] = (
            coordinates["chrom"] + ":" + coordinates["start"].astype(str)
            + "-" + coordinates["end"].astype(str)
        )
        transforms.append(_transform_record(
            "one_based_closed_to_zero_based_half_open", len(coordinates), len(coordinates),
            {"source": "1-based closed", "target": "0-based half-open"},
        ))
    if values.empty or values.columns.duplicated().any():
        raise PeakCountMatrixError("At least one uniquely named sample column is required.")
    numeric = values.apply(pd.to_numeric, errors="coerce")
    if (numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all()
            or (numeric < 0).any().any() or not np.equal(numeric, np.floor(numeric)).all().all()):
        raise PeakCountMatrixError("Counts must be non-negative integers without missing values.")
    if (numeric.sum(axis=0) == 0).any():
        raise PeakCountMatrixError("Every sample must have a positive library size.")
    result = pd.concat([coordinates, numeric.astype(np.int64).reset_index(drop=True)], axis=1)
    if result["peak_id"].duplicated().any():
        raise PeakCountMatrixError("Duplicate peak coordinates are not allowed.")
    result.attrs["transforms"] = transforms
    return result


def _classify_dar(result: pd.DataFrame, padj_threshold: float,
                  lfc_threshold: float) -> pd.DataFrame:
    if not 0 <= padj_threshold <= 1:
        raise ValueError("padj_threshold must be between 0 and 1.")
    if lfc_threshold < 0:
        raise ValueError("lfc_threshold must be non-negative.")
    tested = ~result["padj_is_na"] & ~result["lfc_is_na"]
    significant = (
        tested
        & result["padj"].le(padj_threshold)
        & result["log2FoldChange"].abs().ge(lfc_threshold)
        & result["log2FoldChange"].ne(0)
    )
    result["is_significant"] = significant
    result["accessibility_direction"] = "not_significant"
    result.loc[~tested, "accessibility_direction"] = "not_tested"
    result.loc[significant & result["log2FoldChange"].gt(0), "accessibility_direction"] = "opening"
    result.loc[significant & result["log2FoldChange"].lt(0), "accessibility_direction"] = "closing"
    return result


def _standard_dar(data: pd.DataFrame, source_mode: str, padj_threshold: float,
                  lfc_threshold: float) -> pd.DataFrame:
    result = data.copy()
    result["padj_is_na"] = result["padj"].isna()
    result["lfc_is_na"] = result["log2FoldChange"].isna()
    result["padj"] = result["padj"].fillna(1.0)
    result["log2FoldChange"] = result["log2FoldChange"].fillna(0.0)
    if "pvalue" not in result:
        result["pvalue"] = np.nan
    for column in ("baseMean", "stat"):
        if column not in result:
            result[column] = np.nan
    result["source_mode"] = source_mode
    if "input_row" not in result:
        result["input_row"] = np.arange(len(result), dtype=np.int64)
    result = _classify_dar(result, padj_threshold, lfc_threshold)
    standard_columns = [
        "peak_id", "chrom", "start", "end", "log2FoldChange", "pvalue", "padj",
        "padj_is_na", "lfc_is_na", "is_significant", "accessibility_direction",
        "baseMean", "stat", "input_row", "source_mode",
    ]
    optional_columns = [column for column in ("gene", "annotation") if column in result]
    return result[standard_columns + optional_columns]


def _validate_size_factors(size_factors: Mapping[str, float] | pd.Series | None,
                           samples: pd.Index) -> pd.Series:
    if size_factors is None:
        raise ValueError("User-supplied size factors are required for the selected normalization.")
    factors = pd.Series(size_factors, dtype=float)
    missing = samples.difference(factors.index)
    extra = factors.index.difference(samples)
    if len(missing) or len(extra):
        details = []
        if len(missing):
            details.append("missing: " + ", ".join(map(str, missing)))
        if len(extra):
            details.append("unexpected: " + ", ".join(map(str, extra)))
        raise ValueError("Size factor sample mismatch (" + "; ".join(details) + ").")
    factors = factors.loc[samples]
    if not np.isfinite(factors).all() or factors.le(0).any():
        raise ValueError("Size factors must be finite positive numbers.")
    return factors / float(np.exp(np.log(factors).mean()))


def _fit_deseq2_with_fixed_size_factors(dds: Any, factors: pd.Series) -> None:
    """Run PyDESeq2's public fitting stages with externally selected factors."""
    dds.obs["size_factors"] = factors.loc[dds.obs_names].to_numpy()
    dds.layers["normed_counts"] = dds.X / dds.obs["size_factors"].to_numpy()[:, None]
    dds.var["_normed_means"] = dds.layers["normed_counts"].mean(axis=0)
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.calculate_cooks()
    if dds.refit_cooks:
        dds.refit()
    dds.cooks_outlier()


def run_dar(counts_df: pd.DataFrame, metadata: pd.DataFrame, ref_condition: str,
            test_condition: str, normalization: str, n_cpus: int,
            padj_threshold: float, lfc_threshold: float,
            prefilter_enabled: bool = False, prefilter_total_count: int = 10,
            size_factors: Mapping[str, float] | pd.Series | None = None) -> pd.DataFrame:
    """Estimate DARs with an explicitly selected normalization method."""
    if normalization not in NORMALIZATIONS:
        raise ValueError("Unsupported normalization. Explicitly select a documented method.")
    if n_cpus < 1:
        raise ValueError("n_cpus must be at least 1.")
    if prefilter_total_count < 0:
        raise ValueError("prefilter_total_count must be non-negative.")
    required = {"peak_id", "chrom", "start", "end"}
    if not required.issubset(counts_df.columns):
        raise PeakCountMatrixError("Count matrix must include standardized peak coordinates.")
    if ref_condition == test_condition:
        raise ValueError("Reference and test conditions must differ.")
    if "condition" not in metadata or metadata.index.duplicated().any():
        raise PeakCountMatrixError("Metadata requires a condition column and unique sample names.")
    if counts_df["peak_id"].duplicated().any():
        raise PeakCountMatrixError("Duplicate peak_id values are not allowed.")
    parse_peak_coordinates(counts_df[["chrom", "start", "end"]])
    sample_columns = [column for column in counts_df.columns if column not in required]
    selected_metadata = metadata.loc[metadata["condition"].isin([ref_condition, test_condition])].copy()
    group_counts = selected_metadata["condition"].value_counts()
    if any(group_counts.get(group, 0) < 2 for group in (ref_condition, test_condition)):
        raise InsufficientSampleError("DAR requires at least two samples in each contrast group.")
    missing = selected_metadata.index.difference(sample_columns)
    if len(missing):
        raise PeakCountMatrixError("Metadata samples missing from count matrix: " + ", ".join(map(str, missing)))
    values = counts_df.set_index("peak_id").loc[:, selected_metadata.index].apply(pd.to_numeric, errors="coerce")
    if (values.isna().any().any() or not np.isfinite(values.to_numpy()).all() or (values < 0).any().any()
            or not np.equal(values, np.floor(values)).all().all()):
        raise PeakCountMatrixError("Counts must be non-negative integers without missing values.")
    values = values.astype(np.int64)
    if prefilter_enabled:
        keep = values.sum(axis=1) >= prefilter_total_count
    else:
        keep = pd.Series(True, index=values.index)
    filtered = values.loc[keep]
    if filtered.empty:
        raise PeakCountMatrixError("Pre-filtering removed every peak.")
    if filtered.sum(axis=0).eq(0).any():
        raise PeakCountMatrixError("Every selected sample must retain a positive library size after filtering.")
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dds = DeseqDataSet(counts=filtered.T, metadata=selected_metadata, design="~condition",
                       refit_cooks=True, n_cpus=n_cpus)
    if normalization == "deseq2_median_of_ratios":
        if size_factors is not None:
            raise ValueError("size_factors may only be supplied with user_supplied_size_factors.")
        dds.deseq2()
        fitted_factors = pd.Series(dds.obs["size_factors"], index=dds.obs_names, dtype=float)
    else:
        if normalization == "total_reads_in_peaks":
            if size_factors is not None:
                raise ValueError("size_factors may only be supplied with user_supplied_size_factors.")
            libraries = filtered.sum(axis=0).astype(float)
            fitted_factors = libraries / float(np.exp(np.log(libraries).mean()))
        else:
            fitted_factors = _validate_size_factors(size_factors, selected_metadata.index)
        _fit_deseq2_with_fixed_size_factors(dds, fitted_factors)
    stats = DeseqStats(dds, contrast=["condition", test_condition, ref_condition], n_cpus=n_cpus)
    stats.summary()
    raw = stats.results_df.copy()
    source_rows = pd.Series(np.arange(len(counts_df), dtype=np.int64), index=counts_df["peak_id"])
    coordinates = counts_df.set_index("peak_id").loc[raw.index, ["chrom", "start", "end"]]
    coordinates["input_row"] = source_rows.loc[raw.index].to_numpy()
    result = _standard_dar(coordinates.join(raw).reset_index(), "count_matrix",
                           padj_threshold, lfc_threshold).sort_values("padj", kind="stable")
    minimum_n = min(group_counts[ref_condition], group_counts[test_condition])
    if minimum_n == 2:
        warnings = ["At least one group has n=2; dispersion estimates are unstable and results are exploratory."]
    elif minimum_n == 3:
        warnings = ["At least one group has n=3; statistical power is limited for ATAC-seq."]
    else:
        warnings = []
    result.attrs["warnings"] = warnings
    result.attrs["sample_size_warning"] = bool(warnings)
    result.attrs["minimum_group_size"] = int(minimum_n)
    result.attrs["normalization"] = normalization
    result.attrs["size_factors"] = {str(sample): float(fitted_factors.loc[sample])
                                    for sample in selected_metadata.index}
    result.attrs["prefilter"] = {"enabled": prefilter_enabled, "total_count_threshold": prefilter_total_count,
                                  "input_peaks": len(values), "excluded_peaks": int((~keep).sum()),
                                  "analyzed_peaks": len(filtered)}
    result.attrs["samples_by_condition"] = {str(key): int(value) for key, value in group_counts.items()}
    return result


def read_dar_table(file_obj: BinaryIO | TextIO | bytes | str, sep: str,
                   padj_threshold: float, lfc_threshold: float,
                   coordinate_system: str,
                   column_map: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Read a DAR table; ``column_map`` maps standard names to source names."""
    raw = _read_frame(file_obj, sep)
    renamed = raw.rename(columns={source: target for target, source in (column_map or {}).items()})
    normalized = {str(column).strip().lower(): column for column in renamed.columns}
    for target, aliases in DAR_ALIASES.items():
        if target not in renamed:
            matches = [normalized[alias] for alias in aliases if alias in normalized]
            if len(matches) == 1:
                renamed = renamed.rename(columns={matches[0]: target})
    validation = validate_dar_table(renamed, coordinate_system, padj_threshold, lfc_threshold)
    if not validation.valid:
        raise DARSchemaError("; ".join(validation.errors))
    return validation.data


def validate_dar_table(df: pd.DataFrame, coordinate_system: str,
                       padj_threshold: float, lfc_threshold: float) -> ValidationResult:
    """Validate and standardize an analyzed DAR table without dropping NA padj."""
    errors = []
    required = {"chrom", "start", "end", "log2FoldChange", "padj"}
    missing = required.difference(df.columns)
    if missing:
        return ValidationResult(False, errors=("Missing required columns: " + ", ".join(sorted(missing)),))
    if coordinate_system not in {"0-based", "1-based"}:
        return ValidationResult(False, errors=("coordinate_system must be '0-based' or '1-based'.",))
    result = df.copy()
    for column in ("start", "end", "log2FoldChange", "padj", "pvalue", "baseMean", "stat"):
        if column not in result:
            continue
        original = result[column]
        converted = pd.to_numeric(original, errors="coerce")
        if (original.notna() & converted.isna()).any():
            errors.append(f"{column} must be numeric when provided.")
        result[column] = converted
    if result["chrom"].isna().any() or result["chrom"].astype(str).str.strip().eq("").any():
        errors.append("chrom must be non-empty and non-missing.")
    if result[["start", "end", "log2FoldChange"]].isna().any().any():
        errors.append("Coordinates and log2FoldChange must be numeric and non-missing.")
    if not np.isfinite(result[["start", "end"]].dropna().to_numpy()).all():
        errors.append("Coordinates must be finite.")
    finite_lfc = np.isfinite(result["log2FoldChange"].fillna(np.inf))
    if not finite_lfc.all():
        errors.append("log2FoldChange must be finite.")
    if result["padj"].dropna().lt(0).any() or result["padj"].dropna().gt(1).any():
        errors.append("padj must be between 0 and 1 or NA.")
    if "pvalue" in result and (result["pvalue"].dropna().lt(0).any() or result["pvalue"].dropna().gt(1).any()):
        errors.append("pvalue must be between 0 and 1 or NA.")
    if "baseMean" in result and (result["baseMean"].dropna().lt(0).any()
                                  or not np.isfinite(result["baseMean"].dropna()).all()):
        errors.append("baseMean must be finite, non-negative, or NA.")
    if "stat" in result and not np.isfinite(result["stat"].dropna()).all():
        errors.append("stat must be finite or NA.")
    for column in ("start", "end"):
        if result[column].notna().any() and not np.equal(result[column].dropna(), np.floor(result[column].dropna())).all():
            errors.append(f"{column} must contain integers.")
    transforms = []
    if coordinate_system == "1-based" and not errors:
        result["start"] -= 1
        transforms.append(_transform_record(
            "one_based_closed_to_zero_based_half_open", len(result), len(result),
            {"source": "1-based closed", "target": "0-based half-open"},
        ))
    if not errors and ((result["start"] < 0).any() or (result["end"] <= result["start"]).any()):
        errors.append("Coordinates require start >= 0 and end > start.")
    if errors:
        return ValidationResult(False, errors=tuple(dict.fromkeys(errors)), transforms=tuple(transforms))
    result["start"] = result["start"].astype(np.int64)
    result["end"] = result["end"].astype(np.int64)
    if "peak_id" not in result:
        result["peak_id"] = (result["chrom"].astype(str) + ":" + result["start"].astype(str)
                             + "-" + result["end"].astype(str))
    if (result["peak_id"].isna().any() or result["peak_id"].astype(str).str.strip().eq("").any()):
        return ValidationResult(False, errors=("peak_id must be non-empty and non-missing.",))
    if result["peak_id"].duplicated().any():
        return ValidationResult(False, errors=("Duplicate peak_id values are not allowed.",))
    return ValidationResult(
        True,
        _standard_dar(result, "dar_table", padj_threshold, lfc_threshold),
        transforms=tuple(transforms),
    )


def standardize_chromosomes(df: pd.DataFrame, build: str) -> tuple[pd.DataFrame, TransformLog]:
    """Explicitly standardize chromosome names for a selected build."""
    aliases = {"GRCh38": "hg38", "GRCm38": "mm10"}
    selected = aliases.get(build, build)
    if selected not in {"hg38", "mm10"}:
        raise GenomeBuildError("Supported builds are hg38/GRCh38 and mm10/GRCm38.")
    result = df.copy()
    if "chrom" not in result:
        raise CoordinateSystemError("chrom column is required.")
    if result["chrom"].isna().any():
        raise CoordinateSystemError("Chromosome names must not be missing.")
    original = result["chrom"].astype(str)
    if original.str.strip().eq("").any():
        raise CoordinateSystemError("Chromosome names must not be empty.")
    normalized = original.str.strip().str.replace(r"^chr", "", regex=True, flags=re.IGNORECASE)
    normalized = normalized.mask(normalized.str.upper().eq("MT"), "M")
    result["chrom"] = "chr" + normalized
    changed = result["chrom"] != original
    if "peak_id" in result and changed.any():
        if not {"start", "end"}.issubset(result.columns):
            raise CoordinateSystemError("start and end are required to update peak_id during chromosome conversion.")
        result["original_peak_id"] = result["peak_id"]
        result["peak_id"] = (
            result["chrom"] + ":" + result["start"].astype(str)
            + "-" + result["end"].astype(str)
        )
        if result["peak_id"].duplicated().any():
            raise CoordinateSystemError("Chromosome conversion produced duplicate peak_id values.")
    log = TransformLog("chromosome_names", bool(changed.any()), int(changed.sum()), len(result), 1.0,
                       {"build": selected, "rule": "add chr prefix; MT -> chrM"})
    return result, log


def load_gene_annotation(build: str) -> pd.DataFrame:
    """Load and checksum the pinned GENCODE table for an explicit build."""
    aliases = {"GRCh38": "hg38", "GRCm38": "mm10"}
    selected = aliases.get(build, build)
    if selected not in {"hg38", "mm10"}:
        raise GenomeBuildError("Supported builds are hg38/GRCh38 and mm10/GRCm38.")
    try:
        manifest = json.loads((REFERENCE_DIR / "manifest.json").read_text(encoding="utf-8"))
        record = manifest["files"][selected]
        path = REFERENCE_DIR / record["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ReferenceAnnotationError(f"Cannot load reference metadata for {selected}.") from error
    if digest != record["sha256"]:
        raise ReferenceAnnotationError(f"Reference checksum mismatch for {selected}.")
    table = pd.read_csv(path, sep="\t", compression="gzip")
    expected = {"chrom", "tss", "strand", "gene_id", "gene_symbol", "gene_type"}
    if not expected.issubset(table.columns) or table.empty or table["gene_id"].duplicated().any():
        raise ReferenceAnnotationError(f"Invalid reference table for {selected}.")
    table["reference_build"] = record["build"]
    table["reference_release"] = record["release"]
    table.attrs["reference"] = record
    return table


def _value(record: Any, name: str, default: Any = None) -> Any:
    return getattr(record, name, default)


def _edge(peak: Any, gene: Any, method: str, distance: int) -> dict[str, Any]:
    return {"peak_id": peak.peak_id, "chrom": peak.chrom, "start": int(peak.start), "end": int(peak.end),
            "gene_id": gene.gene_id, "gene_symbol": gene.gene_symbol, "mapping_method": method,
            "mapping_evidence_type": "distance_based_candidate" if method == "nearest_tss" else "promoter_overlap",
            "distance_to_tss": int(distance), "gene_strand": gene.strand,
            "reference_build": _value(gene, "reference_build"),
            "reference_release": _value(gene, "reference_release"),
            "atac_log2FoldChange": _value(peak, "log2FoldChange"),
            "atac_padj": _value(peak, "padj"),
            "atac_padj_is_na": _value(peak, "padj_is_na"),
            "atac_lfc_is_na": _value(peak, "lfc_is_na")}


EDGE_COLUMNS = [
    "peak_id", "chrom", "start", "end", "gene_id", "gene_symbol", "mapping_method",
    "mapping_evidence_type", "distance_to_tss", "gene_strand", "reference_build",
    "reference_release", "atac_log2FoldChange", "atac_padj", "atac_padj_is_na",
    "atac_lfc_is_na",
]


def map_peaks_to_promoters(peaks: pd.DataFrame, genes: pd.DataFrame, upstream: int,
                           downstream: int) -> pd.DataFrame:
    """Return every strand-aware promoter overlap as a separate edge."""
    if upstream < 0 or downstream < 0:
        raise ValueError("Promoter windows must be non-negative.")
    _validate_mapping_inputs(peaks, genes)
    rows = []
    for chrom, chromosome_peaks in peaks.groupby("chrom", sort=False):
        chromosome_genes = genes.loc[genes["chrom"] == chrom].copy()
        if chromosome_genes.empty:
            continue
        strands = {}
        for strand in ("+", "-"):
            strand_genes = chromosome_genes.loc[chromosome_genes["strand"] == strand].sort_values("tss")
            strands[strand] = (strand_genes["tss"].to_numpy(), list(strand_genes.itertuples(index=False)))
        for peak in chromosome_peaks.itertuples(index=False):
            for strand, (tss_values, gene_records) in strands.items():
                if strand == "+":
                    low, high = peak.start - downstream, peak.end + upstream
                else:
                    low, high = peak.start - upstream, peak.end + downstream
                left = np.searchsorted(tss_values, low, side="left")
                right = np.searchsorted(tss_values, high, side="left")
                for gene in gene_records[left:right]:
                    rows.append(_edge(peak, gene, "promoter", 0 if peak.start <= gene.tss < peak.end
                                      else min(abs(peak.start - gene.tss), abs(peak.end - 1 - gene.tss))))
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def map_peaks_to_nearest_tss(peaks: pd.DataFrame, genes: pd.DataFrame, max_distance: int) -> pd.DataFrame:
    """Map peaks to all equally nearest TSS genes within an explicit distance."""
    if max_distance < 0:
        raise ValueError("max_distance must be non-negative.")
    _validate_mapping_inputs(peaks, genes)
    rows = []
    for chrom, chromosome_peaks in peaks.groupby("chrom", sort=False):
        chromosome_genes = genes.loc[genes["chrom"] == chrom].sort_values("tss")
        if chromosome_genes.empty:
            continue
        tss = chromosome_genes["tss"].to_numpy()
        records = list(chromosome_genes.itertuples(index=False))
        for peak in chromosome_peaks.itertuples(index=False):
            left = int(np.searchsorted(tss, peak.start, side="left"))
            right = int(np.searchsorted(tss, peak.end, side="left"))
            candidates = list(range(left, right))
            if not candidates:
                if left:
                    left_start = int(np.searchsorted(tss, tss[left - 1], side="left"))
                    candidates.extend(range(left_start, left))
                if right < len(tss):
                    right_end = int(np.searchsorted(tss, tss[right], side="right"))
                    candidates.extend(range(right, right_end))
            distances = {index: (0 if peak.start <= tss[index] < peak.end else
                                 min(abs(peak.start - tss[index]), abs(peak.end - 1 - tss[index])))
                         for index in set(candidates)}
            if distances:
                nearest = min(distances.values())
                if nearest <= max_distance:
                    for index in sorted(i for i, distance in distances.items() if distance == nearest):
                        rows.append(_edge(peak, records[index], "nearest_tss", nearest))
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def _validate_mapping_inputs(peaks: pd.DataFrame, genes: pd.DataFrame) -> None:
    peak_columns = {"peak_id", "chrom", "start", "end"}
    gene_columns = {"chrom", "tss", "strand", "gene_id", "gene_symbol"}
    if not peak_columns.issubset(peaks.columns):
        raise CoordinateSystemError("Peaks require peak_id, chrom, start, and end columns.")
    if not gene_columns.issubset(genes.columns):
        raise ReferenceAnnotationError("Genes require chrom, tss, strand, gene_id, and gene_symbol columns.")
    if peaks["peak_id"].duplicated().any():
        raise CoordinateSystemError("Duplicate peak_id values are not allowed for mapping.")
    parse_peak_coordinates(peaks[["chrom", "start", "end"]])
    if not genes["strand"].isin(["+", "-"]).all():
        raise ReferenceAnnotationError("Gene strands must be '+' or '-'.")
    tss = pd.to_numeric(genes["tss"], errors="coerce")
    if (tss.isna().any() or not np.isfinite(tss).all()
            or not np.equal(tss, np.floor(tss)).all() or tss.lt(0).any()):
        raise ReferenceAnnotationError("Gene TSS coordinates must be non-negative integers.")
    if genes["gene_id"].isna().any() or genes["gene_id"].astype(str).str.strip().eq("").any():
        raise ReferenceAnnotationError("gene_id must be non-empty and non-missing.")


def merge_peak_gene_evidence(*edge_tables: pd.DataFrame) -> pd.DataFrame:
    """Merge only duplicate peak-gene evidence, preserving every biological edge."""
    nonempty = [table.copy() for table in edge_tables if table is not None and not table.empty]
    if not nonempty:
        return pd.DataFrame()
    combined = pd.concat(nonempty, ignore_index=True)
    key = ["peak_id", "gene_id"]
    if not set(key + ["mapping_method", "mapping_evidence_type"]).issubset(combined.columns):
        raise ValueError("Edge tables require peak_id, gene_id, mapping_method, and mapping_evidence_type.")
    rows = []
    for _, group in combined.groupby(key, sort=False, dropna=False):
        row = group.iloc[0].to_dict()
        methods = list(dict.fromkeys(group["mapping_method"].astype(str)))
        evidence_types = list(dict.fromkeys(group["mapping_evidence_type"].astype(str)))
        row["mapping_methods"] = methods
        row["mapping_evidence_types"] = evidence_types
        row["mapping_method"] = methods[0] if len(methods) == 1 else "multiple"
        row["edge_id"] = f"edge_{len(rows) + 1}"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_atac_qc(peaks: pd.DataFrame, edges: pd.DataFrame, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize counts and explicit settings without inferring thresholds."""
    mapped = set(edges["peak_id"]) if edges is not None and not edges.empty else set()
    input_peaks = len(peaks)
    summary = {
        "input_peaks": input_peaks,
        "mapped_peaks": len(mapped),
        "unmapped_peaks": input_peaks - len(mapped),
        "mapping_coverage": len(mapped) / input_peaks if input_peaks else 0.0,
        "mapping_edges": 0 if edges is None else len(edges),
        "settings": dict(settings),
    }
    if "padj_is_na" in peaks:
        summary["padj_is_na"] = int(peaks["padj_is_na"].astype(bool).sum())
    if "accessibility_direction" in peaks:
        summary["accessibility_directions"] = {
            str(key): int(value) for key, value in peaks["accessibility_direction"].value_counts().items()
        }
    if edges is not None and not edges.empty and "mapping_method" in edges:
        summary["mapping_methods"] = {
            str(key): int(value) for key, value in edges["mapping_method"].value_counts().items()
        }
    return summary


def export_peaks_as_bed(dar_df: pd.DataFrame, direction: str,
                        thresholds: Mapping[str, float]) -> str:
    """Export significant opening, closing, or all peaks as 0-based BED."""
    if direction not in {"opening", "closing", "all"}:
        raise ValueError("direction must be opening, closing, or all.")
    padj = float(thresholds["padj"])
    lfc = float(thresholds["log2FoldChange"])
    if not 0 <= padj <= 1 or lfc < 0:
        raise ValueError("BED thresholds require padj in [0, 1] and non-negative log2FoldChange.")
    if direction == "all":
        selected = dar_df
    else:
        tested = ~dar_df["padj_is_na"].astype(bool)
        if "lfc_is_na" in dar_df:
            tested &= ~dar_df["lfc_is_na"].astype(bool)
        selected = dar_df.loc[tested & dar_df["padj"].le(padj)]
        if direction == "opening":
            selected = selected.loc[selected["log2FoldChange"].ge(lfc)
                                    & selected["log2FoldChange"].gt(0)]
        else:
            selected = selected.loc[selected["log2FoldChange"].le(-lfc)
                                    & selected["log2FoldChange"].lt(0)]
    return "".join(f"{row.chrom}\t{int(row.start)}\t{int(row.end)}\t{row.peak_id}\n"
                   for row in selected.itertuples(index=False))
