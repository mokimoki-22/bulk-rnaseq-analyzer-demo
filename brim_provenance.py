"""Offline provenance construction shared by BRIM analysis workflows.

This module has no UI or session-state dependency. Missing information remains
None rather than being inferred from a filename, a threshold, or the host.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import hashlib
from importlib import metadata
import io
import json
import platform
from typing import Any, BinaryIO

import pandas as pd


def file_checksum(file_obj: BinaryIO | bytes) -> str:
    """Hash all original bytes with SHA-256, preserving a stream's position."""
    if isinstance(file_obj, bytes):
        return hashlib.sha256(file_obj).hexdigest()
    if not all(hasattr(file_obj, name) for name in ("read", "seek", "tell")):
        raise TypeError("Checksum input must be bytes or a seekable binary stream.")
    position = file_obj.tell()
    digest = hashlib.sha256()
    try:
        file_obj.seek(0)
        while True:
            block = file_obj.read(1024 * 1024)
            if not isinstance(block, bytes):
                raise TypeError("Checksum input must be a binary stream, not text.")
            if not block:
                break
            digest.update(block)
    finally:
        file_obj.seek(position)
    return digest.hexdigest()


def collect_environment() -> dict[str, Any]:
    """Collect local runtime versions and an offset-aware generation time."""
    now = datetime.now().astimezone()
    packages = {}
    for name in ("pydeseq2", "pandas", "numpy", "scipy", "scikit-learn",
                 "statsmodels", "streamlit", "plotly", "gseapy", "decoupler"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "timestamp": now.isoformat(), "timezone": now.tzname(),
        "python": platform.python_version(), "os": platform.platform(),
        "packages": packages,
    }


def _copy_json_mapping(value: Mapping[str, Any], category: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Manifest {category} must be a mapping.")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Manifest {category} must contain finite JSON-compatible values: {error}") from error


def build_manifest(inputs: Mapping[str, Any], settings: Mapping[str, Any],
                   counts: Mapping[str, Any], services: Mapping[str, Any]) -> dict[str, Any]:
    """Build one manifest from explicit settings; do not derive significance.

Pass app_version in settings; it is placed under environment. Modality slots
are left empty when no corresponding analysis was performed.
"""
    input_copy = _copy_json_mapping(inputs, "inputs")
    setting_copy = _copy_json_mapping(settings, "settings")
    count_copy = _copy_json_mapping(counts, "counts")
    service_copy = _copy_json_mapping(services, "services")
    environment = collect_environment()
    environment["app_version"] = setting_copy.pop("app_version", None)
    for category in (input_copy, setting_copy, count_copy):
        category.setdefault("rna", None)
        category.setdefault("atac", None)
    service_copy.setdefault("external_services_used", [])
    return {"environment": environment, "inputs": input_copy, "settings": setting_copy,
            "counts": count_copy, "services": service_copy}


def render_manifest_markdown(manifest: Mapping[str, Any]) -> str:
    """Render every manifest field as readable, lossless JSON sections."""
    copied = _copy_json_mapping(manifest, "document")
    sections = ["# BRIM provenance", ""]
    for category, values in copied.items():
        sections.extend([f"## {category}", "", "```json",
                         json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False), "```", ""])
    return "\n".join(sections)


def describe_rna_data(raw_csv: str, results: pd.DataFrame,
                      metadata_frame: pd.DataFrame | None) -> tuple[dict, dict]:
    """Describe the exact exported RNA matrix and count NA flags separately.

For legacy in-memory results with missing flags, NA counts are unknown. In
particular, padded padj==1 values must not be counted as formerly missing.
"""
    raw = pd.read_csv(io.StringIO(raw_csv), index_col=0)
    matrix = {
        "file_name": "0.1_Raw_Counts.csv", "sha256": file_checksum(raw_csv.encode("utf-8")),
        "representation": "BRIM exported count matrix (UTF-8 CSV), after input preparation",
    }
    sample_counts = None
    if metadata_frame is not None and "condition" in metadata_frame:
        selected = metadata_frame.loc[metadata_frame.index.isin(raw.columns)]
        sample_counts = {str(key): int(value) for key, value in selected["condition"].value_counts().items()}
    na_counts = {}
    for flag in ("padj_is_na", "lfc_is_na"):
        if flag not in results:
            na_counts[flag] = None
        else:
            if results[flag].isna().any() or not pd.api.types.is_bool_dtype(results[flag]):
                raise ValueError(f"RNA {flag} must contain non-missing boolean flags.")
            na_counts[flag] = int(results[flag].sum())
    return matrix, {
        "input_genes": len(raw), "input_samples": len(raw.columns), "result_genes": len(results),
        "samples_by_condition": sample_counts, "na_counts": na_counts,
    }
