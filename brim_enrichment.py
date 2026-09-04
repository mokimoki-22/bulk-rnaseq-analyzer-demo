"""Reliable pathway enrichment helpers for BRIM.

KEGG libraries bundled with BRIM are evaluated locally by GSEApy.  This keeps
the custom count-matrix background and avoids depending on Enrichr/Speedrichr
being available at analysis time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


APP_DIR = Path(__file__).resolve().parent
LOCAL_GENE_SETS = {
    "KEGG_2019_Mouse": APP_DIR / "references" / "gene_sets" / "KEGG_2019_Mouse.gmt",
    "KEGG_2021_Human": APP_DIR / "references" / "gene_sets" / "KEGG_2021_Human.gmt",
    "GO_Biological_Process_2021": APP_DIR / "references" / "gene_sets" / "GO_Biological_Process_2021.gmt",
}


def resolve_gene_set(gene_set: str) -> str:
    """Return a bundled GMT path when available, otherwise the online name."""
    local_path = LOCAL_GENE_SETS.get(gene_set)
    if local_path is not None:
        if not local_path.is_file() or local_path.stat().st_size == 0:
            raise FileNotFoundError(f"Bundled gene-set file is missing: {local_path}")
        return str(local_path)
    return gene_set


def run_overrepresentation(
    gene_list: Iterable[str], gene_set: str, background: Iterable[str]
):
    """Run ORA locally and return results ranked by statistical significance."""
    import gseapy as gp

    genes = [str(gene).strip() for gene in gene_list if str(gene).strip()]
    background_genes = [
        str(gene).strip() for gene in background if str(gene).strip()
    ]
    if not genes:
        raise ValueError("No genes were supplied for pathway analysis.")
    if not background_genes:
        raise ValueError("No background genes were supplied for pathway analysis.")

    resolved_gene_set = resolve_gene_set(gene_set)
    result = gp.enrichr(
        gene_list=genes,
        gene_sets=resolved_gene_set,
        background=background_genes,
        outdir=None,
    ).results
    if result is None or result.empty:
        raise ValueError("No pathway overlaps were found for the selected genes.")

    # GSEApy does not guarantee that its result frame is ordered by
    # significance.  Every UI consumer selects rows with ``head(Top N)``, so
    # enforce one canonical ranking here before plots, tables, and downloads
    # receive the data.  Stable sorting keeps output deterministic when values
    # are tied.
    ranking = [
        ("Adjusted P-value", True),
        ("P-value", True),
        ("Combined Score", False),
        ("Term", True),
    ]
    columns = [column for column, _ in ranking if column in result.columns]
    ascending = [direction for column, direction in ranking if column in result.columns]
    if columns:
        result = result.sort_values(
            columns,
            ascending=ascending,
            na_position="last",
            kind="mergesort",
        )
    return result.reset_index(drop=True)
