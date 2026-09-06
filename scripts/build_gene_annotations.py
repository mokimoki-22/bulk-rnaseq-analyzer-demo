"""Build compact BRIM gene/TSS tables from pinned GENCODE GTF files."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re

import pandas as pd


RELEASES = {
    "hg38": {
        "source": "GENCODE", "species": "human", "build": "GRCh38", "release": "48",
        "assembly": "GRCh38.p14",
        "download_url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.annotation.gtf.gz",
        "source_page": "https://www.gencodegenes.org/human/release_48.html",
    },
    "mm10": {
        "source": "GENCODE", "species": "mouse", "build": "GRCm38", "release": "M25",
        "assembly": "GRCm38.p6",
        "download_url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz",
        "source_page": "https://www.gencodegenes.org/mouse/release_M25.html",
    },
}
ATTRIBUTE = re.compile(r'(\S+) "([^"]*)"')


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_table(gtf_path: Path) -> pd.DataFrame:
    """Extract one row per GENCODE gene with a zero-based TSS coordinate."""
    rows = []
    with gzip.open(gtf_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene":
                continue
            chrom, start, end, strand = fields[0], int(fields[3]), int(fields[4]), fields[6]
            attrs = dict(ATTRIBUTE.findall(fields[8]))
            gene_id = attrs.get("gene_id")
            if not gene_id or strand not in {"+", "-"}:
                raise ValueError(f"Invalid gene row at {gtf_path}:{line_number}")
            rows.append({
                "chrom": chrom if chrom.startswith("chr") else f"chr{chrom}",
                "tss": start - 1 if strand == "+" else end - 1,
                "strand": strand,
                "gene_id": gene_id,
                "gene_symbol": attrs.get("gene_name", gene_id),
                "gene_type": attrs.get("gene_type", ""),
            })
    result = pd.DataFrame(rows)
    if result.empty or result["gene_id"].duplicated().any():
        raise ValueError(f"Expected a non-empty, gene-unique GENCODE table: {gtf_path}")
    return result.sort_values(["chrom", "tss", "gene_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for alias, release in RELEASES.items():
        gtf = args.input_dir / Path(release["download_url"]).name
        table = build_table(gtf)
        output = args.output_dir / f"{alias}_genes.tsv.gz"
        table.to_csv(output, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
        files[alias] = {
            **release,
            "file": output.name,
            "rows": len(table),
            "source_sha256": sha256(gtf),
            "sha256": sha256(output),
            "generation_script": "scripts/build_gene_annotations.py",
            "generation_script_sha256": sha256(Path(__file__).resolve()),
            "generation_command": "python scripts/build_gene_annotations.py --input-dir <download-dir> --output-dir references/genome_annotations",
            "license_terms": "GENCODE open-access data: https://www.gencodegenes.org/pages/data_access.html",
        }
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_policy": "Pinned; update data and manifest in a dedicated commit and record it in CHANGELOG.md.",
        "files": files,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
