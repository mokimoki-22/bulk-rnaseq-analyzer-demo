"""Reproducible 200,000-peak benchmark for the Phase 1 interval implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import brim_atac  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", choices=("hg38", "mm10"), default="hg38")
    parser.add_argument("--peaks", type=int, default=200_000)
    args = parser.parse_args()
    if args.peaks < 1:
        raise ValueError("--peaks must be positive.")

    genes = brim_atac.load_gene_annotation(args.build)
    rng = np.random.default_rng(20260906)
    sampled = genes.iloc[np.arange(args.peaks) % len(genes)].reset_index(drop=True)
    offsets = rng.integers(-80_000, 80_001, size=args.peaks)
    starts = np.maximum(0, sampled["tss"].to_numpy() + offsets)
    ends = starts + 250
    peaks = pd.DataFrame({
        "peak_id": sampled["chrom"] + ":" + pd.Series(starts).astype(str)
                   + "-" + pd.Series(ends).astype(str),
        "chrom": sampled["chrom"], "start": starts, "end": ends,
    })
    # A coordinate may repeat when the same annotation row and random offset recur.
    peaks["peak_id"] += ":" + peaks.index.astype(str)

    start = time.perf_counter()
    promoter = brim_atac.map_peaks_to_promoters(peaks, genes, upstream=2_000, downstream=500)
    promoter_seconds = time.perf_counter() - start
    start = time.perf_counter()
    nearest = brim_atac.map_peaks_to_nearest_tss(peaks, genes, max_distance=100_000)
    nearest_seconds = time.perf_counter() - start
    print(json.dumps({
        "build": args.build,
        "input_peaks": args.peaks,
        "reference_genes": len(genes),
        "promoter_edges": len(promoter),
        "nearest_tss_edges": len(nearest),
        "promoter_seconds": round(promoter_seconds, 3),
        "nearest_tss_seconds": round(nearest_seconds, 3),
        "total_seconds": round(promoter_seconds + nearest_seconds, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
