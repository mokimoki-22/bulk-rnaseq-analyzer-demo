"""Deterministic synthetic inputs for the Phase 6 (Level 3 motif import) tests.

Nothing here is real data or real tool output; rows are constructed explicitly (no random numbers) so every
boundary is visible in the test.  Names of motifs added by later steps are strings made up for the tests, not
copies of any tool's output.
"""

from __future__ import annotations

import pandas as pd


THRESHOLDS = {"atac_padj": 0.05, "atac_lfc": 1.0}
_COLUMNS = ["peak_id", "chrom", "start", "end", "log2FoldChange", "pvalue", "padj", "padj_is_na", "lfc_is_na"]

# (peak_id, chrom, start, end, lfc, padj, padj_is_na, lfc_is_na): every boundary case is written out.
BOUNDARY_ROWS = [
    ("open_clear", "chr1", 100, 200, 2.0, 0.001, False, False),
    ("open_padj_boundary", "chr1", 300, 400, 1.0, 0.05, False, False),          # padj == 0.05 and lfc == 1: included
    ("open_padj_just_out", "chr1", 500, 600, 2.0, 0.0500001, False, False),     # just above the padj limit
    ("open_lfc_just_out", "chr1", 700, 800, 0.9999, 0.001, False, False),       # just below the lfc limit
    ("closing_clear", "chr1", 900, 1000, -2.0, 0.001, False, False),
    ("closing_boundary", "chr1", 1100, 1200, -1.0, 0.05, False, False),
    ("closing_lfc_just_out", "chr1", 1300, 1400, -0.9999, 0.001, False, False),
    ("zero_lfc_significant", "chr1", 1500, 1600, 0.0, 0.001, False, False),     # lfc == 0 is never opening/closing
    ("not_tested_padj_na", "chr1", 1700, 1800, 3.0, 1.0, True, False),          # padj NA: excluded everywhere
    ("not_tested_lfc_na", "chr1", 1900, 2000, 0.0, 0.001, False, True),         # lfc NA: excluded everywhere
    ("nonsig_a", "chr2", 100, 200, 0.2, 0.7, False, False),
    ("nonsig_b", "chr2", 300, 400, -0.3, 0.9, False, False),
]
_OPENING = {"open_clear", "open_padj_boundary"}
_CLOSING = {"closing_clear", "closing_boundary"}
_NOT_TESTED = {"not_tested_padj_na", "not_tested_lfc_na"}


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame([{"peak_id": r[0], "chrom": r[1], "start": r[2], "end": r[3], "log2FoldChange": r[4],
                          "pvalue": r[5], "padj": r[5], "padj_is_na": r[6], "lfc_is_na": r[7]} for r in rows],
                        columns=_COLUMNS)


def boundary_dar_table() -> pd.DataFrame:
    """The 12 explicit boundary rows."""
    return _frame(BOUNDARY_ROWS)


def expected_boundary_sets() -> dict[str, set[str]]:
    """The expected peak-set membership of ``boundary_dar_table`` under ``THRESHOLDS``."""
    tested = {row[0] for row in BOUNDARY_ROWS} - _NOT_TESTED
    return {"opening": set(_OPENING), "closing": set(_CLOSING), "background": tested}


def synthetic_dar_table(with_whitespace_ids: bool = False) -> pd.DataFrame:
    """About 300 peaks: the boundary rows plus a deterministic block of opening, closing, non-significant and NA rows."""
    rows = list(BOUNDARY_ROWS)
    for i in range(120):
        rows.append((f"open_{i:03d}", "chr3", 10_000 + 500 * i, 10_300 + 500 * i, 1.5 + (i % 5) * 0.1, 0.001 + (i % 7) * 0.001,
                     False, False))
    for i in range(60):
        rows.append((f"closing_{i:03d}", "chr4", 10_000 + 500 * i, 10_300 + 500 * i, -(1.5 + (i % 5) * 0.1),
                     0.001 + (i % 7) * 0.001, False, False))
    for i in range(100):
        rows.append((f"nonsig_{i:03d}", "chr5", 10_000 + 500 * i, 10_300 + 500 * i, (i % 9 - 4) * 0.1, 0.2 + (i % 8) * 0.1,
                     False, False))
    for i in range(10):
        rows.append((f"na_{i:03d}", "chr6", 10_000 + 500 * i, 10_300 + 500 * i, 0.0, 1.0, i % 2 == 0, i % 2 == 1))
    frame = _frame(rows)
    if with_whitespace_ids:
        extra = _frame([("spaced peak", "chr7", 100, 200, 2.0, 0.001, False, False),
                        ("tabbed\tpeak", "chr7", 300, 400, -2.0, 0.001, False, False)])
        frame = pd.concat([frame, extra], ignore_index=True)
    return frame


def colliding_id_dar_table() -> pd.DataFrame:
    """Two tested peaks whose identifiers become identical after whitespace replacement."""
    return _frame([("a b", "chr1", 100, 200, 2.0, 0.001, False, False),
                   ("a_b", "chr1", 300, 400, -2.0, 0.001, False, False)])
