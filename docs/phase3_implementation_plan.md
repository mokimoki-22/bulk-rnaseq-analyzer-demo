# Phase 3 Integration core implementation plan

## Status

Planning only.  The user authorized moving the current phase to Phase 3 on
2026-09-20.  Independent Auditor A and Auditor B both issued **CONDITIONAL
NO-GO** for implementation.  No Phase 3 code may be written until the open
questions below are resolved in the design documents and both decisions are
updated to GO.

## Approved scope after the gate is cleared

Phase 3 is limited to Streamlit-free Level 1 analysis and its tests:

1. Add `brim_multiomics.py` with the public functions specified in
   `ARCHITECTURE.md` and design document §14.2.
2. Standardize explicitly declared RNA gene identifiers without network access;
   preserve both `padj_is_na` and `lfc_is_na`.
3. Return a compatibility result for species, explicit genome build, ordered
   contrast, and gene-ID matching evidence.  Reject missing or incompatible
   metadata rather than silently converting or inverting a contrast.
4. Produce a lossless edge-level integration table using independent RNA and
   ATAC thresholds.  Preserve one-to-many mappings and never combine p-values.
5. Produce a deterministic gene-level summary, retain a route to source edges,
   declare the representative-peak rule, and retain mixed opening/closing
   accessibility rather than resolving it by majority vote.
6. Connect RNA-originated invalidation to existing integration state so RNA
   replacement, DEG reruns, contrast changes, and RNA-threshold changes cannot
   leave Level 1 results stale.  ATAC invalidation is already connected.
7. Add `tests/test_multiomics.py`, including all classifications, exact
   threshold boundaries, NA handling, compatibility rejections, one-to-many
   edges, mixed accessibility, deterministic output, RNA-only genes without a
   mapped peak, and fixed-seed negative controls required by design §17.4.

## Explicitly out of scope

No Integration subtab or other Phase 4 UI, plot, export/provenance UI,
enrichment, TF analysis, motif analysis, new network access, or dependencies.

## Implementation blockers requiring a design decision

1. Specify the edge and gene-summary class when RNA and ATAC are both not
   tested.  The existing design defines `rna_not_tested` and `atac_not_tested`
   but not their precedence or a joint class.
2. Specify whether `lfc_is_na` is an explicit public column in the integration
   edge and gene-summary contracts.  Both input modalities carry it; dropping
   it would violate information preservation.
3. Specify whether gene-ID matching rate is informative only or has a defined
   compatibility-stop threshold.  The core will report the rate explicitly;
   it must not invent a threshold.

## Approval checkpoint

After the three decisions are reflected in the design and architecture
documents, obtain independent GO decisions from Auditor A (design/scientific
integrity) and Auditor B (implementation/test evidence) before writing Phase 3
implementation code.
