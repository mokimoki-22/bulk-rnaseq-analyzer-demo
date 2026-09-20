# Phase 3 Integration core implementation plan

## Status

Planning only.  The user authorized moving the current phase to Phase 3 on
2026-09-20 and delegated the open design decisions to Auditor C.  Auditor C
resolved them on 2026-09-20; the design and architecture documents now record
the resulting contract.  No Phase 3 code may be written until Auditor A and
Auditor B independently update their decisions to GO.

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

## Resolved decisions (Auditor C, delegated authority)

1. `both_not_tested` is an explicit edge class when both modalities are not
   tested; `rna_not_tested` and `atac_not_tested` apply only when that modality
   alone is not tested.  A not-tested result is `padj_is_na OR lfc_is_na`.
   Gene aggregation follows the exact precedence in design §10.3.
2. Both modalities' `padj_is_na` and `lfc_is_na` are public edge columns.
   Gene summaries retain RNA flags and unique-peak counts for each ATAC NA and
   tested state, plus sorted `source_edge_ids`.
3. Zero shared genes blocks integration.  Any nonzero overlap is permitted;
   the two matching rates and component counts are returned, with a strong
   warning below 80%.  No universal numeric stop threshold is scientifically
   defensible.

## Approval checkpoint

After the three decisions are reflected in the design and architecture
documents, obtain independent GO decisions from Auditor A (design/scientific
integrity) and Auditor B (implementation/test evidence) before writing Phase 3
implementation code.
