# Phase 2 audit remediation plan

## Scope and decision

This plan addresses the two independent Phase 2 audit decisions issued on
2026-09-20.  It remains wholly within Phase 2: it adds no RNA--ATAC
integration, TF analysis, motif analysis, network access, dependency, or
Phase 3 UI.

The design document's ATAC-only visualizations in section 11.1 are treated as
Phase 2 completion requirements, not deferred work.  This avoids changing the
approved design merely to clear the audit gate.

## Implementation sequence

1. **Make project status truthful before and after the code change.**
   Update `PLAN.md` to state that Phase 2 remains active and that this audit
   remediation is its next task; correct the contradictory Phase 2 and Phase 7
   status entries.  Update `README.md` only to describe behavior actually
   delivered by this remediation.  Do not advance the current phase.

   Render the existing Phase 2 UI inside the always-visible `ATAC-seq`
   subtab of `Multi-omics`.  No Integration subtab, placeholder, or control
   is added in this remediation because integration remains Phase 3 work.

   Add the remaining section 6.2 Phase 2 controls rather than deferring them:
   a validation summary (row/valid-peak/significant-DAR counts, chromosome
   list, duplicate/missing/invalid-coordinate diagnostics, log2FC direction,
   and count-matrix sample-size warning); an optional user-provided peak--gene
   mapping upload; and the unmapped-peak result table.  Parse and validate the
   optional mapping in `brim_atac.py`, retain its original evidence type,
   score, and source without transforming them into a BRIM score, and record
   it in provenance.  This is the explicit implementation of design sections
   6.1--6.2 and 7.5, not a design change or a deferral.

2. **Preserve the identity of an ATAC input.**
   Extend the ATAC upload signature from mode/name/size to include the
   SHA-256 of the original uploaded bytes, using
   `brim_provenance.file_checksum()` without consuming the upload stream.
   Store only an input-provenance record (file name, byte size, checksum, and
   input mode) in session state; do not duplicate a potentially large upload.
   A changed digest must invoke the existing ATAC reset path, which also
   clears mapping and every present or future integration/TF/motif result.

3. **Produce one shared manifest and package for RNA-only, ATAC-only, and
   future combined work.**
   Keep manifest construction in the Streamlit-free `brim_provenance.py`
   module.  Add a small ATAC data-description helper there, parallel to the
   existing RNA helper, to calculate checksums and counts from explicit values.
   `collect_all_results()` will then pass explicit optional `rna` and `atac`
   slots to the existing `build_manifest()` path.  For ATAC, record source
   mode and source file identity, DAR thresholds and contrast, normalization
   and prefiltering/size-factor information where applicable, validation and
   coordinate-transform logs, mapping settings, genome build/species,
   reference metadata, and DAR/edge/NA/mapping counts.  No values are inferred
   when they were not supplied; they remain null or explicitly unavailable.

   Enable Package Export and standalone JSON/Markdown manifest download when
   either RNA DEG results or ATAC results exist.  An ATAC-only ZIP contains
   `ATAC/peak_counts.csv` for count-matrix input, `ATAC/dar_standardized.csv`,
   `ATAC/dar_significant.csv`, `ATAC/peak_gene_edges.csv` when mappings exist,
   `ATAC/unmapped_peaks.csv` after annotation, and
   `ATAC/atac_validation.json`.  When annotation was run, it also contains
   `Provenance/reference_manifest.json` constructed from the selected
   reference record returned by `load_gene_annotation()`; it is not invented
   for an unannotated result.  Every package has the same
   `Provenance/manifest.json` and `Provenance/manifest.md` structure used by
   RNA export.  It does not introduce Phase 3 integration files.

4. **Complete the specified ATAC-only result views.**
   In addition to the existing DAR volcano/table/download, render from the
   stored ATAC results and mapping outputs: chromosome-wise DAR counts,
   opening/closing counts, peak--TSS distance distribution, annotation-method
   counts, and mapping coverage.  These are descriptive views only and do not
   alter DAR statistics, peak--gene edges, or thresholds.

   Every ATAC plot, including the existing volcano plot, receives a caption
   stating its thresholds, displayed n, and analysis unit.  The results area
   continues to display the count of untested peaks as a separate metric;
   tests will protect that distinction.

5. **Test the contracts, then verify all supported environments.**
   Add provenance unit tests for ATAC input checksums and complete manifest
   fields, AppTest coverage for an ATAC-only package/standalone manifest, and
   a same-name/same-size/different-content upload replacement test proving
   that stale ATAC and downstream state is cleared.  Add UI regression checks
   for the new subtab, validation summary, optional mapping, unmapped table,
   captions, untested metric, and descriptive views.

   Manifest regression cases cover both input modes and assert, as applicable,
   a value or explicit null for: original-byte SHA-256, source mode, coordinate
   system and transform log, DAR-table column mapping, normalization,
   prefilter, size factors, contrast, species/build, reference release and
   checksum, NA-flag counts, DAR/edge/unmapped/mapping counts, and
   user-provided mapping provenance.  Package tests assert each conditional
   artifact above.  Preserve all existing RNA-only tests and run the relevant
   test groups locally.  Push only after they pass and require the
   Ubuntu/Windows x Python 3.11/3.12 CI matrix to pass before requesting a
   renewed Phase 3 gate decision.

## Acceptance criteria

- An ATAC-only result can be exported without RNA DEG results; its ZIP and
  standalone manifests are generated by `brim_provenance.build_manifest()`.
- The manifest records all available ATAC provenance listed above, including
  the original upload SHA-256, and uses explicit nulls for unavailable fields.
- Replacing an upload with different bytes but the same name and size clears
  ATAC, mapping, and downstream state.
- All five design section 11.1 ATAC-only descriptive views are present.
- The Phase 2 ATAC-seq subtab, validation summary, optional mapping, and
  unmapped-peak table satisfy design sections 6.1--6.2 without adding Phase 3
  controls.
- Every ATAC plot has a threshold/n/unit caption and the untested-peak metric
  remains distinct from non-significant peaks.
- `PLAN.md` and `README.md` accurately reflect the delivered Phase 2 state;
  Phase 3 remains unstarted.
- New regression tests pass alongside existing relevant tests and the four CI
  jobs pass.

## Open questions resolved by this plan

- **ATAC-only package shape:** use the existing shared ZIP/manifest structure;
  the ZIP contains only artifacts that exist for that modality.
- **Section 11.1 visualizations:** implement them now as Phase 2 scope rather
  than defer them.
- **User-provided mapping and validation UI:** implement them now as explicit
  Phase 2 requirements; do not move them to a later phase.
- **Reference manifest:** include it only after annotation, from the exact
  fixed reference record used by that annotation.
