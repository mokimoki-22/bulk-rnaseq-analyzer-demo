# Phase 4 Integration UI implementation plan

## Scope

Phase 4 adds only the Level 1 Integration subtab, its local class-specific
KEGG/GO ORA, shared-export/provenance artifacts, and UI regression tests.  It
does not add Level 2/3 TF or motif controls, processing, exports, network
access, or dependencies.

## Auditor C decisions

- `last_contrast` is display-only.  Integration requires explicit
  `rna_contrast` and `atac_contrast` objects with exact `reference` and `test`
  labels.  RNA stores its object after successful DEG.  DAR-table validation
  requires both labels before accepting the table; blank or identical labels
  are rejected.  No label/file-name parsing or automatic log2FC inversion is
  allowed.  A changed input, contrast, or rerun clears the object and all
  integration results; both objects are retained verbatim in summaries and the
  manifest.
- Human can run bundled local KEGG and GO-BP.  Mouse can run only bundled local
  `KEGG_2019_Mouse.gmt`; Mouse GO-BP is disabled because the bundled GO library
  contains human symbols.  No cross-species conversion, online fallback, new
  dependency, or network call is permitted.  UI and provenance state this
  reason in English and Japanese and distinguish it from an executed ORA.
- Default to gene-summary display; edge display is optional and explicitly
  warns about one-to-many repetition.  Show every class in the initial filter,
  with discordant classes as visible as concordant classes.
- Quadrants include only tested, mapped, non-mixed values.  Do not plot any
  not-tested result using its compatibility fill values.  Show separate counts
  for `both_not_tested`, `rna_not_tested`, `atac_not_tested`, mixed genes,
  RNA-only-no-mapped-peak genes, and missing ATAC coordinates.  Captions state
  analysis unit, four thresholds, total/drawn counts, and exclusion counts.
- Class ORA operates on deduplicated gene summaries.  Eligible classes are
  concordant, discordant, `atac_only`, `rna_only_on_mapped_peak`, and
  `mixed_accessibility`; the latter is explicitly direction-agnostic.
  Background is the tested, mapped gene-summary universe with at least one
  tested ATAC peak, including `not_significant`.  Not-tested, unmapped, and
  `rna_only_no_mapped_peak` genes are excluded from both input and background.
- Use local species-specific KEGG/GO-BP resources only, behind explicit action.
  Warn for inputs below 20 genes, record zero-overlap results, and state that
  ORA padj is a new independent test—not a continuation or combination of RNA
  or ATAC padj.  Use non-causal wording.
- Export only existing Integration results: edges, gene summary, summary JSON,
  executed ORA, notebook, and shared manifest.  Record compatibility/ID-match
  evidence, contrast/species/build, thresholds, class/mapping counts, plot
  exclusions, and complete ORA history.

## Acceptance tests

AppTest covers prerequisites; compatibility failure/low-match warning;
invalidation on upstream changes; all class/filter views; excluded quadrant
values and captions; English/Japanese non-causal text; mocked local ORA input,
background, warnings, zero-overlap, and independent-test wording; Integration
export/provenance; and absence of Level 2/3 controls.

It also verifies structured RNA/DAR-table contrasts, contrast reset and
provenance, mismatch blocking without ATAC inversion, Human/Mouse library
availability, disabled Mouse GO without runner invocation, and Mouse KEGG.
