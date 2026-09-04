# Repository Instructions

## Core principles

- Prefer correct, simple, maintainable code over clever code.
- Implement only the currently requested task.
- Do not implement future features unless required by the current task.
- Do not make broad architectural changes without reporting them first.
- Do not silently invent missing requirements.
- Record unresolved requirements as OPEN QUESTIONS.

## Architecture

- Keep UI, business logic, data access, external integrations, and export logic separated.
- Do not put core calculations directly inside UI event handlers.
- Prefer small, testable modules.
- Avoid unnecessary global mutable state.
- Prefer public APIs of external libraries.
- Do not modify third-party source code unless explicitly requested.

## Scope control

Before editing:
1. Read PLAN.md.
2. Read the relevant section of ARCHITECTURE.md.
3. Inspect existing code and tests.

During implementation:
- Change only files required for the current task.
- Preserve unrelated user changes.
- Do not perform unrelated refactoring.
- Do not add dependencies without a clear need.

## Coding

- Prefer readable code over compact code.
- Use clear names.
- Use type hints where appropriate.
- Add docstrings to public functions/classes.
- Avoid unexplained magic numbers.
- Validate external/user input.
- Do not swallow exceptions silently.
- Provide actionable error messages.

## Testing

Every new feature should include tests where practical.

Before finishing:
1. Run existing relevant tests.
2. Run newly added tests.
3. Fix failures caused by the change.
4. Run the tests again.
5. Perform a minimal runtime/smoke test when applicable.

Do not weaken or delete valid tests simply to make the implementation pass.

## Completion report

At the end of every task report:

1. What was implemented.
2. Files created.
3. Files modified.
4. Tests executed.
5. Test results.
6. Manual/runtime checks performed.
7. Known limitations.
8. OPEN QUESTIONS.
9. Whether any architecture or dependency decisions changed.
10. Whether any project-specific invariant (see below) was touched.

## Safety

- Never overwrite user source data unless explicitly requested.
- Use safe/atomic writes for important files where practical.
- Do not expose secrets in source code or logs.
- Do not construct unsafe shell commands from user input.

## Documentation

- Keep README.md consistent with actual behavior.
- Update PLAN.md when project scope or phase changes.
- Update ARCHITECTURE.md when architectural boundaries change.
- Do not document features that are not implemented.

---

# Project-specific invariants (BRIM)

These rules override convenience, consistency with existing code, and
visual tidiness of output. Each one exists because violating it silently
produces scientifically wrong results that look correct.

If a task appears to require breaking one of these, **stop and report it
as an OPEN QUESTION**. Do not work around it.

Design document references are to `BRIM_RNA_ATAC_Integration_Design_v2.md`.

## I-1. Statistical integrity

**I-1.1 — NA means "not tested", not "not significant".**
Never replace NA in `padj` or `log2FoldChange` without also setting the
corresponding `padj_is_na` / `lfc_is_na` flag column. DESeq2 emits NA for
independent filtering, Cooks outlier removal, or all-zero counts — none of
which mean the feature was tested and found non-significant.

The existing `run_deg()` currently does bare `fillna(1.0)`. This is a known
defect scheduled for repair in Phase 0.5. **Do not copy this pattern into
new code**, and do not "fix" new code to match it.

Downstream, features with `padj_is_na == True` must be classified as
`rna_not_tested` / `atac_not_tested`, never as `atac_only`,
`rna_only_on_mapped_peak`, or `not_significant`. (Design §3.4, §10.2)

**I-1.2 — Never combine p-values across modalities.**
RNA padj and ATAC padj are independently corrected. Do not multiply them,
take Fisher's combined probability, average them, or derive any joint
significance value. Classification uses each threshold separately.
(Design §10.1, §22)

**I-1.3 — Never merge the TF evidence axes into one score.**
The four axes (motif enrichment, target enrichment, TF expression, TF
activity) are reported side by side and stay separate columns. Do not
create a `combined_score`, `tf_score`, `confidence`, or any weighted sum,
even for sorting convenience.

`n_axes_supported` is a count of supported axes for display ordering only.
It is not a statistic and must not be presented as one. (Design §8.5, §12,
§22)

**I-1.4 — Never collapse mixed directions by majority vote.**
A gene with both significant opening and significant closing peaks is
`mixed_accessibility`. Do not resolve it to a single direction by counting,
by summing log2FC, or by picking the most significant peak.
(Design §10.3)

**I-1.5 — Always report the background used for enrichment.**
TF target enrichment uses the universe of genes that are both peak-mapped
and RNA-tested — not all genes, not all DEGs. The universe definition and
its size must appear in the output and in the manifest. (Design §12.2)

**I-1.6 — Enrichment p-values are new tests.**
Target-enrichment and motif-enrichment padj values are computed from the
level-1 classification as input. They are independent of RNA/ATAC padj and
must never be presented as a continuation or refinement of them.
(Design §12.2, §22)

## I-2. Information preservation

**I-2.1 — Preserve one-to-many peak–gene edges.**
A peak mapping to several genes produces several rows. A gene with several
peaks keeps all of them. Do not deduplicate to the largest |log2FC|, the
smallest padj, or the nearest TSS.

Gene-level summaries are secondary aggregations that must retain
`representative_peak_rule` and a path back to the source peaks.
(Design §4.3, §8.4)

**I-2.2 — Record conversions, never apply them silently.**
Chromosome naming (`chr1` vs `1`), coordinate system (0-based vs 1-based),
gene ID mapping, and log2FC inversion are all recorded with their transform
log and success rate. Genome build is never inferred — the user selects it.
(Design §4.4)

**I-2.3 — Normalization is an explicit choice, not a default applied
silently.** ATAC count matrices must not reuse the RNA normalization path
without the user selecting a method. The selected method goes into the
manifest. (Design §7.2)

## I-3. Module boundaries

**I-3.1 — Analysis modules are Streamlit-free.**
`brim_atac.py`, `brim_multiomics.py`, `brim_tf_integration.py`, and
`brim_provenance.py` must not import streamlit, read or write
`st.session_state`, call `st.error` / `st.warning` / `st.stop`, or build
download widgets.

They take DataFrames plus explicit settings and return DataFrames or typed
results. Validation problems are returned or raised, never displayed.
(Design §4.2)

**I-3.2 — Thresholds are passed in, never read from global state.**
No analysis function reads a threshold from session state or a module-level
default. All thresholds are function arguments.

## I-4. External access and distribution

**I-4.1 — Do not add network calls.**
The complete allowed set is the two that already exist:
`mygene.info` (gene ID mapping) and `string-db.org` (network image). Both
must stay behind explicit user action and be recorded in the manifest as
`external_services_used`.

The ATAC and integration features add no network calls at all.
(Design §16.3)

**I-4.2 — Do not add dependencies that break Windows portable
distribution.** Specifically: nothing requiring a genome FASTA, BAM files,
or an external binary (bedtools, HOMER, MEME, samtools). Interval
operations are implemented in pandas/NumPy, or with `bioframe` only after
a documented comparison. (Design §18, §19)

**I-4.3 — Provenance is generated through `brim_provenance.py`, for RNA
too.** Do not create a second, thinner manifest path for RNA-only analysis.
The existing `reproducibility_report.json` is replaced by the shared
generator. (Design §16.1)

## I-5. Rejected designs — do not implement

These were evaluated and rejected. Implementing them is scope violation,
not initiative.

**I-5.1 — No built-in motif table.** Motif enrichment enters only by
importing external results (HOMER etc.). BRIM writes BED files and shows
the command; it does not scan sequence itself. Rationale and re-evaluation
conditions are recorded in Design §12.4, §12.5.

**I-5.2 — No correlation-based peak–gene linking as a primary path.**
Cross-sample expression/accessibility correlation requires ~10+ samples;
this project targets 3–6 per group. Distance- and promoter-based mapping
is the primary path. (Design §9.5, §2.1)

**I-5.3 — No footprinting.** TOBIAS/HINT-style analysis requires BAM.
Out of scope. (Design §3.2, §21.3)

**I-5.4 — No causal language.** Output text says "concordant",
"discordant", "no corresponding evidence", "candidate". It does not say
that accessibility change caused expression change, or that a TF drives a
phenotype. (Design §2)

## I-6. Backward compatibility

**I-6.1 — RNA-only workflows must not change.**
A user who never touches the Multi-omics tab must see the same behavior as
in v1.1.0, except for the additions specified in Phase 0.5 (NA flags,
unified manifest, external-service notices). Existing RNA tests must keep
passing without modification.

**I-6.2 — Level gating.** Level 2 requires level 1 results; level 3
requires level 2. Upstream changes invalidate all downstream levels. Never
leave stale TF or motif results attached to changed inputs.
(Design §15.1)
