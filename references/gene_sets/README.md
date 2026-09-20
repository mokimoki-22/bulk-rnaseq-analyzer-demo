# Bundled KEGG gene-set libraries

BRIM uses these GMT files for local over-representation analysis and GSEA so
KEGG analysis does not depend on Enrichr/Speedrichr availability at run time.

- `KEGG_2019_Mouse.gmt`: Enrichr `KEGG_2019_Mouse` library
- `KEGG_2021_Human.gmt`: Enrichr `KEGG_2021_Human` library
- `GO_Biological_Process_2021.gmt`: Enrichr `GO_Biological_Process_2021` library

`GO_Biological_Process_2021.gmt` contains human gene symbols. In Phase 4
integration ORA, BRIM therefore enables it for Human only. Mouse runs the
bundled `KEGG_2019_Mouse.gmt` only; BRIM does not perform cross-species
conversion or an online fallback.

Source endpoint:
`https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=<library>`

For existing RNA-only ORA/GSEA, the statistical background is the genes in the
active BRIM count matrix. Phase 4 integration ORA has a separate, explicit
background: RNA-tested gene summaries with at least one ATAC-tested mapped
peak, including `not_significant`. It excludes not-tested, unmapped, and
`rna_only_no_mapped_peak` genes. GSEApy performs the local hypergeometric test
and Benjamini-Hochberg multiple-testing correction.
