# Bundled KEGG gene-set libraries

BRIM uses these GMT files for local over-representation analysis and GSEA so
KEGG analysis does not depend on Enrichr/Speedrichr availability at run time.

- `KEGG_2019_Mouse.gmt`: Enrichr `KEGG_2019_Mouse` library
- `KEGG_2021_Human.gmt`: Enrichr `KEGG_2021_Human` library
- `GO_Biological_Process_2021.gmt`: Enrichr `GO_Biological_Process_2021` library

Source endpoint:
`https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=<library>`

The statistical background remains the genes present in the active BRIM count
matrix. GSEApy performs the local hypergeometric test and Benjamini-Hochberg
multiple-testing correction.
