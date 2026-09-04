# Bundled TF networks

BRIM reads these compressed CSV files locally so TF activity estimation does
not depend on a live database URL.

- CollecTRI and DoRothEA were retrieved through decoupler 2.1.4.
- Human-to-mouse orthologs use the current HGNC HCOP download:
  `https://storage.googleapis.com/public-download-files/hcop/human_mouse_hcop_fifteen_column.txt.gz`
- Translation follows decoupler defaults: minimum 3 supporting orthology
  resources and at most 5 orthologs per gene.
- Human edges that converge on the same mouse TF-target pair are consolidated.
  The best DoRothEA confidence is retained and weights within the retained
  class are averaged; CollecTRI weights are averaged.
- Generated on 2026-08-06 with `tools/generate_tf_networks.py`.

The generator is a maintenance utility and requires internet access. Normal
BRIM analysis uses only the files in this directory.
