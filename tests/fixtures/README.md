# Pre-Phase-0.5 compatibility fixture

`phase05_before/` was recorded from unchanged production code at `16450fa` and
committed separately at `22ad00b` before the NA/manifest implementation.

- `counts.csv`, `metadata.csv`: deterministic eight-gene, six-sample input.
- `deg_results.csv`: actual PyDESeq2 results, preserving all six original columns.
- `reproducibility_report.json`, `results.zip`: actual populated AppTest download
  payloads. Only Plotly static-image conversion was mocked.
- `capture.json`: source hash, Python/package versions, and decoded ZIP contents
  for review. The binary ZIP preserves the original bytes and member metadata.

Do not regenerate these files to accommodate a regression. The manual recorder
refuses to overwrite an existing capture or run on changed production code.
Compare decoded data, not ZIP headers or generation timestamps. Numerical checks
allow relative tolerance `1e-6`; p-values use zero absolute tolerance so that
replacement of tiny nonzero values by zero cannot pass. Package versions are
recorded, not silently enforced by skipping tests on different environments.

The UI harness mocks only static image conversion and file-upload/HTTP input
boundaries. It captures the data handed to real download widgets and retains
the application script, statistical engine, JSON serialization, and ZIP writer.
