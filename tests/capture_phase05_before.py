"""Record immutable pre-Phase-0.5 fixtures; never run against modified code.

Run manually from the repository root: python tests/capture_phase05_before.py.
The recorder refuses to overwrite an existing capture or record a changed app.
"""

import hashlib
import importlib
import importlib.metadata
import io
import json
import platform
import subprocess
import sys
import zipfile

from rna_support import FIXTURES, ROOT, LEGACY_COLUMNS, capture_downloads, result_app, small_counts


def main():
    """Run unchanged production code and save its actual exported payloads."""
    sys.path.insert(0, str(ROOT))
    if FIXTURES.exists():
        raise SystemExit("Capture already exists; refusing to overwrite the baseline.")
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    baseline_source = subprocess.check_output(git + ["show", "16450fa:Bulk_RNAseq_Analyzer.py"], cwd=ROOT)
    current_source = (ROOT / "Bulk_RNAseq_Analyzer.py").read_bytes().replace(b"\r\n", b"\n")
    if current_source != baseline_source.replace(b"\r\n", b"\n"):
        raise SystemExit("Production code differs from 16450fa; capture is not allowed.")
    module = importlib.import_module("Bulk_RNAseq_Analyzer")
    counts, metadata = small_counts()
    result = module.run_deg(counts, metadata, "control", "treated", n_cpus=1)
    assert list(result.columns) == LEGACY_COLUMNS
    app = result_app(result)
    with capture_downloads() as downloads:
        app.run()
    assert not app.exception, [error.value for error in app.exception]
    report = downloads["reproducibility_report.json"]
    archive = downloads["results.zip"]
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        members = {name: zipped.read(name).decode("utf-8") for name in zipped.namelist()}
    capture = {
        "source_commit": "16450fa", "source_sha256_lf": hashlib.sha256(current_source).hexdigest(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in (
            "pydeseq2", "numpy", "pandas", "scipy", "scikit-learn", "anndata", "streamlit"
        )},
        "zip_members": members,
        "note": "Actual UI payloads. Only Plotly static image conversion was mocked. Timestamp/ZIP metadata are not golden equality targets.",
    }
    FIXTURES.mkdir(parents=True)
    counts.to_csv(FIXTURES / "counts.csv", lineterminator="\n")
    metadata.to_csv(FIXTURES / "metadata.csv", lineterminator="\n")
    result.to_csv(FIXTURES / "deg_results.csv", lineterminator="\n")
    (FIXTURES / "reproducibility_report.json").write_text(report, encoding="utf-8")
    (FIXTURES / "results.zip").write_bytes(archive)
    (FIXTURES / "capture.json").write_text(json.dumps(capture, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(result), "zip_members": list(members), "packages": capture["packages"]}))


if __name__ == "__main__":
    main()
