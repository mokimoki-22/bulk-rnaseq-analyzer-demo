"""Phase 6 Level 3 motif import core tests (Streamlit-free), built up step by step.

Step 1: peak sets, BED files, commands, README and the export bundle.  Step 2: reading motif results.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pandas as pd
import pytest

import brim_atac
import brim_motif_import as mi
from motif_support import (HOMER_DEFAULT_ROWS, HOMER_HEADER, THRESHOLDS, boundary_dar_table, colliding_id_dar_table,
                           expected_boundary_sets, generic_motif_csv, homer_known_text, synthetic_dar_table)


def _sets(dar=None, thresholds=THRESHOLDS, build="hg38", species="Human"):
    return mi.build_peak_sets(boundary_dar_table() if dar is None else dar, thresholds, build, species)


def test_boundaries_are_inclusive_and_untested_or_zero_lfc_peaks_are_excluded():
    peak_sets = _sets()
    expected = expected_boundary_sets()
    assert set(peak_sets.opening["name"]) == expected["opening"]
    assert set(peak_sets.closing["name"]) == expected["closing"]
    assert set(peak_sets.background["name"]) == expected["background"]
    everything = set(peak_sets.opening["name"]) | set(peak_sets.closing["name"]) | set(peak_sets.background["name"])
    assert not everything & {"not_tested_padj_na", "not_tested_lfc_na"}          # I-1.1: untested peaks are never exported
    assert "open_padj_just_out" not in set(peak_sets.opening["name"])
    assert "open_lfc_just_out" not in set(peak_sets.opening["name"])
    assert "zero_lfc_significant" in set(peak_sets.background["name"])
    assert "zero_lfc_significant" not in set(peak_sets.opening["name"]) | set(peak_sets.closing["name"])


def test_zero_lfc_threshold_still_excludes_lfc_zero_from_opening_and_closing():
    peak_sets = _sets(thresholds={"atac_padj": 0.05, "atac_lfc": 0.0})
    assert "zero_lfc_significant" not in set(peak_sets.opening["name"]) | set(peak_sets.closing["name"])
    assert {"open_lfc_just_out", "open_clear"} <= set(peak_sets.opening["name"])
    assert "closing_lfc_just_out" in set(peak_sets.closing["name"])


def test_counts_are_consistent_and_the_background_is_all_tested_peaks():
    dar = synthetic_dar_table()
    peak_sets = _sets(dar)
    counts = peak_sets.counts
    tested = int((~dar["padj_is_na"] & ~dar["lfc_is_na"]).sum())
    assert counts["n_dar_total"] == len(dar) and counts["n_tested"] == tested == counts["n_background"]
    assert counts["n_not_tested_excluded"] == len(dar) - tested == 10 + 2
    assert counts["n_opening"] == len(peak_sets.opening) and counts["n_closing"] == len(peak_sets.closing)
    assert counts["n_opening"] + counts["n_closing"] < counts["n_background"]        # non-significant peaks stay in the background


def test_opening_and_closing_bed_match_the_existing_export_when_no_identifier_is_changed():
    dar = synthetic_dar_table()
    peak_sets = _sets(dar)
    thresholds = {"padj": 0.05, "log2FoldChange": 1.0}
    assert mi.peak_set_bed_text(peak_sets, "opening") == brim_atac.export_peaks_as_bed(dar, "opening", thresholds)
    assert mi.peak_set_bed_text(peak_sets, "closing") == brim_atac.export_peaks_as_bed(dar, "closing", thresholds)


def test_existing_bed_export_is_unchanged_and_still_includes_untested_peaks_so_it_is_not_the_background():
    text = brim_atac.export_peaks_as_bed(boundary_dar_table(), "all", {"padj": 0.05, "log2FoldChange": 1.0})
    assert "not_tested_padj_na" in text and "not_tested_lfc_na" in text


def test_bed_format_is_four_tab_separated_columns_zero_based_with_lf_endings_and_input_order():
    text = mi.peak_set_bed_text(_sets(), "opening")
    assert text == "chr1\t100\t200\topen_clear\nchr1\t300\t400\topen_padj_boundary\n"
    assert "\r" not in mi.peak_set_bed_text(_sets(), "background")
    with pytest.raises(mi.MotifImportError):
        mi.peak_set_bed_text(_sets(), "everything")


def test_whitespace_in_identifiers_is_replaced_and_counted_and_collisions_stop_the_export():
    peak_sets = _sets(synthetic_dar_table(with_whitespace_ids=True))
    names = set(peak_sets.background["name"])
    assert {"spaced_peak", "tabbed_peak"} <= names and not [n for n in names if re.search(r"\s", n)]
    assert peak_sets.counts["n_ids_sanitized"] == 2
    assert _sets(synthetic_dar_table()).counts["n_ids_sanitized"] == 0
    with pytest.raises(mi.MotifImportError, match="collide"):
        _sets(colliding_id_dar_table())


def test_empty_and_small_peak_sets_write_no_file_no_command_and_carry_warnings():
    dar = boundary_dar_table().assign(padj=0.9)               # nothing is significant any more
    peak_sets = _sets(dar)
    assert peak_sets.opening.empty and peak_sets.closing.empty and not peak_sets.background.empty
    codes = [(w["code"], w["peak_set"]) for w in peak_sets.warnings]
    assert ("empty_peak_set", "opening") in codes and ("empty_peak_set", "closing") in codes
    assert mi.build_homer_commands(peak_sets) == []
    assert mi.homer_unavailable_reason(peak_sets)["code"] == "no_peak_sets"
    bundle = mi.build_motif_bundle(peak_sets, "9.9.9", "2026-09-21T00:00:00")
    assert set(bundle) == {"MotifAnalysis/all_peaks_background.bed", "MotifAnalysis/motif_analysis_README.txt"}
    small = _sets()                                            # 2 opening and 2 closing peaks: below the small-set limit
    assert [w["code"] for w in small.warnings].count("small_peak_set") == 2
    assert all(w["message"] and w["message_ja"] for w in small.warnings)


def test_one_empty_peak_set_omits_only_its_command():
    dar = boundary_dar_table()
    dar = dar.loc[~dar["peak_id"].isin(["closing_clear", "closing_boundary"])]
    commands = mi.build_homer_commands(_sets(dar))
    assert len(commands) == 2 and "opened_peaks" in commands[1] and not any("closed_peaks" in c for c in commands)


def test_file_names_are_shell_safe_and_follow_the_thresholds():
    assert mi.bed_file_name("opening", THRESHOLDS) == "opened_peaks_padj0.05_lfc1.bed"
    assert mi.bed_file_name("closing", THRESHOLDS) == "closed_peaks_padj0.05_lfc1.bed"
    assert mi.bed_file_name("background", THRESHOLDS) == "all_peaks_background.bed"
    assert mi.bed_file_name("opening", {"atac_padj": 0.01, "atac_lfc": 0.585}) == "opened_peaks_padj0.01_lfc0.585.bed"
    for kind in ("opening", "closing", "background"):
        for thresholds in (THRESHOLDS, {"atac_padj": 1e-05, "atac_lfc": 0.0}):
            assert re.match(r"^[A-Za-z0-9._-]+$", mi.bed_file_name(kind, thresholds))
    with pytest.raises(mi.MotifImportError, match="unsafe"):
        mi.bed_file_name("opening", {"atac_padj": 0.05, "atac_lfc": 1e20})
    with pytest.raises(mi.MotifImportError):
        mi.bed_file_name("other", THRESHOLDS)


def test_homer_commands_use_the_allow_listed_genome_and_generated_file_names_only():
    dar = synthetic_dar_table()
    assert mi.build_homer_commands(_sets(dar, build="hg38")) == [
        "perl configureHomer.pl -install hg38",
        "findMotifsGenome.pl opened_peaks_padj0.05_lfc1.bed hg38 homer_opening/ -size 200 -bg all_peaks_background.bed",
        "findMotifsGenome.pl closed_peaks_padj0.05_lfc1.bed hg38 homer_closing/ -size 200 -bg all_peaks_background.bed",
    ]
    assert "mm10" in mi.build_homer_commands(_sets(dar, build="mm10", species="Mouse"))[1]
    for build in ("hg19", "hg38; rm -rf /", "", "$(evil)"):
        peak_sets = _sets(dar, build=build)
        assert mi.build_homer_commands(peak_sets) == [], build
        assert mi.homer_unavailable_reason(peak_sets)["code"] == "genome_not_supported"
    for command in mi.build_homer_commands(_sets(dar)):
        assert not re.search(r"[;&|$`<>\\]", command)                         # no shell metacharacters in any command


def test_fingerprint_is_deterministic_and_changes_with_every_input_it_depends_on():
    dar = synthetic_dar_table()
    base = _sets(dar).peakset_fingerprint
    assert base == _sets(dar).peakset_fingerprint and len(base) == 64
    assert _sets(dar, thresholds={"atac_padj": 0.01, "atac_lfc": 1.0}).peakset_fingerprint != base
    assert _sets(dar, thresholds={"atac_padj": 0.05, "atac_lfc": 1.5}).peakset_fingerprint != base
    assert _sets(dar, build="mm10").peakset_fingerprint != base
    assert _sets(dar, species="Mouse").peakset_fingerprint != base
    changed = dar.copy()
    changed.loc[changed["peak_id"] == "open_000", "start"] += 1
    assert _sets(changed).peakset_fingerprint != base
    prefixed = dar.copy()
    prefixed["chrom"] = prefixed["chrom"].str.replace("chr", "", regex=False)
    assert _sets(prefixed).peakset_fingerprint != base


def test_chromosome_naming_is_recorded_warned_when_mixed_and_never_converted():
    dar = synthetic_dar_table()
    assert _sets(dar).counts["chrom_style"] == {"chr_prefixed": len(_sets(dar).background), "bare": 0}
    mixed = dar.copy()
    mixed.loc[mixed["chrom"] == "chr3", "chrom"] = "3"
    peak_sets = _sets(mixed)
    assert peak_sets.counts["chrom_style"]["chr_prefixed"] > 0 and peak_sets.counts["chrom_style"]["bare"] == 120
    assert "mixed_chromosome_naming" in [w["code"] for w in peak_sets.warnings]
    assert "3\t10000\t10300\topen_000\n" in mi.peak_set_bed_text(peak_sets, "opening")        # written exactly as given
    assert "mixed_chromosome_naming" not in [w["code"] for w in _sets(dar).warnings]


def test_readme_states_the_conditions_the_bed_files_and_the_required_sentences():
    peak_sets = _sets(synthetic_dar_table())
    text = mi.render_motif_readme(peak_sets, "9.9.9", "2026-09-21T00:00:00")
    assert "BRIM 9.9.9" in text and "2026-09-21T00:00:00" in text
    assert "padj <= 0.05" in text and "|log2FC| >= 1" in text and "log2FC = 0 is never opening or closing" in text
    assert mi.THRESHOLD_SOURCE_NOTE in text and mi.THRESHOLD_SOURCE_NOTE_JA in text
    assert "Genome build / genome build: hg38" in text and "0-based half-open (BED)" in text
    assert mi.BACKGROUND_DEFINITION in text and mi.BACKGROUND_DEFINITION_JA in text
    assert f"tested / 検定済み: {peak_sets.counts['n_tested']}" in text
    assert "tf_candidates.csv keeps motif_status=not_run by design" in text
    assert "tf_candidates_with_motif.csv" in text and "motif_status=not_runのままです" in text
    for command in mi.build_homer_commands(peak_sets):
        assert command in text
    assert "BRIM does not bundle or run it" in text and "-size 200 is an example value" in text
    assert "WSL" in text
    assert text == mi.render_motif_readme(peak_sets, "9.9.9", "2026-09-21T00:00:00")          # a pure function
    unavailable = mi.render_motif_readme(_sets(synthetic_dar_table(), build="hg19"), "9.9.9", "t")
    assert "hg19" in unavailable and "perl configureHomer" not in unavailable


def test_readme_text_has_no_causal_wording():
    text = mi.render_motif_readme(_sets(synthetic_dar_table()), "9.9.9", "t")
    assert not re.search(r"\b(regulates?|drives?|driven|causes?|caused|causing)\b", text, re.IGNORECASE)
    assert not re.search(r"制御する|引き起こ|原因", text)


def test_bundle_contains_the_written_files_and_the_recorded_checksums_match_their_bodies():
    peak_sets = _sets(synthetic_dar_table())
    bundle = mi.build_motif_bundle(peak_sets, "9.9.9", "t")
    assert set(bundle) == {"MotifAnalysis/opened_peaks_padj0.05_lfc1.bed", "MotifAnalysis/closed_peaks_padj0.05_lfc1.bed",
                           "MotifAnalysis/all_peaks_background.bed", "MotifAnalysis/motif_analysis_README.txt"}
    checksums = mi.bundle_sha256(bundle)
    import hashlib
    for path, text in bundle.items():
        assert checksums[path] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert peak_sets.bed_sha256["opening"] == checksums["MotifAnalysis/opened_peaks_padj0.05_lfc1.bed"]
    assert peak_sets.bed_sha256["background"] == checksums["MotifAnalysis/all_peaks_background.bed"]
    assert mi.PeakSets.__dataclass_params__.frozen


def test_invalid_inputs_are_rejected_with_actionable_messages():
    dar = boundary_dar_table()
    with pytest.raises(mi.MotifImportError, match="missing required columns: padj_is_na"):
        _sets(dar.drop(columns=["padj_is_na"]))
    for bad in ({"atac_padj": 2, "atac_lfc": 1}, {"atac_padj": 0.05, "atac_lfc": -1}, {"atac_padj": float("nan"), "atac_lfc": 1},
                {"atac_padj": 0.05}, {"atac_padj": "x", "atac_lfc": 1}):
        with pytest.raises(mi.MotifImportError):
            _sets(dar, thresholds=bad)
    with pytest.raises(mi.MotifImportError, match="True/False"):
        _sets(dar.assign(padj_is_na=None))
    with pytest.raises(mi.MotifImportError, match="whitespace"):
        _sets(dar.assign(chrom="chr 1"))
    with pytest.raises(mi.MotifImportError, match="whole numbers"):
        _sets(dar.assign(start=1.5))
    with pytest.raises(mi.MotifImportError, match="whole numbers"):
        _sets(dar.assign(end=dar["start"] - 1))
    with pytest.raises(mi.MotifImportError, match="finite"):
        _sets(dar.assign(log2FoldChange=float("inf")))


def test_module_is_streamlit_free_and_never_runs_processes_or_touches_the_network():
    source = pathlib.Path(mi.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source and "st.session_state" not in source
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"streamlit", "subprocess", "socket", "requests", "urllib", "urllib3", "http", "httpx", "aiohttp",
                 "ftplib", "shlex", "multiprocessing"}
    assert not imported & forbidden, imported & forbidden
    assert "os.system" not in source and "os.popen" not in source and "Popen" not in source


# ----------------------------------------------------------------------------------------------
# Step 2: reading motif results (D1, D10)
# ----------------------------------------------------------------------------------------------

def _homer(text=None, name="knownResults.txt", **kwargs):
    return mi.read_motif_results((homer_known_text() if text is None else text).encode("utf-8"), name, "homer_known", **kwargs)


def _generic(text=None, column_map=None, name="motif.csv", encoding="utf-8"):
    return mi.read_motif_results((generic_motif_csv() if text is None else text).encode(encoding), name, "generic",
                                 column_map={"motif_name": "motif", "padj": "q_value", "pvalue": "p_value"}
                                 if column_map is None else column_map)


def test_homer_known_results_are_read_exactly_as_reported_and_nothing_is_recomputed():
    table = _homer()
    rows = table.rows
    assert len(rows) == len(HOMER_DEFAULT_ROWS) == table.record["n_rows"]
    assert rows["motif_name"].iloc[0] == HOMER_DEFAULT_ROWS[0][0]
    assert rows["padj"].iloc[:5].tolist() == [0.0001, 0.01, 0.02, 0.5, 0.3]          # the q-value (Benjamini) column as is
    assert rows["pvalue"].iloc[0] == 1e-30 and rows["pvalue"].iloc[1] == 1e-10
    assert rows["pct_target"].iloc[0] == 73.17 and rows["pct_background"].iloc[0] == 32.85        # '%' removed, value kept
    assert rows["enrichment_score"].isna().all()                                                # BRIM derives no score
    assert rows["source_line"].tolist() == list(range(2, 2 + len(rows)))
    record = table.record
    assert (record["n_target_sequences_reported"], record["n_background_sequences_reported"]) == (123, 4567)
    assert record["tool"] == "homer_known" and record["delimiter"] == "tab" and record["encoding"] == "utf-8-sig"
    assert record["column_map"]["padj"] == "q-value (Benjamini)" and record["column_map"]["motif_name"] == "Motif Name"
    assert len(record["sha256"]) == 64 and record["byte_size"] == len(homer_known_text().encode("utf-8"))
    assert record["file_name"] == "knownResults.txt" and record["peak_set_in_file"] is None
    assert "peak_set" not in rows.columns


def test_missing_values_stay_missing_and_are_counted_never_filled():
    table = _homer()
    assert table.rows["padj"].isna().sum() == 1 == table.record["n_missing_padj"]        # the NA q-value row
    assert table.rows.loc[table.rows["motif_name"].str.startswith("Myc"), "padj"].isna().all()
    assert not table.record["no_padj_reported"] and table.warnings == []


def test_header_recognition_ignores_case_and_spacing_and_extra_columns():
    header = [h.upper().replace("(", "  (") for h in HOMER_HEADER] + ["Extra Column"]
    rows = [row + ("x",) for row in HOMER_DEFAULT_ROWS]
    table = _homer(homer_known_text(rows, header))
    assert table.rows["padj"].iloc[0] == 0.0001
    assert table.record["n_target_sequences_reported"] == 123


def test_homer_log_p_value_column_is_not_mistaken_for_the_p_value_column():
    table = _homer()
    assert table.rows["pvalue"].iloc[0] == 1e-30 and table.rows["pvalue"].iloc[0] != -69.1


def test_de_novo_motif_files_and_unrecognised_headers_are_rejected_with_a_clear_message():
    with pytest.raises(mi.MotifImportError, match="de novo"):
        _homer(">ATGCAAAT\tMotif1\t8.5\n")
    with pytest.raises(mi.MotifImportError, match="q-value \\(Benjamini\\)"):
        _homer(homer_known_text(header=["Motif Name", "Consensus", "P-value", "Log P-value", "Log P-pvalue"]))
    with pytest.raises(mi.MotifImportError, match="knownResults"):
        _homer("Gene\tScore\nA\t1\n")
    with pytest.raises(mi.MotifImportError, match="no header"):
        _homer("# only a comment\n\n")


def test_generic_file_needs_a_confirmed_column_map_and_reads_the_mapped_columns():
    text = generic_motif_csv()
    with pytest.raises(mi.MotifImportError, match="column map is required"):
        mi.read_motif_results(text.encode("utf-8"), "m.csv", "generic")
    assert mi.suggest_column_map(["Motif", "Q-Value", "p-value", "Peak_Set", "Other"]) == {
        "motif_name": "Motif", "padj": "Q-Value", "pvalue": "p-value", "peak_set": "Peak_Set"}
    table = _generic(text)
    assert table.rows["padj"].iloc[:2].tolist() == [0.0001, 0.01] and table.rows["padj"].isna().sum() == 1
    assert table.record["peak_set_in_file"] is None
    assert table.record["column_map"] == {"motif_name": "motif", "padj": "q_value", "pvalue": "p_value"}
    with pytest.raises(mi.MotifImportError, match="not found in the file"):
        _generic(text, column_map={"motif_name": "motif", "padj": "missing_column"})
    with pytest.raises(mi.MotifImportError, match="at least one of padj or pvalue"):
        _generic(text, column_map={"motif_name": "motif"})
    with pytest.raises(mi.MotifImportError, match="motif_name"):
        _generic(text, column_map={"padj": "q_value"})


def test_generic_peak_set_column_must_hold_a_single_known_peak_set():
    column_map = {"motif_name": "motif", "padj": "q_value", "pvalue": "p_value", "peak_set": "peak_set"}
    assert _generic(column_map=column_map).record["peak_set_in_file"] == "opening"
    upper = generic_motif_csv([("Stat3", "0.1", "0.01", "CLOSING")])
    assert _generic(upper, column_map=column_map).record["peak_set_in_file"] == "closing"
    mixed = generic_motif_csv([("Stat3", "0.1", "0.01", "opening"), ("Myc", "0.2", "0.02", "closing")])
    with pytest.raises(mi.MotifImportError, match="Split the file"):
        _generic(mixed, column_map=column_map)
    with pytest.raises(mi.MotifImportError, match="one value, opening or closing"):
        _generic(generic_motif_csv([("Stat3", "0.1", "0.01", "background")]), column_map=column_map)


def test_only_a_pvalue_gives_no_padj_and_brim_never_derives_one():
    text = "motif,p_value\nStat3,0.001\nMyc,0.2\n"
    table = _generic(text, column_map={"motif_name": "motif", "pvalue": "p_value"})
    assert table.rows["padj"].isna().all() and table.record["no_padj_reported"] is True
    assert table.rows["pvalue"].tolist() == [0.001, 0.2]
    assert [w["code"] for w in table.warnings] == ["no_padj_reported"] and "does not adjust" in table.warnings[0]["message"]


def test_invalid_probabilities_are_reported_with_line_numbers_at_most_five_at_a_time():
    rows = [("A", "1.5", "0.1", "x"), ("B", "abc", "0.1", "x"), ("C", "inf", "0.1", "x"), ("D", "-0.1", "0.1", "x"),
            ("E", "0.1", "2", "x"), ("F", "0.1", "0.1", "x"), ("G", "nan_text", "0.1", "x")]
    with pytest.raises(mi.MotifImportError) as error:
        _generic(generic_motif_csv(rows))
    message = str(error.value)
    assert "line 2: padj 1.5 is outside 0-1" in message and "line 3: padj 'abc' is not a number" in message
    assert "line 4: padj 'inf' is not a finite number" in message and "and 1 more" in message
    assert message.count("line ") == 5


def test_empty_motif_names_and_empty_or_headerless_files_are_rejected():
    with pytest.raises(mi.MotifImportError, match="motif name is empty"):
        _generic(generic_motif_csv([("", "0.1", "0.1", "x")]))
    with pytest.raises(mi.MotifImportError, match="empty"):
        mi.read_motif_results(b"", "e.txt", "generic", column_map={"motif_name": "m", "padj": "q"})
    with pytest.raises(mi.MotifImportError, match="no data rows"):
        _generic("motif,q_value,p_value,peak_set\n")
    with pytest.raises(mi.MotifImportError, match="tool must be"):
        mi.read_motif_results(b"x", "x", "other")


def test_size_and_row_limits_are_enforced(monkeypatch):
    monkeypatch.setattr(mi, "MAX_IMPORT_BYTES", 200)
    with pytest.raises(mi.MotifImportError, match="larger than"):
        _homer()
    monkeypatch.setattr(mi, "MAX_IMPORT_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(mi, "MAX_IMPORT_ROWS", 3)
    with pytest.raises(mi.MotifImportError, match="more than 3 data rows"):
        _homer()
    assert len(_generic(generic_motif_csv([("A", "0.1", "0.1", "x")] * 3)).rows) == 3
    monkeypatch.setattr(mi, "MAX_IMPORT_BYTES", 1024)
    with pytest.raises(mi.MotifImportError, match="limit of 1,024 bytes"):
        mi.read_motif_results(b"m,q\n" + b"a,0.1\n" * 400, "big.csv", "generic", column_map={"motif_name": "m", "padj": "q"})


def test_encodings_are_detected_and_recorded_and_unreadable_files_are_rejected():
    assert _homer().record["encoding"] == "utf-8-sig"
    bom = mi.read_motif_results(b"\xef\xbb\xbf" + homer_known_text().encode("utf-8"), "k.txt", "homer_known")
    assert bom.rows["motif_name"].iloc[0].startswith("Stat3") and bom.record["encoding"] == "utf-8-sig"       # BOM removed
    japanese = generic_motif_csv([("転写因子A", "0.1", "0.01", "x")])
    table = _generic(japanese, encoding="cp932")
    assert table.record["encoding"] == "cp932" and table.rows["motif_name"].iloc[0] == "転写因子A"
    with pytest.raises(mi.MotifImportError, match="not readable text"):
        mi.read_motif_results(b"motif,q\n\x81\x20,0.1\n", "bad.csv", "generic", column_map={"motif_name": "motif", "padj": "q"})


def test_delimiters_are_detected_and_unsupported_ones_get_a_helpful_message():
    tab = generic_motif_csv(delimiter="\t")
    assert _generic(tab).record["delimiter"] == "tab" and _generic().record["delimiter"] == "comma"
    with pytest.raises(mi.MotifImportError, match="Semicolon"):
        _generic("motif;q_value;p_value\nA;0.1;0.1\n")
    with pytest.raises(mi.MotifImportError, match="tab or comma"):
        _generic("motif q_value p_value\nA 0.1 0.1\n")


def test_comments_and_blank_lines_are_skipped_and_line_numbers_stay_accurate():
    text = "# a comment\n\nmotif,q_value,p_value,peak_set\nA,0.1,0.1,x\n\n# note\nB,0.2,0.2,x\n"
    table = _generic(text)
    assert table.rows["motif_name"].tolist() == ["A", "B"]
    assert table.rows["source_line"].tolist() == [4, 7] and table.record["n_lines_skipped"] == 4


def test_duplicate_motif_rows_are_all_kept():
    table = _homer()
    assert (table.rows["motif_name"].str.lower().str.startswith("stat3")).sum() == 3      # three Stat3 motifs stay as rows


def test_file_names_are_reduced_to_a_safe_display_form():
    long_name = "x" * 500 + ".txt"
    assert len(_homer(name=long_name).record["file_name"]) == 200
    assert _homer(name="C:\\Users\\me\\out\\knownResults.txt").record["file_name"] == "knownResults.txt"
    assert _homer(name="../../etc/known\tResults.txt").record["file_name"] == "known_Results.txt"
    assert _homer(name="").record["file_name"] == "unnamed"


def test_non_numeric_percentages_are_kept_as_missing_and_do_not_stop_the_import():
    rows = [row[:6] + ("n/a",) + row[7:] for row in HOMER_DEFAULT_ROWS]
    table = _homer(homer_known_text(rows))
    assert table.rows["pct_target"].isna().all() and table.rows["padj"].iloc[0] == 0.0001


def test_homer_header_constant_matches_the_documented_required_columns():
    assert set(mi.HOMER_REQUIRED_HEADERS) <= {h.lower() for h in HOMER_HEADER}