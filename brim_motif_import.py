"""Level 3 motif import contracts: peak sets, BED files, commands and (later) result import.

This module is Streamlit-free (AGENTS.md I-3.1).  BRIM never scans sequence, never runs an external tool and makes no
network call (I-4, I-5.1): it writes BED files, shows the command a user can run outside BRIM, and imports the
result.  All thresholds, the genome build and the species are explicit arguments (I-3.2).  The decisions behind each
rule are recorded in ``docs/phase6_implementation_plan.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


class MotifImportError(ValueError):
    """Raised when peak sets or a motif result cannot be built or imported with the given inputs."""


PEAK_SETS = ("opening", "closing")
SMALL_PEAK_SET_WARNING = 100
COORDINATE_CONVENTION = "0-based half-open (BED)"
# The genome names HOMER understands for the builds BRIM supports.  Nothing typed by a user is ever placed in a command.
HOMER_GENOMES = {"hg38": "hg38", "mm10": "mm10"}
_SAFE_FILE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_REQUIRED_DAR_COLUMNS = ("peak_id", "chrom", "start", "end", "log2FoldChange", "padj", "padj_is_na", "lfc_is_na")

BACKGROUND_DEFINITION = (
    "Background: all ATAC peaks that were tested (padj and log2FC both available); peaks that were not tested "
    "(NA) are excluded. Opening and closing peaks are part of the background."
)
BACKGROUND_DEFINITION_JA = (
    "背景: 検定済みの全ATAC peak（padjとlog2FCがともに得られたもの）。未検定（NA）のpeakは含めません。"
    "openingとclosingのpeakも背景に含まれます。"
)
THRESHOLD_SOURCE_NOTE = (
    "The peak-set thresholds are the ATAC thresholds of the Level 1 integration run (the default 0.05 / 1 applies "
    "when the ATAC validation report carried none)."
)
THRESHOLD_SOURCE_NOTE_JA = (
    "peak集合の閾値は、レベル1の統合を実行したときのATAC閾値です（ATAC検証レポートに閾値がなかった場合は"
    "既定の0.05 / 1が使われます）。"
)


@dataclass(frozen=True)
class PeakSets:
    """The exported peak sets and everything needed to state, verify and reproduce them."""

    opening: pd.DataFrame
    closing: pd.DataFrame
    background: pd.DataFrame
    counts: dict[str, Any]
    thresholds: dict[str, float]
    genome_build: str
    species: str
    bed_sha256: dict[str, str]
    peakset_fingerprint: str
    warnings: list[dict[str, str]] = field(default_factory=list)


def _format_number(value: float) -> str:
    return format(float(value), "g")


def _validated_thresholds(thresholds: Mapping[str, Any]) -> dict[str, float]:
    try:
        padj, lfc = float(thresholds["atac_padj"]), float(thresholds["atac_lfc"])
    except (KeyError, TypeError, ValueError) as error:
        raise MotifImportError("Peak sets require numeric atac_padj and atac_lfc thresholds.") from error
    if not (math.isfinite(padj) and math.isfinite(lfc)) or not 0 <= padj <= 1 or lfc < 0:
        raise MotifImportError("atac_padj must be within [0, 1] and atac_lfc must be a non-negative finite number.")
    return {"atac_padj": padj, "atac_lfc": lfc}


def _boolean_flags(dar: pd.DataFrame, column: str) -> pd.Series:
    flags = dar[column]
    if flags.isna().any() or not flags.isin([True, False, 0, 1]).all():
        raise MotifImportError(f"DAR column {column} must contain only True/False values.")
    return flags.astype(bool)


def _chrom_style(chroms: pd.Series) -> dict[str, int]:
    prefixed = int(chroms.astype(str).str.lower().str.startswith("chr").sum())
    return {"chr_prefixed": prefixed, "bare": int(len(chroms) - prefixed)}


def bed_file_name(kind: str, thresholds: Mapping[str, Any]) -> str:
    """Return the shell-safe BED file name for ``opening``, ``closing`` or ``background``."""
    values = _validated_thresholds(thresholds)
    tag = f"padj{_format_number(values['atac_padj'])}_lfc{_format_number(values['atac_lfc'])}"
    names = {"opening": f"opened_peaks_{tag}.bed", "closing": f"closed_peaks_{tag}.bed",
             "background": "all_peaks_background.bed"}
    if kind not in names:
        raise MotifImportError("BED kind must be opening, closing or background.")
    if not _SAFE_FILE_NAME.match(names[kind]):
        raise MotifImportError("The threshold values produce a BED file name with unsafe characters; use plain decimals.")
    return names[kind]


def build_peak_sets(dar: pd.DataFrame, thresholds: Mapping[str, Any], genome_build: str, species: str) -> PeakSets:
    """Split a standardized DAR table into opening, closing and background peak sets.

    Only tested peaks (``padj`` and ``log2FC`` not NA) are ever exported (I-1.1).  Boundaries are inclusive and a
    log2FC of exactly 0 is never opening or closing, matching the DAR classification.  Peak identifiers are made
    shell- and BED-safe by replacing whitespace; the number replaced is recorded (I-2.2).  Chromosome names and
    coordinates are never converted.
    """
    missing = [column for column in _REQUIRED_DAR_COLUMNS if column not in dar.columns]
    if missing:
        raise MotifImportError("The DAR table is missing required columns: " + ", ".join(missing) + ".")
    values = _validated_thresholds(thresholds)
    padj_limit, lfc_limit = values["atac_padj"], values["atac_lfc"]
    padj_na, lfc_na = _boolean_flags(dar, "padj_is_na"), _boolean_flags(dar, "lfc_is_na")
    tested = ~padj_na & ~lfc_na
    padj = pd.to_numeric(dar["padj"], errors="coerce")
    lfc = pd.to_numeric(dar["log2FoldChange"], errors="coerce")
    starts = pd.to_numeric(dar["start"], errors="coerce")
    ends = pd.to_numeric(dar["end"], errors="coerce")
    if (tested & (padj.isna() | lfc.isna() | ~np.isfinite(padj.fillna(np.inf)) | ~np.isfinite(lfc.fillna(np.inf)))).any():
        raise MotifImportError("Tested peaks must have finite padj and log2FoldChange values.")
    if starts.isna().any() or ends.isna().any() or (starts % 1 != 0).any() or (ends % 1 != 0).any() \
            or (starts < 0).any() or (ends < starts).any():
        raise MotifImportError("Peak start and end must be non-negative whole numbers with end >= start.")
    chroms = dar["chrom"].astype(str)
    if chroms.str.strip().eq("").any() or chroms.str.contains(r"\s", regex=True).any():
        raise MotifImportError("Chromosome names must be non-empty and must not contain whitespace.")

    opening_mask = tested & padj.le(padj_limit) & lfc.ge(lfc_limit) & lfc.gt(0)
    closing_mask = tested & padj.le(padj_limit) & lfc.le(-lfc_limit) & lfc.lt(0)
    original_ids = dar["peak_id"].astype(str)
    sanitized = original_ids.str.replace(r"\s", "_", regex=True)
    exported = tested
    if sanitized[exported].duplicated().any():
        examples = sorted(set(sanitized[exported][sanitized[exported].duplicated(keep=False)]))[:5]
        raise MotifImportError("Peak identifiers collide after whitespace replacement: " + ", ".join(examples) + ".")
    frame = pd.DataFrame({"chrom": chroms, "start": starts.astype("Int64"), "end": ends.astype("Int64"),
                          "name": sanitized})

    def part(mask: pd.Series) -> pd.DataFrame:
        return frame.loc[mask].reset_index(drop=True)

    opening, closing, background = part(opening_mask), part(closing_mask), part(exported)
    n_sanitized = int((original_ids[exported] != sanitized[exported]).sum())
    style = _chrom_style(chroms[exported])
    counts = {
        "n_dar_total": int(len(dar)), "n_tested": int(tested.sum()), "n_not_tested_excluded": int((~tested).sum()),
        "n_opening": int(len(opening)), "n_closing": int(len(closing)), "n_background": int(len(background)),
        "n_ids_sanitized": n_sanitized, "chrom_style": style,
    }
    warnings: list[dict[str, str]] = []
    for kind, part_frame in (("opening", opening), ("closing", closing)):
        if part_frame.empty:
            warnings.append({"code": "empty_peak_set", "peak_set": kind,
                             "message": f"The {kind} peak set has 0 peaks; no BED file or command is written for it.",
                             "message_ja": f"{kind}のpeak集合は0件のため、BEDファイルとコマンドは出力しません。"})
        elif len(part_frame) < SMALL_PEAK_SET_WARNING:
            warnings.append({"code": "small_peak_set", "peak_set": kind,
                             "message": f"The {kind} peak set has only {len(part_frame)} peaks (fewer than "
                                        f"{SMALL_PEAK_SET_WARNING}); treat the motif result as exploratory.",
                             "message_ja": f"{kind}のpeak集合は{len(part_frame)}件（{SMALL_PEAK_SET_WARNING}件未満）です。"
                                           "motif結果は探索的に解釈してください。"})
    if style["chr_prefixed"] and style["bare"]:
        warnings.append({"code": "mixed_chromosome_naming", "peak_set": "all",
                         "message": "Chromosome names are a mix of chr-prefixed and bare names. Check that they match "
                                    "the genome used by the external tool; BRIM does not convert them.",
                         "message_ja": "染色体名の表記が混在しています。外部ツールのgenomeと表記が合うか確認してください"
                                       "（BRIMは変換しません）。"})
    bed_texts = {"opening": _bed_text(opening), "closing": _bed_text(closing), "background": _bed_text(background)}
    bed_sha256 = {kind: hashlib.sha256(text.encode("utf-8")).hexdigest() for kind, text in bed_texts.items()}
    fingerprint = hashlib.sha256(json.dumps({
        "thresholds": values, "genome_build": str(genome_build), "species": str(species),
        "coordinate_convention": COORDINATE_CONVENTION, "chrom_style": style, "bed_sha256": bed_sha256,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    return PeakSets(opening=opening, closing=closing, background=background, counts=counts, thresholds=values,
                    genome_build=str(genome_build), species=str(species), bed_sha256=bed_sha256,
                    peakset_fingerprint=fingerprint, warnings=warnings)


def _bed_text(frame: pd.DataFrame) -> str:
    return "".join(f"{row.chrom}\t{int(row.start)}\t{int(row.end)}\t{row.name}\n" for row in frame.itertuples(index=False))


def peak_set_bed_text(peak_sets: PeakSets, which: str) -> str:
    """Return the 4-column BED text (chrom, start, end, name; 0-based half-open) of one peak set."""
    frames = {"opening": peak_sets.opening, "closing": peak_sets.closing, "background": peak_sets.background}
    if which not in frames:
        raise MotifImportError("BED kind must be opening, closing or background.")
    return _bed_text(frames[which])


def homer_unavailable_reason(peak_sets: PeakSets) -> dict[str, str] | None:
    """Explain why no command is shown, or return None when commands can be shown."""
    if peak_sets.genome_build not in HOMER_GENOMES:
        return {"code": "genome_not_supported",
                "message": f"No command is shown because the genome build '{peak_sets.genome_build}' is not one BRIM "
                           "can name for the external tool (hg38 and mm10 are supported).",
                "message_ja": f"genome build「{peak_sets.genome_build}」は外部ツール用に名前を示せるbuild"
                              "（hg38、mm10）ではないため、コマンドを表示しません。"}
    if peak_sets.background.empty:
        return {"code": "empty_background", "message": "No command is shown because there are no tested peaks.",
                "message_ja": "検定済みpeakがないため、コマンドを表示しません。"}
    if peak_sets.opening.empty and peak_sets.closing.empty:
        return {"code": "no_peak_sets", "message": "No command is shown because both peak sets are empty.",
                "message_ja": "openingとclosingの両方が0件のため、コマンドを表示しません。"}
    return None


def build_homer_commands(peak_sets: PeakSets) -> list[str]:
    """Return the command lines a user can run outside BRIM (never executed by BRIM).

    Only the genome name from a fixed allow-list and the BRIM-generated file names are inserted; nothing a user typed
    can reach a command.  ``-size 200`` is an example value, not a recommendation.
    """
    if homer_unavailable_reason(peak_sets) is not None:
        return []
    genome = HOMER_GENOMES[peak_sets.genome_build]
    background = bed_file_name("background", peak_sets.thresholds)
    commands = [f"perl configureHomer.pl -install {genome}"]
    for kind, frame, output in (("opening", peak_sets.opening, "homer_opening/"),
                                ("closing", peak_sets.closing, "homer_closing/")):
        if not frame.empty:
            commands.append(f"findMotifsGenome.pl {bed_file_name(kind, peak_sets.thresholds)} {genome} {output} "
                            f"-size 200 -bg {background}")
    return commands


def render_motif_readme(peak_sets: PeakSets, app_version: str, generated_at: str) -> str:
    """Return the bilingual README shipped with the BED files (a pure function of its inputs)."""
    counts, thresholds = peak_sets.counts, peak_sets.thresholds
    commands = build_homer_commands(peak_sets)
    reason = homer_unavailable_reason(peak_sets)
    lines = [
        "BRIM motif analysis files / BRIMのmotif解析用ファイル",
        "=" * 60,
        f"Generated by BRIM {app_version} at {generated_at} / BRIM {app_version} が {generated_at} に生成",
        "",
        "What is in this folder / このフォルダの内容",
    ]
    for kind in ("opening", "closing", "background"):
        n = counts["n_background"] if kind == "background" else counts[f"n_{kind}"]
        state = f"{bed_file_name(kind, thresholds)} ({n} peaks)" if n else f"(not written: 0 {kind} peaks / 0件のため出力なし)"
        lines.append(f"- {kind}: {state}")
    lines += [
        "",
        f"Peak-set thresholds / peak集合の閾値: padj <= {_format_number(thresholds['atac_padj'])}, "
        f"|log2FC| >= {_format_number(thresholds['atac_lfc'])} (log2FC = 0 is never opening or closing)",
        THRESHOLD_SOURCE_NOTE, THRESHOLD_SOURCE_NOTE_JA,
        f"Genome build / genome build: {peak_sets.genome_build}  Species / 種: {peak_sets.species}",
        f"Coordinates / 座標系: {COORDINATE_CONVENTION}. Chromosome names and coordinates are written as in the DAR "
        "table and are not converted / DAR表のまま出力し、変換していません。",
        f"Chromosome naming counts / 染色体名の表記の件数: {counts['chrom_style']}",
        f"Peak identifiers with whitespace replaced by '_' / 空白を'_'に置換したpeak ID数: {counts['n_ids_sanitized']}",
        "",
        BACKGROUND_DEFINITION, BACKGROUND_DEFINITION_JA,
        f"Peaks in the DAR table / DAR表のpeak数: {counts['n_dar_total']}; tested / 検定済み: {counts['n_tested']}; "
        f"not tested and excluded / 未検定で除外: {counts['n_not_tested_excluded']}",
        "",
        "How to use / 使い方",
        "1. Prepare the external tool by following that tool's own documentation (BRIM does not bundle or run it). "
        "HOMER is intended for Linux/macOS-type environments, so on Windows a layer such as WSL is usually needed; "
        "check the tool's documentation. / 外部ツールは、そのツールの文書に従って用意してください"
        "（BRIMは同梱も実行もしません）。HOMERはLinux/macOS系向けで、Windowsでは通常WSLなどが必要です。",
        "2. Put the BED files in one folder. / BEDファイルを1つのフォルダに置きます。",
        "3. Run the command(s) below outside BRIM. -size 200 is an example value. / 下のコマンドをBRIMの外で実行します。"
        "-size 200は例です。",
    ]
    if commands:
        lines += ["   (The first line is needed only if the genome is not installed yet; it is run by you, outside BRIM, "
                  "and may download data. / 1行目はgenomeが未導入の場合のみ必要で、利用者がBRIMの外で実行し、"
                  "データをダウンロードすることがあります。)"] + [f"   {command}" for command in commands]
    else:
        lines.append(f"   {reason['message']} / {reason['message_ja']}" if reason else "   (no command)")
    lines += [
        "4. Import the resulting result files (for example homer_opening/knownResults.txt) in the BRIM Level 3 import "
        "area, choosing opening or closing. / できた結果ファイル（例: homer_opening/knownResults.txt）を、"
        "BRIMのLevel 3の取り込み欄で、openingまたはclosingを選んで読み込みます。",
        "",
        "About the exported tables / 出力される表について",
        "- Integration/tf_candidates.csv keeps motif_status=not_run by design. Integration/tf_candidates_with_motif.csv "
        "is the view with the motif columns. / tf_candidates.csvは設計上motif_status=not_runのままです。"
        "motif列付きのビューはtf_candidates_with_motif.csvです。",
        "- Motif results are produced by an external tool that BRIM does not run or verify, and they are candidates "
        "for further checking. / motif結果はBRIMが実行も検証もしない外部ツールの出力で、追加の確認が必要な候補です。",
        "",
    ]
    return "\n".join(lines)


def build_motif_bundle(peak_sets: PeakSets, app_version: str, generated_at: str) -> dict[str, str]:
    """Return ``MotifAnalysis/`` file contents by archive path (empty peak sets are not written)."""
    bundle: dict[str, str] = {}
    for kind in ("opening", "closing", "background"):
        text = peak_set_bed_text(peak_sets, kind)
        if text:
            bundle[f"MotifAnalysis/{bed_file_name(kind, peak_sets.thresholds)}"] = text
    bundle["MotifAnalysis/motif_analysis_README.txt"] = render_motif_readme(peak_sets, app_version, generated_at)
    return bundle


def bundle_sha256(bundle: Mapping[str, str]) -> dict[str, str]:
    """Return the SHA-256 of each exported file body (UTF-8), for the manifest."""
    return {path: hashlib.sha256(text.encode("utf-8")).hexdigest() for path, text in sorted(bundle.items())}
