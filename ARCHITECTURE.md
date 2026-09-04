# ARCHITECTURE.md — BRIM

- 対応設計書: `docs/BRIM_RNA_ATAC_Integration_Design_v2.md`（版2.0）
- 最終更新: 2026-09-04

このファイルは境界と責務を定義する。仕様の詳細は設計書を参照する。

---

## 1. レイヤ境界

```
┌─────────────────────────────────────────────┐
│ Bulk_RNAseq_Analyzer.py                     │  UI層
│  - Streamlit widget、タブ構成                │
│  - session state の読み書き                  │
│  - エラー表示、ダウンロードボタン             │
└───────────────┬─────────────────────────────┘
                │  DataFrame + 明示的な設定値
                ▼
┌─────────────────────────────────────────────┐
│ brim_enrichment.py     （既存）              │  解析層
│ brim_tf_networks.py    （既存）              │  Streamlit非依存
│ brim_atac.py           （Phase 1）           │
│ brim_multiomics.py     （Phase 3）           │
│ brim_tf_integration.py （Phase 5）           │
│ brim_provenance.py     （Phase 0.5）         │
└───────────────┬─────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│ references/            参照データ            │
│ sample_data/           サンプルデータ         │
└─────────────────────────────────────────────┘
```

### 境界の規則

**解析層は Streamlit に依存しない。** 次を行わない。

- `import streamlit`
- `st.session_state` の読み書き
- `st.error` / `st.warning` / `st.stop` の呼び出し
- ダウンロードUIの生成
- 暗黙の閾値取得（すべて引数で受ける）

解析関数は DataFrame と明示的な設定値を受け取り、DataFrame または型付き結果を返す。
検証上の問題は戻り値または例外として返し、表示はしない。

**UI層は計算を持たない。** イベントハンドラ内に直接計算を書かず、解析層の関数を呼ぶ。

---

## 2. ファイル構成

```text
brim-app/
├─ Bulk_RNAseq_Analyzer.py       # Streamlit UI、既存RNA機能
├─ i18n.py                       # 多言語（既存）
├─ brim_enrichment.py            # KEGG/GO/GSEA（既存）
├─ brim_tf_networks.py           # TF activity: CollecTRI/DoRothEA（既存）
├─ brim_atac.py                  # ATAC入力、検証、DAR推定、annotation
├─ brim_multiomics.py            # RNA–ATAC統合、分類、要約
├─ brim_tf_integration.py        # TF候補推定、motif結果の統合
├─ brim_provenance.py            # manifest、checksum、環境情報
├─ references/
│  ├─ *.csv                      # 既存の自動走査対象（_EXTERNAL_REFS）
│  └─ genome_annotations/
│     ├─ hg38_genes.parquet
│     ├─ mm10_genes.parquet
│     └─ manifest.json
├─ sample_data/
│  └─ multiomics/
├─ test_atac.py
├─ test_multiomics.py
├─ test_tf_integration.py
├─ test_provenance.py
├─ test_multiomics_ui.py
└─ docs/
   ├─ multiomics_input_format.md
   └─ motif_analysis_guide.md
```

初期実装では package 化しない。安定後に `brim_core/` へ移行できるよう、
各モジュールの公開関数を限定する。

**参照データの2系統は意図的に分離する。**
既存の `references/*.csv` は glob で自動走査され `_EXTERNAL_REFS` に読み込まれる。
genome annotation はサイズと構造が異なるため `references/genome_annotations/` に
Parquet + manifest で配置し、既存の走査対象に含めない。統合しないことが設計判断である。

---

## 3. モジュール責務と公開API

### 3.1 `brim_atac.py`（Phase 1）

ATAC-seq入力の読み込み、検証、DAR推定、peak–gene mapping。

```python
read_peak_count_matrix(file_obj, sep, coordinate_column_mode) -> pd.DataFrame
parse_peak_coordinates(index_or_columns) -> pd.DataFrame
run_dar(counts_df, metadata, ref_condition, test_condition,
        normalization, n_cpus) -> pd.DataFrame
read_dar_table(file_obj, sep, column_map) -> pd.DataFrame
validate_dar_table(df, coordinate_system) -> ValidationResult
standardize_chromosomes(df, build) -> tuple[pd.DataFrame, TransformLog]
load_gene_annotation(build) -> pd.DataFrame
map_peaks_to_promoters(peaks, genes, upstream, downstream) -> pd.DataFrame
map_peaks_to_nearest_tss(peaks, genes, max_distance) -> pd.DataFrame
merge_peak_gene_evidence(*edge_tables) -> pd.DataFrame
summarize_atac_qc(peaks, edges, settings) -> dict
export_peaks_as_bed(dar_df, direction, thresholds) -> str
```

`run_dar` は既存の `run_deg` と同じく pydeseq2 を使う。行が遺伝子か peak かの
違いを除き処理は同一。正規化方式は引数で受け、既定を暗黙適用しない。

### 3.2 `brim_multiomics.py`（Phase 3）

RNA DEG結果とpeak–gene edgeの統合、分類、gene単位要約。

```python
standardize_rna_results(deg_results, gene_id_type) -> pd.DataFrame
check_integration_compatibility(rna_meta, atac_meta) -> CompatibilityResult
integrate_peak_gene_edges(rna, edges, thresholds) -> pd.DataFrame
classify_integration_edges(integrated, thresholds) -> pd.DataFrame
summarize_integration_by_gene(integrated, thresholds) -> pd.DataFrame
extract_gene_set(summary, integration_class) -> list[str]
build_integration_summary(edges, genes, settings) -> dict
```

### 3.3 `brim_tf_integration.py`（Phase 5–6）

TF候補の推定と、外部motif結果の統合。

```python
build_universe(edges, rna) -> set[str]
test_target_enrichment(gene_set, universe, tf_network) -> pd.DataFrame
attach_tf_expression(tf_table, deg_results) -> pd.DataFrame
attach_tf_activity(tf_table, tf_activity_results) -> pd.DataFrame
read_motif_results(file_obj, tool, column_map) -> pd.DataFrame
normalize_tf_symbols(motif_df, species) -> tuple[pd.DataFrame, UnmatchedReport]
attach_motif_enrichment(tf_table, motif_df, peak_set) -> pd.DataFrame
build_tf_summary(tf_table, settings) -> dict
```

TF network の取得は既存 `brim_tf_networks.py` の
`load_collectri_network` / `load_dorothea_network` を再利用する。
TF activity の推定は既存 `infer_tf_activity` の結果を受け取るのみで、再計算しない。

### 3.4 `brim_provenance.py`（Phase 0.5）

manifest生成。**RNA単独解析でもこのモジュールを経由する。**

```python
file_checksum(file_obj) -> str
collect_environment() -> dict
build_manifest(inputs, settings, counts, services) -> dict
render_manifest_markdown(manifest) -> str
```

RNA用とATAC用で別々のmanifest生成経路を作らない。既存の
`reproducibility_report.json` はこの出力に置き換える。

### 3.5 例外

UIで解釈できる専用例外を定義する。英語の安定した error code と人向け詳細を持たせ、
多言語UI側で翻訳できるようにする。

```python
DARSchemaError
PeakCountMatrixError
CoordinateSystemError
GenomeBuildError
ReferenceAnnotationError
ContrastMismatchError
GeneIdentifierError
MotifImportError
InsufficientSampleError
```

---

## 4. データフロー

```text
RNA-seq raw count                  ATAC peak count matrix
    ↓ run_deg (既存)                   ↓ read_peak_count_matrix
deg_results                            ↓ run_dar
（padj_is_na を保持）                   │
    │                          または  │  read_dar_table
    │                                  ↓
    │                          標準DARテーブル（§8.1）
    │                                  ↓ map_peaks_to_*
    │                          peak–gene edges（§8.2）
    │                                  │
    └──────────────┬───────────────────┘
                   ↓ classify_integration_edges
        統合edgeテーブル（§8.3）／gene summary（§8.4）   ← レベル1
                   ↓ test_target_enrichment 他
        TF候補テーブル（§8.5、motif列はNA）              ← レベル2
                   ↓ attach_motif_enrichment
        TF候補テーブル（motif列が埋まる）                 ← レベル3
```

**2つの入力モードは標準DARテーブルで合流する。** 下流は入力モードを意識しない。
モードの区別は `source_mode` 列とmanifestにのみ残る。

---

## 5. UI構成

```text
Upload | DEG | Multi-omics | Visualization | Network | Meta | Export | Info
```

`Multi-omics` タブ内はサブタブ。

```text
Multi-omics
  ├ ATAC-seq          （常時表示）
  └ Integration       （ATAC結果とRNA DEG結果が揃った場合のみ表示）
```

`Integration` サブタブは段階開示。レベル1の結果を表示した後に、レベル2の追加ボタンが
現れる。レベル2の後にレベル3。先頭で3択を提示しない。

段階開示により、レベルの状態は各 session state キーが None かどうかの2値になる。
レベル間遷移という概念を持たない。

---

## 6. Session state

### 6.1 追加キー

```text
atac_input_mode              # count_matrix / dar_table
atac_counts_df
atac_metadata
atac_normalization
atac_input_df
atac_validated_df
atac_validation_report
atac_genome_build
atac_species
atac_contrast
atac_mapping_settings
atac_peak_gene_edges
atac_unmapped_peaks
atac_results
integration_edge_results
integration_gene_results
integration_settings
integration_summary
integration_enrichment
integration_tf_results          # レベル2
integration_motif_results       # レベル3
integration_motif_source
integration_provenance
```

### 6.2 リセット関数

既存の `reset_data_results()` と `_DATA_RESULT_DEFAULTS` にATAC状態を混在させない。
別の辞書と関数を追加する。

```python
reset_atac_results()
reset_integration_results()
reset_peak_mapping_results()
reset_tf_integration_results()
```

### 6.3 Invalidation規則

| 変更 | クリアする結果 |
|---|---|
| ATAC入力モード変更 | ATAC全体、Integration全体、TF、motif |
| ATACファイル変更 | ATAC annotation以降、Integration全体、TF、motif |
| 正規化方式変更 | DAR推定以降すべて |
| genome build変更 | ATAC annotation以降、Integration全体、TF、motif |
| promoter window変更 | peak–gene edge以降、Integration全体、TF、motif |
| ATAC閾値変更 | ATAC分類、Integration分類、enrichment、TF、motif |
| RNA DEG再実行 | Integration全体、TF、motif |
| RNA閾値変更 | Integration分類、enrichment、TF、motif |
| contrast方向変更 | Integration全体、TF、motif |
| species変更 | ATACとIntegration全体、TF、motif |

レベル2・3の結果は常に上流の変更で破棄する。上流が変わったまま古いTF結果が残る
状態を許さない。

RNAデータ差し替え時は `reset_integration_results()` と
`reset_tf_integration_results()` を呼ぶが、独立して読み込まれたATAC原表は保持可能。
ただし species / contrast 不一致時は Integration を無効化する。

---

## 7. 外部通信

許可されているのは既存の2つのみ。

| 関数 | 送信先 | 送信内容 |
|---|---|---|
| `run_online_mapping()` | mygene.info | gene ID |
| `get_string_network_img()` | string-db.org | 遺伝子リスト |

いずれもユーザーの明示操作の後にのみ実行し、送信先と内容種別を事前表示する。
manifest に `external_services_used` として記録する。

**ATAC・統合機能は外部通信を追加しない。**

---

## 8. 依存関係

### 8.1 既存を再利用する

- pydeseq2 — RNA DEG および ATAC DAR
- scipy — Fisher正確検定
- statsmodels — BH補正
- pandas / numpy — データ操作、interval join
- plotly / matplotlib / seaborn — 可視化
- streamlit — UI層のみ

新規の統計エンジンを実装しない。DAR推定はRNAと同じ pydeseq2 を使う。

### 8.2 追加候補（未決）

- `bioframe` — genomic interval操作
- `pyarrow` — Parquet参照ファイル

追加前に NumPy/pandas のみの chromosome 別 interval 処理と性能を比較する。

### 8.3 追加してはいけないもの

Windows portable 配布を壊す依存。具体的には、ゲノムFASTA、BAMファイル、または
外部バイナリ（bedtools、HOMER、MEME、samtools）を必要とするもの。

---

## 9. 参照データの管理

`references/genome_annotations/` 以下のファイルには manifest.json で次を付与する。

- source
- release
- download URL
- license / terms
- build
- SHA-256 checksum
- generation script

BRIM本体には完全なGTFではなく、必要列に限定した圧縮済み gene/TSS table を同梱する。

対応build: hg38 / GRCh38、mm10 / GRCm38。
将来候補: hg19 / GRCh37、mm39 / GRCm39。

---

## 10. 変更時の手続き

このファイルの内容を変更する場合は、次を行う。

1. 変更を実装前に報告する（AGENTS.md「Do not make broad architectural changes
   without reporting them first」）
2. 設計書の該当章との整合を確認する
3. このファイルと設計書の両方を更新する

境界（§1）と禁止事項（§8.3）の変更は、設計書§21の差別化の主張に直接影響するため、
特に慎重に扱う。
