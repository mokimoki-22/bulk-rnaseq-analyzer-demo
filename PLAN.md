# PLAN.md — BRIM v2.0 実装計画

- 対応設計書: `B`docs/BRIM_RNA_ATAC_Integration_Design_v2.md`（版2.0, 2026-09-04）
- 現在Phase: **Phase 1**
- 最終更新: 2026-09-04

各タスクは現在Phaseの範囲のみを実装する。将来Phaseの機能を先取りしない。
Phaseを進めるときはこのファイルの「現在Phase」を更新する。

---

## Phase 0: 安全な土台

**状態:** 完了（2026-09-04）

**作業**
- 現在版をバックアップ／tag化（v1.1.0）
- Git repositoryを準備
- OSI承認LICENSEを追加
- README.md、CHANGELOG.md、CONTRIBUTING.md を追加
- AGENTS.md、PLAN.md、ARCHITECTURE.md を配置
- 現在のテストを全件成功させる
- CIを設定する

**完了条件**
ATAC追加前のbaseline testが再現可能。

**このPhaseで触らないもの**
`Bulk_RNAseq_Analyzer.py` の解析ロジック。ドキュメントとCIのみ。

---

## Phase 0.5: 既存コードの前提整備

**状態:** 完了（2026-09-04）
**設計書参照:** §3.4, §16.1, §16.3

このPhaseをATAC実装より前に置く理由は、統合分類のロジックが `padj_is_na`
に依存するためである。後から追加すると分類実装の書き直しが発生する。

**作業**

1. `run_deg()` に NA フラグを追加
   ```python
   res["padj_is_na"] = res["padj"].isna()
   res["lfc_is_na"] = res["log2FoldChange"].isna()
   res["padj"] = res["padj"].fillna(1.0)
   res["log2FoldChange"] = res["log2FoldChange"].fillna(0.0)
   res["stat"] = res["stat"].fillna(0.0)
   ```
   既存のfillnaは維持する。下流の挙動を変えないこと。

2. `brim_provenance.py` を新規作成
   - `file_checksum(file_obj) -> str`
   - `collect_environment() -> dict`
   - `build_manifest(inputs, settings, counts, services) -> dict`
   - `render_manifest_markdown(manifest) -> str`

3. RNA単独解析のmanifest生成を `brim_provenance.py` 経由へ移行
   既存の `reproducibility_report.json` を置き換える。

4. 外部サービス使用の明示
   - `run_online_mapping()`（mygene.info）実行前に送信先と送信内容種別を表示
   - `get_string_network_img()`（string-db.org）も同様
   - manifest に `external_services_used` を記録

5. 回帰テスト
   - `test_provenance.py` を新規作成
   - 既存RNAテストが全件成功すること

**完了条件**
既存RNA機能の出力が意味的に変化せず、NA情報とmanifestが追加されている。

**Acceptance criteria**
- 既存の全テストがパスする
- `deg_results` に `padj_is_na` / `lfc_is_na` 列が存在する
- NA件数がmanifestに記録される
- RNA count matrix の SHA-256 がmanifestに含まれる
- 外部サービス使用時に画面表示があり、manifestに記録される

**完了記録**
- 段階1の変更前DEG・JSON・ZIP fixtureを独立コミット`22ad00b`に保存してから実装。
- 監査項目1〜5（既存6列の数値互換性、NA4ケース、結果ありExport、共有manifestの
  単体・接続、結果ありUIと外部サービス記録）をすべて検証。Phase 2への繰り延べなし。
- `Provenance/manifest.json` / `Provenance/manifest.md`を既存ZIPに追加し、
  単独ダウンロードとの内容一致をテスト。旧JSON全項目の対応表はCHANGELOGに記載。
- Windows / Python 3.12で全37テスト成功。既存3テストは変更なし。
  人工小規模データでの既存PyDESeq2 dispersion fallback警告3件あり。
- AppTestで初期起動、英語・日本語の結果あり画面、単一／複数Studyの読み込みを確認。
  HTTP通信と静的画像変換はモック。GitHub Actionsの4環境実行は未確認。

---

## Phase 1: ATAC core

**状態:** 未着手
**設計書参照:** §7.1–7.3, §8.1–8.2, §9, §14.1

**作業**
- `brim_atac.py` を新規作成（Streamlit非依存）
- peak count matrix の読み込みと座標解析（index形式・3列形式）
- DAR推定（pydeseq2、正規化方式の選択）
- DAR schema validation
- gene/TSS reference の読み込み（hg38, mm10）
- promoter overlap mapping
- nearest TSS mapping
- unit tests（`test_atac.py`）
- sample data（両入力モード）

**完了条件**
StreamlitなしでATAC annotationが完結し、期待edgeと一致する。
両入力モードが同一の標準DARテーブルへ収束する。

**Acceptance criteria**
- `chr1:100200-100700` 形式と3列形式の両方を解析できる
- count matrix から DAR を推定できる
- n=1 で `InsufficientSampleError` を送出する
- n=2 で警告フラグを返す（例外ではない）
- padj の NA を保持し `padj_is_na` を立てる
- start/end逆転、負座標、非数値、padj範囲外を拒否する
- promoter overlap が strand を考慮する
- nearest TSS の tie で全遺伝子を保持する
- 一対多 mapping が保持される
- 最大距離外が unmapped になる

**未決事項（Phase 1中に決定）**
- interval join を pandas/NumPy で実装するか `bioframe` を採用するか
- gene annotation の source と release
- count matrix モードの既定正規化方式
- peak数20万超の実行時間対策

---

## Phase 2: ATAC UI

**状態:** 未着手
**設計書参照:** §6.1–6.2, §11.1, §15

**作業**
- `Multi-omics` タブを追加（既存7タブ→8タブ）
- `ATAC-seq` サブタブ
- 入力モード選択／column mapping／正規化選択
- validation summary（サンプル数警告を含む）
- annotation settings
- ATAC plot／table／download
- session state追加と `reset_atac_results()`
- UI tests

**完了条件**
RNA入力なしでATAC単独解析が完了する。

**Acceptance criteria**
- 両入力モードでATAC単独解析が通る
- サンプル数警告が表示される
- ATAC入力変更で下流結果がクリアされる
- RNAのみを使う既存操作が変化していない
- 日本語・英語の主要UIが表示される

---

## Phase 3: Integration core（レベル1）

**状態:** 未着手
**設計書参照:** §8.3–8.4, §10, §14.2

**作業**
- `brim_multiomics.py` を新規作成（Streamlit非依存）
- compatibility check
- edge classification（`rna_not_tested` / `atac_not_tested` を含む）
- gene summary（mixed accessibility の保持）
- thresholds
- unit tests（`test_multiomics.py`）
- 陰性対照テスト（§17.4）

**完了条件**
人工fixtureで全分類とmixed peakを正しく処理し、陰性対照でスコアが低下する。

**Acceptance criteria**
- 全分類を生成する
- `padj_is_na` の遺伝子が `atac_only` へ混入しない
- contrast逆向き・species不一致を拒否する
- mixed を単一方向へ潰さない
- 元edgeが失われない
- 同じ入力と設定から同じ出力を得る
- peak–gene対応をシャッフルすると concordant/discordant 比が偶然水準に近づく
- RNAラベルをシャッフルすると分類の偏りが消失する

---

## Phase 4: Integration UI（レベル1）

**状態:** 未着手
**設計書参照:** §6.3, §11.2, §16.2

**作業**
- `Integration` サブタブ（ATAC結果とRNA結果が揃った場合のみ表示）
- compatibility check 表示
- quadrant plot
- evidence table（edge単位／gene単位）
- class filter
- class別 KEGG／GO enrichment
- discordant群の可視性確保（下部に埋もれさせない）
- export／provenance
- UI tests

**完了条件**
BRIM sample RNAとsample ATACから一連の解析・出力が完了する。

**Acceptance criteria**
- RNA結果がない場合 Integration サブタブが表示されない
- genome build不一致・contrast方向不一致を表示する
- ATAC変更時に古い統合結果が消える
- Export に必要ファイルが含まれる

---

## Phase 5: TF候補推定（レベル2）

**状態:** 未着手
**設計書参照:** §6.4, §8.5, §12.1–12.3, §14.3

**作業**
- `brim_tf_integration.py` を新規作成（Streamlit非依存）
- 背景（universe）の構築：mapped かつ tested 遺伝子
- 標的濃縮（Fisher正確検定 + BH補正）
- TF発現の結合（`deg_results` から）
- TF activity の結合（既存 `tf_collectri` / `tf_dorothea` から）
- 3軸テーブル（motif列は未実行として空欄表示）
- 段階開示UI（レベル1実行後にボタンが出る）
- unit tests（`test_tf_integration.py`）

**完了条件**
sample dataからTF候補が提示され、背景と多重検定の扱いが結果に明記される。

**Acceptance criteria**
- Fisher検定が既知の分割表で正しい値を返す
- 背景が mapped かつ tested に限定される
- 背景の定義と件数が画面に表示される
- 遺伝子集合が20未満のとき警告フラグが立つ
- BH補正が適用される
- 3軸が単一スコアへ合成されていない
- レベル1未実行時にレベル2ボタンが出ない
- レベル2の限界（翻訳後修飾TF、DBバイアス）が画面に明示される

---

## Phase 6: motif統合（レベル3）

**状態:** 未着手
**設計書参照:** §7.6, §12.4, §14.3

**作業**
- BED書き出し（`opened_peaks`, `closed_peaks`, `all_peaks_background`）
- `motif_analysis_README.txt` の生成
- 実行コマンドの提示（genome build を設定から自動補完）
- motif結果のインポートとTFシンボル正規化
- 3軸テーブルへの結合
- motif出典の記録（ツール、版、database、閾値、背景）
- unit tests
- `docs/motif_analysis_guide.md`

**完了条件**
外部ツールの結果を読み込み、motif列が埋まる。出典と解析条件がmanifestに残る。

**Acceptance criteria**
- BED出力が閾値どおりのpeakを含む
- opening/closing が別ファイルに分離される
- HOMER複合名（`Stat3(Stat)/mES-...`）からシンボルを抽出できる
- 照合できなかったmotifの件数と一覧が表示される
- opening/closing が `peak_set` 列で区別される
- motif未実行時に該当列がNAとなり他軸が影響を受けない
- 入力閾値が現在のBRIM設定と食い違う場合に警告する

---

## Phase 7: 検証と公開

**状態:** 未着手
**設計書参照:** §17.3, §21

**作業**
- 公開paired datasetによるcase study
  - Corces et al. 2016 造血（GSE74912）— CI用軽量fixture
  - ENCODE cell line — パイプライン統一済み対照
  - TCGA ATAC atlas — peak–gene linking の照合
- 既存手法／単純nearest-geneとの比較
- wet-lab researcherによる操作確認
- installation guide と tutorial
- `CITATION.cff`
- GitHub release
- DOI対応

**完了条件**
第三者がREADMEのみで同じ結果を再現できる。

---

## Phase順序を変更してはいけない理由

- **Phase 0.5 は Phase 3 より前**：分類ロジックが `padj_is_na` に依存する
- **Phase 5・6 は Phase 7 より前**：TF軸がないと Statement of Need が成立しない
  （設計書§21.2）。レベル2の3軸はいずれもRNA由来であり、レベル3を含めて初めて
  ATACを入れた見返りが出力に現れる
