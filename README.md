# BRIM — Bulk RNA-seq Insight in Minutes

BRIMは、Streamlitで動作するBulk RNA-seq解析アプリです。カウント行列の読込み、QC、
PyDESeq2による差次的発現解析、可視化、経路濃縮解析、TF activity推定、ネットワーク表示、
デコンボリューション、出力を提供します。

v1.1.0をRNA-onlyの基準線として固定し、`PLAN.md`に定義されたPhase順で
RNA-seqとATAC-seqの統合機能を追加しています。Phase 1ではStreamlit非依存の
ATAC core（count matrix / 解析済みDARの標準化、DAR推定、GENCODE TSSを使う
promoter / nearest-TSS mapping）を提供します。UIはPhase 2で追加します。

## 必要環境

- Python 3.11 または 3.12
- pip

依存関係は`requirements.txt`で管理します。

## 起動方法

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run Bulk_RNAseq_Analyzer.py
```

macOS/Linuxでは仮想環境を有効化した後、同じ`pip install`と`streamlit run`コマンドを
使用します。

## テスト

```powershell
python -m pytest -q
```

基準線テストは、既存モジュールのimport、人工count matrixに対する`run_deg()`、
Streamlit AppTestによるアプリ起動、および既存の主要タブ描画を確認します。
さらに、保存済みの変更前DEG・Exportとの互換性、NAフラグ、manifestの内容と
共有生成器への接続、結果ありUI、外部サービス操作を検証します。
HTTP通信と画像変換はテスト内でモックし、解析・JSON/ZIP生成は実コードを実行します。
ATAC coreのテストでは両入力形式、NA保持、3種類の明示的な正規化、座標変換、
strand-aware mapping、nearest-TSS tie、一対多edge、固定GENCODE参照も検証します。

## ATAC core（Phase 1、Python API）

`brim_atac.py`はStreamlitをimportせず、DataFrameと明示設定だけを受け取ります。
count matrixは`chr1:100200-100700`形式または`chrom,start,end`列形式、解析済みDARは
標準列または対応aliasを受け付けます。座標系、genome build、正規化方式、padj/LFC閾値は
推測せず呼び出し側が指定します。サンプル入力は`sample_data/BRIM_ATAC_*`にあります。

count matrixの正規化はDESeq2 median-of-ratios、total reads in peaks、user-supplied
size factorsに対応します。事前フィルタは既定無効で、有効時の初期候補は全サンプル合計
count 10未満の除外です。選択方式、size factor、入力・除外・解析peak数は返却DataFrameの
`attrs`に保持され、Phase 2で共有manifestへ接続します。

## RNA結果とprovenance

DEG結果には`padj_is_na` / `lfc_is_na`が含まれます。これらはPyDESeq2の元の
欠損値を表し、従来の数値補完（padj=1、log2FoldChange=0）は維持されます。
`padj_is_na=True`を「検定済みで有意でない」と解釈しないでください。

ExportのZIPには既存のCSV等に加えて`Provenance/manifest.json`と
`Provenance/manifest.md`が含まれます。単独ダウンロードも同じ内容です。
旧`reproducibility_report.json`からの項目対応はCHANGELOGに記載しています。

manifestは環境・入力・設定・件数・外部サービスを記録します。アップロード原本の
SHA-256は`inputs.rna.source_files`、アプリが保持してZIPに出力するcount matrixの
SHA-256は`inputs.rna.count_matrix`に記録します。後者はID変換・重複処理後の行列で、
UTF-8 CSVの実際の出力バイト列を対象とします（OSの改行差でもハッシュは変わります）。
サンプルデータや既にメモリ上にある結果で原本がない場合、原本情報を捏造せず空欄にします。
NAフラグのない旧結果のNA件数は`null`です。

## ネットワーク通信

通常のローカル解析では、同梱のgene-setおよびTF networkを使用します。既存機能のうち、
ユーザー操作でオンラインID mappingを選択した場合はmygene.infoへ遺伝子IDを、
STRING networkの取得を実行した場合はstring-db.orgへ遺伝子リストを送信します。
該当操作の前に送信先・内容種別を表示し、ユーザーによる照会をmanifestへ記録します。
実行したHTTP試行は、その後の入力検証の成否によらず記録します。
manifestの`services`は次のように区別します。

- `external_services_used`: 現在の入力に属し、成功または部分成功した照会のサービス名。
- `events`: 現在の入力に属する照会。失敗した照会も結果を明示して保持します。
- `external_service_events`: セッション全体の照会履歴。入力差し替えでは消去しません。
  `event_id`で現在の`events`と対応し、`source`に入力名・SHA-256（複数StudyではStudy名も）、
  `input_outcome`に`accepted` / `rejected` / `not_applicable`を記録します。
  全Studyの採用前に失敗した場合、その読み込みの照会はすべて`rejected`です。

各照会の`lookup_outcome`は`success` / `partial` / `unmapped` / `failed`、
`access`は`network` / `cache`です。`requests`には実際のHTTP試行ごとに日時・
HTTP status・結果・例外の型を記録します。送信IDや応答本文、例外メッセージは保存しません。
キャッシュ再利用の照会も履歴に残りますが、その照会の`requests`は空で、
過去のHTTPを新規通信として数えません。通信途中は`pending`として記録されます。
履歴はStreamlitセッション内の記録であり、アプリ終了を越える永続ログではありません。
manifestのダウンロードは従来どおりDEG結果があるときに利用できます。

## ライセンスと同梱データ

BRIMのコードおよび本リポジトリで新規作成した文書は[MIT License](LICENSE)です。
第三者由来の同梱データには、それぞれの出典・利用条件が適用され、MIT Licenseの対象外です。

特に`references/drg_mouse.csv`および`references/spinal_cord_mouse.csv`は、
Petitprez et al., *Genome Medicine* 12, 86 (2020),
[doi:10.1186/s13073-020-00783-w](https://doi.org/10.1186/s13073-020-00783-w)
に由来するGPL-3.0データです。これらのファイルを再配布または改変する際はGPL-3.0と
原著の条件に従ってください。コードと当該データは別個に管理されます。

同梱のKEGG/GO gene setとCollecTRI/DoRothEA networkについては、各ディレクトリの
READMEを参照してください。

`references/genome_annotations/`には、GENCODE Human Release 48（GRCh38.p14）と
Mouse Release M25（GRCm38.p6）から生成したgene/TSS gzip TSVを同梱します。
GENCODEデータは[open access](https://www.gencodegenes.org/pages/data_access.html)です。
release、公式download URL、元GTFと生成物のSHA-256、生成スクリプトは同ディレクトリの
`manifest.json`に固定しています。release更新は結果の再現性に影響するため、参照データと
manifestを別コミットで意図的に更新し、CHANGELOGへ記録します。

## 開発への参加

開発・テスト・変更方針は[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。
設計上の境界は[ARCHITECTURE.md](ARCHITECTURE.md)、実装順序は[PLAN.md](PLAN.md)、
科学的・技術的な不変条件は[AGENTS.md](AGENTS.md)に定義されています。
