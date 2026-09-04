# BRIM — Bulk RNA-seq Insight in Minutes

BRIMは、Streamlitで動作するBulk RNA-seq解析アプリです。カウント行列の読込み、QC、
PyDESeq2による差次的発現解析、可視化、経路濃縮解析、TF activity推定、ネットワーク表示、
デコンボリューション、出力を提供します。

現在はv1.1.0を基準線として固定しています。RNA-seqとATAC-seqの統合機能は、
`PLAN.md`に定義されたPhase順に追加します。

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

## ネットワーク通信

通常のローカル解析では、同梱のgene-setおよびTF networkを使用します。既存機能のうち、
ユーザー操作でオンラインID mappingを選択した場合はmygene.infoへ遺伝子IDを、
STRING networkの取得を実行した場合はstring-db.orgへ遺伝子リストを送信します。

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

## 開発への参加

開発・テスト・変更方針は[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。
設計上の境界は[ARCHITECTURE.md](ARCHITECTURE.md)、実装順序は[PLAN.md](PLAN.md)、
科学的・技術的な不変条件は[AGENTS.md](AGENTS.md)に定義されています。
