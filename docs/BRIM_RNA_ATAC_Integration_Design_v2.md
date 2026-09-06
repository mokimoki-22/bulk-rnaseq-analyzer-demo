# BRIM RNA-seq / ATAC-seq統合機能 設計書

- 文書版: 2.0
- 作成日: 2026-08-28
- 改訂日: 2026-09-04
- 実装基盤: `Bulk_RNAseq_Analyzer.py`（BRIM v1.1.0）
- 対象アプリ: BRIM — Bulk RNA-seq Insight in Minutes
- 想定次期バージョン: BRIM v2.0.0
- 状態: 実装前設計

## 0. 版1.0からの主な変更点

| 項目 | 版1.0 | 版2.0 | 理由 |
|---|---|---|---|
| ATAC入力 | 解析済みDAR表のみ | **カウント行列も受ける** | wet研究者が到達できる受け渡し地点はカウント行列であるため |
| TF・motif解析 | 後続リリース（Phase 6） | **初期リリースに含む** | 統合ツールとしての差別化の本体であるため |
| motif取得方法 | 未定（ローカルscan候補） | **外部結果のインポートのみ + BED出力による支援** | ゲノム配列同梱がportable配布と両立しないため |
| Integration UI | レベル選択 | **段階開示** | 初見ユーザーが選択できないため／状態遷移が単純になるため |
| タブ構成 | 9タブ | **8タブ（Multi-omics に統合）** | 横幅と既存利用者への影響 |
| RNA側padj | 既存のまま | **NA情報を保持するよう修正** | 「検定されなかった」と「有意でない」の混同を防ぐため |
| provenance | ATAC側のみ詳細 | **RNA側も同一形式に統一** | 記録の非対称を解消するため |
| サンプル数 | 記述なし | **少サンプル時の挙動と警告を明記** | 想定ユーザーの実験規模に直結するため |

---

## 1. 結論

ATAC-seq機能は別アプリとして完成させてから結合するのではなく、解析ロジックを独立モジュールとして実装し、画面は既存BRIMへ段階的に追加する。

実装順は次のとおりとする。

1. 既存RNA側のpadj NA情報を保持する修正を行う（§3.4）。
2. ATAC-seqの入力・検証・DAR推定・peak annotationを純粋関数として実装する。
3. BRIMへ `Multi-omics` タブを追加し、まずATAC単独解析を有効化する。
4. ATAC単独解析が安定した後、既存RNA-seq結果と接続するIntegrationを追加する。
5. Integration結果に対しTF推定（レベル2）を追加する。
6. motif結果のインポートとBED出力支援（レベル3）を追加する。

この構成により、既存RNA-seq機能を維持しながら、各段階を独立にテストできる。

## 2. 目的

bulk RNA-seqとbulk ATAC-seqの結果を統合し、再現可能かつ透明な形で提示する。提示する内容は次のとおり。

- differential accessible regions（DAR）の推定または読み込みと要約
- peakから遺伝子への対応付けと、その根拠の保持
- RNA発現変化とchromatin accessibility変化の一致・不一致分類
- 変化を駆動している候補転写因子の提示（3軸の根拠を分離したまま）
- 解析条件・参照データ・入力ファイル情報を含む再現性出力

本機能は因果関係を断定するものではない。ATAC-seqの変化を転写変化の原因と表示せず、「整合する」「不整合である」「対応する証拠がない」という観察レベルの表現を用いる。転写因子についても「駆動因子である」ではなく「候補である」「根拠が n 軸で支持される」という表現を用いる。

### 2.1 想定ユーザーと実験規模

主たる想定ユーザーは、自身の実験でbulk RNA-seqとbulk ATAC-seqを取得したwet-lab研究者である。想定する実験規模は**各群3〜6サンプル**である。

この想定は設計全体を規定する。

- 大規模コホートを前提とした相関ベースのpeak–gene linkingは主機能に置かない（§9.5）
- 群間比較に基づく解析を中心に据える
- サンプル数が不足する場合、機能を無効化するのではなく**警告を表示した上で実行し、結果の解釈上の制約を明示する**

## 3. 対象範囲

### 3.1 初期リリースで実装するもの

- 既存BRIMで生成したRNA-seq DEG結果の利用
- **ATAC-seq peak count matrixの読み込みとDAR推定（pydeseq2）**
- 解析済みATAC-seq DAR結果（CSV／TSV）の読み込み
- humanおよびmouseのgene annotation
- promoter overlapとnearest TSSによるpeak–gene対応
- ユーザー提供peak–gene対応表の読み込み
- peak単位・peak–gene edge単位・gene単位の結果表示
- RNA–ATAC整合性分類（レベル1）
- **TF候補の推定：標的濃縮、TF発現、TF activityの3軸表示（レベル2）**
- **motif enrichment結果のインポートと3軸への統合（レベル3）**
- **motif解析用BEDファイルの書き出しと実行コマンドの提示**
- quadrant plot、分類件数、結果テーブル、距離分布
- 分類別のKEGG／GO over-representation analysis
- CSV、JSON、Markdown、ZIPによる結果出力
- **RNA側を含む統一provenance manifest**
- サンプルデータと自動テスト

### 3.2 初期リリースでは実装しないもの

- FASTQからのread QC、adapter trimming、alignment
- BAMからのpeak calling、consensus peak setの構築
- BAMを必要とするfootprinting（TOBIAS、HINT-ATAC等）
- ゲノム配列を必要とするローカルmotif scan
- single-cell RNA-seq／single-cell ATAC-seq
- Hi-C、ChIP-seq、CUT&RUN、DNA methylationの統合
- ATAC変化からRNA変化への因果推論
- 任意のLLMによる自動解釈
- 外部文献、研究者、臨床試験、研究資金の検索

これらはBRIM v2.0の責務に含めない。

FASTQ〜consensus peak count matrixの生成は、受託解析または既存パイプライン（nf-core/atacseq等）の成果物として得られることを前提とする。BRIMはその出力を起点とする。

### 3.3 後続リリース候補

- 内蔵promoter motif表によるmotif濃縮（§12.5に検討記録を残す）
- ユーザー提供enhancer–gene link／ABC scoreの利用
- 複数コントラストのRNA–ATAC統合（既存Metaタブとの接続）
- genome buildの追加（hg19、mm39）
- サンプル数が十分な場合の相関ベースpeak–gene linking

### 3.4 実装前に必要な既存コードの修正

ATAC実装に着手する前に、既存RNA側の以下を修正する。

**修正1：padj NA情報の保持（必須・優先）**

現行の `run_deg()` はDESeq2結果のNAを次のように処理している。

```python
res["padj"] = res["padj"].fillna(1.0)
res["log2FoldChange"] = res["log2FoldChange"].fillna(0.0)
res["stat"] = res["stat"].fillna(0.0)
```

DESeq2のpadj=NAは「有意でない」ではなく、全カウントゼロ、Cooks距離による外れ値除去、またはindependent filteringによる除外を意味する。とくにindependent filteringは全遺伝子の3〜4割に及ぶことがある。

これを1.0に潰すと、統合分類において「検定した結果RNAが変化しなかった遺伝子」と「RNAを検定していない遺伝子」が区別できず、`atac_only`（クロマチンは開いたが発現は変わらない）という生物学的に注目すべき分類に、統計的に評価されていない遺伝子が混入する。

既存の下流処理を壊さないため、fillnaは維持したままフラグ列を追加する。

```python
res["padj_is_na"] = res["padj"].isna()
res["lfc_is_na"] = res["log2FoldChange"].isna()
res["padj"] = res["padj"].fillna(1.0)
res["log2FoldChange"] = res["log2FoldChange"].fillna(0.0)
res["stat"] = res["stat"].fillna(0.0)
```

統合分類ではこのフラグを参照し、`rna_not_tested` を独立した分類として扱う（§10.2）。

この修正は設計書§4.3「情報を失わない」および§4.4「入力を暗黙に修正しない」の原則を既存RNA側にも適用するものである。

**修正2：外部通信の明示（§16.3）**
**修正3：provenance生成の共通化（§16.1）**

## 4. 基本設計原則

### 4.1 既存BRIMを基盤にする

以下の既存機能を再利用する。

- species選択（`SPECIES_MAP`：human／mouse）
- DEG結果とcontrast情報
- **pydeseq2によるdifferential analysis（ATAC DAR推定にも使用）**
- KEGG／GO enrichment（`brim_enrichment.py`）
- **TF activity（`brim_tf_networks.py`：CollecTRI／DoRothEA）**
- Plotly／Matplotlibの描画設定
- 多言語表示（`i18n`）
- analysis log
- ZIP exportとreproducibility report
- Streamlit AppTestを用いた回帰テスト
- `_DATA_RESULT_DEFAULTS` 辞書によるsession stateリセット

### 4.2 UIと解析ロジックを分離する

`Bulk_RNAseq_Analyzer.py` には画面制御だけを追加し、ATACおよび統合計算はStreamlitへ依存しないモジュールに置く。既存の `brim_enrichment.py` および `brim_tf_networks.py` と同じパターンを踏襲する。

解析関数の中で次を行わない。

- `st.session_state` への直接アクセス
- `st.error`、`st.warning`、`st.stop` の呼び出し
- ファイルダウンロードUIの生成
- 暗黙の閾値取得

解析関数はDataFrameと明示的な設定値を受け取り、DataFrameまたは型付き結果を返す。

### 4.3 情報を失わない

- 一つのpeakが複数遺伝子に対応する場合、一対多のedgeを保持する。
- 一つの遺伝子に複数peakが対応する場合、元peakを保持する。
- 最も大きいlog2FCだけへ暗黙に集約しない。
- p値を根拠なく合成しない。
- **検定されなかったことと有意でないことを区別する（§3.4）。**
- **TF候補の3軸を単一スコアへ統合しない。**
- gene-level summaryは二次的な要約とし、元edgeへ戻れるようにする。

### 4.4 入力を暗黙に修正しない

- 無効な座標、非数値、無限値はエラーにする。
- 重複peak、欠損値、未知染色体は件数と対象を表示する。
- 染色体名の `chr1` と `1` の変換は、変換内容を記録した上で行う。
- gene ID変換は対応率を表示する。
- genome buildは推測せず、ユーザーの明示選択を必須とする。
- **正規化方式は推測せず、明示選択とする（§7.2）。**

### 4.5 制約を隠さない

機能が対象としない範囲を、結果画面上に明示する。ドキュメントのみに記載して画面では黙る、という形にしない。

- サンプル数が推奨を下回る場合、結果とともに警告を表示する
- motif結果が外部由来である場合、その出典と解析条件を結果に併記する
- peak–gene mappingでunmappedとなったpeakの件数と割合を常に表示する

## 5. 対象ユーザーフロー

```text
RNA-seq raw count                  ATAC-seq peak count matrix
    ↓ 既存BRIM DEG                     ↓ validation
RNA DEG result                         ↓ 正規化方式の選択
（padj_is_naフラグを保持）              ↓ pydeseq2によるDAR推定
    │                                  │
    │                          または  │
    │                          解析済みDAR表の読み込み
    │                                  ↓
    │                          標準DARテーブル
    │                                  ↓ peak annotation
    │                          peak–gene edges
    │                                  │
    └──────────────┬───────────────────┘
                   ↓
        【レベル1】RNA–ATAC比較
        concordant / discordant / single-modality / not_tested
                   ↓
        【レベル2】TF候補推定（追加ファイル不要）
        標的濃縮 × TF発現 × TF activity
                   ↓
        【レベル3】motif濃縮の追加
        BED出力 → 外部ツール → 結果インポート
                   ↓
        plots / tables / enrichment / export
```

ATAC単独解析はRNA結果がなくても実行できる。Integrationは、有効なRNA DEG結果とATAC annotationの両方が存在する場合のみ有効化する。

レベル2および3は、レベル1の結果が存在する場合のみ追加できる。

## 6. UI設計

### 6.1 トップレベルタブ

既存タブへ `Multi-omics` を1つ追加する。

```text
Upload | DEG | Multi-omics | Visualization | Network | Meta | Export | Info
```

版1.0では `ATAC-seq` と `Integration` を独立タブとしていたが、次の理由により1タブへ統合する。

- 既存7タブに2つ追加すると横幅を超えやすい（本アプリはタブラベルを18pxに設定しているため顕著）
- RNAのみを使う既存利用者に、常時2つの未使用タブが見える状態になる

`Multi-omics` タブ内はサブタブで構成する。既存コードでもDEGタブ・Networkタブ内で `st.tabs` の入れ子を使用しており、パターンとして確立している。

```text
Multi-omics
  ├ ATAC-seq          （常時表示）
  └ Integration       （ATAC結果とRNA DEG結果が揃った場合のみ表示）
```

### 6.2 ATAC-seqサブタブ

```text
1. Input
   - 入力モードの選択
       ○ Peak count matrix（BRIMがDAR推定を行う）
       ○ 解析済みDAR table（既に差次的解析を終えている場合）
   - species / genome build
   - test condition / reference condition
   - column mapping（DAR tableモード時）
   - 正規化方式（count matrixモード時）
2. Validation
   - 行数、有効peak数、significant DAR数
   - chromosome一覧
   - 重複・欠損・無効座標
   - log2FCの向き
   - サンプル数と警告（count matrixモード時）
3. Peak annotation
   - mapping method
   - promoter window
   - nearest TSSの最大距離
   - user-provided mapping
4. Results
   - opening／closing DAR
   - annotation type
   - TSS距離分布
   - peak–gene edge table
   - unmapped peak table
```

### 6.3 Integrationサブタブ（段階開示）

版1.0ではタブ内を固定構成としていたが、解析の深さを段階的に開示する形へ変更する。

```text
1. Compatibility check
   - species一致
   - genome build明示
   - contrast方向一致
   - gene ID対応率
2. Thresholds
   - RNA padj / RNA absolute log2FC
   - ATAC padj / ATAC absolute log2FC
3. 【レベル1】RNA–ATAC比較   ［実行ボタン］
   - 分類件数、mapping coverage
   - quadrant plot
   - evidence table（edge単位／gene単位）
   - 分類別KEGG／GO
   ────────────────────────────────
4. 【レベル2】TF候補の推定    ［追加ボタン・追加ファイル不要］
   - 3軸テーブル（motif列は未実行として空欄表示）
   - 分類ごとの切り替え
   - TF選択による標的遺伝子・peakの表示
   ────────────────────────────────
5. 【レベル3】motif濃縮の追加  ［追加ボタン］
   - BED書き出しと実行コマンドの提示
   - 外部結果のインポート
   - 3軸テーブルのmotif列が埋まる
   ────────────────────────────────
6. Export
   - 統合結果とprovenance
```

段階開示を採用する理由。

- 初見のユーザーは、レベル2/3が何を与えるか事前には判断できない。レベル1の結果を見た後に選択肢を提示するほうが判断できる。
- 状態遷移が「未実行／実行済み」の2値になり、レベル間遷移時のinvalidation規則が不要になる。
- 途中で止めたユーザーに不要な計算が走らない。
- 未実装フェーズにおいて、該当ボタンを表示しないだけで段階リリースが成立する。

ただし段階開示は後続機能の存在に気づかれにくいため、レベル1実行前にタブ冒頭で3段階の存在を1行で示す。

### 6.4 レベル2のテーブル表示

レベル2の中心は次のテーブルである。motif列は最初から表示し、未実行であることを明示する。

```text
TF        motif濃縮   標的濃縮      TF発現        TF activity
Stat3     （未実行）   padj 3e-08   log2FC 1.4↑   z = 4.2↑
Fosl1     （未実行）   padj 2e-05   log2FC 2.2↑   z = 3.1↑
Myc       （未実行）   padj 1e-04   変化なし       z = 2.8↑
```

motif列を空のまま提示することには2つの意図がある。

- レベル3で何が加わるかを、説明ではなくテーブルの形で示す
- レベル2単独では3軸のうち2軸がRNA由来であり、ATACの寄与が入力集合の絞り込みに限られることを、構造として可視化する

### 6.5 Contrast方向の確認

RNAとATACの符号を比較する前に、両方が同じcontrast方向であることを確認する。

```text
RNA:  Drug vs Control
ATAC: Drug vs Control    → 統合可能

RNA:  Drug vs Control
ATAC: Control vs Drug    → ブロックし、反転または再入力を要求
```

ATACのlog2FC反転を許可する場合は、ユーザーが明示的に実行し、provenanceへ `atac_log2fc_inverted: true` を記録する。

count matrixモードでは、BRIMがcontrastを構成するため方向の不一致は原理的に発生しない。DAR tableモードでのみこの確認を行う。

## 7. 入力仕様

### 7.1 入力モードの併存

ATAC入力は2モードを受け付ける。両者は同一の標準DARテーブル（§8.1）へ合流する。

```text
peak count matrix ──→ [BRIMがDAR推定] ─┐
                                        ├→ 標準DARテーブル → annotation → 統合
解析済みDAR table ──→ [読み込み・検証] ─┘
```

版1.0はDAR tableモードのみを想定していたが、次の理由でcount matrixモードを初期リリースに含める。

- 受託解析やnf-core等の標準パイプラインでは、consensus peakのcount matrixまでが成果物として提供されることが多い。どの群を比較するかは研究者が決めるため、DAR表は納品されない。したがってcount matrixが研究者に渡る自然な受け渡し地点である。
- BRIMのRNA側はraw countを受け付ける。ATAC側のみ解析済み結果を要求すると、同一アプリ内で入力水準が非対称になる。
- 既存の `run_deg()` はpydeseq2にcount matrixとmetadataを渡す構成であり、行が遺伝子かpeakかの違いを除けば同一の処理が適用できる。統計エンジンを新規に実装する必要がない。

### 7.2 Peak count matrixモード

**入力形式**

行がpeak、列がサンプルの整数カウント行列。peak座標は次のいずれかで与える。

- index に `chr1:100200-100700` 形式の文字列
- 先頭3列が `chrom`, `start`, `end`

metadata（サンプルと群の対応）は、RNA側と同じ入力UIを再利用する。

**正規化方式の選択（必須）**

ATAC-seqではリードの相当割合がpeak外に分布し、その割合はサンプル間で変動する。RNA-seqのように「リードの大半が定量対象である」という前提が成り立たないため、peak内リードのみを用いた正規化が妥当かは自明でない。

したがって正規化方式は既定値を暗黙適用せず、明示選択とする。

| 選択肢 | 内容 |
|---|---|
| DESeq2 median-of-ratios | pydeseq2の既定。peak内カウントに基づく |
| Total reads in peaks | peak内総リード数によるスケーリング |
| User-supplied size factors | ユーザーがサイズファクターを与える |

選択した方式はprovenanceへ記録する。既定の選択肢を用意することは妨げないが、選択したことがログに残る形とする。

既定候補として最初に提示する方式は **DESeq2 median-of-ratios** とする。ただし、これは
暗黙の既定値として解析へ適用せず、ユーザーが明示的に選択してから実行する。
globalなaccessibility変化がある場合はpeakの大半が不変という前提が崩れ、
median-of-ratiosが偏りうる旨を、Phase 2 UIの選択肢の近くに1行で表示する。
選択値はprovenanceへ記録する。

**事前フィルタ**

count matrixの事前フィルタは既定で無効とする。UIではチェックボックスを既定オフにし、
有効化した場合だけ閾値を編集可能にする。初期閾値は「全サンプル合計count < 10を除外」。
適用の有無、閾値、入力peak数、除外peak数、解析対象peak数を画面とprovenanceに記録する。

**サンプル数の扱い**

各群のサンプル数を表示し、次の基準で警告する。機能は無効化しない。

| 各群のn | 表示 |
|---|---|
| n >= 4 | 警告なし |
| n = 3 | 検出力が限られる旨を表示 |
| n = 2 | 分散推定が不安定である旨を強く表示。結果は探索的である旨を明記 |
| n = 1 | DAR推定は実行しない（pydeseq2が分散を推定できないため） |

ATAC-seqはRNA-seqより群内変動が大きく、同じサンプル数でも検出力が低い傾向がある。この点を警告文に含める。

**出力**

pydeseq2の結果を標準DARテーブルへ変換する。RNA側と同様、padjのNAはフラグ列 `padj_is_na` として保持した上でfillnaする（§3.4と同一方針）。

### 7.3 DAR tableモード

**必須列**

| 標準列 | 内容 | 必須 | 制約 |
|---|---|---:|---|
| `chrom` | 染色体 | Yes | 空欄不可 |
| `start` | 0-based開始座標 | Yes | 0以上の整数 |
| `end` | 終了座標 | Yes | `end > start` |
| `log2FoldChange` | accessibility変化 | Yes | 有限数 |
| `padj` | 多重検定補正p値 | Yes | 0〜1、またはNA |

padjのNAは許容し、`padj_is_na` として保持する。DiffBindやDESeq2の出力にはNAが含まれうるため、NA行を無条件に除外しない。

初期版ではBEDの0-based, half-open座標を標準とする。1-based入力は自動推測せず、入力設定で明示的に変換する。

**任意列**

| 標準列 | 内容 |
|---|---|
| `peak_id` | peak識別子。未指定時は座標から生成 |
| `pvalue` | 未補正p値 |
| `baseMean` | 平均accessibility等 |
| `stat` | 統計量 |
| `gene` | 既存annotationがある場合の遺伝子 |
| `annotation` | promoter、intronic等 |

DiffBind、DESeq2ベースDAR、edgeRベースDAR等の代表的な列名をaliasとして認識する。ただし、最終的なcolumn mappingを画面で表示して確認させる。

### 7.4 RNA入力

初期版では現在のBRIMセッションにある `deg_results` を利用する。最低限必要な列は以下である。

- gene IDまたはgene symbol（index可）
- `log2FoldChange`
- `padj`
- `padj_is_na`（§3.4の修正により追加）

将来は、解析済みDEG表を直接読み込むモードを追加できる設計にする。その場合 `padj_is_na` 列がない入力を受けることになるため、欠如時は全てFalseとして扱い、その旨をprovenanceへ記録する。

### 7.5 ユーザー提供peak–gene map

任意入力として以下を受け付ける。

| 列 | 必須 | 内容 |
|---|---:|---|
| `peak_id` または座標3列 | Yes | 入力DARとの対応 |
| `gene` | Yes | 対応遺伝子 |
| `evidence_type` | No | ABC、Cicero、curated、custom等 |
| `score` | No | 元手法が出力したscore |
| `source` | No | ファイル名、DB、解析法 |

外部scoreをBRIM独自scoreへ無理に変換しない。元の値と意味を保持する。

### 7.6 motif enrichment結果（レベル3）

| 列 | 必須 | 内容 |
|---|---:|---|
| `tf_symbol` | Yes | 転写因子シンボル |
| `motif_id` | No | JASPAR ID等 |
| `pvalue` | No | 未補正p値 |
| `padj` | No | 補正p値 |
| `enrichment_score` | No | 濃縮の強さ |
| `direction_or_peak_set` | Yes | `opening` / `closing` / ユーザー定義のpeak集合名 |

HOMERの `knownResults.txt` を第一の対応形式とし、それ以外は汎用CSVとしてcolumn mappingで受ける。§7.3のalias認識と同じパターンを用いる。

インポート時に併せて次を入力させ、provenanceへ記録する。BRIMは外部ツールの実行条件を検証できないため、記録によって担保する。

- 解析に用いたATAC padj / log2FC閾値
- 背景（background）の設定
- 使用ツール名とバージョン
- 使用したmotif database

入力された閾値が現在のBRIMのATAC閾値と一致しない場合、警告を表示する。処理は継続する。

## 8. データモデル

### 8.1 標準DARテーブル

```text
peak_id
chrom
start
end
log2FoldChange
pvalue
padj
padj_is_na               # 検定されなかったことを示す
lfc_is_na                # log2FoldChangeが推定されなかったことを示す
is_significant
accessibility_direction  # opening / closing / not_significant / not_tested
input_row
source_mode              # count_matrix / dar_table
```

### 8.2 Peak–gene edgeテーブル

```text
edge_id
peak_id
chrom
start
end
gene_id
gene_symbol
mapping_method            # promoter / nearest_tss / user_supplied
mapping_evidence_type
distance_to_tss
gene_strand
reference_build
reference_release
atac_log2FoldChange
atac_padj
atac_padj_is_na
atac_lfc_is_na
```

### 8.3 統合edgeテーブル

```text
edge_id
peak_id
gene_symbol
mapping_method
distance_to_tss
rna_log2FoldChange
rna_padj
rna_padj_is_na
atac_log2FoldChange
atac_padj
atac_padj_is_na
rna_significant
atac_significant
integration_class
```

### 8.4 Gene summaryテーブル

```text
gene_symbol
rna_log2FoldChange
rna_padj
rna_padj_is_na
n_mapped_peaks
n_opening_peaks
n_closing_peaks
n_significant_peaks
accessibility_pattern     # opening / closing / mixed / no_significant_peak
representative_peak_id
representative_peak_rule
integration_class
```

`representative_peak_rule` を必須列とし、代表peakがどの規則で選ばれたかを明示する。初期値は `smallest_atac_padj_then_largest_abs_lfc` とするが、元peak一覧を常に保持する。

### 8.5 TF候補テーブル（レベル2／3）

```text
tf_symbol
peak_set                      # 対象とした分類または方向
target_enrichment_padj        # CollecTRI標的の濃縮（レベル2）
target_enrichment_source      # collectri / dorothea
n_targets_in_set
tf_rna_log2FoldChange         # TF自身の発現（レベル2）
tf_rna_padj
tf_activity_score             # 既存TF activity推定（レベル2）
tf_activity_source
motif_enrichment_padj         # motif濃縮（レベル3、未実行時はNA）
motif_enrichment_score
motif_source                  # NA / imported
motif_source_tool
n_axes_supported              # 支持された軸の数（表示補助。統合スコアではない）
```

`n_axes_supported` は表示上のソート補助であり、統計量ではない。単一の「TF score」として提示しない。

## 9. Peak–gene mapping

### 9.1 Mappingの優先順位

一つの方法へ強制的に絞らず、mapping edgeを併存させる。

1. user-provided link
2. promoter overlap
3. nearest TSS within maximum distance
4. unmapped

同一peak–geneが複数方法で支持された場合は一行へ統合し、`mapping_methods` へ複数の根拠を記録する。

### 9.2 Promoter定義

初期値はTSSに対してstrand-awareで以下とする。

```text
upstream:   2,000 bp
downstream:   500 bp
```

値はユーザーが変更可能とし、出力へ保存する。

### 9.3 Distal mapping

nearest TSSは距離に基づく候補であり、機能的enhancer–gene関係とは表示しない。

- 初期最大距離: 100 kb
- 同距離の遺伝子: すべて保持
- 最大距離外: unmapped
- ラベル: `distance_based_candidate`

初期値100 kbの根拠を明記する。大規模コホートを用いた相関ベースのpeak–gene linking（TCGA ATAC atlas等）では500 kbが用いられるが、これは多数のサンプルにわたる相関により偽陽性を除去できることを前提とする。本アプリは少サンプルを想定し相関フィルタを前提としないため、距離のみで候補を広げると偽陽性が増加する。より保守的な100 kbを初期値とし、ユーザーが変更可能とする。

この設定値と、それが距離に基づく候補にすぎないことは、結果画面とprovenanceの双方に記録する。

### 9.4 Reference annotation

初期対応buildを次とする。

- Human: hg38 / GRCh38 — GENCODE Release 48（GRCh38.p14）
- Mouse: mm10 / GRCm38 — GENCODE Release M25（GRCm38.p6）

将来追加候補: hg19 / GRCh37、mm39 / GRCm39

参照ファイルには必ず以下を付与する。

- source
- release
- download URL
- license／terms
- build
- SHA-256 checksum
- generation script

releaseは再現性のため上記に固定し、実装時点の最新版へ自動追従しない。
release更新は意図的な判断として参照データ・manifestを別コミットで更新し、
CHANGELOGへ記録する。source pageはHuman Release 48を
`https://www.gencodegenes.org/human/release_48.html`、Mouse Release M25を
`https://www.gencodegenes.org/mouse/release_M25.html`とする。

BRIM本体へは完全なGTFではなく、必要列に限定したgzip圧縮TSVのgene/TSS tableを同梱する。
Parquetは追加エンジンを必要とするためPhase 1では採用せず、Windows portable環境で
標準ライブラリとpandasだけで読める形式を優先する。保存形式は`load_gene_annotation()`の
内部へ隠蔽し、将来の形式変更で利用側APIを変えない。

既存BRIMは `references/*.csv` をglobで自動走査して `_EXTERNAL_REFS` へ読み込む方式を採っている。genome annotationはサイズと構造が異なるため `references/genome_annotations/` 以下に圧縮TSVとmanifestで配置し、既存のglob対象と分離する。両者を1つの機構へ統合しないことを意図的な設計判断として記録する。

### 9.5 相関ベースlinkingを主機能に置かない理由

サンプル横断の発現–accessibility相関によるpeak–gene linkingは、既存ツール（Linkage等）が採用している方式であり、有効な手法である。ただし信頼できる相関の推定には概ね10サンプル以上を要する。

本アプリは各群3〜6サンプルを想定するため、この方式は主機能に置かない。サンプル数が十分な場合の追加機能として後続リリースの候補とする（§3.3）。

## 10. RNA–ATAC分類規則（レベル1）

### 10.1 Significance

RNAとATACは別々の閾値を使う。

```text
RNA significant:
    rna_padj_is_na == False
    and rna_padj <= RNA_PADJ_THRESHOLD
    and abs(rna_log2FoldChange) >= RNA_LFC_THRESHOLD

ATAC significant:
    atac_padj_is_na == False
    and atac_padj <= ATAC_PADJ_THRESHOLD
    and abs(atac_log2FoldChange) >= ATAC_LFC_THRESHOLD
```

初期値: RNA padj 0.05 / RNA |log2FC| 1.0 / ATAC padj 0.05 / ATAC |log2FC| 1.0

### 10.2 Edge-level classification

| 条件 | `integration_class` | 表示上の意味 |
|---|---|---|
| RNA up、ATAC opening | `concordant_activation` | 発現上昇とaccessibility上昇が整合 |
| RNA down、ATAC closing | `concordant_repression` | 発現低下とaccessibility低下が整合 |
| RNA down、ATAC opening | `discordant_open_down` | 方向が不一致 |
| RNA up、ATAC closing | `discordant_closed_up` | 方向が不一致 |
| RNA非有意、ATAC有意 | `atac_only` | ATACのみ閾値を満たす |
| RNA有意、ATAC非有意 | `rna_only_on_mapped_peak` | RNAのみ閾値を満たす |
| **RNAが検定されていない** | **`rna_not_tested`** | **RNA側がindependent filtering等で評価されていない** |
| **ATACが検定されていない** | **`atac_not_tested`** | **ATAC側が評価されていない** |
| 両方非有意 | `not_significant` | 統合上の有意証拠なし |

peakが対応しない有意RNA遺伝子は、gene-levelで `rna_only_no_mapped_peak` とする。

`rna_not_tested` および `atac_not_tested` は版2.0で新設した分類である。版1.0の規則では、これらが `atac_only` または `not_significant` に混入していた。とくに `atac_only` は「クロマチンは開いたが発現は変化しない」という解釈上注目される群であるため、統計的に評価されていない遺伝子の混入は解釈を歪める。

### 10.3 Gene-level aggregation

同一遺伝子にopeningとclosingの有意peakが混在する場合、`mixed_accessibility` とする。多数決で一方向へ自動決定しない。

gene-level分類は以下の優先規則を使う。

1. RNAが検定されていない場合は `rna_not_tested` とする。
2. 有意peakがない場合はRNA-onlyまたはnot significant。
3. 有意peakが一方向のみなら、その方向を採用する。
4. openingとclosingが混在する場合はmixedとする。
5. mixedの場合はconcordant／discordantへ単純化しない。

### 10.4 相関係数

RNA log2FCとATAC log2FCの相関は探索的指標としてのみ表示する。

- peak–gene edge単位とgene summary単位を区別する。
- 一対多edgeによる擬似反復を明記する。
- p値を主要な統合判断に使わない。
- サンプル数、使用edge数、集約規則を併記する。

### 10.5 discordant群の扱い

`discordant_open_down` および `discordant_closed_up` は件数が少ない一方、転写抑制因子の関与や転写後制御の可能性を示唆する探索的価値がある。結果画面ではconcordant群と同等の可視性を与え、一覧の下部に埋もれさせない。

ただしdiscordantは技術的要因（mapping誤り、細胞組成変化、時点のずれ）によっても生じるため、生物学的解釈を断定する表示は行わない。

## 11. 可視化

### 11.1 ATAC単独

- DAR volcano plot
- chromosome別DAR件数
- opening／closing件数
- peak–TSS distance histogram
- annotation method bar chart
- mapping coverage
- **検定されなかったpeakの件数**

### 11.2 RNA–ATAC統合（レベル1）

- RNA log2FC × ATAC log2FC quadrant plot
- integration class bar chart
- class別のgene table
- 複数peakを持つgeneのaccessibility pattern table
- concordant geneのheatmap
- class別KEGG／GO dot plot

### 11.3 TF候補（レベル2／3）

- 3軸テーブル（§6.4）
- TFごとの根拠バッジ表示
- 選択TFの標的遺伝子とpeakのdrill-down
- 分類ごとの切り替え（concordant / discordant / opening全体 等）

全図で閾値、n数、解析単位をcaptionへ表示する。

## 12. TF候補の推定

### 12.1 3軸の定義

| 軸 | データ源 | 判定内容 | レベル |
|---|---|---|---|
| motif濃縮 | ATAC + 外部motif解析 | 変化したpeak内にそのTFの結合配列が濃縮するか | 3 |
| 標的濃縮 | RNA + CollecTRI/DoRothEA | そのTFの既知標的が該当分類に偏るか | 2 |
| TF発現 | RNA | TF遺伝子自体の発現が変化したか | 2 |
| TF activity | RNA + CollecTRI/DoRothEA | 既存のTF activity推定が支持するか | 2 |

（表示上は「標的濃縮」「TF発現」「TF activity」をまとめてレベル2の3列とし、motifを加えて計4列とする。）

### 12.2 レベル2の実装

新規の統計計算は標的濃縮のみである。TF発現は `deg_results` から、TF activityは既存の `tf_collectri` / `tf_dorothea` から取得する。

**標的濃縮の計算**

レベル1で分類した遺伝子集合（例: `concordant_activation`）に対し、各TFの既知標的が有意に偏るかをFisher正確検定で評価する。

**背景（universe）の設定**

背景には**peak–geneでmapされ、かつRNA側で検定された遺伝子**を用いる。全遺伝子を背景にしてはならない。mapされていない遺伝子は原理的にconcordantになりえず、`rna_not_tested` の遺伝子も分類対象外であるため、これらを背景に含めると濃縮が過大評価される。

背景の定義と件数を結果画面へ表示する。

**多重検定補正**

CollecTRIは数百のTFを含むため、BH法で補正する。このpadjはレベル1の分類を入力とする新規の検定であり、RNAおよびATACのpadjとは独立である。結果画面とドキュメントの双方でこの点を明示し、レベル1のpadjと混同させない。

**遺伝子集合が小さい場合**

対象分類の遺伝子数が20未満の場合、検定は実行するが結果を探索的として表示し、警告を付す。少サンプル実験ではdiscordant群が数十遺伝子にとどまることが多いため、この状況は例外ではなく常態として扱う。

### 12.3 レベル2の限界の明示

レベル2の3軸はいずれもRNA-seqとキュレーション済みデータベースに由来し、ATACの寄与は入力遺伝子集合の絞り込みに限られる。この事実を結果画面に明示する。

とくに次の2点を記載する。

- 核移行や翻訳後修飾で活性化する転写因子（NF-κB、STAT、SMAD、HIF-1α等）は、mRNA発現も既知標的の応答も弱く出ることがあり、レベル2では検出されない場合がある。
- CollecTRI等のキュレーションデータベースは研究の蓄積量に依存するため、報告の少ない転写因子は構造的に検出されにくい。

これらはレベル3で部分的に補われる。

### 12.4 レベル3：motif濃縮

**取得方法はインポートのみとする**

ローカルでのmotif scanは初期リリースに含めない。peak配列の取得にはゲノムFASTA（ヒト・マウスで各約3 GB）が必要であり、同梱・初回ダウンロード・ユーザー指定のいずれもportable配布またはwet-lab利用者の到達性と両立しない。

**BED出力とコマンド提示による支援**

インポートのみとする代わりに、外部ツールの実行を最大限支援する。レベル3の追加ボタンを押した時点で、次を提供する。

出力ファイル:

```text
opened_peaks_padj<th>_lfc<th>.bed     # opening方向の有意DAR
closed_peaks_padj<th>_lfc<th>.bed     # closing方向の有意DAR
all_peaks_background.bed              # 全peak（背景用）
motif_analysis_README.txt             # 閾値、件数、genome build、生成日時
```

提示するコマンド例（genome buildはBRIMの設定から自動で埋める）:

```text
# HOMERにゲノムが未導入の場合
perl configureHomer.pl -install mm10

# opening / closing を別々に解析する
findMotifsGenome.pl opened_peaks.bed mm10 out_open/ -size 200 -bg all_peaks_background.bed
findMotifsGenome.pl closed_peaks.bed mm10 out_close/ -size 200 -bg all_peaks_background.bed
```

この設計は、外部ツール利用時の典型的な失敗要因のうち次を解消する。

- どのpeak集合を渡すべきかが不明である
- opening と closing を分離せず、濃縮が相殺される
- 背景の設定が不適切である（ATACでは全peakを背景とするのが妥当な場合が多い）
- 閾値の記録が残らず、後から解析条件を再構成できない

解消できないのは外部ツールのインストール環境の問題のみである。この点はドキュメントに明記する。

**背景に全peakを用いる理由**

HOMERの既定背景はGC含量を揃えたランダムゲノム配列である。ATAC-seqではpeak領域自体がゲノム全体と異なる配列組成を持つため、全peakを背景とすることで「開いている領域の中で、変化した領域に特異的なmotif」を検出できる。既定背景では、開いた領域一般に共通するmotifが一律に濃縮する。

**インポートと統合**

`tf_symbol` をキーにレベル2のテーブルへ結合する。HOMERの出力は `Stat3(Stat)/mES-Stat3-ChIP-Seq/Homer` のような複合名を含むため、シンボル抽出の正規化を行う。照合できなかったmotifの件数と一覧を表示する（§4.4のgene ID対応率と同じ方針）。

opening と closing は別々のpeak集合として保持し、`peak_set` 列で区別する。1ファイルに混在する場合は分割を要求する。

### 12.5 内蔵promoter motif表を採用しない判断（記録）

検討したが初期リリースには含めない。判断の根拠を将来の再検討のために記録する。

想定した構成: 参照ゲノムのプロモーター領域（TSS±2 kb）をJASPAR PWMで事前スキャンし、「遺伝子 × TF」の対応表（数十MB）を同梱する。ゲノムFASTAは表の生成時にのみ必要で、配布物には含まれない。

採用しなかった理由:

- 対象がプロモーターに限られる。有意DARに占めるプロモーターの割合は一般に2割前後であり、遠位領域を扱えない。ATACの情報の多くは遠位エンハンサーにある。
- 解析単位が遺伝子に丸められる。実際に開いたpeakがプロモーターでなくとも、その遺伝子のプロモーターmotifを計上することになる。
- CollecTRI等のキュレーション情報もプロモーター近傍の実験に由来するものが多く、レベル2の標的濃縮との独立性が想定より低い可能性がある。独立した第4の軸として機能しない懸念がある。
- 表の生成・版管理・閾値選択の妥当性検証というコストが継続的に発生する。

再検討の条件: レベル3-インポートの利用実績が乏しく、外部ツールの実行が到達性の障壁になっていることがユーザーからのフィードバックで確認された場合。その際は「プロモーター限定である」旨を結果画面に常時表示することを必須とする。

## 13. モジュール構成

既存ファイルを維持し、以下を追加する。

```text
brim-app/
├─ Bulk_RNAseq_Analyzer.py       # Streamlit UI、既存RNA機能
├─ brim_enrichment.py            # 既存
├─ brim_tf_networks.py           # 既存
├─ brim_atac.py                  # ATAC入力、検証、DAR推定、annotation
├─ brim_multiomics.py            # RNA–ATAC統合、分類、要約
├─ brim_tf_integration.py        # TF候補推定、motif結果の統合
├─ brim_provenance.py            # manifest、checksum、設定保存（RNA側も使用）
├─ references/
│  ├─ *.csv                      # 既存の自動走査対象
│  └─ genome_annotations/
│     ├─ hg38_genes.parquet
│     ├─ mm10_genes.parquet
│     └─ manifest.json
├─ sample_data/
│  └─ multiomics/
│     ├─ sample_rna_counts.csv
│     ├─ sample_atac_counts.csv      # count matrixモード用
│     ├─ sample_atac_dar.csv         # DAR tableモード用
│     ├─ sample_peak_gene_map.csv
│     ├─ sample_motif_results.txt    # レベル3インポート用
│     └─ expected_summary.json
├─ test_atac.py
├─ test_multiomics.py
├─ test_tf_integration.py
├─ test_provenance.py
├─ test_multiomics_ui.py
└─ docs/
   ├─ multiomics_input_format.md
   └─ motif_analysis_guide.md      # 外部ツール実行手順
```

初期実装では過度なpackage化を避けるが、`brim_atac.py`、`brim_multiomics.py`、`brim_tf_integration.py`、`brim_provenance.py` はStreamlit非依存にする。機能が安定した後に `brim_core/` packageへ移行できるよう、公開関数を限定する。

## 14. 公開関数案

### 14.1 `brim_atac.py`

```python
read_peak_count_matrix(file_obj, sep, coordinate_column_mode, coordinate_system) -> pd.DataFrame
parse_peak_coordinates(index_or_columns) -> pd.DataFrame
run_dar(counts_df, metadata, ref_condition, test_condition,
        normalization, n_cpus, padj_threshold, lfc_threshold,
        prefilter_enabled=False, prefilter_total_count=10,
        size_factors=None) -> pd.DataFrame
read_dar_table(file_obj, sep, padj_threshold, lfc_threshold,
               coordinate_system, column_map=None) -> pd.DataFrame
validate_dar_table(df, coordinate_system, padj_threshold,
                   lfc_threshold) -> ValidationResult
standardize_chromosomes(df, build) -> tuple[pd.DataFrame, TransformLog]
load_gene_annotation(build) -> pd.DataFrame
map_peaks_to_promoters(peaks, genes, upstream, downstream) -> pd.DataFrame
map_peaks_to_nearest_tss(peaks, genes, max_distance) -> pd.DataFrame
merge_peak_gene_evidence(*edge_tables) -> pd.DataFrame
summarize_atac_qc(peaks, edges, settings) -> dict
export_peaks_as_bed(dar_df, direction, thresholds) -> str
```

### 14.2 `brim_multiomics.py`

```python
standardize_rna_results(deg_results, gene_id_type) -> pd.DataFrame
check_integration_compatibility(rna_meta, atac_meta) -> CompatibilityResult
integrate_peak_gene_edges(rna, edges, thresholds) -> pd.DataFrame
classify_integration_edges(integrated, thresholds) -> pd.DataFrame
summarize_integration_by_gene(integrated, thresholds) -> pd.DataFrame
extract_gene_set(summary, integration_class) -> list[str]
build_integration_summary(edges, genes, settings) -> dict
```

### 14.3 `brim_tf_integration.py`

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

### 14.4 `brim_provenance.py`

```python
file_checksum(file_obj) -> str
collect_environment() -> dict          # Python/package versions, platform
build_manifest(inputs, settings, counts, services) -> dict
render_manifest_markdown(manifest) -> str
```

RNA単独解析時もこのモジュールを経由してmanifestを生成する（§16.1）。

### 14.5 例外

UIで解釈できる専用例外を定義する。

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

例外メッセージは英語の安定したerror codeと、人向け詳細を持たせ、多言語UI側で翻訳できるようにする。

## 15. Session state

追加候補:

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

段階開示の採用により、レベルの状態は各キーがNoneかどうかの2値で表現される。レベル間の遷移という概念を持たない。

### 15.1 Invalidation規則

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

レベル2・3の結果は常に上流の変更で破棄する。上流が変わったまま古いTF結果が残る状態を許さない。

既存の `reset_data_results()` にATAC状態を直接混在させず、以下を追加する。

```python
reset_atac_results()
reset_integration_results()
reset_peak_mapping_results()
reset_tf_integration_results()
```

RNAデータ差し替え時は `reset_integration_results()` と `reset_tf_integration_results()` を呼ぶが、独立して読み込まれたATAC原表は保持可能とする。ただしspecies／contrast不一致時はIntegrationを無効化する。

## 16. Exportと再現性

### 16.1 provenanceの統一

版1.0ではATAC側にのみ詳細なmanifestを定義していたが、これは既存RNA側の `reproducibility_report.json`（timestamp、app_version、species、DEG閾値、analysis_logのみ）と非対称であり、同一ZIP内に詳細度の異なる2種類の記録が併存することになる。

`brim_provenance.py` を汎用のmanifest生成モジュールとして実装し、**RNA単独解析時も同モジュールを経由させる**。既存の `reproducibility_report.json` はその出力へ置き換える。

manifestに含める項目:

- BRIM version
- Pythonと主要package version（pydeseq2、pandas、numpy、scipy等）
- timestampとtimezone
- speciesとgenome build
- reference annotation release/checksum
- 入力ファイル名とSHA-256（**RNA count matrixを含む**）
- 入力モード（count_matrix / dar_table）
- column mapping
- coordinate system
- 正規化方式
- 各群のサンプル数
- contrast方向
- RNA／ATAC閾値
- promoter window
- nearest TSS最大距離
- gene ID変換方法と対応率
- log2FC反転の有無
- padj NAの件数（RNA／ATAC別）
- TF解析の背景定義と件数
- motif結果の出典（ツール、バージョン、database、閾値、背景設定）
- 使用した外部サービス（§16.3）
- 各処理の件数

### 16.2 出力構成

```text
RNA/
  raw_counts.csv
  deg_results.csv
ATAC/
  peak_counts.csv                 # count matrixモード時
  dar_standardized.csv
  dar_significant.csv
  peak_gene_edges.csv
  unmapped_peaks.csv
  atac_validation.json
Integration/
  rna_atac_edges.csv
  rna_atac_gene_summary.csv
  integration_summary.json
  enrichment_by_class.csv
  tf_candidates.csv               # レベル2以降
  motif_import_record.json        # レベル3実行時
  Analysis_Notebook_Multiomics.md
MotifAnalysis/                    # レベル3のBED出力を行った場合
  opened_peaks.bed
  closed_peaks.bed
  all_peaks_background.bed
  motif_analysis_README.txt
Provenance/
  manifest.json
  manifest.md
  reference_manifest.json
```

### 16.3 外部サービスの明示

既存BRIMは次の外部通信を行っている。

- `run_online_mapping()`: gene IDを mygene.info へ送信
- `get_string_network_img()`: 遺伝子リストを STRING-db へ送信

これらは新機能ではないが、本設計の方針（入力データを無断で外部送信しない）の対象である。次を実装する。

- 該当機能の実行前に、送信先サービス名と送信内容の種別を画面に表示する
- ユーザーの明示操作を必要とする（自動実行しない）
- manifestへ `external_services_used` として記録する

未発表データの外部送信を機関ポリシーで制限している利用者が存在するため、この明示は機能の可否ではなく情報提供として必要である。

ATAC／統合機能では新たな外部通信を追加しない。

## 17. テスト計画

### 17.1 Unit test

`test_atac.py`:

- 正常なDAR表を受理する
- peak count matrixのindex形式（`chr1:100-200`）を解析する
- peak count matrixの座標3列形式を解析する
- count matrixからのDAR推定が既知の結果を再現する
- n=1の群でDAR推定を拒否する
- n=2で警告を返す
- 正規化方式の違いが結果に反映される
- start/end逆転を拒否する
- 負の座標を拒否する
- NaN、inf、文字列log2FCを拒否する
- padj範囲外を拒否する
- **padjのNAを保持し `padj_is_na` を立てる**
- 0-based／1-based変換を検証する
- 染色体名変換を検証する
- promoter overlapをstrand別に検証する
- nearest TSSの距離とtieを検証する
- 一対多mappingを保持する
- 最大距離外をunmappedにする
- BED出力が閾値どおりのpeakを含む
- BED出力でopening/closingが分離される

`test_multiomics.py`:

- 全分類を検証する（`rna_not_tested` / `atac_not_tested` を含む）
- **`padj_is_na` の遺伝子が `atac_only` へ混入しない**
- contrast逆向きを拒否する
- species不一致を拒否する
- RNA-only／ATAC-onlyを保持する
- 複数peakのmixed判定を検証する
- threshold境界値を検証する
- 元edgeが失われないことを検証する
- 同じ入力と設定から同じ出力を得る

`test_tf_integration.py`:

- 標的濃縮のFisher検定が既知の分割表で正しい値を返す
- 背景がmapped かつ tested 遺伝子に限定される
- 遺伝子集合が20未満のとき警告フラグが立つ
- BH補正が適用される
- TF発現・TF activityの結合でキーが失われない
- HOMER形式の複合名からTFシンボルを抽出する
- 照合できなかったmotifが報告される
- opening/closingが `peak_set` で区別される
- motif未実行時に該当列がNAとなり、他の軸が影響を受けない
- 3軸を単一スコアへ合成しない（`n_axes_supported` は件数のみ）

`test_provenance.py`:

- RNA単独解析でもmanifestが生成される
- 入力ファイルのSHA-256が記録される
- 外部サービス使用が記録される
- 設定値の欠落がない

### 17.2 UI regression test

`Streamlit AppTest` で以下を検証する。

- ATAC単独で解析できる（両入力モード）
- RNA結果がない場合、Integrationサブタブが表示されない
- レベル1未実行時にレベル2ボタンが出ない
- レベル2未実行時にレベル3ボタンが出ない
- ATAC変更時に古い統合結果・TF結果・motif結果が消える
- genome build不一致を表示する
- contrast方向不一致を表示する
- サンプル数警告が表示される
- サンプルデータから期待件数が得られる
- 日本語と英語の主要UIが表示される
- Exportに必要ファイルが含まれる
- **RNAのみを使う既存操作が変化していない**

### 17.3 Scientific regression dataset

人工データだけでなく、公開済みのpaired bulk RNA-seq／ATAC-seqデータセットから小規模な固定fixtureを作成する。

候補データセット:

| データセット | 用途 | 備考 |
|---|---|---|
| Corces et al. 2016 造血（GSE74912） | CI用の軽量fixture | ピークcount matrixがGEOから単一ファイルで取得可能。FACSソート済みで細胞系譜が既知 |
| ENCODE cell line（K562、GM12878等） | パイプライン統一済みの対照 | 同一cell lineでRNA/ATACが揃い、処理が統一されている |
| TCGA ATAC atlas（Corces et al. 2018） | peak–gene linking の参照 | 同一検体でRNA/ATACが揃う。UCSC XenaからLog2Counts行列を取得可能。原著が相関ベースのpeak–gene linkを公開しており、mapping結果の照合に使える |

fixture作成の要件:

- 原論文とaccessionを記録する
- 再配布ライセンスを確認する
- 既知の方向性を持つ遺伝子／TFを含める
- 完全データではなくテストに必要な最小部分を使用する
- fixture生成scriptを保存する

### 17.4 陰性対照テスト

統合結果が構造を持たないデータに対して有意な出力を返さないことを確認する。実装の妥当性検証として必須とする。

- peak–gene対応をランダムに入れ替えたとき、concordant/discordantの比率が偶然水準に近づく
- RNAのサンプルラベルを入れ替えたとき、統合分類の偏りが消失する
- TF標的濃縮において、遺伝子集合をランダム抽出したときpadjが一様分布に近づく
- motif結果をランダムなTFへ付け替えたとき、3軸一致TFが減少する

各テストは統計的な閾値ではなく、方向性（スコアが有意に低下すること）で判定する。

## 18. 性能要件

初期目標:

- DAR 100,000〜200,000行を一般的なPCで処理可能
- peak count matrixからのDAR推定は、20万peak × 8サンプルで実用時間内（数分〜十数分）
- peak annotationは可能な限り10秒〜60秒以内
- Streamlit rerunごとにannotationやDAR推定を再計算しない
- reference annotationは `st.cache_data` またはモジュールキャッシュで保持
- ファイルhashとmapping設定をcache keyに含める

ATACのpeak数はRNAの遺伝子数の5〜10倍になるため、pydeseq2の実行時間はRNAより長くなる。実行前に推定行数を表示し、長時間になる旨を伝える。

巨大DataFrameをsession stateへ重複保存しない。標準DAR、edge、gene summaryの責務を分け、不要なcopyを減らす。

外部bedtools実行を初期必須依存にしない。Windows portable配布を維持できるPython実装を優先する。

## 19. 依存関係方針

初期版は既存依存関係を最大限利用する。DAR推定は既存のpydeseq2を、標的濃縮は既存のscipyを、TF networkは既存の `brim_tf_networks.py` を用いるため、これらに追加依存は生じない。

追加候補はinterval joinの実装比較後に決定する。

- `bioframe`: Python内でgenomic interval操作を行う候補
- `pyarrow`: compactなParquet参照ファイルを読む候補

追加依存は以下を満たすこと。

- Windows／macOSで導入可能
- OSI承認ライセンス
- portable配布に含められる
- テスト可能
- メンテナンスが継続している

依存追加前に、NumPy／pandasのみのchromosome別interval処理との性能を比較する。

## 20. 実装フェーズ

### Phase 0: 安全な土台

- 現在版をバックアップ／tag化
- Git repositoryを準備
- OSI承認LICENSEを追加
- README、CHANGELOG、CONTRIBUTINGを追加
- 現在のテストを全件成功させる
- CIを設定する

完了条件: ATAC追加前のbaseline testが再現可能。

### Phase 0.5: 既存コードの前提整備

- `run_deg()` に `padj_is_na` / `lfc_is_na` を追加（§3.4）
- `brim_provenance.py` を作成し、RNA単独解析のmanifest生成を移行（§16.1）
- 外部サービス使用の明示表示を追加（§16.3）
- 上記に対する回帰テスト

完了条件: 既存RNA機能の出力が意味的に変化せず、NA情報とmanifestが追加されている。

このPhaseをATAC実装より前に置く理由は、統合分類のロジックが `padj_is_na` に依存するためである。後から追加すると分類実装の書き直しが発生する。

### Phase 1: ATAC core

- `brim_atac.py`
- peak count matrixの読み込みと座標解析
- DAR推定（pydeseq2、正規化方式の選択）
- DAR schema validation
- gene/TSS reference
- promoter overlap
- nearest TSS
- unit tests
- sample data（両モード）

完了条件: StreamlitなしでATAC annotationが完結し、期待edgeと一致する。両入力モードが同一の標準DARテーブルへ収束する。

### Phase 2: ATAC UI

- `Multi-omics` タブと `ATAC-seq` サブタブ
- 入力モード選択／column mapping／正規化選択
- validation summary（サンプル数警告を含む）
- annotation settings
- ATAC plot／table／download
- session reset
- UI tests

完了条件: RNA入力なしでATAC単独解析が完了する。

### Phase 3: Integration core（レベル1）

- `brim_multiomics.py`
- compatibility check
- edge classification（`rna_not_tested` を含む）
- gene summary
- thresholds
- unit tests
- 陰性対照テスト

完了条件: 人工fixtureで全分類とmixed peakを正しく処理し、陰性対照でスコアが低下する。

### Phase 4: Integration UI（レベル1）

- `Integration` サブタブ
- quadrant plot
- evidence table
- class filter
- class別enrichment
- discordant群の可視性確保
- export／provenance
- UI tests

完了条件: BRIM sample RNAとsample ATACから一連の解析・出力が完了する。

### Phase 5: TF候補推定（レベル2）

- `brim_tf_integration.py`
- 標的濃縮（Fisher、BH補正、背景定義）
- TF発現・TF activityの結合
- 3軸テーブル（motif列は空欄表示）
- 段階開示UI
- unit tests

完了条件: sample dataからTF候補が提示され、背景と多重検定の扱いが結果に明記される。

### Phase 6: motif統合（レベル3）

- BED書き出しとコマンド提示
- motif結果のインポートとシンボル正規化
- 3軸テーブルへの結合
- motif出典の記録
- unit tests
- `docs/motif_analysis_guide.md`

完了条件: 外部ツールの結果を読み込み、motif列が埋まる。出典と解析条件がmanifestに残る。

### Phase 7: 検証と公開

- 公開paired datasetによるcase study
- 既存手法／単純nearest-geneとの比較
- wet-lab researcherによる操作確認
- installation guideとtutorial
- GitHub release
- DOI対応

完了条件: 第三者がREADMEのみで同じ結果を再現できる。

版1.0との違いとして、TF・motif機能を公開前（Phase 5・6）へ移動した。理由は§21に記す。

## 21. JOSSと差別化の方針

### 21.1 既存ツールとの関係

RNA-seqとATAC-seqを統合するツールは既に複数存在する。それぞれの位置づけを整理する。

| ツール | 差次的解析 | RNA×ATAC統合 | TF軸 | GUI | 想定サンプル数 |
|---|---|---|---|---|---|
| Linkage (2025) | 外部依存 | あり（相関ベース） | なし | あり | 10以上を推奨 |
| genomeSidekick (2022) | 外部依存 | 限定的 | なし | あり | 制約なし（結果表を受けるのみ） |
| SPACE | — | あり | なし | あり | 自データの投入不可 |
| UTAP2 / Galaxy | あり（モダリティ別） | なし | なし | あり | 制約なし |
| TF-Prioritizer (2023) | あり | あり | あり（footprinting含む） | なし | 制約なし |
| diffTF | あり | あり | あり | なし | 制約なし |
| **BRIM v2.0** | **あり（両モダリティ）** | **あり（群間比較ベース）** | **あり（motifはインポート）** | **あり** | **3〜6を想定** |

Linkageは本機能と最も近いが、上流の差次的解析を外部パイプラインに委ね、信頼できる統合には概ね10サンプル以上を要する相関ベースの手法を中核とする。TF-PrioritizerおよびdiffTFはTF軸まで到達するが、コマンドライン専用であり、footprintingのためBAMを必要とする。

### 21.2 Statement of Needの骨格

新規性の主張は「新しいRNA–ATAC統計手法」ではなく、次に置く。

> 既存のGUI型統合ツールは前処理済みデータと大規模コホートを前提とし、転写因子軸まで到達するツールはコマンドライン専用でBAMを要求する。BRIMは、標準的な実験デザイン（各群3〜6サンプル）のcount matrixを起点として、差次的解析、peak–gene対応、整合性分類、転写因子候補の提示までを、根拠を分離したまま単一のGUIで完結させる。

主張の構成要素は以下の4点であり、いずれか単独では既存ツールとの差にならない。

1. count matrixから開始できる（wet-lab研究者が到達できる入力水準）
2. 少サンプルの群間比較で機能する
3. GUIで完結する
4. その上でTF軸まで到達する

TF軸のみを新規性として主張しない。TF-PrioritizerおよびdiffTFが先行しており、footprintingを用いる分それらのほうが科学的に深い。

### 21.3 想定される指摘と対応

**footprintingを行わない点**

BAMを必要とせず、portable配布とGUI完結を優先した設計上の選択である。motif濃縮は配列の存在を示すのみで結合を示さない。この限界を隠さず、ドキュメントと結果画面に明記する。

**motifを自前で計算しない点**

ゲノム配列の同梱が配布形態と両立しないため。代わりにBED出力とコマンド提示で外部ツール実行を支援する（§12.4）。内蔵表を採用しない判断の根拠は§12.5に記録した。

**少サンプルでの偽陽性**

サンプル数を表示し、閾値を下回る場合は警告する。結果を「探索的」と明示する。陰性対照テスト（§17.4）を公開し、構造のないデータで有意な出力が出ないことを示す。

### 21.4 JOSS要件

- 公開Git repositoryで継続的に開発する
- OSI承認licenseを明示する
- 機能ごとのcommitとreleaseを残す
- `CITATION.cff` を用意する
- 自動テストとCIを公開する
- 入力仕様、tutorial、FAQを用意する
- 実研究での利用例を示す
- 外部ユーザーのfeedbackをissueとして残す
- 既存ツールとの差をStatement of Needで説明する（§21.2）
- AI支援を使用した範囲を記録し、投稿時に開示する

## 22. 科学的注意事項

アプリ内およびdocumentationに以下を明示する。

**peak–gene対応について**

- nearest geneは調節標的の証明ではない。
- distal peakは遠隔遺伝子を調節する場合がある。
- 距離に基づく候補は偽陽性を含む。相関ベースのフィルタを適用していない。

**統合分類について**

- promoter accessibilityと発現の一致は因果関係を証明しない。
- ATACとRNAの不一致は解析失敗とは限らない。
- 異なる時点、試料、batchの結果は直接比較できない場合がある。
- bulk dataは細胞組成変化の影響を受ける。
- peak数に依存するgene-level biasが生じ得る。
- RNAとATACで独立に多重検定補正されたpadjを、統合後の新しい統計的有意性として扱わない。
- 検定されなかった遺伝子（independent filtering等）は `rna_not_tested` として分離しており、非有意とは異なる。

**転写因子候補について**

- 標的濃縮とTF activityはいずれもキュレーション済みデータベースに依存し、研究の蓄積が少ないTFは検出されにくい。
- 核移行や翻訳後修飾で活性化するTFは、mRNA発現および既知標的の応答が弱く、レベル2では検出されない場合がある。
- motif濃縮は結合配列の存在を示すのみで、実際の結合を示さない。
- 3軸は独立した根拠であり、単一のスコアへ統合していない。支持軸数はソート補助であって統計量ではない。
- 標的濃縮のpadjは、レベル1の分類を入力とする新規の検定であり、RNA／ATACのpadjとは独立である。

**サンプル数について**

- ATAC-seqはRNA-seqより群内変動が大きく、同一サンプル数でも検出力が低い傾向がある。
- 各群3サンプル未満では分散推定が不安定であり、結果は探索的である。
- 遺伝子集合が小さい場合、濃縮解析の検出力は著しく低下する。

**enrichmentについて**

- enrichment結果は入力gene集合とbackgroundに依存する。
- TF標的濃縮の背景は、mapされかつ検定された遺伝子に限定している。

## 23. 受け入れ基準

BRIM v2.0の初期統合版は、以下をすべて満たした時点で完成とする。

**既存機能の保全**

- 既存RNA-seqテストがすべて成功する。
- RNAのみを使う既存ユーザーの操作が壊れていない。
- RNA単独解析でもmanifestが生成される。

**ATAC単独**

- peak count matrixからDAR推定が実行できる。
- 解析済みDAR tableを読み込める。
- 両モードが同一の標準DARテーブルへ収束する。
- ATAC DARの正常／異常入力テストが成功する。
- hg38およびmm10でpeak annotationが動作する。
- 一対多peak–gene edgeを保持する。
- サンプル数警告が適切に表示される。

**統合（レベル1）**

- contrast不一致を検出して統合を止める。
- RNA–ATAC全分類を正しく生成する。
- `rna_not_tested` が `atac_only` へ混入しない。
- mixed accessibilityを単一方向へ潰さない。
- 陰性対照テストでスコアが低下する。

**TF（レベル2・3）**

- 標的濃縮の背景がmappedかつtested遺伝子に限定される。
- 3軸が単一スコアへ合成されていない。
- motif未実行時にも他の軸が正しく表示される。
- BED出力がopening/closingを分離し、閾値を記録している。
- motif結果の出典と解析条件がmanifestに残る。

**共通**

- 解析条件と参照releaseが出力へ含まれる。
- 外部サービス使用が明示・記録される。
- サンプルデータから一連の操作を完了できる。
- Windows portable配布で起動できる。
- READMEから第三者が再現できる。

## 24. 保留事項

実装前またはPhase 1中に決定する。

1. interval joinをpandas／NumPyで実装するか、`bioframe` を採用するか。
2. gene annotation sourceとrelease。
3. mm10を初期同梱し、mm39を追加配布にするか。
4. count matrixモードの既定正規化方式をどれにするか。
5. peak数が20万を超える場合の実行時間対策（事前フィルタを提案するか）。
6. 解析済みRNA DEG表の直接uploadをv2.0へ含めるか。
7. motif結果のインポートでHOMER以外にどの形式を初期対応とするか。
8. BRIMのsubtitleをmulti-omics対応に合わせて変更するか。

subtitle変更案:

> BRIM — Bulk Regulatory Integration in Minutes

名称変更は機能実装とは分離して決定する。

## 25. 参考となる既存ソフトウェア

**GUI型の統合・可視化ツール**

- Linkage (Xu et al., BMC Genomics 2025): R Shiny。ATAC-seqとRNA-seqからcis制御領域を相関ベースで予測する。上流の差次的解析は外部パイプラインに依存し、マッチした10サンプル以上を推奨する。
- genomeSidekick (Front Bioinform 2022): R Shiny。解析済みのRNA／ATAC／ChIP結果表を受け取り、volcano plotとGO解析を提供する。統計解析は外部で行う前提。
- SPACE: がんコホートのアクセシビリティを可視化するブラウザ型リソース。ユーザーデータの投入は不可。

**パイプライン型**

- UTAP2 (BMC Bioinformatics 2025): GUI型パイプライン。RNA-Seq、ChIP-Seq、ATAC-Seq等に対応するが、モダリティ横断の統合解釈は提供しない。
- Galaxy: 汎用ワークフロープラットフォーム。
- nf-core/atacseq: ATAC-seqの標準前処理パイプライン。consensus peak count matrixを出力する。

**TF軸の統合手法（CLI）**

- TF-Prioritizer (GigaScience 2023): ATAC/DNase peakとRNA countから、footprinting、結合アフィニティ、機械学習を経てTFを優先順位付けする。Javaパイプライン。
- diffTF (Berest et al., Genome Biology 2019): 差次的TF活性を計算し、RNA-seqと統合してTFを活性化因子／抑制因子に分類する。Snakemake。
- TOBIAS (Bentsen et al., Nat Commun 2020): Tn5バイアス補正を伴うfootprinting。BAMを要する。

**基盤ツール**

- DESeq2 / PyDESeq2: differential analysis。本アプリのRNA DEGおよびATAC DARに使用。
- edgeR / DiffBind: DAR解析の代表的な実装。入力互換の対象。
- CollecTRI / DoRothEA / decoupleR: TF activity inference。既存BRIMで使用。
- HOMER / MEME Suite: motif enrichment。レベル3のインポート対象。
- JASPAR / HOCOMOCO: motif database。

既存手法を適切に引用し、BRIM独自の貢献をUI、入力互換性、透明なevidence保持、再現性、portableな研究workflowとして位置付ける（§21.2）。

## 26. 改訂履歴

| 版 | 日付 | 内容 |
|---|---|---|
| 1.0 | 2026-08-28 | 初版 |
| 2.0 | 2026-09-04 | count matrix入力の追加、TF・motif機能の初期リリースへの繰り上げ、段階開示UIへの変更、既存RNA側のNA処理およびprovenance統一の追加、タブ構成の変更、サンプル数方針の明記、検証データセットと陰性対照テストの具体化、既存ツール比較の更新 |
