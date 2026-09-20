# Phase 5 TF候補推定（Level 2）実装計画

- 状態: **実装中**（2026-09-20着手）。監査役A・Bの計画レビューGOと監査役Cの着手決定をPLAN.mdに記録済み。
- 決定者: 監査役C（AGENTS.md「Planning decision authority」、ユーザー常設委任 2026-09-20）
- 前提: Phase 4完了（A/B GO、`924b31d`の4環境CI成功）。移行監査でAは計画のみGO・実装
  CONDITIONAL NO-GO、Bは同条件のCONDITIONAL NO-GO。条件は、Cの決定を本書とPLAN.mdへ記録し、
  A・Bが本書にGOを出し、Cが着手を記録すること。
- 不変条件I-1〜I-6は全て維持する。新規のネットワーク通信・依存関係は追加しない（`scipy`は
  既に`requirements.txt`にある）。

## 1. 範囲

追加するもの:
- Streamlit非依存の`brim_tf_integration.py`。
- Integrationサブタブ内のLevel 2 UI（実行ボタン、結果表、drill-down、限界の説明）。
- 共有exportとmanifestへの`tf_candidates.csv`・`tf_summary.json`・`tf_level2`ブロック。
- Level 1/2/3の1行ステージ表示（設計書§6.3。Level 3は「この版では利用不可」と表示）。
- 回帰テストと合成テスト入力。

追加しないもの:
- motif/Level 3の操作・BED出力・取り込み（motif軸は明示的な`not_run`表示のまま）。
- 新規ネットワーク通信、新規依存関係、新しいTF activity推定、RNA-only workflowの変更（I-6.1）。
- ORA計算とORA exportの命名変更。

## 2. 監査役Cの決定

### D1 背景遺伝子（universe）
- Level 2のuniverseはPhase 4のORA背景と同一とし、`brim_integration_enrichment.build_ora_background(summary)`
  を再利用して定義が分岐しないようにする。
  - 含む: RNAのpadjとlog2FCがともに検定済み（NAでない）で、検定済みATAC peakが1つ以上対応付いた遺伝子。
    `not_significant`を含む。
  - 除く: `rna_not_tested`、`atac_not_tested`、`both_not_tested`、`rna_only_no_mapped_peak`、未対応遺伝子。
- 設計書§12.2・I-1.5の「peak対応付き・RNA検定済み」は「分類可能な対応付き遺伝子」と読む。全遺伝子でも
  DEGのみでもない。ATAC未検定のpeakしか持たない遺伝子はconcordant/atac_onlyになり得ず、I-1.1により
  「有意でない」とも扱えないため、背景から除く。設計書§12.2に注記を追記する（A・Bはこの読みが
  I-1.5の再解釈に当たらないことを確認する。異議があれば、字義どおりの「対応付き ∩ RNA検定済み」へ
  変更する。変更は`build_universe`のみ）。記録方法は§5.5に従う（AGENTS.mdの不変条件本文は変更しない）。
- 画面とmanifestに`universe_definition`（英/日）、`universe_size`、`n_mapped_rna_tested`（字義どおりの
  サイズ）、`n_excluded_atac_not_tested`、`n_rna_only_no_mapped_peak`、`n_rna_not_tested`を表示・記録する。
- 関数は`build_universe(summary)`（Level 1のgene summaryを入力）。設計書§14.3・ARCHITECTURE §3.4の
  `build_universe(edges, rna)`を更新する。
- 受入テスト: 合成summaryで未検定・未対応の遺伝子がuniverseに無く、`not_significant`が有ること。
  `build_universe(summary) == build_ora_background(summary)`。UIとmanifestに件数が出ること。

### D2 検定する遺伝子集合・BH・Fisher
- 遺伝子集合は利用者が1つずつ選ぶ9種（`SET_DEFINITIONS`）。
  - 単一class7種: `concordant_activation`、`concordant_repression`、`discordant_open_down`、
    `discordant_closed_up`、`atac_only`、`rna_only_on_mapped_peak`、`mixed_accessibility`。
  - 既存classの和集合2種: `concordant_all`（activation ∪ repression）、`discordant_all`
    （open_down ∪ closed_up）。方向の解消はしない（I-1.4）。
  - `mixed_accessibility`は方向非依存で、Phase 4と同じ注記を表示する。
  - 各集合の遺伝子は`集合 ∩ universe`。除かれた件数を記録する。
- Fisher: a = |集合 ∩ ターゲット|、b = |集合| − a、c = |ターゲット ∩ universe| − a、
  d = |universe| − a − b − c。対立仮説は`greater`（片側、過剰表現）。`fisher_alternative="greater"`を
  全行とmanifestに記録し、`odds_ratio`と`fold_enrichment`も出力する。
- ターゲット集合は各TFについてuniverseに制限し、`n_targets_in_universe`と`n_targets_in_set`を出力する。
- BHは1つの遺伝子集合の内側で、検定したTF（`n_targets_in_universe ≥ min_targets`）全体に適用する。
  `n_tests`を集合ごとに出力する。集合をまたぐ補正はしない。画面・manifestに「padjは選択した遺伝子集合内、
  検定したTFのみで補正。複数集合の閲覧は探索的な多重性で補正していない」と明記する。集合内の該当0のTFは
  p = 1として族に残す。
- 受入テスト: 既知の表[[3,1],[1,3]]で`greater`のp = 17/70 ≈ 0.242857（両側の0.4857と異なる）。
  BH([0.01,0.04,0.03,0.005]) = [0.02,0.04,0.04,0.02]で単調。集合とターゲットがuniverseの部分集合。
  `n_tests`が`min_targets`以上のTF数に等しい。集合間の結果が独立。

### D3 ネットワーク・閾値・シンボル整合
- Phase 5のUIとmanifestはCollecTRIのみ。`test_target_enrichment`は`network`と`network_source`ラベルを
  引数にとるため、DoRothEA追加時にAPI変更は不要。DoRothEAと「両方」はPhase 5では提供しない
  （既存のsession stateは`tf_dorothea`のconfidence levelを保持せず、標的濃縮とactivityの整合を
  取れないため。非ブロッカーとして後続Phaseで再検討）。`target_enrichment_source`は常に`collectri`。
- `min_targets`は引数（既定10、UIスライダー5〜30、universe内のターゲット数で数える）。manifestに記録する。
- シンボル整合: 両側を`strip()`と`casefold()`で比較する。`gene_key`・RNAの`gene_symbol`・edgeのシンボルは
  変更しない。casefold比較は記録される変換（I-2.2）で、`n_universe_symbols`、
  `n_universe_symbols_in_network`、`n_network_targets_matched`、`n_casefold_collisions`を記録する。
  universe内の衝突は衝突した全シンボルに対応させる。別名展開と種間変換はしない。
- universeのシンボルがネットワークと1つも一致しない場合は、原因（種またはgene ID型の不一致）を示す
  エラーで停止する。一致率による一般的な停止はしない（キュレーション被覆率は自然に低いため）。
  一致件数は常に表示する。
- Level 1が`rna_gene_id_type == "gene_symbol"`で実行された場合のみLevel 2を実行できる。`gene_id`
  の場合はボタンを無効にして英/日のメッセージを出す。ID変換とネットワーク通信はしない。
- 受入テスト: 大文字のRNAシンボルがMouseネットワークのTF（`Myc`）と一致する。一致レポートが正確。
  一致0でエラー。gene_idモードでボタンが無効。

### D4 TF activity軸
既存の`tf_collectri`はサンプル×TFのスコア行列で、群間統計量もp値もない。新しい統計検定は追加せず、
記述的な規則を既存スコアに適用する。
- 群は`metadata["condition"]`と構造化`rna_contrast`（reference/test）から決める。源は`tf_collectri`のみ。
- 列: `tf_activity_score` = mean(test) − mean(reference)（p値なしの記述的な差）、`tf_activity_source =
  "collectri"`、`tf_activity_n_ref`、`tf_activity_n_test`、`tf_activity_status`。
- status: `supported_up`（testの全スコアがreferenceの全スコアより大）、`supported_down`（逆）、
  `not_separated`、`insufficient_samples`（いずれかの群が3サンプル未満。評価可能に数えない）、
  `not_estimated`（TFが行列に無い）、`not_run`（activity結果が無い、またはcontrastのサンプルが行列に
  全て含まれない）。
- 完全分離規則は記述的な規則であり推測統計ではない。各群3サンプルでは、帰無のTFの約10%が通過し得る。
  限界の説明にその旨を含める。
- 古い結果の検出: Level 2実行時に`activity_fingerprint`（activity行列のindex・columns・丸めた値の
  sha256、未実行なら`null`）を保存する。描画のたびに現在の`tf_collectri`から再計算し、不一致なら
  `integration_tf_results`を消去して`reset_tf_integration_results()`を呼び、「TF Activity結果が変わった
  ため、Level 2結果を消去しました。再実行してください」を表示する。正規化・contrast・閾値の変更は
  既存の`reset_contrast_results` → `reset_integration_results`の連鎖で`tf_collectri`ごと消える。
- 未実行のセルは「not run / 未実行」と表示し、その軸は`n_axes_evaluable`に数えない。TFタブでTF Activityを
  実行してからLevel 2を再実行するよう注記する。BRIMはactivityを自動実行しない。
- 受入テスト: 完全分離で`supported_up/down`、重なりで`not_separated`。2 vs 2で`insufficient_samples`。
  `tf_collectri = None`で`not_run`かつ軸が除外される。activity再推定でTF結果が消える。p値列は作られない。

### D5 TF expression軸
- データは標準化したRNA表（`standardize_rna_results(deg_results, "gene_symbol")`）。照合はD3と同じ
  strip/casefold。
- 列: `tf_rna_log2FoldChange`、`tf_rna_padj`、`tf_rna_padj_is_na`、`tf_rna_lfc_is_na`、`tf_expression_status`。
- status: `supported_up/down`（NAでなく、`padj ≤ rna_padj`かつ`|lfc| ≥ rna_lfc`。閾値はLevel 1の設定を
  引数で受け取る＝I-3.2）、`not_significant`（検定済みで閾値未満）、`not_tested`（いずれかのNAフラグが
  True。`log2FoldChange`と`padj`は空欄で、補完しない＝I-1.1）、`not_in_rna_results`、`ambiguous_symbol`
  （casefold一致が複数）。
- 支持に数えるのは`supported_*`のみ。`not_tested`・`not_in_rna_results`・`ambiguous_symbol`は評価可能にも
  数えない。
- 受入テスト: NAのTFは`not_tested`で値が空欄。padj = 1.0でNAフラグ無しは`not_significant`。
  `padj = rna_padj`と`|lfc| = rna_lfc`の境界は支持。符号でup/down。

### D6 `n_axes_supported`
- Level 2の軸は3つ。標的濃縮は`padj ≤ alpha`（既定0.05、引数）かつ`n_targets_in_set ≥ 1`で支持。
  expressionとactivityは`supported_*`で支持。motif軸は`not_run`で数えない。
- 列: `n_axes_supported`（支持軸数）、`n_axes_evaluable`（支持/非支持が確定した軸数。`not_tested`、
  `not_run`、`not_estimated`、`insufficient_samples`、`not_in_rna_results`、`ambiguous_symbol`は除く。
  これらは「支持」に数えない）。表示は「支持 / 評価可能」（例: 2 / 3）。
- 既定の並べ替えは`n_axes_supported`降順 → `target_enrichment_padj`昇順 → `tf_symbol`。同順位の
  決め手は標的濃縮の1軸のみ。
- 表示専用の保証: 合成スコアを作らない（I-1.3）。`score`・`confidence`・`combined`・`weighted`を含む
  列名は作らない（設計書指定の`tf_activity_score`のみ例外）。軸の積・平均・p値結合をしない（I-1.2）。
  `n_axes_supported`は検定・閾値・フィルタの入力に使わない。UIでは英日で「支持軸数（並べ替えの補助。
  統計量ではありません）」と表示する。
- 受入テスト: 1軸が反転すると数がちょうど1変わる。`not_run`軸は数えない。`n_axes_*`列を除いても他の列が
  変わらない。禁止列名のガードテスト。同順位の並びが決定的。

### D7 小さい集合・空の集合
- 遺伝子0個: 実行しない。`empty_gene_set`を英/日メッセージで返し、実行履歴に記録する。行は作らない。
- 1〜19個: 実行する。全行に`small_gene_set_warning=True`（`SMALL_GENE_SET_THRESHOLD=20`）を付け、表の上に
  「探索的: 20遺伝子未満」を英/日で表示する。
- universe内のターゲットが`min_targets`未満のTFは検定から除外し、`n_tfs_below_min_targets`として
  サマリーとUIに表示する（BHの族に含めない）。
- 受入テスト: 0個で`empty_gene_set`かつ行なし。1個と19個で警告、20個で警告なし。`min_targets`未満の除外と
  件数。`n_tests`が正しい。

### D8 export・manifest・無効化・限界の説明
- exportはLevel 2結果があるときだけ共有exportへ追加する。
  - `Integration/tf_candidates.csv`: 実行した全遺伝子集合を連結し、`gene_set`列を持つ。設計書§8.5の列を
    維持し、D2〜D7の列を加える。`motif_enrichment_padj`・`motif_enrichment_score`は空欄、
    `motif_status="not_run"`、`motif_source`は空欄。同じ集合の再実行はその集合の行を置き換える。
  - `Integration/tf_summary.json`: `build_tf_summary`の出力。
  - Phase 4の既存ファイル名は変更しない。
- manifestは`integration_provenance["tf_level2"]`と`settings["integration"]["tf_level2"]`に記録し、
  `brim_provenance`と既存のexport経路で生成する（I-4.3）。項目: `status`、`network`
  {source, organism, file, n_edges}、`min_targets`、`alpha`、`fisher_alternative`、`bh_scope`、
  集合ごとの`n_tests`、`universe`（定義・件数）、`gene_sets`（名前・遺伝子数・universe外で除いた数・
  小集合警告）、`symbol_matching`、`expression_rule`（使った閾値）、`activity_rule`、
  `activity_fingerprint`、`motif_axis="not_run"`、`n_axes_supported_note`、`input_fingerprint`、
  `run_history`、`external_services_used=[]`、`limitations_text`（英/日）。
- 無効化: **§5.1（レビュー後のC決定）が正**。設計書§15.1の各項目は既存の`reset_integration_results` →
  `reset_tf_integration_results`の経路で処理し、全項目をテストで確認する。TF専用の無効化関数、Level 1再実行時の
  リセット（Phase 4挙動の意図的な変更）、描画時・export時の古さ検査は§5.1に定める。
  （旧記述の「Level 1再実行でORA結果も消える現行の整合と一致」は事実と異なるため撤回した。）
- 限界の説明は、Level 2表の上に常時表示し、manifestの`limitations_text`にも複写する。
  - EN: All three Level 2 columns come from RNA-seq and curated regulatory databases; ATAC only narrows
    the input gene set. / TFs regulated by nuclear translocation or post-translational modification (for
    example NF-κB, STAT, SMAD, HIF-1α) may show weak mRNA change and weak target response, and may not be
    detected. / Curated databases reflect research volume, so TFs with few reports are structurally hard
    to detect. / Target-enrichment padj is a new test computed from the Level 1 classification; it is
    independent of RNA and ATAC padj and is corrected within one gene set only. / Candidates are
    hypotheses and show no evidence that a TF regulates these genes or drives a phenotype. / Motif
    enrichment: not run. Number of supported axes is a sorting aid, not a statistic. / Activity
    "supported" is a descriptive group-separation rule without a p-value.
  - JA: レベル2の3列はいずれもRNA-seqとキュレーション済み制御データベースに由来し、ATACは入力遺伝子集合の
    絞り込みにのみ寄与します。／核移行や翻訳後修飾で活性化するTF（NF-κB、STAT、SMAD、HIF-1αなど）は、
    mRNA発現も既知標的の応答も弱く、検出されないことがあります。／キュレーションDBは研究の蓄積量に依存するため、
    報告の少ないTFは構造的に検出されにくくなります。／標的濃縮のpadjはレベル1の分類を入力とする新規の検定で、
    RNA・ATACのpadjとは独立であり、1つの遺伝子集合内でのみ補正されています。／候補は仮説であり、TFが
    これらの遺伝子を制御する、あるいは表現型を引き起こすことを示す根拠ではありません。／motif濃縮は未実行です。
    支持軸数は並べ替えの補助であり統計量ではありません。／activityの「支持」はp値を伴わない記述的な
    群分離規則です。
- 表現は非因果的にする（"candidate"、"concordant"、"supported axis"、"no corresponding evidence"）。
  "regulates"・"drives"・"causes"は使わない。
- 受入テスト: Level 2の後にのみexport ZIPへ`Integration/tf_candidates.csv`と`tf_summary.json`が入る。
  manifestに上記の全項目がある（`external_services_used == []`を含む）。§15.1の各上流変更で
  `integration_tf_results is None`になる（パラメータ化AppTest）。Level 1再実行でTF結果とORA結果が消える。
  限界の説明が英日で表示され、因果を示す動詞を含まない（正規表現ガード）。

### D9 ステージ表示・ORA命名
- Level 1/2/3の1行ステージ表示はPhase 5の範囲とする（設計書§6.3。テキストのみ）。「Level 1: RNA–ATAC
  比較 → Level 2: TF候補（Level 1の後）。Level 3（motif）はこの版では利用できません。」（英日、`ui()`）。
  Level 3を実装済みのように書かない。
- ORAの実行ID付きCSV命名は不要。Phase 5はORA exportに触れない。Level 2は`tf_candidates.csv`が1ファイル
  （`gene_set`列）で、実行履歴はmanifestにあるため、上書き問題を持たない。

### 監査役Bの質問への回答
- **Q1（fixture）:** `sample_data/`にRNAとATACの対応するペアは無い。決定: 決定的な合成ペアを
  `tests/tf_support.py::synthetic_tf_level1_state(seed=0, species="Mouse")`として作る（大きなデータは
  コミットしない）。`sample_data/README.md`に説明する。同梱のmouse CollecTRIから、ターゲット数が最大の
  TF（同数なら名前順）を「仕込みTF」として選び、約400遺伝子のuniverse（仕込みTFのターゲット＋おとり）、
  NAフラグ付きの`deg_results`形式のRNA、1遺伝子1peakと少数の一対多のedge、少数のATAC NA行を作る。
  仕込みTFのターゲット約25個をconcordant_activationにし、残りは帰無のノイズにする。実際の
  `brim_multiomics`のLevel 1関数を通すため、summaryは本番と同じ形になる。3 vs 3の合成activity行列が
  仕込みTFを分離し、シャッフルした陰性対照には信号がない。完了基準「サンプルデータでTF候補が得られる」は、
  仕込みTFが`n_axes_supported ≥ 2`かつ標的濃縮`padj < 0.05`で、陰性対照では`padj ≤ 0.05`のTFが0個で
  あることで満たす。PLAN.mdで「サンプルデータ」はこの文書化された合成ペアを指すと明記する。
- **Q2（ターゲット集合の取得）:** `brim_tf_networks`は取得手段を公開していない（ネットワークの読込みと
  activity推定のみ。decouplerは遅延import、Streamlit非依存）。決定: `load_collectri_network`を再利用し、
  ターゲット集合の構築は`brim_tf_integration.py`に置く（Streamlit非依存、I-3.1）。`brim_tf_networks`は
  変更しない。
- **Q3（ORA export）:** Phase 5はORA exportに触れない。変わるのは上記のreset挙動のみ。

## 3. ファイル別の実装計画

**新規: `brim_tf_integration.py`**（streamlit・session_state・ネットワークを使わない。
`numpy`、`pandas`、`scipy.stats`、`brim_multiomics`、`brim_integration_enrichment`をimport）。
```python
SET_DEFINITIONS: dict[str, tuple[str, ...]]      # 9集合（D2）
SMALL_GENE_SET_THRESHOLD = 20
class TFIntegrationError(ValueError): ...

def build_universe(summary) -> list[str]
def describe_universe(summary) -> dict
def resolve_gene_set(summary, set_name, universe) -> tuple[list[str], int]
def build_target_sets(network, universe, min_targets) -> tuple[dict[str, frozenset[str]], dict]
def benjamini_hochberg(pvalues) -> np.ndarray
def test_target_enrichment(gene_set, universe, network, min_targets, network_source) -> pd.DataFrame
def attach_tf_expression(tf_table, rna_results, thresholds) -> pd.DataFrame
def attach_tf_activity(tf_table, activity_scores, sample_groups, reference, test, source) -> pd.DataFrame
def add_motif_placeholder(tf_table) -> pd.DataFrame      # motif_status="not_run"
def count_supported_axes(tf_table, alpha) -> pd.DataFrame  # n_axes_evaluable追加、並べ替え（D6）
def get_tf_targets_in_set(tf_symbol, gene_set, network, edges) -> pd.DataFrame  # drill-down。edge行を保持（I-2.1）
def compute_fingerprints(summary, thresholds, contrasts, activity_scores) -> dict
def run_level2(summary, set_name, network, rna_results, thresholds, contrasts,
               activity_scores, sample_conditions, min_targets, alpha, network_source, activity_meta) -> dict   # 実装: edgesは不要（drill-downのみ）
def build_tf_summary(tf_table, settings) -> dict          # manifestブロック（D8）
UNIVERSE_DEFINITION / _JA, LIMITATIONS_EN / _JA, BH_SCOPE_NOTE / _JA
```

**変更: `Bulk_RNAseq_Analyzer.py`**
- `_render_integration_ui`: 1行ステージ表示。Level 1結果があるときだけLevel 2ブロック（遺伝子集合の選択、
  `min_targets`スライダー、「Level 2 TF候補を実行」ボタン。Level 1のID型が`gene_symbol`でない場合や
  未対応のspeciesでは無効化してメッセージ）。universeの説明、結果表、探索的・小集合の警告、「not run」表示、
  限界の説明、TF drill-down、fingerprintによる古い結果の検出、`log_analysis`、
  `try/except TFIntegrationError → st.error`。
- Level 1の実行ハンドラ: 新しい結果の保存前に`reset_tf_integration_results()`を呼ぶ（§5.1。Phase 4挙動の
  意図的な変更）。`invalidate_tf_level2_results()`の新設、`tf_collectri_meta`の追加（§5.4）、`reset_data_results`等
  の追加も含む。
- 共有export生成（1137〜1170行付近）: `Integration/tf_candidates.csv`、`Integration/tf_summary.json`、
  `tf_level2`のmanifestブロックを追加。
- RNA-only workflow、`tf_collectri`、その他のRNAロジックは変更しない。

**文書の変更:** PLAN.md（Phase 5の節。Current development statusは、A/Bの本書へのGOとCの着手決定の後に
更新）、設計書§12.2への注記（universe、片側Fisher、BHの範囲）と§14.3（署名）、ARCHITECTURE §3.4
（署名の修正）、本書、`sample_data/README.md`（合成ペアの説明）、README/CHANGELOG（実装済みの挙動のみ、
完了時）、i18nは`ui()`で英日。

**テスト**
- 新規`tests/tf_support.py`: 上記の合成入力ビルダー。
- 新規`tests/test_tf_integration.py`: Fisher既知表と片側/両側の差、BH既知ベクトル・単調性・族の大きさ、
  universeとORA背景の一致・除外class、シンボルcasefoldと記録件数・一致0のエラー、集合のuniverse制限、
  小集合0/1/19/20個、`min_targets`除外、expression（NA非補完・境界・`not_in_rna_results`）、
  activity（分離・`insufficient_samples`・`not_run`）、軸カウント（`not_run`除外・禁止列名ガード・
  決定的な並べ替え）、入力変更でfingerprintが変わること、仕込みTFの検出と陰性対照が帰無であること、
  `brim_tf_integration.py`がStreamlitをimportしないこと（I-3.1、既存の検査と同形式）、ネットワーク通信なし。
- 新規`tests/test_tf_integration_ui.py`（AppTest）: Level 1が無ければLevel 2ボタンが無く、あれば有る。
  `gene_id`型でボタン無効。universeの定義・件数と英日の限界説明が表示される。motif列と（activity無しの
  とき）activity列が「not run」。古いactivity fingerprintで結果が消える。§15.1の各上流変更とLevel 1再実行での
  無効化（パラメータ化）。exportとmanifestの項目。Level 3の操作が無い。英日の文言に因果の動詞が無い。
- 既存テストは変更しない。ローカルではIntegration関連のファイルを実行する。既知の9件のPyDESeq2失敗は
  変わらず、CI（Ubuntu/Windows × Python 3.11/3.12）を正とする。

## 4. 受入基準（PLAN.mdのPhase 5との対応）
D1〜D9の受入テストに加えて、PLAN.mdの次の項目を満たす。
- Fisherが既知の表で正しい。
- universeが検定済み・対応付き済み（D1）に限られ、定義と件数が表示される。
- 20遺伝子未満の集合で警告フラグが立つ。
- BHが適用され、範囲が結果に明記される。
- 3軸が単一スコアへ合成されない。
- Level 1実行前はLevel 2ボタンが表示されない。
- 限界（翻訳後修飾で活性化するTF、DBの偏り）が画面に表示される。
- 完了基準は§5.6のとおり（合成の陽性・陰性対照での回帰検証であり、実データ・生物学的妥当性の検証ではない）。

## 5. 計画レビュー後の監査役C決定（2026-09-20）— 本節が旧記述に優先する

監査役A（CONDITIONAL NO-GO）とB（GO、3点の反映が条件）のレビュー指摘を、監査役Cが次のとおり決定した。

### 5.1 TF専用の無効化・Level 1再実行・古さ検査（A1、A2/B1）
- `invalidate_tf_level2_results()`（新規、`Bulk_RNAseq_Analyzer.py`）: `integration_tf_results = None`にし、
  `integration_provenance`があれば`tf_level2`キーを`pop`する。ORA（`integration_enrichment`、`ora_history`）と
  `integration_motif_*`には触れない。
- `reset_tf_integration_results()`は従来どおり`integration_enrichment`、`integration_tf_results`、
  `integration_motif_*`を消去し、加えて`integration_provenance["tf_level2"]`を`pop`する。
- §15.1の各項目（RNA再実行、RNA/ATAC閾値、contrast、species、mapping、入力変更）は既存の
  `reset_integration_results` → `reset_tf_integration_results`の経路で処理される。全項目をテストで確認する。
- Level 1の実行ハンドラ（〜2048行）は現在resetを一切呼ばない（再実行は`ora_history=[]`にするだけで
  `integration_enrichment`は残る）。**Phase 4挙動の意図的な変更**: 新しい結果を保存する前に
  `reset_tf_integration_results()`を呼ぶ。TF・ORA・motifの結果が消え、「Level 1再実行後にORA結果が残るが
  履歴は空」という不整合が解消する（文書化された修正）。変更するのはORAのsession stateの寿命のみで、
  ORAの計算・ORA exportの命名・列は変えない。CHANGELOGに1行を記録し、READMEの該当記述があれば整合させる。
  既存テストがLevel 1再実行後のORA残存を主張している場合は、テストを変更せず作業を止めてCへ差し戻す。
- 古い結果の検出（描画時とexport時の両方）: `compute_fingerprints`で現在の状態から再計算する。
  - Level 1 fingerprint（gene summary・`integration_settings["thresholds"]`・
    `integration_settings["rna_contrast"]`）の不一致 → `reset_tf_integration_results()`＋通知。
  - `activity_fingerprint`の不一致 → `invalidate_tf_level2_results()`のみ（ORAは残す）＋通知
    「TF Activity結果が変わったため、Level 2結果を消去しました。再実行してください」。
  - exportは描画に依存しない。共有export生成時にも同じ検査を行い、現在でないTF結果は
    `tf_candidates.csv`・`tf_summary.json`・`tf_level2`を出力に含めない。
- manifestの`tf_level2`は`integration_tf_results`から組み立てる（単一の源）。`None`なら
  `settings["integration"]["tf_level2"]`キー自体を出力しない。`integration_provenance["tf_level2"]`は
  実行時の写しで、無効化のたびに`pop`する。
- 受入テスト（D8に追加）: TF専用の無効化で`integration_tf_results`が`None`、`provenance["tf_level2"]`が無く、
  `integration_enrichment`と`ora_history`が保持される。`reset_tf_integration_results()`後に`tf_level2`が無い。
  古いTF結果がsession stateに残ってもZIPに`tf_candidates.csv`・`tf_summary.json`・`tf_level2`が入らない。
  Level 1再実行で`integration_enrichment`・`integration_tf_results`が`None`になり、旧ORA CSVがZIPに残らない。

### 5.2 D5の境界とD4の非有限値（A3）
- D5: `supported_up/down`は、NA・非有限でなく、`padj ≤ rna_padj`かつ`|lfc| ≥ rna_lfc`かつ**`lfc != 0`**
  （Level 1のRNA有意判定`brim_multiomics.py`と同じ規則。閾値は`integration_settings["thresholds"]`から引数で受ける）。
  `lfc == 0`は閾値を満たしても`not_significant`。値が非有限の場合は`not_tested`（値は空欄、補完しない）。
  テスト: `lfc == 0`かつ`padj ≤ rna_padj`で`not_significant`。`rna_lfc = 0`かつ`lfc = 0`は支持されない。
- D4: 対象TFの列、またはcontrastのどちらかの群のサンプルのスコアに非有限値（NaN・inf）が1つでもあれば
  `not_estimated`とし、`tf_activity_score`は空欄。NaNを除いた再計算はしない。`supported_up`は
  testの最小値がreferenceの最大値より大きい場合（同値は`not_separated`）。
- 群メンバーの源: `tf_collectri.index`（サンプル名）を、TFタブが使うRNAメタデータの`condition`列
  （`integration_settings["rna_contrast"]`のreference/test）で引く。contrastのサンプルが`tf_collectri.index`に
  全て含まれなければ`not_run`。indexにあるがcontrast外のサンプルは使わず、`activity_n_samples_ignored`に記録する。
- テスト: TF列のNaN、いずれかの群のNaNで`not_estimated`。同値で`not_separated`。contrastのサンプル欠落で`not_run`。

### 5.3 設定値の取得源（B2）
- 閾値: `integration_settings["thresholds"]`（`_integration_thresholds()`の現在値ではない）。
- RNA contrast: `integration_settings["rna_contrast"]`（`st.session_state["rna_contrast"]`の現在値ではない）。
- 種: `integration_settings["species"]`（Human→"human"、Mouse→"mouse"。他は実行不可）。
- gene IDの型: `integration_settings["rna_gene_id_type"]`（`gene_symbol`のみ実行可）。
- これらを`run_level2`の引数へ明示的に渡す（モジュール内でsession stateを読まない）。
- テスト: Level 1実行後にRNA閾値・contrast・ウィジェットを変更しても（reset前に）、Level 2は実行時の
  `integration_settings`の値で動作し、manifestに同じ値が記録される。

### 5.4 activityの出自の記録（A4、選択肢a）
- TFタブが`tf_collectri`を保存する箇所（約5072行）で、同時に`st.session_state["tf_collectri_meta"]`へ
  `{"method_requested", "method_used", "tmin", "normalization", "organism", "network": "collectri"}`を保存する
  （追加のみ、RNA-only挙動は不変）。`tf_collectri_meta`は`tf_collectri`が消される全ての箇所で同時に`None`にする:
  `_DATA_RESULT_DEFAULTS`（初期化と`reset_data_results`）、`reset_contrast_results`のキー一覧、正規化方法変更時
  （〜2313行）、遺伝子長読込み時（〜2356行）。session state初期化（〜201行）にも追加する。TFタブのウィジェットは
  実行後に変更できるため、meta値は実行時の値のみを信頼する。
- `tf_activity_min_targets`（TFタブのtmin＝activity推定に使うTF集合）はLevel 2の`min_targets`
  （universe内ターゲット数の下限）とは別の量。manifestの`activity_rule`と`network`に別名で記録し、UIも別の
  パラメータとして注記する。
- organism: Level 2のネットワークorganismは`integration_settings["species"]`から決める。
  `tf_collectri_meta["organism"]`と異なる場合はactivity軸を評価せず、実行を止めて英日のエラー
  （「TF Activityは別の生物種で推定されています。TFタブで再実行してください」）を出す。
  `tf_collectri_meta`が`None`のときはactivity軸を`not_run`とし、`activity_fingerprint`のみで識別する旨を
  manifestに記録する。
- `activity_fingerprint`は`tf_collectri`のindex・columns・丸めた値のsha256＋`tf_collectri_meta`のJSON。
- テスト: metaが`tf_collectri`と同時に保存される。4つの消去経路でmetaが`None`になる。organism不一致で実行が
  止まる。`tf_activity_min_targets`と`min_targets`が別々に記録される。TFタブのウィジェット変更が記録値を変えない。

### 5.5 D1の記録方法（A5）
- 監査役Cは不変条件を放棄・再解釈する権限を持たず、AGENTS.mdの不変条件本文の編集も委任範囲外である。
  したがってAGENTS.md I-1.5は変更しない。D1は設計書§12.2の注記、本書、PLAN.mdに記録する。
  D1の読み（分類可能な対応付き遺伝子）がI-1.5の明確化に当たるかは、監査役Aが明確化と判断済みで、
  監査役Bは着手時のGOで明示的に確認する。I-1.5の開示義務は、字義どおりの件数（`n_mapped_rna_tested`）の併記で
  満たす。A・Bのいずれかが衝突と判断する場合は、字義どおりの定義へ戻す（変更は`build_universe`のみ）。
- AGENTS.md I-1.5に設計書§12.2への参照を1行追記するかどうかは、ユーザーへのOPEN QUESTION（§6）。
  実装の着手を妨げない。

### 5.6 合成テスト入力の固定値と完了条件（A5）
- `synthetic_tf_level1_state`の固定パラメータ（変更・調整・seed探索をしない）:
  - `seed=0`（`numpy.random.default_rng(0)`）、species="Mouse"。同梱CollecTRIから、ターゲット数最大（同数なら
    名前順で先頭）のTFを仕込みTFとする（ターゲット数≥60を`assert`）。
  - `n_universe=400`: 仕込みTFのターゲット60（名前順先頭）＋おとり340（仕込みTFの非ターゲットから、名前順に
    ソートした上で`rng.choice`）。
  - 仕込み標的25（上記60のうち名前順先頭25）をconcordant_activationとする。おとり15をconcordant_activationに
    加え、合計40。残りのおとりは他のclassと`not_significant`にseedで割り当てる。
  - NA割合: RNAの検定NA 5%、ATAC peakのNA 5%（各`rng.choice`、仕込み標的はNAにしない）。一対多edgeは全遺伝子の約5%。
  - activity: 3 vs 3。仕込みTFはreference=[-0.5, 0.0, 0.5]、test=[2.5, 3.0, 3.5]（決定的）。他のTFは`N(0, 1)`の
    ノイズ。仕込みTFのRNA: `log2FoldChange=1.5`、`padj=0.001`。
  - 陰性対照: 同じseedで、universe内の遺伝子と集合classの対応を並べ替える（集合の内容は同数）。
- 陰性対照の主張（「`padj ≤ 0.05`のTFが0個」）は**この固定seedに対する回帰検査**であり、一般的な偽陽性率の
  保証ではない。
- **完了条件**: 文書化した合成の陽性・陰性対照ペア（`tests/tf_support.py`、固定seed）で、仕込みTFが候補
  （`n_axes_supported ≥ 2`かつ標的濃縮`padj < 0.05`）として得られ、陰性対照では（この固定seedで）
  `padj ≤ 0.05`のTFが0個であり、背景と多重検定の扱いが結果に明記されること。実データでの検証ではなく、
  生物学的な妥当性を示すものではない。README/CHANGELOGは合成テストでの検証にとどめ、生物学的検証を
  示唆する表現を使わない。PLAN.mdのPhase 5完了条件も同じ文言に揃える。

### 5.7 テスト表現（B3）
- `brim_tf_integration.py`のソース文字列に`import streamlit`と`st.session_state`が含まれないこと
  （`tests/test_atac.py:75`と同形式）。加えて、`ast`で`import`/`from ... import`を走査し`streamlit`が無いこと、
  `requests`・`urllib`・`socket`等のネットワークモジュールが無いことを確認する。

### 5.8 追加の限界説明・表示（推奨事項を採用）
D8の`LIMITATIONS_EN / _JA`に次を追加する。
- EN: Target enrichment ignores the sign of regulation (activation vs repression) in the network. / Padj is
  BH-corrected within one gene set across tested TFs; TFs with overlapping target sets are not independent
  tests. / The three axes are all RNA-derived (target enrichment uses the RNA-derived Level 1 classes;
  expression and activity use RNA) and are not independent evidence. / Some CollecTRI entries are complexes
  and may appear as not_in_rna_results. / Activity "separated_up/separated_down" is a descriptive rule; about
  10% of null TFs pass it with 3 vs 3 samples.
- JA: 標的濃縮はネットワークの制御の符号（活性化・抑制）を考慮しません。／padjは1つの遺伝子集合内の検定したTFで
  BH補正されており、標的集合が重なるTFは独立な検定ではありません。／3つの軸はいずれもRNA由来であり
  （標的濃縮はRNA由来のレベル1分類を使用）、独立な根拠ではありません。／CollecTRIの複合体エントリは
  `not_in_rna_results`になることがあります。／activityの「separated_up/separated_down」は記述的な規則で、
  3 vs 3では帰無のTFの約10%が通過します。
- activity列の表示ラベルは`separated_up` / `separated_down`とし（内部の値は一貫して同名に統一する）、
  「約10%」の注記をactivity列の直近に置く（限界の説明ブロックだけにしない）。
- `n_casefold_collisions`は`n_casefold_collisions_universe`と`n_casefold_collisions_network`に分けて記録する。

### 5.9 A・Bの最終レビューでの追記事項（非ブロッカー、実装時に反映）
- D6: 軸は、expressionが`supported_*`、またはactivityが`separated_*`（内部値は`supported_*`と同義。
  表示ラベルのみ`separated_*`）のとき支持として数える。用語は§5.2・テストでも統一する。
- `activity_fingerprint`に、contrastのreference/testそれぞれのサンプル名リストを含める（`metadata["condition"]`の
  付け替えで、contrastが同じままでも結果が無効化されるようにする）。
- §5.1の受入テストに追加: Level 1再実行では、リセットが**新しい結果の保存前**に行われ、新しいLevel 1の出力が
  消えないこと。
- 正規化変更（〜2313行）と遺伝子長読込み（〜2356行）は`reset_integration_results`を呼ばずに`tf_collectri`を
  消すため、描画時・export時のfingerprint照合が「以前は非null、今はnull」を不一致として検出することを
  テストする（正規化変更経路のテストを追加）。
- Level 2のUI fixtureがactivityを使う場合は、`tf_collectri`と同時に`tf_collectri_meta`も設定する
  （metaが無いとactivity軸は§5.4により`not_run`）。既存の`tests/test_existing_result_views.py`のfixtureは変更しない。
- 陰性対照の「0個」は、実装の早い段階で実行して確認する。満たさない場合はseedを探索せず、Cへ差し戻す。
- 仕込みTFはSp1（ターゲット1341）になる見込みで、他のTFも陽性対照で濃縮され得るが、検証項目（仕込みTFが
  `n_axes_supported ≥ 2`かつ標的濃縮`padj < 0.05`）には影響しない。

## 6. OPEN QUESTIONS
- **ユーザーへ（非ブロッカー、1件）:** AGENTS.md I-1.5に、設計書§12.2の明確化への参照を1行追記するか。
  「追記しない」場合は設計書§12.2の注記が正となり、A・BのGOがゲートとなる。A・Bが衝突と判断すれば
  字義どおりの定義へ戻す。実装はこの回答を待たない。
- **A・Bへ（着手GOの条件）:** D1がI-1.5の再解釈に当たらないことの確認（Bは明示的に）。Level 1再実行でORA結果が
  消えるPhase 4挙動の意図的な変更の承認。
- D4の完全分離規則は記述的で推測統計ではないこと、D3でDoRothEAを提供せずCollecTRIのみとすることは、
  非ブロッカーとして引き続きA・Bの確認事項。

## 7. 実装順序（監査役C決定、2026-09-20）
各ステップは小さな単位で、自分のテストを持つ。ローカルでIntegration関連のテストを実行する（既知のPyDESeq2の9件の
ローカル失敗は想定内）。現ステップのテストが通るまで次へ進まない。

1. **テスト入力と純関数群。** `tests/tf_support.py`（§5.6の固定パラメータ、調整・seed探索をしない）と、
   `brim_tf_integration.py`の`build_universe`・`describe_universe`・`resolve_gene_set`・`build_target_sets`・
   `benjamini_hochberg`・`test_target_enrichment`。Fisher既知表と片側/両側、BH既知ベクトル、universe = ORA背景、
   casefold、小集合（0/1/19/20）をテストする。**陰性対照をここで実行して確認する**（§5.9）。`padj ≤ 0.05`のTFが
   あればseedを探索せず、Cへ差し戻す。
2. **軸。** `attach_tf_expression`・`attach_tf_activity`・motifプレースホルダ・`count_supported_axes`（D6、
   `separated_*`の用語）。§5.2の境界（lfc=0、非有限値、同値、contrastサンプル欠落）、禁止列名ガード、決定的な並びをテストする。
3. **fingerprint・`run_level2`・`build_tf_summary`・drill-down。** `activity_fingerprint`にcontrastのサンプル名リストを
   含める。入力変更でfingerprintが変わること、合成の端から端までの検査（仕込みTFが`n_axes_supported ≥ 2`かつ
   `padj < 0.05`、陰性対照が帰無）、ソース文字列とASTによるStreamlit・ネットワーク不使用のテスト（§5.7）。
4. **session state・無効化・Level 1再実行のリセット。** `tf_collectri_meta`（4つの消去経路で`None`、生物種不一致の扱い）、
   `invalidate_tf_level2_results()`、`reset_tf_integration_results()`が`tf_level2`を`pop`、Level 1の新しい結果の
   保存前にリセット。既存テストがLevel 1再実行後のORA残存を主張していれば、テストを変更せず作業を止めてCへ差し戻す。
   §5.1・§5.9のテスト（正規化変更経路を含む）。
5. **UI。** ステージ1行表示（Level 3は「この版では利用できません」）、Level 2ブロック、無効状態、universeの説明、
   §5.8を含む英日の限界説明とactivity列直近の「約10%」注記、motifとactivityの「not run」表示、drill-down、
   描画時の古さ検査。AppTest（`tests/test_tf_integration_ui.py`）と、英日文言に因果の動詞が無いことの検査。
6. **export・manifest・provenance。** `Integration/tf_candidates.csv`、`tf_summary.json`、`tf_level2`ブロック
   （`integration_tf_results`のみから構築）。export時にも同じ古さ検査を行い、古いTF結果は出力しない。ZIPの内容をテストする。
7. **文書。** 設計書§12.2（注記）・§14.3、ARCHITECTURE §3.4、`sample_data/README.md`、README、CHANGELOG。
   READMEとCHANGELOGは実装済みの挙動のみを書き、合成対照での検証にとどめ、生物学的検証を示唆しない。
   PLAN.mdのPhase 5完了条件を§5.6に揃える。
8. **検証と実装後監査。** Integration関連テストをローカルで実行し、pushして4環境（Ubuntu/Windows × Python 3.11/3.12）の
   CIを確認する。その後、監査役A（設計・科学的整合性・不変条件・文書）とB（コード・テスト・export・provenance・CI証拠）に
   監査させる。両者GOの後でのみPhase 5を完了とする。non-GOなら修正してCIを再実行し、A・Bの新しい判定を得る。

**Cへ差し戻す条件:** 陰性対照が満たされない。既存テストが§5の決定と衝突する。変更がI-1〜I-6、RNA-only workflow、
ORAの計算・export命名に触れる、または範囲・依存関係・ネットワーク通信を追加する。計画と設計書の実際の衝突。
有効なテストの弱体化・skip・削除が必要になる。

**CIの確認点:** ステップ4と6の終わりに早期のCI確認としてpushしてよい（任意）。A・B監査の前に、最終commitのCIは必須。