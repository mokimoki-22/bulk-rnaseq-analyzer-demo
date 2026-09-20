# Phase 6 motif取り込み（Level 3）実装計画

- 状態: **実装中**（2026-09-21着手）。監査役A・Bが§7を含む本書にGOを発行し、監査役Cが着手を決定した。PLAN.mdに記録済み。
- 決定者: 監査役C（AGENTS.md「Planning decision authority」、ユーザー常設委任 2026-09-20）
- 前提: Phase 5完了（`9f4142d`、A・B実装後監査GO）。Phase 5→6移行監査で、A・Bとも計画のみGO、実装はNO-GO。`cd0a2a4`は`9f4142d`にPLAN.md・phase5計画書を足しただけの文書commit（コード差分なし）。
- 不変条件I-1〜I-6は全て維持する。新規のネットワーク通信・依存関係・外部バイナリ呼び出しは追加しない。motif scanは実装しない（I-5.1）。BRIMは外部ツールを実行せず、BED出力、コマンド提示、結果のimportだけを行う。

## 1. 範囲

追加するもの:
- Streamlit非依存の`brim_motif_import.py`（peak集合とBED、コマンド・手順文、結果の読込みと検証、TFシンボル正規化、結合ビュー、manifestブロック）。
- Integrationサブタブ内のLevel 3ブロック（BED準備とダウンロード、コマンド提示、結果import、motif列の表示）。Level 2の現在の結果があるときだけ表示する。
- 共有exportへの`MotifAnalysis/`、`Integration/motif_*`、manifestの`tf_level3`ブロック（別ブロック）。
- `docs/motif_analysis_guide.md`（§D13の内容規則に従う）。
- 1行ステージ表示の更新（Level 3は「Level 2の後に利用可能」）。

追加しないもの:
- 内蔵motif表、motif scan、de novo結果の取込み、HOMER以外の専用パーサ（他ツールは汎用CSV/TSVのみ）。
- 1つのpeak集合への複数ツールの併存、TF別名・family辞書、`n_axes_supported`へのmotif算入、motifと他軸の合成スコア。
- `export_peaks_as_bed`の変更（後述）、`tf_level2`ブロック・`tf_candidates.csv`・Level 2の関数の意味の変更。
- 新規ネットワーク通信、新規依存、RNA-only workflowの変更（I-6.1）。
- サンプル用のmotif結果ファイルの同梱（実際のツール出力と誤認されるため）。

## 2. 監査役Cの決定（計画レビュー後の§7が、ここの記述に優先する）

### D1 対応ツール・形式・列の対応（A1、B Q2）
- 第一級対応は HOMER の `knownResults.txt`（`tool="homer_known"`）。それ以外は汎用CSV/TSV（`tool="generic"`）で、列対応を画面で確認する（設計書§7.3のalias提案と同じ形式）。de novo結果は取り込まない。
- HOMER形式の認識は、ヘッダー行を小文字化・空白正規化して前方一致で行う。
  - 必須: `motif name`、`p-value`、`q-value (benjamini)`。
  - 任意: `log p-value`、`# of target sequences with motif(of N)`、`% of target sequences with motif`、背景側の同名2列。
  - 区切りはタブ。
- 対応の固定:
  - `motif_name` ← Motif Name
  - `pvalue` ← P-value
  - `padj` ← `q-value (Benjamini)`（ツールが報告した値。BRIMは再計算も再補正もしない。I-1.6）
  - `pct_target`、`pct_background`は「%」を外した数値をそのまま保持する。
  - `enrichment_score`は空欄。HOMERの列から導出せず、BRIMは強度を計算しない。
- HOMERヘッダーの「(of N)」から`n_target_sequences_reported`と`n_background_sequences_reported`を記録する（§D5、§D6の照合に使う）。
- 汎用CSV/TSVの列対応キー: `motif_name`（必須）、`padj`と`pvalue`（少なくとも一方は必須）、`enrichment_score`、`motif_id`、`peak_set`（任意）。alias候補は大文字小文字非依存の固定リスト（`q-value`、`qvalue`、`fdr`、`padj`など）。提案は初期値にすぎず、利用者の確認が必要。`peak_set`列を対応付けた場合、値が`opening`または`closing`の1種類だけでなければ、分割を求めるエラー（設計書§12.4）。
- ヘッダーが認識できない場合、または`>`で始まるmotifファイルの場合は、「HOMER knownResults.txt（Motif Name, q-value (Benjamini)）またはCSV/TSVを選んでください。de novo結果は取り込めません」というエラーを返す。
- `padj`列が無い（または全て欠損）場合は取り込めるが、`no_padj_reported`と表示する。BRIMは`pvalue`を補正しない。
- 根拠: 公式仕様書を参照できないため、認識は前方一致にとどめ、不明ならエラーにする。実際のツール出力との突合は§6のOPEN QUESTIONとPhase 7で行う。
- 受入テスト:
  - HOMER風の合成表を読める。padjは`q-value (Benjamini)`の値そのもの。
  - de novo、motifファイル、列不足のファイルは、対処が分かるエラーで拒否される。
  - 汎用CSVで、alias提案、列対応、`peak_set`列の混在エラーが働く。
  - pvalueのみの入力で`no_padj_reported`になり、padjが導出されない。

### D2 TFシンボル正規化（A2、A13、A10）
- 抽出規則`extract_tf_symbols(motif_name)`:
  1. 深さ0の`(`または`/`の直前までを取る（`(POU,Homeobox/HMG)`のように括弧内に`/`があっても切らない）。
  2. `:`の連続で分割してヘテロダイマーを成分に分ける。
  3. 各成分をstripする。空の成分は捨てる。`(var.2)`などの括弧は手順1で落ちる。
- 照合は既存の`fold_symbol`（strip + casefold。`brim_tf_integration`から再利用し、Phase 5と規則を共有する）を使う。
- 参照シンボル`reference_symbols`は呼び出し側が渡す。中身はRNA結果のgene symbolとCollecTRIのTF（source）の和集合。
- 結果のTF表記は参照側の綴りとする。参照内でcasefold衝突がある場合は名前順で先頭を使い、件数を記録する。
- 1つのmotif行が複数の成分に対応する場合は、各成分に1行ずつ展開する（一対多を保持。I-2.1の趣旨）。`motif_form="heterodimer"`と相手成分を記録する。一部の成分だけ照合できた行は`partially_matched`とする。
- 同じTFに複数のmotif行がある場合は、全行を`motif_symbol_map.csv`に残す。表示用の代表は、報告padjが最小、同値ならpvalueが最小、さらに同値ならmotif名順の1行とする。`representative_motif_rule="smallest_reported_padj_then_pvalue_then_motif_name"`、`n_motifs_for_tf`を記録する。padjが欠損の行は代表の候補から外し、全て欠損なら`no_padj_reported`とする。
- 別名・旧シンボル・familyは解決しない。未照合として報告する（例: `AP-1(bZIP)`）。辞書は作らない（新しいデータ依存になるため。後続Phase）。
- 未照合レポート`UnmatchedReport`（dict）:
  - 件数: `n_motif_rows`、`n_rows_matched`、`n_rows_partially_matched`、`n_rows_unmatched`、`match_rate`、`reference_size`
  - 一覧: `unmatched`（motif名、抽出シンボル、理由コード `no_symbol_extracted` / `not_in_reference`）
  - 規則: `rule_text`（英日）
- 画面には件数と全一覧を出す。`motif_symbol_map.csv`には全件を書く。manifestには件数と先頭50件の名前だけを置き、全件はCSVを参照する。
- 照合できた行が0（一部照合を含む）の場合は実行を止め、種または遺伝子シンボルの不一致を示すエラーを返す。一致率が低いだけでは止めない（HOMERにはfamily名が多い）。
- 受入テスト:
  - `Stat3(Stat)/mES-Stat3-ChIP-Seq(GSE11431)/Homer`（合成）→`Stat3`。
  - `Oct4:Sox17(POU,Homeobox/HMG)/…`→2成分。片方未照合なら`partially_matched`。
  - 大小文字違い（`STAT3`、`stat3`）が同じTFに照合される。
  - 同じTFの複数motifで代表規則が決定的。全行が`symbol_map`に残る。
  - `AP-1(bZIP)`が未照合として一覧に出る。全て未照合ならエラー。

### D3 peak集合とBED出力（A3、A18、B）
- 新しいStreamlit非依存関数`build_peak_sets`を作る。既存の`brim_atac.export_peaks_as_bed`は変更しない。理由: `"all"`が検定されなかったpeakまで出力する既存の挙動を`tests/test_atac.py:528`が固定しており、背景には使えないため。
- 入力は`atac_results`（標準DAR表）と、`integration_settings["thresholds"]`の`atac_padj`・`atac_lfc`、`genome_build`、`species`。ライブのウィジェット値は使わない（I-3.2）。
- 検定済みpeakは`~padj_is_na & ~lfc_is_na`。
  - `opening`: 検定済みかつ`padj ≤ atac_padj`かつ`lfc ≥ atac_lfc`かつ`lfc > 0`。
  - `closing`: 検定済みかつ`padj ≤ atac_padj`かつ`lfc ≤ -atac_lfc`かつ`lfc < 0`。
  - `background`: 検定済みの全peak（opening・closing・非有意を含む。`padj_is_na`は除く。I-1.1）。
  - 境界は等号を含む。`_classify_dar`の規則とも一致させる。
- BEDの形式:
  - 4列（`chrom`、`start`、`end`、`name`）。0-based half-open。LF改行。UTF-8。
  - 入力表の順序を保つ（決定的）。
  - 座標系はREADMEとmanifestに記録する（I-2.2）。座標変換や染色体名の変換はしない。DAR表のchrom表記のままとし、その旨を記録する。
- `name`はpeak_idの空白類（スペース、タブ、改行）を`_`に置換したもの。置換した件数を`n_ids_sanitized`として変換記録に残す（I-2.2）。置換後にidが衝突する場合は、衝突例を示すエラーで止める。
- 出力ファイル名は`opened_peaks_padj{g}_lfc{g}.bed`、`closed_peaks_padj{g}_lfc{g}.bed`、`all_peaks_background.bed`（`{g}`は`format(x,"g")`）。名前は`^[A-Za-z0-9._-]+$`に一致することを`assert`する（シェル安全）。
- 集合が0件の場合、そのBEDは書かない。画面に「0 peak」と表示し、その集合のコマンドは出さない。`SMALL_PEAK_SET_WARNING = 100`未満では「探索的」の警告を出す（BRIM自身の目安であり、ツールの要件ではない）。
- 集合の記録: `counts`（`n_dar_total`、`n_tested`、`n_not_tested_excluded`、`n_opening`、`n_closing`、`n_background`、`n_ids_sanitized`）、各BEDのsha256、`peakset_fingerprint`（閾値・build・species・座標規約・3つのBED本文のsha256のハッシュ）。
- 受入テスト:
  - 境界（padj = 閾値、lfc = 閾値）は含む。わずかに外側は含まない。lfc = 0は含まない。`padj_is_na`と`lfc_is_na`のpeakは全集合に含まれない。
  - opening/closingの本文が`export_peaks_as_bed`と一致する（idの置換が無い前提）。
  - 背景の件数 = 検定済みpeak数。
  - 空白を含むidが置換され、件数が記録される。衝突でエラーになる。
  - 0件の集合でBEDが書かれない。ファイル名が正規表現に一致する。
  - `export_peaks_as_bed`は無変更で既存テストが通る。

### D4 コマンド文と手順（A12、B Q1）
- genome名は`integration_settings["genome_build"]`（Level 1実行時に利用者が明示選択したもの）から取る。推測しない（I-2.2）。許可リスト`{"hg38": "hg38", "mm10": "mm10"}`（現在BRIMが対応するbuild）にあるときだけコマンドを出す。他のbuildでは、コマンドを出さずに理由を表示する。
- コマンド本文に挿入するのは、許可リスト由来のgenome名と、D3で生成したファイル名だけにする。利用者が入力した文字列は入れない。BRIMは実行しない（モジュールに`subprocess`、`os.system`、`socket`を入れない。ASTで検査する）。
- 表示するコマンド（設計書§12.4の文面。`-size 200`は「例」と明記する）:
  ```
  perl configureHomer.pl -install hg38
  findMotifsGenome.pl opened_peaks_padj0.05_lfc1.bed hg38 homer_opening/ -size 200 -bg all_peaks_background.bed
  findMotifsGenome.pl closed_peaks_padj0.05_lfc1.bed hg38 homer_closing/ -size 200 -bg all_peaks_background.bed
  ```
  - 1行目は「HOMERにgenomeが未導入の場合のみ。この手順は利用者がBRIMの外で実行し、データをダウンロードすることがある」と注記する。
- 平易な手順（英日）:
  1. 外部ツールを、そのツールの文書に従って用意する。HOMERはLinux/macOS系の環境向けなので、Windowsでは通常WSLなどが必要（詳細はツールの公式文書で確認する）。
  2. BEDファイルを1つのフォルダに置く。
  3. 上のコマンドを実行する。
  4. `homer_opening/knownResults.txt`などを、BRIMのLevel 3の取り込み欄で、opening/closingを選んで読み込む。
- 受入テスト:
  - hg38とmm10でコマンド文が期待どおり。許可リスト外のbuildでコマンドが出ない。
  - 空白・`;`・`$`を含む値が入る経路が存在しない（ファイル名の正規表現）。
  - 0件の集合のコマンドが出ない。
  - モジュールが`subprocess`、`os.system`、ネットワーク系を`import`しない。

### D5 importの紐付け・古さ・閾値照合（A4、A16、B Q4）
- importは、BED準備（§D11のsession記録`integration_motif_source`）が存在し、Level 2が現在の結果であるときだけ可能。各importは、必ずpeak集合（`opening`または`closing`）を利用者が選んで紐付ける。ファイルには集合情報が無いため。
- 紐付けの記録:
  - 「BRIMが今回生成したBEDを解析した」（既定）か、「別のファイルを解析した」か。
  - 別のファイルの場合は、利用者がATACのpadjとlog2FC閾値を入力する（プリフィルしない。プリフィルすると食い違いを検出できなくなるため）。
  - どちらの場合も、`peakset_fingerprint`（import時の現在値）と`genome_build`を記録する。
  - 外部ツールに使ったgenomeが`{build}`であることを利用者が確認するチェックボックスを必須とし、`genome_attested: true`を記録する。BRIMはgenomeを推測しない。
- 閾値照合は、入力した数値を`float`にして厳密に比較する（`abs_tol=1e-12`）。許容幅は設けない。食い違いは警告（処理は継続。設計書§7.6）で、`threshold_matches_current: false`と双方の値をmanifestに記録する。
- 古さ（stale）は警告ではなくブロックとする。`peakset_fingerprint`を、render時・export時・import時に現在の`atac_results`と`integration_settings`から再計算し、格納値と比較する。不一致ならmotif結果を消去して通知する（Phase 5 §5.1と同じ設計）。
- HOMERヘッダーの件数の照合（BRIMが生成したBEDを選んだ場合）: `n_target_sequences_reported`が対象peak数と異なれば、警告を出す。両方の数値を表示し、「ツールが配列を除外・統合した場合に差が出ることがある」と併記する。ブロックはしない。
- 受入テスト:
  - 閾値が0.05のところに0.01を入力すると警告が出て、importは成功し、manifestに不一致が記録される。同値なら警告なし。
  - fingerprintが不一致だとブロックされる（importが付かず、消去される）。
  - BED未準備、Level 2が現在でない、genome確認なし、別ファイルなのに閾値未入力の場合は、それぞれ拒否される。
  - ヘッダーの件数とBED件数が食い違うと警告が出る。

### D6 背景（A5、I-1.5）
- BRIMの背景は「検定済みの全peak」（D3の`background`）。件数と定義（英日）を、画面・README・manifestに記録する（I-1.5）。
- import時に、背景の設定を選択させる: `brim_all_tested_peaks`（既定）、`tool_default`、`other`（説明を入力）。
  - 既定以外を選んだ場合は`background_differs_from_brim: true`を記録し、画面に「このmotif結果はBRIMの背景とは別の背景との比較なので、別の結果として解釈してください」と表示する。
  - 既定を選んだ場合は、HOMERヘッダーの背景件数と`n_background`を照合し、食い違えば警告する。
- 受入テスト: 各選択肢でフラグが正しく記録される。既定以外で警告が出る。背景件数が定義どおり。

### D7 件数と置換（A6、B Q3）
- motifの結果は、遺伝子集合ごとではなくpeak集合ごとに持つ。1つのpeak集合につきactiveなimportは1つ（`opening`・`closing`で最大2つ）。同じ集合への再importはその集合の結果を置換し、`history`に追記する。opening/closingは常に別々に保持し、混在ファイルは拒否する。
- 1つのpeak集合に対する複数ツールの併存は、Phase 6では対応しない（記録して後続Phaseへ）。
- BED準備を再実行してfingerprintが変わった場合、既存のimportは新しい集合に紐付かないので消去する。fingerprintが同じなら維持する。
- 受入テスト: 再importで置換され、履歴が残る。opening/closingが独立に保たれる。fingerprint変更で消去される。

### D8 Level 2表との関係・`n_axes_supported`（A7、B）
- 保存済みのLevel 2表、`tf_candidates.csv`、`tf_level2`ブロックは変更しない。motifは別の表として保持し、結合は表示用に派生させる。
- `attach_motif_enrichment(tf_table, motif_tf_tables, peak_sets=("opening","closing"), alpha=0.05)`は、Level 2表のコピーに、各peak集合ごとの列を追加する:
  - `motif_{p}_status`、`motif_{p}_padj`、`motif_{p}_score`、`motif_{p}_n_motifs`、`motif_{p}_motif_name`、`motif_{p}_source_tool`
  - プレースホルダ列（`motif_enrichment_padj`、`motif_enrichment_score`、`motif_status`、`motif_source`）は、この派生ビューでは除く。「not_run」と値が同居しないようにするため。
  - 結合はLevel 2表の左結合で、行と順序は元の表のとおり。
- opening/closingは、Level 2の遺伝子集合（`concordant_activation`など）と自動で対応付けない。両方を並べて表示する。
- `n_axes_supported`と`n_axes_evaluable`にmotifを算入しない。理由: (a) peak集合と遺伝子集合の対応が一意でない、(b) motifのpadj閾値は外部ツール側の設定に依存する、(c) Level 2の契約とテストを維持する、(d) PLAN.mdの受入基準に含まれない。画面に「支持軸数はレベル2の3軸のみ。motifは別列」と明記する。統合スコアは作らない（I-1.3）。motifのpadjはBRIMで補正・再計算・結合しない。
- Level 2表に無いTFがmotif結果にだけある場合は、別表`motif_only_tfs`として表示する（motifはCollecTRIと独立に候補を挙げるため）。`motif_results.csv`には全TFを書く。
- 受入テスト:
  - 結合後もLevel 2の行数・順序・Level 2の各列が不変。保存済みのLevel 2表を変更しない。
  - opening/closingの列が別々に入る。
  - 禁止列名（`score`の合成、`confidence`、`combined`、`weighted`）が無い（既存のガードと同形式）。
  - `n_axes_supported`が結合の前後で同じ。
  - motif結果だけにあるTFが`motif_only_tfs`に出る。

### D9 状態語彙（A8）
- `motif_{p}_status`の値:
  - `not_run`: BED準備もimportもない（既存プレースホルダと同じ語。Level 2の表・CSVはこのまま）。
  - `not_imported`: BEDは準備済みで、そのpeak集合のimportがない。
  - `not_in_result`: importはあるが、そのTFの行がない（「ツールのmotifデータベースにその行がない」ことを意味し、「濃縮されていない」ではない）。
  - `no_padj_reported`: TFの行はあるが、報告されたpadjがない。
  - `reported_padj_le_alpha` / `reported_padj_gt_alpha`: ツールが報告したpadjと表示用の`alpha`（0.01/0.05/0.1、既定0.05）の比較。`alpha`は表示専用で、何も数えず、フィルタにも使わない。manifestに記録する。
- 「not enriched」という語は使わない。`not_tested`はmotifでは使わない（HOMERは未検定を報告しないため）。
- 受入テスト: 各状態が期待どおりに出る。`not enriched`を含む文言が無い。`alpha`の変更が他を変えない。

### D10 ファイル制限・検証（A11）
- サイズ ≤ 10 MB（`MAX_IMPORT_BYTES`）、行数 ≤ 20,000（`MAX_IMPORT_ROWS`）、0行はエラー。
- エンコーディングは`utf-8-sig`を試し、失敗したら`cp932`。それも失敗ならエラー。使った方式を記録する。
- 区切りは、ヘッダー行のタブの有無で決める（タブがあればタブ、なければカンマ）。どちらも無ければエラー。セミコロンは非対応（エラーで「CSVまたはTSVで保存し直してください」と案内する）。
- p/qは`[0,1]`。範囲外、非数値の文字列、`inf`は、ファイルの行番号付きの最大5件を挙げるエラー。空欄・`NA`・`NaN`は欠損として扱い（補完しない。I-1.1）、件数を報告する。
- 重複するmotif名は全行を保持し、D2の代表規則で扱う。
- ファイル名は表示・記録用にパス区切りを除去し、200文字までとする。
- 受入テスト: 各エラーの条件と、行番号を含むメッセージ。欠損が補完されない。エンコーディングとdelimiterが記録される。10MB超・20,000行超で拒否される。

### D11 session state・無効化（A16、A17、A20）
- 既存キーを流用する（新しいキーを増やさない）:
  - `integration_motif_source`: BED準備の記録（閾値、build、species、`counts`、各BEDのsha256、`peakset_fingerprint`、`generated_at`、`app_version`）。BED本文は保存せず、必要時に`atac_results`から再生成して、fingerprintで一致を確認する。
  - `integration_motif_results`: `{"imports": {peak_set: {...}}, "history": [...]}`。各importは、`record`（D5〜D10の記録）、`rows`、`symbol_map`、`tf_table`、`unmatched`を持つ。
- 無効化の表:

  | 変更 | motif | 実装 |
  |---|---|---|
  | Level 1再実行、§15.1の上流変更 | 消去 | 既存の`reset_tf_integration_results`（`integration_motif_*`を`None`にする）。加えて`tf_level3`をprovenanceから`pop` |
  | 正規化変更など`reset`を呼ばない経路 | 検出して消去 | render・export・importでのfingerprint再計算 |
  | TF Activity結果の変更（`invalidate_tf_level2_results`） | 消去 | Level 3はLevel 2を要する（I-6.2）ため、同関数に`integration_motif_*`の消去と`tf_level3`の`pop`を加える。ORAは触れない |
  | Level 2のボタン再実行（1つの遺伝子集合の置換） | 維持 | motifはLevel 2の出力に依存しないため。結合ビューを再計算する |
  | BED準備の再実行 | fingerprintが同じなら維持、違えばimportを消去 | D7 |
  | 同じpeak集合への再import | その集合のimportを置換 | 履歴は残る |
  | motifの`alpha`変更 | 何も消さない | 表示専用 |

- 新規`reset_motif_results()`（`integration_motif_source`・`integration_motif_results`・`tf_level3`のprovenance）を追加する。
- Level 3のブロックは、現在のLevel 2の結果が1つ以上（executed）あるときだけ表示する。Level 1・2の前には表示しない。
- 受入テスト: 上表の各行のうち、`reset`経路とfingerprint経路と`invalidate_tf_level2_results`をパラメータ化して確認する。Level 2の再実行でmotifが残ること。Level 1再実行で`integration_motif_*`と`tf_level3`が消えること。既存の`test_tf_integration_ui`の該当テストは、変更せずに通ること。

### D12 manifest・exportの構成（A9、A20、B）
- 別ブロック`tf_level3`を新設する（`integration_provenance["tf_level3"]`と`settings["integration"]["tf_level3"]`）。`tf_level2`の意味・fingerprint・テスト契約は変えない。`tf_level2.motif_axis == "not_run"`はLevel 2の実行に関する記述として固定し、Level 3の状態は`tf_level3.status`（`bed_prepared_no_import` / `imported`）に置く。`tf_level3`に「`tf_level2.motif_axis`はLevel 2の実行のみを記述する」旨の注記を入れる。
- `tf_level3`の内容:
  - `peak_sets`: 閾値、genome build、species、座標規約（0-based half-open）、背景の定義（英日）、`counts`、BED名とsha256、`peakset_fingerprint`、`n_ids_sanitized`。
  - `imports`: peak集合ごとに、`import_id`（ファイルsha256・peak集合・fingerprintのハッシュ）、`tool`、`tool_version`（未入力は`"not provided"`）、`motif_database`（同）、ソースファイル（名前・sha256・サイズ・エンコーディング・区切り）、列対応、紐付け方式、申告した閾値と`threshold_matches_current`、背景の申告と`background_differs_from_brim`、`genome_attested`、件数（行・欠損・照合）、未照合の件数と先頭50件、警告。
  - 規則: `symbol_normalization_rule`、`representative_motif_rule`、`motif_alpha_display`、`n_axes_note`、`independence_note`。
  - 共通: `limitations_text`（英日）、`external_services_used: []`、`external_tool_executed_by_brim: false`、`exported_files_sha256`。
  - 履歴: `import_history`。
- 単一の源: ブロックは`integration_motif_source`と`integration_motif_results`だけから組み立てる（`build_motif_summary`）。無効化のたびに`pop`する。
- exportは、現在のLevel 2があり、fingerprintが現在と一致するときだけ、次を出力に含める:
  - `MotifAnalysis/`（BED3種 + `motif_analysis_README.txt`）: BED準備をした場合だけ（利用者が生成しないと出ない）。
  - `Integration/motif_results.csv`（peak集合 × TF、代表行、`n_motifs_for_tf`など）。
  - `Integration/motif_symbol_map.csv`（motif行 → TF、照合状態、理由。未照合を含む）。
  - `Integration/tf_candidates_with_motif.csv`（Level 2表とD8の結合ビュー、`gene_set`列付き）。
  - `Integration/motif_import_record.json`（`tf_level3`の`imports`と履歴）。
- 既存の`Integration/tf_candidates.csv`は変更しない（`motif_status = not_run`のまま）。
- 受入テスト: importの後だけ上記ファイルがZIPに入る。BED準備だけなら`MotifAnalysis/`のみ。古いmotif結果はZIPにも`tf_level3`にも入らない。sha256が本文と一致する。`tf_level2`ブロックは既存テストの値のまま。`external_services_used == []`。

### D13 限界の文言・文書の内容規則・ライセンス（A14、A15、B Q8）
- `LIMITATIONS_EN / _JA`（Level 3、画面に常時表示し、manifestにも複写する）:
  - EN:
    - Motif results are imported from an external tool that BRIM did not run and cannot verify; BRIM records the conditions you entered.
    - A motif match shows that a binding sequence is present in the peaks; it does not show that the TF binds there.
    - The motif p-value and padj are a separate test computed by the external tool from ATAC peaks and a background; they are independent of RNA, ATAC and Level 2 padj, and BRIM does not correct, recompute or combine them.
    - Opening and closing peak sets are analyzed separately and are not paired with Level 2 gene sets automatically.
    - The motif axis and the CollecTRI-based target enrichment may be less independent than they appear, because curated regulatory databases partly rest on experiments near promoters.
    - Motif names that could not be matched to a gene symbol (families, aliases, complexes) are listed and not used; a TF that is not in the result is "not in result", not "not enriched".
    - When one TF has several motifs, BRIM shows the motif with the smallest reported padj; all rows are kept in the export, and choosing the best of several is not corrected.
    - Candidates are hypotheses, not evidence that a TF regulates genes or drives a phenotype. The supported-axis count is a sorting aid, not a statistic, and it does not include the motif axis.
  - JA: 上記に対応する8文（外部ツールでBRIMは実行・検証しない／配列の存在であって結合を示さない／motifのp・padjは外部ツールの独立した検定でBRIMは補正・再計算・結合しない／opening・closingは別々で遺伝子集合と自動対応しない／motifとCollecTRI由来の標的濃縮は見かけほど独立でない可能性／未照合motifは一覧のみで、結果に無いTFは「結果に無い」であり「濃縮なし」ではない／複数motifは最小padjの1つを表示し全行は出力に残る／候補は仮説で因果を示さず、支持軸数は並べ替えの補助でmotifを含まない）。実装時に上のENと同じ意味の完全な日本語文を`brim_motif_import.py`に書く。
- Level 2の画面（限界の一覧とキャプション）にある「Motif enrichment: not run」の文は、motifのimportが現在あるときだけ表示から除く（`level2_limitations_for_display`）。Level 2の定数と`tf_level2`ブロックは変更しない。
- 因果表現: 「regulates」「drives」「causes」を使わない。英日の文言をregexで検査する（Phase 5のガードと同形式）。
- `docs/motif_analysis_guide.md`の内容規則（B Q8）:
  - 書けるのは、(a) BRIMの挙動でテストされているもの（ファイル名、BRIMが読む列、BRIMが記録する項目）、(b) コードが生成するコマンド文と同一の文（テストでコマンド文の一致を確認する）、(c) 外部ツールについては、名前、「別のソフトウェアであり、固有のライセンス・条件がある。BRIMは同梱も実行もしない」、「インストール・版・出力形式・引用の方法はそのツールの公式文書で確認する」だけ。
  - 書かないもの: ツールの内部動作、性能、閾値の推奨、引用文の再掲、出典なしの外部主張。URLは、未確認のため載せない。
- 受入テスト: ガイドのコマンド節が生成関数の出力と一致する。ガイドと画面に因果動詞が無い。

### D14 UI・既存テストとの関係・AppTestの数（B）
- `_render_integration_ui`の1行ステージ表示を「Level 3 (motif): after Level 2, import results from an external tool」（英日）に更新する。Level 2のキャプションの「（Level 3はこの版では利用できません）」も更新する。
- 上記の更新により、`tests/test_tf_integration_ui.py`の次の2つのアサーションは、Level 3が提供される事実と矛盾する。
  - 244行目: `"Level 3 (motif) is not available in this version"`。
  - 393行目: `"レベル3（motif）はこの版では利用できません"`。
  - これらを新しい文言の同等の確認に置き換える（削除・緩和はしない）。**A・Bは計画レビューで、この置換が「有効なテストの弱体化」に当たらないことを明示的に確認する。**どちらかが異議を出した場合は、ここで止めてユーザーの判断を仰ぐ。
  - 251行目（Level 1後にキーに`motif`を含むボタンが無い）は、Level 3がLevel 2の実行後にしか出ないため無変更で通る。Level 3のボタンのキーは`tf_level3_*`とする。通らなければ止めてCへ差し戻す。
- Level 3ブロックの構成:
  1. 「motif解析用ファイルを準備」ボタン（BED準備、件数、警告、コマンドと手順の表示、`MotifAnalysis.zip`のダウンロード）。
  2. 取り込み欄（peak集合、ツール、ファイル、汎用のときの列対応、紐付け方式、背景、genome確認、版とデータベース）。
  3. 取り込み結果（照合件数、未照合の一覧、警告、peak集合ごとのmotif列付きのLevel 2表、`motif_only_tfs`、限界の説明）。
- AppTestは最大3件（CIが遅いため）。論理はモジュールの単体テストに置く。
  - AppTestが`file_uploader`を扱えない場合は、モジュールで作った結果をsession stateに置いて描画を確認する（アップロードの取り込み部分は、取り込みハンドラを薄いラッパにして、モジュールの単体テストで担保する）。
  - (a) ゲート（Level 2前は非表示、Level 2後に表示）とBED準備の表示。
  - (b) 取り込み済み状態のmotif列表示と古さ検出。
  - (c) exportの内容（ビルド関数を直接呼ぶ）。
  - AppTestのタイムアウトは既存の300秒の定数と同じにする。

### D15 テスト入力（fixture）
- 新規`tests/motif_support.py`（決定的、seedなし）:
  - `synthetic_dar_table()`: 約300 peak。opening、closing、非有意、`padj_is_na`、`lfc_is_na`の各行と、境界行（`padj = 0.05`ちょうど、`padj = 0.0500001`、`lfc = 1.0`ちょうど、`lfc = 0.9999`、`lfc = 0`で有意）を明示的に構築する。空白を含むpeak_idを1件含める。別の小さな表で、id衝突を作る。
  - `homer_known_text(...)`: HOMER風の合成表。複合名（`Stat3(Stat)/…`、`Oct4:Sox17(POU,Homeobox/HMG)/…`、`AP-1(bZIP)/…`）、大小文字違い、同じTFの重複、非数値のp（拒否用）、欠損、未照合motif。名前は本物のツール出力の転載ではなく、テスト用に作った文字列であることを明記する。
  - `generic_motif_csv(...)`: 汎用のCSV（列違い、セミコロン、範囲外のp、`inf`、`peak_set`混在、cp932、サイズ超過・行数超過の生成）。
  - 閾値の食い違い（開いたpeak集合に0.01を申告）と、古い`peakset_fingerprint`の作り方。
- AppTest用: Phase 5の`synthetic_tf_level1_state`と`run_level2`の出力を再利用し、仕込みTF（fixtureの`planted_tf`）の名前で`f"{planted}(Zf)/Fixture/Homer"`のようなmotif行を作る（Level 2表の行に結合される）。`atac_results`には`synthetic_dar_table`を置き、閾値を`integration_settings`と一致させる。既存の`tests/test_tf_integration_ui.py`と`tests/tf_support.py`は変更しない（ヘルパーはimportまたは最小限の複製）。
- 受入テスト: fixtureの決定性（2回生成して同一）。

### D16 CI・gateの扱い（B Q5、Q6、Q7）
- Q7: `cd0a2a4`はPLAN.mdとphase5計画書だけの変更で、コードは`9f4142d`と同一。着手記録の前に、主エージェントが`cd0a2a4`のCI結果をGitHubで確認してPLAN.mdに書く。確認できない場合は、最新の確認済みrun `35522031618`（`9f4142d`、4環境成功）を根拠とし、その旨を記録する。
- Q6: 着手前の追加の安定性確認は必須としない。`dabbb32`のWindows/3.12失敗の原因は不明の残余リスクとして受け入れ、次を設ける。
  - Phase 6のAppTestは3件までにする。
  - CIチェックポイント1（§5のステップ3の後）と最終commitで、Windows/3.12ジョブを1回再実行して結果を記録する。
  - Phase 6のcommitでWindowsのジョブが失敗した場合は、原因を特定するまで次のステップへ進まない（サイン済みのログ確認は、ユーザーまたは主エージェントが可能なら行う）。
- Q5（gate文言）: 本書へのA・B GO後、Cが次の趣旨をPLAN.mdに記録する。「Phase 6の計画（`docs/phase6_implementation_plan.md`）にA・BがGOを発行し、監査役Cが着手を決定した（日付）。実装の完了は、実装後監査でA・Bの両GO（コード・テスト・export・provenance・CI）を得るまで宣言しない。Phase 7へは進まない。」

## 3. ファイル別の実装計画

**新規: `brim_motif_import.py`**（streamlit、session_state、ネットワーク、subprocessを使わない。`hashlib`、`io`、`json`、`re`、`math`、`dataclasses`、`numpy`、`pandas`と、`brim_tf_integration.fold_symbol`をimportする）:
```python
class MotifImportError(ValueError): ...
PEAK_SETS = ("opening", "closing")
MAX_IMPORT_BYTES = 10 * 1024 * 1024
MAX_IMPORT_ROWS = 20_000
SMALL_PEAK_SET_WARNING = 100
HOMER_GENOMES = {"hg38": "hg38", "mm10": "mm10"}

@dataclass(frozen=True)
class PeakSets: opening, closing, background (DataFrame chrom/start/end/name),
                counts: dict, thresholds: dict, genome_build: str, species: str,
                fingerprints: dict, peakset_fingerprint: str

def build_peak_sets(dar, thresholds, genome_build, species) -> PeakSets
def peak_set_bed_text(peak_sets, which) -> str                  # which: opening/closing/background
def bed_file_name(kind, thresholds) -> str
def build_homer_commands(peak_sets) -> list[str]                # genome許可リスト外は空
def render_motif_readme(peak_sets, app_version, generated_at) -> str   # 英日、純関数
def build_motif_bundle(peak_sets, app_version, generated_at) -> dict[str, str]   # "MotifAnalysis/…" -> 本文
def suggest_column_map(columns) -> dict[str, str]
def read_motif_results(data: bytes, file_name: str, tool: str, column_map=None) -> MotifTable
def extract_tf_symbols(motif_name) -> list[str]
def normalize_tf_symbols(motif_rows, reference_symbols) -> tuple[pd.DataFrame, dict]  # symbol_map, UnmatchedReport
def summarize_motif_by_tf(symbol_map) -> pd.DataFrame           # 代表行と n_motifs_for_tf
def compare_thresholds(declared, current) -> dict
def import_motif_result(table, peak_sets, peak_set, declaration, reference_symbols, imported_at) -> dict
def attach_motif_enrichment(tf_table, motif_tf_tables, peak_sets=PEAK_SETS, alpha=0.05,
                            source_prepared=False) -> pd.DataFrame
def motif_only_tfs(tf_table, motif_tf_table) -> pd.DataFrame
def build_motif_summary(source, imports) -> dict                # tf_level3
def build_motif_export_files(peak_sets, source, imports, tf_runs) -> dict[str, str]
def level2_limitations_for_display(limits, motif_present) -> list[str]
LIMITATIONS_EN / _JA, BACKGROUND_DEFINITION / _JA, SYMBOL_RULE_TEXT / _JA, STATUS_* 定数
```
（設計書§14.3の`read_motif_results`・`normalize_tf_symbols`・`attach_motif_enrichment`の署名は、上記に更新する。）

**変更: `Bulk_RNAseq_Analyzer.py`**
- `reset_motif_results()`の新設。`invalidate_tf_level2_results()`に`integration_motif_*`の消去と`tf_level3`の`pop`を追加。`reset_tf_integration_results()`に`tf_level3`の`pop`を追加。
- `_current_motif_state(genes)`（fingerprint照合で古さを検出。session stateは変更しない）、`_render_tf_level3_ui(genes, edges, lang)`、Level 2表示関数へのmotif結合ビューの追加（既定の引数で従来の表示を保つ）。2箇所の`_render_tf_level2_ui`呼び出しを、Level 2 → Level 3の順に描く共通の呼び出しに置き換える。ステージ表示とキャプションの文言更新。
- 共有export生成（1151〜1195行付近）: `build_motif_export_files`の結果を`files`に足し、`exported_provenance["tf_level3"]`を追加する。Level 2が現在の場合だけ。ロジックはモジュール側に置く。
- 論理は追加せず、`atac_results`、`integration_settings`、`deg_results`、CollecTRIのsourceを読んで、モジュールの関数へ渡すだけにする。

**文書:** PLAN.md（Phase 6の節と現在地。着手決定の記録）、設計書§12.4（列の対応、背景、生成物の名前、`tf_level3`）・§14.3・§16.2、ARCHITECTURE（モジュール、境界）、README、CHANGELOG、`docs/motif_analysis_guide.md`、本書。READMEとCHANGELOGは実装済みの挙動のみを書き、実際のツール出力での検証を示唆しない。

**テスト:** 新規`tests/motif_support.py`、`tests/test_motif_import.py`（ステップ1〜3のモジュール単体）、`tests/test_motif_import_ui.py`（AppTest最大3件とsession state・export）。既存テストは、D14の2つのアサーション（置換）を除き変更しない。

## 4. 受入基準（PLAN.md Phase 6との対応）

| PLAN.mdの基準 | 対応する決定・テスト |
|---|---|
| BED出力が閾値どおりのpeakを含む | D3（境界・NA除外・`export_peaks_as_bed`との一致） |
| opening/closingが別ファイルに分離される | D3（別名のBED）、D7 |
| HOMER複合名からシンボルを抽出できる | D2 |
| 照合できなかったmotifの件数と一覧が表示される | D2（画面・CSV・manifest） |
| opening/closingが`peak_set`列で区別される | D8、D12（`motif_results.csv`） |
| motif未実行時に該当列がNAとなり他軸が影響を受けない | D8、D9（`not_run`、Level 2表・CSVが不変） |
| 入力閾値が現在のBRIM設定と食い違う場合に警告する | D5（厳密比較・警告・記録） |
| motif列が埋まる（完了条件） | D8（結合ビュー） |
| 出典と解析条件がmanifestに残る | D12（`tf_level3`） |
| motif結果のimport、3軸テーブルへの結合、unit tests、ガイド | D1〜D8、D15、D13 |

- 検証は決定的な合成データで行う。実際のツール出力・実データでの検証ではなく、生物学的妥当性を示すものではない。READMEとCHANGELOGも同じ範囲にとどめる（Phase 5 §5.6と同じ扱い）。
- 不変条件の確認: I-1.1（NA非補完、未検定peak除外）、I-1.2・I-1.3（合成スコアなし、motifのpadjを結合しない）、I-1.5（背景の記録）、I-1.6（motifのpadjは独立した新しい検定）、I-2.2（座標規約・id置換の記録、buildは利用者選択）、I-3.1（モジュールはStreamlit非依存）、I-3.2（閾値は引数）、I-4（ネットワーク・外部バイナリなし）、I-5.1（scanなし）、I-5.4（因果語なし）、I-6.1（RNA-only不変）、I-6.2（Level 3はLevel 2を要し、上流変更で消去）。

## 5. 実装順序

各ステップは小さく、自分のテストを持つ。ローカルでは、Phase 6関連のファイルとIntegration関連のファイルを実行する（既知のPyDESeq2の9件のローカル失敗は想定内）。現ステップのテストが通るまで次へ進まない。

0. **着手前。** A・Bが本書にGO。CがPLAN.mdに着手決定とD16のCI記録を書く。
1. **peak集合とBED。** `tests/motif_support.py`（`synthetic_dar_table`）、`build_peak_sets`、`peak_set_bed_text`、`bed_file_name`、fingerprint、`build_homer_commands`、README生成、`build_motif_bundle`。D3・D4のテスト。
2. **結果の読込み。** `suggest_column_map`、`read_motif_results`（HOMER、汎用）、検証（D10）、HOMER風fixture。D1・D10のテスト。
3. **正規化・結合・要約。** `extract_tf_symbols`、`normalize_tf_symbols`、`summarize_motif_by_tf`、`compare_thresholds`、`import_motif_result`、`attach_motif_enrichment`、`motif_only_tfs`、`build_motif_summary`、`build_motif_export_files`、`level2_limitations_for_display`、禁止列名・因果語・AST（streamlit、ネットワーク、subprocessなし）のガード。D2・D5〜D9・D12・D13のテスト。**CIチェックポイント1**: ここでpushし、4環境（Ubuntu/Windows × 3.11/3.12）を確認する。Windows/3.12を1回再実行して記録する。
4. **session state・無効化。** `reset_motif_results`、`invalidate_tf_level2_results`と`reset_tf_integration_results`の追加、`_current_motif_state`。D11の表をパラメータ化して確認する（既存の関数ローダ方式。AppTestは使わない）。既存テストが失敗すれば、テストを変更せず止めてCへ差し戻す。
5. **UI。** ステージ表示、Level 3ブロック、Level 2表へのmotif列、D14の2つのアサーションの置換、AppTest最大2件。英日文言の因果語検査。
6. **export・manifest。** 共有exportへの追加、`tf_level3`、古い場合の除外、ZIPの内容（D12）。AppTest（c）またはビルド関数の直接テスト。**CIチェックポイント2**（推奨）。
7. **文書。** D13に従う。設計書、ARCHITECTURE、README、CHANGELOG、`docs/motif_analysis_guide.md`、PLAN.md。ガイドのコマンド節と生成関数の出力の一致テスト。
8. **検証と実装後監査。** ローカルでPhase 6関連を実行し、pushして4環境のCIを確認する（Windows/3.12は1回再実行して記録）。その後、監査役A（設計・科学的整合性・不変条件・文書）とB（コード・テスト・export・provenance・CI）に監査させる。両者GOの後でのみPhase 6を完了とする。

**Cへ差し戻す条件:** 既存テストがD14の2つ以外で失敗する。Level 2の契約（`tf_candidates.csv`、`tf_level2`、`n_axes_*`）の変更が必要になる。I-1〜I-6、RNA-only workflow、ORAの計算・export命名に触れる。範囲・依存関係・ネットワーク通信・外部プロセスの追加が必要になる。有効なテストの弱体化・skip・削除が必要になる。実際のHOMER出力と想定ヘッダーが異なると判明した（再設計せず、認識規則の修正案をCへ）。AppTestが`file_uploader`を扱えず、取り込み部分の担保方法を変える必要がある。Windowsのジョブが失敗する。

**CIの確認点:** ステップ3の後（必須）、ステップ6の後（推奨）、A・B監査の前の最終commit（必須）。

## 6. OPEN QUESTIONS

- **ユーザーへ（非ブロッカー、2件）:**
  1. Phase 5から継続: AGENTS.md I-1.5に設計書§12.2の明確化への参照を1行追記するか。監査役Cは不変条件本文を編集できない。
  2. 実際のHOMER `knownResults.txt`（ユーザー自身のデータからの出力）を、Phase 7の検証で使わせてもらえるか。手元にあれば、想定ヘッダーとの一致確認だけに使う。無い場合、Phase 6は合成データによる検証にとどまる。
- **A・Bへ（本書のGOの条件）:** D14の2つのアサーション置換が「有効なテストの弱体化」に当たらないことの確認。D8で`n_axes_supported`にmotifを算入しない判断（設計書§8.5・§12.1の4軸表現との差）が、I-1.3の範囲内であることの確認。

**Cが決定しなかったこと（権限外・実施不能）:**
- AGENTS.mdの不変条件本文の編集（ユーザー）。
- 実際のHOMER出力・公式仕様との突合、HOMERの引用文・ライセンス条項の記載（外部の一次情報が必要で、ネットワーク・実データを使えないため。文書は主張を避ける設計とした）。
- 実データでの生物学的検証（Phase 7）。
- `dabbb32`のWindows/3.12失敗の原因特定（サイン済みのログが必要）。受け入れとチェックポイントの再実行で扱う。
- D14の置換アサーションの承認（A・Bの確認事項）と、`cd0a2a4`のCI結果の確認（主エージェントまたはユーザー。私の環境では`gh`が使えなかった）。
- 範囲外として後続Phaseに送ったもの: 複数ツールの併存、TF別名・family辞書、motifを`n_axes_supported`へ算入すること、de novo結果、HOMER以外の専用パーサ。

参照した既存の事実（決定の根拠）: `brim_atac.export_peaks_as_bed`の`"all"`は検定されなかったpeakも出し、`tests/test_atac.py:528`がそれを固定している。`_current_tf_level2_runs`と`invalidate_tf_level2_results`、`reset_tf_integration_results`は`Bulk_RNAseq_Analyzer.py`の317〜340行と2056行付近にある。`tests/test_tf_integration_ui.py`の244・251・393行がLevel 3の不在を確認している。genome buildはhg38とmm10だけが選べる（1677行）。

## 7. 計画レビュー後の監査役C決定（2026-09-21）— 本節が旧記述に優先する

監査役A（条件付きGO、E1〜E6）とB（条件付きNO-GO、6点）の指摘を、監査役Cが次のとおり決定した。AとBはともに、D14
（既存アサーション2件の置換）を「有効なテストの弱体化」に当たらないと明示的に判断した（実装前にユーザーの判断を
仰ぐ必要はない）。不変条件には触れていない。

### 7.1 D3追記: 染色体名の表記（A-E5）
- `counts`に`chrom_style`（`{"chr_prefixed": n, "bare": n}`）を記録する。`chr`で始まる染色体名と、始まらない染色体名が
  両方あれば警告を出す（「染色体名の表記が混在しています。外部ツールのgenomeと表記が合うか確認してください」）。
  染色体名の変換はしない（I-2.2）。`chrom_style`は`peakset_fingerprint`に含める。
- 受入テスト: 表記混在で警告が出て件数が記録される。変換されない。

### 7.2 D4置換・追記: genome build、閾値の出所、README（A-minor）
- genome名は`integration_settings["genome_build"]`から取る。これはATACのアノテーション/マッピングのbuildで、speciesに応じて
  事前選択された値（hg38/mm10）を利用者が確認したものであり、「Level 1で明示選択した」とは書かない。推測しない。
- Level 3のpeak集合の閾値は、`integration_settings["thresholds"]`の`atac_padj`・`atac_lfc`（Level 1の値）を使う。ATAC検証
  レポートに閾値が無く既定の0.05/1.0が使われた場合も同じで、その旨を`motif_analysis_README.txt`に書く。
- `motif_analysis_README.txt`の必須文言: 「`tf_candidates.csv`は設計上`motif_status=not_run`のままです。motif列付きの
  ビューは`tf_candidates_with_motif.csv`です。」

### 7.3 D5置換: 古さの規則を1つにする（A-E3、A-E4、A-E1）
- 現在のLevel 2結果が無い、または`peakset_fingerprint`が現在値と一致しないmotif結果は、**どこでも不在として扱う**。
  - render: Level 2と同じく消去する（`reset_motif_results()`を呼び、通知を出す）。
  - export: 状態を変更せず、出力に含めない。
  - import: 状態を変更せず、拒否する（次のrenderで消去される）。
  - Level 3のrenderとexportは、それぞれ自分で`_current_tf_level2_runs`を呼ぶ。`invalidate_tf_level2_results`が
    Level 2のrenderでしか動かないことに依存しない。
- genome確認の文言は「peak座標と外部ツールのgenomeが同じbuild（`{build}`）である」とする。
- `peakset_fingerprint`の再計算は、fingerprint入力の要約をキーにsession内でキャッシュしてよい（任意）。
- 閾値の不一致、または既定以外の背景の場合、該当peak集合のmotif列の見出しにバッジ（「閾値不一致」「別の背景」）を表示する。

### 7.4 D8追記: 結合のキー、ヘテロダイマー表示、記録列（B-4、A-E6、A-E1）
- Level 2表との結合のキーは`fold_symbol(tf_symbol)`（strip + casefold）とする。綴りの一致には依存しない。
- `motif_{p}_motif_name`は代表motifの全名（例: `Oct4:Sox17(POU,Homeobox/HMG)/…`）を出す。`motif_{p}_match_status`
  （`matched` / `partially_matched`）と`motif_{p}_form`（`single` / `heterodimer`）を加え、画面に表示する。
- `motif_{p}_threshold_matches_current`と`motif_{p}_background_differs_from_brim`を、`tf_candidates_with_motif.csv`と
  `motif_results.csv`にも列として持つ（manifestだけにしない）。
- 受入テスト: ヘテロダイマー・部分照合が画面の列に出る。綴りが違っても`fold_symbol`で結合される。

### 7.5 D11置換: 無効化はインラインで行う（B-1、A-E3）
- `invalidate_tf_level2_results()`は、`integration_tf_results = None`に加えて、`integration_motif_source`と
  `integration_motif_results`を`None`にし、`integration_provenance`から`tf_level2`と`tf_level3`を**関数内で直接**除く。
  `reset_motif_results()`は呼ばない。理由: 既存の関数ローダテストは`_RESET_FUNCTIONS`の関数だけを実行し、未定義名を参照すると
  `NameError`になるため。`tests/test_tf_integration_ui.py`の`_RESET_FUNCTIONS`は変更しない（既存テストの変更は不要）。
- `reset_tf_integration_results()`も、`tf_level3`の除去をインラインで行う。
- `reset_motif_results()`は、renderの古さ検出（§7.3）専用とし、上記2関数から呼ばない。
- D11の表の「古さ」の行は「renderで消去、exportとimportは変更しない」とする。

### 7.6 D12追記: exportの古いコピー、NaN、記録列（B-3、A-E1）
- exportは、`exported_provenance`から保存済みの`tf_level3`も除いた上で、現在のLevel 2とfingerprintが一致するときだけ、
  現在の値から再構築して入れる。古いコピーが`analysis_notebook.md`やmanifestに漏れない。
- `tf_level3`はNaNを含まない（欠損は`null`。JSONは`allow_nan=False`で書く）。
- `motif_results.csv`と`tf_candidates_with_motif.csv`は、`threshold_matches_current`と`background_differs_from_brim`の列を持つ。

### 7.7 D13置換: 因果語の除去とガード（A-E2）
- EN: 「Candidates are hypotheses; they do not show a functional effect of the TF on genes or on a phenotype.」
- JA: 「候補は仮説であり、TFが遺伝子や表現型に機能的な影響を与えることを示すものではありません。」
- ガード規則: Level 3の英日の文言（定数と画面文字列）に、次の語を含まない。EN: `regulates?`、`drives?`、`driven`、
  `causes?`、`caused`、`causing`。JA: `制御する`、`引き起こ`、`原因`。テストは正規表現で全文を検査する。Level 2の既存定数は変更しない。

### 7.8 D14置換: 既存アサーションの置換の条件、AppTestとアップロード（B-2、B-5）
- 2つのアサーションの置換は、A・Bが明示的に承認済み（弱体化に当たらない）。条件: (1) 置換するのは
  `tests/test_tf_integration_ui.py`の244行目と393行目の2つだけ。(2) 新しい英日の文言に「after Level 2」「Level 2の後」を含める。
  (3) 251行目、254〜264行目、435〜447行目は無変更。(4) Level 2のキャプションを書き換えても、motifのimportが無いときは
  `Motif enrichment: not run`の部分文字列を保つ（264行目）。(5) Level 2の前にLevel 3ブロックが描画されないことを確認する
  アサーションを1つ追加する。
- `file_uploader`は、インストール済みのStreamlit 1.50.0のAppTestで扱えない（確定事実。requirementsは1.39以上を許す）。
  設計は、薄いアップロードハンドラ + 純粋な`read_motif_results(bytes, …)`・`import_motif_result`の単体テスト + session stateの注入
  である。この事実は停止条件ではない（§5の該当条件は削除）。
- AppTestは3件以内。`build_motif_export_files`のテストは、モジュール単体（BEDのみ、古いものの除外、sha256）で行う。exportの
  ビルド関数はアプリのスクリプト内にあり、importできない。wiringは、`capture_downloads`を使うAppTest 1件と、session stateを
  変更して2回目の`app.run()`を行う古い結果のケースで確認する（合計3件）。AppTestは`tests/test_tf_integration_ui.py`の
  `_level1_snapshot`と`_tf_app`をimportして再利用し、遅い実際のLevel 1実行を複製しない。

### 7.9 §5への追記（B-2、B-6）
- ステップ8に**手動確認**を加える: 実行中のアプリで、実際のアップロード経路を確認する（合成ファイルをimportする）。結果を
  証拠として記録し、CIではない手動確認と明記する。
- ステップ7に、`ARCHITECTURE.md`の64行目付近、177〜187行目、254〜256行目の修正を加える（`read_motif_results`・
  `normalize_tf_symbols`・`attach_motif_enrichment`を`brim_tf_integration.py`の古い署名で書いている箇所を、
  `brim_motif_import.py`と本書の署名に直す）。
- HOMERのパーサは、実際の出力に対して**未検証**である（ヘッダーの綴りは記憶に基づく）。READMEとCHANGELOGにその旨を書く。
- §3の`Bulk_RNAseq_Analyzer.py`の項: `invalidate_tf_level2_results`と`reset_tf_integration_results`にmotif消去と`tf_level3`
  除去をインラインで追加する。`reset_motif_results()`はrenderの古さ検出専用。Level 3のrenderとexportはそれぞれ
  `_current_tf_level2_runs`を呼ぶ。