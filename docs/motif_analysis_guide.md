# motif解析（レベル3）の手順

BRIMは、motifのスキャンも外部ツールの実行もしません。BRIMがするのは、(1) 外部ツールに渡すBEDファイルを書き出すこと、
(2) BRIMの外で実行するコマンドを表示すること、(3) 得られた結果を取り込んで、レベル2の表に別の列として並べることです。

この文書に書くのは、BRIMの挙動と、BRIMが生成するコマンド文だけです。外部ツールについては、次の3点以外は書きません。
- 外部ツールは別のソフトウェアで、固有のライセンス・利用条件があります。BRIMはそれらを同梱も実行もしません。
- インストール、版、出力形式、引用の方法は、そのツールの公式文書で確認してください。
- 実際のツール出力に対するBRIMの読み取り規則は未検証です（下記「未検証の点」）。

## 前提

- レベル1を実行し、続けてレベル2を実行してあること（レベル3は、現在のレベル2の結果があるときだけ表示されます）。
- ATACの結果（DAR）と、ATACの注釈設定で選んだgenome build（hg38またはmm10）があること。

## 手順

1. Integrationタブのレベル3で「motif解析用ファイルを準備」を押します。次のファイルが作られ、ZIPでダウンロードできます。
   - `opened_peaks_padj<閾値>_lfc<閾値>.bed`: opening（開いた）peak。
   - `closed_peaks_padj<閾値>_lfc<閾値>.bed`: closing（閉じた）peak。
   - `all_peaks_background.bed`: 背景（検定済みの全peak。未検定のpeakは含めません）。
   - `motif_analysis_README.txt`: 閾値、件数、genome build、座標系、コマンド、生成日時。
   閾値は、レベル1の統合を実行したときのATAC閾値です。0件のpeak集合のファイルとコマンドは作られません。
2. 外部ツールを、そのツールの公式文書に従って用意します。
3. BEDファイルを1つのフォルダに置き、BRIMが表示するコマンドをBRIMの外で実行します（`-size 200`は例です）。
4. 得られた結果ファイル（例: `homer_opening/knownResults.txt`）を、BRIMの取り込み欄で、openingまたはclosingを選んで取り込みます。
   BRIMがBEDを書き出したファイルを解析した場合は「BRIMが書き出したBEDファイル」を、別のファイルを解析した場合は、その閾値を
   入力してください（自動では埋めません。食い違いを検出できるようにするためです）。

## BRIMが表示するコマンド（genome buildがhg38またはmm10のとき）

BRIMは、許可リストにあるgenome名（hg38、mm10）と、BRIMが生成したファイル名だけをコマンドに入れます。次の例はhg38、
閾値が既定（padj 0.05、|log2FC| 1）の場合です。

```text
perl configureHomer.pl -install hg38
findMotifsGenome.pl opened_peaks_padj0.05_lfc1.bed hg38 homer_opening/ -size 200 -bg all_peaks_background.bed
findMotifsGenome.pl closed_peaks_padj0.05_lfc1.bed hg38 homer_closing/ -size 200 -bg all_peaks_background.bed
```

1行目は、genomeが未導入の場合だけ必要です。この操作は利用者がBRIMの外で行い、データをダウンロードすることがあります。
対応OSとセットアップ要件は、外部ツールの公式文書で確認してください。

## 取り込みで記録される内容

- どのpeak集合（opening / closing）の結果か、ツール名・版・データベース（入力した場合）、ファイル名とsha256。
- 使った閾値と、現在のBRIMの閾値との一致・不一致（不一致は警告し、記録します。取り込みは続けます）。
- 背景の選択（BRIMの検定済み全peak背景 / ツール既定 / その他）。BRIMの背景でない場合は、別の結果として扱われます。
- peak座標と外部ツールのgenomeが同じbuildであるという利用者の確認。
- motif名からTFシンボルを読み取り、遺伝子シンボルと大文字小文字を無視して照合した結果。照合できなかったmotif名の件数と一覧。
  別名・旧シンボル・ファミリーは解決せず、未照合として表示します。

## 表の読み方

- レベル2の表の`motif_opening_*`・`motif_closing_*`列に、取り込んだ結果が並びます。支持軸数にはmotifを数えません。
- 状態: `reported_padj_le_alpha` / `reported_padj_gt_alpha`（ツールが報告したpadjと、表示用の閾値の比較。何も数えず、
  フィルタにも使いません）、`no_padj_reported`、`not_in_result`（その結果にそのTFがない。「濃縮されなかった」ではありません）、
  `not_imported`（BEDは準備済みで、そのpeak集合の取り込みがない）、`not_run`（準備も取り込みもない）。
- 1つのTFに複数のmotifがある場合は、報告padjが最小のmotifを表示します。全行は`motif_symbol_map.csv`に残ります。
- `Integration/tf_candidates.csv`は、設計上`motif_status=not_run`のままです。motif列付きの表は
  `Integration/tf_candidates_with_motif.csv`です。

## 限界

- motifの結果は、BRIMが実行も検証もしていない外部ツールの出力です。BRIMは、入力された条件を記録します。
- motifの一致は、peak内に結合配列があることを示すもので、TFがそこに結合することを示すものではありません。
- motifのp値とpadjは、外部ツールが計算した別の検定です。RNA・ATAC・レベル2のpadjとは独立で、BRIMは補正・再計算・結合をしません。
- 候補は仮説であり、TFの機能的な影響を示すものではありません。

## 未検証の点

HOMERの`knownResults.txt`の見出しの綴り（Motif Name、P-value、q-value (Benjamini)など）に対するBRIMの認識規則は、記憶に基づいて
作られており、実際のツール出力では検証していません。認識できない場合は、取り込みが分かりやすいエラーで止まります。実際の出力での
検証は次のPhaseで行います。
