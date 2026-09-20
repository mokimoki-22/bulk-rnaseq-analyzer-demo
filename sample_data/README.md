# ATAC sample data

Phase 1のStreamlit非依存ATAC coreを試すための人工データです。実研究データではありません。

- `BRIM_ATAC_peak_counts.csv`: 120 peaks × 6 samplesの非負整数count matrix。
  `peak_id`は0-based half-openの`chrom:start-end`形式です。先頭30 peaksはTreatedで増加、
  次の30 peaksは減少、残り60 peaksは概ね不変になる固定データです。
- `BRIM_ATAC_metadata.csv`: Control 3 samples、Treated 3 samplesの対応表です。
- `BRIM_ATAC_DAR.csv`: 解析済みDAR tableモードのschema確認用です。padj NA行を1行含みます。

これらはBRIMのテスト用に新規生成したデータで、リポジトリのMIT Licenseの対象です。

## Phase 5 のTF候補テスト用の合成データ

Phase 5（TF候補、Level 2）のテストは、同梱データではなく`tests/tf_support.py`の
`synthetic_tf_level1_state()`が生成する決定的な合成のRNA+ATAC結果を使います。実研究データではなく、
生物学的な妥当性を示すものでもありません。

- 同梱のMouse CollecTRIで標的数が最大のTFを「仕込みTF」とし、その標的のうち25遺伝子を
  `concordant_activation`にします（固定seed 0、背景400遺伝子、詳細は`docs/phase5_implementation_plan.md`§5.6）。
- 3 vs 3のTF activity行列では、仕込みTFだけが群間で完全に分離します。
- 陰性対照は、同じ背景でclassの割り当てを並べ替えたものです。「該当するTFが0個」という確認は、この固定seedに対する
  回帰検査であり、一般的な偽陽性率の保証ではありません。