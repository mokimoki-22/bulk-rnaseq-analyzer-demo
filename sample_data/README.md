# ATAC sample data

Phase 1のStreamlit非依存ATAC coreを試すための人工データです。実研究データではありません。

- `BRIM_ATAC_peak_counts.csv`: 120 peaks × 6 samplesの非負整数count matrix。
  `peak_id`は0-based half-openの`chrom:start-end`形式です。先頭30 peaksはTreatedで増加、
  次の30 peaksは減少、残り60 peaksは概ね不変になる固定データです。
- `BRIM_ATAC_metadata.csv`: Control 3 samples、Treated 3 samplesの対応表です。
- `BRIM_ATAC_DAR.csv`: 解析済みDAR tableモードのschema確認用です。padj NA行を1行含みます。

これらはBRIMのテスト用に新規生成したデータで、リポジトリのMIT Licenseの対象です。
