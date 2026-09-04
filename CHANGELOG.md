# Changelog

このプロジェクトは[Keep a Changelog](https://keepachangelog.com/ja/1.1.0/)の形式を
参考に記録します。

## [Unreleased]

### Added

- GitHub ActionsによるWindows/Ubuntu、Python 3.11/3.12のbaseline CI
- 既存RNA workflowの回帰基準テスト
- Phase 0.5: `run_deg()`が補完前のNAを`padj_is_na` / `lfc_is_na`として保持。
  既存6列の計算、padj/LFC/statの補完と並べ替えは維持。
- Streamlit非依存の`brim_provenance.py`を追加し、RNA Exportを共有manifest生成器へ移行。
  `Provenance/manifest.json`と`Provenance/manifest.md`を既存ZIPへ追加。
  単独ダウンロードはZIP内と同一内容。既存CSV・notebookのパスは維持。
- 元のアップロードファイル名・SHA-256、出力count matrixのSHA-256、サンプル数、
  NAフラグ件数、現在のcontrast、ID変換の方法・対応率、環境・package版を記録。
- mygene.info / string-db.orgの送信先とデータ種別を実行前に表示。
  明示操作による照会を`services.external_services_used`と`services.events`へ記録。
  既存キャッシュは維持し、照会記録は新規通信の有無を断定しない。
- 変更前コード`16450fa`のDEG・JSON・ZIPを`22ad00b`で先に記録。
  実PyDESeq2の数値互換性、4種類のNA、Export内容、共有生成器接続、結果ありUI、
  外部サービス操作のテストを追加。監査項目1〜5の繰り延べなし。

### Changed

- Phase 0.5監査・指摘1: 入力検証に失敗してもmygene.infoの照会記録を破棄しない。
  単一／複数Studyとも全HTTP試行を呼び出し前に登録し、失敗・部分成功・キャッシュを区別。
  `services.events`は現在の入力、追加の`services.external_service_events`はセッション全履歴。
  `external_services_used`は現在の入力に属する成功／部分成功照会のサービス名とする。
  timeout・非200・不正応答・部分的mapping・入力検証失敗の回帰テストを追加。
- 監査で追加指摘されたMeta／TF／Interactionの結果あり検証と、既存InteractionのNA対応は
  ユーザー指定によりPhase 2へ繰り延べ（PLAN参照）。既存37テスト・段階1fixtureは変更しない。
- MIT著作権者を当面「Motoki Morita」とする。公開前に所属機関の知財ポリシーを確認して確定する。
- 旧`reproducibility_report.json`の情報は以下へ移行（日時はExport生成時刻であり固定値ではない）。

| 旧項目 | 新manifest内の対応先 |
|---|---|
| `timestamp` | `environment.timestamp`（UTC offset付き）および`environment.timezone` |
| `app_version` | `environment.app_version` |
| `species` | `settings.species` |
| `deg_parameters.lfc_threshold` | `settings.rna.lfc_threshold` |
| `deg_parameters.padj_threshold` | `settings.rna.padj_threshold` |
| `deg_parameters.normalization` | `settings.rna.normalization` |
| `deg_parameters.low_count_filtering.enabled` | `settings.rna.low_count_filtering.enabled` |
| `deg_parameters.low_count_filtering.min_count` | `settings.rna.low_count_filtering.min_count` |
| `deg_parameters.low_count_filtering.min_samples` | `settings.rna.low_count_filtering.min_samples` |
| `contrasts`（解析履歴全体） | `settings.rna.analysis_log`（全レコードを保持） |

manifestのトップレベルは`environment / inputs / settings / counts / services`。
`inputs.atac`・`settings.atac`・`counts.atac`は未実行を示す`null`。
RNA UIではgenome buildを選択しないため`settings.genome_build`も`null`とし推測しない。
旧結果などNAフラグが存在しない場合、NA件数は`null`（不明）とし、padj=1から逆算しない。

## [1.1.0] - 2026-09-04

### Baseline

- Bulk RNA-seq解析アプリとしての現行実装を初回Gitコミットおよび`v1.1.0`タグで固定
- RNA-seq / ATAC-seq統合機能の実装前設計を追加
