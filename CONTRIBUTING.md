# Contributing to BRIM

## 開発環境

Python 3.11または3.12で仮想環境を作成し、依存関係をインストールします。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## 変更方針

- 作業前に`AGENTS.md`、`PLAN.md`、`ARCHITECTURE.md`を確認してください。
- 現在のPhase外の機能を先取りしないでください。
- RNA-only workflowの意味的な挙動を変えないでください。
- 解析コードとStreamlit UIを分離し、解析モジュールはStreamlitへ依存させません。
- 追加機能には実用的な範囲でテストを添え、`python -m pytest -q`を実行してください。
- 新しい外部通信、Windows portable配布を壊す依存、科学的な不変条件に反する変更は行わないでください。

## 変更提案

変更の目的、影響範囲、実行したテストを記載してください。設計境界または依存関係を変更する
場合は、実装前に提案し、`ARCHITECTURE.md`と対応設計書の整合を確認してください。

## ライセンス

コードはMIT Licenseです。ただし`references/`配下には第三者由来データが含まれます。
再配布時は[README.md](README.md)および各参照データの付帯文書に記載された条件を確認してください。
