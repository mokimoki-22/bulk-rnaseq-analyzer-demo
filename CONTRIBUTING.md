# Contributing to BRIM

## 開発環境

Python 3.11または3.12で仮想環境を作成し、依存関係をインストールします。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## 作業の再開とPC切り替え

`PLAN.md`の`Current development status`を、現在のPhaseと次の作業を示す唯一の
現在地として扱います。新しいPCやCodexで作業を再開する場合は、最初に`AGENTS.md`、
`PLAN.md`の同節、関連する`ARCHITECTURE.md`を読んでください。

複数PCで作業する場合は、以下の手順を守ります。

1. 作業開始時に`git status`を確認する。未コミット変更がなければ、
   `git pull --ff-only origin main`で最新化する。
2. 未コミット変更がある場合はpullしない。変更内容を確認し、必要な変更だけを
   commitするか、判断が必要なら報告する。
3. 作業終了時に`git status`を確認し、必要な変更だけをcommitして
   `git push origin main`を実行する。
4. push後に`git status`を確認し、`main...origin/main`に差分がないことを確かめる。

`git reset --hard`、強制push、未確認のstash操作は使用しません。GitHub認証が必要な
場合は、認証を完了してから通常のpushを行います。

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
