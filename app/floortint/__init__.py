"""FloorTint — 平面図（レイアウト）と床仕上図から「平面着色」を全自動で作る。

処理の流れ（設計書 §4）:
  pdfdoc   取込・ページ種別・縮尺・位置合わせ
  legend   凡例（記号色 → 仕様）の読み取り
  regions  色別ハッチ線 → 塗り領域、矢印 → 貼り方向
  cleanplan 平面図の整理（寸法・通り芯・表を除外）と家具の足元抽出
  textures 貼り方に応じた床模様の生成（デフォルメ倍率つき）
  compose  合成（床 → 影 → 家具白 → 線画 → 室名）
  pipeline 上記をまとめる入口と、調整可能なパラメータ
"""
from floortint.pipeline import Params, run, RunResult  # noqa: F401
