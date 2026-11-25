"""正解データが無い状態で精度を数値で追うための指標。改善の前後比較に使う。

領域の指標は regions.extract_regions が内部で計算する（diag["_quality"] → 段階 4 の指標）:
  hatch_recall     材種ごとのハッチ線の画素のうち、その材種の領域に入った割合（領域の取りこぼし）
  hatch_precision  領域の画素のうち、ハッチ線から間隔 1.5 本分（最低 2.5pt）以内にある割合（領域のはみ出し）
  boundary_pt      領域の輪郭画素から床仕上図の黒い長い線（壁・境界線）までの距離の中央値／95 パーセンタイル
  overlap_m2       材種の異なる領域どうしの重なり（重なり解決を有効にすると 0）
線画の指標はここで計算する:
  small_components 整理後の線画で、塗り範囲の中・家具の外に残った文字サイズの小さな線の成分数
                   （椅子の部品なども含むので上限の目安。減れば消し残しが減っている）
  furniture_ratio  塗り範囲に占める家具（白抜き）の面積比
"""
from __future__ import annotations

import cv2
import numpy as np


def lineart_metrics(lineart: np.ndarray, furniture: np.ndarray, painted: np.ndarray, S: float, mm_per_pt: float,
                    glyph_pt: float = 4.5) -> dict:
    ink = (lineart.min(axis=2) < 200).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    gmax = glyph_pt * S
    small = 0
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w <= gmax and h <= gmax and a >= 3:
            cx, cy = x + w // 2, y + h // 2
            if painted[cy, cx] > 0 and furniture[cy, cx] == 0:
                small += 1
    pa = float((painted > 0).sum())
    return {"small_components": int(small), "furniture_ratio": round(float(((furniture > 0) & (painted > 0)).sum()) / max(1.0, pa), 4),
            "painted_m2": round(pa / (S / mm_per_pt * 1000) ** 2, 1)}


def report(ex, res) -> dict:
    """抽出結果と合成結果から、精度と性能の要約を 1 つの辞書にする（run_cases の quality.json、画面の表示用）。"""
    st4 = ex.stages[3].metrics if len(ex.stages) > 3 else {}
    st7 = ex.stages[6].metrics if len(ex.stages) > 6 else {}
    lm = lineart_metrics(ex.clean.lineart, ex.clean.furniture, res.painted, ex.S, ex.mm_per_pt)
    times = {s.name: round(s.seconds, 2) for s in ex.stages}
    if res.stage is not None:
        times[res.stage.name] = round(res.stage.seconds, 2)
    return {
        "regions": len(ex.regions),
        "hatch_recall_mean": st4.get("ハッチ線の取り込み率（平均）"), "hatch_precision_mean": st4.get("領域のはみ出し無し率（平均）"),
        "boundary_median_mm": st4.get("輪郭と黒線の距離 中央値 [mm]"), "boundary_p95_mm": st4.get("輪郭と黒線の距離 95% [mm]"),
        "arrows_assigned": f"{sum(1 for r in ex.regions if r.direction is not None)}/{len(ex.regions)}",
        "furniture": st7.get("家具と判定"), "chair_symbols": ex.clean.stats.get("chair_symbols"),
        **lm, "timings_s": times, "total_s": round(sum(times.values()), 2),
    }
