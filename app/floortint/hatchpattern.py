"""同じ色でハッチの模様だけが違う床仕上図（MTG 大阪など）への対応。

床仕上図の規約は「材種ごとに色違いのハッチ線」だが、1 色（マゼンタなど）で 斜線／格子（クロス）／点 の模様を変えて材種を分ける図面もある。
色で分ける既存の処理をそのまま使えるように、前処理で「色＋模様の種類」ごとに仮想の色を割り当て直す（凡例の記号片も同じ規則で塗り替える）。
模様の種類: 斜線（45°／135° の単線）、格子（45° と 135° が同じ場所に重なる）、点（短い線片）、水平・垂直線。
"""
from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np
from floortint import pdfio

from floortint.pdfdoc import BLACK, WHITE, Drawing, is_gray


def _angle_class(d: Drawing) -> str | None:
    segs = d.segments()
    if not segs:
        return None
    lens = [math.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in segs]
    if max(lens) < 2.5:
        return "dot"                                   # 2.5pt 未満の線片だけ（点模様）。什器で切れた短いハッチ片は向きで分ける
    x0, y0, x1, y1 = max(zip(segs, lens), key=lambda z: z[1])[0]
    a = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
    for c, ref in (("h0", 0.0), ("h45", 45.0), ("h90", 90.0), ("h135", 135.0), ("h0", 180.0)):
        if abs(a - ref) < 8:
            return c
    return "other"


def recolor_by_pattern(drs: list[Drawing], page_rect: pdfio.Rect, min_count: int = 20, only_colors: set | None = None) -> tuple[list[Drawing], dict]:
    """色ごとに模様の種類を数え、2 種類以上あれば種類ごとに仮想の色を付けた Drawing の列を返す（元の列は変えない）。
    only_colors を渡すとその色だけ（凡例で同じ色が 2 行以上ある色）。戻り値 (図形の列, 診断 {色: {種類: 本数}})。"""
    by_color: dict[tuple, list[int]] = {}
    for i, d in enumerate(drs):
        if d.color is None or d.fill is not None or d.color in (BLACK, WHITE) or is_gray(d.color):
            continue
        if only_colors is not None and d.color not in only_colors:
            continue
        by_color.setdefault(d.color, []).append(i)
    out = list(drs)
    info: dict = {}
    for col, idx in by_color.items():
        cls = {i: _angle_class(drs[i]) for i in idx}
        counts: dict[str, int] = {}
        for c in cls.values():
            if c:
                counts[c] = counts.get(c, 0) + 1
        major = {c: n for c, n in counts.items() if n >= min_count and c != "other"}
        if len(major) < 2:
            continue
        # 45° と 135° が同じ場所に重なるものは「格子」
        cross_ids: set[int] = set()
        if "h45" in major and "h135" in major:
            W, H = int(page_rect.width) + 2, int(page_rect.height) + 2
            def raster(c: str) -> np.ndarray:
                m = np.zeros((H, W), np.uint8)
                for i in idx:
                    if cls[i] == c:
                        for x0, y0, x1, y1 in drs[i].segments():
                            cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 255, 1)
                return m
            a, b = raster("h45"), raster("h135")
            # ハッチ間隔: 片方の線の中心間の最近傍距離の中央値（無ければ 12pt）
            pts = np.array([[(d_.rect.x0 + d_.rect.x1) / 2, (d_.rect.y0 + d_.rect.y1) / 2] for i, d_ in ((i, drs[i]) for i in idx) if cls[i] == "h45"], np.float32)
            sp = 12.0
            if len(pts) > 5:
                from scipy.spatial import cKDTree
                dd, _ = cKDTree(pts).query(pts, k=2)
                sp = float(np.median(dd[:, 1])) or 12.0
            k = int(max(5, min(60, sp * 2.5))) | 1
            ker = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            both = (cv2.dilate(a, ker) > 0) & (cv2.dilate(b, ker) > 0)
            for i in idx:
                if cls[i] in ("h45", "h135"):
                    m = np.zeros((H, W), np.uint8)
                    for x0, y0, x1, y1 in drs[i].segments():
                        cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 255, 1)
                    on = m > 0
                    if on.any() and (both & on).sum() >= 0.5 * on.sum():
                        cross_ids.add(i)
        # 仮想の色: 元の色を種類ごとに少しずつ変える（彩度は保つ）
        order = ["cross", "h45", "h135", "h0", "h90", "dot"]
        shift = {"cross": (0.0, 0.0, 0.0), "h45": (-0.08, 0.04, 0.0), "h135": (0.0, -0.08, 0.06), "h0": (0.06, 0.0, -0.08), "h90": (-0.04, -0.04, 0.08), "dot": (0.08, 0.08, -0.08)}
        used: dict[str, int] = {}
        colors: dict[str, tuple] = {}
        for i in idx:
            c = "cross" if i in cross_ids else cls[i]
            if c not in order:
                continue
            nc = tuple(round(min(1.0, max(0.0, v + s)), 3) for v, s in zip(col, shift[c]))
            out[i] = replace(drs[i], color=nc)
            used[c] = used.get(c, 0) + 1
            colors[c] = nc
        info[",".join(f"{v:.2f}" for v in col)] = {"classes": used, "colors": {c: ",".join(f"{v:.2f}" for v in nc) for c, nc in colors.items()}}
    return out, info


def pattern_pitch_overrides(drs: list[Drawing], info: dict) -> dict[str, float]:
    """格子（X の格子）と点模様は平行線の切片からは間隔が取れないので、図形の中心どうしの最近傍距離（同じ X の 2 本は中心が同じなので除く）で
    間隔を決める。戻り値 {材種キー（色）: 間隔 pt}。"""
    from scipy.spatial import cKDTree
    out: dict[str, float] = {}
    for base, rec in info.items():
        for c in ("cross", "dot"):
            key = rec.get("colors", {}).get(c)
            if not key:
                continue
            pts = np.array([[(d.rect.x0 + d.rect.x1) / 2, (d.rect.y0 + d.rect.y1) / 2] for d in drs
                            if d.color is not None and ",".join(f"{v:.2f}" for v in d.color) == key], np.float32)
            if len(pts) < 10:
                continue
            if c == "dot":
                # 1 つの点印が複数の線片でできている（v 字など）ので、2pt 以内の点を 1 つの印にまとめてから最近傍を測る
                W, H = int(pts[:, 0].max()) + 4, int(pts[:, 1].max()) + 4
                m = np.zeros((H, W), np.uint8)
                for x, y in pts:
                    m[int(y), int(x)] = 255
                m = cv2.dilate(m, np.ones((5, 5), np.uint8))
                n, _, _, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
                pts = cen[1:].astype(np.float32)
                if len(pts) < 10:
                    continue
            dd, _ = cKDTree(pts).query(pts, k=min(4, len(pts)))
            nn = []
            for row in dd:
                cand = [v for v in row[1:] if v > 0.5]
                if cand:
                    nn.append(cand[0])
            if nn:
                out[key] = float(np.median(nn))
    return out

