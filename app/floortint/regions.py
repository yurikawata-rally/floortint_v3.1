"""塗り領域の復元と貼り方向。

床仕上図では材種ごとに色違いのハッチ線が引かれている。色別にラスタ化し、ハッチ間隔から決めた核で
閉じると、閉じた塗り領域になる（曲線境界も再現できる）。貼り方向は黒塗りの矢印（三角＋軸）で示される。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from floortint import pdfio

from floortint.legend import Material
from floortint.pdfdoc import BLACK, Drawing, touches


@dataclass
class Region:
    id: int
    material_key: str
    polygon: np.ndarray                     # (N,2) pt、床仕上図の座標
    holes: list[np.ndarray] = field(default_factory=list)
    area_pt2: float = 0.0
    direction: float | None = None          # 貼り方向 [deg]、None は矢印なし
    centroid: tuple[float, float] = (0.0, 0.0)
    enabled: bool = True

    def contains(self, x: float, y: float) -> bool:
        if cv2.pointPolygonTest(self.polygon.astype(np.float32), (float(x), float(y)), False) < 0:
            return False
        return not any(cv2.pointPolygonTest(h.astype(np.float32), (float(x), float(y)), False) >= 0 for h in self.holes)

    def to_dict(self, mm_per_pt: float) -> dict:
        return {"id": self.id, "material": self.material_key, "area_m2": round(self.area_pt2 * (mm_per_pt / 1000) ** 2, 2),
                "direction": self.direction, "centroid": [round(self.centroid[0], 1), round(self.centroid[1], 1)],
                "enabled": self.enabled, "n_vertices": int(len(self.polygon)), "n_holes": len(self.holes)}


def hatch_pitch(segs: list[tuple[float, float, float, float]]) -> tuple[float, float]:
    """平行ハッチの間隔 [pt] と角度 [deg]。角度の最頻方向に直交する切片の差の中央値。"""
    if not segs:
        return 6.0, 0.0
    ang = np.array([math.atan2(y1 - y0, x1 - x0) % math.pi for x0, y0, x1, y1 in segs])
    hist, edges = np.histogram(ang, bins=36, range=(0, math.pi))
    th = edges[np.argmax(hist)] + (edges[1] - edges[0]) / 2
    nx, ny = -math.sin(th), math.cos(th)
    sel = [(x0, y0) for (x0, y0, x1, y1), a in zip(segs, ang) if abs(a - th) < 0.1 or abs(abs(a - th) - math.pi) < 0.1]
    icp = np.array(sorted(set(round(x * nx + y * ny, 1) for x, y in sel)))
    d = np.diff(icp)
    d = d[(d > 0.4) & (d < 60)]
    if not len(d):
        return 6.0, math.degrees(th)
    # レンガ模様のように線が二重になる場合は中央値が小さく出るので、上側の分位も見る
    return float(max(np.median(d), np.percentile(d, 85))), math.degrees(th)


def legend_bbox(zones: list[pdfio.Rect], pad: float = 12.0) -> pdfio.Rect | None:
    """凡例表全体の矩形（記号欄ゾーンの和を少し広げたもの）。"""
    if not zones:
        return None
    r = pdfio.Rect(zones[0])
    for z in zones[1:]:
        r |= z
    return pdfio.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)


def _mask_from_segments(segs, W, H, S, thick=2) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in segs:
        cv2.line(m, (int(round(x0 * S)), int(round(y0 * S))), (int(round(x1 * S)), int(round(y1 * S))), 255, thick)
    return m


def extract_regions(page: pdfio.Page, drs: list[Drawing], materials: list[Material], zones: list[pdfio.Rect],
                    S: float = 3.0, min_area_pt2: float = 60.0, hole_min_pt2: float = 40.0,
                    close_factor: float = 1.5, simplify_pt: float = 0.6, debug: dict | None = None,
                    resolve_overlap: bool = True, snap_to_lines: bool = True, grow_factor: float = 0.75,
                    barrier_min_pt: float = 4.0, pitch_override: dict | None = None) -> tuple[list[Region], dict]:
    """凡例に載っている色ごとに、凡例ゾーン外のストロークから領域を復元する。

    1. 色ごとにハッチ線をラスタ化し、ハッチ間隔 × close_factor の核で閉じる（従来どおり）
    2. resolve_overlap: 複数の材種が同じ画素を主張したら、ハッチ線に近い方に与える（二重塗りを無くす）
    3. snap_to_lines: 床仕上図の黒い線（壁・境界線）で領域を分割し、ハッチ線を含まない断片（線を越えて
       にじみ出た部分）を捨て、残った領域を黒い線まで広げる（間隔 × grow_factor まで、線は越えない）
    debug を渡すと、材種ごとに「ハッチ線」「閉じた後」「確定」の縮小画像を入れる。"""
    from floortint.debugview import shrink
    W, H = int(page.rect.width * S) + 1, int(page.rect.height * S) + 1
    regions: list[Region] = []
    diag: dict[str, dict] = {}
    legend_rect = legend_bbox(zones)
    idx_of: list[Material] = []
    hatch_masks: dict[str, np.ndarray] = {}
    metric_masks: dict[str, np.ndarray] = {}
    closed_masks: dict[str, np.ndarray] = {}
    # ---- 1. 色ごとの閉じたマスク
    for m in materials:
        if m.color is None:
            continue
        segs = []
        n = 0
        for d in drs:
            if d.color != m.color:
                continue
            if any(touches(z, d.rect) for z in zones) or (legend_rect is not None and legend_rect.contains(d.rect)):
                continue
            n += 1
            segs += d.segments()
        if not segs:
            diag[m.key] = {"strokes": 0}
            continue
        pitch, ang = hatch_pitch(segs)
        if pitch_override and m.key in pitch_override:
            pitch = float(pitch_override[m.key])             # 点模様など、平行線の間隔が取れない模様の外部推定
        mask = _mask_from_segments(segs, W, H, S)
        ks = max(5, int(pitch * S * close_factor) | 1)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks)))
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        # 取り込み率の評価には、ハッチの向き（線分の 10% 以上を占める向き。ヘリンボーンは 2 つ）の線だけを使う。
        # 材種色で描かれた領域の輪郭線は壁の黒線と重なるため、領域に入らなくても見た目に影響しない
        angs = np.array([math.atan2(sg[3] - sg[1], sg[2] - sg[0]) % math.pi for sg in segs])
        hist, edges = np.histogram(angs, bins=36, range=(0, math.pi))
        dom = [edges[b] + (edges[1] - edges[0]) / 2 for b in np.where(hist >= 0.10 * len(segs))[0]]
        on_hatch = [any(abs(((a - t + math.pi / 2) % math.pi) - math.pi / 2) <= math.radians(10) for t in dom) for a in angs]
        metric_masks[m.key] = _mask_from_segments([sg for sg, ok in zip(segs, on_hatch) if ok], W, H, S) if any(on_hatch) else mask
        hatch_masks[m.key] = mask
        closed_masks[m.key] = closed
        idx_of.append(m)
        diag[m.key] = {"strokes": n, "pitch_pt": round(pitch, 2), "hatch_deg": round(ang, 1), "kernel_px": ks}
    if not idx_of:
        return regions, diag
    # ---- 2. 重なりの解決: ハッチ線に近い材種が勝つ
    label = np.zeros((H, W), np.int16)
    best = np.full((H, W), np.inf, np.float32)
    for i, m in enumerate(idx_of, start=1):
        closed = closed_masks[m.key] > 0
        if resolve_overlap:
            # 距離はその材種のハッチ間隔で正規化する（間隔の細かい模様が常に勝つのを防ぐ）
            dist = cv2.distanceTransform((hatch_masks[m.key] == 0).astype(np.uint8), cv2.DIST_L2, 3) / max(diag[m.key]["pitch_pt"] * S, 1.0)
            win = closed & (dist < best)
            best[win] = dist[win]
        else:
            win = closed & (label == 0)
        label[win] = i
    del best
    # ---- 3. 黒い線で分割し、にじみを捨て、線まで広げる
    ink_d = None
    if snap_to_lines:
        # 境界にするのは長い黒線（壁・区画線）だけ。椅子や記号の短い線は境界にしない
        black = [sg for d in drs if d.color == BLACK and d.fill is None
                 and not (legend_rect is not None and legend_rect.contains(d.rect)) for sg in d.segments()
                 if math.hypot(sg[2] - sg[0], sg[3] - sg[1]) >= barrier_min_pt]
        ink = _mask_from_segments(black, W, H, S, thick=1) if black else np.zeros((H, W), np.uint8)
        ink_d = cv2.dilate(ink, np.ones((3, 3), np.uint8)) > 0
        all_black = [sg for d in drs if d.color == BLACK and d.fill is None
                     and not (legend_rect is not None and legend_rect.contains(d.rect)) for sg in d.segments()]
        ink_all = _mask_from_segments(all_black, W, H, S, thick=1) > 0 if all_black else ink_d
        dist_line = cv2.distanceTransform((~ink_d).astype(np.uint8), cv2.DIST_L2, 3)
        for i, m in enumerate(idx_of, start=1):
            mine = label == i
            if not mine.any():
                continue
            pitch = diag[m.key]["pitch_pt"]
            # 分割して、ハッチ線を含まない断片を落とす
            parts = (mine & ~ink_d).astype(np.uint8)
            n, lab = cv2.connectedComponents(parts, connectivity=4)
            has = np.zeros(n, bool)
            has[np.unique(lab[(hatch_masks[m.key] > 0) & (lab > 0)])] = True
            has[0] = False
            keep = has[lab]
            dropped = int(((parts > 0) & ~keep).sum())
            label[mine & ~keep & ~ink_d] = 0
            # 黒い線まで広げる（空いている画素だけ、線は越えない）
            cur = keep.copy()
            grown = 0
            k = int(pitch * S * grow_factor) + 2
            free = (label == 0) & ~ink_d
            free &= dist_line <= k                         # 線との隙間だけを埋める（線の無い方向へは広げない）
            for _ in range(k):
                nxt = (cv2.dilate(cur.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & free & ~cur
                if not nxt.any():
                    break
                cur |= nxt; grown += int(nxt.sum())
            cur |= (hatch_masks[m.key] > 0) & (label == i)          # ハッチ線そのものは必ず含める
            # 線の画素そのものは隣接する側に含める（同じ材種の両側から挟まれた線は埋めて、領域を分断しない）
            filled = cv2.morphologyEx(cur.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
            onto = filled & ~cur & ink_d & (label == 0)
            cur |= onto
            label[(label == i) & ~cur] = 0
            label[cur] = i
            diag[m.key].update({"dropped_px": dropped, "grown_px": grown})
    # ---- 4. 輪郭
    rid = 0
    for i, m in enumerate(idx_of, start=1):
        final = (label == i).astype(np.uint8) * 255
        cnts, hier = cv2.findContours(final, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if debug is not None:
            debug[m.key] = {"ハッチ線": hatch_masks[m.key], "閉じた後": closed_masks[m.key], "確定": final}
        found = 0
        if hier is not None:
            hier = hier[0]
            for ci, c in enumerate(cnts):
                if hier[ci][3] != -1:
                    continue                                   # 穴は親側で扱う
                a = cv2.contourArea(c) / (S * S)
                if a < min_area_pt2:
                    continue
                poly = cv2.approxPolyDP(c, simplify_pt * S, True).reshape(-1, 2).astype(np.float64) / S
                holes = []
                j = hier[ci][2]
                while j != -1:
                    ha = cv2.contourArea(cnts[j]) / (S * S)
                    if ha >= hole_min_pt2:
                        holes.append(cv2.approxPolyDP(cnts[j], simplify_pt * S, True).reshape(-1, 2).astype(np.float64) / S)
                        a -= ha
                    j = hier[j][0]
                M = cv2.moments(c)
                cx, cy = (M["m10"] / M["m00"] / S, M["m01"] / M["m00"] / S) if M["m00"] else (poly[:, 0].mean(), poly[:, 1].mean())
                regions.append(Region(rid, m.key, poly, holes, a, None, (cx, cy)))
                rid += 1; found += 1
        diag[m.key]["regions"] = found
    # ---- 5. 精度指標（正解なしで追える値）
    q: dict = {"per_material": {}}
    recs, precs = [], []
    claims = np.zeros((H, W), np.uint8)
    for m in idx_of:
        claims += (metric_masks[m.key] > 0).astype(np.uint8)
    shared = claims > 1                                    # 隣り合う材種のハッチが重なる画素
    for i, m in enumerate(idx_of, start=1):
        hatch = (metric_masks[m.key] > 0) & ~shared
        mine = label == i
        pitch = diag[m.key]["pitch_pt"]
        tol = max(int(pitch * S * 1.5), int(2.5 * S)) | 1
        near = cv2.dilate(hatch.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol, tol))) > 0
        rec = float((hatch & mine).sum() / max(1, hatch.sum())); prec = float((mine & near).sum() / max(1, mine.sum()))
        recs.append(rec); precs.append(prec)
        q["per_material"][m.name] = {"hatch_recall": round(rec, 4), "hatch_precision": round(prec, 4)}
    q["hatch_recall_mean"] = round(float(np.mean(recs)), 4); q["hatch_precision_mean"] = round(float(np.mean(precs)), 4)
    q["overlap_m2"] = 0.0 if resolve_overlap else None
    q["shared_border_px"] = int(shared.sum())
    if ink_d is not None:
        dist = cv2.distanceTransform((~ink_all).astype(np.uint8), cv2.DIST_L2, 3)
        edge = cv2.morphologyEx((label > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        d = dist[edge] / S
        if len(d):
            q["boundary_pt"] = {"median": round(float(np.median(d)), 2), "p95": round(float(np.percentile(d, 95)), 2),
                                "within_1.5pt": round(float((d <= 1.5).mean()), 3)}
    diag["_quality"] = q
    if debug:
        union = np.zeros((H, W), bool)
        for imgs in debug.values():
            union |= imgs["ハッチ線"] > 0
        ys, xs = np.where(union)
        if len(xs):
            pad = int(10 * S)
            y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad); x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
            for k, imgs in debug.items():
                debug[k] = {nm: shrink(255 - im[y0:y1, x0:x1], 1200) for nm, im in imgs.items()}
    return regions, diag


@dataclass
class Arrow:
    cx: float
    cy: float
    angle: float          # [deg] 0 = 水平、90 = 垂直（mod 180）
    length: float


def extract_arrows(drs: list[Drawing], zones: list[pdfio.Rect]) -> list[Arrow]:
    """黒塗りの細長い軸（長さ 12〜80pt、太さ 3pt 以下）を矢印とみなす。頭（小三角）は補助。"""
    out = []
    legend_rect = legend_bbox(zones)
    for d in drs:
        if d.fill != BLACK or d.color not in (None, BLACK):
            continue
        if any(touches(z, d.rect) for z in zones) or (legend_rect is not None and legend_rect.contains(d.rect)):
            continue
        pts = []
        for s in d.segments():
            pts += [(s[0], s[1]), (s[2], s[3])]
        if len(pts) < 3:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.array(pts, np.float32))
        L, T = max(w, h), min(w, h)
        if not (12 <= L <= 80 and T <= 3.2):
            continue
        a = ang if w >= h else ang + 90
        out.append(Arrow(float(cx), float(cy), float(a % 180), float(L)))
    return out


def assign_directions(regions: list[Region], arrows: list[Arrow]) -> int:
    """矢印の中心を含む最小の領域に方向を与える。戻り値は割り当てた矢印の数。"""
    n = 0
    for ar in arrows:
        cands = [r for r in regions if r.contains(ar.cx, ar.cy)]
        if cands:
            r = min(cands, key=lambda z: z.area_pt2)
        else:
            # 境界線の上に置かれた矢印: 10pt 以内で最も近い領域に与える
            near = [(abs(cv2.pointPolygonTest(z.polygon.astype(np.float32), (float(ar.cx), float(ar.cy)), True)), z) for z in regions]
            near = [(d, z) for d, z in near if d <= 10.0]
            if not near:
                continue
            r = min(near, key=lambda t: t[0])[1]
        if r.direction is None:
            r.direction = round(ar.angle, 1)
        n += 1
    return n


def regions_bbox(regions: list[Region]) -> pdfio.Rect | None:
    if not regions:
        return None
    xs = np.concatenate([r.polygon[:, 0] for r in regions]); ys = np.concatenate([r.polygon[:, 1] for r in regions])
    return pdfio.Rect(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
