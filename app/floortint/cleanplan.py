"""平面図の整理と家具の足元抽出。

寸法線（灰色）・通り芯（破線）・席マーカー（橙・青の塗り）・図面枠外の表や注記を落とし、
壁・家具・建具・植栽だけの線画にする。家具の足元は、線画で囲まれた小さな閉領域として拾う。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from floortint import pdfio

from floortint.pdfdoc import BLACK, WHITE, Drawing, is_gray, render_drawings, spans, touches


@dataclass
class CleanParams:
    margin_pt: float = 14.0          # 床領域の外接矩形からこの幅までを図面部とみなす
    drop_gray: bool = True           # 灰色の線（寸法線）を落とす
    drop_dashed: bool = True         # 破線（通り芯・想定線）を落とす
    drop_markers: bool = True        # 橙・青などの席マーカーを落とす
    keep_plants: bool = True         # 緑（植栽）は残す
    keep_room_names: bool = True
    furniture_min_m2: float = 0.04   # 家具とみなす閉領域の面積 [m²]
    furniture_max_m2: float = 6.0
    line_width_boost: float = 1.0    # 線幅の倍率（細すぎる線を太らせる）
    drop_stroke_text: bool = True    # 線分で描かれた小さな文字（寸法値など）を落とす
    glyph_pt: float = 4.5            # 文字候補の大きさの上限 [pt]（紙の上の大きさ。縮尺によらない）
    text_single_pt: float = 3.6      # 単独の図形はこの高さ以下なら文字。これより大きい滑らかな弧・楕円は椅子として残す
    exclude_door_arcs: bool = True   # 扉の開閉弧を家具判定から外す（弧と壁で囲まれた扇形に影を付けない）
    chair_symbols: bool = True       # 第 3 段: 椅子の記号（曲線を含む小さな図形の群）の凸包を家具にする
    chair_min_mm: float = 350.0      # その群の大きさの範囲 [mm]
    chair_max_mm: float = 900.0
    chair_group_gap_mm: float = 60.0 # 群にまとめる距離 [mm]
    dashed_long_pt: float = 60.0     # これより長い破線だけを通り芯とみなして落とす
    close_gaps_px: int = 2           # 家具判定の前に線の小さな切れ目を埋める幅 [px]（第 1 段）
    chair_gap_mm: float = 300.0      # 第 2 段: 実寸でこの幅までの切れ目を埋め、小さな閉領域（椅子・破線の什器）を追加で拾う
    chair_max_m2: float = 1.0        # 第 2 段で追加する閉領域の面積上限 [m²]（机は第 1 段で閉じているので、椅子だけ）
    drop_green_hatch: bool = True    # 緑でも直線の長いハッチ（転用什器などの印）は植栽でないので落とす
    use_layers: bool = True          # PDF のレイヤーがあればレイヤーで仕分け、色の規則（灰色・破線・マーカー・緑ハッチ）は使わない
    layer_overrides: dict = field(default_factory=dict)   # {レイヤー名: 落とすか} 画面での上書き
    color_rules_with_layers: bool = False   # レイヤーがあっても色の規則を併用する
    color_rules_unnamed: bool = True        # レイヤー名の無い図形（レイヤー外で描かれた寸法値など）には色の規則を使う
    drop_grid: bool = True                  # 等間隔に並ぶ同じ長さの直線の群（OA フロアの割付マス目・タイル割）を落とす
    grid_min_lines: int = 8                 # 群とみなす本数


def _glyph_candidate(d: Drawing, glyph_pt: float) -> bool:
    """線分だけで描かれた、紙の上で文字サイズ（glyph_pt 以下）の図形。曲線を含むもの（椅子など）は除く。"""
    if d.fill is not None or d.rect.width > glyph_pt or d.rect.height > glyph_pt:
        return False
    if any(it[0] in ("c", "qu") for it in d.items):
        return False
    return all(abs(x1 - x0) + abs(y1 - y0) <= glyph_pt * 1.5 for x0, y0, x1, y1 in d.segments())


def _is_smooth_arc(d: Drawing) -> bool:
    """折れ線で描かれた滑らかな弧・楕円（椅子の背や座面）か。曲がる向きが一定で、1 頂点あたりの曲がりが小さい。
    数字や仮名は角（90° 近い曲がり）や曲がりの反転を含むので当たらない。"""
    import math
    segs = d.segments()
    if len(segs) < 3:
        return False
    angs = [math.atan2(y1 - y0, x1 - x0) for x0, y0, x1, y1 in segs]
    turns = []
    for a, b in zip(angs, angs[1:]):
        t = (b - a + math.pi) % (2 * math.pi) - math.pi
        turns.append(t)
    if any(abs(t) > math.radians(75) for t in turns):
        return False                                   # 角（文字の折れ）
    sign = 1 if sum(turns) >= 0 else -1
    if any(sign * t < -math.radians(35) for t in turns):
        return False                                   # 曲がりの反転（2・3・5・S など）
    total = abs(sum(turns))
    return math.radians(50) <= total <= math.radians(400)


def stroke_text_ids(drs: list[Drawing], zone: pdfio.Rect, glyph_pt: float, single_pt: float = 3.6) -> set[int]:
    """文字とみなす図形の添字。文字は紙の上の大きさで描かれ、同じ高さ・同じ下端で横（または縦書き・回転で縦）に
    並ぶ（単語）ことを使う。単独の図形は single_pt 以下なら落とし、それより大きくても滑らかな弧・楕円（椅子）でなければ落とす。"""
    cand = [i for i, d in enumerate(drs) if touches(d.rect, zone) and _glyph_candidate(d, glyph_pt)]
    out: set[int] = set()

    def runs(order, same, near):
        run: list[int] = []
        for i in order:
            if run and same(drs[run[-1]].rect, drs[i].rect) and near(drs[run[-1]].rect, drs[i].rect):
                run.append(i); continue
            if len(run) >= 2:
                out.update(run)
            run = [i]
        if len(run) >= 2:
            out.update(run)
    # 横並び（通常の文字列）
    runs(sorted(cand, key=lambda i: (round(drs[i].rect.y1 * 2), drs[i].rect.x0)),
         lambda q, r: abs(r.height - q.height) <= max(0.45, 0.2 * q.height) and abs(r.y1 - q.y1) <= 0.6,
         lambda q, r: -0.5 <= (r.x0 - q.x1) <= max(1.2, 0.6 * q.height))
    # 縦並び（90° 回転した寸法値など）
    runs(sorted(cand, key=lambda i: (round(drs[i].rect.x1 * 2), drs[i].rect.y0)),
         lambda q, r: abs(r.width - q.width) <= max(0.45, 0.2 * q.width) and abs(r.x1 - q.x1) <= 0.6,
         lambda q, r: -0.5 <= (r.y0 - q.y1) <= max(1.2, 0.6 * q.width))
    for i in cand:
        if i in out:
            continue
        d = drs[i]
        if d.rect.height <= single_pt or not _is_smooth_arc(d):
            out.add(i)
    return out


def _is_door_arc(d: Drawing, mm_per_pt: float) -> bool:
    """扉の開閉弧: ベジェ曲線だけで、外接矩形がほぼ正方形（辺 500〜1300mm）、始点と終点の距離が辺の √2 倍程度。"""
    if d.fill is not None or d.close or not d.items or any(it[0] != "c" for it in d.items) or len(d.items) > 6:
        return False
    w, h = d.rect.width * mm_per_pt, d.rect.height * mm_per_pt
    if not (450 <= w <= 1400 and 450 <= h <= 1400 and 0.7 <= w / h <= 1.43):
        return False
    p0, p1 = d.items[0][1], d.items[-1][4]
    import math
    chord = math.hypot(p1.x - p0.x, p1.y - p0.y) * mm_per_pt
    side = (w + h) / 2
    return 1.15 * side <= chord <= 1.6 * side


def _is_green_hatch(d: Drawing) -> bool:
    """緑の直線 1 本で 6pt より長いもの。植栽は短い線や曲線の集まりなので当たらない。"""
    if any(it[0] == "c" for it in d.items) or len(d.items) > 2:
        return False
    import math
    return any(math.hypot(x1 - x0, y1 - y0) > 6.0 for x0, y0, x1, y1 in d.segments())


@dataclass
class CleanResult:
    lineart: np.ndarray              # (H,W,3) RGB、白地に線画（植栽の緑を含む）
    furniture: np.ndarray            # (H,W) uint8 家具の足元マスク
    plants: np.ndarray               # (H,W) uint8 植栽マスク
    enclosed: np.ndarray             # (H,W) int32 閉領域ラベル（0=線または外）
    texts: list[tuple[float, float, float, str]] = field(default_factory=list)   # (x, y, size, text) pt
    stats: dict = field(default_factory=dict)
    ink: np.ndarray | None = None    # 家具判定に使った線（切れ目を埋めた後）
    plant_img: np.ndarray | None = None   # 植栽だけを描いた画像（白地）
    struct_img: np.ndarray | None = None  # 躯体だけを描いた画像（白地）


def _is_green(c) -> bool:
    r, g, b = c
    return g > 0.35 and g > r + 0.1 and g > b + 0.1


def _is_marker(c) -> bool:
    # 橙 (1,0.5,0)、青 (0,0.5,1)、赤・ピンクなど彩度の高い色
    r, g, b = c
    return (max(c) - min(c)) > 0.4 and not _is_green(c)


def grid_line_ids(drs: list[Drawing], zone: pdfio.Rect, mm_per_pt: float, min_lines: int = 8) -> set[int]:
    """OA フロアの割付マス目など: 同じ向き・ほぼ同じ長さの直線が等間隔（300〜1000mm、ばらつき 5% 以内）に min_lines 本以上並ぶ群。
    壁や什器の線は長さがそろわず等間隔にもならないので当たらない。戻り値は図形の添字。"""
    import math
    cands = []
    for i, d in enumerate(drs):
        if d.fill is not None or not touches(d.rect, zone):
            continue
        segs = d.segments()
        if len(segs) != 1:
            continue
        x0, y0, x1, y1 = segs[0]
        L = math.hypot(x1 - x0, y1 - y0)
        if L * mm_per_pt < 1500:
            continue
        ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
        if min(ang, 180 - ang) < 1.0:
            cands.append((i, "h", (y0 + y1) / 2, L, min(x0, x1), max(x0, x1)))
        elif abs(ang - 90) < 1.0:
            cands.append((i, "v", (x0 + x1) / 2, L, min(y0, y1), max(y0, y1)))
    out: set[int] = set()
    for ori in ("h", "v"):
        rows = [c for c in cands if c[1] == ori]
        if len(rows) < min_lines:
            continue
        # 長さで群にまとめる（±10%）→ 位置の間隔が等しいか
        rows.sort(key=lambda c: c[3])
        groups: list[list] = []
        for c in rows:
            if groups and abs(groups[-1][-1][3] - c[3]) <= 0.1 * c[3]:
                groups[-1].append(c)
            else:
                groups.append([c])
        for g in groups:
            if len(g) < min_lines:
                continue
            # 同じ範囲（始点・終点がそろう）のものだけ
            g.sort(key=lambda c: c[2])
            pos = np.array([c[2] for c in g]); st = np.array([c[4] for c in g]); en = np.array([c[5] for c in g])
            ok = (np.abs(st - np.median(st)) < 0.1 * np.median(en - st)) & (np.abs(en - np.median(en)) < 0.1 * np.median(en - st))
            sel = [c for c, o in zip(g, ok) if o]
            if len(sel) < min_lines:
                continue
            pos = np.array([c[2] for c in sel])
            gaps = np.diff(pos) * mm_per_pt
            gaps = gaps[gaps > 5]
            if len(gaps) < min_lines - 1:
                continue
            med = float(np.median(gaps))
            if 250 <= med <= 1200 and (np.abs(gaps - med) <= 0.05 * med).mean() >= 0.8:
                out.update(c[0] for c in sel)
    return out


def select_drawings(drs: list[Drawing], zone: pdfio.Rect, p: CleanParams,
                    layer_plan=None) -> tuple[list[tuple[Drawing, dict]], list[Drawing]]:
    """残す図形と（描画属性の上書き）、植栽図形を返す。
    layer_plan（layers.LayerPlan）があり has_layers なら、レイヤーで落とす・植栽・躯体を決め、色の規則は既定で使わない。"""
    keep: list[tuple[Drawing, dict]] = []
    plants: list[Drawing] = []
    dropped: dict[str, int] = {"図面部の外": 0, "レイヤー": 0, "席マーカー": 0, "長い破線": 0, "灰色の線": 0, "線分の文字": 0, "緑のハッチ": 0, "割付のマス目": 0}
    grid_ids = grid_line_ids(drs, zone, getattr(p, "mm_per_pt", 1.0), p.grid_min_lines) if p.drop_grid else set()
    by_layer = layer_plan is not None and layer_plan.has_layers and p.use_layers
    color_rules = (not by_layer) or p.color_rules_with_layers
    drop_set = layer_plan.drop_set if by_layer else set()
    plant_set = layer_plan.plant_set if by_layer else set()
    structure_set = layer_plan.structure_set if by_layer else set()
    # 線分で描かれた文字（寸法値など）はレイヤーに関係なく落とす（日鉄の平面図は STRUCTURE レイヤーの中に寸法値の線分がある）
    text_ids = stroke_text_ids(drs, zone, p.glyph_pt, p.text_single_pt) if p.drop_stroke_text else set()
    structures: list[Drawing] = []
    for idx, d in enumerate(drs):
        if not touches(d.rect, zone):
            dropped["図面部の外"] += 1
            continue
        col = d.color if d.color is not None else d.fill
        if col is None:
            continue
        if by_layer and (d.layer or "") in drop_set:
            dropped["レイヤー"] += 1
            continue
        if idx in grid_ids:
            dropped["割付のマス目"] += 1
            continue
        unnamed_rules = by_layer and p.color_rules_unnamed and not d.layer
        is_plant = ((d.layer or "") in plant_set or (unnamed_rules and _is_green(col))) if by_layer else _is_green(col)
        if is_plant:
            if (color_rules or unnamed_rules) and p.drop_green_hatch and _is_green_hatch(d):
                dropped["緑のハッチ"] += 1
                continue
            if p.keep_plants:
                plants.append(d)
                keep.append((d, {}))
            continue
        if color_rules or unnamed_rules:
            if p.drop_markers and (_is_marker(col) or (d.fill is not None and _is_marker(d.fill))):
                dropped["席マーカー"] += 1
                continue
            if p.drop_dashed and d.dashed and max(d.rect.width, d.rect.height) > p.dashed_long_pt:
                dropped["長い破線"] += 1
                continue                               # 長い破線 = 通り芯・想定線。短い破線（既存什器）は残す
            if p.drop_gray and d.color is not None and d.color not in (BLACK, WHITE) and is_gray(d.color) and d.fill is None:
                dropped["灰色の線"] += 1
                continue
        if idx in text_ids and (color_rules or unnamed_rules or (by_layer and (d.layer or "") in structure_set)):
            dropped["線分の文字"] += 1                # 名前の無い図形と躯体レイヤーの中の文字だけ（什器レイヤーの小さな線は残す）
            continue
        if by_layer and (d.layer or "") in structure_set:
            structures.append(d)
        ov: dict = {}
        if d.color is not None and d.color != WHITE:
            ov["color"] = BLACK
        if d.fill is not None and d.fill not in (WHITE,):
            ov["fill"] = BLACK
        if d.color is not None and p.line_width_boost != 1.0:
            ov["width"] = max(d.width, 0.2) * p.line_width_boost
        keep.append((d, ov))
    p.last_dropped = dropped                       # 診断用（select_drawings の呼び出し元が読む）
    p.last_structures = structures
    p.last_by_layer = by_layer
    return keep, plants


def _texts(page: pdfio.Page, zone: pdfio.Rect) -> list[tuple[float, float, float, str]]:
    import re
    out = []
    for b, t, sz in spans(page):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if not zone.contains(pdfio.Point(cx, cy)):
            continue
        if re.fullmatch(r"[\d.,×xX*+()\-\s/㎡mMhH]+", t):
            continue                                       # 寸法・席数などの数値
        if not re.search(r"[ぁ-んァ-ン一-龥]", t):
            continue                                       # 日本語を含む語だけ室名候補
        if sz < 3.5 or len(t) > 24:
            continue
        out.append((b[0], b[1], sz, t))
    return out


def furniture_mask(ink: np.ndarray, px_per_mm: float, p: CleanParams) -> tuple[np.ndarray, np.ndarray, dict]:
    """線画の ink マスクから、囲まれた小領域（家具）を拾う。"""
    inv = (ink == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    H, W = ink.shape
    amin = p.furniture_min_m2 * (1000 * px_per_mm) ** 2
    amax = p.furniture_max_m2 * (1000 * px_per_mm) ** 2
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if amin <= a <= amax:
            keep[i] = True
    fm = keep[lab].astype(np.uint8) * 255
    fm = cv2.dilate(fm, np.ones((3, 3), np.uint8))
    return fm, lab, {"components": int(n - 1), "furniture_components": int(keep.sum())}


def chair_footprints(keep: list[tuple[Drawing, dict]], door_set: set[int], mm_per_pt: float, S: float, H: int, W: int,
                     p: "CleanParams") -> tuple[np.ndarray, int]:
    """第 3 段: 椅子の記号（曲線を含む 350〜900mm の小さな図形の群）を凸包で家具の足元にする。
    背もたれが大きく開いた椅子や、座面と背が別の図形で描かれた椅子は、線で囲まれないのでここで拾う。"""
    from floortint.legend import _cluster
    lo, hi = p.chair_min_mm / mm_per_pt, p.chair_max_mm / mm_per_pt
    cand = [d for d, _ in keep if id(d) not in door_set and not _is_green(d.color or d.fill or (0, 0, 0))
            and lo * 0.3 <= max(d.rect.width, d.rect.height) <= hi]
    fm = np.zeros((H, W), np.uint8)
    if not cand:
        return fm, 0
    groups = _cluster([d.rect for d in cand], gap=p.chair_group_gap_mm / mm_per_pt / 2)
    n = 0
    for g in groups:
        ds = [cand[i] for i in g]
        if len(ds) > 14:
            continue
        r = pdfio.Rect(ds[0].rect)
        for d in ds[1:]:
            r |= d.rect
        w, h = r.width, r.height
        if not (lo <= w <= hi and lo <= h <= hi and 0.4 <= w / max(h, 1e-6) <= 2.5):
            continue
        if not any(any(it[0] == "c" for it in d.items) or _is_smooth_arc(d) for d in ds):
            continue                                   # 曲線を含まない群（記号・箱）は対象外
        pts = []
        for d in ds:
            for x0, y0, x1, y1 in d.segments():
                pts += [(x0, y0), (x1, y1)]
        if len(pts) < 4:
            continue
        hull = cv2.convexHull(np.array(pts, np.float32)).reshape(-1, 2)
        area_m2 = cv2.contourArea(hull) * (mm_per_pt / 1000) ** 2
        if area_m2 > p.chair_max_m2:
            continue
        cv2.fillPoly(fm, [np.round(hull * S).astype(np.int32)], 255)
        n += 1
    return fm, n


def clean_layout(page: pdfio.Page, drs: list[Drawing], zone: pdfio.Rect, S: float, px_per_mm: float,
                 p: CleanParams, layer_plan=None) -> CleanResult:
    import time
    t0 = time.time()
    p.mm_per_pt = S / px_per_mm                     # 割付マス目の判定（実寸の間隔）に使う
    keep, plants = select_drawings(drs, zone, p, layer_plan)
    W, H = page.rect.width, page.rect.height
    plant_ids = set(id(d) for d in plants)
    # 植栽と躯体は線画から外して別に描く（色を変えたり塗ったりできるように）
    struct_ids = set(id(d) for d in getattr(p, "last_structures", []))
    base = [(d, ov) for d, ov in keep if id(d) not in plant_ids and id(d) not in struct_ids]
    lineart = render_drawings(W, H, base, S)
    plant_img = render_drawings(W, H, [(d, {}) for d in plants], S) if plants else None
    struct_img = render_drawings(W, H, [(d, ov) for d, ov in keep if id(d) in struct_ids], S) if struct_ids else None
    pmask = (plant_img.min(axis=2) < 235).astype(np.uint8) * 255 if plant_img is not None else np.zeros(lineart.shape[:2], np.uint8)
    mm_per_pt = 1.0 / px_per_mm * S
    doors = [d for d, _ in keep if p.exclude_door_arcs and _is_door_arc(d, mm_per_pt)]
    door_set = set(id(d) for d in doors)
    t_render = time.time() - t0; t0 = time.time()
    # 家具判定用の線: 線画の黒画素を 1px 太らせて細線の途切れを防ぎ、扉の開閉弧（別描画）を除く
    ink = ((lineart.min(axis=2) < 200) & (pmask == 0)).astype(np.uint8)
    if struct_img is not None:
        ink |= (struct_img.min(axis=2) < 200).astype(np.uint8)
    ink = cv2.dilate(ink, np.ones((3, 3), np.uint8))
    if doors:
        door_img = render_drawings(W, H, [(d, {"color": BLACK, "width": max(d.width, 0.5)}) for d in doors], S)
        door_ink = cv2.dilate((door_img.min(axis=2) < 200).astype(np.uint8), np.ones((5, 5), np.uint8))
        ink[door_ink > 0] = 0
    if p.close_gaps_px > 0:
        k = np.ones((p.close_gaps_px * 2 + 1, p.close_gaps_px * 2 + 1), np.uint8)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    fm, lab, st = furniture_mask(ink, px_per_mm, p)
    # 第 2 段: 背もたれが開いた椅子や、破線（短い線の列）で描かれた什器。実寸 chair_gap_mm までの切れ目を埋めて
    # 小さな閉領域だけを追加する（大きな切れ目を閉じても、面積の上限で部屋や机の連結は入らない）
    added = 0
    gap_px = int(round(p.chair_gap_mm * px_per_mm))
    if gap_px >= 2:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_px | 1, gap_px | 1))
        ink2 = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k2)
        import dataclasses
        q = dataclasses.replace(p, furniture_max_m2=p.chair_max_m2)
        fm2, _, st2 = furniture_mask(ink2, px_per_mm, q)
        new = (fm2 > 0) & (fm == 0)
        new &= ink2 == ink                         # 埋めた分（線の上）は家具にしない
        added = int(new.sum())
        fm[new] = 255
        st["furniture_components_pass2"] = st2.get("furniture_components", 0)
    st["furniture_added_px_pass2"] = added
    # 第 3 段: 椅子の記号の凸包
    n3 = 0
    if p.chair_symbols:
        fm3, n3 = chair_footprints(keep, door_set, mm_per_pt, S, fm.shape[0], fm.shape[1], p)
        new3 = (fm3 > 0) & (fm == 0)
        st["furniture_added_px_pass3"] = int(new3.sum())
        fm[new3] = 255
    st["chair_symbols"] = n3
    fm[pmask > 0] = 0
    texts = _texts(page, zone) if p.keep_room_names else []
    st.update({"kept_drawings": len(keep), "plant_drawings": len(plants), "texts": len(texts), "door_arcs": len(doors),
               "dropped": getattr(p, "last_dropped", {}), "total_drawings": len(drs),
               "t_render": round(t_render, 3), "t_furniture": round(time.time() - t0, 3)})
    st["by_layer"] = bool(getattr(p, "last_by_layer", False))
    st["structure_drawings"] = len(struct_ids)
    res = CleanResult(lineart, fm, pmask, lab, texts, st)
    res.ink = ink * 255                            # デバッグ表示用（家具判定に使った線）
    res.plant_img = plant_img
    res.struct_img = struct_img
    return res
