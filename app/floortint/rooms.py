"""平面図の閉領域（部屋・ゾーン）を取り出す。指示書モード／手動モードで塗る単位。

図面整理（cleanplan）が作る閉領域ラベル enclosed のうち、面積が min_m2 以上で、図面部の縁や画像の縁に触れない成分を
「部屋」とする。什器の足元や柱は部屋の中の穴になっているので、塗るときは穴を埋める（什器は後で白く抜く）。
ただし、部屋の中に別の部屋（開放オフィスの中の会議室など）があれば、それは埋めない。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import re

import numpy as np


@dataclass
class Room:
    id: int
    label: int                                  # enclosed のラベル値
    area_m2: float
    bbox: tuple[int, int, int, int]             # px (x0, y0, x1, y1)
    centroid: tuple[float, float]               # px
    name: str = ""                              # 室名（文字が部屋の中にあれば）
    seed: str | None = None                     # 指示書の色キー（割り当て後）
    seed_cover: float = 0.0                     # その色が部屋を覆う割合
    material_key: str | None = None             # 材種（割り当て後）
    direction: float | None = None
    core: str = ""                              # "EV" | "階段" | ""  コア（既定で塗らない）

    def to_dict(self) -> dict:
        return {"id": self.id, "area_m2": round(self.area_m2, 1), "name": self.name, "core": self.core, "seed": self.seed,
                "seed_cover": round(self.seed_cover, 3), "material": self.material_key, "bbox": list(self.bbox)}


@dataclass
class RoomSet:
    rooms: list[Room] = field(default_factory=list)
    room_map: np.ndarray | None = None          # (H,W) int32 各画素が属する部屋 id（穴を埋めた後。0=どの部屋でもない）
    big_union: np.ndarray | None = None         # (H,W) bool 部屋（穴を埋める前）の和

    def by_id(self, rid: int) -> Room | None:
        return next((r for r in self.rooms if r.id == rid), None)

    def mask(self, rid: int) -> np.ndarray:
        return ((self.room_map == rid).astype(np.uint8)) * 255


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """2 値マスク（bool）の穴（外から到達できない 0）を埋める。"""
    h, w = mask.shape
    ff = np.ones((h + 2, w + 2), np.uint8)          # 1 = 背景（縁を 1 で囲む）、0 = マスク
    ff[1:-1, 1:-1] = (~mask).astype(np.uint8)
    cv2.floodFill(ff, None, (0, 0), 2)              # 縁から届く背景を 2 にする
    return ff[1:-1, 1:-1] != 2                      # 届かない背景（穴）とマスク自身


def find_rooms(enclosed: np.ndarray, zone_mask: np.ndarray | None, px_per_mm: float, min_m2: float = 2.0,
               texts: list | None = None, S: float = 1.0, max_rooms: int = 400) -> RoomSet:
    """部屋の一覧と、各画素の部屋 id マップ。texts は (x, y, size, text) pt（平面図座標）。"""
    H, W = enclosed.shape
    amin = min_m2 * (1000 * px_per_mm) ** 2
    ids, counts = np.unique(enclosed, return_counts=True)
    border = set(np.unique(np.concatenate([enclosed[0], enclosed[-1], enclosed[:, 0], enclosed[:, -1]])).tolist())
    if zone_mask is not None:
        border |= set(np.unique(enclosed[zone_mask == 0]).tolist())
    big = [(int(i), int(c)) for i, c in zip(ids, counts) if i > 0 and c >= amin and i not in border]
    big.sort(key=lambda z: -z[1])
    big = big[:max_rooms]
    big_union = np.isin(enclosed, [i for i, _ in big])
    rs = RoomSet(big_union=big_union)
    room_map = np.zeros((H, W), np.int32)
    # 大きい部屋から描き、小さい部屋（大きい部屋の中の会議室）が上書きする
    for n, (lab, cnt) in enumerate(big, start=1):
        comp = enclosed == lab
        ys, xs = np.where(comp)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        sub = comp[y0:y1, x0:x1]
        filled = fill_holes(sub)
        # 穴のうち、他の部屋（min_m2 以上の別成分）は埋めない
        other = big_union[y0:y1, x0:x1] & ~sub
        filled &= ~other
        rm = room_map[y0:y1, x0:x1]
        rm[filled] = n
        rs.rooms.append(Room(n, lab, float(cnt) / (1000 * px_per_mm) ** 2, (int(x0), int(y0), int(x1), int(y1)),
                             (float(xs.mean()), float(ys.mean()))))
    rs.room_map = room_map
    # 室名: 文字の位置が部屋の中なら付ける（大きい文字を優先）
    if texts:
        for x, y, sz, t in sorted(texts, key=lambda z: -z[2]):
            px, py = int(x * S), int(y * S)
            if 0 <= px < W and 0 <= py < H:
                rid = int(room_map[py, px])
                r = rs.by_id(rid)
                if r is not None and not r.name:
                    r.name = t
    return rs


def room_overlay(lineart: np.ndarray, rs: RoomSet, colors: dict[int, tuple[int, int, int]] | None = None,
                 alpha: float = 0.45, number: bool = True) -> np.ndarray:
    """部屋を色分けして番号を描いた確認用画像。colors は id → RGB（無ければ自動の淡い色）。"""
    out = lineart.copy()
    rm = rs.room_map
    pal = [(255, 200, 150), (170, 210, 255), (190, 240, 180), (255, 230, 150), (230, 190, 255), (255, 190, 200),
           (190, 230, 230), (220, 220, 190)]
    layer = np.zeros_like(out)
    hit = np.zeros(rm.shape, bool)
    for r in rs.rooms:
        m = rm == r.id
        c = (colors or {}).get(r.id, pal[r.id % len(pal)])
        layer[m] = c
        hit |= m
    out[hit] = (out[hit].astype(np.float32) * (1 - alpha) + layer[hit].astype(np.float32) * alpha).astype(np.uint8)
    if number:
        for r in rs.rooms:
            cx, cy = int(r.centroid[0]), int(r.centroid[1])
            cv2.putText(out, str(r.id), (cx - 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, str(r.id), (cx - 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mark_core_rooms(rs: RoomSet, segments: list[tuple[float, float, float, float]], S: float, mm_per_pt: float,
                    max_m2: float = 60.0) -> int:
    """EV シャフト（矩形に対角線 2 本の X 印）と階段（等間隔の平行線が多数）の部屋に core を付ける。segments は pt。
    戻り値は印を付けた数。"""
    import math
    segs = np.array(segments, np.float32) if segments else np.zeros((0, 4), np.float32)
    n = 0
    for r in rs.rooms:
        if r.area_m2 > max_m2 or r.area_m2 < 1.5:
            continue
        x0, y0, x1, y1 = [v / S for v in r.bbox]
        bw, bh = x1 - x0, y1 - y0
        if len(segs) == 0:
            continue
        inside = (segs[:, [0, 2]].min(axis=1) >= x0 - 2) & (segs[:, [0, 2]].max(axis=1) <= x1 + 2) & \
                 (segs[:, [1, 3]].min(axis=1) >= y0 - 2) & (segs[:, [1, 3]].max(axis=1) <= y1 + 2)
        sub = segs[inside]
        if len(sub) == 0:
            continue
        dx, dy = sub[:, 2] - sub[:, 0], sub[:, 3] - sub[:, 1]
        length = np.hypot(dx, dy)
        diag = math.hypot(bw, bh)
        # X 印: 対角線に近い向きで長さが対角線の 7 割以上の線が、両向きにある
        ang = np.degrees(np.arctan2(dy, dx)) % 180
        a1, a2 = math.degrees(math.atan2(bh, bw)) % 180, math.degrees(math.atan2(-bh, bw)) % 180
        long = length >= 0.7 * diag
        if long.any() and (np.abs(ang[long] - a1) < 6).any() and (np.abs(ang[long] - a2) < 6).any():
            r.core = "EV"; n += 1
            continue
        # 階段: 同じ向きの線が 7 本以上、長さが短辺の 6 割以上、間隔が 200〜400mm
        short = min(bw, bh)
        for a in (0.0, 90.0):
            sel = (np.abs(((ang - a) + 90) % 180 - 90) < 3) & (length >= 0.6 * short)
            if sel.sum() >= 7:
                pos = np.sort((sub[sel, 1] + sub[sel, 3]) / 2 if a == 0.0 else (sub[sel, 0] + sub[sel, 2]) / 2)
                gaps = np.diff(pos) * mm_per_pt
                gaps = gaps[gaps > 20]
                if len(gaps) >= 6 and 150 <= np.median(gaps) <= 450:
                    r.core = "階段"; n += 1
                    break
    return n


# 共用部（室名で判定。既定で塗らない）: トイレ・PS/EPS/DS・機械室・電気室・EV・階段・湯沸室・廊下・ホールなど
COMMON_RE = re.compile(r"WC|W\.C|トイレ|便所|化粧室|EPS|(?<![A-Z])PS(?![A-Z])|(?<![A-Z])DS(?![A-Z])|MECH|機械室|電気室|ELV|ELEV|(?<![A-Z])EV(?![A-Z])|ＥＶ|エレベ|階段|STAIR|"
                       r"パイプ|湯沸|給湯室|廊下|共用|(?<![A-Z])SK(?![A-Z])|PIT|シャフト|ＰＳ|ＥＰＳ|ＤＳ|非常用|避難|ゴミ|ごみ|塵芥|空調機械|AHU|受水|ポンプ", re.I)


CORRIDOR_RE = re.compile(r"廊下|ホール|共用|ロビー|LOBBY|HALL|CORRIDOR", re.I)


def mark_common_rooms(rs: RoomSet, extra_pattern: str = "", max_m2: float = 80.0, corridor_max_m2: float = 400.0) -> int:
    """室名が共用部の語に当たる部屋に core="共用" を付ける（EV・階段の印が付いていない部屋だけ）。
    WC・PS・機械室などは小さい部屋（max_m2 以下）に限る: 湯沸コーナーの文字が壁の無い大部屋の中にあると大部屋が共用になるのを防ぐ。
    廊下・ホールは corridor_max_m2 まで。戻り値は付けた数。"""
    rx = COMMON_RE
    if extra_pattern:
        try:
            rx = re.compile(COMMON_RE.pattern + "|" + extra_pattern, re.I)
        except re.error:
            pass
    n = 0
    for r in rs.rooms:
        if r.core or not r.name:
            continue
        if CORRIDOR_RE.search(r.name) and r.area_m2 <= corridor_max_m2:
            r.core = "共用"; n += 1
        elif rx.search(r.name) and r.area_m2 <= max_m2:
            r.core = "共用"; n += 1
    return n


def split_mask_by_lines(mask: np.ndarray, lines: list, thickness: int = 3) -> list[np.ndarray]:
    """マスクを線（(x0,y0,x1,y1) px。マスクの外接矩形まで延長する）で分割し、面積の大きい順に部分マスクを返す。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    bx0, bx1, by0, by1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    m = (mask > 0).astype(np.uint8)
    for x0, y0, x1, y1 in lines:
        dx, dy = x1 - x0, y1 - y0
        L = float(np.hypot(dx, dy))
        if L < 1:
            continue
        ext = float(np.hypot(bx1 - bx0, by1 - by0)) + 10
        ux, uy = dx / L, dy / L
        pa = (int(round(x0 - ux * ext)), int(round(y0 - uy * ext))); pb = (int(round(x1 + ux * ext)), int(round(y1 + uy * ext)))
        cv2.line(m, pa, pb, 0, thickness)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    parts = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 20:
            continue
        parts.append(((lab == i).astype(np.uint8) * 255, int(st[i, cv2.CC_STAT_AREA])))
    parts.sort(key=lambda z: -z[1])
    # 線の太さで消えた画素を、最も近い部分に戻す（境界をきれいにする）
    out = [pm for pm, _ in parts]
    if len(out) >= 2:
        lost = (mask > 0) & (m == 0)
        if lost.any():
            idx = np.zeros(mask.shape, np.int32)
            for i, pm in enumerate(out, start=1):
                idx[pm > 0] = i
            dist = cv2.distanceTransform((idx == 0).astype(np.uint8), cv2.DIST_L2, 3)
            # 最近傍ラベル: 各部分ごとの距離で比べる（部分は少ないので単純に回す）
            best = np.full(mask.shape, np.inf, np.float32); who = np.zeros(mask.shape, np.int32)
            for i, pm in enumerate(out, start=1):
                d_ = cv2.distanceTransform((pm == 0).astype(np.uint8), cv2.DIST_L2, 3)
                sel = lost & (d_ < best)
                best[sel] = d_[sel]; who[sel] = i
            for i, pm in enumerate(out, start=1):
                pm[lost & (who == i)] = 255
            del dist
    return out
