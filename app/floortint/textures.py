"""床模様の生成。凡例の「寸法」と「貼り方」から、拡大（デフォルメ）した模様を描く。

貼り方:
  流し       同じ向きに並べる（タイルカーペットの既定）
  市松       1 枚ごとに 90° 回して並べる。毛並みの縞が縦横交互になる
  乱         列ごとの継ぎ目位置を無作為にずらす
  アシュラー  列ごとに 1/2 ずらす（馬踏み目地）。板状の塩ビタイルで多い
  ブリック    レンガ積み。アシュラーと同じずらしで、目地をやや強く出す
  ヘリンボーン 板を 45° ずつ互い違いに組む（ハの字）
模様は id マップ（どの画素がどの 1 枚に属するか）を作り、1 枚ごとの明暗・目地・木目を載せてから回転する。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 材種ごとの既定色（品番の実物色は持たないので、画面で差し替える前提）
DEFAULT_COLORS = {
    "carpet": ["#b8b5b0", "#a9adb3", "#c4bfb6", "#9c9a96", "#b3b8b0", "#c9c2b8", "#a3a7ad", "#bdb4ab"],
    "vinyl": ["#c9a97c", "#b58f63", "#d5b98f", "#a98a6a", "#cfb08a"],
    "wood": ["#c9a97c", "#b58f63", "#d5b98f"],
    "tile": ["#d9d6d0", "#c7c3bc"],
    "existing": ["#e3e1dd"],
    "other": ["#cfcac2"],
}


@dataclass
class TextureParams:
    deform: float = 1.5            # 模様の拡大倍率（1 = 実寸）。target_mm > 0 のときは使わない
    target_mm: float = 600.0       # 1 枚の「面積の平方根」をこの見かけの実寸まで拡大する（0 = deform の固定倍率）。
                                   # 500 角タイルは 1.2 倍、250×1000 のヘリンボーン板も 1.2 倍（300×1200）、150×900 の板は 1.63 倍
    long_cap_mm: float = 1200.0    # 長辺の見かけの上限。板が長くなりすぎるのを防ぐ（0 = 制限なし）
    joint: float = 0.5             # 目地の濃さ 0〜1
    variation: float = 0.06        # 1 枚ごとの明暗のばらつき（比率）
    grain: float = 0.5             # 木目の強さ 0〜1（板状の材だけ）
    fiber: float = 0.35            # カーペットの毛並みの強さ 0〜1
    seed: int = 7


def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32)


def _hash01(a: np.ndarray, b: np.ndarray, seed: int) -> np.ndarray:
    """整数 id の組から [0,1) の疑似乱数。"""
    x = (a.astype(np.int64) * 73856093) ^ (b.astype(np.int64) * 19349663) ^ (seed * 83492791)
    x = (x ^ (x >> 13)) * 1274126177
    return ((x ^ (x >> 16)) & 0xFFFFFF).astype(np.float32) / 0xFFFFFF


def _id_maps(pattern: str, X: np.ndarray, Y: np.ndarray, tw: float, th: float, seed: int):
    """画素 → (id_a, id_b, 向き 0=横/1=縦, u, v) を返す。tw は貼り方向に沿う辺、th は直交する辺。
    u, v は 1 枚の中の位置 [px]（横向きなら u∈[0,tw), v∈[0,th)。縦向きの板は u∈[0,th), v∈[0,tw)）。"""
    if pattern == "ヘリンボーン":
        w = th
        r = max(2, int(round(tw / th)))
        i = np.floor(X / w).astype(np.int64); j = np.floor(Y / w).astype(np.int64)
        d = np.mod(i - j, 2 * r)
        horiz = d < r
        ida = np.where(horiz, i - d, i)
        idb = np.where(horiz, j, j - (2 * r - 1 - d))
        return ida, idb, (~horiz).astype(np.uint8), X - ida * w, Y - idb * w
    j = np.floor(Y / th).astype(np.int64)
    if pattern in ("アシュラー", "ブリック"):
        off = (j % 2) * (tw / 2)
    elif pattern == "乱":
        off = np.floor(_hash01(j, np.zeros_like(j), seed) * 8) / 8 * tw
    else:
        off = 0.0
    i = np.floor((X + off) / tw).astype(np.int64)
    orient = np.zeros_like(i, dtype=np.uint8)
    if pattern == "市松":
        orient = ((i + j) % 2).astype(np.uint8)
    return i, j, orient, (X + off) - i * tw, Y - j * th


def adjust_image(img: np.ndarray, saturation: float = 1.0, brightness: float = 1.0) -> np.ndarray:
    """床材写真の彩度・明るさを調節する（1 = そのまま）。"""
    if saturation == 1.0 and brightness == 1.0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _sample_images(images: list[np.ndarray], ida, idb, orient, u, v, tw: float, th: float, seed: int,
                   pattern: str) -> np.ndarray:
    """1 枚ごとに写真を選んで貼る。写真は 1 枚の実物（横向き tw×th）として縮尺を合わせる。
    乱・市松・ヘリンボーンでは枚ごとに写真を替え、180° 回した向きも混ぜて繰り返しを目立たなくする。"""
    W = max(2, int(round(tw))); H = max(2, int(round(th)))
    horiz = [cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA) for im in images]
    vert = [np.ascontiguousarray(np.rot90(im)) for im in horiz]           # 縦向き: (W,H) → 高さ W、幅 H
    n = len(images)
    pick = (_hash01(ida, idb, seed + 11) * n).astype(np.int64) % n
    flip = _hash01(ida, idb, seed + 12) < 0.5 if pattern != "流し" else np.zeros_like(ida, dtype=bool)
    out = np.zeros((*ida.shape, 3), np.uint8)
    ui = np.clip(u.astype(np.int64), 0, 10**9); vi = np.clip(v.astype(np.int64), 0, 10**9)
    for k in range(n):
        for o in (0, 1):
            sel = (pick == k) & (orient == o)
            if not sel.any():
                continue
            img = horiz[k] if o == 0 else vert[k]
            h, w = img.shape[:2]
            uu = np.minimum(ui[sel], w - 1); vv = np.minimum(vi[sel], h - 1)
            f = flip[sel]
            uu = np.where(f, w - 1 - uu, uu); vv = np.where(f, h - 1 - vv, vv)
            out[sel] = img[vv, uu]
    return out


def effective_deform(sw: float, sh: float, p: TextureParams) -> float:
    """実際に使う拡大倍率。見かけの大きさ指定なら、1 枚の面積の平方根（幾何平均）を target_mm にする倍率。
    短辺で揃えると細長い板の長辺が 2m を超えて不自然になるので、面積で揃え、さらに長辺を long_cap_mm で抑える。1〜8 倍に制限。"""
    if p.target_mm and p.target_mm > 0:
        d = p.target_mm / max(1.0, (sw * sh) ** 0.5)
        if p.long_cap_mm and p.long_cap_mm > 0:
            d = min(d, p.long_cap_mm / max(1.0, max(sw, sh)))
        return float(min(8.0, max(1.0, d)))
    return float(p.deform)


def make_texture(base_hex: str, kind: str, pattern: str, size_mm: tuple[float, float] | None, angle_deg: float,
                 w_px: int, h_px: int, px_per_mm: float, p: TextureParams, flat: bool = False,
                 images: list[np.ndarray] | None = None) -> np.ndarray:
    """w_px×h_px の RGB 模様。angle_deg は貼り方向（画像の x 軸から反時計回り、deg）。
    images（床材の写真。1 枚 = 製品 1 枚）があれば、生成模様の代わりに写真を 1 枚ずつ貼る。"""
    base = hex_to_rgb(base_hex)
    if flat:
        return np.tile(base.astype(np.uint8), (h_px, w_px, 1))
    if size_mm is None:
        # 寸法が凡例に無いとき: 板状の貼り方なら板、それ以外はタイル
        size_mm = (900.0, 150.0) if pattern in ("ヘリンボーン", "アシュラー", "ブリック") and kind != "carpet" else (500.0, 500.0)
    sw, sh = size_mm
    if kind in ("vinyl", "wood") and sw < sh:
        sw, sh = sh, sw                              # 板は長辺を貼り方向に
    deform = effective_deform(sw, sh, p)
    tw = max(4.0, sw * px_per_mm * deform)
    th = max(4.0, sh * px_per_mm * deform)
    if pattern == "ヘリンボーン":
        angle_deg += 45.0                            # ハの字の向きが貼り方向（矢印）になるように
    ang = angle_deg % 360
    axis_aligned = abs(ang % 90) < 0.5
    if axis_aligned:
        # 回転が 90° 単位なら、座標の入れ替えで済ませて大きな作業画像を作らない
        Dh, Dw = h_px, w_px
        yy = np.arange(Dh, dtype=np.float32)[:, None]; xx = np.arange(Dw, dtype=np.float32)[None, :]
        q = int(round(ang / 90)) % 4
        X, Y = (xx, yy) if q == 0 else (yy, -xx) if q == 1 else (-xx, -yy) if q == 2 else (-yy, xx)
        X = np.broadcast_to(X, (Dh, Dw)) + 100000.0; Y = np.broadcast_to(Y, (Dh, Dw)) + 100000.0
    else:
        D = int(np.ceil(np.hypot(w_px, h_px))) + int(max(tw, th)) + 4
        D = min(D, 9000)
        Dh = Dw = D
        yy = np.arange(D, dtype=np.float32)[:, None]; xx = np.arange(D, dtype=np.float32)[None, :]
        X, Y = np.broadcast_to(xx, (D, D)), np.broadcast_to(yy, (D, D))
    xs, ys = X, Y
    ida, idb, orient, u, v = _id_maps(pattern, xs, ys, tw, th, p.seed)
    tone = 1.0 + (_hash01(ida, idb, p.seed) - 0.5) * 2 * p.variation
    use_images = bool(images)
    if use_images:
        img = _sample_images(images, ida, idb, orient, u, v, tw, th, p.seed, pattern).astype(np.float32)
    else:
        img = np.ones((Dh, Dw, 3), np.float32) * base[None, None, :]
    # 木目・毛並み（写真を貼るときは不要）
    plank = kind in ("vinyl", "wood") and (sw / sh) >= 2.0
    if use_images:
        pass
    elif plank and p.grain > 0:
        across = np.where(orient == 1, xs, ys)
        phase = _hash01(ida, idb, p.seed + 1) * 6.283
        f1 = 2 * np.pi / max(th / 3.5, 3.0)
        g = (np.sin(across * f1 + phase) * 0.6 + np.sin(across * f1 * 2.7 + phase * 1.7) * 0.4)
        tone *= 1.0 + g * 0.08 * p.grain
        along = np.where(orient == 1, ys, xs)
        tone *= 1.0 + np.sin(along * 0.05 + phase * 3) * 0.015 * p.grain
    elif kind in ("carpet", "existing") and p.fiber > 0:
        along = np.where(orient == 1, xs, ys)
        stripe = np.sin(along * (2 * np.pi / max(th / 14, 2.0)) + _hash01(ida, idb, p.seed + 2) * 6.283)
        tone *= 1.0 + stripe * 0.03 * p.fiber
        rng = np.random.default_rng(p.seed)
        noise = rng.normal(0, 0.012 * p.fiber, (Dh, Dw)).astype(np.float32)
        tone = tone * (1.0 + cv2.GaussianBlur(noise, (0, 0), 1.2))
    img *= tone[:, :, None]
    # 目地: id が変わる画素
    edge = (ida != np.roll(ida, 1, axis=1)) | (idb != np.roll(idb, 1, axis=1)) | \
           (ida != np.roll(ida, 1, axis=0)) | (idb != np.roll(idb, 1, axis=0))
    jw = 1 if max(tw, th) < 40 else 2
    if jw > 1:
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((jw, jw), np.uint8)) > 0
    strength = p.joint * (0.35 if kind in ("carpet", "existing") else 0.55)
    if pattern == "ブリック":
        strength *= 1.3
    img[edge] *= (1.0 - strength)
    img = np.clip(img, 0, 255)
    if axis_aligned:
        return img.astype(np.uint8)
    # 回転して中央を切り出す
    M = cv2.getRotationMatrix2D((Dw / 2, Dh / 2), -angle_deg, 1.0)
    rot = cv2.warpAffine(img, M, (Dw, Dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    y0, x0 = (Dh - h_px) // 2, (Dw - w_px) // 2
    return rot[y0:y0 + h_px, x0:x0 + w_px].astype(np.uint8)


def swatch(base_hex: str, kind: str, pattern: str, size_mm, p: TextureParams, px: int = 96,
           images: list[np.ndarray] | None = None) -> np.ndarray:
    """凡例表示用の小さな見本（実寸 4m 四方を px に収める）。"""
    px_per_mm = px / 4000.0
    return make_texture(base_hex, kind, pattern, size_mm, 0.0, px, px, px_per_mm,
                        TextureParams(**{**p.__dict__, "deform": 1.0, "target_mm": 0.0}), images=images)
