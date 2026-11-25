"""各段階の処理状況・性能・途中画像を記録する（改善のためのデバッグ用）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class StageLog:
    name: str                                  # 段階名（例: "3 凡例"）
    seconds: float = 0.0
    status: str = "ok"                         # ok | warn | error
    metrics: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)  # 表示名 → np.ndarray（縮小済み）
    notes: list = field(default_factory=list)   # 警告・所見

    def warn(self, msg: str) -> None:
        self.notes.append(msg)
        if self.status == "ok":
            self.status = "warn"


def shrink(img: np.ndarray, max_w: int = 1600) -> np.ndarray:
    """表示用に縮小。2 値マスクは白黒の RGB にする。"""
    if img is None:
        return img
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    if w > max_w:
        s = max_w / w
        img = cv2.resize(img, (max_w, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def label_image(lab: np.ndarray, max_w: int = 1600) -> np.ndarray:
    """連結成分ラベルを色分けした画像（0 は黒）。"""
    h, w = lab.shape
    s = min(1.0, max_w / w)
    small = cv2.resize(lab.astype(np.int32), (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
    rng = np.random.default_rng(3)
    n = int(small.max()) + 1
    lut = rng.integers(60, 255, size=(n, 3), dtype=np.uint8)
    lut[0] = 0
    counts = np.bincount(small.ravel(), minlength=n)
    counts[0] = 0
    lut[int(np.argmax(counts))] = 235                # いちばん大きい成分（建物の外や廊下）は薄い灰色にして目立たせない
    return lut[small]


def overlay_mask(base: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45, max_w: int = 1600) -> np.ndarray:
    b = shrink(base, max_w).astype(np.float32)
    m = shrink(mask, max_w)[:, :, 0] > 0
    col = np.array(color, np.float32)
    b[m] = b[m] * (1 - alpha) + col * alpha
    return b.astype(np.uint8)


def side_by_side(a: np.ndarray, b: np.ndarray, max_w: int = 1600) -> np.ndarray:
    a, b = shrink(a, max_w // 2), shrink(b, max_w // 2)
    h = max(a.shape[0], b.shape[0])
    pad = lambda x: np.pad(x, ((0, h - x.shape[0]), (0, 0), (0, 0)), constant_values=255)
    return np.concatenate([pad(a), np.full((h, 6, 3), 200, np.uint8), pad(b)], axis=1)


def ink_overlay(plan_gray: np.ndarray, floor_gray: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """平面図のインクを赤、床仕上図のインク（平面図座標へ戻したもの）を青で重ねる。位置合わせの確認用。"""
    h, w = plan_gray.shape
    out = np.full((h, w, 3), 255, np.uint8)
    a = plan_gray < 200
    f = np.zeros_like(a)
    fh, fw = floor_gray.shape
    ys, xs = np.where(floor_gray < 200)
    xs2, ys2 = xs - dx, ys - dy
    ok = (xs2 >= 0) & (xs2 < w) & (ys2 >= 0) & (ys2 < h)
    f[ys2[ok], xs2[ok]] = True
    out[a] = (220, 40, 40)
    out[f] = (40, 80, 220)
    out[a & f] = (40, 40, 40)
    return out
