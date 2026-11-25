"""PDF の取込。ページ種別（表題欄の図面名称）、縮尺、通り芯ラベル、平面図と床仕上図の位置合わせ。

座標は PyMuPDF の pt（左上原点、y は下向き）。全て A3 横（1191×842pt）を想定するが依存はしない。
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

import numpy as np
from floortint import pdfio

BLACK = (0.0, 0.0, 0.0)
WHITE = (1.0, 1.0, 1.0)


def rgb_key(c) -> tuple[float, float, float]:
    return tuple(round(float(x), 2) for x in c)


def is_gray(c, tol=0.08) -> bool:
    r, g, b = c
    return abs(r - g) < tol and abs(g - b) < tol


@dataclass
class PageInfo:
    index: int
    title: str
    kind: str                 # plan | floor | finish_table | other
    scale_denom: int | None
    width: float
    height: float
    floor: str | None = None  # 階（"7" など）。平面図と床仕上図の対応づけに使う

    @property
    def label(self) -> str:
        sc = f"（1/{self.scale_denom}）" if self.scale_denom else ""
        fl = f" {self.floor}F" if self.floor else ""
        return f"p{self.index + 1}: {self.title or '（名称なし）'}{fl}{sc}"


def spans(page: pdfio.Page) -> list[tuple[tuple[float, float, float, float], str, float]]:
    """(bbox, text, size) の一覧。空白のみは除く。"""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                t = s["text"].strip()
                if t:
                    out.append((tuple(s["bbox"]), t, float(s["size"])))
    return out


def page_title(page: pdfio.Page) -> str:
    """表題欄「図面名称:」の右下にある文字を図面名とする。無ければ下端付近の大きい文字。"""
    sp = spans(page)
    lab = [b for b, t, _ in sp if t.startswith("図面名称") or t.startswith("図面名")]
    if lab:
        x0, y0, x1, y1 = lab[0]
        cand = [(b, t) for b, t, _ in sp if b[0] >= x0 - 2 and b[0] < x1 + 160 and y1 - 2 < b[1] < y1 + 18
                and not t.startswith("図面名")]
        cand.sort(key=lambda z: (round(z[0][1]), z[0][0]))
        words = []
        for _, w in cand:
            if re.match(r"(縮尺|日付|図面番号|\*-\d+|[0-9]{4}\.[0-9]{2})", w):
                break                                  # 表題欄の隣の項目に入ったら打ち切る
            words.append(w)
        t = " ".join(words).strip()
        if t:
            return t
    h = page.rect.height
    big = [(sz, b, t) for b, t, sz in sp if b[1] > h * 0.85 and sz >= 10 and not re.fullmatch(r"[\d.,/() A-Za-z-]+", t)]
    big.sort(key=lambda z: -z[0])
    return big[0][2] if big else ""


def page_scale(page: pdfio.Page) -> int | None:
    """表題欄の縮尺。"1/200(A3)"、"S=1:150(A3)"、"縮尺: 1/250"、"1:250"、"SCALE 1:100" の形を順に探す。"""
    txt = page.get_text()
    for pat in (r"1[/:]\s?(\d{2,4})\s*\(?A\d", r"(?:縮尺|SCALE)[:：=\s]*(?:S\s*=\s*)?1[/:]\s?(\d{2,4})", r"S\s*=\s*1[/:]\s?(\d{2,4})",
                r"(?<![\d.])1[/:](\d{2,4})(?![\d.])"):
        m = re.findall(pat, txt, re.I)
        if m:
            v = int(m[0])
            if 20 <= v <= 1000:
                return v
    return None


FLOOR_RE = re.compile(r"(?<![0-9A-Za-z])([0-9０-９]{1,2})\s*[FＦ階](?![0-9A-Za-z])")
_Z2H = str.maketrans("０１２３４５６７８９", "0123456789")


def floor_label(page: pdfio.Page) -> str | None:
    """ページの文字から階（「7F」「８階」など）の最頻値を取る。無ければ None。"""
    import collections
    c = collections.Counter(m.group(1).translate(_Z2H).lstrip("0") or "0" for m in FLOOR_RE.finditer(page.get_text()))
    return c.most_common(1)[0][0] if c else None


def classify(title: str) -> str:
    t = title.replace(" ", "")
    if "床仕上" in t:
        return "floor"
    if "仕上表" in t:
        return "finish_table"
    bad = ("転用", "面積", "天井", "照明", "電源", "申請", "解体", "壁", "建具", "鍵", "サイン", "什器高さ", "展開", "造作", "床工事")
    if ("平面図" in t or "レイアウト" in t) and not any(x in t for x in bad):
        return "plan"
    return "other"


def scan(doc: pdfio.Document) -> list[PageInfo]:
    out = []
    for i, p in enumerate(doc):
        title = page_title(p)
        out.append(PageInfo(i, title, classify(title), page_scale(p), p.rect.width, p.rect.height, floor_label(p)))
    # 単ページで名称が取れない場合は本文の語で判定
    if len(out) == 1 and out[0].kind == "other":
        txt = doc[0].get_text()
        if "凡例" in txt and ("床仕上" in txt or "貼り" in txt):
            out[0].kind = "floor"
        elif "平面図" in txt or "レイアウト" in txt:
            out[0].kind = "plan"
    return out


def pick_pages(infos: list[PageInfo]) -> tuple[int | None, int | None]:
    """(平面図ページ, 床仕上図ページ) の既定候補。"""
    plan = next((i.index for i in infos if i.kind == "plan"), None)
    floor = pick_floor_page(infos, infos[plan].floor if plan is not None else None)
    return plan, floor


def pick_floor_page(infos: list[PageInfo], plan_floor: str | None) -> int | None:
    """床仕上図ページの既定。平面図と同じ階のものを優先する（1F と 7F が同じファイルにある場合など）。"""
    floors = [i for i in infos if i.kind == "floor"]
    if not floors:
        return None
    if plan_floor:
        same = [i for i in floors if i.floor == plan_floor]
        if same:
            return same[0].index
    return floors[0].index


GRID_RE = re.compile(r"^[XY]\d{1,2}[a-z]?$")


def grid_labels(page: pdfio.Page) -> dict[str, list[tuple[float, float]]]:
    out: dict[str, list[tuple[float, float]]] = {}
    for b, t, _ in spans(page):
        if GRID_RE.match(t) or re.fullmatch(r"[A-Z]|\d{1,2}", t) and (b[2] - b[0]) < 14:
            out.setdefault(t, []).append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    return out


@dataclass
class Alignment:
    dx: float          # 床仕上図の座標 = 平面図の座標 + (dx, dy)
    dy: float
    method: str
    residual: float = 0.0
    matched: int = 0


def align(plan: pdfio.Page, floor: pdfio.Page) -> Alignment:
    """通り芯ラベルの同名対応で平行移動量を求める。取れなければ壁のラスタ相関で合わせる。"""
    ga, gb = grid_labels(plan), grid_labels(floor)
    # X1/Y2 形式を優先し、無ければ丸付きの 1〜9 / A〜Z（出現が 4 回以下のものだけ）も使う
    for strict in (True, False):
        dx, dy = [], []
        for k in set(ga) & set(gb):
            if strict and not GRID_RE.match(k):
                continue
            if not strict and (len(ga[k]) > 4 or len(gb[k]) > 4):
                continue
            for xa, ya in ga[k]:
                xb, yb = min(gb[k], key=lambda q: abs(q[0] - xa) + abs(q[1] - ya))
                dx.append(xb - xa); dy.append(yb - ya)
        if len(dx) >= 3:
            mx, my = statistics.median(dx), statistics.median(dy)
            res = float(np.median(np.abs(np.array(dx) - mx)) + np.median(np.abs(np.array(dy) - my)))
            if res < 2.0:
                return Alignment(mx, my, "grid" if strict else "grid-bubble", res, len(dx))
    # フォールバック: 黒い線だけのラスタで位相相関
    import cv2
    def ink(p):
        pix = p.get_pixmap(matrix=pdfio.Matrix(1, 1), colorspace=pdfio.csGRAY)
        a = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).astype(np.float32)
        return 255 - a
    a, b = ink(plan), ink(floor)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    (sx, sy), resp = cv2.phaseCorrelate(a[:h, :w], b[:h, :w])
    return Alignment(float(sx), float(sy), "phase", 1.0 - float(resp), 0)


def touches(a: pdfio.Rect, b: pdfio.Rect) -> bool:
    """矩形が重なるか。pdfio.Rect.intersects は幅 0・高さ 0 の矩形（水平・垂直の単線）を常に False にするので使わない。"""
    return not (a.x1 < b.x0 or a.x0 > b.x1 or a.y1 < b.y0 or a.y0 > b.y1)


def mm_per_pt(scale_denom: int | None) -> float:
    return (scale_denom or 200) * 25.4 / 72.0


@dataclass
class Drawing:
    """get_drawings() の 1 件を扱いやすくした形。"""
    rect: pdfio.Rect
    items: list
    color: tuple | None
    fill: tuple | None
    width: float
    dashes: str | None
    close: bool
    even_odd: bool
    layer: str | None = None         # PDF のレイヤー（OCG）名。無ければ None
    fill_opacity: float = 1.0
    stroke_opacity: float = 1.0

    @classmethod
    def from_raw(cls, g: dict) -> "Drawing":
        return cls(g["rect"], g["items"], rgb_key(g["color"]) if g.get("color") is not None else None,
                   rgb_key(g["fill"]) if g.get("fill") is not None else None,
                   float(g.get("width") or 0.0), g.get("dashes"), bool(g.get("closePath")), bool(g.get("even_odd")),
                   g.get("layer") or None, float(g.get("fill_opacity") if g.get("fill_opacity") is not None else 1.0),
                   float(g.get("stroke_opacity") if g.get("stroke_opacity") is not None else 1.0))

    @property
    def dashed(self) -> bool:
        return bool(self.dashes) and self.dashes not in ("[] 0", "[]")

    def segments(self) -> list[tuple[float, float, float, float]]:
        """直線要素だけ (x0,y0,x1,y1)。曲線は制御点を折れ線で近似。"""
        out = []
        for it in self.items:
            if it[0] == "l":
                out.append((it[1].x, it[1].y, it[2].x, it[2].y))
            elif it[0] == "re":
                r = it[1]
                out += [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1), (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
            elif it[0] == "qu":
                q = it[1]; pts = [q.ul, q.ur, q.lr, q.ll]
                out += [(pts[i].x, pts[i].y, pts[(i + 1) % 4].x, pts[(i + 1) % 4].y) for i in range(4)]
            elif it[0] == "c":
                p = it[1:]
                out += [(p[i].x, p[i].y, p[i + 1].x, p[i + 1].y) for i in range(len(p) - 1)]
        return out


def drawings(page: pdfio.Page) -> list[Drawing]:
    return [Drawing.from_raw(g) for g in page.get_drawings()]


def render_drawings(width: float, height: float, items: list[tuple[Drawing, dict]], scale: float) -> np.ndarray:
    """(Drawing, 上書き属性) の列を白地に描き、RGB 配列（H,W,3）で返す（OpenCV で描く。pdfio.render_drawings）。"""
    return pdfio.render_drawings(width, height, items, scale)
