"""凡例の読み取り。記号欄の色つきハッチ小片を見つけ、右隣の文字を仕様として構造化する。

床仕上図の凡例は「記号 | 仕様 | 工事区分」の表で、記号欄は図面と同じ色のハッチ線で描かれている。
白い記号（ビル標準・変更なし）は色を持たないので、文字から拾う。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from floortint import pdfio

from floortint.pdfdoc import BLACK, WHITE, Drawing, drawings, is_gray, spans

PATTERNS = ["ヘリンボーン", "アシュラー", "ブリック", "市松", "乱", "流し"]
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[×xX]\s*(\d+(?:\.\d+)?)(?:\s*[×xX]\s*t?\s*(\d+(?:\.\d+)?))?")


@dataclass
class Material:
    key: str                       # 色の識別子 "r,g,b"（白は "existing"）
    color: tuple[float, float, float] | None
    lines: list[str] = field(default_factory=list)
    kind: str = "carpet"           # carpet | vinyl | wood | tile | existing | other
    product: str = ""
    maker: str = ""
    size_mm: tuple[float, float] | None = None
    thickness: float | None = None
    pattern: str = "流し"          # 流し | 市松 | 乱 | アシュラー | ブリック | ヘリンボーン
    note: str = ""
    canceled: bool = False
    swatch: tuple[float, float, float, float] | None = None
    order: int = 0

    @property
    def name(self) -> str:
        if self.kind == "existing":
            return "既存床（変更なし）"
        base = {"carpet": "タイルカーペット", "vinyl": "塩ビタイル", "wood": "フローリング", "tile": "タイル"}.get(self.kind, "床材")
        return f"{base} {self.product}".strip()

    @property
    def spec(self) -> str:
        s = " / ".join(x for x in [self.product, self.maker] if x)
        if self.size_mm:
            s += f"  {self.size_mm[0]:g}×{self.size_mm[1]:g}" + (f"×t{self.thickness:g}" if self.thickness else "")
        return s

    @property
    def hex(self) -> str:
        if not self.color:
            return "#e6e6e6"
        return "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in self.color)

    def to_dict(self) -> dict:
        return {"key": self.key, "color": self.hex, "name": self.name, "kind": self.kind, "product": self.product,
                "maker": self.maker, "size_mm": self.size_mm, "thickness": self.thickness, "pattern": self.pattern,
                "note": self.note, "canceled": self.canceled, "lines": self.lines}


def _looks_like_spec(text: str) -> bool:
    """「AB1520-3 全厚8t 川島セルコン」のように、貼り方の語が無くても品番＋メーカーなら仕様行とみなす。"""
    from floortint.pptinstr import CODE_RE, MAKERS
    t = re.sub(r"\s+", "", text)
    return bool(CODE_RE.search(text)) and (any(mk in t for mk in MAKERS) or bool(re.search(r"全厚|\d+t\b|mm", text)))


def _cluster(rects: list[pdfio.Rect], gap: float = 3.0) -> list[list[int]]:
    """矩形を近接で連結して群にする（小さい記号片の検出用）。
    矩形を gap だけ広げてラスタに描き、連結成分で群を決める（矩形数の 2 乗にならない）。"""
    n = len(rects)
    if n == 0:
        return []
    import cv2
    import numpy as np
    a = np.array([[r.x0, r.y0, r.x1, r.y1] for r in rects], np.float64)
    x0, y0 = a[:, 0].min() - gap - 1, a[:, 1].min() - gap - 1
    R = 2.0                                               # 0.5pt の解像度
    W = int((a[:, 2].max() - x0 + gap + 1) * R) + 1; H = int((a[:, 3].max() - y0 + gap + 1) * R) + 1
    img = np.zeros((H, W), np.uint8)
    for r in a:
        cv2.rectangle(img, (int((r[0] - gap - x0) * R), int((r[1] - gap - y0) * R)),
                      (int((r[2] + gap - x0) * R), int((r[3] + gap - y0) * R)), 255, -1)
    _, lab = cv2.connectedComponents(img, connectivity=8)
    groups: dict[int, list[int]] = {}
    for i, r in enumerate(a):
        cx, cy = int(((r[0] + r[2]) / 2 - x0) * R), int(((r[1] + r[3]) / 2 - y0) * R)
        groups.setdefault(int(lab[min(H - 1, cy), min(W - 1, cx)]), []).append(i)
    return list(groups.values())


def _kind(text: str) -> str:
    if "変更なし" in text or "ビル標準" in text and "貼" not in text:
        return "existing"
    if "タイルカーペット" in text or "カーペット" in text:
        return "carpet"
    if "フローリング" in text or "木質" in text or "無垢" in text:
        return "wood"
    if "塩ビ" in text or "ビニル" in text or "ビニール" in text or "長尺" in text:
        return "vinyl"
    if "タイル" in text or "石" in text:
        return "tile"
    return "other"


def _parse(m: Material) -> Material:
    # "RE5251-15,RE5263-15" ＋ 次行 "/川島織物セルコン" のように品番とメーカーが行で割れている場合をつなぐ
    joined: list[str] = []
    for ln in m.lines:
        if ln.startswith("/") and joined:
            joined[-1] = joined[-1] + ln
        else:
            joined.append(ln)
    m.lines = joined
    text = "　".join(m.lines)
    m.kind = _kind(text)
    for p in PATTERNS:
        if p + "貼" in text or p + "張" in text or (p in text and p in ("ヘリンボーン", "アシュラー", "ブリック")):
            m.pattern = p; break
    sm = SIZE_RE.search(text)
    if sm:
        m.size_mm = (float(sm.group(1)), float(sm.group(2)))
        if sm.group(3):
            m.thickness = float(sm.group(3))
    # 品番/メーカー: "AB1820-1/川島織物セルコン" や "GA3684/東リ/500×500×t6.8" の形
    for ln in m.lines:
        if "/" in ln and not ln.startswith("OA") and "貼り" not in ln.split("/")[0]:
            parts = [x.strip() for x in ln.split("/") if x.strip()]
            parts = [x for x in parts if not SIZE_RE.search(x)]
            if parts:
                m.product = re.sub(r"\s*\(.*?\)\s*$", "", parts[0]).strip()
                m.maker = parts[1] if len(parts) > 1 else ""
                break
    if not m.product:
        # 「AB1520-3　全厚8t」「川島セルコン」のように / で区切らない書き方: 品番の形とメーカー名の辞書で拾う
        from floortint.pptinstr import parse_legend_text
        g = parse_legend_text(text)
        if g.get("product"):
            m.product = g["product"]; m.maker = m.maker or g.get("maker", "")
            if m.kind == "other" and g.get("kind"):
                m.kind = g["kind"]
    m.canceled = "中止" in text
    notes = [ln for ln in m.lines if re.search(r"\d{6}_|中止|統合|オプション|※", ln)]
    m.note = " ".join(notes)
    return m


def duplicate_swatch_colors(page: pdfio.Page, drs: list[Drawing]) -> set[tuple]:
    """凡例で同じ色の記号片が 2 つ以上ある色（模様だけで材種を分けている図面）。"""
    mats, _ = parse_legend(page, drs, dedup=False)
    seen: dict[tuple, set] = {}
    for m in mats:
        if m.color is not None and m.product and not m.canceled:      # 品番の無い注記行（「転用」など）や中止行は数えない
            seen.setdefault(m.color, set()).add(m.product)
    return {c for c, prods in seen.items() if len(prods) >= 2}


def parse_legend(page: pdfio.Page, drs: list[Drawing] | None = None, dedup: bool = True) -> tuple[list[Material], list[pdfio.Rect]]:
    """凡例の材種一覧と、除外すべき凡例ゾーンの矩形を返す。dedup=False なら同じ色の行も全部返す（模様違いの検出用）。"""
    drs = drs if drs is not None else drawings(page)
    sp = spans(page)
    # 色つきストロークを色ごとに近接クラスタ化し、小さい群を記号片候補にする
    by_color: dict[tuple, list[Drawing]] = {}
    for d in drs:
        if d.color is None or d.color in (BLACK, WHITE):
            continue
        by_color.setdefault(d.color, []).append(d)
    cands: list[tuple[tuple, pdfio.Rect]] = []
    for col, ds in by_color.items():
        if is_gray(col):
            continue                                    # 灰色は中止行の打ち消しハッチなど。材種ではない
        rects = [d.rect for d in ds if d.rect.width <= 50 and d.rect.height <= 50]   # 記号片は小さい（MTG 大阪は 44pt 角）
        if len(rects) < 3:
            continue
        for grp in _cluster(rects, gap=2.5):
            r = pdfio.Rect(rects[grp[0]])
            for i in grp[1:]:
                r |= rects[i]
            if 5 <= r.width <= 50 and 5 <= r.height <= 50 and len(grp) >= 3:
                cands.append((col, r))
    # 右隣に仕様らしい文字がある候補だけ採用
    mats: list[Material] = []
    zones: list[pdfio.Rect] = []
    seen = set()
    # 凡例が複数列のとき、右隣の列の文字を拾わないよう、列の x 位置から文字の探索幅を決める
    col_xs = sorted(set(round(r.x0) for _, r in cands))
    note_xs = [b[0] for b, t, _ in sp if t.startswith("備考")]           # 備考欄も右側の境界になる
    def text_width(x1: float) -> float:
        nxt = [x for x in col_xs + note_xs if x > x1 + 20]
        return min(230.0, (min(nxt) - x1 - 6)) if nxt else 230.0
    for col, r in sorted(cands, key=lambda z: (round(z[1].x0 / 150), z[1].y0)):
        cy = (r.y0 + r.y1) / 2
        tw = text_width(r.x1)
        near = [(b, t) for b, t, _ in sp if r.x1 < b[0] < r.x1 + tw and abs((b[1] + b[3]) / 2 - cy) < 16]
        text = " ".join(t for _, t in near)
        if not near or not (re.search(r"貼|カーペット|タイル|フロア|変更なし|\*\*\*", text) or _looks_like_spec(text)):
            continue
        key = ",".join(f"{c:.2f}" for c in col)
        if key in seen and dedup:
            continue
        seen.add(key)
        near.sort(key=lambda z: (round(z[0][1]), z[0][0]))
        lines = [t for _, t in near if not t.endswith("工事") and t != "*****"]
        if not lines:
            continue                                    # 未使用行（*****）
        m = Material(key=key, color=col, lines=lines, swatch=tuple(r), order=len(mats))
        mats.append(_parse(m))
        zones.append(pdfio.Rect(r.x0 - 4, r.y0 - 14, r.x1 + tw + 10, r.y1 + 14))
    # 白い記号（変更なし）: 文字だけで拾う
    for b, t, _ in sp:
        if "変更なし" in t or ("ビル標準" in t and "カーペット" in t):
            cy = (b[1] + b[3]) / 2
            tw = text_width(b[0] - 20)
            near = [(bb, tt) for bb, tt, _ in sp if abs((bb[1] + bb[3]) / 2 - cy) < 14 and -20 <= (bb[0] - b[0]) < tw]
            near.sort(key=lambda z: (round(z[0][1]), z[0][0]))
            lines = [tt for _, tt in near if not tt.endswith("工事")]
            if not any(m.kind == "existing" for m in mats):
                m = Material(key="existing", color=None, lines=lines, kind="existing", pattern="流し", order=len(mats))
                m.size_mm = (500, 500)
                mats.append(m)
                zones.append(pdfio.Rect(b[0] - 30, b[1] - 14, b[0] + 240, b[3] + 14))
            break
    return mats, zones
