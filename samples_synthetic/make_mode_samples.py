"""3 つの着色モード（床仕上図から／指示書から／指示なし）を試すサンプル一式を作る。

  ..\\.venv\\Scripts\\python.exe make_mode_samples.py            # このフォルダに書く
  ..\\.venv\\Scripts\\python.exe make_mode_samples.py 出力先フォルダ

平面図はどのモードでも layered_plan_3F.pdf（make_layered_plan.py の出力。無ければ先に作る）。
このスクリプトが作るのは、平面図と一緒に入れる資料:

  floor_finish_3F.pdf             床仕上図モード。材種ごとに色の違うハッチ線、凡例表（記号欄に同じ色のハッチ小片、右に仕様）、
                                  貼り方向の黒塗り矢印、材種の境の見切り線。用紙の上の位置を平面図から右へ 12pt・下へ 8pt
                                  ずらしてあり、通り芯の符号（X1…/Y1…）で重ね直す（pdfdoc.align）
  instruction_ppt_3F.pdf          指示書モード（PPT を PDF に書き出した形を模したもの。PowerPoint で作ったものではない）。
                                  平面図の画像、半透明の塗り図形、凡例（色見本・品番・製品写真・リンク）、貼り方向の矢印
  instruction_handwritten_3F.pdf  指示書モード（平面図 PDF に iPad で描き足した形）。斜線のハッチ・囲み、引出線つきの手書きメモと写真

指示なしモードは平面図だけで動く（部屋ごとに画面で割り当てる）。
画像（PNG/JPG）の指示書は作らない。この図面のスクリーンショットでは、図面部分の判定（instruct._drawing_roi）が線の密な
階段に寄り、倍率の照合も繰り返しの多い柱割りで偽の一致を拾って、位置合わせが外れる（2026-10-30 実測）。

材種は全モードで同じ（品番・メーカー・リンク先は架空）:
  A 執務室  タイルカーペット 市松貼り           SMP-101  青
  B 通路    タイルカーペット 流し貼り           SMP-205  橙  執務室の中央の通路だけを塗る（部屋の一部を塗る例）
  C 会議室  タイルカーペット アシュラー貼り     SMP-310  緑
  D 応接室  塩ビタイル（木目） ヘリンボーン貼り SMV-012  桃
  EVホール  既存床（変更なし）  床仕上図の凡例にだけ載せる（白い記号）。指示書・手動モードでは室名で共用部と判定され、既定で塗らない

PPT 指示書の図面画像の作成に pypdfium2・Pillow・numpy（floortint の依存と同じ）を使うので、アプリの仮想環境で動かす。
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from make_layered_plan import (BLACK, COLUMNS, LAYERS, MM_PER_PT, OX, OY, PAGE_H, Plan, build, dimensions, furnishings,
                               grid, paper, room_names, shell, title_block, write_pdf)

HERE = Path(__file__).resolve().parent
PLAN_PDF = HERE / "layered_plan_3F.pdf"


@dataclass
class Mat:
    key: str
    use: str                 # 使う部屋（凡例・メモの補足）
    kind: str                # 凡例の 1 行目（種類と貼り方）
    code: str                # 品番（架空）
    size: str
    thick: str
    color: tuple             # 床仕上図のハッチ・PPT の図形の色
    pen: tuple               # 手書きのペンの色（RGB 0〜255）
    rects: list              # 塗る範囲 [mm]（壁の内法）
    hatch_deg: float
    hatch_pitch: float       # [mm]
    arrows: list             # 貼り方向 (x, y, deg) [mm]
    photo: str               # 製品写真の模様


MATS = [
    Mat("A", "執務室", "タイルカーペット　市松貼り", "SMP-101", "500×500", "6.5", (0.0, 0.45, 0.95), (25, 75, 240),
        [(150, 150, 9600, 10750), (12000, 150, 21550, 10750)], 45, 150, [(4200, 5400, 0), (17400, 5400, 0)], "checker"),
    Mat("B", "通路", "タイルカーペット　流し貼り", "SMP-205", "500×500", "6.5", (1.0, 0.55, 0.0), (255, 130, 0),
        [(9600, 150, 12000, 10750)], 135, 120, [(10800, 5400, 90)], "stripe"),
    Mat("C", "会議室", "タイルカーペット　アシュラー貼り", "SMP-310", "250×1000", "6.5", (0.1, 0.65, 0.3), (0, 155, 65),
        [(150, 10850, 7150, 16050), (7250, 10850, 14350, 16050)], 45, 150, [(3600, 14600, 0), (10800, 14600, 0)], "ashlar"),
    Mat("D", "応接室", "塩ビタイル（木目）　ヘリンボーン貼り", "SMV-012", "150×900", "3", (0.85, 0.15, 0.6), (230, 25, 140),
        [(14450, 10850, 21550, 16050)], 135, 100, [(18000, 13900, 90)], "herringbone"),
]
EXISTING = ("既存床（変更なし）", "ビル標準タイルカーペット　EVホール")


# ---------------------------------------------------------------- 幾何
def _clip(p0, d, r):
    """直線 p0 + t·d と軸平行の矩形 r の交わる t の区間（Liang–Barsky）。"""
    t0, t1 = -math.inf, math.inf
    for dv, q, lo, hi in ((d[0], p0[0], r[0], r[2]), (d[1], p0[1], r[1], r[3])):
        if abs(dv) < 1e-12:
            if q < lo or q > hi:
                return None
            continue
        a, b = (lo - q) / dv, (hi - q) / dv
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    return (t0, t1) if t1 > t0 else None


def hatch_segments(r, pitch, deg, holes=(), min_len=0.0):
    """矩形 r の中の平行線（角度 deg、間隔 pitch）。holes の矩形は抜く。線の位置は原点基準なので隣の範囲と揃う。"""
    a = math.radians(deg)
    d, n = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    s = [x * n[0] + y * n[1] for x, y in ((r[0], r[1]), (r[2], r[1]), (r[0], r[3]), (r[2], r[3]))]
    out = []
    for k in range(math.floor(min(s) / pitch) + 1, math.ceil(max(s) / pitch)):
        p0 = (k * pitch * n[0], k * pitch * n[1])
        iv = _clip(p0, d, r)
        if iv is None:
            continue
        pieces = [iv]
        for h in holes:
            hv = _clip(p0, d, h)
            if hv is not None:
                pieces = [(lo, hi) for a0, b0 in pieces for lo, hi in ((a0, min(b0, hv[0])), (max(a0, hv[1]), b0)) if hi - lo > 1e-6]
        for lo, hi in pieces:
            if hi - lo > min_len:
                out.append(((p0[0] + lo * d[0], p0[1] + lo * d[1]), (p0[0] + hi * d[0], p0[1] + hi * d[1])))
    return out


def to_paper(x, y):
    """図面の mm → 平面図の紙の上の pt（左上原点）。"""
    return OX + x / MM_PER_PT, OY + y / MM_PER_PT


# ---------------------------------------------------------------- 製品写真（架空。模様だけの合成画像）
def product_photo(kind: str, seed: int, n: int = 160) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    if kind == "checker":           # 市松: 方向違いの 2 トーン
        base = np.where(((xx // 40) + (yy // 40)) % 2 == 0, 118, 98)
        tint = (1.0, 1.03, 1.12)
    elif kind == "stripe":          # 流し: 細い横縞
        base = 150 + 14 * np.sin(yy / 2.5)
        tint = (1.12, 1.0, 0.82)
    elif kind == "ashlar":          # アシュラー: 半分ずつずらした長方形
        row = yy // 20
        base = np.where(((xx + (row % 2) * 40) % 80 < 2) | (yy % 20 < 2), 80, 120)
        tint = (0.92, 1.08, 0.95)
    else:                           # ヘリンボーン: 斜めの板
        u, v = (xx + yy) // 18, (xx - yy) % 72
        base = np.where((u % 2 == 0) == (v < 36), 150, 122) + 10 * np.sin(xx / 3.0)
        tint = (1.15, 0.92, 0.7)
    g = base + rng.normal(0, 7, (n, n))
    rgb = np.stack([np.clip(g * t, 0, 255) for t in tint], axis=2).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


def render_plan(scale: float, crop_pt: tuple[float, float, float, float]) -> Image.Image:
    """平面図 PDF を画像にして、紙の上の pt の範囲で切り出す。"""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(PLAN_PDF))
    im = doc[0].render(scale=scale).to_pil().convert("RGB")
    doc.close()
    return im.crop(tuple(int(round(v * scale)) for v in crop_pt))


# ---------------------------------------------------------------- 床仕上図
FLOOR_LAYERS = ["躯体", "間仕切", "建具", "階段", "床仕上", "見切", "貼り方向", "通り芯", "室名", "凡例", "図枠"]


def arrow_mm(P: Plan, layer, cx, cy, deg, length=1200.0, shaft=30.0, head=150.0, head_w=110.0):
    """貼り方向の両矢印: 黒塗りの細長い軸（regions.extract_arrows が拾う）と両端の三角。"""
    a = math.radians(deg)
    u, n = (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))
    h = length / 2 - head
    P.poly(layer, [(cx + s * u[0] * h + t * n[0] * shaft / 2, cy + s * u[1] * h + t * n[1] * shaft / 2)
                   for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))], closed=True, stroke=None, fill=BLACK)
    for s in (-1, 1):
        tip = (cx + s * u[0] * length / 2, cy + s * u[1] * length / 2)
        bx, by = cx + s * u[0] * h, cy + s * u[1] * h
        P.poly(layer, [tip, (bx + n[0] * head_w / 2, by + n[1] * head_w / 2), (bx - n[0] * head_w / 2, by - n[1] * head_w / 2)],
               closed=True, stroke=None, fill=BLACK)


def floor_legend(P: Plan) -> None:
    """凡例表「記号 | 仕様 | 工事区分」。記号欄は図面と同じ色のハッチ小片、仕様は 2 行（種類と貼り方／品番/メーカー/寸法×厚さ）。"""
    x0, xa, xb, x1 = 40.0, 90.0, 350.0, 400.0
    top, hh, rh = 682.0, 14.0, 22.0
    rows = MATS + [None]
    bottom = top + hh + rh * len(rows)
    P.text("凡例", "床仕上凡例", x0 + 2, top - 8, 8, align="left", to_pdf=paper)
    P.rect("凡例", x0, top, x1, bottom, to_pdf=paper, width=0.4)
    for x in (xa, xb):
        P.line("凡例", x, top, x, bottom, to_pdf=paper, width=0.25)
    for i in range(len(rows)):
        P.line("凡例", x0, top + hh + rh * i, x1, top + hh + rh * i, to_pdf=paper, width=0.25)
    for s, x in (("記号", (x0 + xa) / 2), ("仕様", (xa + xb) / 2), ("工事区分", (xb + x1) / 2)):
        P.text("凡例", s, x, top + hh / 2, 6, to_pdf=paper)
    for i, m in enumerate(rows):
        cy = top + hh + rh * i + rh / 2
        sw = (50.0, cy - 7, 80.0, cy + 7)
        if m is None:
            P.rect("凡例", *sw, to_pdf=paper, width=0.3)                       # 白い記号（変更なし）
            lines = EXISTING
        else:
            P.rect("凡例", *sw, to_pdf=paper, stroke=m.color, width=0.3)
            for p, q in hatch_segments(sw, 3.0, m.hatch_deg):
                P.line("凡例", *p, *q, to_pdf=paper, stroke=m.color, width=0.3)
            lines = (f"{m.kind}　（{m.use}）", f"{m.code}/架空床材/{m.size}×t{m.thick}")
        P.text("凡例", lines[0], 94, cy - 4.5, 6.5, align="left", to_pdf=paper)
        P.text("凡例", lines[1], 94, cy + 4.5, 6.5, align="left", to_pdf=paper)
        P.text("凡例", "B工事" if m is not None else "－", (xb + x1) / 2, cy, 6.5, to_pdf=paper)
    P.text("凡例", "※ 品番・メーカーは架空", x0 + 2, bottom + 8, 6, align="left", to_pdf=paper)


def make_floor_finish(dst: Path) -> None:
    F = Plan(FLOOR_LAYERS)
    shell(F, outline=True)
    for m in MATS:
        for r in m.rects:
            for p, q in hatch_segments(r, m.hatch_pitch, m.hatch_deg, holes=COLUMNS, min_len=20.0):
                F.line("床仕上", *p, *q, stroke=m.color, width=0.18)
        for cx, cy, deg in m.arrows:
            arrow_mm(F, "貼り方向", cx, cy, deg)
    for x in (9600, 12000):                                   # 執務室と通路の見切り
        F.line("見切", x, 150, x, 10750, width=0.25)
    grid(F)
    room_names(F)
    floor_legend(F)
    title_block(F, "3F 床仕上図", "床仕上図モード確認用")
    write_pdf(F, dst, title="3F 床仕上図（架空のサンプル・床仕上図モード確認用）", creator="make_mode_samples.py", shift=(12.0, -8.0))


# ---------------------------------------------------------------- PPT 指示書（16:9 のスライドを PDF に書き出した形）
SLIDE_W, SLIDE_H = 960.0, 540.0
CROP = (36.0, 46.0, 1014.0, 666.0)          # 平面図の紙の上の図面部 [pt]（通り芯の符号から外壁まで）
IMG_X, IMG_Y, IMG_W = 24.0, 64.0, 610.0     # スライドの上の図面画像の位置 [pt]（左上原点）
K = IMG_W / (CROP[2] - CROP[0])
IMG_H = (CROP[3] - CROP[1]) * K


def slide_mm(x, y):
    """図面の mm → スライドの PDF の pt。"""
    px, py = to_paper(x, y)
    return IMG_X + (px - CROP[0]) * K, SLIDE_H - (IMG_Y + (py - CROP[1]) * K)


def slide_pt(x, y):
    """スライドの pt（左上原点）→ PDF の pt。"""
    return x, SLIDE_H - y


def ppt_arrow(P: Plan, layer, x0, y0, x1, y1):
    """PPT の矢印: 線 1 本と両端の小さな三角（pptinstr.extract_arrows が拾う）。座標は mm。"""
    P.line(layer, x0, y0, x1, y1, to_pdf=slide_mm, width=1.2)
    (ax, ay), (bx, by) = slide_mm(x0, y0), slide_mm(x1, y1)
    L = math.hypot(bx - ax, by - ay)
    u = ((bx - ax) / L, (by - ay) / L)
    n = (-u[1], u[0])
    for (px, py), s in (((ax, ay), -1), ((bx, by), 1)):
        tip = (px + s * u[0] * 6, py + s * u[1] * 6)
        P.poly(layer, [tip, (px + n[0] * 3.5, py + n[1] * 3.5), (px - n[0] * 3.5, py - n[1] * 3.5)],
               closed=True, to_pdf=lambda x, y: (x, y), stroke=None, fill=BLACK)


def make_ppt(dst: Path) -> None:
    P = Plan(["図面", "範囲", "矢印", "凡例", "文字"])
    im = render_plan(2.0, CROP)
    P.image("図面", "Im0", im.width, im.height, im.tobytes(), IMG_X, SLIDE_H - IMG_Y - IMG_H, IMG_W, IMG_H)
    # 範囲: 材種ごとの半透明の矩形。人が描いた図形なので壁の芯あたりまでかかる（部屋への割り当てはアプリが部屋の形で整える）
    over = 100
    for m in MATS:
        for x0, y0, x1, y1 in m.rects:
            r = (x0 - over if x0 not in (9600, 12000) else x0, y0 - over, x1 + over if x1 not in (9600, 12000) else x1, y1 + over)
            P.rect("範囲", *r, to_pdf=slide_mm, stroke=None, fill=m.color, alpha=0.45)
    ppt_arrow(P, "矢印", 18000, 12700, 18000, 15100)          # 応接室: ヘリンボーンの向き
    ppt_arrow(P, "矢印", 10800, 2400, 10800, 8400)            # 通路: 流し貼りの向き
    # 文字
    P.text("文字", "3F 床材指示書", 24, 30, 15, align="left", to_pdf=slide_pt)
    P.text("文字", "サンプルオフィス（架空）　塗りの色が凡例の材種を示す", 24, 50, 8, align="left", to_pdf=slide_pt)
    P.text("文字", "※ 品番・メーカー・リンク先・写真はすべて架空", 24, 524, 7, align="left", to_pdf=slide_pt)
    # 凡例: 色見本（不透明）の右に 4 行、その右に製品写真。4 行目に製品ページへのリンク
    P.text("凡例", "床材凡例", 650, 56, 9, align="left", to_pdf=slide_pt)
    for i, m in enumerate(MATS):
        y = 74.0 + 90.0 * i
        P.rect("凡例", 650, y, 678, y + 16, to_pdf=slide_pt, stroke=None, fill=m.color)
        for j, s in enumerate((m.kind, f"{m.code} / 架空床材", f"{m.size}　{m.use}", "製品ページ（架空のリンク）")):
            P.text("凡例", s, 686, y + 5 + 11 * j, 8, align="left", to_pdf=slide_pt)
        P.link((686, SLIDE_H - (y + 43), 790, SLIDE_H - (y + 33)), f"https://example.com/floortint-sample/{m.code}")
        ph = product_photo(m.photo, i)
        P.image("凡例", f"Ph{i}", ph.width, ph.height, ph.tobytes(), 860, SLIDE_H - y - 56, 56, 56)
    write_pdf(P, dst, title="3F 床材指示書（架空のサンプル・PPT 書き出し形式）", creator="make_mode_samples.py",
              page_size=(SLIDE_W, SLIDE_H))


# ---------------------------------------------------------------- 手書き指示書（平面図 PDF に描き足し）
def _wobble(rng: random.Random, pts, amp):
    """手で引いた線らしく、点の間を分けて横に少し揺らす。"""
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        k = max(1, int(L / 1500))
        nx, ny = -(y1 - y0) / max(L, 1e-6), (x1 - x0) / max(L, 1e-6)
        for i in range(1, k + 1):
            t = i / k
            j = rng.uniform(-amp, amp) if i < k else 0.0
            out.append((x0 + (x1 - x0) * t + nx * j, y0 + (y1 - y0) * t + ny * j))
    return out


def _scribble(P: Plan, rng: random.Random, text: str, x, y, color):
    """手書きの文字の代わり: 1 文字 = 小さな折れ線 1 本（短く小さい線の集まりを、アプリは文字の塊として扱う）。"""
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        cx = x + 7.5 * i
        pts = [(cx + rng.uniform(0, 5.5), y + rng.uniform(-4.5, 4.5)) for _ in range(rng.randint(3, 5))]
        P.poly("手書き", pts, to_pdf=paper, stroke=color, width=1.2, cap=1)


def make_handwritten(dst: Path) -> None:
    P = Plan(LAYERS + ["手書き"])
    shell(P); furnishings(P); grid(P); dimensions(P); room_names(P)          # 平面図と同じ内容（黒い線の一致で「描き足し」と判定される）
    title_block(P, "3F 平面図", "レイヤー確認用")
    rng = random.Random(3)
    pen = {m.key: tuple(c / 255 for c in m.pen) for m in MATS}
    # 区画: 部屋の内側に斜線（壁から 400 離す）。通路だけは囲みで示す
    for m in MATS:
        if m.key == "B":
            continue
        for x0, y0, x1, y1 in m.rects:
            r = (x0 + 400, y0 + 400, x1 - 400, y1 - 400)
            for p, q in hatch_segments(r, 900, 45 if m.key != "D" else 135, min_len=2200):
                P.poly("手書き", _wobble(rng, [p, q], 50), stroke=pen[m.key], width=2.4, cap=1)
    loop = [(9800, 600), (11800, 600), (11800, 10350), (9800, 10350), (9800, 700)]
    P.poly("手書き", _wobble(rng, loop, 40), stroke=pen["B"], width=2.4, cap=1)
    # メモ（品番の走り書き）と引出線。メモは図面の右の余白に置き、線の先を部屋に落とす
    for m, (tx, ty), target in ((MATS[0], (1035, 150), (18000, 2400)), (MATS[1], (1035, 260), (10800, 9000)),
                                (MATS[2], (1035, 420), (10800, 14500)), (MATS[3], (1035, 500), (18000, 14800))):
        _scribble(P, rng, m.code, tx, ty, pen[m.key])
        P.poly("手書き", _wobble(rng, [to_paper(*target), (tx - 5, ty)], 1.5), to_pdf=paper, stroke=pen[m.key], width=1.6, cap=1)
    # 写真（応接室の床材の見本）と引出線
    ph = product_photo("herringbone", 7)
    P.image("手書き", "Ph0", ph.width, ph.height, ph.tobytes(), 1060, PAGE_H - 630, 70, 70)
    P.poly("手書き", [to_paper(19500, 15300), (1056, 595)], to_pdf=paper, stroke=pen["D"], width=1.6, cap=1)
    write_pdf(P, dst, title="3F 平面図に手書きの指示（架空のサンプル・iPad 手書き形式）", creator="make_mode_samples.py")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out.mkdir(parents=True, exist_ok=True)
    if not PLAN_PDF.exists():
        write_pdf(build(), PLAN_PDF)
    jobs = (("floor_finish_3F.pdf", make_floor_finish), ("instruction_ppt_3F.pdf", make_ppt),
            ("instruction_handwritten_3F.pdf", make_handwritten))
    for name, fn in jobs:
        fn(out / name)
        print(f"{out / name}  ({(out / name).stat().st_size // 1024} KB)")
