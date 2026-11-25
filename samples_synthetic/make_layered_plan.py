"""レイヤー（OCG）付きの架空の平面図 PDF を作る。floortint のレイヤー仕分け（floortint/layers.py）の確認用。

  python make_layered_plan.py                 # このフォルダに layered_plan_3F.pdf を書く
  python make_layered_plan.py 出力先.pdf

外部ライブラリは使わず PDF を直接書く。図面・物件名は架空で、実案件のデータは含まない。
A3 横・1/100、通り芯 X1〜X5 × Y1〜Y4（7200 × 5400 グリッド）の 1 フロア。
建物の部品（shell・grid・room_names・title_block）と PDF の書き出し（write_pdf）は make_mode_samples.py でも使う。

レイヤーと狙い（左が layers.RULES での分類）:
  躯体       躯体        外壁・柱（塗り）
  間仕切     その他      間仕切壁（塗り）。規則に当たらないので「残す」
  建具       建具        扉の開閉弧・枠、EV の扉
  階段       建具        段鼻の線（壁に触れないので階段室は 1 部屋になり、コア判定に当たる）
  什器_新規  什器        机・椅子・会議卓・応接セット
  什器_転用  什器        青い線のロッカー。色の規則なら「席マーカー」で落ちるが、レイヤーなら残る
  植栽       植栽        鉢植え（緑）
  1-STB3     その他      建物側の大梁の想定線（Y2 通りの破線）。規則に当たらず「残す」になり、執務室が南北 2 部屋に割れる。
                         画面のレイヤー一覧で「落とす」に反転すると 1 部屋に戻る（README の既知の課題の再現）
  寸法       寸法        寸法線（黒）。色の規則（灰色の線・長い破線）では落ちない
  通り芯     通り芯      一点鎖線と符号
  OA割付     割付        OA フロアの 600 角の割付線（灰色）
  WP         WP          席の橙の点
  人物       人物        着席・立ちの人（黒）。色の規則では落ちない
  記号       文字・記号  EV シャフトの X 印（落としても、部屋のコア判定は全図形の線で行う）
  室名       文字・記号  室名の文字（図形ではないのでレイヤー一覧には出ない）
  図枠       図枠        図枠と表題欄
"""
from __future__ import annotations

import math
import sys
import zlib
from pathlib import Path

PAGE_W, PAGE_H = 1190.55, 841.89          # A3 横 [pt]
SCALE = 100
MM_PER_PT = SCALE * 25.4 / 72.0
OX, OY = 165.0, 175.0                     # 通り芯 X1/Y1 の交点の紙の上の位置（左上原点）[pt]
GX = [0, 7200, 14400, 21600, 28800]       # 通り芯 X1〜X5 [mm]
GY = [0, 5400, 10800, 16200]              # 通り芯 Y1〜Y4 [mm]
KAPPA = 0.5523                            # 1/4 円を 3 次ベジェで近似する係数

BLACK = (0.0, 0.0, 0.0)
STRUCT_FILL = (0.45, 0.45, 0.45)
WALL_FILL = (0.65, 0.65, 0.65)
OA_GRAY = (0.72, 0.72, 0.72)
BEAM_GRAY = (0.5, 0.5, 0.5)
REUSE_BLUE = (0.1, 0.35, 0.9)
PLANT_GREEN = (0.15, 0.6, 0.2)
WP_ORANGE = (1.0, 0.55, 0.1)

LAYERS = ["躯体", "間仕切", "建具", "階段", "什器_新規", "什器_転用", "植栽", "1-STB3",
          "寸法", "通り芯", "OA割付", "WP", "人物", "記号", "室名", "図枠"]


def mm(x: float, y: float) -> tuple[float, float]:
    """図面の mm（X1/Y1 原点・y 下向き）→ PDF の pt（左下原点・y 上向き）。"""
    return OX + x / MM_PER_PT, PAGE_H - (OY + y / MM_PER_PT)


def paper(x: float, y: float) -> tuple[float, float]:
    """紙の上の pt（左上原点・y 下向き）→ PDF の pt。図枠・表題欄用。"""
    return x, PAGE_H - y


def _f(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _arc(c, rx, ry, a0, a1):
    """楕円弧（a0→a1 [deg]、|a1-a0| ≤ 90）の 3 次ベジェの制御点。座標は mm（y 下向き）。"""
    t0, t1 = math.radians(a0), math.radians(a1)
    k = 4.0 / 3.0 * math.tan((t1 - t0) / 4.0)
    p0 = (c[0] + rx * math.cos(t0), c[1] + ry * math.sin(t0))
    p3 = (c[0] + rx * math.cos(t1), c[1] + ry * math.sin(t1))
    p1 = (p0[0] - k * rx * math.sin(t0), p0[1] + k * ry * math.cos(t0))
    p2 = (p3[0] + k * rx * math.sin(t1), p3[1] - k * ry * math.cos(t1))
    return p0, p1, p2, p3


class Plan:
    """レイヤーごとに内容ストリームの断片をためる。図形 1 つ = PDF のパス 1 つ（CAD の書き出しと同じ粒度）。"""

    def __init__(self, layers=LAYERS):
        self.layers = list(layers)
        self.ops: dict[str, list[str]] = {name: [] for name in self.layers}
        self.images: dict[str, tuple[int, int, bytes]] = {}   # 名前 → (幅 px, 高さ px, RGB 8bit の生データ)
        self.alphas: dict[float, str] = {}                     # 不透明度 → ExtGState 名
        self.links: list[tuple[tuple, str]] = []               # (PDF の pt の矩形, URI)

    # ---- 図形
    def _emit(self, layer, body, stroke=BLACK, fill=None, width=0.25, dash=None, alpha=None, cap=None):
        st = []
        if alpha is not None:
            st.append("/%s gs" % self.alphas.setdefault(alpha, f"GS{len(self.alphas)}"))
        if stroke is not None:
            st.append("%s %s %s RG %s w" % (_f(stroke[0]), _f(stroke[1]), _f(stroke[2]), _f(width)))
            st.append("[%s] 0 d" % " ".join(_f(v) for v in dash) if dash else "[] 0 d")
            if cap is not None:
                st.append("%d J %d j" % (cap, cap))
        if fill is not None:
            st.append("%s %s %s rg" % tuple(_f(v) for v in fill))
        paint = "B" if stroke is not None and fill is not None else ("f" if fill is not None else "S")
        self.ops[layer].append("q " + " ".join(st) + "\n" + body + "\n" + paint + " Q")

    def poly(self, layer, pts, closed=False, to_pdf=mm, **kw):
        q = [to_pdf(*p) for p in pts]
        body = "%s %s m " % (_f(q[0][0]), _f(q[0][1])) + " ".join("%s %s l" % (_f(x), _f(y)) for x, y in q[1:])
        self._emit(layer, body + (" h" if closed else ""), **kw)

    def line(self, layer, x0, y0, x1, y1, **kw):
        self.poly(layer, [(x0, y0), (x1, y1)], **kw)

    def rect(self, layer, x0, y0, x1, y1, **kw):
        self.poly(layer, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True, **kw)

    def arcs(self, layer, c, rx, ry, a0, a1, closed=False, **kw):
        """a0→a1 [deg] の楕円弧（90° ごとに分割）。"""
        n = max(1, math.ceil(abs(a1 - a0) / 90.0 - 1e-9))
        step = (a1 - a0) / n
        body = []
        for i in range(n):
            p0, p1, p2, p3 = (mm(*p) for p in _arc(c, rx, ry, a0 + i * step, a0 + (i + 1) * step))
            if i == 0:
                body.append("%s %s m" % (_f(p0[0]), _f(p0[1])))
            body.append("%s %s %s %s %s %s c" % (_f(p1[0]), _f(p1[1]), _f(p2[0]), _f(p2[1]), _f(p3[0]), _f(p3[1])))
        self._emit(layer, " ".join(body) + (" h" if closed else ""), **kw)

    def circle(self, layer, cx, cy, r, **kw):
        self.arcs(layer, (cx, cy), r, r, 0, 360, closed=True, **kw)

    # ---- 文字
    def text(self, layer, s, x, y, size, font="F1", align="center", rotate=False, to_pdf=mm):
        """(x, y) は文字列の中心（align="left" なら左端の中心）。F1 = 日本語（UCS-2）、F2 = Helvetica。"""
        w = sum(0.5 if ord(ch) < 128 else 1.0 for ch in s) * size if font == "F1" else 0.56 * size * len(s)
        px, py = to_pdf(x, y)
        off = -w / 2 if align == "center" else 0.0
        if rotate:                                            # 90° 左回り（下から上へ読む縦の寸法値）
            tm = "0 1 -1 0 %s %s" % (_f(px + 0.35 * size), _f(py + off))
        else:
            tm = "1 0 0 1 %s %s" % (_f(px + off), _f(py - 0.35 * size))
        enc = "<%s>" % s.encode("utf-16-be").hex().upper() if font == "F1" else "(%s)" % s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        self.ops[layer].append("q 0 g BT /%s %s Tf %s Tm %s Tj ET Q" % (font, _f(size), tm, enc))

    # ---- 画像・リンク（指示書の見本用。座標は PDF の pt、左下原点）
    def image(self, layer, name, w_px, h_px, rgb: bytes, x, y, w, h):
        self.images[name] = (w_px, h_px, rgb)
        self.ops[layer].append("q %s 0 0 %s %s %s cm /%s Do Q" % (_f(w), _f(h), _f(x), _f(y), name))

    def link(self, rect, uri):
        self.links.append((tuple(rect), uri))

    # ---- 部品
    def wall(self, layer, x0, y0, x1, y1, t, openings=(), fill=WALL_FILL, width=0.18):
        """軸に平行な壁を塗りの矩形で描く。openings は (開口の始点 mm, 幅 mm)。"""
        horiz = y0 == y1
        a, b = sorted((x0, x1)) if horiz else sorted((y0, y1))
        pos, pieces = a, []
        for s, w in sorted(openings):
            if s > pos:
                pieces.append((pos, s))
            pos = s + w
        if pos < b:
            pieces.append((pos, b))
        for p0, p1 in pieces:
            if horiz:
                self.rect(layer, p0, y0 - t / 2, p1, y0 + t / 2, fill=fill, width=width)
            else:
                self.rect(layer, x0 - t / 2, p0, x0 + t / 2, p1, fill=fill, width=width)

    def door(self, horiz, wall_at, start, width, swing, t=100):
        """片開き扉: 開閉弧（ベジェ 1 本）と扉の線、反対側の面に沓摺の線（部屋を閉じる）。swing は開く側（+1 / -1）。"""
        fs, fo = wall_at + swing * t / 2, wall_at - swing * t / 2
        if horiz:
            self.line("建具", start, fs, start, fs + swing * width, width=0.25)
            a0 = 90.0 if swing > 0 else -90.0
            self.arcs("建具", (start, fs), width, width, a0, 0.0, width=0.18)
            self.line("建具", start, fo, start + width, fo, width=0.18)
        else:
            self.line("建具", fs, start, fs + swing * width, start, width=0.25)
            a0 = 0.0 if swing > 0 else 180.0
            self.arcs("建具", (fs, start), width, width, a0, 90.0, width=0.18)
            self.line("建具", fo, start, fo, start + width, width=0.18)

    def chair(self, cx, cy, back, layer="什器_新規"):
        """椅子: 座面の円 ＋ 背もたれの半楕円。back は背の向き（up / down / left / right）。"""
        self.circle(layer, cx, cy, 250, width=0.2)
        dx, dy, a0 = {"up": (0, -120, 180), "down": (0, 120, 0), "left": (-120, 0, 90), "right": (120, 0, 270)}[back]
        rx, ry = (280, 200) if back in ("up", "down") else (200, 280)
        self.arcs(layer, (cx + dx, cy + dy), rx, ry, a0, a0 + 180, width=0.25)

    def person(self, cx, cy):
        self.circle("人物", cx, cy, 110, width=0.2)
        self.arcs("人物", (cx, cy), 230, 100, 0, 360, closed=True, width=0.2)

    def planter(self, cx, cy):
        self.circle("植栽", cx, cy, 300, stroke=PLANT_GREEN, width=0.25)
        for i in range(6):
            a = math.radians(i * 60 + 15)
            self.circle("植栽", cx + 300 * math.cos(a), cy + 300 * math.sin(a), 200, stroke=PLANT_GREEN, width=0.2)


COLUMNS = [(gx - 400, gy - 400, gx + 400, gy + 400) for gx in GX for gy in GY]     # 柱 800 角 [mm]
ROOM_NAMES = (("執務室", 3000, 4600), ("会議室A", 3600, 11500), ("会議室B", 10800, 11500), ("応接室", 18000, 11600),
              ("EVホール", 24600, 6750), ("階段", 23400, 4850), ("男子WC", 23400, 12000), ("女子WC", 27000, 12000),
              ("EV", 26100, 4050), ("EV", 27900, 4050), ("PS", 26100, 1350), ("EPS", 27900, 1350))


def build() -> Plan:
    P = Plan()
    shell(P)
    furnishings(P)
    grid(P)
    dimensions(P)
    room_names(P)
    title_block(P, "3F 平面図", "レイヤー確認用")
    return P


def shell(P: Plan, outline: bool = False) -> None:
    """躯体・間仕切・建具・階段。outline=True なら壁を塗らずに黒の外形線だけで描く（床仕上図の描き方）。"""
    T_EXT, T_PART = 300, 100
    part = {"fill": None, "width": 0.25} if outline else {}
    # ---- 躯体: 外壁（グリッドの外周に中心）と柱 800 角
    W, H = GX[-1], GY[-1]
    for x0, y0, x1, y1 in ((0, 0, W, 0), (0, H, W, H), (0, 0, 0, H), (W, 0, W, H)):
        P.wall("躯体", x0, y0, x1, y1, T_EXT, **({"fill": None, "width": 0.35} if outline else {"fill": STRUCT_FILL, "width": 0.35}))
    for c in COLUMNS:
        P.rect("躯体", *c, fill=STRUCT_FILL, width=0.35)

    # ---- 間仕切（開口つき）と扉
    P.wall("間仕切", 0, 10800, 21600, 10800, T_PART, openings=[(5600, 900), (12800, 900), (19600, 900)], **part)
    P.wall("間仕切", 7200, 10800, 7200, 16200, T_PART, **part)
    P.wall("間仕切", 14400, 10800, 14400, 16200, T_PART, **part)
    P.wall("間仕切", 21600, 0, 21600, 16200, T_PART, openings=[(6300, 900)], **part)
    P.wall("間仕切", 21600, 5400, 28800, 5400, T_PART, openings=[(22800, 900)], **part)
    P.wall("間仕切", 25200, 0, 25200, 5400, T_PART, **part)
    P.wall("間仕切", 27000, 0, 27000, 5400, T_PART, **part)
    P.wall("間仕切", 25200, 2700, 28800, 2700, T_PART, **part)
    P.wall("間仕切", 21600, 8100, 28800, 8100, T_PART, openings=[(22400, 800), (26000, 800)], **part)
    P.wall("間仕切", 25200, 8100, 25200, 16200, T_PART, **part)
    for start in (5600, 12800, 19600):
        P.door(True, 10800, start, 900, +1)                  # 会議室A・会議室B・応接室（室内へ開く）
    P.door(False, 21600, 6300, 900, +1)                      # 執務室 → EVホール
    P.door(True, 5400, 22800, 900, +1)                       # 階段 → EVホール
    P.door(True, 8100, 22400, 800, +1)                       # 男子WC
    P.door(True, 8100, 26000, 800, +1)                       # 女子WC
    for x0 in (25500, 27300):                                # EV の扉（引戸 2 枚。開口は設けない）
        P.rect("建具", x0, 5450, x0 + 600, 5500, width=0.2)
        P.rect("建具", x0 + 600, 5500, x0 + 1200, 5550, width=0.2)

    # ---- 階段: 段鼻 15 本（踏面 240）と中央の手摺。壁には触れない。
    # 踏面を 250 以上にすると、等間隔の長い直線の群として OA 割付のマス目の判定（cleanplan.grid_line_ids）に当たり、段鼻が消える
    for i in range(15):
        y = 900 + i * 240
        P.line("階段", 21900, y, 24900, y, width=0.2)
    P.rect("階段", 23350, 750, 23450, 4410, width=0.2)


def furnishings(P: Plan) -> None:
    """記号・什器・植栽・人物・WP・梁の想定線・OA 割付（平面図だけに描くもの）。"""
    # EV シャフトの X 印（記号）。部屋の内法（壁の内側）に収める。はみ出すとコア判定（rooms.mark_core_rooms）に当たらない
    for x0, x1 in ((25250, 26950), (27050, 28650)):
        P.line("記号", x0, 2750, x1, 5350, width=0.2)
        P.line("記号", x1, 2750, x0, 5350, width=0.2)

    # ---- 執務室: 6 人掛けの島 × 8、椅子・人・WP
    for ix in (1200, 6000, 12000, 16800):
        for iy in (1800, 7200):
            for i in range(3):
                for j in range(2):
                    x0, y0 = ix + i * 1200, iy + j * 700
                    P.rect("什器_新規", x0, y0, x0 + 1200, y0 + 700, width=0.25)
                    cx = x0 + 600
                    cy = iy - 350 if j == 0 else iy + 1400 + 350
                    P.chair(cx, cy, "up" if j == 0 else "down")
                    P.circle("WP", cx, cy, 75, stroke=None, fill=WP_ORANGE)
                    if (i + j + ix // 1200) % 2 == 0:
                        P.person(cx, cy)
    # 転用のロッカー（青）
    for x0 in (300, 1200, 2100, 3000, 12000, 12900, 13800, 14700):
        P.rect("什器_転用", x0, 200, x0 + 900, 650, stroke=REUSE_BLUE, width=0.3)
        P.line("什器_転用", x0, 200, x0 + 900, 650, stroke=REUSE_BLUE, width=0.2)
    for cx, cy in ((800, 4500), (10200, 4500), (21000, 4500), (27900, 6750)):
        P.planter(cx, cy)
    for cx, cy in ((23200, 7300), (26200, 6200)):          # EVホールの立ち姿
        P.person(cx, cy)

    # ---- 会議室A・B・応接室
    P.rect("什器_新規", 1800, 12900, 5400, 14100, width=0.25)
    for x in (2400, 3600, 4800):
        P.chair(x, 12550, "up"); P.chair(x, 14450, "down")
    P.chair(1450, 13500, "left"); P.chair(5750, 13500, "right")
    P.rect("什器_新規", 9600, 12900, 12000, 14100, width=0.25)
    for x in (10200, 11400):
        P.chair(x, 12550, "up"); P.chair(x, 14450, "down")
    P.chair(9250, 13500, "left"); P.chair(12350, 13500, "right")
    P.rect("什器_新規", 16200, 14800, 19200, 15600, width=0.25)        # ソファ
    P.line("什器_新規", 16200, 15350, 19200, 15350, width=0.2)
    P.rect("什器_新規", 16800, 13400, 18600, 14100, width=0.25)        # ローテーブル
    P.chair(17100, 12700, "up"); P.chair(18300, 12700, "up")

    # ---- 1-STB3: 建物側の大梁の想定線。Y2 通りを柱から柱へ結ぶ 1 本の折れ線（執務室を南北に横断する）。
    # 1 本の直線にすると、長さと範囲が OA 割付の線とそろったときに割付のマス目の一部として落ちる
    P.poly("1-STB3", [(150, 5400), (7200, 5400), (14400, 5400), (21550, 5400)], stroke=BEAM_GRAY, width=0.18, dash=(4, 2))

    # ---- OA割付（600 角）
    for x in range(600, 21600, 600):
        P.line("OA割付", x, 150, x, 10750, stroke=OA_GRAY, width=0.1)
    for y in range(600, 10800, 600):
        P.line("OA割付", 150, y, 21550, y, stroke=OA_GRAY, width=0.1)


def grid(P: Plan) -> None:
    """通り芯（一点鎖線）と符号。符号 X1…/Y1… は床仕上図・指示書との位置合わせ（pdfdoc.align）にも使われる。"""
    W, H = GX[-1], GY[-1]
    for k, gx in enumerate(GX, start=1):
        P.line("通り芯", gx, -3600, gx, H + 1000, width=0.18, dash=(6, 1.5, 1, 1.5))
        P.circle("通り芯", gx, -4000, 400, width=0.25)
        P.text("通り芯", f"X{k}", gx, -4000, 7, font="F2")
    for k, gy in enumerate(GY, start=1):
        P.line("通り芯", -3600, gy, W + 1000, gy, width=0.18, dash=(6, 1.5, 1, 1.5))
        P.circle("通り芯", -4000, gy, 400, width=0.25)
        P.text("通り芯", f"Y{k}", -4000, gy, 7, font="F2")


def dimensions(P: Plan) -> None:
    """寸法（黒）: 上と左に 2 段（スパンと全長）。"""
    def dim_h(y, xs):
        P.line("寸法", xs[0], y, xs[-1], y, width=0.15)
        for x in xs:
            P.line("寸法", x, y + 300, x, -450, width=0.15)                # 引出線（外壁の手前まで）
            P.line("寸法", x - 120, y + 120, x + 120, y - 120, width=0.3)  # 斜めの止め
        for a, b in zip(xs, xs[1:]):
            P.text("寸法", str(b - a), (a + b) / 2, y - 250, 6, font="F2")

    def dim_v(x, ys):
        P.line("寸法", x, ys[0], x, ys[-1], width=0.15)
        for y in ys:
            P.line("寸法", x + 300, y, -450, y, width=0.15)
            P.line("寸法", x - 120, y + 120, x + 120, y - 120, width=0.3)
        for a, b in zip(ys, ys[1:]):
            P.text("寸法", str(b - a), x - 250, (a + b) / 2, 6, font="F2", rotate=True)

    dim_h(-1500, GX); dim_h(-2600, [GX[0], GX[-1]])
    dim_v(-1500, GY); dim_v(-2600, [GY[0], GY[-1]])


def room_names(P: Plan) -> None:
    for name, x, y in ROOM_NAMES:
        P.text("室名", name, x, y, 9)


def title_block(P: Plan, title: str, note: str) -> None:
    """図枠と表題欄（紙の上の pt）。図面名称の欄の文字でページの種類（平面図／床仕上図）が決まる（pdfdoc.classify）。"""
    P.rect("図枠", 20, 20, PAGE_W - 20, PAGE_H - 20, to_pdf=paper, width=0.7)
    tx0, tx1, ty = 870.0, PAGE_W - 20, [701.0, 725.0, 753.0, 777.0, PAGE_H - 20]
    P.rect("図枠", tx0, ty[0], tx1, ty[-1], to_pdf=paper, width=0.5)
    for y in ty[1:-1]:
        P.line("図枠", tx0, y, tx1, y, to_pdf=paper, width=0.3)
    P.line("図枠", 1030, ty[2], 1030, ty[4], to_pdf=paper, width=0.3)
    rows = (("工事名", "サンプルオフィス（架空）", tx0, ty[0]), ("図面名称", title, tx0, ty[1]),
            ("縮尺", "1/100(A3)", tx0, ty[2]), ("図面番号", "A-03", 1030, ty[2]),
            ("日付", "2026.10.30", tx0, ty[3]), ("備考", note, 1030, ty[3]))
    for label, value, x, y in rows:
        P.text("図枠", label, x + 4, y + 6, 5.5, align="left", to_pdf=paper)
        P.text("図枠", value, x + 8, y + 16.5, 9 if label != "図面名称" else 11, align="left", to_pdf=paper)


def write_pdf(P: Plan, path: Path, title: str = "3F 平面図（架空のサンプル・レイヤー確認用）", creator: str = "make_layered_plan.py",
              page_size: tuple[float, float] = (PAGE_W, PAGE_H), shift: tuple[float, float] = (0.0, 0.0)) -> None:
    """shift [pt] はページ全体の平行移動（同じ CAD データを別の用紙位置に出した図面の再現）。"""
    objs: list[bytes] = []

    def add(b: bytes | str) -> int:
        objs.append(b.encode("latin-1") if isinstance(b, str) else b)
        return len(objs)

    def pdf_text(s: str) -> str:
        return "<FEFF%s>" % s.encode("utf-16-be").hex().upper()

    catalog = add("")                     # 後で埋める
    pages = add("")
    page = add("")
    font_desc = add("<< /Type /FontDescriptor /FontName /HeiseiKakuGo-W5 /Flags 4 /FontBBox [-92 -250 1010 922] "
                    "/ItalicAngle 0 /Ascent 752 /Descent -221 /CapHeight 737 /StemV 114 >>")
    cid_font = add("<< /Type /Font /Subtype /CIDFontType0 /BaseFont /HeiseiKakuGo-W5 "
                   "/CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 2 >> "
                   f"/FontDescriptor {font_desc} 0 R /DW 1000 /W [1 95 500 231 325 500] >>")
    f1 = add(f"<< /Type /Font /Subtype /Type0 /BaseFont /HeiseiKakuGo-W5 /Encoding /UniJIS-UCS2-H /DescendantFonts [{cid_font} 0 R] >>")
    f2 = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    ocg = {name: add(f"<< /Type /OCG /Name {pdf_text(name)} >>") for name in P.layers}

    content = []
    for i, name in enumerate(P.layers):
        if P.ops[name]:
            content.append(f"/OC /L{i} BDC\n" + "\n".join(P.ops[name]) + "\nEMC")
    if shift != (0.0, 0.0):
        content = [f"q 1 0 0 1 {_f(shift[0])} {_f(shift[1])} cm"] + content + ["Q"]
    data = zlib.compress("\n".join(content).encode("latin-1"), 9)
    contents = add(f"<< /Length {len(data)} /Filter /FlateDecode >>\nstream\n".encode("latin-1") + data + b"\nendstream")
    info = add(f"<< /Title {pdf_text(title)} /Creator ({creator}) >>")
    extra = ""
    if P.alphas:
        extra += " /ExtGState << " + " ".join(f"/{nm} << /Type /ExtGState /ca {_f(a)} /CA {_f(a)} >>" for a, nm in P.alphas.items()) + " >>"
    if P.images:
        xo = []
        for nm, (w, h, rgb) in P.images.items():
            z = zlib.compress(rgb, 9)
            ref = add(f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 "
                      f"/Filter /FlateDecode /Length {len(z)} >>\nstream\n".encode("latin-1") + z + b"\nendstream")
            xo.append(f"/{nm} {ref} 0 R")
        extra += " /XObject << " + " ".join(xo) + " >>"
    annots = ""
    if P.links:
        refs_a = [add(f"<< /Type /Annot /Subtype /Link /Rect [{' '.join(_f(v) for v in r)}] /Border [0 0 0] "
                      f"/A << /S /URI /URI ({uri}) >> >>") for r, uri in P.links]
        annots = " /Annots [" + " ".join(f"{a} 0 R" for a in refs_a) + "]"

    refs = " ".join(f"{ocg[n]} 0 R" for n in P.layers)
    props = " ".join(f"/L{i} {ocg[n]} 0 R" for i, n in enumerate(P.layers))
    objs[catalog - 1] = (f"<< /Type /Catalog /Pages {pages} 0 R /OCProperties << /OCGs [{refs}] "
                         f"/D << /Name {pdf_text('既定')} /Order [{refs}] /ON [{refs}] /OFF [] >> >> >>").encode("latin-1")
    objs[pages - 1] = f"<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>".encode("latin-1")
    objs[page - 1] = (f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 {page_size[0]} {page_size[1]}] "
                      f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> /Properties << {props} >>{extra} >> "
                      f"/Contents {contents} 0 R{annots} >>").encode("latin-1")

    out = bytearray(b"%PDF-1.6\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root {catalog} 0 R /Info {info} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


if __name__ == "__main__":
    dst = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("layered_plan_3F.pdf")
    P = build()
    write_pdf(P, dst)
    print(f"{dst}  ({dst.stat().st_size // 1024} KB, " + ", ".join(f"{n}:{len(P.ops[n])}" for n in P.layers) + ")")
