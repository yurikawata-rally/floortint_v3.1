"""PDF の読み取り層（pypdfium2 製。BSD-3 / Apache-2.0）。

以前は PyMuPDF（AGPL）を使っていた。社内方針で AGPL を避けるため、FloorTint と QuantityTakeoff が使っていた範囲だけを
pypdfium2（pdfium）の上に同じ形で作り直した。呼び出し側のコードは ``pymupdf`` を ``pdfio`` に読み替えるだけで動く。

提供するもの:
  open(path | stream=bytes) → Document（len / [i] / for）
  Document[i] → Page: rect, rotation, number, parent, get_drawings(), get_text() / get_text("dict") / get_text("text", clip=),
                      get_pixmap(matrix=, colorspace=, alpha=, clip=), get_image_info(xrefs=True), get_links(), remove_rotation()
  Rect / Point / Quad / Matrix / csRGB / csGRAY / Pixmap（画像の実データ取り出し）
  render_drawings(width, height, [(Drawing, 上書き属性)], scale) → RGB 配列（OpenCV で描く。pdfdoc から呼ぶ）

座標はすべて「表示どおり」（左上原点・右下向き・ページの回転を反映）で、PyMuPDF の remove_rotation() 後と同じ。
図形（get_drawings）の辞書は PyMuPDF と同じキー: rect, items（("l",p,q) / ("c",p,c1,c2,q) / ("re",Rect)）, color, fill, width,
dashes, closePath, even_odd, layer（OCG 名）, fill_opacity, stroke_opacity。線幅は図形の変換行列の拡大率を掛けた紙の上の値。
"""
from __future__ import annotations

import builtins
import ctypes
import math

import cv2
import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as R


# ---------------------------------------------------------------- 幾何
class Point:
    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        if isinstance(x, (tuple, list, Point)):
            x, y = (x.x, x.y) if isinstance(x, Point) else (x[0], x[1])
        self.x = float(x); self.y = float(y)

    def __iter__(self):
        yield self.x; yield self.y

    def __getitem__(self, i):
        return (self.x, self.y)[i]

    def __repr__(self):
        return f"Point({self.x:.4g}, {self.y:.4g})"


class Rect:
    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, *args):
        if len(args) == 1:
            a = args[0]
            vals = (a.x0, a.y0, a.x1, a.y1) if isinstance(a, Rect) else tuple(a)
        elif len(args) == 0:
            vals = (0.0, 0.0, 0.0, 0.0)
        elif len(args) == 2:
            vals = (args[0][0], args[0][1], args[1][0], args[1][1])
        else:
            vals = args
        self.x0, self.y0, self.x1, self.y1 = (float(v) for v in vals)

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def is_empty(self) -> bool:
        return self.x0 >= self.x1 or self.y0 >= self.y1

    @property
    def tl(self) -> Point:
        return Point(self.x0, self.y0)

    @property
    def br(self) -> Point:
        return Point(self.x1, self.y1)

    def get_area(self) -> float:
        return 0.0 if self.is_empty else (self.x1 - self.x0) * (self.y1 - self.y0)

    def __iter__(self):
        yield self.x0; yield self.y0; yield self.x1; yield self.y1

    def __getitem__(self, i):
        return (self.x0, self.y0, self.x1, self.y1)[i]

    def __len__(self):
        return 4

    def __eq__(self, o):
        return isinstance(o, Rect) and tuple(self) == tuple(o)

    def __hash__(self):
        return hash(tuple(self))

    def __and__(self, o) -> "Rect":
        o = Rect(o)
        x0, y0, x1, y1 = max(self.x0, o.x0), max(self.y0, o.y0), min(self.x1, o.x1), min(self.y1, o.y1)
        if x0 > x1 or y0 > y1:
            return Rect(0.0, 0.0, 0.0, 0.0)
        return Rect(x0, y0, x1, y1)

    def __or__(self, o) -> "Rect":
        o = Rect(o)
        return Rect(min(self.x0, o.x0), min(self.y0, o.y0), max(self.x1, o.x1), max(self.y1, o.y1))

    def intersects(self, o) -> bool:
        """PyMuPDF と同じく、面積のある重なりがあるときだけ True（幅 0・高さ 0 の矩形は常に False）。"""
        o = Rect(o)
        if self.is_empty or o.is_empty:
            return False
        return max(self.x0, o.x0) < min(self.x1, o.x1) and max(self.y0, o.y0) < min(self.y1, o.y1)

    def contains(self, o) -> bool:
        if isinstance(o, Point) or (isinstance(o, (tuple, list)) and len(o) == 2):
            x, y = (o.x, o.y) if isinstance(o, Point) else (o[0], o[1])
            return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1
        o = Rect(o)
        return self.x0 <= o.x0 and self.y0 <= o.y0 and o.x1 <= self.x1 and o.y1 <= self.y1

    def __contains__(self, o) -> bool:
        return self.contains(o)

    def __repr__(self):
        return f"Rect({self.x0:.4g}, {self.y0:.4g}, {self.x1:.4g}, {self.y1:.4g})"


class Quad:
    __slots__ = ("ul", "ur", "ll", "lr")

    def __init__(self, ul, ur, ll, lr):
        self.ul, self.ur, self.ll, self.lr = Point(ul), Point(ur), Point(ll), Point(lr)


class Matrix:
    """拡大率だけを持つ（描画用）。Matrix(s) / Matrix(sx, sy)。"""
    __slots__ = ("a", "b", "c", "d", "e", "f")

    def __init__(self, *args):
        if len(args) == 1:
            self.a = self.d = float(args[0])
        else:
            self.a, self.d = float(args[0]), float(args[1])
        self.b = self.c = self.e = self.f = 0.0


class _Colorspace:
    def __init__(self, n: int, name: str):
        self.n, self.name = n, name


csRGB = _Colorspace(3, "DeviceRGB")
csGRAY = _Colorspace(1, "DeviceGray")


def _mcompose(A, B):
    """行ベクトル規約（p' = p·M）で A を掛けてから B を掛ける行列。"""
    a, b, c, d, e, f = A
    a2, b2, c2, d2, e2, f2 = B
    return (a * a2 + b * c2, a * b2 + b * d2, c * a2 + d * c2, c * b2 + d * d2, e * a2 + f * c2 + e2, e * b2 + f * d2 + f2)


def _mapply(M, x, y):
    a, b, c, d, e, f = M
    return (a * x + c * y + e, b * x + d * y + f)


def _mscale(M) -> float:
    a, b, c, d, _, _ = M
    return math.sqrt(abs(a * d - b * c))


_FLIP_UNIT = (1.0, 0.0, 0.0, -1.0, 0.0, 1.0)      # (u, v 下向き) → (u, v 上向き)


# ---------------------------------------------------------------- 文書・ページ
class Pixmap:
    """描画結果または画像の実データ。samples は行優先の bytes、n はチャネル数（1 または 3）、alpha は 0。"""

    def __init__(self, *args):
        self.alpha = 0
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            arr = args[0]
        elif len(args) == 2 and isinstance(args[0], Document):
            arr = args[0]._image_array(int(args[1]))
        elif len(args) == 2 and isinstance(args[0], _Colorspace):
            arr = args[1]._arr()
            if args[0].n == 1 and arr.ndim == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            elif args[0].n == 3 and arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif len(args) == 2 and isinstance(args[0], Pixmap):
            arr = args[0]._arr()
        else:
            raise TypeError("Pixmap(array) / Pixmap(doc, xref) / Pixmap(csRGB, pix) / Pixmap(pix, 0)")
        arr = np.ascontiguousarray(arr)
        self._a = arr
        self.h, self.w = int(arr.shape[0]), int(arr.shape[1])
        self.n = 1 if arr.ndim == 2 else int(arr.shape[2])
        self.samples = arr.tobytes()

    def _arr(self) -> np.ndarray:
        return self._a

    def tobytes(self, output: str = "png") -> bytes:
        a = self._a
        bgr = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png" if output == "png" else f".{output}", bgr)
        return buf.tobytes()

    def save(self, path: str) -> None:
        with builtins.open(path, "wb") as f:
            f.write(self.tobytes("png" if str(path).lower().endswith(".png") else "jpg"))


def _wbuf(n: int):
    return (ctypes.c_ushort * n)()


def _wstr(buf, ln) -> str:
    return bytes(buf)[:ln.value].decode("utf-16-le", "ignore").rstrip("\x00")


def _pdf_string(raw: bytes) -> str:
    """PDF 文字列（BOM 付き UTF-16BE か PDFDocEncoding≒Latin-1、まれに UTF-8）を復号する。"""
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", "ignore").rstrip("\x00")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "ignore")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", "ignore")


_OC_NAME_CACHE: dict[int, str | None] = {}


def _oc_layer(raw_obj) -> str | None:
    """図形の marked content（/OC）から OCG の名前を取る。無ければ None。"""
    nm = R.FPDFPageObj_CountMarks(raw_obj)
    for i in range(nm):
        mk = R.FPDFPageObj_GetMark(raw_obj, i)
        b = _wbuf(64); ln = ctypes.c_ulong(); R.FPDFPageObjMark_GetName(mk, b, 128, ln)
        if _wstr(b, ln) != "OC":
            continue
        n = ctypes.c_ulong()
        if not R.FPDFPageObjMark_GetParamBlobValue(mk, b"Name", None, 0, n) or n.value == 0:
            vb = _wbuf(256); vl = ctypes.c_ulong()
            if R.FPDFPageObjMark_GetParamStringValue(mk, b"Name", vb, 512, vl):
                return _wstr(vb, vl) or None
            return None
        buf = (ctypes.c_ubyte * n.value)()
        R.FPDFPageObjMark_GetParamBlobValue(mk, b"Name", buf, n.value, n)
        return _pdf_string(bytes(buf)[:n.value]) or None
    return None


class Page:
    def __init__(self, doc: "Document", index: int):
        self.parent = doc
        self.number = index
        self._pg = doc._pdf[index]
        self.rotation = int(self._pg.get_rotation())
        w, h = self._pg.get_size()
        self.rect = Rect(0.0, 0.0, float(w), float(h))
        bx0, by0, bx1, by1 = self._pg.get_cropbox() if hasattr(self._pg, "get_cropbox") else self._pg.get_mediabox()
        mb = self._pg.get_mediabox()
        bx0, by0, bx1, by1 = max(bx0, mb[0]), max(by0, mb[1]), min(bx1, mb[2]), min(by1, mb[3])
        rot = self.rotation % 360
        if rot == 90:
            self._D = (0.0, 1.0, 1.0, 0.0, -by0, -bx0)
        elif rot == 180:
            self._D = (-1.0, 0.0, 0.0, 1.0, bx1, -by0)
        elif rot == 270:
            self._D = (0.0, -1.0, -1.0, 0.0, by1, bx1)
        else:
            self._D = (1.0, 0.0, 0.0, -1.0, -bx0, by1)
        self._drawings = None
        self._textpage = None
        self._images = None

    # ---- 互換
    def remove_rotation(self) -> None:
        """座標は常に表示どおりなので何もしない（PyMuPDF 互換）。"""
        return None

    def _to_display(self, x: float, y: float) -> tuple[float, float]:
        return _mapply(self._D, x, y)

    def _from_display(self, x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f = self._D
        det = a * d - b * c
        ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
        ie, if_ = -(e * ia + f * ic), -(e * ib + f * id_)
        return (ia * x + ic * y + ie, ib * x + id_ * y + if_)

    def _obj_matrix(self, o) -> tuple:
        """入れ子のフォームも含めた、図形の座標 → ページのユーザ空間の行列。"""
        M = tuple(o.get_matrix().get())
        c = getattr(o, "container", None)
        depth = 0
        while c is not None and depth < 16:
            M = _mcompose(M, tuple(c.get_matrix().get()))
            c = getattr(c, "container", None); depth += 1
        return M

    # ---- 図形
    def get_drawings(self) -> list[dict]:
        if self._drawings is None:
            self._drawings = self._read_drawings()
        return self._drawings

    def _read_drawings(self) -> list[dict]:
        out: list[dict] = []
        r_, g_, b_, a_ = ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint(), ctypes.c_uint()
        wv = ctypes.c_float(); fill_m = ctypes.c_int(); stroke_m = ctypes.c_int()
        px = ctypes.c_float(); py = ctypes.c_float()
        cont_layers: dict[int, str | None] = {}
        for o in self._pg.get_objects(max_depth=8):
            if o.type != R.FPDF_PAGEOBJ_PATH:
                continue
            raw = o.raw
            n = R.FPDFPath_CountSegments(raw)
            if n <= 0:
                continue
            M = _mcompose(self._obj_matrix(o), self._D) if o.level else _mcompose(tuple(o.get_matrix().get()), self._D)
            pts = []
            for i in range(n):
                sg = R.FPDFPath_GetPathSegment(raw, i)
                R.FPDFPathSegment_GetPoint(sg, px, py)
                pts.append((R.FPDFPathSegment_GetType(sg), _mapply(M, px.value, py.value), bool(R.FPDFPathSegment_GetClose(sg))))
            items: list = []
            cur = None; start = None; closed = False; bez: list = []
            xs: list[float] = []; ys: list[float] = []
            for typ, p, cl in pts:
                xs.append(p[0]); ys.append(p[1])
                if typ == R.FPDF_SEGMENT_MOVETO:
                    cur = p; start = p; bez = []
                elif typ == R.FPDF_SEGMENT_LINETO:
                    if cur is not None:
                        items.append(("l", Point(cur), Point(p)))
                    cur = p
                elif typ == R.FPDF_SEGMENT_BEZIERTO:
                    bez.append(p)
                    if len(bez) == 3:
                        if cur is not None:
                            items.append(("c", Point(cur), Point(bez[0]), Point(bez[1]), Point(bez[2])))
                        cur = bez[2]; bez = []
                if cl:
                    closed = True
                    if cur is not None and start is not None and typ == R.FPDF_SEGMENT_LINETO and abs(cur[0] - start[0]) + abs(cur[1] - start[1]) < 1e-6:
                        pass
            if not items:
                continue
            # 矩形（re 演算子）: moveto ＋ 3〜4 lineto で軸に平行、閉じている
            if closed and 4 <= len(items) <= 5 and all(it[0] == "l" for it in items):
                q = [items[0][1]] + [it[2] for it in items[:3]]
                if len(items) == 4 or (abs(items[3][2].x - q[0].x) + abs(items[3][2].y - q[0].y) < 1e-6):
                    ax = sorted({round(v.x, 3) for v in q}); ay = sorted({round(v.y, 3) for v in q})
                    if len(ax) == 2 and len(ay) == 2 and all(round(v.x, 3) in ax and round(v.y, 3) in ay for v in q):
                        items = [("re", Rect(ax[0], ay[0], ax[1], ay[1]))]
            R.FPDFPath_GetDrawMode(raw, fill_m, stroke_m)
            color = None; fill = None; s_op = 1.0; f_op = 1.0
            if stroke_m.value and R.FPDFPageObj_GetStrokeColor(raw, r_, g_, b_, a_):
                color = (r_.value / 255.0, g_.value / 255.0, b_.value / 255.0); s_op = a_.value / 255.0
            if fill_m.value and R.FPDFPageObj_GetFillColor(raw, r_, g_, b_, a_):
                fill = (r_.value / 255.0, g_.value / 255.0, b_.value / 255.0); f_op = a_.value / 255.0
            R.FPDFPageObj_GetStrokeWidth(raw, wv)
            sc = _mscale(M)
            width = float(wv.value) * sc if color is not None else 0.0
            nd = R.FPDFPageObj_GetDashCount(raw)
            dashes = "[] 0"
            if nd > 0:
                arr = (ctypes.c_float * nd)()
                R.FPDFPageObj_GetDashArray(raw, arr, nd)
                ph = ctypes.c_float(); R.FPDFPageObj_GetDashPhase(raw, ph)
                dashes = "[" + " ".join(f"{v * sc:g}" for v in arr) + f"] {ph.value * sc:g}"
            layer = _oc_layer(raw)
            if layer is not None and o.level:
                # フォーム XObject の中の図形は親フォームのレイヤー（/OC）を pdfium が引き継がせる。PyMuPDF は引き継がない（レイヤー無し扱い）ので揃える
                cont = getattr(o, "container", None)
                ck = id(cont)
                if ck not in cont_layers:
                    cont_layers[ck] = _oc_layer(cont.raw) if cont is not None else None
                if layer == cont_layers[ck]:
                    layer = None
            out.append({"rect": Rect(min(xs), min(ys), max(xs), max(ys)), "items": items, "color": color, "fill": fill, "width": width,
                        "dashes": dashes, "closePath": closed and fill is None, "even_odd": fill_m.value == R.FPDF_FILLMODE_ALTERNATE, "layer": layer,
                        "fill_opacity": f_op, "stroke_opacity": s_op})
        return out

    # ---- 文字
    def _tp(self):
        if self._textpage is None:
            self._textpage = self._pg.get_textpage()
        return self._textpage

    def get_text(self, option: str = "text", clip: Rect | None = None):
        tp = self._tp()
        if option == "dict":
            return {"blocks": [{"type": 0, "lines": [{"spans": [sp]} for sp in self._spans()]}]}
        if clip is not None:
            # 表示座標の矩形 → ユーザ空間の矩形
            cs = [self._from_display(clip.x0, clip.y0), self._from_display(clip.x1, clip.y1), self._from_display(clip.x0, clip.y1), self._from_display(clip.x1, clip.y0)]
            xs = [c[0] for c in cs]; ys = [c[1] for c in cs]
            return tp.get_text_bounded(left=min(xs), bottom=min(ys), right=max(xs), top=max(ys))
        n = tp.count_chars()
        return tp.get_text_range(0, n) if n > 0 else ""

    def _spans(self) -> list[dict]:
        """文字を「同じ文字オブジェクト・同じ大きさで、横に続くもの」ごとにまとめる（PyMuPDF の span に相当）。"""
        tp = self._tp()
        n = tp.count_chars()
        spans: list[dict] = []
        cur: dict | None = None
        last_obj = None
        fm = R.FS_MATRIX()
        for i in range(n):
            code = R.FPDFText_GetUnicode(tp, i)
            ch = chr(code) if code else ""
            if ch in ("\r", "\n", "￾", "\x00", ""):
                continue                                          # pdfium の改行は無視し、位置だけでつなぐ
            R.FPDFText_GetMatrix(tp, i, fm)
            size = float(R.FPDFText_GetFontSize(tp, i)) * math.sqrt(abs(fm.a * fm.d - fm.b * fm.c)) * _mscale(self._D)
            l, b, r, t = tp.get_charbox(i)
            obj = R.FPDFText_GetTextObject(tp, i)
            key = ctypes.cast(obj, ctypes.c_void_p).value if obj else None
            if l == r == b == t == 0:                         # 空白など箱の無い文字
                if cur is not None and ch.strip() == "":
                    cur["text"] += ch
                continue
            (x0, y0), (x1, y1) = self._to_display(l, b), self._to_display(r, t)
            bb = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            same = False; gap = 0.0
            if cur is not None and abs(size - cur["size"]) < 0.05:
                eff = max(size, 2.0)                                   # 極小フォント（寸法値）は 2pt 相当で判定
                lim = 1.0 * eff if key == last_obj else 0.6 * eff
                cb = cur["bbox"]
                gx = max(bb[0], cb[0]) - min(bb[2], cb[2]); gy = max(bb[1], cb[1]) - min(bb[3], cb[3])   # 箱どうしの隙間（重なれば負）
                cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2; ccx, ccy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
                horiz = abs(cy - ccy) < 0.6 * eff and -0.5 * eff <= gx <= lim               # 横書き（左右どちら向きでも）
                vert = abs(cx - ccx) < 0.6 * eff and -0.5 * eff <= gy <= lim                 # 縦書き・回転
                same = horiz or vert; gap = gx if horiz else gy
            if same:
                if key != last_obj and gap > 0.2 * max(size, 2.0) and not cur["text"].endswith(" ") and ch != " ":
                    cur["text"] += " "                                 # 別の文字オブジェクトで隙間があれば空白で区切る（PyMuPDF と同じ）
                cur["text"] += ch
                cur["bbox"] = (min(cur["bbox"][0], bb[0]), min(cur["bbox"][1], bb[1]), max(cur["bbox"][2], bb[2]), max(cur["bbox"][3], bb[3]))
            else:
                cur = {"text": ch, "bbox": bb, "size": size}
                spans.append(cur)
            last_obj = key
        return spans

    # ---- 描画
    def get_pixmap(self, matrix: Matrix | None = None, colorspace: _Colorspace = csRGB, alpha: bool = False, clip: Rect | None = None, dpi=None) -> Pixmap:
        s = float(matrix.a) if matrix is not None else 1.0
        W, H = self.rect.width, self.rect.height
        crop = (0.0, 0.0, 0.0, 0.0)
        if clip is not None:
            c = Rect(clip) & self.rect
            crop = (c.x0, H - c.y1, W - c.x1, c.y0)
        bm = self._pg.render(scale=s, crop=crop, rev_byteorder=True, draw_annots=True, may_draw_forms=False, grayscale=(colorspace.n == 1))
        arr = bm.to_numpy()
        if arr.ndim == 3:
            arr = arr[:, :, :3] if colorspace.n == 3 else cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif colorspace.n == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return Pixmap(np.ascontiguousarray(arr))

    # ---- 画像
    def _image_objs(self) -> list:
        if self._images is None:
            self._images = []
            for o in self._pg.get_objects(max_depth=8):
                if o.type == R.FPDF_PAGEOBJ_IMAGE:
                    xref = self.parent._register_image(self, o)
                    self._images.append((xref, o))
        return self._images

    def get_image_info(self, xrefs: bool = False) -> list[dict]:
        out = []
        for xref, o in self._image_objs():
            M = _mcompose(self._obj_matrix(o), self._D)            # 単位正方形（v 上向き） → 表示座標
            T = _mcompose(_FLIP_UNIT, M)                           # (u, v 下向き) → 表示座標（PyMuPDF の transform）
            cs = [_mapply(M, 0, 0), _mapply(M, 1, 0), _mapply(M, 0, 1), _mapply(M, 1, 1)]
            xs = [c[0] for c in cs]; ys = [c[1] for c in cs]
            w = ctypes.c_uint(); h = ctypes.c_uint()
            R.FPDFImageObj_GetImagePixelSize(o.raw, w, h)
            out.append({"xref": xref, "bbox": (min(xs), min(ys), max(xs), max(ys)), "transform": T, "width": int(w.value), "height": int(h.value)})
        return out

    def get_images(self) -> list:
        return [(xref,) for xref, _ in self._image_objs()]

    # ---- リンク
    def get_links(self) -> list[dict]:
        out = []
        pos = ctypes.c_int(0)
        link = R.FPDF_LINK()
        while R.FPDFLink_Enumerate(self._pg.raw, pos, link):
            rc = R.FS_RECTF()
            d: dict = {}
            if R.FPDFLink_GetAnnotRect(link, rc):
                (x0, y0), (x1, y1) = self._to_display(rc.left, rc.bottom), self._to_display(rc.right, rc.top)
                d["from"] = Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            act = R.FPDFLink_GetAction(link)
            if act:
                n = R.FPDFAction_GetURIPath(self.parent._pdf.raw, act, None, 0)
                if n > 0:
                    buf = ctypes.create_string_buffer(n)
                    R.FPDFAction_GetURIPath(self.parent._pdf.raw, act, buf, n)
                    d["uri"] = buf.raw[:n].decode("utf-8", "ignore").rstrip("\x00")
            if d:
                out.append(d)
        return out


class Document:
    def __init__(self, src):
        self._pdf = pdfium.PdfDocument(src)
        self._pages: dict[int, Page] = {}
        self._image_registry: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self._pdf)

    @property
    def metadata(self) -> dict:
        """PyMuPDF 互換の小文字キー（producer, creator, title, ...）。"""
        try:
            md = self._pdf.get_metadata_dict()
        except Exception:  # noqa: BLE001
            md = {}
        return {k.lower(): (v or "") for k, v in md.items()}

    @property
    def page_count(self) -> int:
        return len(self._pdf)

    def __getitem__(self, i: int) -> Page:
        if i < 0:
            i += len(self)
        if i not in self._pages:
            self._pages[i] = Page(self, i)
        return self._pages[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def close(self) -> None:
        """図形・文字のキャッシュと pdfium の文書を解放する（A3 の CAD 図面で図形辞書が 1GB 近くになる）。"""
        for pg in self._pages.values():
            pg._drawings = None; pg._textpage = None; pg._images = None
        self._pages.clear(); self._image_registry.clear()
        try:
            self._pdf.close()
        except Exception:  # noqa: BLE001
            pass

    def _register_image(self, page: Page, obj) -> int:
        xref = 100000 * (page.number + 1) + len([k for k in self._image_registry if k // 100000 == page.number + 1]) + 1
        self._image_registry[xref] = (page, obj)
        return xref

    def _image_array(self, xref: int) -> np.ndarray:
        page, o = self._image_registry[xref]
        img = pdfium.PdfImage(o.raw, page=page._pg, pdf=self._pdf) if not isinstance(o, pdfium.PdfImage) else o
        bm = img.get_bitmap(render=False)
        pil = bm.to_pil().convert("RGB")
        return np.array(pil)


def open(path=None, stream: bytes | None = None, filetype: str | None = None) -> Document:  # noqa: A001
    if stream is not None:
        return Document(bytes(stream))
    return Document(str(path))


# ---------------------------------------------------------------- 図形の描画（PyMuPDF の Shape の代わり）
def _flatten_items(items, steps: int = 12) -> list[list[tuple[float, float]]]:
    """items を折れ線（サブパス）の列にする。連続する要素は 1 本につなぐ。"""
    paths: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for it in items:
        if it[0] == "l":
            p, q = (it[1].x, it[1].y), (it[2].x, it[2].y)
            if cur and abs(cur[-1][0] - p[0]) + abs(cur[-1][1] - p[1]) < 1e-6:
                cur.append(q)
            else:
                if cur:
                    paths.append(cur)
                cur = [p, q]
        elif it[0] == "c":
            p0, p1, p2, p3 = [(v.x, v.y) for v in it[1:5]]
            pts = []
            for k in range(1, steps + 1):
                t = k / steps; u = 1 - t
                pts.append((u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0],
                            u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]))
            if cur and abs(cur[-1][0] - p0[0]) + abs(cur[-1][1] - p0[1]) < 1e-6:
                cur += pts
            else:
                if cur:
                    paths.append(cur)
                cur = [p0] + pts
        elif it[0] == "re":
            r = it[1]
            if cur:
                paths.append(cur); cur = []
            paths.append([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1), (r.x0, r.y0)])
        elif it[0] == "qu":
            q = it[1]
            if cur:
                paths.append(cur); cur = []
            paths.append([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)])
    if cur:
        paths.append(cur)
    return paths


STROKE_MODE = "plain"            # "plain": 線幅どおり（1/8px 精度のアンチエイリアス）/ "hairline+1": 細線を 2px に / "all+1": 常に 1px 足す。回帰で plain が最も近かった


def _stroke_px(lw: float, scale: float) -> int:
    w = lw * scale
    if STROKE_MODE == "all+1":
        return max(1, int(round(w))) + 1
    if STROKE_MODE == "hairline+1":
        return 2 if w < 1.5 else max(1, int(round(w)))
    return max(1, int(round(w)))


def render_drawings(width: float, height: float, items: list, scale: float) -> np.ndarray:
    """(Drawing, 上書き属性 {color, fill, width}) の列を白地に描き、RGB 配列 (H, W, 3) で返す。"""
    W, H = int(math.ceil(width * scale)), int(math.ceil(height * scale))
    img = np.full((H, W, 3), 255, np.uint8)
    for d, ov in items:
        color = ov.get("color", d.color) if ov else d.color
        fill = ov.get("fill", d.fill) if ov else d.fill
        lw = ov.get("width", d.width) if ov else d.width
        paths = _flatten_items(d.items)
        if not paths:
            continue
        # 座標は 1/8 px 精度（shift=3）でアンチエイリアス描画。太さの扱いは STROKE_MODE（既定 plain）
        polys = [np.round(np.array(p, np.float64) * scale * 8).astype(np.int32).reshape(-1, 1, 2) for p in paths]
        if fill is not None:
            fc = tuple(int(round(v * 255)) for v in fill)
            cv2.fillPoly(img, polys, fc, lineType=cv2.LINE_8, shift=3)
        if color is not None:
            sc = tuple(int(round(v * 255)) for v in color)
            th = _stroke_px(lw or 0.0, scale)
            cv2.polylines(img, polys, bool(d.close) or fill is not None, sc, th, lineType=cv2.LINE_AA, shift=3)
    return img


def image_to_pdf_bytes(img: np.ndarray, dpi: int, title: str = "") -> bytes:
    """RGB 画像を 1 ページの PDF（Flate 圧縮、無劣化）にする。用紙寸法は dpi から逆算。"""
    import zlib
    h, w = img.shape[:2]
    rgb = np.ascontiguousarray(img[:, :, :3])
    data = zlib.compress(rgb.tobytes(), 6)
    pw, ph = w * 72.0 / dpi, h * 72.0 / dpi
    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.3f} {ph:.3f}] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>".encode())
    objs.append(f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(data)} >>".encode() + b"\nstream\n" + data + b"\nendstream")
    content = f"q {pw:.3f} 0 0 {ph:.3f} 0 0 cm /Im0 Do Q".encode()
    objs.append(f"<< /Length {len(content)} >>".encode() + b"\nstream\n" + content + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)

