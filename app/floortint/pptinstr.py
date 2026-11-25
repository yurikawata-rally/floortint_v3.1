"""指示書（PPT → PDF、または図面 PDF に iPad で手書き）の読み取り v2。

PPT 指示書の構造（2026-10 受領の 5 案件で確認）:
  - 図面はスライドに貼った画像（CAD の PDF をレンダリングした PNG/JPEG。縮小・トリミングあり。1 枚に複数階のことも）
  - 範囲は半透明の塗り図形（矩形・フリーフォーム。alpha 0.3〜0.7）。同じ材種は同じ色
  - 凡例は図面の外に「色見本（小さな塗り）＋品番／メーカー／貼り方の文字＋製品写真（小さな画像）＋製品ページへのリンク」
  - 貼り方向は図形の上に矢印（線＋三角）
手書き指示書（iPad）の構造:
  - 図面 PDF そのものに、色の線（斜線ハッチ・囲み・引出線）と文字・写真が描き足されている。レイヤー無し、線幅 1〜7pt
位置合わせ:
  - PPT: 画像の実データ（xref）を取り出し、平面図ページのレンダリングに多倍率テンプレート相関と特徴点（ORB）で重ねる。
    両者が一致すれば高信頼。画像の配置（transform）で図形座標 → 画像座標 → 平面図座標を結ぶ
  - 手書き: 図面が同じ PDF なので恒等（黒い線の一致率で確認）
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from floortint import pdfio

from floortint.pdfdoc import Drawing, render_drawings

MAKERS = {  # 表記ゆれ → 正式名
    "川島織物セルコン": "川島織物セルコン", "川島セルコン": "川島織物セルコン", "川島織物": "川島織物セルコン", "カワシマ": "川島織物セルコン", "Kawashima": "川島織物セルコン",
    "東リ": "東リ", "TOLI": "東リ", "Toli": "東リ", "サンゲツ": "サンゲツ", "Sangetsu": "サンゲツ", "sangetsu": "サンゲツ", "SANGETSU": "サンゲツ",
    "田島ルーフィング": "田島ルーフィング", "田島": "田島ルーフィング", "タジマ": "田島ルーフィング", "リリカラ": "リリカラ", "シンコール": "シンコール",
    "スミノエ": "住江織物", "住江織物": "住江織物", "住江": "住江織物", "ミリケン": "ミリケン", "Milliken": "ミリケン", "インターフェイス": "インターフェイス",
    "住商インテリア": "住商インテリア", "アドヴァン": "アドヴァン", "ニッタイ": "ニッタイ工業", "名古屋モザイク": "名古屋モザイク", "LIXIL": "LIXIL",
}
_MAKER_KEYS = sorted(MAKERS, key=len, reverse=True)
CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{1,5}[-‐]?\d{2,6}[A-Z]{0,3}(?:[-‐]\d{1,5}[A-Z]{0,4})?(?:[-‐][A-Z]{1,4})?|\d{4}[-‐]\d{3,5})(?![A-Za-z0-9])")
# 品番の頭文字から材種を推定（文字に種類の語が無いとき）
PREFIX_KIND = [(r"^(RE|OT|TTN|LF|GX|PWT|PST|PM|TS|WD|FS|IS)", "vinyl"), (r"^(NTH|NT|DT|GA|GX|CB|AB|AA|TZ|GT|LP|PX|TGA|\d{4}-)", "carpet")]
PREFIX_MAKER = [(r"^(RE|AB|CB|AA|TZ|GA)\d", None), (r"^(RE|AB|CB)", "川島織物セルコン"), (r"^(NTH|NT|DT|OT|IS|PM|TS|WD)", "サンゲツ"), (r"^(GA|GX|TTN|PST|PWT|FS)", "東リ"), (r"^(TZ|LF|\d{4}-)", "田島ルーフィング"), (r"^(AA|LP)", "リリカラ")]
PREFIX_MAKER = [(pat, mk) for pat, mk in PREFIX_MAKER if mk]
PATTERNS = [("ヘリンボーン", r"ヘリンボーン"), ("アシュラー", r"アシュラー"), ("ブリック", r"ブリック"), ("市松", r"市松"), ("乱", r"乱貼|乱張|ランダム|乱"), ("流し", r"流し")]
KINDS = [("vinyl", r"塩ビ|塩化ビニ|フロアタイル|ビニル|LVT|木目|石目|CPT"), ("tile", r"タイル貼|セラミック|磁器|タイル(?!カーペット)"), ("carpet", r"タイルカーペット|カーペット|TC\b")]


@dataclass
class LegendItem:
    color: tuple                      # RGB 0〜1（PDF の塗り色）
    text: str = ""
    product: str = ""
    maker: str = ""
    kind: str = "carpet"
    pattern: str = "流し"
    size_mm: tuple | None = None
    photo: bytes | None = None        # 製品写真（PNG/JPEG）
    url: str = ""
    bbox: tuple = (0, 0, 0, 0)
    note: str = ""
    existing: bool = False            # 「ビル標準」「既存」など、そのまま残す床

    def to_dict(self) -> dict:
        return {"color": "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in self.color), "product": self.product, "maker": self.maker,
                "kind": self.kind, "pattern": self.pattern, "size_mm": self.size_mm, "url": self.url, "photo": self.photo is not None, "existing": self.existing, "text": self.text[:60]}


@dataclass
class PlanImage:
    xref: int
    bbox: tuple                       # ページ上の配置（pt）。ページ外にはみ出すことがある
    transform: tuple                  # 単位正方形 → ページ pt（a,b,c,d,e,f）
    native: np.ndarray | None = None  # (H,W,3)
    M_page_to_plan: np.ndarray | None = None   # ページ pt → 平面図 S px（2x3）
    scale: float = 0.0                # 画像 px → 平面図 r px の倍率
    corr: float = 0.0
    inliers: int = 0
    method: str = ""


def pptx_to_pdf(data: bytes, timeout: int = 180) -> bytes:
    """PPTX を LibreOffice（soffice --headless）で PDF にする。図形・画像・リンクが PDF に残る。"""
    import shutil
    import subprocess
    import tempfile
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if exe is None:
        raise RuntimeError("LibreOffice（soffice）が無いので PPTX を読めません。PDF に書き出してから入れてください")
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.pptx")
        with open(src, "wb") as f:
            f.write(data)
        env = dict(os.environ, HOME=td)
        subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", td, src], check=True, timeout=timeout,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
        with open(os.path.join(td, "in.pdf"), "rb") as f:
            return f.read()


def parse_legend_text(text: str) -> dict:
    """「タイルカーペット貼 GA-144 / 東リ ※市松貼」のような文字から品番・メーカー・種類・貼り方を取る。"""
    t = text.replace("／", "/").replace("　", " ")
    out = {"product": "", "maker": "", "kind": "carpet", "pattern": "流し", "size_mm": None}
    codes = [c for c in CODE_RE.findall(t) if not re.fullmatch(r"(CH|FL|OA|EV|PS|EPS|DS|WC|H|W|D|P|A|B|C)\d+", c)]
    if codes:
        out["product"] = codes[0]
    t_ns = re.sub(r"\s+", "", t)
    for mk in _MAKER_KEYS:
        if mk in t_ns:
            out["maker"] = MAKERS[mk]
            break
    if not out["maker"] and codes:
        for pat, mk in PREFIX_MAKER:
            if re.match(pat, codes[0]):
                out["maker"] = mk
                break
    kind = None
    for k, pat in KINDS:
        if re.search(pat, t):
            kind = k
            break
    if kind is None and codes:
        for pat, k in PREFIX_KIND:
            if re.match(pat, codes[0]):
                kind = k
                break
    out["kind"] = kind or "carpet"
    out["existing"] = bool(re.search(r"ビル標準|既存|現状|変更なし", t))
    for k, pat in PATTERNS:
        if re.search(pat, t):
            out["pattern"] = k
            break
    sm = re.search(r"(\d{3,4})\s*[x×X]\s*(\d{3,4})", t)
    if sm:
        out["size_mm"] = (float(sm.group(1)), float(sm.group(2)))
    elif re.search(r"□\s*500|500角", t):
        out["size_mm"] = (500.0, 500.0)
    return out


def _saturated(c) -> bool:
    return c is not None and (max(c) - min(c)) >= 0.25


def _page_images(doc: pdfio.Document, page: pdfio.Page) -> list[dict]:
    out = []
    for info in page.get_image_info(xrefs=True):
        b = info.get("bbox"); tr = info.get("transform")
        if not b or not tr or not info.get("xref"):
            continue
        out.append({"xref": int(info["xref"]), "bbox": tuple(b), "transform": tuple(tr), "w": int(info.get("width", 0)), "h": int(info.get("height", 0))})
    return out


def _extract_image(doc: pdfio.Document, xref: int) -> np.ndarray | None:
    try:
        pix = pdfio.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4:
            pix = pdfio.Pixmap(pdfio.csRGB, pix)
        if pix.alpha:
            pix = pdfio.Pixmap(pix, 0)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        return arr[:, :, :3].copy()
    except Exception:  # noqa: BLE001
        return None


def _ink(rgb: np.ndarray, thr: int = 170) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) < thr


def align_image_to_plan(img: np.ndarray, plan_rgb: np.ndarray, scale_range=(0.4, 2.5), steps: int = 30) -> tuple[np.ndarray | None, float, float, int, str]:
    """画像（図面のレンダリング）を平面図のレンダリングに重ねる。戻り値 (2x3 行列 画像px→平面図px, 倍率, 相関, 内点数, 方法)。
    テンプレート相関（多倍率）と ORB の両方を試し、相関が高い方を採る。ORB は倍率の検証にも使う。"""
    A = _ink(img).astype(np.float32); B = _ink(plan_rgb).astype(np.float32)
    work = 0.5
    Bw = cv2.GaussianBlur(cv2.resize(B, None, fx=work, fy=work, interpolation=cv2.INTER_AREA), (0, 0), 2.0)
    base = plan_rgb.shape[1] / max(1, img.shape[1])
    best = (None, 0.0, -1.0)
    for sc in base * np.exp(np.linspace(math.log(scale_range[0]), math.log(scale_range[1]), steps)):
        f = sc * work
        if A.shape[0] * f < 30 or A.shape[1] * f < 30:
            continue
        t = cv2.GaussianBlur(cv2.resize(A, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), (0, 0), 2.0)
        if t.shape[0] >= Bw.shape[0] or t.shape[1] >= Bw.shape[1]:
            # テンプレートの方が大きい（画像にページ外まで入っている）: 画像側を縮めて逆に探す
            continue
        rm = cv2.matchTemplate(Bw, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(rm)
        if mx > best[2]:
            best = (np.float32([[sc, 0, loc[0] / work], [0, sc, loc[1] / work]]), float(sc), float(mx))
    M_t, sc_t, corr_t = best
    # ORB（倍率を含む相似変換）
    M_o, n_in, sc_o = None, 0, 0.0
    try:
        orb = cv2.ORB_create(nfeatures=8000)
        ga, gb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.cvtColor(plan_rgb, cv2.COLOR_RGB2GRAY)
        ka, da = orb.detectAndCompute(ga, None); kb, db = orb.detectAndCompute(gb, None)
        if da is not None and db is not None and len(ka) >= 50 and len(kb) >= 50:
            knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
            good = [m for m, n in (x for x in knn if len(x) == 2) if m.distance < 0.75 * n.distance]
            if len(good) >= 20:
                src = np.float32([ka[m.queryIdx].pt for m in good]); dst = np.float32([kb[m.trainIdx].pt for m in good])
                M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=5000, confidence=0.995)
                if M is not None and inl is not None:
                    n_in = int(inl.sum()); sc_o = float(math.hypot(M[0, 0], M[0, 1]))
                    rot = abs(math.degrees(math.atan2(M[0, 1], M[0, 0])))
                    if n_in >= 30 and rot < 2.0:
                        M_o = np.float32([[sc_o, 0, M[0, 2]], [0, sc_o, M[1, 2]]])
    except Exception:  # noqa: BLE001
        pass
    if M_o is not None and (M_t is None or corr_t < 0.5 or abs(sc_o - sc_t) / max(sc_t, 1e-6) < 0.05):
        # ORB が取れて倍率が相関と一致（または相関が弱い）→ ORB を採る。相関はテンプレートの値をそのまま報告
        return M_o, sc_o, max(corr_t, 0.0), n_in, "orb+template" if M_t is not None and abs(sc_o - sc_t) / max(sc_t, 1e-6) < 0.05 else "orb"
    if M_t is not None:
        return M_t, sc_t, corr_t, n_in, "template"
    return None, 0.0, 0.0, 0, "none"


def _page_to_image_px(pi: PlanImage, W: int, H: int) -> np.ndarray:
    """ページ pt → 画像 px の 2x3 行列（画像の配置 transform から）。"""
    a, b, c, d, e, f = pi.transform
    # x_pt = a*u + c*v + e, y_pt = b*u + d*v + f（u,v は 0〜1、v は上が 0）。回転なしを前提に逆算
    sx = W / a if abs(a) > 1e-6 else 0.0
    sy = H / d if abs(d) > 1e-6 else 0.0
    return np.float32([[sx, 0, -e * sx], [0, sy, -f * sy]])


def _compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """B の後に A を適用する 2x3 行列（A∘B）。"""
    A3 = np.vstack([A, [0, 0, 1]]); B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2].astype(np.float32)


def find_plan_images(doc: pdfio.Document, page: pdfio.Page, plan_page: pdfio.Page, r: float = 2.0, S: float = 3.0,
                     min_native_w: int = 600) -> list[PlanImage]:
    """ページ内の大きな画像を全部取り出し、平面図に位置合わせして相関の高い順に返す。"""
    pix = plan_page.get_pixmap(matrix=pdfio.Matrix(r, r), alpha=False)
    plan_rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    out = []
    page_area = page.rect.width * page.rect.height
    for info in _page_images(doc, page):
        if info["w"] < min_native_w:
            continue
        b = info["bbox"]
        vis = pdfio.Rect(b) & page.rect
        if vis.is_empty or vis.width * vis.height < 0.04 * page_area:
            continue
        img = _extract_image(doc, info["xref"])
        if img is None or img.shape[1] < min_native_w:
            continue
        # 図面らしさ: 細い線が多い（写真は面）
        ink = _ink(img).astype(np.uint8)
        thin = ink & ~cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        if thin.mean() < 0.01 or ink.mean() > 0.5:
            continue
        pi = PlanImage(info["xref"], b, info["transform"], img)
        M, sc, corr, n_in, method = align_image_to_plan(img, plan_rgb)
        pi.scale, pi.corr, pi.inliers, pi.method = sc, corr, n_in, method
        if M is not None:
            Mi = _page_to_image_px(pi, img.shape[1], img.shape[0])          # ページ pt → 画像 px
            M_plan_r = M                                                   # 画像 px → 平面図 r px
            to_S = np.float32([[S / r, 0, 0], [0, S / r, 0]])              # 平面図 r px → S px
            pi.M_page_to_plan = _compose(to_S, _compose(M_plan_r, Mi))
        out.append(pi)
    out.sort(key=lambda z: -(z.corr + 0.001 * z.inliers))
    return out


def read_legend(doc: pdfio.Document, page: pdfio.Page, exclude: list[pdfio.Rect], drs: list[Drawing]) -> list[LegendItem]:
    """凡例: 図面画像の外にある小さな塗り（色見本）ごとに、文字・近くの小さな画像・リンクを集める。
    文字の対応づけは 2 通りの配置を扱う。
      右配置: 見本の右に 1〜4 行（見本が縦に並ぶ）。各見本の「先頭行」= 見本の上端に最も近い行、ブロック = 次の見本の先頭行の手前まで
      下配置: 見本の下に 1〜3 行（見本が格子に並ぶ）。見本と左端が揃う行を、次の見本（同じ列）の手前まで"""
    spans = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for sp in l["spans"]:
                t = sp["text"].strip()
                if t:
                    spans.append((pdfio.Rect(sp["bbox"]), t))
    links = [(pdfio.Rect(l["from"]), l.get("uri", "")) for l in page.get_links() if l.get("uri")]
    imgs = [(pdfio.Rect(i["bbox"]), i["xref"], i["w"]) for i in _page_images(doc, page)]
    def outside(r: pdfio.Rect) -> bool:
        """図面画像の外（半分以上はみ出していれば外とみなす: 凡例が画像の縁にかかることがある）"""
        a = max(1e-6, r.get_area())
        return not any((r & e).get_area() > 0.5 * a for e in exclude)
    spans = [(r, t) for r, t in spans if outside(r)]
    swatches: list[tuple[pdfio.Rect, tuple]] = []
    for d in drs:
        r = d.rect
        if r.width > 60 or r.height > 40 or r.width < 8 or r.height < 5 or not outside(r):
            continue
        if d.fill is not None and (_saturated(d.fill) or 0.25 <= min(d.fill) <= max(d.fill) <= 0.9):
            swatches.append((r, tuple(round(c, 3) for c in d.fill)))
        elif d.fill is None and d.color is not None and not _saturated(d.color) and d.width <= 2 and d.items and d.items[0][0] == "re":
            swatches.append((r, (1.0, 1.0, 1.0)))              # 白枠だけの見本（ビル標準など）
    merged: list[tuple[pdfio.Rect, tuple]] = []
    for r, c in sorted(swatches, key=lambda z: (z[0].y0, z[0].x0)):
        if any(c2 == c and abs(r2.y0 - r.y0) < 8 and abs(r2.x0 - r.x0) < 80 for r2, c2 in merged):
            continue
        merged.append((r, c))
    if not merged:
        return []
    # 行にまとめる: y 中心が近いスパンを同じ行とし、行内は x 順に並べて隙間 12pt 以内をつなぐ
    rows: list[list[tuple[pdfio.Rect, str]]] = []
    for r, t in sorted(spans, key=lambda z: (z[0].y0 + z[0].y1) / 2):
        cy = (r.y0 + r.y1) / 2
        if rows and abs((rows[-1][0][0].y0 + rows[-1][0][0].y1) / 2 - cy) < 3.5:
            rows[-1].append((r, t))
        else:
            rows.append([(r, t)])
    col_starts = [sw.x0 for sw, _ in merged]                     # 見本の列の左端では行を切る（格子配置で隣の列の文字をつながない）
    lines: list[tuple[pdfio.Rect, str]] = []
    for row in rows:
        cur = None
        for r, t in sorted(row, key=lambda z: z[0].x0):
            if cur is not None and -2 <= r.x0 - cur[0].x1 < 12 and not any(abs(r.x0 - cx) <= 6 for cx in col_starts):
                cur = (cur[0] | r, cur[1] + " " + t)
            else:
                if cur is not None:
                    lines.append(cur)
                cur = (pdfio.Rect(r), t)
        if cur is not None:
            lines.append(cur)
    # 各見本の先頭行（右配置・下配置）
    def anchor(sw: pdfio.Rect):
        right = [(abs(r.y0 - sw.y0) + 0.02 * abs(r.x0 - sw.x1), r, t) for r, t in lines if max(sw.x1 - 8, sw.x0 + 0.5 * sw.width) <= r.x0 <= sw.x1 + 45 and abs(r.y0 - sw.y0) <= sw.height + 6]
        below = [((r.y0 - sw.y1) + 0.5 * abs(r.x0 - sw.x0), r, t) for r, t in lines if abs(r.x0 - sw.x0) <= 25 and -2 <= r.y0 - sw.y1 <= 30]
        if right:
            return "R", min(right)[1]
        if below:
            return "B", min(below)[1]
        return None, None
    anchors = [(sw, c, *anchor(sw)) for sw, c in merged]
    items: list[LegendItem] = []
    for sw, c, side, a0 in anchors:
        block: list[tuple[pdfio.Rect, str]] = []
        if a0 is not None:
            if side == "R":
                col_x = a0.x0
                # 同じ列の行を y 順に並べ、先頭行から次の見本の先頭行の手前まで
                col = sorted([(r, t) for r, t in lines if abs(r.x0 - col_x) <= 40 and r.y0 >= a0.y0 - 1], key=lambda z: z[0].y0)
                nxt = [a2.y0 for sw2, c2, side2, a2 in anchors if a2 is not None and side2 == "R" and abs(a2.x0 - col_x) <= 40 and a2.y0 > a0.y0 + 1]
                y_end = min(nxt) - 1 if nxt else a0.y0 + 90
                block = [(r, t) for r, t in col if r.y0 < y_end and r.y0 - a0.y0 < 90]
            else:
                col = sorted([(r, t) for r, t in lines if abs(r.x0 - sw.x0) <= 25 and r.y0 >= sw.y1 - 2], key=lambda z: z[0].y0)
                nxt = [sw2.y0 for sw2, c2, s2, a2 in anchors if abs(sw2.x0 - sw.x0) <= 25 and sw2.y0 > sw.y1]
                y_end = min(nxt) - 1 if nxt else sw.y1 + 60
                block = [(r, t) for r, t in col if r.y0 < y_end and r.y0 - sw.y1 < 60]
        text = " ".join(t for _, t in block)
        if not text.strip():
            continue                                             # 文字の無い小さな塗り（矢印の頭など）は見本ではない
        li = LegendItem(c, text=text, bbox=tuple(sw))
        li.__dict__.update(parse_legend_text(text))
        for lr, uri in links:
            if any(lr.intersects(r) for r, _ in block):
                li.url = uri
                break
        # 写真: 見本または文字ブロックの近く（右 220pt・下 130pt）の小さな画像
        ref = pdfio.Rect(sw)
        for r, _ in block:
            ref |= r
        area = pdfio.Rect(ref.x0 - 10, ref.y0 - 10, ref.x1 + 220, ref.y1 + 130)
        best = None
        for ir, xref, iw in imgs:
            if ir.intersects(area) and outside(ir) and ir.width < 220 and ir.height < 220:
                d = max(0, ir.x0 - ref.x1) + max(0, ir.y0 - ref.y1)
                if best is None or d < best[0]:
                    best = (d, xref)
        if best is not None:
            try:
                pix = pdfio.Pixmap(doc, best[1])
                if pix.n - pix.alpha >= 4:
                    pix = pdfio.Pixmap(pdfio.csRGB, pix)
                li.photo = pix.tobytes("png")
            except Exception:  # noqa: BLE001
                pass
        items.append(li)
    return items


def extract_arrows(drs: list[Drawing], inside: pdfio.Rect) -> list[tuple[float, float, float]]:
    """図面画像の上の矢印（線＋小さな三角の塗り）→ (中点 x, 中点 y, 角度 deg)。両矢印は 1 本として扱う。"""
    tris = [d for d in drs if d.fill is not None and d.rect.width < 14 and d.rect.height < 14 and inside.contains(d.rect)
            and sum(1 for it in d.items if it[0] == "l") in (2, 3)]
    lines = [d for d in drs if d.fill is None and d.color is not None and inside.contains(d.rect) and len(d.items) == 1 and d.items[0][0] == "l"]
    out = []
    for d in lines:
        x0, y0, x1, y1 = d.segments()[0]
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 12 or L > 500:
            continue
        near = any(min(math.hypot(t.rect.x0 + t.rect.width / 2 - x, t.rect.y0 + t.rect.height / 2 - y) for x, y in ((x0, y0), (x1, y1))) < 8 for t in tris)
        if near:
            out.append(((x0 + x1) / 2, (y0 + y1) / 2, math.degrees(math.atan2(-(y1 - y0), x1 - x0)) % 180))
    return out


@dataclass
class PptInstruction:
    plan_images: list = field(default_factory=list)
    legend: list = field(default_factory=list)
    seeds_masks: dict = field(default_factory=dict)      # 色（RGB 0〜1） → mask（平面図 S px）
    seed_alpha: dict = field(default_factory=dict)
    arrows: list = field(default_factory=list)           # 平面図 S px 座標の (x, y, deg)
    warnings: list = field(default_factory=list)
    chosen: int = -1
    is_annotated_plan: bool = False


def read_ppt_instruction(doc: pdfio.Document, page: pdfio.Page, plan_page: pdfio.Page, S: float, out_hw: tuple[int, int],
                         min_corr: float = 0.3) -> PptInstruction:
    """PPT 由来の指示書ページを読む。図面画像を平面図に合わせ、塗り図形を平面図座標の種マスクにし、凡例・矢印も返す。"""
    res = PptInstruction()
    drs = [Drawing.from_raw(g) for g in page.get_drawings()]
    pis = find_plan_images(doc, page, plan_page, S=S)
    res.plan_images = pis
    if not pis or pis[0].M_page_to_plan is None or pis[0].corr < min_corr and pis[0].inliers < 60:
        res.warnings.append("図面画像と平面図の位置合わせが取れない（相関 %.2f）" % (pis[0].corr if pis else 0.0))
        if not pis or pis[0].M_page_to_plan is None:
            return res
    pi = pis[0]; res.chosen = 0
    img_rect = pdfio.Rect(pi.bbox) & page.rect
    res.legend = read_legend(doc, page, [pdfio.Rect(p.bbox) for p in pis], drs)
    legend_colors = [np.float32(li.color) for li in res.legend]
    def in_legend(c) -> bool:
        return any(float(np.abs(np.float32(c) - lc).max()) < 0.03 for lc in legend_colors)
    # 範囲の図形: 塗りで、図面画像の上にあるもの（見本は画像の外）。彩度のある色、半透明の白、凡例にある灰色
    marks = [d for d in drs if d.fill is not None and (_saturated(d.fill) or (d.fill_opacity < 0.95 and min(d.fill) > 0.85) or in_legend(d.fill))
             and d.rect.intersects(img_rect) and (d.rect.width > 4 or d.rect.height > 4) and not (d.rect.width < 14 and d.rect.height < 14)]
    PH, PW = out_hw
    W, H = page.rect.width, page.rect.height
    by_color: dict[tuple, list[Drawing]] = {}
    for d in marks:
        by_color.setdefault(tuple(round(c, 3) for c in d.fill), []).append(d)
    # ページ pt → 平面図 S px。render_drawings は S_page px/pt で描くので、その分を戻す
    S_page = 2.0
    M_render_to_plan = _compose(pi.M_page_to_plan, np.float32([[1 / S_page, 0, 0], [0, 1 / S_page, 0]]))
    for c, group in by_color.items():
        img = render_drawings(W, H, [(d, {"color": (0, 0, 0), "fill": (0, 0, 0)}) for d in group], S_page)
        m = (img.min(axis=2) < 128).astype(np.uint8) * 255
        m = cv2.warpAffine(m, M_render_to_plan, (PW, PH), flags=cv2.INTER_LINEAR)
        m = (m > 127).astype(np.uint8) * 255
        if m.any():
            res.seeds_masks[c] = m
            res.seed_alpha[c] = float(np.median([d.fill_opacity for d in group]))
    for x, y, deg in extract_arrows(drs, img_rect):
        px, py = pi.M_page_to_plan @ np.float32([x, y, 1.0])
        res.arrows.append((float(px), float(py), deg))
    return res


# ---------------------------------------------------------------- 手書き（図面 PDF に描き足し）
def is_annotated_plan(instr_page: pdfio.Page, plan_page: pdfio.Page, r: float = 1.0) -> tuple[bool, float]:
    """指示書ページが平面図ページそのもの（に描き足したもの）かを、黒い線の一致率で判定する。
    貼り付けた写真（画像）の領域は除外し、色の付いた線（手書き）も除外して比べる。"""
    if abs(instr_page.rect.width - plan_page.rect.width) > 2 or abs(instr_page.rect.height - plan_page.rect.height) > 2:
        return False, 0.0
    def black(pg):
        pix = pg.get_pixmap(matrix=pdfio.Matrix(r, r), colorspace=pdfio.csRGB, alpha=False)
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, 3).astype(np.int16)
        g = (rgb.max(2) < 200) & ((rgb.max(2) - rgb.min(2)) < 40)
        return cv2.dilate(g.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    a, b = black(instr_page), black(plan_page)
    mask = np.ones_like(a)
    for pg in (instr_page, plan_page):
        for info in pg.get_image_info():
            x0, y0, x1, y1 = [int(v * r) for v in info["bbox"]]
            mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    a &= mask; b &= mask
    u = (a | b).sum()
    if u == 0:
        return False, 0.0
    iou = float((a & b).sum() / u)
    return iou >= 0.6, iou


@dataclass
class Callout:
    """手書きの吹き出し（文字の塊や貼った写真）。引出線の先（target）で領域・部屋に紐づける。"""
    kind: str                          # "text" | "photo"
    bbox: tuple                        # ページ pt
    png: bytes | None = None           # 切り出し画像
    target: tuple | None = None        # 平面図 S px (x, y)。引出線が無ければ None
    color: tuple | None = None         # 引出線の色（RGB 0〜1）
    room_id: str | None = None         # パイプラインで埋める
    text: str = ""                     # PDF に文字があれば

    def to_dict(self) -> dict:
        return {"kind": self.kind, "bbox": [round(v) for v in self.bbox], "target": [round(v) for v in self.target] if self.target else None,
                "room": self.room_id, "text": self.text}


def _stroke_geom(d: Drawing) -> tuple[float, float, tuple, tuple]:
    """線の長さ・端点間距離・始点・終点。"""
    segs = d.segments()
    if not segs:
        return 0.0, 0.0, (d.rect.x0, d.rect.y0), (d.rect.x1, d.rect.y1)
    L = sum(math.hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in segs)
    s0, e0 = (segs[0][0], segs[0][1]), (segs[-1][2], segs[-1][3])
    return L, math.hypot(e0[0] - s0[0], e0[1] - s0[1]), s0, e0


def _cluster_rects(rects: list[pdfio.Rect], gap: float) -> list[list[int]]:
    """矩形を近さ（gap 以内）でつなぐ。"""
    n = len(rects); parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        ri = pdfio.Rect(rects[i]); ri.x0 -= gap; ri.y0 -= gap; ri.x1 += gap; ri.y1 += gap
        for j in range(i + 1, n):
            if ri.intersects(rects[j]) or ri.contains(rects[j]):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def classify_handwriting(drs: list[Drawing], zone: pdfio.Rect, images: list[tuple[pdfio.Rect, int]]) -> dict:
    """手書きの色の線を 区画（ハッチ・囲み）／引出線／文字 に分ける。
    文字: 短く小さい線が集まったもの。引出線: 長い開いた線で、端が文字の塊・写真・図面の外のどれかに近いもの。残りが区画。"""
    marks = [d for d in drs if _saturated(d.color if d.color is not None else d.fill)]
    geom = {id(d): _stroke_geom(d) for d in marks}
    small = [d for d in marks if geom[id(d)][0] < 45 and max(d.rect.width, d.rect.height) < 25 and d.width >= 0.8]   # 細い線（CAD 記号）は文字でない
    text_groups: list[pdfio.Rect] = []
    if small:
        rects = [d.rect for d in small]
        for g in _cluster_rects(rects, 14.0):
            if len(g) >= 3:
                r = pdfio.Rect(rects[g[0]])
                for i in g[1:]:
                    r |= rects[i]
                if r.width >= 20 or r.height >= 20:
                    text_groups.append(r)
    text_ids = set()
    for d in small:
        if any(tg.intersects(d.rect) for tg in text_groups):
            text_ids.add(id(d))
    anchors = [(r, "text") for r in text_groups] + [(r, "photo") for r, _ in images]
    def near_anchor(pt, tol=40.0):
        best = None
        for r, kind in anchors:
            dx = max(r.x0 - pt[0], 0, pt[0] - r.x1); dy = max(r.y0 - pt[1], 0, pt[1] - r.y1)
            dd = math.hypot(dx, dy)
            if dd <= tol and (best is None or dd < best[0]):
                best = (dd, r, kind)
        return best
    leaders, regions, texts = [], [], []
    zone_out = pdfio.Rect(zone.x0 - 25, zone.y0 - 25, zone.x1 + 25, zone.y1 + 25)   # 図面の縁の少し外までは区画の線とみなす
    for d in marks:
        if id(d) in text_ids:
            texts.append(d); continue
        L, E, s0, e0 = geom[id(d)]
        if L < 7 or max(d.rect.width, d.rect.height) < 7:
            continue
        if d.fill is not None and not d.color and d.rect.width < 20 and d.rect.height < 20:
            continue                                        # 小さな塗り（CAD 記号）
        is_open = E > 0.3 * L or L < 20
        if is_open and L >= 50:
            a_s, a_e = near_anchor(s0), near_anchor(e0)
            out_s, out_e = not zone_out.contains(pdfio.Point(*s0)), not zone_out.contains(pdfio.Point(*e0))
            if (a_s is not None) != (a_e is not None):
                src, tgt = (a_s, e0) if a_s is not None else (a_e, s0)
                leaders.append((d, src[1], src[2], tgt)); continue
            if out_s != out_e:
                leaders.append((d, None, "", e0 if out_s else s0)); continue
            if out_s and out_e:
                continue
        regions.append(d)
    return {"regions": regions, "leaders": leaders, "texts": texts, "text_groups": text_groups}


def handwritten_seeds(instr_page: pdfio.Page, S: float, out_hw: tuple[int, int], zone: pdfio.Rect, mm_per_pt: float,
                      merge_de: float = 30.0) -> tuple[dict[tuple, np.ndarray], list[str], dict, list[Callout]]:
    """手書きの色の線（斜線ハッチ・囲み）を色ごとにまとめ、ハッチは間隔で閉じて面にする。引出線・文字は落とし、吹き出しとして返す。
    戻り値: 色 → mask（平面図 S px）, 警告, 診断, 吹き出し。"""
    from floortint.instruct import _cluster_colors
    drs = [Drawing.from_raw(g) for g in instr_page.get_drawings()]
    images = []
    for info in instr_page.get_image_info(xrefs=True):
        r = pdfio.Rect(info["bbox"])
        if info.get("xref") and (not zone.contains(r)) and r.width > 30 and r.height > 30:
            images.append((r, int(info["xref"])))
    cls = classify_handwriting(drs, zone, images)
    marks = [d for d in cls["regions"] if d.rect.intersects(zone)]
    diag: dict = {"colored_drawings": len(cls["regions"]) + len(cls["leaders"]) + len(cls["texts"]),
                  "leaders": len(cls["leaders"]), "text_strokes": len(cls["texts"]), "text_groups": len(cls["text_groups"]), "photos": len(images)}
    W, H = instr_page.rect.width, instr_page.rect.height
    PH, PW = out_hw
    sx, sy = PW / W, PH / H
    # 吹き出し: 文字の塊と写真。引出線の先を平面図座標にする
    callouts: list[Callout] = []
    def crop(r: pdfio.Rect) -> bytes:
        rr = pdfio.Rect(r.x0 - 4, r.y0 - 4, r.x1 + 4, r.y1 + 4) & instr_page.rect
        return instr_page.get_pixmap(matrix=pdfio.Matrix(2, 2), clip=rr, alpha=False).tobytes("png")
    def pdf_text(r: pdfio.Rect) -> str:
        try:
            return " ".join(instr_page.get_text("text", clip=r).split())
        except Exception:  # noqa: BLE001
            return ""
    anchor_map: dict[tuple, Callout] = {}
    for r in cls["text_groups"]:
        co = Callout("text", tuple(r), crop(r), text=pdf_text(r)); callouts.append(co); anchor_map[tuple(r)] = co
    for r, xref in images:
        near = [c for c in callouts if c.kind == "text" and pdfio.Rect(r.x0 - 10, r.y1 - 5, r.x1 + 10, r.y1 + 30).intersects(pdfio.Rect(c.bbox))]   # 写真の下の文字
        co = Callout("photo", tuple(r), crop(r), text=" ".join(c.text for c in near)); callouts.append(co); anchor_map[tuple(r)] = co
    for d, src_rect, kind, tgt in cls["leaders"]:
        col = d.color if d.color is not None else d.fill
        t = (float(tgt[0] * sx), float(tgt[1] * sy))
        if src_rect is not None and tuple(src_rect) in anchor_map:
            co = anchor_map[tuple(src_rect)]
            if co.target is None:
                co.target, co.color = t, col
            else:
                callouts.append(Callout(co.kind, co.bbox, co.png, t, col, text=co.text))
        else:
            callouts.append(Callout("leader", tuple(d.rect), None, t, col))
    if not marks:
        return {}, ["色の線が無い"], diag, callouts
    cols = np.array([[int(round(c * 255)) for c in (d.color if d.color is not None else d.fill)] for d in marks], np.uint8)
    w = np.array([max(d.rect.width, 1) * max(d.rect.height, 1) for d in marks], np.float32)
    assign = _cluster_colors(cols, w, merge_de, 8)
    out: dict[tuple, np.ndarray] = {}
    px_per_mm = S / mm_per_pt
    for k in range(assign.max() + 1):
        keep = [d for d, a in zip(marks, assign) if a == k]
        if not keep:
            continue
        img = render_drawings(W, H, [(d, {"color": (0, 0, 0), "fill": (0, 0, 0) if d.fill is not None else None, "width": max(d.width, 1.5)}) for d in keep], S)
        m = (img.min(axis=2) < 128).astype(np.uint8) * 255
        if m.shape != (PH, PW):
            m = cv2.resize(m, (PW, PH), interpolation=cv2.INTER_NEAREST)
        # ハッチの間隔: 線の塊どうしの最近傍距離の中央値 → その 1.3 倍で閉じる
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        close_px = int(600 * px_per_mm)
        if n > 3:
            from scipy.spatial import cKDTree
            pts = cen[1:]
            d2, _ = cKDTree(pts).query(pts, k=min(3, len(pts)))
            nn = float(np.median(d2[:, 1])) if d2.shape[1] > 1 else close_px
            close_px = int(min(max(nn * 1.3, 300 * px_per_mm), 3000 * px_per_mm))
        k_ = max(3, close_px | 1)
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_, k_)))
        # 穴を埋めて面にする。細長いまま残る成分（引出線の残り）は落とす
        from floortint.rooms import fill_holes
        filled = fill_holes(closed > 0).astype(np.uint8) * 255
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        keep_m = np.zeros_like(filled)
        for i in range(1, n2):
            a = st2[i, cv2.CC_STAT_AREA]; bw, bh = st2[i, cv2.CC_STAT_WIDTH], st2[i, cv2.CC_STAT_HEIGHT]
            if a < 2.0 * (1000 * px_per_mm) ** 2:
                continue
            if a < 0.25 * bw * bh and max(bw, bh) > 6 * min(bw, bh):
                continue
            keep_m[lab2 == i] = 255
        if keep_m.any():
            rgb = tuple(float(v) / 255 for v in np.average(cols[assign == k], axis=0, weights=w[assign == k]))
            out[tuple(round(c, 3) for c in rgb)] = keep_m
        diag[f"color{k}"] = {"drawings": len(keep), "close_px": close_px}
    return out, [], diag, callouts
