"""指示書（手書きマーカー／塗りつぶし）を読み、色ごとの「種」を平面図の座標に重ねて部屋に割り当てる。

入力は 3 通り。
  ベクタ PDF   CAD 図面の上に PDF 注釈ツールで描いたマーカー（半透明・太い線）。図形の色・不透明度・線幅で拾う。
  ラスタ PDF   図面の画像にタブレットで塗った画像が貼ってある PDF。彩度の高い画素を拾う。
  画像ファイル  PNG/JPG。ラスタ PDF と同じ。
位置合わせは、ベクタなら通り芯ラベル（床仕上図と同じ方法）、ラスタなら線画の特徴点（ORB）で拡大縮小つきの変換を求める。
色は似た色をまとめて最大 max_colors 種にし、画面で材種を対応づける（手書きの品番は読まない）。
割り当ては、色の塊（blob）が部屋の中に収まっていれば「部屋全体」、大きな部屋を細く通るだけなら「塗った形（通路）」。
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from floortint import pdfio

from floortint import pdfdoc
from floortint.pdfdoc import Drawing, render_drawings
from floortint.rooms import RoomSet


@dataclass
class InstructParams:
    min_width_pt: float = 4.0        # ベクタ: これ以上太い色つき線はマーカー
    use_opacity: bool = True         # ベクタ: 不透明度 < 1 の色つき図形はマーカー
    sat_min: int = 40                # ラスタ: 彩度（0〜255）の下限。淡い桃色のマーカーも拾う
    val_min: int = 70                # ラスタ: 明度の下限
    min_blob_m2: float = 0.6         # これより小さい塊は無視（植栽・記号）
    merge_de: float = 22.0           # 似た色をまとめる距離（Lab）
    max_colors: int = 8
    raster_scale: float = 2.0        # ラスタ解析の解像度 [px/pt]
    room_cover: float = 0.7          # 部屋全体にする: 色の領域が部屋のこの割合以上を覆う
    fill_cover: float = 0.35         # 塗りつぶし（輪でない）なら、この割合でも部屋全体
    cover_min: float = 0.12          # 画面で「部屋ごと」を強制したとき、この割合以上の部屋を取る
    share_min: float = 0.3           # 小さい部屋: 領域のこの割合以上がその部屋にあれば部屋全体
    small_room_m2: float = 60.0      # この面積以下を小さい部屋とみなす
    stroke_max_mm: float = 1500.0    # 塊の平均の太さがこれ以下なら「線」、超えれば「塗りつぶし」
    band_close_mm: float = 150.0     # 塊の小さな切れ目を閉じる幅
    band_simplify_mm: float = 120.0  # 塗った形モード: 輪郭をこの精度で直線化
    fills_as_marker: bool = False    # ベクタ: 色つきの塗りつぶし（不透明でも）をマーカーにする（PPT の図形など）
    marker_kinds: str = "both"       # both | fill | stroke  拾うマーカーの描き方を限る（塗り＝範囲、線＝柱・扉の印 など）
    manual_scale: float = 1.0        # 手動補正（ラスタ）
    manual_dx: float = 0.0           # [pt]
    manual_dy: float = 0.0


@dataclass
class Seed:
    key: str
    rgb: tuple[int, int, int]
    name: str
    mask: np.ndarray                 # (H,W) uint8 平面図座標（S px/pt）
    area_m2: float = 0.0
    blobs: int = 0
    legend: object = None            # pptinstr.LegendItem（色が一致した凡例）

    @property
    def hex(self) -> str:
        return "#%02x%02x%02x" % self.rgb

    def to_dict(self) -> dict:
        d = {"key": self.key, "color": self.hex, "name": self.name, "area_m2": round(self.area_m2, 1), "blobs": self.blobs}
        if self.legend is not None:
            d["legend"] = self.legend.to_dict()
        return d


@dataclass
class Paint:
    """塗る単位。room_id があれば部屋全体、無ければ塗った形。"""
    seed: str
    mask: np.ndarray
    room_id: int | None = None
    mode: str = "room"               # room | band
    area_m2: float = 0.0


@dataclass
class InstructionResult:
    seeds: list[Seed] = field(default_factory=list)
    kind: str = ""                   # vector | raster | image | ppt | annotated
    align: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0
    legend: list = field(default_factory=list)       # pptinstr.LegendItem（PPT の凡例から読んだ材種）
    arrows: list = field(default_factory=list)       # (x, y, deg) 平面図 S px
    seed_alpha: dict = field(default_factory=dict)   # 色キー → 図形の不透明度
    diag: dict = field(default_factory=dict)
    callouts: list = field(default_factory=list)     # pptinstr.Callout（手書きの文字・写真と引出線の先）


# ---------------------------------------------------------------- 色
def _lab(rgb: np.ndarray) -> np.ndarray:
    a = np.asarray(rgb, np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def color_name(rgb: tuple[int, int, int]) -> str:
    r, g, b = [c / 255 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.2:
        return "灰"
    deg = h * 360
    for lim, nm in ((15, "赤"), (40, "橙"), (65, "黄"), (170, "緑"), (200, "水色"), (250, "青"), (290, "紫"), (335, "桃"), (361, "赤")):
        if deg < lim:
            return nm
    return "色"


def _cluster_colors(cols: np.ndarray, weights: np.ndarray, de: float, kmax: int) -> np.ndarray:
    """色（RGB 0〜255）を貪欲にまとめ、各色の属する群の添字を返す。重みの大きい色から種にする。"""
    lab = _lab(cols)
    order = np.argsort(-weights)
    centers: list[np.ndarray] = []
    cw: list[float] = []
    assign = np.full(len(cols), -1, np.int32)
    for i in order:
        best, bd = -1, 1e9
        for j, c in enumerate(centers):
            d = float(np.linalg.norm(lab[i] - c))
            if d < bd:
                best, bd = j, d
        if best >= 0 and bd <= de:
            assign[i] = best
            centers[best] = (centers[best] * cw[best] + lab[i] * weights[i]) / (cw[best] + weights[i]); cw[best] += weights[i]
        elif len(centers) < kmax:
            assign[i] = len(centers); centers.append(lab[i].copy()); cw.append(float(weights[i]))
        else:
            assign[i] = best if best >= 0 else 0
    return assign


# ---------------------------------------------------------------- ベクタ
def _is_marker(d: Drawing, p: InstructParams) -> bool:
    col = d.color if d.color is not None else d.fill
    if col is None or (max(col) - min(col)) < 0.3:
        return False
    if p.marker_kinds == "fill" and d.fill is None:
        return False
    if p.marker_kinds == "stroke" and d.fill is not None:
        return False
    if p.use_opacity and ((d.color is not None and d.stroke_opacity < 0.99) or (d.fill is not None and d.fill_opacity < 0.99)):
        return True
    if d.color is not None and d.width >= p.min_width_pt:
        return True
    if p.fills_as_marker and d.fill is not None and max(d.rect.width, d.rect.height) >= 4.0:
        return True
    return False


def vector_seeds(instr_page: pdfio.Page, plan_page: pdfio.Page, S: float, px_per_mm: float, p: InstructParams,
                 res: InstructionResult, out_hw: tuple[int, int], plan_lineart: np.ndarray | None = None,
                 marks: list[Drawing] | None = None) -> list[Seed]:
    if marks is None:
        drs = pdfdoc.drawings(instr_page)
        marks = [d for d in drs if _is_marker(d, p)]
    res.align["markers"] = len(marks)
    if not marks:
        res.warnings.append("マーカー（半透明または太い色つき線）が見つからない。線幅の下限を下げるか、ラスタとして読む")
        return []
    al = pdfdoc.align(plan_page, instr_page)
    al.dx += p.manual_dx; al.dy += p.manual_dy
    res.align.update({"method": al.method, "dx": round(al.dx, 2), "dy": round(al.dy, 2), "residual": round(al.residual, 3), "scale": 1.0})
    if al.method == "phase":
        # 通り芯で合わない（PPT に貼った縮小図面の上に図形を描いた指示書など）: 図面の線を倍率つきで重ねて合わせる
        r = p.raster_scale
        pix = instr_page.get_pixmap(matrix=pdfio.Matrix(r, r), alpha=False)
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy()
        ink, _ = _ink_and_marks(rgb, p)
        if plan_lineart is not None:
            g = cv2.resize(plan_lineart.min(axis=2), None, fx=r / S, fy=r / S, interpolation=cv2.INTER_AREA)
            plan_ink = g < 200
        else:
            ppix = plan_page.get_pixmap(matrix=pdfio.Matrix(r, r), alpha=False)
            plan_ink, _ = _ink_and_marks(np.frombuffer(ppix.samples, np.uint8).reshape(ppix.h, ppix.w, ppix.n)[:, :, :3], p)
        hint = None; best = 0.0
        for info in instr_page.get_image_info():
            b = info.get("bbox")
            if b and (b[2] - b[0]) * (b[3] - b[1]) > best:
                best = (b[2] - b[0]) * (b[3] - b[1]); hint = (int(b[0] * r), int(b[1] * r), int(b[2] * r), int(b[3] * r))
        roi = hint if hint else _drawing_roi(ink)
        base = plan_ink.shape[1] / max(1, roi[2] - roi[0])
        M, sc, corr = _template_align(ink, plan_ink, roi, base * np.exp(np.linspace(np.log(0.3), np.log(1.5), 30)))
        if M is not None and corr >= 0.12:
            res.align.update({"method": "template", "scale": round(sc, 4), "corr": round(corr, 3), "dx": round(float(M[0, 2]) / r, 1), "dy": round(float(M[1, 2]) / r, 1)})
            MS = np.float32([[M[0, 0], M[0, 1], M[0, 2] * S / r], [M[1, 0], M[1, 1], M[1, 2] * S / r]])
            return _vector_seeds_with_matrix(instr_page, marks, MS, S, p, out_hw)
        res.warnings.append(f"指示書と平面図の位置合わせが弱い（相関 {corr:.2f}）。ずれていれば手動補正を使う")
    cols = np.array([[int(round(c * 255)) for c in (d.color if d.color is not None else d.fill)] for d in marks], np.uint8)
    w = np.array([max(d.rect.width, 1) * max(d.rect.height, 1) for d in marks], np.float32)
    assign = _cluster_colors(cols, w, p.merge_de, p.max_colors)
    W, H = instr_page.rect.width, instr_page.rect.height
    PH, PW = out_hw
    seeds = []
    for k in range(assign.max() + 1):
        group = [d for d, a in zip(marks, assign) if a == k]
        if not group:
            continue
        img = render_drawings(W, H, [(d, {"color": (0, 0, 0), "fill": (0, 0, 0) if d.fill is not None else None}) for d in group], S)
        m = (img.min(axis=2) < 128).astype(np.uint8) * 255
        # 指示書座標 → 平面図座標（平行移動）
        M = np.float32([[1, 0, -al.dx * S], [0, 1, -al.dy * S]])
        m = cv2.warpAffine(m, M, (PW, PH), flags=cv2.INTER_NEAREST)
        rgb = tuple(int(v) for v in np.average(cols[assign == k], axis=0, weights=w[assign == k]))
        seeds.append(Seed(f"s{k}", rgb, color_name(rgb), m))
    return seeds


def _vector_seeds_with_matrix(instr_page: pdfio.Page, marks: list[Drawing], MS: np.ndarray, S: float, p: InstructParams,
                              out_hw: tuple[int, int]) -> list[Seed]:
    """マーカー図形を指示書の座標で描き、行列 MS（指示書 S px → 平面図 S px）で平面図座標に移す。"""
    cols = np.array([[int(round(c * 255)) for c in (d.color if d.color is not None else d.fill)] for d in marks], np.uint8)
    w = np.array([max(d.rect.width, 1) * max(d.rect.height, 1) for d in marks], np.float32)
    assign = _cluster_colors(cols, w, p.merge_de, p.max_colors)
    W, H = instr_page.rect.width, instr_page.rect.height
    PH, PW = out_hw
    seeds = []
    for k in range(assign.max() + 1):
        group = [d for d, a in zip(marks, assign) if a == k]
        if not group:
            continue
        img = render_drawings(W, H, [(d, {"color": (0, 0, 0), "fill": (0, 0, 0) if d.fill is not None else None}) for d in group], S)
        m = (img.min(axis=2) < 128).astype(np.uint8) * 255
        m = cv2.warpAffine(m, MS, (PW, PH), flags=cv2.INTER_LINEAR)
        m = (m > 127).astype(np.uint8) * 255
        rgb = tuple(int(v) for v in np.average(cols[assign == k], axis=0, weights=w[assign == k]))
        seeds.append(Seed(f"s{k}", rgb, color_name(rgb), m))
    return seeds


# ---------------------------------------------------------------- ラスタ
def _ink_and_marks(rgb: np.ndarray, p: InstructParams) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    marks = (hsv[:, :, 1] >= p.sat_min) & (hsv[:, :, 2] >= p.val_min)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = (gray < 150) & ~cv2.dilate(marks.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return ink, marks


def _orb_align(src_ink: np.ndarray, dst_ink: np.ndarray) -> tuple[np.ndarray | None, int, float]:
    """src → dst の相似変換（拡大縮小・回転・平行移動）。戻り値 (2x3 行列, 内点数, 内点率)。"""
    orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31)
    a = (~src_ink).astype(np.uint8) * 255; b = (~dst_ink).astype(np.uint8) * 255
    ka, da = orb.detectAndCompute(a, None); kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None, 0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for m, n in (x for x in knn if len(x) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, len(good), 0.0
    src = np.float32([ka[m.queryIdx].pt for m in good]); dst = np.float32([kb[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=5000, confidence=0.995)
    if M is None:
        return None, 0, 0.0
    n = int(inl.sum())
    return M, n, n / max(1, len(good))


def _phase_align(src_ink: np.ndarray, dst_ink: np.ndarray) -> tuple[np.ndarray, float]:
    """拡大縮小を何段か試して位相相関が最も強いものを取る（ORB の予備）。"""
    best = (None, -1.0)
    b = dst_ink.astype(np.float32)
    for s in np.linspace(0.5, 1.6, 23):
        a = cv2.resize(src_ink.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        (dx, dy), resp = cv2.phaseCorrelate(a[:h, :w], b[:h, :w])
        if resp > best[1]:
            best = (np.float32([[s, 0, dx], [0, s, dy]]), float(resp))
    return best


def _template_align(src_ink: np.ndarray, dst_ink: np.ndarray, roi: tuple[int, int, int, int], scales: np.ndarray,
                    work: float = 0.5) -> tuple[np.ndarray | None, float, float]:
    """src の roi（指示書の図面部分）を倍率を変えながら dst（平面図）に重ね、相関が最大の倍率と位置を取る。
    線はぼかして密度にしてから比べる。戻り値 (2x3 行列 src→dst, 倍率, 相関)。"""
    x0, y0, x1, y1 = roi
    crop = src_ink[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return None, 1.0, 0.0
    dst = cv2.resize(dst_ink.astype(np.float32), None, fx=work, fy=work, interpolation=cv2.INTER_AREA)
    dst = cv2.GaussianBlur(dst, (0, 0), 2.0)
    best = (None, 1.0, -1.0)
    for sc in scales:
        f = sc * work
        t = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if t.shape[0] < 40 or t.shape[1] < 40 or t.shape[0] >= dst.shape[0] or t.shape[1] >= dst.shape[1]:
            continue
        t = cv2.GaussianBlur(t, (0, 0), 2.0)
        rmap = cv2.matchTemplate(dst, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(rmap)
        if mx > best[2]:
            best = ((loc[0] / work, loc[1] / work, sc), sc, float(mx))
    if best[0] is None:
        return None, 1.0, 0.0
    lx, ly, sc = best[0]
    M = np.float32([[sc, 0, lx - sc * x0], [0, sc, ly - sc * y0]])
    return M, sc, best[2]


def _drawing_roi(ink: np.ndarray) -> tuple[int, int, int, int]:
    """細い線が密な領域（図面）の外接矩形。写真（面の濃淡）や手書き（まばら）を除く。"""
    u = ink.astype(np.uint8)
    thin = u & ~cv2.morphologyEx(u, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))   # 太い塊（写真の暗部）を除く
    dens = cv2.blur(thin.astype(np.float32), (121, 121))
    thr = max(0.02, 0.35 * float(dens.max()))
    m = (dens >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return 0, 0, ink.shape[1], ink.shape[0]
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    return int(x), int(y), int(x + w), int(y + h)


def raster_seeds(rgb: np.ndarray, px_per_pt_src: float, plan_page: pdfio.Page, S: float, px_per_mm: float,
                 p: InstructParams, res: InstructionResult, out_hw: tuple[int, int],
                 roi_hint: tuple[int, int, int, int] | None = None, plan_lineart: np.ndarray | None = None) -> list[Seed]:
    """rgb は指示書の画像（任意の解像度）。平面図の線画と重ねて位置合わせし、色の塊を平面図座標に移す。
    指示書の図面は縮小したスクリーンショットで、写真や手書きが周りにあることが多いので、
    色の塊の範囲を図面部分とみなして切り出し、倍率を変えながらテンプレート相関で合わせる。弱ければ特徴点（ORB）も試す。"""
    ink, marks = _ink_and_marks(rgb, p)
    r = p.raster_scale
    if plan_lineart is not None:
        # 整理後の線画（壁・什器が黒で描かれている）を r px/pt に縮小して使う。元の PDF は躯体が薄い灰色のことがある
        f = r / S
        g = cv2.resize(plan_lineart.min(axis=2), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        plan_ink = g < 200
    else:
        pix = plan_page.get_pixmap(matrix=pdfio.Matrix(r, r), alpha=False)
        plan_rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
        plan_ink, _ = _ink_and_marks(plan_rgb, p)
    # 図面部分: PDF なら最大の画像の範囲、無ければ細い線が密な領域
    H, W = ink.shape
    roi = roi_hint if roi_hint is not None else _drawing_roi(ink)
    roi = (max(0, roi[0]), max(0, roi[1]), min(W, roi[2]), min(H, roi[3]))
    # 図面部分の外の彩度（写真・色見本）は無視する
    box = np.zeros_like(marks); box[roi[1]:roi[3], roi[0]:roi[2]] = True
    marks &= box
    # 倍率の範囲: 図面部分が平面図の 0.3〜1.5 倍の大きさになる範囲
    base = plan_ink.shape[1] / max(1, roi[2] - roi[0])
    scales = base * np.exp(np.linspace(np.log(0.3), np.log(1.5), 30))
    M, sc, corr = _template_align(ink, plan_ink, roi, scales)
    method = "template"
    n_inl, ratio = 0, 0.0
    if M is None or corr < 0.15:
        M2, n_inl, ratio = _orb_align(ink, plan_ink)
        if M2 is not None and n_inl >= 25:
            M, method = M2, "orb"
        elif M is None:
            M, resp = _phase_align(ink, plan_ink); method = "phase"
        res.warnings.append(f"位置合わせの相関が弱い（{corr:.2f}）。ずれていれば手動補正（倍率・平行移動）を使う")
    res.align["roi"] = [int(v) for v in roi]; res.align["corr"] = round(corr, 3)
    # 手動補正（倍率は中心まわり、平行移動は pt）
    if p.manual_scale != 1.0 or p.manual_dx or p.manual_dy:
        A = np.vstack([M, [0, 0, 1]])
        cx, cy = plan_ink.shape[1] / 2, plan_ink.shape[0] / 2
        T = np.array([[p.manual_scale, 0, cx * (1 - p.manual_scale) + p.manual_dx * r], [0, p.manual_scale, cy * (1 - p.manual_scale) + p.manual_dy * r], [0, 0, 1]])
        M = (T @ A)[:2]
    sc = float(math.hypot(M[0, 0], M[0, 1]))
    res.align.update({"method": method, "scale": round(sc, 4), "dx": round(float(M[0, 2]) / r, 1), "dy": round(float(M[1, 2]) / r, 1),
                      "inliers": n_inl, "inlier_ratio": round(ratio, 3), "rotation_deg": round(math.degrees(math.atan2(M[0, 1], M[0, 0])), 2)})
    # 確認画像: 指示書の線を平面図に重ねる
    warped = cv2.warpAffine(ink.astype(np.uint8) * 255, M, (plan_ink.shape[1], plan_ink.shape[0]))
    chk = np.full((*plan_ink.shape, 3), 255, np.uint8)
    chk[plan_ink] = (0, 0, 200); chk[warped > 0] = (200, 0, 0); chk[plan_ink & (warped > 0)] = (0, 0, 0)
    res.images["位置合わせ（赤=指示書、青=平面図、黒=一致）"] = chk
    # 色ごとに分けてから塊を取る（隣り合う色が 1 つの連結成分にならないように）
    mk = cv2.morphologyEx(marks.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    if not mk.any():
        res.warnings.append("色の塊が見つからない（彩度の下限を下げる、または最小面積を下げる）")
        return []
    q = (rgb[mk] // 16).astype(np.int32)                       # 16 階調に量子化して色の度数表を作る
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    cols = np.stack([(uniq // 256) * 16 + 8, ((uniq // 16) % 16) * 16 + 8, (uniq % 16) * 16 + 8], axis=1).astype(np.uint8)
    assign = _cluster_colors(cols, cnt.astype(np.float32), p.merge_de, p.max_colors)
    pix_cluster = np.full(mk.shape, -1, np.int32)
    pix_cluster[mk] = assign[inv]
    # 平面図座標での面積の下限（指示書 px → 平面図 px は sc 倍、平面図 r px/pt → mm）
    px_per_mm_src = (r / S) * px_per_mm / max(sc, 1e-6)
    amin = p.min_blob_m2 * (1000 * px_per_mm_src) ** 2
    PH, PW = out_hw
    MS = (M * (S / r)).astype(np.float32)                      # 指示書 px → 平面図 S px
    seeds = []
    n_blobs = 0
    for k in range(assign.max() + 1):
        cm = (pix_cluster == k).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cm, connectivity=8)
        ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= amin]
        if not ids:
            continue
        n_blobs += len(ids)
        m = np.isin(lab, ids).astype(np.uint8) * 255
        m = cv2.warpAffine(m, MS, (PW, PH), flags=cv2.INTER_LINEAR)
        m = (m > 127).astype(np.uint8) * 255
        sel = assign == k
        col = tuple(int(v) for v in np.average(cols[sel], axis=0, weights=cnt[sel]))
        seeds.append(Seed(f"s{k}", col, color_name(col), m))
    res.align["blobs"] = n_blobs
    return seeds


# ---------------------------------------------------------------- 入口
def read_instruction(data: bytes, filename: str, page_idx: int, plan_page: pdfio.Page, S: float, px_per_mm: float,
                     p: InstructParams, out_hw: tuple[int, int], plan_lineart: np.ndarray | None = None,
                     zone: pdfio.Rect | None = None) -> InstructionResult:
    """out_hw は平面図の線画の (H, W) px。種のマスクをこの大きさで作る。
    経路: (1) 図面 PDF に描き足した手書き（ページが平面図と同じ）→ 恒等変換で色の線を拾う
          (2) PPT 由来（図面の画像＋塗り図形）→ 画像の実データで位置合わせ、凡例・矢印も読む
          (3) それ以外は従来（ベクタのマーカー／ラスタ）。"""
    import time
    t0 = time.time()
    res = InstructionResult()
    name = (filename or "").lower()
    if name.endswith((".pptx", ".ppt")):
        from floortint import pptinstr
        data = pptinstr.pptx_to_pdf(data)
        name += ".pdf"
        res.align["converted"] = "pptx → pdf（LibreOffice）"
    if name.endswith(".pdf"):
        from floortint import pptinstr
        doc = pdfio.open(stream=data, filetype="pdf")
        page = doc[min(page_idx, len(doc) - 1)]
        if page.rotation:
            page.remove_rotation()
        seeds = None
        annot, iou = pptinstr.is_annotated_plan(page, plan_page)
        res.align["plan_ink_iou"] = round(iou, 3)
        if annot and p.marker_kinds != "fill":
            res.kind = "annotated"
            res.align.update({"method": "identity（図面に描き足した手書き）", "scale": 1.0, "dx": 0.0, "dy": 0.0})
            masks, warns, diag, callouts = pptinstr.handwritten_seeds(page, S, out_hw, zone or plan_page.rect, 1.0 / px_per_mm * S if px_per_mm else 70.0, p.merge_de)
            res.warnings += warns; res.diag = diag; res.callouts = callouts
            seeds = [Seed(f"s{i}", tuple(int(round(c * 255)) for c in col), color_name(tuple(int(round(c * 255)) for c in col)), m)
                     for i, (col, m) in enumerate(masks.items())]
        else:
            ppt = pptinstr.read_ppt_instruction(doc, page, plan_page, S, out_hw)
            good = bool(ppt.plan_images) and ppt.chosen >= 0 and (ppt.plan_images[ppt.chosen].corr >= 0.3 or ppt.plan_images[ppt.chosen].inliers >= 60)
            # PPT 経路は「プレゼン由来の PDF」か「品番つきの凡例がある」ときだけ。タブレットで塗った図面のスクショ（ispace）は従来のラスタ経路へ
            meta = " ".join(str(v) for v in doc.metadata.values()).lower() if hasattr(doc, "metadata") else ""
            ppt_like = any(k in meta for k in ("powerpoint", "impress", "libreoffice", "keynote", "google slides"))
            # さらに、塗り図形に彩度のある色が無ければ（スクショに塗った画像だけの資料）ラスタ経路へ
            has_sat = any(pptinstr._saturated(c) for c in ppt.seeds_masks)
            good = good and (ppt_like or any(li.product for li in ppt.legend)) and has_sat
            res.align["ppt_like"] = ppt_like
            if good and ppt.seeds_masks and p.marker_kinds != "stroke":
                res.kind = "ppt"
                pi = ppt.plan_images[ppt.chosen]
                res.align.update({"method": pi.method, "scale": round(pi.scale, 4), "corr": round(pi.corr, 3), "inliers": pi.inliers,
                                  "images": [{"xref": x.xref, "corr": round(x.corr, 3), "scale": round(x.scale, 3)} for x in ppt.plan_images]})
                res.warnings += ppt.warnings
                res.legend = ppt.legend; res.arrows = ppt.arrows
                seeds = []
                for i, (col, m) in enumerate(ppt.seeds_masks.items()):
                    rgb = tuple(int(round(c * 255)) for c in col)
                    sd = Seed(f"s{i}", rgb, color_name(rgb), m)
                    sd.legend = next((li for li in ppt.legend if max(abs(a - b) for a, b in zip(li.color, col)) < 0.03), None)
                    res.seed_alpha[sd.key] = ppt.seed_alpha.get(col, 1.0)
                    seeds.append(sd)
            elif good and p.marker_kinds != "stroke" and not ppt.seeds_masks:
                res.warnings += ppt.warnings
        if seeds is not None:
            out = _finish_seeds(seeds, px_per_mm, p)
            res.seeds = out
            res.seconds = time.time() - t0
            return res
        drs = [Drawing.from_raw(g) for g in page.get_drawings()]
        marks = [d for d in drs if _is_marker(d, p)]
        # マーカー図形の外接矩形の面積がページの 0.5% 以上ならベクタ。凡例の小さな色見本だけなら画像（ラスタ）として読む
        mark_area = sum(d.rect.width * d.rect.height for d in marks)
        page_area = page.rect.width * page.rect.height
        img_area = 0.0
        for info in page.get_image_info():
            b = info.get("bbox")
            if b:
                img_area += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        res.align["marker_drawings"] = len(marks); res.align["marker_area_ratio"] = round(mark_area / page_area, 4)
        res.align["image_area_ratio"] = round(min(1.0, img_area / page_area), 3)
        if marks and (mark_area >= 0.005 * page_area or img_area < 0.05 * page_area or p.fills_as_marker) and (img_area < 0.3 * page_area or p.fills_as_marker):
            res.kind = "vector"
            # 図面の画像（最大の画像）の外にあるマーカーは凡例の見本なので除く
            big = max((info.get("bbox") for info in page.get_image_info() if info.get("bbox")), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), default=None)
            if big is not None and (big[2] - big[0]) * (big[3] - big[1]) >= 0.15 * page_area:
                br = pdfio.Rect(big)
                marks = [d for d in marks if d.rect.intersects(br) or br.contains(d.rect) or d.rect.contains(br)]
                res.align["markers_in_drawing"] = len(marks)
            seeds = vector_seeds(page, plan_page, S, px_per_mm, p, res, out_hw, plan_lineart, marks)
        else:
            res.kind = "raster"
            r = p.raster_scale
            pix = page.get_pixmap(matrix=pdfio.Matrix(r, r), alpha=False)
            rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy()
            # 最大の画像（図面のスクリーンショット）の範囲を図面部分とする
            hint = None
            best = 0.0
            for info in page.get_image_info():
                b = info.get("bbox")
                if b and (b[2] - b[0]) * (b[3] - b[1]) > best:
                    best = (b[2] - b[0]) * (b[3] - b[1]); hint = (int(b[0] * r), int(b[1] * r), int(b[2] * r), int(b[3] * r))
            if best < 0.15 * page_area:
                hint = None
            seeds = raster_seeds(rgb, r, plan_page, S, px_per_mm, p, res, out_hw, hint, plan_lineart)
    else:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(data)).convert("RGB")
        rgb = np.array(im)
        # 画像は解像度が不明。幅を平面図の解析解像度に合わせてから特徴点を探す
        r = p.raster_scale
        target_w = int(plan_page.rect.width * r)
        if rgb.shape[1] > target_w * 1.5:
            f = target_w * 1.2 / rgb.shape[1]
            rgb = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        res.kind = "image"
        seeds = raster_seeds(rgb, r, plan_page, S, px_per_mm, p, res, out_hw, None, plan_lineart)
    res.seeds = _finish_seeds(seeds, px_per_mm, p)
    res.seconds = time.time() - t0
    return res


def _finish_seeds(seeds: list[Seed], px_per_mm: float, p: InstructParams) -> list[Seed]:
    """小さな塊を落とし、面積・塊数を付け、面積順にキーを振る（凡例の対応はキーに依らず Seed に持つ）。"""
    out = []
    amin = p.min_blob_m2 * (1000 * px_per_mm) ** 2
    for s in seeds:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(s.mask, connectivity=8)
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= amin]
        if not keep:
            continue
        s.mask = np.isin(lab, keep).astype(np.uint8) * 255
        s.blobs = len(keep)
        s.area_m2 = float((s.mask > 0).sum()) / (1000 * px_per_mm) ** 2
        out.append(s)
    out.sort(key=lambda z: -z.area_m2)
    for i, s in enumerate(out):
        s.key = f"s{i}"
    return out


# ---------------------------------------------------------------- 部屋への割り当て
def _regions_from_seeds(seeds: list[Seed], px_per_mm: float, p: InstructParams) -> list[dict]:
    """色の塊ごとに {seed, kind, mask, stroke, area}。kind は
      loop   輪（閉じた線）。mask は内側＋線
      fill   塗りつぶし（太さが stroke_max_mm より大きい）。mask は塊そのもの
      stroke 開いた線。mask は線そのもの（部屋を区切る境界として使う）
    入れ子の輪・塗りは、外側の輪の mask から外す。"""
    from floortint.rooms import fill_holes
    regs: list[dict] = []
    k = max(3, int(p.band_close_mm * px_per_mm) | 1)
    for s in seeds:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(s.mask, connectivity=8)
        for i in range(1, n):
            blob = lab == i
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            x0, y0, x1, y1 = max(0, x - k), max(0, y - k), min(blob.shape[1], x + w + k), min(blob.shape[0], y + h + k)
            sub = blob[y0:y1, x0:x1]
            closed = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
            filled = fill_holes(closed)
            hole = int(filled.sum()) - int(closed.sum())
            # 平均の太さ = 2×面積 ÷ 周長（穴の輪郭も含める）。マーカー線なら線幅、塗りつぶしなら数 m になる
            cs, _ = cv2.findContours(closed.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            per = sum(cv2.arcLength(c, True) for c in cs)
            thick_mm = 2 * float(closed.sum()) / max(per, 1.0) / px_per_mm
            if hole >= max(p.min_blob_m2 * (1000 * px_per_mm) ** 2, 0.25 * closed.sum()):
                kind = "loop"; reg = filled
            elif thick_mm > p.stroke_max_mm:
                kind = "fill"; reg = closed
            else:
                kind = "stroke"; reg = closed
            full = np.zeros(blob.shape, bool); full[y0:y1, x0:x1] = reg
            st = np.zeros(blob.shape, bool); st[y0:y1, x0:x1] = closed
            regs.append({"seed": s.key, "kind": kind, "mask": full, "stroke": st, "area": int(full.sum()), "thick_mm": round(thick_mm)})
    for a in regs:
        if a["kind"] != "loop":
            continue
        for b in regs:
            if a is b or b["area"] >= 0.9 * a["area"]:
                continue
            if (b["mask"] & a["mask"]).sum() >= 0.9 * b["area"]:
                a["mask"] = a["mask"] & ~b["mask"]
        a["area"] = int(a["mask"].sum())
    return regs


def assign_rooms(seeds: list[Seed], rs: RoomSet, px_per_mm: float, p: InstructParams,
                 mode_override: dict[str, str] | None = None) -> list[Paint]:
    """色の領域を部屋に割り当てる。mode_override は {色キー: "auto"|"room"|"band"}。
    小さい部屋（small_room_m2 以下）: 領域に room_cover 以上覆われる／領域の share_min 以上がその部屋にある → 部屋全体。
    大きい部屋（開放オフィス）: 輪の内側・塗りつぶしはその形のまま。開いた線は部屋を区切る境界として、
    線で分けた区画ごとに「最も長く接する線の色」を与える（通路を線で区切る描き方）。"""
    paints: list[Paint] = []
    rm = rs.room_map
    if rm is None or not seeds:
        return paints
    px_m2 = (1000 * px_per_mm) ** 2
    area_of = {r.id: int((rm == r.id).sum()) for r in rs.rooms}
    small_ids = [r.id for r in rs.rooms if r.area_m2 <= p.small_room_m2]
    big_ids = [r.id for r in rs.rooms if r.area_m2 > p.small_room_m2]
    regs = _regions_from_seeds(seeds, px_per_mm, p)
    want = lambda k: (mode_override or {}).get(k, "auto")
    # ---- 小さい部屋: 部屋全体
    taken: dict[int, tuple[str, float]] = {}
    for g in regs:
        if want(g["seed"]) == "band":
            continue
        ids, cnt = np.unique(rm[g["mask"]], return_counts=True)
        for rid, c in zip(ids, cnt):
            rid = int(rid)
            if rid <= 0:
                continue
            cover, share = c / max(1, area_of[rid]), c / max(1, g["area"])
            if rid in small_ids:
                whole = cover >= p.room_cover or share >= p.share_min or (g["kind"] == "fill" and cover >= p.fill_cover)
            else:
                whole = cover >= p.room_cover or (want(g["seed"]) == "room" and cover >= p.cover_min)
            if whole:
                prev = taken.get(rid)
                if prev is None or prev[1] < cover:
                    taken[rid] = (g["seed"], float(cover))
    for rid, (skey, cover) in taken.items():
        r = rs.by_id(rid)
        if r is not None:
            r.seed, r.seed_cover = skey, cover
            paints.append(Paint(skey, rs.mask(rid), rid, "room", r.area_m2))
    # ---- 大きい部屋: 輪の内側・塗りつぶしはそのまま、開いた線で区切る
    big_free = [rid for rid in big_ids if rid not in taken]
    if not big_free:
        return paints
    big_mask = np.isin(rm, big_free)
    eps = p.band_simplify_mm * px_per_mm
    shapes: list[tuple[str, np.ndarray]] = []          # (seed, mask) 輪・塗り
    strokes = [g for g in regs if g["kind"] == "stroke" and want(g["seed"]) != "room"]
    for g in regs:
        if g["kind"] == "stroke" or want(g["seed"]) == "room":
            continue
        m = g["mask"] & big_mask
        if m.sum() < p.min_blob_m2 * px_m2:
            continue
        shapes.append((g["seed"], m))
    occupied = np.zeros(rm.shape, bool)
    for _, m in shapes:
        occupied |= m
    stroke_union = np.zeros(rm.shape, bool)
    for g in strokes:
        stroke_union |= g["stroke"]
    free = big_mask & ~occupied & ~stroke_union
    # 線で区切られた区画 → 最も長く接する線の色
    pieces: dict[str, np.ndarray] = {}
    if strokes and free.any():
        n, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
        ring = np.ones((5, 5), np.uint8)
        near = {g["seed"]: cv2.dilate(g["stroke"].astype(np.uint8), ring) > 0 for g in strokes}
        near_any = np.zeros(rm.shape, bool)
        for v in near.values():
            near_any |= v
        for i in range(1, n):
            comp = lab == i
            if comp.sum() < 0.5 * px_m2:
                continue
            edge = comp & near_any
            if not edge.any():
                continue
            best, bl = None, 0
            for skey, nm in near.items():
                l = int((edge & nm).sum())
                if l > bl:
                    best, bl = skey, l
            # 接する長さが 1.5m 未満なら無視（線が部屋をかすめただけ）
            if best is None or bl < 1.5 * 1000 * px_per_mm * 5:
                continue
            pieces[best] = pieces.get(best, np.zeros(rm.shape, bool)) | comp
    # 線そのものは線の色に（区画の隙間を埋める）
    for g in strokes:
        m = g["stroke"] & big_mask & ~occupied
        if m.any():
            pieces[g["seed"]] = pieces.get(g["seed"], np.zeros(rm.shape, bool)) | m
    def simplified(m: np.ndarray) -> np.ndarray:
        u = m.astype(np.uint8) * 255
        cs, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = np.zeros_like(u)
        for c in cs:
            cv2.fillPoly(poly, [cv2.approxPolyDP(c, eps, True)], 255)
        poly[~big_mask] = 0
        return poly
    for skey, m in shapes:
        poly = simplified(m) if eps > 0 else m.astype(np.uint8) * 255
        paints.append(Paint(skey, poly, None, "band", float((poly > 0).sum()) / px_m2))
    for skey, m in pieces.items():
        if m.sum() < p.min_blob_m2 * px_m2:
            continue
        poly = simplified(m) if eps > 0 else m.astype(np.uint8) * 255
        paints.append(Paint(skey, poly, None, "band", float((poly > 0).sum()) / px_m2))
    return paints
