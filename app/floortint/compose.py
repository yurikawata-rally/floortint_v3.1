"""合成。床の模様 → 既存床 → 家具の影 → 家具の白 → 線画 → 室名 の順に重ねる。"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from floortint.legend import Material
from floortint.regions import Region
from floortint.textures import DEFAULT_COLORS, TextureParams, hex_to_rgb, make_texture

FONT_CANDIDATES = ["/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf", "/usr/share/fonts/truetype/ipafont-gothic/ipagp.ttf",
                   "/usr/share/fonts/opentype/ipafont/ipagp.ttf"]


@dataclass
class ComposeParams:
    mode: str = "material"            # material | flat
    texture: TextureParams = field(default_factory=TextureParams)
    shadow_px: int = 10               # 影のずらし量 [px]（右下方向）
    shadow_blur: float = 6.0          # 影のぼかし σ [px]
    shadow_alpha: float = 0.38        # 影の濃さ
    line_alpha: float = 0.85          # 線画の濃さ
    furniture_white: int = 252        # 家具の白
    paint_existing: bool = False      # 「変更なし」領域を塗る
    existing_color: str = "#e3e1dd"
    room_min_m2: float = 3.0          # 既存床として塗る部屋の最小面積
    default_direction: float = 0.0    # 矢印の無い領域の貼り方向 [deg]
    text_scale: float = 1.0
    text_color: tuple = (60, 60, 60)
    colors: dict = field(default_factory=dict)        # material_key → hex（画面での上書き）
    deform: dict = field(default_factory=dict)        # material_key → 固定倍率（0 または未指定 = 全体設定に従う）
    directions: dict = field(default_factory=dict)    # region_id → deg
    disabled: set = field(default_factory=set)        # region_id
    plant_mode: str = "original"      # original（元の色のまま）| line（線を指定色に）| fill（指定色で塗る）| hide
    plant_color: str = "#7fb069"
    plant_alpha: float = 0.55         # fill のときの塗りの濃さ
    structure_mode: str = "line"      # line（線画と同じ）| fill（指定色で塗る）| hide
    structure_color: str = "#9a9a9a"
    images: dict = field(default_factory=dict)        # material_key → 床材写真の列（ライブラリから）


def material_color(m: Material, p: ComposeParams) -> str:
    if m.key in p.colors:
        return p.colors[m.key]
    if p.mode == "flat" and m.color is not None:
        r, g, b = m.color
        mix = lambda c: int(round((c * 0.55 + 0.45) * 255))   # 白と混ぜて淡く
        return "#%02x%02x%02x" % (mix(r), mix(g), mix(b))
    pal = DEFAULT_COLORS.get(m.kind, DEFAULT_COLORS["other"])
    return pal[m.order % len(pal)]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for f in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _poly_px(poly: np.ndarray, off: tuple[float, float], S: float) -> np.ndarray:
    return np.round((poly - np.array(off)) * S).astype(np.int32)


def region_mask(r: Region, off: tuple[float, float], S: float, H: int, W: int) -> np.ndarray:
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [_poly_px(r.polygon, off, S)], 255)
    for h in r.holes:
        cv2.fillPoly(m, [_poly_px(h, off, S)], 0)
    return m


def _ink_alpha(img: np.ndarray, line_alpha: float) -> np.ndarray:
    """255 - (255 - img) * line_alpha を uint8 のまま計算する（float32 の全画素コピーを作らない）。"""
    inv = cv2.subtract(np.full_like(img, 255), img)
    return cv2.subtract(np.full_like(img, 255), cv2.convertScaleAbs(inv, alpha=float(line_alpha)))


def _mul255(canvas: np.ndarray, a: np.ndarray) -> np.ndarray:
    """canvas * a / 255（乗算合成）を uint8 のまま。"""
    return cv2.multiply(canvas, a, scale=1.0 / 255.0)


def compose(H: int, W: int, S: float, off: tuple[float, float], px_per_mm: float,
            regions: list[Region], materials: dict[str, Material], lineart: np.ndarray, furniture: np.ndarray,
            enclosed: np.ndarray, texts: list, p: ComposeParams, zone_mask: np.ndarray | None = None,
            debug: dict | None = None, plant_img: np.ndarray | None = None, struct_img: np.ndarray | None = None,
            layers: dict | None = None, room_masks: list | None = None) -> tuple[np.ndarray, np.ndarray]:
    """戻り値: (完成画像 RGB, 塗った範囲のマスク)。off は床仕上図座標→平面図座標の差 (dx, dy)。
    zone_mask は図面部（1）とそれ以外（0）。既存床はこの中だけに塗る。
    plant_img / struct_img は植栽・躯体だけを白地に描いた画像（レイヤーで分けられたとき）。
    layers を渡すと、重ねる順に RGBA のレイヤー画像（床・影・家具・躯体・線画・植栽・文字）を入れる（レイヤー分け書き出し用）。
    room_masks は (材種キー, マスク) の列。指示書／手動モードで、床仕上図の領域の代わりに塗る。
    debug を渡すと、工程ごとの所要時間と途中画像（縮小）を入れる。"""
    import time
    from floortint.debugview import shrink
    tlog: dict[str, float] = {}
    snaps: dict[str, np.ndarray] = {}
    t0 = time.time()

    def mark(name: str, snap: np.ndarray | None = None):
        nonlocal t0
        tlog[name] = round(time.time() - t0, 3); t0 = time.time()
        if debug is not None and snap is not None:
            snaps[name] = shrink(snap, 1400)

    def rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        out = np.dstack([rgb, np.clip(alpha, 0, 255).astype(np.uint8)])
        return out
    canvas = np.full((H, W, 3), 255, np.uint8)
    painted = np.zeros((H, W), np.uint8)
    # 大きい領域から塗り、小さい島が上に来るようにする
    for r in sorted(regions, key=lambda z: -z.area_pt2):
        if not r.enabled or r.id in p.disabled:
            continue
        m = materials.get(r.material_key)
        if m is None or m.canceled:
            continue
        mask = region_mask(r, off, S, H, W)
        # 隣の領域との継ぎ目（多角形の簡略化で生じる 1〜2px の隙間）を埋める: 2px 広げるが、塗り済みには重ねない
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        mask[painted > 0] = 0
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        ang = p.directions.get(r.id, r.direction if r.direction is not None else p.default_direction)
        ov = p.deform.get(m.key, 0) or 0
        tp = TextureParams(**{**p.texture.__dict__, **({"deform": ov, "target_mm": 0.0} if ov > 0 else {})})
        tex = make_texture(material_color(m, p), m.kind, m.pattern, m.size_mm, -ang, x1 - x0, y1 - y0, px_per_mm, tp,
                           flat=(p.mode == "flat"), images=p.images.get(m.key))
        sub = mask[y0:y1, x0:x1] > 0
        canvas[y0:y1, x0:x1][sub] = tex[sub]
        painted[mask > 0] = 255
    for mkey, mask, ang in (room_masks or []):
        m = materials.get(mkey)
        if m is None or m.canceled:
            continue
        mask = mask.copy(); mask[painted > 0] = 0
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        ov = p.deform.get(m.key, 0) or 0
        tp = TextureParams(**{**p.texture.__dict__, **({"deform": ov, "target_mm": 0.0} if ov > 0 else {})})
        tex = make_texture(material_color(m, p), m.kind, m.pattern, m.size_mm, -float(ang or 0.0), x1 - x0, y1 - y0, px_per_mm, tp,
                           flat=(p.mode == "flat"), images=p.images.get(m.key))
        sub = mask[y0:y1, x0:x1] > 0
        canvas[y0:y1, x0:x1][sub] = tex[sub]
        painted[mask > 0] = 255
    mark("床の模様", canvas)
    # 既存床（変更なし）: 囲まれた部屋のうち未着色で家具を含むもの
    if p.paint_existing and "existing" in materials:
        m = materials["existing"]
        amin = p.room_min_m2 * (1000 * px_per_mm) ** 2
        ids, counts = np.unique(enclosed, return_counts=True)
        # 画像の縁に触れる成分（建物の外）は除く
        border = set(np.unique(np.concatenate([enclosed[0], enclosed[-1], enclosed[:, 0], enclosed[:, -1]])).tolist())
        if zone_mask is not None:
            border |= set(np.unique(enclosed[zone_mask == 0]).tolist())
        big = [i for i, c in zip(ids, counts) if i > 0 and c >= amin and i not in border]
        if big:
            cand = np.isin(enclosed, big)
            cand &= painted == 0
            # 家具を含む部屋だけ（家具の輪郭は線なので、家具マスクを少し膨らませて接する部屋を探す）
            near = cv2.dilate(furniture, np.ones((5, 5), np.uint8)) > 0
            rooms = np.unique(enclosed[near & cand])
            rooms = [i for i in rooms if i in big]
            if rooms:
                mask = np.isin(enclosed, rooms) & (painted == 0)
                mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
                ov = p.deform.get("existing", 0) or 0
                tp = TextureParams(**{**p.texture.__dict__, **({"deform": ov, "target_mm": 0.0} if ov > 0 else {})})
                tex = make_texture(p.colors.get("existing", p.existing_color), "existing", "流し", (500, 500), 0.0, W, H,
                                   px_per_mm, tp, flat=(p.mode == "flat"))
                canvas[mask] = tex[mask]
                painted[mask] = 255
        mark("既存床", canvas)
    if layers is not None:
        layers["床"] = rgba(canvas, painted)
    # 家具の影（右下）
    if p.shadow_px > 0 and p.shadow_alpha > 0:
        sh = np.zeros_like(furniture)
        o = int(p.shadow_px)
        sh[o:, o:] = furniture[:-o, :-o] if o > 0 else furniture
        sh = cv2.GaussianBlur(sh.astype(np.float32) / 255, (0, 0), max(0.1, p.shadow_blur))
        sh = np.clip(sh, 0, 1) * p.shadow_alpha
        sh[furniture > 0] = 0
        sh *= (painted > 0)                                # 塗っていない場所（白）には影を落とさない
        canvas = _mul255(canvas, cv2.cvtColor((255 - sh * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB))
        if layers is not None:
            layers["影"] = rgba(np.zeros((H, W, 3), np.uint8), sh * 255)
        mark("影", (255 - sh * 255 / max(p.shadow_alpha, 1e-6)).astype(np.uint8))
    # 家具は白のまま
    canvas[furniture > 0] = p.furniture_white
    if layers is not None:
        layers["家具"] = rgba(np.full((H, W, 3), p.furniture_white, np.uint8), furniture)
    mark("家具の白", canvas)
    # 躯体（レイヤーで分けられたとき）: 塗る／線のまま／消す
    if struct_img is not None and p.structure_mode != "hide":
        sink = struct_img.min(axis=2) < 200
        if p.structure_mode == "fill":
            # 閉じた柱・壁の内側を塗る: 線の外側（画像の縁から到達できる所）以外を塗りつぶす
            closed = cv2.morphologyEx(sink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            ff = closed.copy(); hmask = np.zeros((H + 2, W + 2), np.uint8)
            cv2.floodFill(ff, hmask, (0, 0), 1)
            inside = (ff == 0) | sink
            col = np.array(ImageColor.getrgb(p.structure_color), np.uint8)
            canvas[inside] = col
            if layers is not None:
                layers["躯体"] = rgba(np.broadcast_to(col, (H, W, 3)).copy(), inside.astype(np.uint8) * 255)
        else:
            sa = _ink_alpha(struct_img, p.line_alpha)
            canvas = _mul255(canvas, sa)
            if layers is not None:
                layers["躯体"] = rgba(struct_img, (255 - sa.min(axis=2)))
        mark("躯体", canvas)
    # 線画を乗算で重ねる
    la = _ink_alpha(lineart, p.line_alpha)
    canvas = _mul255(canvas, la)
    if layers is not None:
        layers["線画"] = rgba(lineart, 255 - la.min(axis=2))
    mark("線画", canvas)
    # 植栽（レイヤーで分けられたとき）: 元の色／線を指定色／指定色で塗る／消す
    if plant_img is not None and p.plant_mode != "hide":
        pink = plant_img.min(axis=2) < 235
        pl = np.full((H, W, 3), 255, np.uint8)
        if p.plant_mode == "original":
            pl = plant_img
        elif p.plant_mode == "line":
            pl[pink] = ImageColor.getrgb(p.plant_color)
        else:   # fill: 線で囲まれた内側を薄く塗り、線は指定色
            closed = cv2.morphologyEx(pink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            ff = closed.copy(); hmask = np.zeros((H + 2, W + 2), np.uint8)
            cv2.floodFill(ff, hmask, (0, 0), 1)
            inside = (ff == 0)
            col = np.array(ImageColor.getrgb(p.plant_color), np.float32)
            fillc = (col * p.plant_alpha + 255 * (1 - p.plant_alpha)).astype(np.uint8)
            pl[inside] = fillc
            pl[pink] = (col * 0.75).astype(np.uint8)
        pa = _ink_alpha(pl, p.line_alpha)
        canvas = _mul255(canvas, pa)
        if layers is not None:
            layers["植栽"] = rgba(pl, 255 - pa.min(axis=2))
        mark("植栽", canvas)
    # 室名
    if texts:
        before = canvas.copy()
        im = Image.fromarray(canvas)
        dr = ImageDraw.Draw(im)
        ink = lineart.min(axis=2) < 200
        for x, y, sz, t in texts:
            f = _font(max(8, int(round(sz * S * p.text_scale))))
            px, py = (x - off[0]) * S, (y - off[1]) * S
            bx0, by0, bx1, by1 = dr.textbbox((px, py), t, font=f)
            bx0, by0, bx1, by1 = max(0, int(bx0)), max(0, int(by0)), min(W, int(bx1)), min(H, int(by1))
            if bx1 <= bx0 or by1 <= by0:
                continue
            # CAD が文字を線でも描いている場合（線画に既に文字がある）は二重に描かない
            if ink[by0:by1, bx0:bx1].mean() > 0.06:
                continue
            dr.text((px, py), t, fill=p.text_color, font=f)
        canvas = np.array(im)
        if layers is not None:
            diff = np.abs(canvas.astype(np.int16) - before.astype(np.int16)).max(axis=2) > 0
            layers["文字"] = rgba(canvas, diff.astype(np.uint8) * 255)
        mark("室名", None)
    if debug is not None:
        debug["timings"] = tlog
        debug["images"] = snaps
    return canvas, painted


def overlay(lineart: np.ndarray, regions: list[Region], materials: dict[str, Material], off, S, arrows=None,
            alpha: float = 0.45) -> np.ndarray:
    """検証用: 復元した領域を凡例色で線画に重ね、矢印と方向を描く。"""
    H, W = lineart.shape[:2]
    out = lineart.copy()
    layer = np.full((H, W, 3), 255, np.uint8)
    for r in regions:
        m = materials.get(r.material_key)
        col = hex_to_rgb(m.hex if m and m.color else "#dddddd")
        mask = region_mask(r, off, S, H, W) > 0
        layer[mask] = col
        cv2.polylines(out, [_poly_px(r.polygon, off, S)], True, (0, 0, 0), 2)
        if r.direction is not None:
            cx, cy = (np.array(r.centroid) - np.array(off)) * S
            L = min(W, H) * 0.03
            dx, dy = np.cos(np.radians(r.direction)) * L, np.sin(np.radians(r.direction)) * L
            cv2.arrowedLine(out, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), (0, 0, 0), 3, tipLength=0.25)
            cv2.arrowedLine(out, (int(cx + dx), int(cy + dy)), (int(cx - dx), int(cy - dy)), (0, 0, 0), 3, tipLength=0.25)
    blend = (out.astype(np.float32) * (1 - alpha) + layer.astype(np.float32) * alpha)
    ink = lineart.min(axis=2) < 200
    blend[ink] = out[ink]
    return blend.astype(np.uint8)
