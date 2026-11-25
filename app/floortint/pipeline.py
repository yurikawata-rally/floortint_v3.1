"""入口。抽出（ファイル依存）と合成（見た目のパラメータ依存）を分け、画面側で別々にキャッシュできるようにする。"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from floortint import pdfio

from floortint import layers as layermod
from floortint import pdfdoc
from floortint.cleanplan import CleanParams, CleanResult, clean_layout
from floortint.compose import ComposeParams, compose, overlay
from floortint.legend import Material, duplicate_swatch_colors, parse_legend
from floortint.regions import Arrow, Region, assign_directions, extract_arrows, extract_regions, regions_bbox
from floortint.instruct import InstructParams, InstructionResult, assign_rooms, read_instruction
from floortint.rooms import RoomSet, find_rooms, mark_common_rooms, mark_core_rooms, room_overlay, split_mask_by_lines


@dataclass
class Params:
    dpi: int = 300
    clean: CleanParams = field(default_factory=CleanParams)
    comp: ComposeParams = field(default_factory=ComposeParams)
    region_scale: float = 3.0         # 領域復元のラスタ解像度 [px/pt]
    close_factor: float = 1.5         # ハッチ間隔に対するクロージング核の倍率
    min_region_m2: float = 0.3
    resolve_overlap: bool = True      # 材種の重なりはハッチ線に近い方に与える
    snap_to_lines: bool = True        # 領域を床仕上図の黒線で分割・にじみ除去し、黒線まで広げる
    grow_factor: float = 0.75         # 黒線まで広げる距離（ハッチ間隔に対する倍率）
    barrier_min_mm: float = 300.0     # 境界とみなす黒線の最小長さ [mm]（短い線＝椅子や記号は境界にしない）
    dx_adjust: float = 0.0            # 位置合わせの手動補正 [pt]
    dy_adjust: float = 0.0
    zone_margin_pt: float = 14.0
    scale_override: int | None = None     # 縮尺の手動指定（表題欄から読めない図面）
    # 指示書モード／手動モード
    instr: InstructParams = field(default_factory=InstructParams)
    room_min_m2: float = 2.0          # 部屋とみなす閉領域の最小面積
    materials: list = field(default_factory=list)        # 画面で定義した材種（Material の列）。床仕上図モードでは凡例を使う
    seed_materials: dict = field(default_factory=dict)   # 指示書の色キー → 材種キー
    seed_modes: dict = field(default_factory=dict)       # 指示書の色キー → auto | room | band
    room_materials: dict = field(default_factory=dict)   # 部屋 id → 材種キー（手動の上書き。"" は塗らない）
    room_directions: dict = field(default_factory=dict)  # 部屋 id → 貼り方向 [deg]
    default_material: str | None = None                  # 指示の無い部屋に塗る材種（None = 塗らない）
    paint_core: bool = False                             # EV・階段を塗るか（既定: 塗らない）
    paint_common: bool = False                           # 共用部（室名で判定: WC・PS・機械室など）を塗るか（既定: 塗らない）
    common_pattern: str = ""                             # 共用部とみなす室名の追加パターン（正規表現）
    room_splits: dict = field(default_factory=dict)      # 部屋 id → [(x0,y0,x1,y1) 平面図 S px]。線で部屋を分ける（共用部／専有部の張り分けなど）
    rect_covers: list = field(default_factory=list)      # [{"rect": (x0,y0,x1,y1) S px, "material": キー or ""（塗らない）, "direction": deg}] 矩形で上書き


@dataclass
class Extraction:
    plan_size: tuple[float, float]
    scale_denom: int | None
    mm_per_pt: float
    align: pdfdoc.Alignment
    materials: list[Material]
    regions: list[Region]
    arrows: list[Arrow]
    zone: tuple[float, float, float, float]      # 平面図座標での図面部
    clean: CleanResult
    S: float
    diag: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    stages: list = field(default_factory=list)   # StageLog の列（段階 1〜7）
    layer_plan: object = None                     # layers.LayerPlan（平面図のレイヤー一覧と仕分け）
    mode: str = "floor"                           # floor（床仕上図）| instruction（指示書）| manual（指示なし）
    rooms: RoomSet | None = None                  # 平面図の部屋（指示書・手動モード）
    instruction: InstructionResult | None = None
    paints: list = field(default_factory=list)    # 最後の割り当て結果（render で更新）
    paint_list: list = field(default_factory=list)   # 最後の塗る単位 (材種, マスク, 方向)
    subrooms: dict = field(default_factory=dict)  # 部屋 id → [(ラベル "12a", マスク)]（線で分割したとき）

    @property
    def material_map(self) -> dict[str, Material]:
        return {m.key: m for m in self.materials}


@dataclass
class RunResult:
    image: np.ndarray
    overlay: np.ndarray
    painted: np.ndarray
    extraction: Extraction
    crop: tuple[int, int, int, int]              # 出力のトリミング (x0, y0, x1, y1) px
    stage: object = None                         # 合成段階の StageLog

    def summary(self) -> dict:
        ex = self.extraction
        return {"scale": ex.scale_denom, "align": asdict(ex.align), "materials": [m.to_dict() for m in ex.materials],
                "regions": [r.to_dict(ex.mm_per_pt) for r in ex.regions], "arrows": len(ex.arrows),
                "arrows_assigned": sum(1 for r in ex.regions if r.direction is not None), "diag": ex.diag,
                "clean": ex.clean.stats, "timings": ex.timings, "mode": ex.mode,
                "layers": ex.layer_plan.summary() if ex.layer_plan is not None else None,
                "rooms": [r.to_dict() for r in ex.rooms.rooms] if ex.rooms is not None else None,
                "seeds": [s.to_dict() for s in ex.instruction.seeds] if ex.instruction is not None else None,
                "instruction_align": ex.instruction.align if ex.instruction is not None else None,
                "paints": [{"seed": q.seed, "room": q.room_id, "mode": q.mode, "area_m2": round(q.area_m2, 1), "material": q.material_key}
                           for q in ex.paints]}


def _page_gray(page: pdfio.Page, scale: float = 1.0) -> np.ndarray:
    pix = page.get_pixmap(matrix=pdfio.Matrix(scale, scale), colorspace=pdfio.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w).copy()


def _page_rgb(page: pdfio.Page, scale: float = 1.0) -> np.ndarray:
    pix = page.get_pixmap(matrix=pdfio.Matrix(scale, scale), alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy()


def extract(plan_bytes: bytes, plan_idx: int, floor_bytes: bytes, floor_idx: int, p: Params) -> Extraction:
    """段階 1〜7。各段階の所要時間・指標・途中画像を StageLog に残す。"""
    import cv2
    from floortint.debugview import StageLog, ink_overlay, label_image, overlay_mask, shrink, side_by_side
    stages: list[StageLog] = []
    t = {}; t0 = time.time()
    # ---- 1 取込
    pdoc = pdfio.open(stream=plan_bytes, filetype="pdf"); fdoc = pdfio.open(stream=floor_bytes, filetype="pdf")
    plan, floor = pdoc[plan_idx], fdoc[floor_idx]
    for pg_ in (plan, floor):
        if pg_.rotation:
            pg_.remove_rotation()                       # 回転ページ（図面帳の横向き）: 表示どおりの座標で図形・文字を読む
    scale = p.scale_override or pdfdoc.page_scale(plan) or pdfdoc.page_scale(floor)
    mmpt = pdfdoc.mm_per_pt(scale)
    S = p.dpi / 72.0
    plan_g, floor_g = _page_gray(plan), _page_gray(floor)
    fdr = pdfdoc.drawings(floor)
    pattern_info: dict = {}
    pitch_over: dict = {}
    pdr = pdfdoc.drawings(plan)
    st1 = StageLog("1 取込", time.time() - t0)
    st1.metrics = {"平面図": pdfdoc.page_title(plan) or "（名称なし）", "床仕上図": pdfdoc.page_title(floor) or "（名称なし）",
                   "縮尺": f"1/{scale}" if scale else "不明（1/200 と仮定）", "mm/pt": round(mmpt, 2),
                   "出力 px/pt": round(S, 3), "平面図の図形数": len(pdr), "床仕上図の図形数": len(fdr)}
    if not scale:
        st1.warn("縮尺が表題欄から読めない。1/200 と仮定して寸法を換算する")
    st1.images = {"平面図（入力）": shrink(cv2.cvtColor(plan_g, cv2.COLOR_GRAY2RGB), 1200),
                  "床仕上図（入力）": shrink(cv2.cvtColor(floor_g, cv2.COLOR_GRAY2RGB), 1200)}
    stages.append(st1); t0 = time.time()
    # ---- 2 位置合わせ
    al = pdfdoc.align(plan, floor)
    al.dx += p.dx_adjust; al.dy += p.dy_adjust
    t["align"] = time.time() - t0
    st2 = StageLog("2 位置合わせ", t["align"])
    st2.metrics = {"方法": {"grid": "通り芯ラベル", "grid-bubble": "通り芯（丸付き）", "phase": "位相相関"}.get(al.method, al.method),
                   "dx [pt]": round(al.dx, 2), "dy [pt]": round(al.dy, 2), "残差 [pt]": round(al.residual, 3), "対応点": al.matched,
                   "手動補正": (p.dx_adjust, p.dy_adjust)}
    if al.method == "phase":
        st2.warn("通り芯ラベルが両図に無い（または一致しない）ので画像の相関で合わせた。ずれていれば手動補正を使う")
    if al.residual > 1.0:
        st2.warn(f"位置合わせの残差が {al.residual:.2f}pt と大きい")
    st2.images = {"重ね合わせ（赤=平面図、青=床仕上図、黒=一致）": shrink(ink_overlay(plan_g, floor_g, int(round(al.dx)), int(round(al.dy))), 1600)}
    stages.append(st2); t0 = time.time()
    # ---- 3 凡例
    mats, zones = parse_legend(floor, fdr)
    dup_colors = duplicate_swatch_colors(floor, fdr)
    if dup_colors or not mats:
        # 同じ色で模様（斜線・格子・点）だけ違う床仕上図: 模様ごとに仮想の色を付け直してから凡例を読み直す
        from floortint.hatchpattern import pattern_pitch_overrides, recolor_by_pattern
        fdr2, pattern_info = recolor_by_pattern(fdr, floor.rect, only_colors=dup_colors or None)
        if pattern_info:
            fdr = fdr2
            mats, zones = parse_legend(floor, fdr)
            pitch_over = pattern_pitch_overrides(fdr, pattern_info)
    t["legend"] = time.time() - t0
    st3 = StageLog("3 凡例", t["legend"])
    if pattern_info:
        st3.metrics["同色で模様違いのハッチ（模様ごとに分けた）"] = pattern_info
    colored = {}
    for d in fdr:
        if d.color and d.color not in ((0., 0., 0.), (1., 1., 1.)):
            colored[d.color] = colored.get(d.color, 0) + 1
    legend_colors = {m.color for m in mats if m.color}
    unmatched = {c: n for c, n in colored.items() if c not in legend_colors and n >= 30}
    st3.metrics = {"材種": len(mats), "色つきの線の色数": len(colored), "凡例に無い色（30 本以上）": len(unmatched),
                   "行": [f"{m.name}｜{m.spec}｜{m.pattern}貼り{'｜中止' if m.canceled else ''}" for m in mats]}
    if unmatched:
        st3.warn("凡例に無い色の線が床仕上図にある: " + ", ".join(f"rgb{c}×{n}" for c, n in unmatched.items()) + "（塗られない）")
    if any(m.size_mm is None and m.kind != "existing" for m in mats):
        st3.warn("寸法の無い行がある。板状の貼り方なら 900×150mm、他は 500×500mm を仮定する")
    lr = None
    if zones:
        lr = pdfio.Rect(zones[0])
        for z in zones[1:]:
            lr |= z
        lr = pdfio.Rect(lr.x0 - 10, lr.y0 - 10, lr.x1 + 10, lr.y1 + 10) & floor.rect
        pix = floor.get_pixmap(matrix=pdfio.Matrix(2.5, 2.5), clip=lr, alpha=False)
        st3.images = {"凡例の切り出し": shrink(np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy(), 1600)}
    stages.append(st3); t0 = time.time()
    # ---- 4 領域
    rdebug: dict = {}
    regs, diag = extract_regions(floor, fdr, mats, zones, S=p.region_scale, min_area_pt2=p.min_region_m2 / (mmpt / 1000) ** 2,
                                 hole_min_pt2=0.4 / (mmpt / 1000) ** 2, close_factor=p.close_factor, debug=rdebug,
                                 resolve_overlap=p.resolve_overlap, snap_to_lines=p.snap_to_lines, grow_factor=p.grow_factor,
                                 barrier_min_pt=p.barrier_min_mm / mmpt, pitch_override=pitch_over)
    t["regions"] = time.time() - t0
    st4 = StageLog("4 領域", t["regions"])
    area_m2 = sum(r.area_pt2 for r in regs) * (mmpt / 1000) ** 2
    q = diag.pop("_quality", {})
    bq = q.get("boundary_pt", {})
    st4.metrics = {"領域": len(regs), "着色面積 [m²]": round(area_m2, 1), "ラスタ解像度 [px/pt]": p.region_scale,
                   "ハッチ線の取り込み率（平均）": q.get("hatch_recall_mean"), "領域のはみ出し無し率（平均）": q.get("hatch_precision_mean"),
                   "輪郭と黒線の距離 中央値 [mm]": round(bq["median"] * mmpt, 1) if bq else None,
                   "輪郭と黒線の距離 95% [mm]": round(bq["p95"] * mmpt, 1) if bq else None,
                   "材種ごと": {mats_key_name(mats, k): v for k, v in diag.items()}}
    if q.get("hatch_recall_mean") is not None and q["hatch_recall_mean"] < 0.98:
        st4.warn(f"ハッチ線の取り込み率が {q['hatch_recall_mean']:.3f}。閉じる核の倍率を上げるか、領域の最小面積を下げる")
    for k, v in diag.items():
        if v.get("regions", 0) == 0:
            st4.warn(f"{mats_key_name(mats, k)} は線が {v.get('strokes', 0)} 本あるが領域にならなかった")
    name_of = {m.key: m.name for m in mats}
    for k, imgs in rdebug.items():
        for nm, im in imgs.items():
            st4.images[f"{name_of.get(k, k)}: {nm}"] = im
    stages.append(st4); t0 = time.time()
    # ---- 5 方向
    arrows = extract_arrows(fdr, zones)
    assigned = assign_directions(regs, arrows)
    t["arrows"] = time.time() - t0
    st5 = StageLog("5 貼り方向", t["arrows"])
    with_dir = sum(1 for r in regs if r.direction is not None)
    st5.metrics = {"矢印候補": len(arrows), "領域に入った矢印": assigned, "方向つき領域": f"{with_dir}/{len(regs)}",
                   "角度の分布": sorted({round(a.angle) for a in arrows})}
    if with_dir < len(regs):
        st5.warn(f"{len(regs) - with_dir} 領域に矢印が無い。既定 0°（画面で上書き可）")
    fl_rgb = _page_rgb(floor, 1.5)
    for a in arrows:
        inside = any(r.contains(a.cx, a.cy) for r in regs)
        cv2.circle(fl_rgb, (int(a.cx * 1.5), int(a.cy * 1.5)), 9, (0, 160, 0) if inside else (220, 0, 0), 2)
    st5.images = {"矢印候補（緑=領域内、赤=領域外）": shrink(fl_rgb, 1600)}
    stages.append(st5); t0 = time.time()
    # ---- 6 図面整理 & 7 家具
    bb = regions_bbox(regs)
    if bb is None:
        bb = floor.rect
    zone = pdfio.Rect(bb.x0 - al.dx - p.zone_margin_pt, bb.y0 - al.dy - p.zone_margin_pt,
                        bb.x1 - al.dx + p.zone_margin_pt, bb.y1 - al.dy + p.zone_margin_pt) & plan.rect
    lp = layermod.inventory(pdr, p.clean.layer_overrides, zone)
    cl = clean_layout(plan, pdr, zone, S, S / mmpt, p.clean, lp)
    t["clean"] = time.time() - t0
    st6 = StageLog("6 図面整理", cl.stats.get("t_render", t["clean"]))
    st6.metrics = {"仕分けの方法": "レイヤー名" if cl.stats.get("by_layer") else "色・線種の規則（レイヤー無し）",
                   "図形の総数": cl.stats.get("total_drawings"), "残した図形": cl.stats.get("kept_drawings"),
                   "植栽": cl.stats.get("plant_drawings"), "躯体": cl.stats.get("structure_drawings"),
                   "落とした内訳": cl.stats.get("dropped"), "室名候補": cl.stats.get("texts"),
                   "図面部 [pt]": [round(v) for v in zone],
                   "レイヤー": [f"{'落' if li.drop else '残'} {li.name or '（無名）'}: {li.count} 件（{li.reason}）" for li in lp.layers]}
    if lp.has_layers and not cl.stats.get("by_layer"):
        st6.warn("レイヤーがあるが、色の規則で仕分けている（設定で無効）")
    zx0, zy0, zx1, zy1 = zone
    raw = _page_rgb(plan, 1.5)[int(zy0 * 1.5):int(zy1 * 1.5), int(zx0 * 1.5):int(zx1 * 1.5)]
    cleaned = cl.lineart[int(zy0 * S):int(zy1 * S), int(zx0 * S):int(zx1 * S)]
    st6.images = {"整理前（左）と整理後（右）": side_by_side(raw, cleaned, 1800)}
    stages.append(st6)
    st7 = StageLog("7 家具", cl.stats.get("t_furniture", 0.0))
    fcount = cl.stats.get("furniture_components", 0)
    st7.metrics = {"線で囲まれた成分": cl.stats.get("components"), "家具と判定": fcount,
                   "面積の範囲 [m²]": (p.clean.furniture_min_m2, p.clean.furniture_max_m2), "切れ目を埋める幅 [px]": p.clean.close_gaps_px,
                   "第 2 段（椅子・破線什器）で追加 [px]": cl.stats.get("furniture_added_px_pass2", 0),
                   "第 3 段（椅子の記号の凸包）": f'{cl.stats.get("chair_symbols", 0)} 個 / {cl.stats.get("furniture_added_px_pass3", 0)} px',
                   "家具判定から外した扉の弧": cl.stats.get("door_arcs", 0)}
    if fcount == 0:
        st7.warn("家具が 1 つも見つからない。線幅や切れ目の設定、家具の面積の範囲を見直す")
    sub = (slice(int(zy0 * S), int(zy1 * S)), slice(int(zx0 * S), int(zx1 * S)))
    st7.images = {"判定に使った線（切れ目を埋めた後）": shrink(cl.ink[sub] if cl.ink is not None else cleaned, 1400),
                  "囲まれた成分（色分け）": label_image(cl.enclosed[sub], 1400),
                  "家具の足元（赤）": overlay_mask(cleaned, cl.furniture[sub], (255, 0, 0), 0.5, 1400)}
    stages.append(st7)
    ex = Extraction((plan.rect.width, plan.rect.height), scale, mmpt, al, mats, regs, arrows, tuple(zone), cl, S, diag, t)
    ex.stages = stages
    ex.layer_plan = lp
    pdoc.close(); fdoc.close()
    return ex


def scan_layers(plan_bytes: bytes, plan_idx: int) -> list[dict]:
    """平面図ページのレイヤー一覧（画面の「レイヤー」設定用。抽出の前に軽く走らせる）。"""
    doc = pdfio.open(stream=plan_bytes, filetype="pdf")
    lp = layermod.inventory(pdfdoc.drawings(doc[plan_idx]))
    return [li.to_dict() for li in lp.layers] if lp.has_layers else []


def mats_key_name(mats, key: str) -> str:
    for m in mats:
        if m.key == key:
            return m.name
    return key


def room_paint_list(ex: Extraction, p: Params) -> list[tuple[str, np.ndarray, float]]:
    """指示書・手動モードの塗る単位を (材種キー, マスク, 貼り方向) の列にする。
    順序 = 優先順（compose は先に塗った画素を塗り直さない）: 矩形の上書き → 大きい部屋（線で分割した部分を含む）→ 小さい部屋 → 塗った形（通路）。
    コア（EV・階段）と共用部（室名）は既定で塗らない。部屋ごとの上書き（room_materials）は "12" のほか、分割後の "12a" "12b" でも指定できる。"""
    rs = ex.rooms
    if rs is None:
        return []
    px_per_mm = ex.S / ex.mm_per_pt
    for r in rs.rooms:
        r.seed, r.seed_cover, r.material_key = None, 0.0, None
    paints = assign_rooms(ex.instruction.seeds, rs, px_per_mm, p.instr, p.seed_modes) if ex.instruction is not None else []
    by_room = {q.room_id: q for q in paints if q.room_id is not None}
    arrows = ex.instruction.arrows if ex.instruction is not None else []
    def arrow_dir(mask: np.ndarray) -> float | None:
        for x, y, deg in arrows:
            xi, yi = int(x), int(y)
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi] > 0:
                return float(deg)
        return None
    rm_over = {str(k): v for k, v in p.room_materials.items()}
    rd_over = {str(k): v for k, v in p.room_directions.items()}
    out: list[tuple[str, np.ndarray, float]] = []
    erase = np.zeros(rs.room_map.shape, bool)
    # ---- 矩形の上書き（最優先）
    for cv in p.rect_covers:
        x0, y0, x1, y1 = [int(round(v)) for v in cv.get("rect", (0, 0, 0, 0))]
        x0, x1 = sorted((max(0, x0), min(rs.room_map.shape[1], x1))); y0, y1 = sorted((max(0, y0), min(rs.room_map.shape[0], y1)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        m_ = np.zeros(rs.room_map.shape, np.uint8); m_[y0:y1, x0:x1] = 255
        if cv.get("material"):
            out.append((cv["material"], m_, float(cv.get("direction") or 0.0)))
        else:
            erase[y0:y1, x0:x1] = True
    ex.subrooms = {}
    for r in sorted(rs.rooms, key=lambda z: -z.area_m2):
        q = by_room.get(r.id)
        mkey = p.seed_materials.get(q.seed) if q is not None else None
        if str(r.id) in rm_over:                       # 手動の上書き（"" は塗らない）
            mkey = rm_over[str(r.id)] or None
        elif r.core in ("EV", "階段") and not p.paint_core:
            mkey = None
        elif r.core == "共用" and not p.paint_common:
            mkey = None
        elif mkey is None:
            mkey = p.default_material
        r.material_key = mkey
        if q is not None:
            q.material_key = mkey
        m_room = rs.mask(r.id)
        parts = [("", m_room)]
        if p.room_splits.get(r.id) or p.room_splits.get(str(r.id)):
            lines = p.room_splits.get(r.id) or p.room_splits.get(str(r.id))
            subs = split_mask_by_lines(m_room, lines)
            if len(subs) >= 2:
                parts = [(chr(ord("a") + i), sm) for i, sm in enumerate(subs[:26])]
                ex.subrooms[r.id] = [(f"{r.id}{lab}", sm) for lab, sm in parts]
        for lab, m_ in parts:
            key = f"{r.id}{lab}"
            mk = mkey
            if lab and key in rm_over:
                mk = rm_over[key] or None
            if not mk:
                continue
            if erase.any():
                m_ = m_.copy(); m_[erase] = 0
            ad = arrow_dir(m_)
            out.append((mk, m_, float(rd_over.get(key, rd_over.get(str(r.id), ad if ad is not None else 0.0)))))
    for q in paints:
        if q.room_id is None:
            mkey = p.seed_materials.get(q.seed)
            q.material_key = mkey
            if mkey:
                m_ = q.mask
                if erase.any():
                    m_ = m_.copy(); m_[erase] = 0
                ad = arrow_dir(m_)
                out.append((mkey, m_, ad if ad is not None else 0.0))
    ex.paints = paints
    ex.paint_list = out
    return out


def paint_overlay_with(ex: Extraction, p: Params) -> np.ndarray:
    """割り当てを計算してから検証図を作る（画面の「部屋ごとの割り当て」用）。"""
    mats = {m.key: m for m in p.materials} if p.materials else ex.material_map
    room_paint_list(ex, p)
    return paint_overlay(ex, mats, p.comp)


def paint_overlay(ex: Extraction, mats: dict, comp: ComposeParams | None = None) -> np.ndarray:
    """検証用: 塗る単位（部屋・線で分けた部分・矩形）を材種の色で塗り分け、指示書の色の塊を半透明で重ねる。番号は部屋 id（分割後は id+a/b）。"""
    import cv2
    from floortint.compose import material_color
    from floortint.textures import hex_to_rgb
    rs = ex.rooms
    comp = comp or ComposeParams()
    out = room_overlay(ex.clean.lineart, rs, {r.id: (235, 235, 235) for r in rs.rooms}, 0.55, number=False)
    painted = np.zeros(rs.room_map.shape, bool)
    for mkey, mask, _ in getattr(ex, "paint_list", None) or []:
        m = (mask > 0) & ~painted
        if mkey in mats:
            c = np.array(hex_to_rgb(material_color(mats[mkey], comp)), np.float32)
            base = ex.clean.lineart[m].astype(np.float32)
            out[m] = (base * 0.45 + c * 0.55).astype(np.uint8)
        painted |= m
    for r in rs.rooms:
        labels = [(str(r.id), r.centroid)]
        subs = getattr(ex, "subrooms", {}).get(r.id)
        if subs:
            labels = []
            for key, sm in subs:
                ys, xs = np.where(sm > 0)
                if len(xs):
                    labels.append((key, (float(xs.mean()), float(ys.mean()))))
        for text, (cx, cy) in labels:
            cx, cy = int(cx), int(cy)
            cv2.putText(out, text, (cx - 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, (cx - 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
    if ex.instruction is not None:
        for s in ex.instruction.seeds:
            m = s.mask > 0
            out[m] = (out[m].astype(np.float32) * 0.6 + np.array(s.rgb, np.float32) * 0.4).astype(np.uint8)
            cs, _ = cv2.findContours(s.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cs, -1, tuple(int(v * 0.7) for v in s.rgb), 2)
    return out


def render(ex: Extraction, p: Params, layers: dict | None = None) -> RunResult:
    H, W = ex.clean.lineart.shape[:2]
    off = (ex.align.dx, ex.align.dy)
    mats = ex.material_map
    zx0, zy0, zx1, zy1 = ex.zone
    zone_mask = np.zeros((H, W), np.uint8)
    zone_mask[int(zy0 * ex.S):int(zy1 * ex.S) + 1, int(zx0 * ex.S):int(zx1 * ex.S) + 1] = 1
    from floortint.debugview import StageLog
    t0 = time.time()
    cdebug: dict = {}
    room_masks = None
    if ex.mode != "floor":
        mats = {m.key: m for m in p.materials} if p.materials else mats
        room_masks = room_paint_list(ex, p)
    img, painted = compose(H, W, ex.S, off, ex.S / ex.mm_per_pt, ex.regions, mats, ex.clean.lineart, ex.clean.furniture,
                           ex.clean.enclosed, ex.clean.texts, p.comp, zone_mask, debug=cdebug,
                           plant_img=ex.clean.plant_img, struct_img=ex.clean.struct_img, room_masks=room_masks,
                           layers=layers)
    t_comp = time.time() - t0; t0 = time.time()
    if ex.mode == "floor":
        ov = overlay(ex.clean.lineart, ex.regions, mats, off, ex.S, ex.arrows)
    else:
        ov = paint_overlay(ex, mats, p.comp)
    x0, y0, x1, y1 = ex.zone
    crop = (max(0, int(x0 * ex.S)), max(0, int(y0 * ex.S)), min(W, int(x1 * ex.S) + 1), min(H, int(y1 * ex.S) + 1))
    st8 = StageLog("8 合成", t_comp)
    painted_m2 = float((painted > 0).sum()) / (ex.S / ex.mm_per_pt * 1000) ** 2
    st8.metrics = {"出力 [px]": f"{crop[2] - crop[0]}×{crop[3] - crop[1]}", "塗った面積 [m²]": round(painted_m2, 1),
                   "工程ごとの時間 [s]": cdebug.get("timings", {}), "検証図の生成 [s]": round(time.time() - t0, 3),
                   "モード": p.comp.mode, "拡大率": p.comp.texture.deform, "無効にした領域": sorted(p.comp.disabled)}
    for nm, im in cdebug.get("images", {}).items():
        cs = tuple(int(v * im.shape[1] / W) for v in crop)          # 縮小画像に合わせて図面部を切り出す
        st8.images[nm] = im[cs[1]:cs[3], cs[0]:cs[2]]
    return RunResult(img, ov, painted, ex, crop, st8)


def run(plan_bytes: bytes, plan_idx: int, floor_bytes: bytes, floor_idx: int, p: Params | None = None) -> RunResult:
    p = p or Params()
    return render(extract(plan_bytes, plan_idx, floor_bytes, floor_idx, p), p)


def cropped(a: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = crop
    return a[y0:y1, x0:x1]


def to_pdf_bytes(img: np.ndarray, dpi: int, title: str = "平面着色") -> bytes:
    """完成画像を 1 ページの PDF にする（Flate 圧縮・無劣化。用紙寸法は dpi から逆算）。"""
    return pdfio.image_to_pdf_bytes(img, dpi, title)


def plan_zone(plan: pdfio.Page, pdr: list, lp, margin_pt: float) -> pdfio.Rect:
    """床仕上図が無いときの図面部。レイヤーがあれば注記レイヤーを除いた図形の範囲（外壁の線を切らないよう端まで取る）。
    無ければ、灰色（寸法線）と図枠を除いた図形の端点の 0.2〜99.8 パーセンタイル。"""
    drop = lp.drop_set if lp.has_layers else set()
    xs, ys = [], []
    W, H = plan.rect.width, plan.rect.height
    for d in pdr:
        if lp.has_layers and (d.layer or "") in drop:
            continue
        if d.color is None and d.fill is None:
            continue
        if d.rect.width > W * 0.8 and d.rect.height > H * 0.8:
            continue                                   # 図枠
        if not lp.has_layers and d.color is not None and pdfdoc.is_gray(d.color) and d.color != pdfdoc.BLACK:
            continue                                   # 灰色（寸法線）
        if lp.has_layers and max(d.rect.width, d.rect.height) < 20:
            continue                                   # 小さな図形（文字・記号）は範囲の決定に使わない
        xs += [d.rect.x0, d.rect.x1]; ys += [d.rect.y0, d.rect.y1]
    if len(xs) < 10:
        return plan.rect
    xa, ya = np.array(xs), np.array(ys)
    lo, hi = (0.02, 99.98) if lp.has_layers else (0.2, 99.8)
    x0, x1 = np.percentile(xa, lo), np.percentile(xa, hi)
    y0, y1 = np.percentile(ya, lo), np.percentile(ya, hi)
    return pdfio.Rect(x0 - margin_pt, y0 - margin_pt, x1 + margin_pt, y1 + margin_pt) & plan.rect


def extract_plan(plan_bytes: bytes, plan_idx: int, p: Params, instruction: tuple[bytes, str, int] | None = None,
                 zone_override: tuple[float, float, float, float] | None = None) -> Extraction:
    """床仕上図を使わない抽出（指示書モード／手動モード）。instruction は (bytes, ファイル名, ページ)。"""
    from floortint.debugview import StageLog, label_image, overlay_mask, shrink, side_by_side
    stages: list[StageLog] = []
    t = {}; t0 = time.time()
    pdoc = pdfio.open(stream=plan_bytes, filetype="pdf")
    plan = pdoc[plan_idx]
    if plan.rotation:
        plan.remove_rotation()
    scale = p.scale_override or pdfdoc.page_scale(plan)
    mmpt = pdfdoc.mm_per_pt(scale)
    S = p.dpi / 72.0
    pdr = pdfdoc.drawings(plan)
    st1 = StageLog("1 取込", time.time() - t0)
    st1.metrics = {"平面図": pdfdoc.page_title(plan) or "（名称なし）", "床仕上図": "（使わない）",
                   "縮尺": f"1/{scale}" if scale else "不明（1/200 と仮定）", "mm/pt": round(mmpt, 2), "出力 px/pt": round(S, 3),
                   "平面図の図形数": len(pdr)}
    if not scale:
        st1.warn("縮尺が表題欄から読めない。1/200 と仮定して寸法を換算する")
    st1.images = {"平面図（入力）": shrink(_page_rgb(plan, 1.0), 1200)}
    stages.append(st1); t0 = time.time()
    # ---- 2〜5 は無し（位置合わせ・凡例・領域・矢印は床仕上図のもの）
    lp0 = layermod.inventory(pdr, p.clean.layer_overrides)
    zone = pdfio.Rect(*zone_override) if zone_override else plan_zone(plan, pdr, lp0, p.zone_margin_pt)
    lp = layermod.inventory(pdr, p.clean.layer_overrides, zone)
    cl = clean_layout(plan, pdr, zone, S, S / mmpt, p.clean, lp)
    t["clean"] = time.time() - t0
    st6 = StageLog("6 図面整理", cl.stats.get("t_render", t["clean"]))
    st6.metrics = {"仕分けの方法": "レイヤー名" if cl.stats.get("by_layer") else "色・線種の規則（レイヤー無し）",
                   "図形の総数": cl.stats.get("total_drawings"), "残した図形": cl.stats.get("kept_drawings"),
                   "植栽": cl.stats.get("plant_drawings"), "躯体": cl.stats.get("structure_drawings"),
                   "落とした内訳": cl.stats.get("dropped"), "室名候補": cl.stats.get("texts"),
                   "図面部 [pt]": [round(v) for v in zone],
                   "レイヤー": [f"{'落' if li.drop else '残'} {li.name or '（無名）'}: {li.count} 件（{li.reason}）" for li in lp.layers]}
    zx0, zy0, zx1, zy1 = zone
    raw = _page_rgb(plan, 1.5)[int(zy0 * 1.5):int(zy1 * 1.5), int(zx0 * 1.5):int(zx1 * 1.5)]
    cleaned = cl.lineart[int(zy0 * S):int(zy1 * S), int(zx0 * S):int(zx1 * S)]
    st6.images = {"整理前（左）と整理後（右）": side_by_side(raw, cleaned, 1800)}
    stages.append(st6); t0 = time.time()
    st7 = StageLog("7 家具", cl.stats.get("t_furniture", 0.0))
    fcount = cl.stats.get("furniture_components", 0)
    st7.metrics = {"線で囲まれた成分": cl.stats.get("components"), "家具と判定": fcount,
                   "第 2 段（椅子・破線什器）で追加 [px]": cl.stats.get("furniture_added_px_pass2", 0),
                   "第 3 段（椅子の記号の凸包）": f'{cl.stats.get("chair_symbols", 0)} 個 / {cl.stats.get("furniture_added_px_pass3", 0)} px',
                   "家具判定から外した扉の弧": cl.stats.get("door_arcs", 0)}
    if fcount == 0:
        st7.warn("家具が 1 つも見つからない。線幅や切れ目の設定、家具の面積の範囲を見直す")
    sub = (slice(int(zy0 * S), int(zy1 * S)), slice(int(zx0 * S), int(zx1 * S)))
    st7.images = {"囲まれた成分（色分け）": label_image(cl.enclosed[sub], 1400),
                  "家具の足元（赤）": overlay_mask(cleaned, cl.furniture[sub], (255, 0, 0), 0.5, 1400)}
    stages.append(st7); t0 = time.time()
    # ---- 部屋
    H, W = cl.enclosed.shape
    zone_mask = np.zeros((H, W), np.uint8)
    zone_mask[int(zy0 * S):int(zy1 * S) + 1, int(zx0 * S):int(zx1 * S) + 1] = 1
    rs = find_rooms(cl.enclosed, zone_mask, S / mmpt, p.room_min_m2, cl.texts, S)
    segs = [sg for d in pdr if d.color is not None and pdfdoc.touches(d.rect, zone) for sg in d.segments()]
    n_core = mark_core_rooms(rs, segs, S, mmpt)
    n_common = mark_common_rooms(rs, p.common_pattern)
    t["rooms"] = time.time() - t0
    st9 = StageLog("9 部屋", t["rooms"])
    areas = sorted((r.area_m2 for r in rs.rooms), reverse=True)
    st9.metrics = {"部屋の数": len(rs.rooms), "最小面積 [m²]": p.room_min_m2, "面積の上位 [m²]": [round(a, 1) for a in areas[:8]],
                   "部屋の合計 [m²]": round(sum(areas), 1), "室名が付いた部屋": sum(1 for r in rs.rooms if r.name),
                   "コア（EV・階段。既定で塗らない）": n_core, "共用部（室名で判定。既定で塗らない）": n_common,
                   "共用部の室名": [r.name for r in rs.rooms if r.core == "共用"][:20]}
    if not rs.rooms:
        st9.warn("部屋が見つからない。壁の線が切れている（図面部の外に漏れている）か、最小面積が大きすぎる")
    elif areas and areas[0] > 0.6 * sum(areas) and len(areas) > 3:
        st9.notes.append(f"最大の部屋が {areas[0]:.0f}m² と大きい（開放オフィス）。指示書の細い塗りはこの中で「塗った形」になる")
    st9.images = {"部屋（番号つき）": shrink(room_overlay(cl.lineart, rs)[sub], 1600)}
    stages.append(st9); t0 = time.time()
    # ---- 指示書
    ins = None
    if instruction is not None:
        data, fname, pidx = instruction
        ins = read_instruction(data, fname, pidx, plan, S, S / mmpt, p.instr, (H, W), cl.lineart, zone)
        if ins.seed_alpha:
            # 手直し: seeds のキーは面積順に振り直されるので、不透明度はキーでなく色で引き直す
            ins.seed_alpha = {s_.key: ins.seed_alpha.get(s_.key, 1.0) for s_ in ins.seeds}
        t["instruction"] = ins.seconds
        st10 = StageLog("10 指示書", ins.seconds)
        st10.metrics = {"種類": {"vector": "ベクタ（PDF の図形）", "raster": "ラスタ（PDF の画像）", "image": "画像ファイル", "ppt": "PPT（図面画像＋塗り図形）",
                                 "annotated": "図面に描き足した手書き"}.get(ins.kind, ins.kind),
                        "位置合わせ": ins.align, "色の数": len(ins.seeds),
                        "色": [f"{s.key} {s.name} {s.hex}: {s.area_m2:.1f}m² / {s.blobs} 塊" + (f" → {s.legend.product} {s.legend.maker} {s.legend.pattern}" if s.legend is not None else "") for s in ins.seeds],
                        "凡例": [li.to_dict() for li in ins.legend], "矢印": len(ins.arrows), "吹き出し": [c.to_dict() for c in ins.callouts], "診断": ins.diag}
        for w in ins.warnings:
            st10.warn(w)
        if not ins.seeds:
            st10.warn("指示書から色の塊が取れなかった")
        for k, im in ins.images.items():
            st10.images[k] = shrink(im, 1600)
        if ins.seeds:
            seed_img = cl.lineart.copy()
            for s in ins.seeds:
                m = s.mask > 0
                seed_img[m] = (seed_img[m].astype(np.float32) * 0.4 + np.array(s.rgb, np.float32) * 0.6).astype(np.uint8)
            st10.images["色の塊を平面図に重ねたもの"] = shrink(seed_img[sub], 1600)
        # 吹き出し（手書きの文字・写真）: 引出線の先の部屋に紐づける（部屋の外なら最も近い部屋、半径 2m まで）
        if ins.callouts and rs is not None and rs.room_map is not None:
            for co in ins.callouts:
                if co.target is None:
                    continue
                x, y = int(round(co.target[0])), int(round(co.target[1]))
                if not (0 <= y < rs.room_map.shape[0] and 0 <= x < rs.room_map.shape[1]):
                    continue
                rid = int(rs.room_map[y, x])
                if rid == 0:
                    best = None
                    for r_ in rs.rooms:
                        d_ = math.hypot(r_.centroid[0] - x, r_.centroid[1] - y)
                        if d_ < 2000 * S / mmpt and (best is None or d_ < best[0]):
                            best = (d_, r_.id)
                    rid = best[1] if best else 0
                co.room_id = rid or None
            st10.metrics["吹き出し"] = [c.to_dict() for c in ins.callouts]
        stages.append(st10)
    al = pdfdoc.Alignment(0.0, 0.0, "none")
    ex = Extraction((plan.rect.width, plan.rect.height), scale, mmpt, al, [], [], [], tuple(zone), cl, S, {}, t)
    ex.stages = stages
    ex.layer_plan = lp
    ex.mode = "instruction" if instruction is not None else "manual"
    ex.rooms = rs
    ex.instruction = ins
    pdoc.close()                                   # 図形の辞書（最大で 1GB 近く）を手放す
    return ex
