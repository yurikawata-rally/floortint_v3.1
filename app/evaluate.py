"""指示書モード／手動モードの参照案件を処理し、デザイナーの着色結果（JPG）と比べる。

  python evaluate.py              # 全案件
  python evaluate.py nittetsu7f   # 1 案件

出力: out/instr/<案件>/ に result.png（自動）、designer.png（位置を合わせたデザイナー出力）、compare.png（並べたもの）、
      summary.json。out/instr_eval.json に指標をまとめる。
指標（デザイナー出力は正解ではなく「人の判断の一例」なので、一致率は目安）:
  painted_iou        塗った範囲の IoU（デザイナーが塗った画素 vs 自動で塗った画素。図面部の中）
  furniture_agree    家具の白抜きの一致率（デザイナーが白く残した画素のうち、自動も白く残した割合）と、その逆
  room_paint_agree   部屋ごとに「塗った／塗らない」が一致した割合
  room_group_purity  部屋ごとの材種の分け方の一致（自動の材種ごとに、デザイナーの色群の最頻値が占める割合）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from floortint import pipeline  # noqa: E402
from floortint.instruct import _drawing_roi, _template_align  # noqa: E402
from floortint.legend import Material  # noqa: E402
from floortint.pipeline import Params  # noqa: E402
from run_cases import CASE_PARAMS2, CASES2, CASES2_DESIGNER, LEGEND_TRUTH, OUT, instr_cases_available  # noqa: E402

PAL = ["#b8b5b0", "#c9a97c", "#a9adb3", "#d5b98f", "#9c9a96", "#cfb08a", "#b3b8b0", "#c4bfb6"]


def designer_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """デザイナー出力 → (線, 塗った床, 白)。線は暗い画素、白は明るく彩度の低い画素、塗りはその他で彩度か中間の明るさ。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    ink = v < 110
    white = (v >= 238) & (s < 25)
    painted = ~ink & ~white
    return ink, painted, white


def align_designer(des: np.ndarray, lineart: np.ndarray, S: float, zone: tuple[float, float, float, float]) -> tuple[np.ndarray, dict]:
    """デザイナー画像を自動結果（線画）の座標に合わせて変形する。
    図面部分（細い線が密な範囲）を切り出し、その幅が平面図の図面部の幅に合う倍率を中心に ±20% を探す。"""
    r = 1.0                                           # 作業解像度 1 px/pt
    f = r / S
    plan_ink = cv2.resize(lineart.min(axis=2), None, fx=f, fy=f, interpolation=cv2.INTER_AREA) < 200
    g = min(1.0, 2400 / des.shape[1])
    small = cv2.resize(des, None, fx=g, fy=g, interpolation=cv2.INTER_AREA)
    ink, _, _ = designer_masks(small)
    zone_w = (zone[2] - zone[0]) * r
    cands = []
    # (a) 線が密な範囲を図面部分とみなし、幅が合う倍率の近くを探す
    roi = _drawing_roi(ink)
    s0 = zone_w / max(1, roi[2] - roi[0])
    cands.append(_template_align(ink, plan_ink, roi, s0 * np.exp(np.linspace(np.log(0.8), np.log(1.25), 19)), work=0.5))
    # (b) 画像全体を、図面部が画像の 40〜100% を占めると仮定して広く探す
    full = (0, 0, ink.shape[1], ink.shape[0])
    s1 = zone_w / ink.shape[1]
    cands.append(_template_align(ink, plan_ink, full, s1 * np.exp(np.linspace(np.log(1.0), np.log(2.6), 25)), work=0.5))
    M, sc, corr = max(cands, key=lambda c: c[2])
    if M is None:
        return None, {"corr": 0.0}
    MS = (M * (S / r)).astype(np.float32)
    H, W = lineart.shape[:2]
    warped = cv2.warpAffine(small, MS, (W, H), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return warped, {"corr": round(corr, 3), "scale": round(sc * g, 4), "roi": [int(v) for v in roi]}


def legend_accuracy(found: list[str], truth: list[str]) -> dict:
    from floortint.library import norm_code
    f = [norm_code(x) for x in found if x]; t = [norm_code(x) for x in truth]
    tp = sum(1 for x in set(t) if x in f)
    return {"found": len(f), "truth": len(t), "recall": round(tp / max(1, len(set(t))), 3), "precision": round(sum(1 for x in set(f) if x in t) / max(1, len(set(f))), 3)}


def compare_designer(name: str, img: np.ndarray, res, ex, des_path, d: Path, row: dict) -> None:
    des = cv2.cvtColor(cv2.imread(str(des_path)), cv2.COLOR_BGR2RGB) if des_path and Path(des_path).exists() else None
    if des is None:
        return
    # 位置合わせには躯体（別描画）を含む線画を使う（pypdfium2 化で躯体レイヤーが線画から分かれたため。壁が無いと相関が落ちる）
    ink = getattr(ex.clean, "ink", None)
    base = ex.clean.lineart if ink is None else np.where((ink > 0)[:, :, None], 0, ex.clean.lineart).astype(np.uint8)
    warped, al = align_designer(des, base, ex.S, ex.zone)
    row["designer_align"] = al
    if warped is None:
        return
    x0, y0, x1, y1 = res.crop
    dz = warped[y0:y1, x0:x1]
    cv2.imwrite(str(d / "designer.png"), cv2.cvtColor(dz, cv2.COLOR_RGB2BGR))
    _, d_paint, d_white = designer_masks(dz)
    a_paint = res.painted[y0:y1, x0:x1] > 0
    a_furn = ex.clean.furniture[y0:y1, x0:x1] > 0
    if ex.rooms is not None and ex.rooms.room_map is not None:
        rm = ex.rooms.room_map[y0:y1, x0:x1]; inside = rm > 0
    else:
        zm = np.zeros(res.painted.shape, bool); zx0, zy0, zx1, zy1 = [int(v * ex.S) for v in ex.zone]; zm[zy0:zy1, zx0:zx1] = True
        inside = zm[y0:y1, x0:x1]; rm = None
    inter = (a_paint & d_paint & inside).sum(); union = ((a_paint | d_paint) & inside).sum()
    row["painted_iou"] = round(float(inter / max(1, union)), 3)
    row["painted_area_m2"] = {"auto": round(float(a_paint.sum()) / (1000 * ex.S / ex.mm_per_pt) ** 2, 1),
                              "designer": round(float((d_paint & inside).sum()) / (1000 * ex.S / ex.mm_per_pt) ** 2, 1)}
    dw = d_white & inside; af = a_furn & inside
    row["furniture_agree"] = {"designer_white_covered_by_auto": round(float((dw & (a_furn | ~a_paint)).sum() / max(1, dw.sum())), 3),
                              "auto_furniture_white_in_designer": round(float((af & d_white).sum() / max(1, af.sum())), 3)}
    if rm is not None:
        agree = 0; n = 0; groups = {}
        for r in ex.rooms.rooms:
            m = rm == r.id
            if m.sum() < 50:
                continue
            dp = float((d_paint & m).sum() / m.sum()); ap = float((a_paint & m).sum() / m.sum())
            n += 1; agree += int((dp >= 0.4) == (ap >= 0.4))
            if dp >= 0.4 and ap >= 0.4 and r.material_key:
                groups.setdefault(r.material_key, []).append(dz[m & d_paint].mean(axis=0))
        row["room_paint_agree"] = round(agree / max(1, n), 3); row["rooms_compared"] = n
        cols = [c for v in groups.values() for c in v]
        if len(cols) >= 2:
            lab = cv2.cvtColor(np.array(cols, np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            k = min(6, len(cols))
            _, labels, _ = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 5, cv2.KMEANS_PP_CENTERS)
            labels = labels.ravel().tolist(); i = 0; pure = 0
            for key, v in groups.items():
                ls = labels[i:i + len(v)]; i += len(v)
                pure += max(ls.count(x) for x in set(ls))
            row["room_group_purity"] = round(pure / len(cols), 3)
    h = 900
    a = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)); b = cv2.resize(dz, (int(dz.shape[1] * h / dz.shape[0]), h))
    cv2.imwrite(str(d / "compare.png"), cv2.cvtColor(np.hstack([a, np.full((h, 20, 3), 255, np.uint8), b]), cv2.COLOR_RGB2BGR))


def eval_floor(names: list[str]) -> dict:
    """床仕上図モードの新案件をデザイナー出力と比べる。"""
    out = {}
    for name in names:
        plan, pi, floor, fi = CASES2[name]
        p = Params(dpi=200); p.comp.paint_existing = True
        t0 = time.time()
        res = pipeline.run(plan.read_bytes(), pi, floor.read_bytes(), fi, p)
        ex = res.extraction
        d = OUT / "instr" / name; d.mkdir(parents=True, exist_ok=True)
        img = pipeline.cropped(res.image, res.crop)
        cv2.imwrite(str(d / "result.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(d / "overlay.png"), cv2.cvtColor(pipeline.cropped(res.overlay, res.crop), cv2.COLOR_RGB2BGR))
        row = {"mode": "floor", "elapsed_s": round(time.time() - t0, 1), "scale": ex.scale_denom, "align": ex.align.method, "materials": len(ex.materials),
               "regions": len(ex.regions), "arrows": len(ex.arrows), "furniture": ex.clean.stats.get("furniture_components"),
               "products": [m.product for m in ex.materials]}
        compare_designer(name, img, res, ex, CASES2_DESIGNER.get(name), d, row)
        (d / "summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        out[name] = row
        print(name, json.dumps(row, ensure_ascii=False, default=str))
    return out


def main(argv):
    avail = instr_cases_available()
    names = [a for a in argv if a in avail or a in CASES2] or list(avail)
    allres = {}
    fl = [n for n in names if n in CASES2]
    if fl:
        allres.update(eval_floor(fl))
    for name in [n for n in names if n in avail]:
        plan, pi, ins, ii, des_path = avail[name]
        p = Params(dpi=200)
        for k, v in CASE_PARAMS2.get(name, {}).items():
            setattr(p, k, v)
        t0 = time.time()
        ex = pipeline.extract_plan(plan.read_bytes(), pi, p, (ins.read_bytes(), ins.name, ii) if ins is not None else None)
        t_ex = time.time() - t0
        seeds = ex.instruction.seeds if ex.instruction is not None else []
        p.materials = [Material(s.key, None, kind=(s.legend.kind if s.legend is not None else "carpet"), product=(s.legend.product if s.legend is not None and s.legend.product else s.name),
                                size_mm=(500, 500), pattern=(s.legend.pattern if s.legend is not None else "流し")) for s in seeds]
        for i, m in enumerate(p.materials):
            p.comp.colors[m.key] = PAL[i % len(PAL)]
        p.seed_materials = {s.key: s.key for s in seeds}
        p.materials.append(Material("default", None, kind="carpet", product="既定", size_mm=(500, 500)))
        p.comp.colors["default"] = "#e3e1dd"
        p.default_material = "default"
        t0 = time.time()
        res = pipeline.render(ex, p)
        t_rd = time.time() - t0
        d = OUT / "instr" / name; d.mkdir(parents=True, exist_ok=True)
        img = pipeline.cropped(res.image, res.crop)
        cv2.imwrite(str(d / "result.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(d / "overlay.png"), cv2.cvtColor(pipeline.cropped(res.overlay, res.crop), cv2.COLOR_RGB2BGR))
        for st in ex.stages:
            for nm, im in st.images.items():
                cv2.imwrite(str(d / f"stage{st.name[:2].strip()}_{nm[:14]}.png"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        ins_ = ex.instruction
        row = {"mode": "instruction" if ins is not None else "manual", "extract_s": round(t_ex, 1), "render_s": round(t_rd, 1), "rooms": len(ex.rooms.rooms), "seeds": len(seeds),
               "furniture": ex.clean.stats.get("furniture_components"), "by_layer": ex.clean.stats.get("by_layer"), "scale": ex.scale_denom,
               "paints": len(ex.paints), "kind": ins_.kind if ins_ is not None else None, "instruction": ins_.align if ins_ is not None else None,
               "core_rooms": sum(1 for r in ex.rooms.rooms if r.core in ("EV", "階段")), "common_rooms": sum(1 for r in ex.rooms.rooms if r.core == "共用"),
               "legend": [li.product for li in ins_.legend] if ins_ is not None else [], "arrows": len(ins_.arrows) if ins_ is not None else 0,
               "callouts": len(ins_.callouts) if ins_ is not None else 0,
               "seeds_with_legend": sum(1 for s_ in seeds if s_.legend is not None), "warnings": ins_.warnings if ins_ is not None else []}
        if name in LEGEND_TRUTH and ins_ is not None:
            row["legend_accuracy"] = legend_accuracy([li.product for li in ins_.legend], LEGEND_TRUTH[name])
        # ---- デザイナー出力との比較
        compare_designer(name, img, res, ex, des_path, d, row)
        (d / "summary.json").write_text(json.dumps({**row, "paints": [{"seed": q.seed, "room": q.room_id, "mode": q.mode, "m2": round(q.area_m2, 1)} for q in ex.paints]},
                                                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        allres[name] = row
        print(name, json.dumps({k: v for k, v in row.items() if k not in ("instruction",)}, ensure_ascii=False, default=str))
    prev = {}
    if (OUT / "instr_eval.json").exists():
        try:
            prev = json.loads((OUT / "instr_eval.json").read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}
    prev.update(allres)
    (OUT / "instr_eval.json").write_text(json.dumps(prev, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
