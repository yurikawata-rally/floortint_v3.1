"""参照 4 案件を一括処理し、out/<案件>/ に完成画像・検証オーバーレイ・抽出結果 JSON を書く。

  python run_cases.py            # 4 案件
  python run_cases.py fec7f      # 1 案件だけ
  python run_cases.py --dpi 150  # 解像度を下げて速く
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from floortint import pipeline, quality  # noqa: E402
from floortint.pipeline import Params  # noqa: E402

import os  # noqa: E402

# サンプル案件の置き場。環境変数 FLOORTINT_SAMPLES で差し替えられる（配布物ではサンプルを同梱しない）
DATA = Path(os.environ.get("FLOORTINT_SAMPLES", "/workspace/data/20260916_平面着色作成"))
CASES = {
    "fec7f":   (DATA / "260414_ファーイーストカンパニー様 7F.pdf", 0, DATA / "260421_ファーイーストカンパニー様_1F7F床.pdf", 1),
    "kmj8f":   (DATA / "231110_8Fレイアウト.pdf", 0, DATA / "231218_8F_竣工図書.pdf", 3),
    "jera19f": (DATA / "230731_JERA_JP19F-平面図.pdf", 0, DATA / "230831_竣工図書_JP19F.pdf", 2),
    "asahi8f": (DATA / "260916_旭化成建材様 本社改装_基本設計図書.pdf", 6, DATA / "260916_旭化成建材様 本社改装_基本設計図書.pdf", 13),
}
OUT = Path(os.environ.get("FLOORTINT_OUT", str(Path(__file__).parent / "out")))

# 指示書モードの参照案件（2026-10 受領分）。(平面図, ページ, 指示書, ページ, デザイナーの着色結果) 指示書が None なら手動モード
DATA2 = Path(os.environ.get("FLOORTINT_SAMPLES2", "/workspace/data/20261007_平面着色作成"))
_NT = DATA2 / "日鉄興和不動産ファシリティーズ"
_IS = DATA2 / "ispace(敗戦物件)"
INSTR_CASES = {
    "nittetsu7f": (_NT / "260824_7F_日鉄興和不動産ファシリティーズ様-平面図 1.pdf", 0, _NT / "260813_7F指示書.pdf", 0,
                   _NT / "260820_日鉄興和不動産ファシリティーズ様7F_平面着色.jpg"),
    "nittetsu6f": (_NT / "260824_6F_日鉄興和不動産ファシリティーズ様-平面図 1.pdf", 0, _NT / "260813_6F指示書.pdf", 0,
                   _NT / "260821_日鉄興和不動産ファシリティーズ様6F②_平面着色.jpg"),
    "ispace6f":   (_IS / "241016_ispace本社移転_6F-平面図.pdf", 0, _IS / "241016_ispace_平面着色依頼.pdf", 2, _IS / "241016_ispace本社移転_6F_A.jpg"),
    "ispace7f":   (_IS / "241015_ispace本社移転_7F-平面図.pdf", 0, _IS / "241015_ispace_7F平面着色資料.pdf", 0, _IS / "241017_ispace本社移転_7F のコピー.jpg"),
    "teian4f":    (DATA2 / "提案図04階" / "250701_提案図04階平面図-平面図 (2.0).pdf", 0, None, 0, DATA2 / "提案図04階" / "250702_提案図04階平面着色 (3.0).jpg"),
}
INSTR_NAMES = {"nittetsu7f": "日鉄 7F（ベクタ指示書・1/200）", "nittetsu6f": "日鉄 6F（ベクタ指示書・1/200）",
               "ispace6f": "ispace 6F（画像の指示書・1/150）", "ispace7f": "ispace 7F（画像の指示書・1/150）", "teian4f": "提案図 4F（指示なし・手動）"}

# 2026-10-14 受領分（PPT 指示書・iPad 手書き・床仕上図）。(平面図, ページ, 指示書(PDF/PPTX), ページ(スライド), デザイナーの着色結果)
DATA3 = Path(os.environ.get("FLOORTINT_SAMPLES3", "/workspace/data/20261014_平面着色作成_追加"))
_GC = DATA3 / "GC13F Shallwe"; _CR = DATA3 / "クレスコ札幌(3フロア)"; _MI = DATA3 / "マイクロン(4フロア)"; _NH = DATA3 / "日本住宅ローン"; _SM = DATA3 / "SMC"
_MT = DATA3 / "MTG大阪"; _AO = DATA3 / "青山学院未来共創センター"
INSTR_CASES2 = {
    "gc13f":     (_GC / "260721_GC13F_基本設計図書-平面着色用.pdf", 0, _GC / "260721_GC13F_基本設計図書-平面着色指示.pdf", 0, _GC / "260824_GC13F_基本設計図書-平面着色用-.png"),
    "cresco_A":  (_CR / "250312_CRESCO札幌_レイアウト図_A,B,C案.pdf", 0, _CR / "250311_3案着色依頼.pptx", 0, _CR / "250312_CRESCO札幌_レイアウト図_A案.jpg"),
    "cresco_B":  (_CR / "250312_CRESCO札幌_レイアウト図_A,B,C案.pdf", 1, _CR / "250311_3案着色依頼.pptx", 1, _CR / "250312_CRESCO札幌_レイアウト図_B案.jpg"),
    "cresco_C":  (_CR / "250312_CRESCO札幌_レイアウト図_A,B,C案.pdf", 2, _CR / "250311_3案着色依頼.pptx", 2, _CR / "250312_CRESCO札幌_レイアウト図_C案.jpg"),
    "micron_1F": (_MI / "260825_PlanB_マイクロンメモリジャパン新西条 1.pdf", 0, _MI / "マイクロン　床着色.pptx", 0, _MI / "260825_PlanB_マイクロンメモリジャパン新西条-1F.jpg"),
    "micron_2F": (_MI / "260825_PlanB_マイクロンメモリジャパン新西条 1.pdf", 1, _MI / "マイクロン　床着色.pptx", 0, _MI / "260824_PlanB_マイクロンメモリジャパン新西条-2F.jpg"),
    "micron_3F": (_MI / "260825_PlanB_マイクロンメモリジャパン新西条 1.pdf", 2, _MI / "マイクロン　床着色.pptx", 0, _MI / "260824_PlanB_マイクロンメモリジャパン新西条-3F.jpg"),
    "micron_4F": (_MI / "260825_PlanB_マイクロンメモリジャパン新西条 1.pdf", 3, _MI / "マイクロン　床着色.pptx", 0, _MI / "260825_PlanB_マイクロンメモリジャパン新西条-4F.jpg"),
    "nhl_10y":   (_NH / "250528_日本住宅ローン21F移転_3PLAN.pdf", 0, _NH / "250604_日本住宅ローン本社移転_着色依頼.pptx", 0, _NH / "250609_日本住宅ローン21F移転-10年後レイアウト.jpg"),
    "nhl_A":     (_NH / "250528_日本住宅ローン21F移転_3PLAN.pdf", 1, _NH / "250604_日本住宅ローン本社移転_着色依頼.pptx", 1, None),
    "smc":       (_SM / "点在プラン.pdf", 0, _SM / "SMC_床着色.pptx", 1, _SM / "260323_点在プラン_平面着色.jpg"),
    "aoyama3f":  (_AO / "260916_青山学院大学_未来共創センター_2027_基本設計図書-平面図.pdf", 0, None, 0, _AO / "250924_青山学院大学_未来共創センター_2027-平面図 (1).jpg"),
}
INSTR_NAMES2 = {"gc13f": "GC 13F（iPad 手書き・1/200）", "cresco_A": "クレスコ札幌 A案（PPT・1/150）", "cresco_B": "クレスコ札幌 B案（PPT・1/150）",
                "cresco_C": "クレスコ札幌 C案（PPT・1/150）", "micron_1F": "マイクロン 1F（PPT 1 枚に 4 階・1/150）", "micron_2F": "マイクロン 2F（PPT・1/150）",
                "micron_3F": "マイクロン 3F（PPT・1/150）", "micron_4F": "マイクロン 4F（PPT・1/150）", "nhl_10y": "日本住宅ローン 10年後（PPT・1/250）",
                "nhl_A": "日本住宅ローン 開設A（PPT・1/250）", "smc": "SMC 点在プラン（PPT・縮尺不明→手動）", "aoyama3f": "青山学院 3F（指示なし・手動・1/100）"}
# 案件ごとの既定パラメータ（縮尺の手動指定など）
CASE_PARAMS2 = {"smc": {"scale_override": 150}}   # SMC: 表題欄なし。椅子の丸（約 9pt）と扉の弧（約 15pt）から 1/150 と推定
# 床仕上図モードの追加案件（平面図, ページ, 床仕上図, ページ）とデザイナー出力
CASES2 = {
    "gc13f_floor": (_GC / "260721_GC13F_基本設計図書-平面着色用.pdf", 0, _GC / "260721_GC13F_基本設計図書-床仕上図.pdf", 0),
    "mtg1f":       (_MT / "250808_MTG大阪_レイアウトプラン.pdf", 0, _MT / "250807_2_MTG_C工事 2.pdf", 0),
}
CASES2_DESIGNER = {"gc13f_floor": _GC / "260824_GC13F_基本設計図書-平面着色用-.png", "mtg1f": _MT / "250826_MTG様_入居工事_着色 1.png"}
# PPT 凡例の品番（正解。スライドの文字から人が読んだもの）
LEGEND_TRUTH = {
    "cresco_A": ["GA-144", "RE5012-25CPT", "GA1414W", "LF-5116", "RE7432-50CPT", "RE0082-50CPT", "TZ05-584"],
    "smc": ["NTH-71403", "GA10723T", "TZ12-724", "RE5551-15CPT", "AB1530-1", "AB1720-1", "OT-5345", "AA238802", "4010-7005", "NTH-71403"],
    "nhl_A": ["DT78104", "AB1510-5", "CB1240-1", "OT5324", "NTH70303", "OT5349", "DT7252", "DT75504", "TTN3604"],
    "micron_1F": ["RE0092-50CPT", "RE7405-50CPT", "GX7654", "CB1240-1", "RE5041-15CPT", "NTH-71403", "RE7403-50CPT", "RE7622-50CPT", "NTH-71603", "CB1240-2", "IS-2067"],
}
INSTR_CASES_ALL = {**INSTR_CASES, **INSTR_CASES2}
INSTR_NAMES_ALL = {**INSTR_NAMES, **INSTR_NAMES2}


def samples_available() -> bool:
    return all(plan.exists() and floor.exists() for plan, _, floor, _ in CASES.values())


def instr_samples_available() -> bool:
    return all(plan.exists() and (ins is None or ins.exists()) for plan, _, ins, _, _ in INSTR_CASES.values())


def instr_cases_available() -> dict:
    """存在する指示書モード案件だけ（旧＋新）。"""
    return {k: v for k, v in INSTR_CASES_ALL.items() if v[0].exists() and (v[2] is None or v[2].exists())}


def floor_cases_available() -> dict:
    return {k: v for k, v in {**CASES, **CASES2}.items() if v[0].exists() and v[2].exists()}


FORBIDDEN_IMPORTS = ("pymupdf", "fitz")      # AGPL のライブラリ。配布物に混ざっていないことを --check-deps で確かめる


def check_deps(root: Path | None = None) -> list[str]:
    """floortint/ と app.py に禁止ライブラリの import が無いか。見つかった行を返す（空なら合格）。"""
    import re
    root = root or Path(__file__).parent
    bad = []
    for f in sorted(list((root / "floortint").glob("*.py")) + [root / "app.py", root / "run_cases.py", root / "evaluate.py"]):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
            if re.match(r"\s*(import|from)\s+(" + "|".join(FORBIDDEN_IMPORTS) + r")\b", line):
                bad.append(f"{f.relative_to(root)}:{i}: {line.strip()}")
    return bad


def main(argv: list[str]) -> None:
    if "--check-deps" in argv:
        bad = check_deps()
        print("禁止ライブラリの import: " + ("なし（合格）" if not bad else "\n".join(bad)))
        sys.exit(1 if bad else 0)
    if not samples_available():
        print(f"サンプル案件が見つかりません: {DATA}（環境変数 FLOORTINT_SAMPLES で場所を指定）"); return
    dpi = 300
    if "--dpi" in argv:
        dpi = int(argv[argv.index("--dpi") + 1]); argv = [a for a in argv if a != "--dpi" and a != str(dpi)]
    allc = {**CASES, **CASES2}
    names = [a for a in argv if a in allc] or list(CASES)
    if "--all" in argv:
        names = list(floor_cases_available())
    rows = []
    qual = {}
    for name in names:
        plan, pi, floor, fi = allc[name]
        p = Params(dpi=dpi)
        p.comp.paint_existing = True
        t0 = time.time()
        res = pipeline.run(plan.read_bytes(), pi, floor.read_bytes(), fi, p)
        dt = time.time() - t0
        d = OUT / name; d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "result.png"), cv2.cvtColor(pipeline.cropped(res.image, res.crop), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(d / "overlay.png"), cv2.cvtColor(pipeline.cropped(res.overlay, res.crop), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(d / "furniture.png"), pipeline.cropped(res.extraction.clean.furniture, res.crop))
        s = res.summary(); s["elapsed_s"] = round(dt, 1)
        (d / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        q = quality.report(res.extraction, res); q["elapsed_s"] = round(dt, 1); qual[name] = q
        ex = res.extraction
        area = sum(r.area_pt2 for r in ex.regions) * (ex.mm_per_pt / 1000) ** 2
        rows.append((name, ex.scale_denom, ex.align.method, len(ex.materials), len(ex.regions), round(area),
                     len(ex.arrows), s["arrows_assigned"], ex.clean.stats.get("furniture_components"), round(dt, 1)))
        print(f"{name:<8} 縮尺1/{ex.scale_denom} 位置合わせ={ex.align.method}({ex.align.dx:+.1f},{ex.align.dy:+.1f}) "
              f"材種{len(ex.materials)} 領域{len(ex.regions)} 面積{area:,.0f}m² 矢印{len(ex.arrows)}→{s['arrows_assigned']} "
              f"家具{ex.clean.stats.get('furniture_components')} {dt:.1f}s")
    (OUT / "cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "quality.json").write_text(json.dumps(qual, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, q in qual.items():
        print(f"  quality {name:<8} recall={q['hatch_recall_mean']} prec={q['hatch_precision_mean']} 輪郭中央値={q['boundary_median_mm']}mm "
              f"95%={q['boundary_p95_mm']}mm 小成分={q['small_components']} 家具比={q['furniture_ratio']} 合計={q['total_s']}s")


if __name__ == "__main__":
    main(sys.argv[1:])
