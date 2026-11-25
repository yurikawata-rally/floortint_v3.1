"""FloorTint — 平面図から平面着色を自動生成する Streamlit アプリ。

起動: streamlit run app.py --server.port 8000
左: 入力（床仕上図／指示書／指示なし の 3 モード、サンプル案件 or PDF アップロード）、レイヤー、調整パラメータ
右: 着色結果 / 検証 / 材種と割り当て / 床材ライブラリ / 処理の内訳（デバッグ）/ 書き出し
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from floortint import pdfio
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from floortint import layers as layermod  # noqa: E402
from floortint import library, pdfdoc, pipeline  # noqa: E402
from floortint.compose import material_color  # noqa: E402
from floortint.legend import PATTERNS, Material  # noqa: E402
from floortint.pipeline import Params  # noqa: E402
from floortint.textures import TextureParams, swatch  # noqa: E402
from run_cases import CASES, CASES2, INSTR_NAMES_ALL, instr_cases_available, samples_available  # noqa: E402

try:
    from streamlit_image_coordinates import streamlit_image_coordinates as _img_click
except Exception:  # noqa: BLE001
    _img_click = None

st.set_page_config(page_title="平面着色 FloorTint", page_icon="🎨", layout="wide")
st.markdown("<style>section[data-testid='stSidebar'] {min-width: 400px;} .block-container {padding-top: 3.4rem;}</style>",
            unsafe_allow_html=True)

KINDS = {"carpet": "タイルカーペット", "vinyl": "塩ビタイル", "wood": "フローリング", "tile": "タイル", "other": "その他"}
MODES = ["床仕上図から（自動）", "指示書から（手書き・塗り）", "指示なし（手動で割り当て）"]


# ---------------------------------------------------------------- 共通
@st.cache_data(show_spinner=False)
def _scan(data: bytes) -> list[dict]:
    doc = pdfio.open(stream=data, filetype="pdf")
    return [{"index": i.index, "label": i.label, "kind": i.kind, "floor": i.floor} for i in pdfdoc.scan(doc)]


@st.cache_data(show_spinner=False)
def _scan_layers(data: bytes, idx: int) -> list[dict]:
    return pipeline.scan_layers(data, idx)


@st.cache_data(show_spinner="PPTX を PDF に変換中…（LibreOffice）", max_entries=4)
def _pptx_pdf(data: bytes) -> bytes:
    from floortint import pptinstr
    return pptinstr.pptx_to_pdf(data)


# 抽出結果は cache_resource（オブジェクトをそのまま保持）。cache_data だと pickle 化した複製（A3 300dpi で 0.5GB）を持ち、
# 再描画のたびに復元コピーを作るのでメモリが 2〜3 倍要った。保持は 2 件まで。
@st.cache_resource(show_spinner="図面を読み取り中…（凡例・領域・矢印・家具）", max_entries=2)
def _extract(plan: bytes, pi: int, floor: bytes, fi: int, key: str):
    p = st.session_state["_params"]
    return pipeline.extract(plan, pi, floor, fi, p)


@st.cache_resource(show_spinner="図面を読み取り中…（部屋・指示書・家具）", max_entries=2)
def _extract_plan(plan: bytes, pi: int, ins: bytes | None, ins_name: str, ins_idx: int, key: str):
    p = st.session_state["_params"]
    return pipeline.extract_plan(plan, pi, p, (ins, ins_name, ins_idx) if ins is not None else None)


def _mem_available_mb() -> int | None:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def _estimate_peak_mb(page_w_pt: float, page_h_pt: float, dpi: int) -> int:
    """画面での 1 回の処理（抽出＋合成＋割り当て図＋表示）のピークメモリの目安 [MB]。
    A3 300dpi（17.4Mpx、部屋 113）: 単体で 2.7GB、画面では 4.2GB の実測 → 約 260 バイト/画素＋固定 600MB。"""
    mpx = page_w_pt * page_h_pt * (dpi / 72.0) ** 2 / 1e6
    return int(600 + mpx * 260)


@st.cache_data(show_spinner=False)
def pymupdf_page_size(data: bytes, idx: int) -> tuple[float, float]:
    pg = pdfio.open(stream=data, filetype="pdf")[idx]
    return float(pg.rect.width), float(pg.rect.height)


def _digest(*parts) -> str:
    h = hashlib.sha1()
    for x in parts:
        h.update(x if isinstance(x, bytes) else json.dumps(x, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def _png_bytes(a: np.ndarray) -> bytes:
    b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG", compress_level=3); return b.getvalue()


def _fact(col, name: str, value: str, note: str = ""):
    """項目名を必ず見える大きさで出す（st.metric はラベルが薄くて読めない）。"""
    col.markdown(
        f"<div style='border:1px solid #d9d9d9;border-radius:8px;padding:8px 10px;background:#fafafa'>"
        f"<div style='font-size:0.8rem;color:#444;font-weight:600'>{name}</div>"
        f"<div style='font-size:1.3rem;line-height:1.3'>{value}</div>"
        f"<div style='font-size:0.75rem;color:#777'>{note}</div></div>", unsafe_allow_html=True)


def _apply_pending_load():
    """保存した物件設定を、ウィジェットを作る前に session_state へ流し込む。"""
    pend = st.session_state.pop("_pending_load", None)
    if not pend:
        return
    for k, v in pend.get("widgets", {}).items():
        st.session_state[k] = v
    for k in ("mat_rows", "seed_mat", "seed_mode", "room_mat", "default_mat", "layer_over", "mat_sig", "room_splits", "rect_covers"):
        if k in pend:
            st.session_state[k] = pend[k]
    st.session_state["_loaded_note"] = f"物件設定「{pend.get('name', '')}」を読み込みました"


_apply_pending_load()

# ---------------------------------------------------------------- 左ペイン: 入力
with st.sidebar:
    st.title("🎨 平面着色 FloorTint")
    st.caption("平面図 → 床に模様を着色。家具は白のまま影付き。")
    mode_label = st.radio("着色の根拠", MODES, key="in_mode",
                          help="床仕上図: 床仕上図のハッチと凡例から全自動。指示書: 手書きマーカーや塗りの色で部屋を塗り分け、色と材種を対応づける。"
                               "指示なし: 部屋ごとに手で材種を割り当てる")
    mode = {MODES[0]: "floor", MODES[1]: "instruction", MODES[2]: "manual"}[mode_label]
    has_samples = samples_available() if mode == "floor" else bool(instr_cases_available())
    src_opts = (["サンプル案件"] if has_samples else []) + ["PDF をアップロード"]
    src = st.radio("入力", src_opts, horizontal=True, key="src")
    plan_bytes = floor_bytes = None
    ins_bytes, ins_name, ins_idx = None, "", 0
    plan_idx = floor_idx = 0
    case = None
    if src == "サンプル案件" and mode == "floor":
        names = {"fec7f": "FEC 7F（1/200・6 材種・曲線境界）", "kmj8f": "コニカ 8F（1/150・乱貼り）",
                 "jera19f": "JERA 19F（1/200・ヘリンボーン/アシュラー）", "asahi8f": "旭化成 8F（1/300・13 材種・図面帳）",
                 "gc13f_floor": "GC 13F（1/200・床仕上図あり）", "mtg1f": "MTG 大阪 1F（1/150・図面帳・回転ページ）"}
        allc = {**CASES, **{k: v for k, v in CASES2.items() if v[0].exists() and v[2].exists()}}
        case = st.selectbox("案件", list(allc), format_func=lambda k: names.get(k, k), key="case")
        plan_path, plan_idx, floor_path, floor_idx = allc[case]
        plan_bytes, floor_bytes = plan_path.read_bytes(), floor_path.read_bytes()
        st.caption(f"平面図: {plan_path.name} p{plan_idx + 1}　／　床仕上図: {floor_path.name} p{floor_idx + 1}")
    elif src == "サンプル案件":
        avail = instr_cases_available()
        keys = [k for k, v in avail.items() if (v[2] is not None) == (mode == "instruction")] or list(avail)
        case = st.selectbox("案件", keys, format_func=lambda k: INSTR_NAMES_ALL.get(k, k), key="case2")
        plan_path, plan_idx, ins_path, ins_idx, _ = avail[case]
        from run_cases import CASE_PARAMS2
        if CASE_PARAMS2.get(case, {}).get("scale_override") and not st.session_state.get("scale_override"):
            st.session_state["scale_override"] = int(CASE_PARAMS2[case]["scale_override"])
        plan_bytes = plan_path.read_bytes()
        if ins_path is not None and mode == "instruction":
            ins_bytes, ins_name = ins_path.read_bytes(), ins_path.name
            if ins_name.lower().endswith(".pptx"):
                ins_bytes, ins_name = _pptx_pdf(ins_bytes), ins_name + ".pdf"
        st.caption(f"平面図: {plan_path.name} p{plan_idx + 1}" + (f"　／　指示書: {ins_path.name} p{ins_idx + 1}" if ins_bytes else ""))
    else:
        up_plan = st.file_uploader("平面図（レイアウト）PDF", type=["pdf"], key="up_plan")
        if mode == "floor":
            up_floor = st.file_uploader("床仕上図 PDF（図面帳でも可。空なら平面図と同じファイル）", type=["pdf"], key="up_floor")
        elif mode == "instruction":
            up_ins = st.file_uploader("指示書（PDF / PPTX / PNG / JPG。PPT の塗り図形、iPad の手書き、マーカーのどれでも）", type=["pdf", "pptx", "png", "jpg", "jpeg"], key="up_ins")
        st.caption(f"アップロード上限 {int(os.environ.get('MAX_UPLOAD_MB', '200'))} MB。ファイルはサーバのメモリ上でだけ処理し、保存しません。")
        if up_plan is not None:
            plan_bytes = up_plan.getvalue()
            ps = _scan(plan_bytes)
            dp = next((x["index"] for x in ps if x["kind"] == "plan"), 0)
            plan_idx = st.selectbox("平面図のページ", [x["index"] for x in ps], index=dp, format_func=lambda i: ps[i]["label"], key="pg_plan")
            if mode == "floor":
                floor_bytes = up_floor.getvalue() if up_floor is not None else plan_bytes
                fs = _scan(floor_bytes)
                pf = ps[dp].get("floor")
                fcands = [x for x in fs if x["kind"] == "floor"]
                same = [x for x in fcands if pf and x.get("floor") == pf]
                df = (same or fcands or [{"index": 0}])[0]["index"]
                floor_idx = st.selectbox("床仕上図のページ", [x["index"] for x in fs], index=df, format_func=lambda i: fs[i]["label"], key="pg_floor")
            elif mode == "instruction":
                if up_ins is None:
                    st.info("指示書を入れてください。無ければ「指示なし」モードで部屋ごとに割り当てられます。")
                else:
                    ins_bytes, ins_name = up_ins.getvalue(), up_ins.name
                    if ins_name.lower().endswith(".pptx"):
                        ins_bytes, ins_name = _pptx_pdf(ins_bytes), ins_name + ".pdf"
                    if ins_name.lower().endswith(".pdf"):
                        n = len(pdfio.open(stream=ins_bytes, filetype="pdf"))
                        ins_idx = st.number_input("指示書のページ", 1, max(1, n), 1, key="pg_ins") - 1

    # ---------------------------------------------------------------- レイヤー
    layer_over: dict[str, bool] = dict(st.session_state.get("layer_over", {}))
    if plan_bytes is not None:
        lay = _scan_layers(plan_bytes, plan_idx)
        with st.expander(f"レイヤー（{len(lay)} 層）" if lay else "レイヤー（無し：色・線種で仕分け）"):
            if lay:
                st.caption("PDF のレイヤー名で、寸法・通り芯・人物・文字などを落とします。チェックを外すと復活、付けると落とします。")
                new_over = {}
                for cat in layermod.CATEGORY_ORDER:
                    rows = [li for li in lay if li["category"] == cat]
                    if not rows:
                        continue
                    st.markdown(f"**{layermod.CATEGORY_LABEL[cat]}**")
                    for li in rows:
                        default = li["category"] in ("drop", "hatch")
                        v = st.checkbox(f"{li['name'] or '（無名）'}　{li['count']:,} 件 ・{li['reason']}",
                                        value=layer_over.get(li["name"], default), key=f"lay_{_digest(plan_bytes)[:6]}_{li['name']}")
                        if v != default:
                            new_over[li["name"]] = v
                layer_over = new_over
                st.session_state["layer_over"] = layer_over
            else:
                st.caption("この PDF にはレイヤー情報がありません。灰色の線・破線・席マーカー・小さな文字を色と形で落とします。")

    # ---------------------------------------------------------------- 調整
    st.markdown("---")
    st.markdown("**着色（既定値で全自動。好みで調節）**")
    tmode = st.radio("モード", ["マテリアル着色", "色着色"], horizontal=True, key="mode",
                     help="マテリアル＝床材の模様（または写真）を貼る。色着色＝色で塗り分けるだけ")
    target_mm = st.slider("模様の見かけの大きさ [mm]", 300, 1200, 600, 50, key="target_mm",
                          help="1 枚の面積の平方根をこの実寸に見えるまで拡大する（長辺は 1200mm まで）")
    fixed_deform = st.toggle("拡大率を固定する", value=False, key="fixed_deform")
    deform = st.slider("固定の拡大率（1 = 縮尺どおり）", 1.0, 8.0, 1.5, 0.5, key="deform") if fixed_deform else 1.5
    joint = st.slider("目地の濃さ", 0.0, 1.0, 0.5, 0.05, key="joint")
    variation = st.slider("1 枚ごとの明暗のばらつき", 0.0, 0.15, 0.06, 0.01, key="variation")
    with st.expander("木目・毛並み（写真を使わない材種）"):
        grain = st.slider("木目の強さ（板材）", 0.0, 1.0, 0.5, 0.05, key="grain")
        fiber = st.slider("毛並みの強さ（カーペット）", 0.0, 1.0, 0.35, 0.05, key="fiber")
        seed = st.number_input("乱数の種（乱貼りの並び）", 0, 999, 7, key="seed")
    st.markdown("**家具と影**")
    shadow_px = st.slider("影の長さ [px]", 0, 30, 10, key="shadow_px")
    shadow_blur = st.slider("影のぼかし", 0.0, 20.0, 6.0, 0.5, key="shadow_blur")
    shadow_alpha = st.slider("影の濃さ", 0.0, 0.8, 0.38, 0.02, key="shadow_alpha")
    with st.expander("家具の判定"):
        f_max = st.slider("家具とみなす面積の上限 [m²]", 1.0, 20.0, 6.0, 0.5, key="f_max")
        f_min = st.slider("面積の下限 [m²]", 0.01, 0.5, 0.04, 0.01, key="f_min")
        close_gaps = st.slider("線の切れ目を埋める幅 [px]", 0, 6, 2, key="close_gaps")
        chair_gap = st.slider("椅子・破線の什器: 埋める切れ目 [mm]", 0, 500, 300, 10, key="chair_gap")
        chair_max = st.slider("第 2 段で追加する面積の上限 [m²]", 0.2, 6.0, 1.0, 0.1, key="chair_max")
    st.markdown("**植栽と躯体**")
    plant_mode = st.selectbox("植栽", ["元の色のまま", "線を指定色にする", "指定色で塗る", "消す"], key="plant_mode",
                              help="レイヤーで植栽が分かれているときに効く（レイヤーの無い PDF では緑の線を植栽とみなす）")
    plant_color = st.color_picker("植栽の色", "#7fb069", key="plant_color")
    plant_alpha = st.slider("植栽の塗りの濃さ", 0.1, 1.0, 0.55, 0.05, key="plant_alpha") if plant_mode == "指定色で塗る" else 0.55
    struct_mode = st.selectbox("躯体（柱・壁）", ["線のまま", "指定色で塗る", "消す"], key="struct_mode",
                               help="躯体レイヤー（STRUCTURE など）があるときに効く")
    struct_color = st.color_picker("躯体の色", "#9a9a9a", key="struct_color")
    st.markdown("**線画と文字**")
    line_alpha = st.slider("線画の濃さ", 0.3, 1.0, 0.85, 0.05, key="line_alpha")
    keep_names = st.toggle("室名・エリア名を残す", value=True, key="keep_names")
    text_scale = st.slider("文字の大きさ", 0.6, 2.0, 1.0, 0.1, key="text_scale")
    if mode == "floor":
        st.markdown("**既存床（凡例の「変更なし」）**")
        paint_existing_opt = st.selectbox("塗り方", ["凡例にあれば塗る", "塗る", "塗らない"], key="paint_existing")
        existing_color = st.color_picker("既存床の色", "#e3e1dd", key="existing_color")
    else:
        paint_existing_opt, existing_color = "塗らない", "#e3e1dd"
    if mode != "floor":
        st.markdown("**塗らない部屋（既定）**")
        paint_core = st.toggle("コア（EV・階段）も塗る", value=False, key="paint_core")
        paint_common = st.toggle("共用部（WC・PS・EPS・機械室・湯沸室など室名で判定）も塗る", value=False, key="paint_common")
        common_pattern = st.text_input("共用部とみなす室名の追加（正規表現、| 区切り）", value="", key="common_pattern", placeholder="例: 倉庫|サーバ")
    else:
        paint_core, paint_common, common_pattern = False, False, ""
    with st.expander("抽出の詳細（通常は変更不要）"):
        dpi = st.select_slider("出力解像度 [dpi]", [100, 150, 200, 300], 300, key="dpi")
        margin = st.slider("図面部の余白 [pt]", 0, 60, 14, key="margin")
        if "scale_override" not in st.session_state:
            st.session_state["scale_override"] = 0
        scale_override = st.number_input("縮尺の手動指定 1/n（0 = 表題欄から自動）", 0, 1000, step=50, key="scale_override",
                                         help="表題欄の文字が読めない図面（アウトライン化・画像）で使う。面積と模様の大きさに効く")
        drop_grid = st.toggle("等間隔の直線群（OA 割付のマス目・タイル割）を落とす", True, key="drop_grid")
        if mode == "floor":
            dx_adj = st.number_input("位置合わせ補正 x [pt]", -50.0, 50.0, 0.0, 0.5, key="dx_adj")
            dy_adj = st.number_input("位置合わせ補正 y [pt]", -50.0, 50.0, 0.0, 0.5, key="dy_adj")
            min_region = st.number_input("領域の最小面積 [m²]", 0.05, 5.0, 0.3, 0.05, key="min_region")
            close_factor = st.slider("ハッチを閉じる核の倍率", 1.0, 3.0, 1.5, 0.1, key="close_factor")
            resolve_ov = st.toggle("材種の重なりはハッチ線に近い方へ", True, key="resolve_ov")
            snap_lines = st.toggle("領域を黒い線（壁・境界線）に沿わせる", True, key="snap_lines")
            barrier_mm = st.slider("境界とみなす黒線の最小長さ [mm]", 100, 1000, 300, 50, key="barrier_mm")
            grow_f = st.slider("線まで広げる距離（ハッチ間隔の倍率）", 0.25, 2.0, 0.75, 0.25, key="grow_f")
        else:
            dx_adj = dy_adj = 0.0; min_region, close_factor, resolve_ov, snap_lines, barrier_mm, grow_f = 0.3, 1.5, True, True, 300, 0.75
            room_min = st.slider("部屋とみなす閉領域の最小面積 [m²]", 0.5, 20.0, 2.0, 0.5, key="room_min")
        if mode == "instruction":
            st.markdown("**指示書の読み取り**")
            sat_min = st.slider("色とみなす彩度の下限（画像）", 10, 150, 40, 5, key="sat_min")
            min_blob = st.slider("無視する塊の面積 [m²]（これ未満）", 0.1, 5.0, 0.6, 0.1, key="min_blob")
            merge_de = st.slider("似た色をまとめる距離", 5.0, 50.0, 22.0, 1.0, key="merge_de")
            min_width = st.slider("マーカーとみなす線幅の下限 [pt]（PDF 図形）", 1.0, 12.0, 4.0, 0.5, key="min_width")
            small_room = st.slider("「小さい部屋」の面積の上限 [m²]", 10.0, 200.0, 60.0, 5.0, key="small_room",
                                   help="小さい部屋は色が少しでも入れば部屋全体。大きい部屋は塗った形・線で区切った区画で塗る")
            room_cover = st.slider("部屋全体にする覆いの割合", 0.3, 0.95, 0.7, 0.05, key="room_cover")
            simplify_mm = st.slider("塗った形の直線化 [mm]", 0, 400, 120, 10, key="simplify_mm")
            st.markdown("位置合わせの手動補正（画像の指示書）")
            m_scale = st.number_input("倍率", 0.5, 2.0, 1.0, 0.01, key="m_scale")
            m_dx = st.number_input("x [pt]", -500.0, 500.0, 0.0, 1.0, key="m_dx")
            m_dy = st.number_input("y [pt]", -500.0, 500.0, 0.0, 1.0, key="m_dy")
        chair_sym = st.toggle("椅子の記号の凸包を家具にする（第 3 段）", True, key="chair_sym")
        chair_rng = st.slider("その記号の大きさ [mm]", 200, 1200, (350, 900), 50, key="chair_rng")
        use_layers = st.toggle("レイヤーで仕分ける（あれば）", True, key="use_layers")
        drop_gray = st.toggle("灰色の線（寸法線）を落とす", True, key="drop_gray")
        drop_dashed = st.toggle("長い破線（通り芯）を落とす", True, key="drop_dashed")
        drop_markers = st.toggle("席マーカー（橙・青の塗り）を落とす", True, key="drop_markers")
        drop_glyph = st.toggle("線で描かれた小さな文字を落とす", True, key="drop_glyph")
        glyph_pt = st.slider("その文字の大きさの上限 [pt]", 3.0, 8.0, 4.5, 0.5, key="glyph_pt")
        drop_ghatch = st.toggle("緑の直線ハッチ（植栽でない印）を落とす", True, key="drop_ghatch")
        excl_doors = st.toggle("扉の開閉弧を家具判定から外す", True, key="excl_doors")

    # ---------------------------------------------------------------- 保存・読込
    st.markdown("---")
    with st.expander("物件の設定を保存・読込"):
        st.caption(f"保存先: {library.lib_dir() / 'projects'}（共有フォルダ。環境変数 FLOORTINT_LIBRARY で変更）")
        proj_name = st.text_input("物件名", value=(case or ""), key="proj_name")
        save_clicked = st.button("この設定を保存", width="stretch", key="proj_save")
        projs = library.list_projects()
        if projs:
            load_name = st.selectbox("保存済みの設定", projs, key="proj_pick")
            c1, c2 = st.columns(2)
            if c1.button("読み込む", width="stretch", key="proj_load"):
                d = library.load_project(load_name)
                if d:
                    st.session_state["_pending_load"] = d
                    st.rerun()
            if c2.button("削除", width="stretch", key="proj_del"):
                library.delete_project(load_name); st.rerun()

if plan_bytes is None:
    st.info("左で案件を選ぶか、平面図 PDF を入れてください。")
    st.stop()
if st.session_state.get("_loaded_note"):
    st.success(st.session_state.pop("_loaded_note"))

# ---------------------------------------------------------------- パラメータ
P = Params(dpi=int(dpi), dx_adjust=float(dx_adj), dy_adjust=float(dy_adj), min_region_m2=float(min_region),
           close_factor=float(close_factor), zone_margin_pt=float(margin))
P.clean.drop_gray, P.clean.drop_dashed, P.clean.drop_markers, P.clean.drop_stroke_text = drop_gray, drop_dashed, drop_markers, drop_glyph
P.clean.keep_room_names = keep_names
P.clean.glyph_pt, P.clean.drop_green_hatch, P.clean.exclude_door_arcs = float(glyph_pt), bool(drop_ghatch), bool(excl_doors)
P.clean.furniture_max_m2, P.clean.furniture_min_m2, P.clean.close_gaps_px = float(f_max), float(f_min), int(close_gaps)
P.clean.chair_gap_mm, P.clean.chair_max_m2 = float(chair_gap), float(chair_max)
P.clean.use_layers, P.clean.layer_overrides = bool(use_layers), dict(layer_over)
P.clean.drop_grid = bool(drop_grid)
P.scale_override = int(scale_override) or None
P.paint_core, P.paint_common, P.common_pattern = bool(paint_core), bool(paint_common), str(common_pattern or "")
P.resolve_overlap, P.snap_to_lines, P.barrier_min_mm, P.grow_factor = bool(resolve_ov), bool(snap_lines), float(barrier_mm), float(grow_f)
P.clean.chair_symbols, P.clean.chair_min_mm, P.clean.chair_max_mm = bool(chair_sym), float(chair_rng[0]), float(chair_rng[1])
if mode != "floor":
    P.room_min_m2 = float(room_min)
if mode == "instruction":
    P.instr.sat_min, P.instr.min_blob_m2, P.instr.merge_de, P.instr.min_width_pt = int(sat_min), float(min_blob), float(merge_de), float(min_width)
    P.instr.small_room_m2, P.instr.room_cover, P.instr.band_simplify_mm = float(small_room), float(room_cover), float(simplify_mm)
    P.instr.manual_scale, P.instr.manual_dx, P.instr.manual_dy = float(m_scale), float(m_dx), float(m_dy)
extract_key = _digest(mode, P.dpi, P.dx_adjust, P.dy_adjust, P.min_region_m2, P.close_factor, P.zone_margin_pt, P.clean.__dict__,
                      P.resolve_overlap, P.snap_to_lines, P.barrier_min_mm, P.grow_factor, P.room_min_m2, P.instr.__dict__,
                      P.scale_override, P.common_pattern)
st.session_state["_params"] = P
try:
    _pw, _ph = pymupdf_page_size(plan_bytes, plan_idx)
except Exception:  # noqa: BLE001
    _pw, _ph = 1191.0, 842.0
_MARGIN_MB = 1500                                   # ホストの他プロセスの揺れ（スワップ無し）を見込む余裕
_need, _avail = _estimate_peak_mb(_pw, _ph, P.dpi), _mem_available_mb()
if _avail is not None and _need > _avail - _MARGIN_MB:
    # まず古い抽出結果のキャッシュを捨てて空きを増やし、それでも足りなければ収まる解像度まで自動で下げる
    import gc
    _extract.clear(); _extract_plan.clear(); gc.collect()
    _avail = _mem_available_mb() or _avail
if _avail is not None and _need > _avail - _MARGIN_MB:
    _orig = P.dpi
    for _d in (300, 200, 150, 100):
        if _d < P.dpi and _estimate_peak_mb(_pw, _ph, _d) <= _avail - _MARGIN_MB:
            P.dpi = _d; break
    if P.dpi != _orig:
        st.warning(f"空きメモリ {_avail / 1000:.1f} GB に対し {_orig} dpi では約 {_need / 1000:.1f} GB 必要なので、今回は {P.dpi} dpi で処理します"
                   f"（処理が落ちて画面が再接続になるのを防ぐため）。他のアプリを止めるか、左ペイン「抽出の詳細」で解像度を選び直してください。")
        extract_key = _digest(extract_key, P.dpi)
    else:
        st.warning(f"この図面を {P.dpi} dpi で処理すると約 {_need / 1000:.1f} GB 必要ですが、空きメモリは {_avail / 1000:.1f} GB しかありません。"
                   f"処理が途中で落ちる（画面が再接続になる）ことがあります。他のアプリを止めてください。")
t0 = time.time()
if mode == "floor":
    ex = _extract(plan_bytes, plan_idx, floor_bytes, floor_idx, extract_key)
else:
    ex = _extract_plan(plan_bytes, plan_idx, ins_bytes, ins_name, int(ins_idx), extract_key)
t_ex = time.time() - t0

# ---------------------------------------------------------------- 見た目のパラメータ
P.comp.mode = "material" if tmode == "マテリアル着色" else "flat"
P.comp.texture = TextureParams(deform=float(deform), target_mm=0.0 if fixed_deform else float(target_mm),
                               joint=float(joint), variation=float(variation), grain=float(grain), fiber=float(fiber), seed=int(seed))
P.comp.shadow_px, P.comp.shadow_blur, P.comp.shadow_alpha = int(shadow_px), float(shadow_blur), float(shadow_alpha)
P.comp.line_alpha, P.comp.text_scale = float(line_alpha), float(text_scale)
P.comp.plant_mode = {"元の色のまま": "original", "線を指定色にする": "line", "指定色で塗る": "fill", "消す": "hide"}[plant_mode]
P.comp.plant_color, P.comp.plant_alpha = plant_color, float(plant_alpha)
P.comp.structure_mode = {"線のまま": "line", "指定色で塗る": "fill", "消す": "hide"}[struct_mode]
P.comp.structure_color = struct_color
P.comp.existing_color = existing_color

# ---------------------------------------------------------------- 上部: 読み取り結果
al = ex.align
st.markdown("#### 図面から自動で読み取った内容")
c1, c2, c3, c4, c5, c6 = st.columns(6)
_fact(c1, "図面の縮尺", f"1/{ex.scale_denom}" if ex.scale_denom else "不明（1/200 と仮定）", "表題欄から読み取り")
_fact(c2, "図面の整理", "レイヤー名で仕分け" if ex.clean.stats.get("by_layer") else "色・線種で仕分け",
      f"{ex.clean.stats.get('kept_drawings', 0):,} / {ex.clean.stats.get('total_drawings', 0):,} 図形を残した")
if mode == "floor":
    mats = ex.materials
    n_dir = sum(1 for r in ex.regions if r.direction is not None)
    _fact(c3, "平面図と床仕上図の位置合わせ",
          {"grid": "通り芯ラベルで一致", "grid-bubble": "通り芯（丸付き）で一致", "phase": "画像の相関で推定"}.get(al.method, al.method),
          f"ずれ ({al.dx:+.1f}, {al.dy:+.1f}) pt")
    _fact(c4, "凡例から読んだ床材", f"{len(mats)} 種", "中止行を含む")
    _fact(c5, "復元した塗り領域", f"{len(ex.regions)} か所", f"貼り方向が決まった領域 {n_dir}")
else:
    rs = ex.rooms
    _fact(c3, "部屋（閉じた領域）", f"{len(rs.rooms)} 室",
          f"コア {sum(1 for r in rs.rooms if r.core in ('EV', '階段'))} 室・共用部 {sum(1 for r in rs.rooms if r.core == '共用')} 室は既定で塗らない")
    if ex.instruction is not None:
        ins = ex.instruction
        a = ins.align
        _fact(c4, "指示書の色", f"{len(ins.seeds)} 色" + (f"／凡例 {len(ins.legend)} 件" if ins.legend else ""),
              {"vector": "PDF の図形", "raster": "PDF の画像", "image": "画像ファイル", "ppt": "PPT（図面画像＋塗り図形）", "annotated": "図面に描き足した手書き"}.get(ins.kind, ""))
        meth = a.get("method", "") or ""
        _fact(c5, "指示書の位置合わせ", {"template": "線画の相関", "orb": "特徴点", "phase": "位相相関", "grid": "通り芯", "grid-bubble": "通り芯",
                                        "orb+template": "図面画像の実データで一致", "template(image)": "図面画像の相関"}.get(meth, "同じ図面（恒等）" if meth.startswith("identity") else meth or "-"),
              f"倍率 {a.get('scale', 1.0)}　相関 {a.get('corr', '-')}" if ins.kind not in ("vector", "annotated") else
              (f"黒線の一致率 {a.get('plan_ink_iou', '-')}" if ins.kind == "annotated" else f"ずれ ({a.get('dx', 0)}, {a.get('dy', 0)}) pt"))
    else:
        _fact(c4, "指示書", "なし", "部屋ごとに手で割り当て")
        _fact(c5, "室名が付いた部屋", f"{sum(1 for r in rs.rooms if r.name)} 室", "図面の文字から")
_fact(c6, "白抜きにする家具", f"{ex.clean.stats.get('furniture_components', 0):,} 個", f"読み取り {t_ex:.1f} 秒")
st.caption("上の 6 項目は自動読み取りの結果。詳しい内訳と途中画像は「処理の内訳（デバッグ）」タブ。")

tab_out, tab_chk, tab_leg, tab_lib, tab_dbg, tab_dl = st.tabs(["🎨 着色結果", "🔍 検証", "📋 材種と割り当て", "🗂 床材ライブラリ",
                                                               "🛠 処理の内訳（デバッグ）", "💾 書き出し"])

# ---------------------------------------------------------------- 床材ライブラリ（先に評価して一覧を得る）
lib_items = library.list_materials()
lib_by_id = {m.id: m for m in lib_items}
with tab_lib:
    st.markdown("#### 床材の写真ライブラリ")
    st.caption(f"製品ごとに実物写真（1 枚 = タイル 1 枚・板 1 枚）を 1〜6 枚登録すると、生成模様の代わりに写真を貼ります。"
               f"指示書（PPT）の凡例にある品番・写真・リンクは読み取り時に自動で蓄積します（品番で裏で紐づけ、最終使用日を更新）。"
               f"保存先: {library.lib_dir() / 'materials'}")
    with st.expander("材種セットの書き出し・読み込み（案件ごとに保存・共有）"):
        st.caption("選んだ材種のメタ情報と写真を 1 つの ZIP にまとめます。別の環境のライブラリに読み込むと同じ品番は飛ばします。廃盤になった製品は「廃盤」を付けると候補から外れます。")
        sel_ids = st.multiselect("書き出す材種", [m.id for m in library.list_materials()], key="exp_ids",
                                 format_func=lambda k: next((f"{m.name}（{m.maker}）" if m.maker else m.name for m in library.list_materials() if m.id == k), k))
        if sel_ids:
            st.download_button("材種セット ZIP をダウンロード", library.export_set(sel_ids, case or "materials"), f"materials_{case or 'set'}.zip", "application/zip")
        imp = st.file_uploader("材種セット ZIP を読み込む", type=["zip"], key="imp_set")
        if imp is not None and st.button("読み込む", key="imp_go"):
            r = library.import_set(imp.getvalue())
            st.success("読み込み: " + "、".join(f"{m.name}（{stt}）" for stt, m in r)); st.rerun()
    with st.form("lib_add", clear_on_submit=True):
        l1, l2, l3, l8 = st.columns([2, 1, 1, 1.2])
        ln = l1.text_input("名前（品番）")
        lk = l2.selectbox("種類", list(KINDS), format_func=lambda k: KINDS[k])
        lp = l3.selectbox("貼り方", PATTERNS, index=5)
        lmk = l8.text_input("メーカー")
        l4, l5, l6, l7 = st.columns(4)
        lw = l4.number_input("寸法 幅 [mm]", 50.0, 3000.0, 500.0, 10.0)
        lh = l5.number_input("寸法 奥行 [mm]", 50.0, 3000.0, 500.0, 10.0)
        lsat = l6.slider("彩度", 0.3, 1.7, 1.0, 0.05)
        lbri = l7.slider("明るさ", 0.5, 1.5, 1.0, 0.05)
        lc = st.color_picker("写真が無いときの色", "#b8b5b0")
        lfiles = st.file_uploader("写真（1〜6 枚）", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if st.form_submit_button("ライブラリに追加") and ln:
            m = library.LibMaterial("", ln, lk, lp, (float(lw), float(lh)), lc, [], float(lsat), float(lbri), product=ln, maker=lmk or "")
            library.save_material(m, [(f.name, f.getvalue()) for f in (lfiles or [])])
            st.success(f"「{ln}」を追加しました"); st.rerun()
    if lib_items:
        show_disc = st.toggle("廃盤も表示", value=False, key="lib_show_disc")
        for m in sorted(lib_items, key=lambda z: (z.discontinued, -(z.updated or 0))):
            if m.discontinued and not show_disc:
                continue
            cc = st.columns([1.2, 2.6, 1.6, 1.2, 0.7, 0.7])
            imgs = library.load_images(m)
            with cc[0]:
                st.image(swatch(m.color, m.kind, m.pattern, m.size_mm, P.comp.texture, images=imgs or None), width=96)
            src = {"ppt": "PPT 凡例から自動", "floor": "床仕上図から自動", "import": "読み込み", "manual": "手入力"}.get(m.source, m.source)
            link = f"　<a href='{m.url}' target='_blank'>製品ページ</a>" if m.url else ""
            cc[1].markdown(f"**{m.name}**{'　<b style=\"color:#c00\">廃盤</b>' if m.discontinued else ''}  \n<span style='font-size:0.85em;color:#666'>"
                           f"{m.maker + '　' if m.maker else ''}{KINDS.get(m.kind, m.kind)}　{m.pattern}貼り　{m.size_mm[0]:g}×{m.size_mm[1]:g}　写真 {len(m.images)} 枚"
                           f"  \n{src}　最終使用 {m.last_used or '-'}　案件 {len(m.projects)} 件{link}</span>", unsafe_allow_html=True)
            if imgs:
                with cc[2]:
                    st.image([Image.fromarray(i).resize((48, 48)) for i in imgs[:6]], width=48)
            with cc[3]:
                more = st.file_uploader("写真を追加", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key=f"libadd_{m.id}", label_visibility="collapsed")
                if more:
                    library.save_material(m, [(f.name, f.getvalue()) for f in more]); st.rerun()
            if cc[4].button("復活" if m.discontinued else "廃盤", key=f"libdisc_{m.id}", help="廃盤にすると材種の候補から外れ、使っている案件では警告を出す"):
                m.discontinued = not m.discontinued; library.save_material(m); st.rerun()
            if cc[5].button("削除", key=f"libdel_{m.id}"):
                library.delete_material(m.id); st.rerun()
    else:
        st.info("まだ登録がありません。上のフォームから追加してください。")

# ---------------------------------------------------------------- 材種と割り当て
with tab_leg:
    if mode == "floor":
        st.markdown("#### 凡例から読み取った材種")
        st.caption("色・倍率（0 = 自動）・貼り方・写真（ライブラリ）はここで直せます。")
        colors, deforms, pattern_over, images_over = {}, {}, {}, {}
        lib_opts = ["（生成模様）"] + [m.id for m in lib_items]
        for m in mats:
            cc = st.columns([0.6, 2.4, 1.0, 0.8, 1.1, 1.4])
            default = material_color(m, P.comp)
            with cc[1]:
                st.markdown(f"**{m.name}**  \n<span style='font-size:0.85em;color:#666'>{m.spec}　{m.pattern}貼り"
                            f"{'　<b style=\"color:#c00\">中止</b>' if m.canceled else ''}</span>", unsafe_allow_html=True)
            with cc[2]:
                colors[m.key] = st.color_picker("色", default, key=f"col_{m.key}", label_visibility="collapsed")
            with cc[3]:
                deforms[m.key] = st.number_input("倍率", 0.0, 8.0, 0.0, 0.5, key=f"def_{m.key}", label_visibility="collapsed")
            with cc[4]:
                pattern_over[m.key] = st.selectbox("貼り方", PATTERNS, index=PATTERNS.index(m.pattern) if m.pattern in PATTERNS else 5,
                                                   key=f"pat_{m.key}", label_visibility="collapsed")
            with cc[5]:
                lid = st.selectbox("写真", lib_opts, key=f"lib_{m.key}", label_visibility="collapsed",
                                   format_func=lambda k: k if k == "（生成模様）" else f"写真: {lib_by_id[k].name}" if k in lib_by_id else k)
                if lid in lib_by_id:
                    images_over[m.key] = library.load_images(lib_by_id[lid])
            with cc[0]:
                st.image(swatch(colors[m.key], m.kind, pattern_over[m.key], m.size_mm, P.comp.texture, images=images_over.get(m.key)), width=64)
        P.comp.colors, P.comp.deform, P.comp.images = colors, deforms, images_over
        for m in mats:
            m.pattern = pattern_over.get(m.key, m.pattern)
        st.markdown("#### 復元した領域")
        import pandas as pd
        mm = ex.mm_per_pt
        mat_name = {m.key: m.name for m in mats}
        rows = [{"id": r.id, "有効": r.id not in P.comp.disabled, "材種": mat_name.get(r.material_key, r.material_key),
                 "面積 m²": round(r.area_pt2 * (mm / 1000) ** 2, 1), "貼り方向 °": r.direction, "頂点": len(r.polygon), "穴": len(r.holes)}
                for r in sorted(ex.regions, key=lambda z: -z.area_pt2)]
        edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", key="regions_editor",
                                column_config={"id": st.column_config.NumberColumn(disabled=True), "材種": st.column_config.TextColumn(disabled=True),
                                               "面積 m²": st.column_config.NumberColumn(disabled=True), "頂点": st.column_config.NumberColumn(disabled=True),
                                               "穴": st.column_config.NumberColumn(disabled=True),
                                               "貼り方向 °": st.column_config.NumberColumn(min_value=0, max_value=179, step=1)})
        P.comp.disabled = set(int(r["id"]) for _, r in edited.iterrows() if not r["有効"])
        P.comp.directions = {int(r["id"]): float(r["貼り方向 °"]) for _, r in edited.iterrows() if pd.notna(r["貼り方向 °"])}
    else:
        import pandas as pd
        seeds = ex.instruction.seeds if ex.instruction is not None else []
        legend = list(ex.instruction.legend) if ex.instruction is not None else []
        callouts = list(ex.instruction.callouts) if ex.instruction is not None else []
        # ---- 凡例をライブラリへ裏で蓄積（品番で紐づけ。最終使用日・案件名を更新）
        if legend:
            lkey = "lib_reg_" + _digest([li.product for li in legend], case or "")
            if not st.session_state.get(lkey):
                try:
                    reg = library.register_from_legend(legend, project=(case or st.session_state.get("proj_name") or ""))
                    st.session_state[lkey] = {"new": [m.name for stt, m in reg if stt == "new"], "updated": [m.name for stt, m in reg if stt == "updated"]}
                except Exception as e:  # noqa: BLE001
                    st.session_state[lkey] = {"error": str(e)}
                lib_items = library.list_materials(); lib_by_id = {m.id: m for m in lib_items}
            info = st.session_state.get(lkey, {})
            if info.get("new") or info.get("updated"):
                st.info("凡例の品番をライブラリに蓄積: 新規 " + ("、".join(info.get("new")) or "なし") + "　／　更新 " + ("、".join(info.get("updated")) or "なし"))
        # ---- 材種の定義（指示書の色ごとに 1 行 ＋ 既定の床）。PPT の凡例があれば品番・メーカー・貼り方・写真を自動で入れる
        st.markdown("#### 材種の定義")
        st.caption("指示書の色ごとに材種を 1 行用意しています（PPT の凡例があれば品番・メーカー・種類・貼り方・写真を自動で入れています）。"
                   "名前・種類・貼り方・寸法・色を直すか、ライブラリの写真を選んでください。行は増やせます。")
        sig = _digest([s.key + s.hex for s in seeds], mode, [li.product for li in legend])
        if st.session_state.get("mat_sig") != sig:
            pal = ["#b8b5b0", "#c9a97c", "#a9adb3", "#d5b98f", "#9c9a96", "#cfb08a", "#b3b8b0", "#c4bfb6"]
            rows = []; seed_mat = {}
            for i, s in enumerate(seeds):
                li = s.legend
                lib = library.find_by_product(li.product, lib_items) if li is not None and li.product else None
                photo = lib.id if lib is not None and lib.images and not lib.discontinued else "（生成模様）"
                color = pal[i % len(pal)]
                if lib is not None and lib.images:
                    try:
                        im = np.array(Image.open(lib.dir / lib.images[0]).convert("RGB")).reshape(-1, 3).mean(axis=0)
                        color = "#%02x%02x%02x" % tuple(int(v) for v in im)
                    except Exception:  # noqa: BLE001
                        pass
                rows.append({"キー": f"m{i + 1}", "名前": (li.product if li is not None and li.product else f"{s.name}の床材"),
                             "品番": li.product if li is not None else "", "メーカー": li.maker if li is not None else "",
                             "種類": (li.kind if li is not None else "carpet") or "carpet", "貼り方": (li.pattern if li is not None else "流し") or "流し",
                             "幅mm": float(li.size_mm[0]) if li is not None and li.size_mm else 500.0,
                             "奥行mm": float(li.size_mm[1]) if li is not None and li.size_mm else 500.0,
                             "色": color, "写真": photo})
                seed_mat[s.key] = f"m{i + 1}"
            rows.append({"キー": "m0", "名前": "既定の床（指示なし）", "品番": "", "メーカー": "", "種類": "carpet", "貼り方": "流し", "幅mm": 500.0, "奥行mm": 500.0,
                         "色": "#e3e1dd", "写真": "（生成模様）"})
            st.session_state["mat_rows"] = rows; st.session_state["mat_sig"] = sig
            st.session_state["seed_mat"] = seed_mat
            st.session_state["seed_mode"] = {}; st.session_state["room_mat"] = {}; st.session_state["default_mat"] = "m0"
            st.session_state["room_splits"] = {}; st.session_state["rect_covers"] = []
        disc_names = [m.name for m in lib_items if m.discontinued and any(r.get("写真") == m.id or (r.get("品番") and library.norm_code(r["品番"]) == library.norm_code(m.product or m.name)) for r in st.session_state["mat_rows"])]
        if disc_names:
            st.warning("廃盤の製品が使われています: " + "、".join(disc_names) + "。代替品をライブラリから選ぶか、写真を差し替えてください。")
        lib_opts = ["（生成模様）"] + [m.id for m in lib_items if not m.discontinued or any(r.get("写真") == m.id for r in st.session_state["mat_rows"])]
        for r in st.session_state["mat_rows"]:
            r.setdefault("品番", ""); r.setdefault("メーカー", "")
        mdf = st.data_editor(pd.DataFrame(st.session_state["mat_rows"]), hide_index=True, width="stretch", num_rows="dynamic", key="mat_editor",
                             column_config={"キー": st.column_config.TextColumn(disabled=True, width="small"),
                                            "品番": st.column_config.TextColumn(width="small"), "メーカー": st.column_config.TextColumn(width="small"),
                                            "種類": st.column_config.SelectboxColumn(options=list(KINDS), width="small"),
                                            "貼り方": st.column_config.SelectboxColumn(options=PATTERNS, width="small"),
                                            "幅mm": st.column_config.NumberColumn(min_value=50, max_value=3000, step=10, width="small"),
                                            "奥行mm": st.column_config.NumberColumn(min_value=50, max_value=3000, step=10, width="small"),
                                            "色": st.column_config.TextColumn(width="small", help="#RRGGBB"),
                                            "写真": st.column_config.SelectboxColumn(options=lib_opts, width="medium")})
        mat_rows = []
        used = set()
        for i, r in mdf.iterrows():
            key = str(r.get("キー") or "")
            if not key or key in used:
                key = f"u{i + 1}"
            used.add(key)
            mat_rows.append({"キー": key, "名前": str(r.get("名前") or key), "品番": str(r.get("品番") or ""), "メーカー": str(r.get("メーカー") or ""),
                             "種類": str(r.get("種類") or "carpet"), "貼り方": str(r.get("貼り方") or "流し"),
                             "幅mm": float(r.get("幅mm") or 500), "奥行mm": float(r.get("奥行mm") or 500), "色": str(r.get("色") or "#b8b5b0"),
                             "写真": str(r.get("写真") or "（生成模様）")})
        st.session_state["mat_rows"] = mat_rows
        P.materials, P.comp.colors, P.comp.images = [], {}, {}
        for r in mat_rows:
            lib = lib_by_id.get(r["写真"])
            if lib is not None:
                m = Material(r["キー"], None, kind=lib.kind, product=r["名前"], size_mm=tuple(lib.size_mm), pattern=lib.pattern)
                P.comp.images[r["キー"]] = library.load_images(lib)
                P.comp.colors[r["キー"]] = lib.color
            else:
                m = Material(r["キー"], None, kind=r["種類"], product=r["名前"], size_mm=(r["幅mm"], r["奥行mm"]), pattern=r["貼り方"])
                col = r["色"] if r["色"].startswith("#") and len(r["色"]) == 7 else "#b8b5b0"
                P.comp.colors[r["キー"]] = col
            P.materials.append(m)
        mat_keys = [m.key for m in P.materials]
        mat_label = {m.key: f"{m.key}: {m.product}" for m in P.materials}
        none_label = "（塗らない）"
        if legend:
            with st.expander(f"指示書（PPT）の凡例 {len(legend)} 件"):
                for li in legend:
                    lc = st.columns([0.4, 3.0, 1.0])
                    lc[0].markdown(f"<div style='width:28px;height:28px;border:1px solid #999;background:{li.to_dict()['color']}'></div>", unsafe_allow_html=True)
                    lc[1].markdown(f"**{li.product or '（品番なし）'}**　{li.maker}　{KINDS.get(li.kind, li.kind)}　{li.pattern}貼り"
                                   f"{'　既存のまま' if li.existing else ''}  \n<span style='font-size:0.8em;color:#666'>{li.text[:80]}</span>"
                                   + (f"　[製品ページ]({li.url})" if li.url else ""), unsafe_allow_html=True)
                    if li.photo:
                        lc[2].image(li.photo, width=72)
        if callouts:
            with st.expander(f"手書きの吹き出し（文字・写真） {len(callouts)} 件"):
                st.caption("iPad などで図面に描き足した文字と貼った写真。引出線の先の部屋に紐づけています（部屋の割り当ての参考。文字は PDF に埋め込まれた認識結果があればそれ）。")
                for co in callouts:
                    cc_ = st.columns([1.4, 3.0])
                    if co.png:
                        cc_[0].image(co.png, width=200)
                    cc_[1].markdown(f"**{'写真' if co.kind == 'photo' else '文字'}**　→ 部屋 {co.room_id if co.room_id else '（引出線なし）'}  \n{co.text or ''}")
        # ---- 色 → 材種
        if seeds:
            st.markdown("#### 指示書の色 → 材種")
            seed_mat = dict(st.session_state.get("seed_mat", {})); seed_mode = dict(st.session_state.get("seed_mode", {}))
            for s in seeds:
                cc = st.columns([0.5, 1.6, 2.2, 1.6])
                cc[0].markdown(f"<div style='width:36px;height:36px;border:1px solid #999;background:{s.hex}'></div>", unsafe_allow_html=True)
                cc[1].markdown(f"**{s.name}** {s.hex}  \n<span style='font-size:0.85em;color:#666'>{s.area_m2:.0f} m² / {s.blobs} 塊</span>", unsafe_allow_html=True)
                opts = [none_label] + mat_keys
                cur = seed_mat.get(s.key, none_label)
                v = cc[2].selectbox("材種", opts, index=opts.index(cur) if cur in opts else 0, key=f"sm_{sig}_{s.key}", label_visibility="collapsed",
                                    format_func=lambda k: mat_label.get(k, k))
                seed_mat[s.key] = v
                mo = cc[3].selectbox("割り当て方", ["auto", "room", "band"], index=["auto", "room", "band"].index(seed_mode.get(s.key, "auto")),
                                     key=f"smo_{sig}_{s.key}", label_visibility="collapsed",
                                     format_func=lambda k: {"auto": "自動（部屋／塗った形）", "room": "部屋ごと", "band": "塗った形だけ"}[k])
                seed_mode[s.key] = mo
            st.session_state["seed_mat"], st.session_state["seed_mode"] = seed_mat, seed_mode
            P.seed_materials = {k: v for k, v in seed_mat.items() if v != none_label}
            P.seed_modes = seed_mode
        dopts = [none_label] + mat_keys
        cur = st.session_state.get("default_mat", "m0")
        default_mat = st.selectbox("指示の無い部屋に塗る材種", dopts, index=dopts.index(cur) if cur in dopts else 0, key=f"dm_{sig}",
                                   format_func=lambda k: mat_label.get(k, k))
        st.session_state["default_mat"] = default_mat
        P.default_material = None if default_mat == none_label else default_mat
        # ---- 部屋ごとの上書き（叩き → 後編集: クリックで割り当て／塗らない／線で分割／矩形で塗る。頂点の編集はしない）
        st.markdown("#### 部屋ごとの割り当て（後編集）")
        room_mat = {str(k): v for k, v in st.session_state.get("room_mat", {}).items()}
        room_splits = {int(k): [tuple(l) for l in v] for k, v in st.session_state.get("room_splits", {}).items()}
        rect_covers = [dict(c) for c in st.session_state.get("rect_covers", [])]
        P.room_materials = dict(room_mat); P.room_splits = room_splits; P.rect_covers = rect_covers
        ov_pre = pipeline.paint_overlay_with(ex, P)
        if _img_click is not None:
            e1, e2 = st.columns([1.4, 2.6])
            tool = e1.radio("操作", ["材種を割り当て", "塗らない", "線で分割（2 点）", "矩形で塗る（2 点）", "上書きを解除"], key=f"tool_{sig}",
                            help="材種を割り当て: 部屋をクリック。塗らない: 部屋をクリック。線で分割: 2 点クリックで部屋をその線で分け、分けた部分（12a/12b）に別の材種を割り当てられる（共用部と専有部の張り分けなど）。"
                                 "矩形で塗る: 2 点クリックの矩形を材種で上書き（塗らないも可）。上書きを解除: 部屋の上書き・分割を消す")
            pick = e2.radio("材種", mat_keys + [none_label], horizontal=True, key=f"pick_{sig}", format_func=lambda k: mat_label.get(k, k))
            pend = st.session_state.get("_pend_pt")
            if tool in ("線で分割（2 点）", "矩形で塗る（2 点）"):
                st.caption(("1 点目を置きました。2 点目をクリックしてください。" if pend else "1 点目をクリックしてください。") + "　番号は部屋 id（分割後は id に a/b…）。")
            else:
                st.caption("図の部屋をクリックすると反映します（指示書の割り当てより優先）。番号は部屋 id。")
            crop = ex.zone
            x0, y0, x1, y1 = [int(v * ex.S) for v in crop]
            view = ov_pre[y0:y1, x0:x1].copy()
            if pend:
                import cv2 as _cv
                _cv.circle(view, (int(pend[0] - x0), int(pend[1] - y0)), 12, (255, 0, 0), 3)
            disp_w = 1400
            f = disp_w / view.shape[1]
            small = np.array(Image.fromarray(view).resize((disp_w, max(1, int(view.shape[0] * f)))))
            pt = _img_click(Image.fromarray(small), key=f"click_{sig}_{extract_key}_{tool}")
            if pt and pt.get("x") is not None:
                px, py = int(pt["x"] / f) + x0, int(pt["y"] / f) + y0
                last = st.session_state.get("_last_click")
                if (px, py) != last:
                    st.session_state["_last_click"] = (px, py)
                    H_, W_ = ex.rooms.room_map.shape
                    rid = int(ex.rooms.room_map[py, px]) if 0 <= py < H_ and 0 <= px < W_ else 0
                    sub = next((k for k, sm in ex.subrooms.get(rid, []) if 0 <= py < H_ and 0 <= px < W_ and sm[py, px] > 0), None)
                    changed = False
                    if tool == "材種を割り当て" and rid > 0 and pick != none_label:
                        room_mat[sub or str(rid)] = pick; changed = True
                    elif tool == "塗らない" and rid > 0:
                        room_mat[sub or str(rid)] = ""; changed = True
                    elif tool == "上書きを解除" and rid > 0:
                        for k in [k for k in room_mat if k == str(rid) or (k.startswith(str(rid)) and k[len(str(rid)):].isalpha())]:
                            room_mat.pop(k, None)
                        room_splits.pop(rid, None); changed = True
                    elif tool in ("線で分割（2 点）", "矩形で塗る（2 点）"):
                        if not pend:
                            st.session_state["_pend_pt"] = (px, py); st.rerun()
                        else:
                            st.session_state["_pend_pt"] = None
                            if tool == "線で分割（2 点）":
                                r0 = int(ex.rooms.room_map[int(pend[1]), int(pend[0])]) if 0 <= pend[1] < H_ and 0 <= pend[0] < W_ else 0
                                target = r0 or rid
                                if target > 0:
                                    room_splits.setdefault(target, []).append((int(pend[0]), int(pend[1]), px, py)); changed = True
                            else:
                                rect_covers.append({"rect": (int(min(pend[0], px)), int(min(pend[1], py)), int(max(pend[0], px)), int(max(pend[1], py))),
                                                    "material": "" if pick == none_label else pick, "direction": 0.0}); changed = True
                    if changed:
                        st.session_state["room_mat"] = room_mat; st.session_state["room_splits"] = {str(k): v for k, v in room_splits.items()}
                        st.session_state["rect_covers"] = rect_covers
                        st.rerun()
            if room_splits or rect_covers:
                with st.expander(f"分割線 {sum(len(v) for v in room_splits.values())} 本・矩形 {len(rect_covers)} 個", expanded=False):
                    for rid_, lines in list(room_splits.items()):
                        for j, l in enumerate(lines):
                            c_ = st.columns([4, 1])
                            c_[0].write(f"部屋 {rid_} の分割線 {j + 1}: ({l[0]}, {l[1]}) → ({l[2]}, {l[3]})")
                            if c_[1].button("消す", key=f"dels_{rid_}_{j}"):
                                lines.pop(j)
                                if not lines:
                                    room_splits.pop(rid_)
                                st.session_state["room_splits"] = {str(k): v for k, v in room_splits.items()}; st.rerun()
                    for j, cv_ in enumerate(rect_covers):
                        c_ = st.columns([4, 1])
                        c_[0].write(f"矩形 {j + 1}: {cv_['rect']} → {mat_label.get(cv_['material'], none_label if not cv_['material'] else cv_['material'])}")
                        if c_[1].button("消す", key=f"delr_{j}"):
                            rect_covers.pop(j); st.session_state["rect_covers"] = rect_covers; st.rerun()
        else:
            st.image(ov_pre, width="stretch")
        rows = []
        for r in sorted(ex.rooms.rooms, key=lambda z: -z.area_m2):
            keys = [str(r.id)] + [k for k, _ in ex.subrooms.get(r.id, [])]
            for k in keys:
                rows.append({"id": k, "室名": r.name if k == str(r.id) else "　└ 分割", "面積 m²": round(r.area_m2, 1) if k == str(r.id) else None,
                             "コア": r.core, "指示書の色": (r.seed or ""),
                             "材種（上書き）": (mat_label.get(room_mat[k], none_label) if k in room_mat else "（指示書／既定）")})
        opts_over = ["（指示書／既定）", none_label] + [mat_label[k] for k in mat_keys]
        rdf = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", key=f"rooms_{sig}_{_digest(room_mat)}_{_digest(room_splits)}",
                             column_config={"id": st.column_config.TextColumn(disabled=True), "室名": st.column_config.TextColumn(disabled=True),
                                            "面積 m²": st.column_config.NumberColumn(disabled=True), "コア": st.column_config.TextColumn(disabled=True),
                                            "指示書の色": st.column_config.TextColumn(disabled=True),
                                            "材種（上書き）": st.column_config.SelectboxColumn(options=opts_over)})
        inv = {v: k for k, v in mat_label.items()}
        new_rm = {}
        for _, r in rdf.iterrows():
            v = r["材種（上書き）"]
            if v == none_label:
                new_rm[str(r["id"])] = ""
            elif v in inv:
                new_rm[str(r["id"])] = inv[v]
        if new_rm != room_mat:
            st.session_state["room_mat"] = new_rm
            room_mat = new_rm
        P.room_materials = dict(room_mat); P.room_splits = room_splits; P.rect_covers = rect_covers
        P.materials = [m for m in P.materials]
    with st.expander("抽出の診断"):
        st.json({"align": al.__dict__, "regions": ex.diag, "clean": ex.clean.stats, "timings": ex.timings,
                 "layers": ex.layer_plan.summary() if ex.layer_plan is not None else None,
                 "instruction": ex.instruction.align if ex.instruction is not None else None})

# ---------------------------------------------------------------- 合成
if mode == "floor":
    P.comp.paint_existing = {"凡例にあれば塗る": any(m.kind == "existing" for m in ex.materials), "塗る": True, "塗らない": False}[paint_existing_opt]
t0 = time.time()
res = pipeline.render(ex, P)
t_rd = time.time() - t0
img = pipeline.cropped(res.image, res.crop)
ov = pipeline.cropped(res.overlay, res.crop)

with tab_out:
    st.image(img, width="stretch", caption=f"平面着色（{img.shape[1]}×{img.shape[0]} px、合成 {t_rd:.1f}s）")
    with st.expander("部分拡大"):
        zc1, zc2, zc3 = st.columns(3)
        zx = zc1.slider("中心 x [%]", 0, 100, 50, key="zx"); zy = zc2.slider("中心 y [%]", 0, 100, 50, key="zy")
        zw = zc3.slider("幅 [px]", 300, 2400, 1000, 100, key="zw")
        H, W = img.shape[:2]
        cx, cy = int(W * zx / 100), int(H * zy / 100); zh = int(zw * 0.6)
        x0, y0 = max(0, cx - zw // 2), max(0, cy - zh // 2)
        st.image(img[y0:y0 + zh, x0:x0 + zw], width="stretch")

with tab_chk:
    if mode == "floor":
        st.caption("床仕上図から復元した領域を凡例の色で平面図に重ねたもの。両矢印は貼り方向。ここが合っていれば着色は正しい。")
    else:
        st.caption("部屋を材種の色で塗り分け、指示書の色の塊を半透明で重ねたもの。番号は部屋 id（「材種と割り当て」タブで上書きできる）。")
    st.image(ov, width="stretch")
    with st.expander("家具の足元（白く残して影を付ける範囲）"):
        st.image(pipeline.cropped(ex.clean.furniture, res.crop), width="stretch")
    if mode != "floor" and ex.paints:
        import pandas as pd
        st.markdown("**塗った単位**")
        st.dataframe(pd.DataFrame([{"色": q.seed, "部屋": str(q.room_id) if q.room_id is not None else "（塗った形）", "方法": {"room": "部屋全体", "band": "塗った形"}[q.mode],
                                    "面積 m²": round(q.area_m2, 1), "材種": q.material_key or "（塗らない）"} for q in ex.paints]),
                     hide_index=True, width="stretch")

with tab_dbg:
    import pandas as pd
    stages = list(ex.stages) + ([res.stage] if res.stage is not None else [])
    total = sum(s.seconds for s in stages)
    st.markdown("#### 段階ごとの状況と性能")
    st.caption("各段階の所要時間・指標・途中画像。改善したい段階を特定するために使う。抽出はファイルごとにキャッシュされ、"
               "合成は見た目のパラメータを変えるたびに走る。")
    out_px = (res.crop[2] - res.crop[0]) * (res.crop[3] - res.crop[1])
    m1, m2, m3, m4, m5 = st.columns(5)
    _fact(m1, "処理時間の合計", f"{total:.1f} 秒", "抽出＋合成")
    _fact(m2, "抽出", f"{sum(s.seconds for s in ex.stages):.1f} 秒", "ファイルごとにキャッシュ")
    _fact(m3, "合成", f"{res.stage.seconds:.1f} 秒" if res.stage else "-", "見た目の変更ごとに再実行")
    _fact(m4, "出力画像の大きさ", f"{out_px / 1e6:.1f} Mpx", f"{P.dpi} dpi")
    _fact(m5, "処理速度", f"{out_px / 1e6 / max(total, 1e-6):.2f} Mpx/秒", "出力画素数 ÷ 合計時間")
    icon = {"ok": "✅", "warn": "⚠️", "error": "❌"}
    rows = [{"段階": s.name, "状態": icon.get(s.status, s.status), "時間 [s]": round(s.seconds, 2),
             "割合": 100.0 * s.seconds / max(total, 1e-6),
             "主な指標": "、".join(f"{k}={v}" for k, v in s.metrics.items() if not isinstance(v, (dict, list)))[:120],
             "所見": " / ".join(s.notes)[:160]} for s in stages]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"割合": st.column_config.ProgressColumn("時間の割合", min_value=0.0, max_value=100.0, format="%.0f%%")})
    for s in stages:
        with st.expander(f"{icon.get(s.status, '')} {s.name}　{s.seconds:.2f} s", expanded=(s.status != "ok")):
            for n in s.notes:
                st.warning(n)
            simple = {k: v for k, v in s.metrics.items() if not isinstance(v, (dict, list))}
            nested = {k: v for k, v in s.metrics.items() if isinstance(v, (dict, list))}
            if simple:
                st.dataframe(pd.DataFrame([simple]).T.rename(columns={0: "値"}).astype(str), width="stretch")
            for k, v in nested.items():
                if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
                    st.markdown(f"**{k}**"); st.dataframe(pd.DataFrame(v).T, width="stretch")
                elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                    st.markdown(f"**{k}**")
                    for x in v:
                        st.markdown(f"- {x}")
                else:
                    st.markdown(f"**{k}**"); st.json(v, expanded=False)
            imgs = list(s.images.items())
            if imgs:
                ncol = 2 if len(imgs) > 1 else 1
                for i in range(0, len(imgs), ncol):
                    cols = st.columns(ncol)
                    for c, (nm, im) in zip(cols, imgs[i:i + ncol]):
                        c.image(im, caption=nm, width="stretch")

with tab_dl:
    base = f"floortint_{_digest(plan_bytes)[:6]}"
    d1, d2, d3 = st.columns(3)
    # PNG/PDF の符号化は画面を描き直すたびに走ると重い（A3 300dpi で各 1〜2 秒・数百 MB）ので、押されたときだけ作る
    if d1.toggle("PNG を用意する", value=False, key="mk_png"):
        d1.download_button("PNG をダウンロード", _png_bytes(img), f"{base}.png", "image/png", width="stretch")
    if d2.toggle("PDF を用意する", value=False, key="mk_pdf"):
        d2.download_button("PDF をダウンロード", pipeline.to_pdf_bytes(img, P.dpi), f"{base}.pdf", "application/pdf", width="stretch")
    d3.download_button("抽出結果 JSON", json.dumps(res.summary(), ensure_ascii=False, indent=1).encode(), f"{base}.json",
                       "application/json", width="stretch")
    st.caption("PDF は画像を 1 ページに貼ったもの。用紙寸法は出力解像度から逆算しています。")
    st.markdown("#### レイヤー分けの書き出し")
    st.caption("床・影・家具（白）・躯体・線画・植栽・文字を別々の透過 PNG にして ZIP にします（重ねると完成画像）。手直しに使えます。")
    if st.toggle("レイヤー分け ZIP を用意する", value=False, key="mk_layers"):
        lay: dict = {}
        res2 = pipeline.render(ex, P, layers=lay)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for i, (nm, im) in enumerate(lay.items(), start=1):
                z.writestr(f"{i:02d}_{nm}.png", _png_bytes(pipeline.cropped(im, res2.crop)))
            z.writestr("00_合成.png", _png_bytes(pipeline.cropped(res2.image, res2.crop)))
            z.writestr("summary.json", json.dumps(res2.summary(), ensure_ascii=False, indent=1))
        st.download_button("レイヤー分け ZIP をダウンロード", buf.getvalue(), f"{base}_layers.zip", "application/zip", width="stretch")
        st.caption("含まれるレイヤー: " + "、".join(lay.keys()))

# ---------------------------------------------------------------- 物件設定の保存
if save_clicked and proj_name:
    widget_keys = ["in_mode", "mode", "target_mm", "fixed_deform", "deform", "joint", "variation", "grain", "fiber", "seed", "shadow_px", "shadow_blur",
                   "shadow_alpha", "f_max", "f_min", "close_gaps", "chair_gap", "chair_max", "plant_mode", "plant_color", "plant_alpha", "struct_mode",
                   "struct_color", "line_alpha", "keep_names", "text_scale", "paint_existing", "existing_color", "dpi", "margin", "room_min",
                   "sat_min", "min_blob", "merge_de", "min_width", "small_room", "room_cover", "simplify_mm", "m_scale", "m_dx", "m_dy",
                   "scale_override", "paint_core", "paint_common", "common_pattern", "drop_grid"]
    settings = {"name": proj_name, "widgets": {k: st.session_state[k] for k in widget_keys if k in st.session_state},
                "mat_rows": st.session_state.get("mat_rows"), "seed_mat": st.session_state.get("seed_mat"), "seed_mode": st.session_state.get("seed_mode"),
                "room_mat": st.session_state.get("room_mat"), "default_mat": st.session_state.get("default_mat"), "layer_over": layer_over,
                "mat_sig": st.session_state.get("mat_sig"), "room_splits": st.session_state.get("room_splits"), "rect_covers": st.session_state.get("rect_covers")}
    path = library.save_project(proj_name, settings)
    st.sidebar.success(f"保存しました: {path.name}")
