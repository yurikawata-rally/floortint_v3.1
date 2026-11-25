"""床材ライブラリの材種セット ZIP（書き出し・読み込みの確認用）を作る。

  ..\\.venv\\Scripts\\python.exe make_material_set.py            # このフォルダに書く
  ..\\.venv\\Scripts\\python.exe make_material_set.py 出力先フォルダ

  material_set_3F.zip    正常なセット。一時フォルダのライブラリに登録し、アプリの書き出し（library.export_set）で ZIP にする。
                         実際のライブラリ（app/library/ や FLOORTINT_LIBRARY の先）には触らない
  material_set_edge.zip  書き出しでは出ない形を手で組んだセット。読み込み（library.import_set）の分岐を確かめる

品番・メーカー・リンク先・写真はすべて架空。SMP-101〜SMV-012 は make_mode_samples.py の指示書と同じ品番なので、
material_set_3F.zip を読み込んでから instruction_ppt_3F.pdf を読むと、凡例の 4 品番がライブラリの材種に紐づく（新規登録されない）。

material_set_3F.zip
  SMP-101  タイルカーペット    市松          500×500   写真 3 枚
  SMP-205  タイルカーペット    流し          500×500   写真 2 枚  彩度 0.9・明るさ 1.05
  SMP-310  タイルカーペット    アシュラー    250×1000  写真 2 枚
  SMV-012  塩ビタイル（木目）  ヘリンボーン  150×900   写真 3 枚
  SMT-600  タイル              ブリック      300×600   写真なし（「写真が無いときの色」だけ）
  SMP-099  タイルカーペット    流し          500×500   写真 1 枚  廃盤（一覧では「廃盤も表示」で出る）

material_set_edge.zip（material_set_3F.zip の後に読み込んだときの結果。空のライブラリなら 5 件とも読み込む）
  ＳＭＰ－１０１  全角で書いた SMP-101             飛ばす（品番の正規化 norm_code で同じ品番）
  smp 205         小文字・空白の SMP-205           飛ばす
  SMX-701         写真 2 枚と書いて 1 枚だけ入れる  読み込む。写真 1 枚（ZIP に無い写真は飛ばす）
  SMX-702         写真 7 枚（1 枚目は PNG 1600×1200） 読み込む。写真 6 枚（MAX_IMAGES）、JPEG・長辺 1024 に縮む
  SMX-703         id と名前だけ＋未知の項目         読み込む。種類・貼り方・寸法・色は既定（carpet・流し・500×500）

写真の作成に numpy・Pillow（floortint の依存と同じ）を使い、floortint.library を読み込むので、アプリの仮想環境で動かす。
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "app"))

SET_NAME = "サンプルオフィス3F"
PROJECT = "サンプルオフィス（架空）"
MAKER = "架空床材"
NOTE = "架空のサンプル（材種セットの確認用）"

# 品番, 種類, 貼り方, 寸法 [mm], 写真の質感, 写真の枚数, 下地の色（RGB）, 追加の項目
SET_3F = [
    ("SMP-101", "carpet", "市松", (500.0, 500.0), "loop", 3, (88, 104, 128), {}),
    ("SMP-205", "carpet", "流し", (500.0, 500.0), "stripe", 2, (176, 150, 116), {"saturation": 0.9, "brightness": 1.05}),
    ("SMP-310", "carpet", "アシュラー", (250.0, 1000.0), "plank", 2, (102, 122, 104), {}),
    ("SMV-012", "vinyl", "ヘリンボーン", (150.0, 900.0), "wood", 3, (164, 120, 82), {}),
    ("SMT-600", "tile", "ブリック", (300.0, 600.0), "", 0, (207, 200, 187), {}),
    ("SMP-099", "carpet", "流し", (500.0, 500.0), "loop", 1, (128, 128, 124), {"discontinued": True}),
]


def piece_photo(style: str, w: int, h: int, seed: int, base: tuple, noise: float = 1.0) -> Image.Image:
    """製品 1 枚を正面から撮った写真の代わり。style: loop（ループパイル）| stripe（細い縞）| plank（長手の筋）| wood（木目）。
    seed ごとにロットの色の揺らぎと周辺減光を変え、縁を少し暗くして 1 枚の外形が分かるようにする。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n = rng.normal(0, 1, (h, w)) * noise
    if style == "loop":
        g = 9 * n + 5 * np.sin(yy / 1.6 + rng.uniform(0, 6))
    elif style == "stripe":
        g = 10 * np.sin(yy / 3.0) + 7 * n
    elif style == "plank":                         # 長辺（縦）方向に流れる筋
        g = 8 * np.sin(xx / 2.2 + 2 * np.sin(yy / 40)) + 8 * n
    else:                                          # 木目: 長辺（横）方向にうねる年輪
        yw = yy + 6 * np.sin(xx / 90 + rng.uniform(0, 6)) + 3 * np.sin(xx / 23)
        g = 14 * np.sin(yw / 2.8) + 8 * np.sin(yw / 7.3) + 4 * n
    shade = 1 + rng.uniform(-0.04, 0.04)
    vign = 1 - 0.25 * ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
    rgb = np.array(base, np.float32)[None, None, :] * shade * vign[..., None] + g[..., None]
    e = max(2, min(w, h) // 60)
    rgb[:e] *= 0.8; rgb[-e:] *= 0.8; rgb[:, :e] *= 0.8; rgb[:, -e:] *= 0.8
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _photo_px(kind: str, size_mm: tuple[float, float], long_px: int = 600) -> tuple[int, int]:
    """写真の幅・高さ [px]。寸法の比のまま長辺を long_px に。板（塩ビ・木）は長辺を横にする（textures.make_texture と同じ向き）。"""
    sw, sh = size_mm
    if kind in ("vinyl", "wood") and sw < sh:
        sw, sh = sh, sw
    f = long_px / max(sw, sh)
    return max(2, int(round(sw * f))), max(2, int(round(sh * f)))


def _encode(im: Image.Image, fmt: str = "PNG") -> bytes:
    b = io.BytesIO()
    im.save(b, fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    return b.getvalue()


def make_set_3f(dst: Path) -> None:
    from floortint import library
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["FLOORTINT_LIBRARY"] = tmp
        ids = []
        for i, (code, kind, pattern, size, style, nimg, base, extra) in enumerate(SET_3F):
            m = library.LibMaterial(f"{library._slug(code)}_sample", code, kind, pattern, size, "#%02x%02x%02x" % base,
                                    product=code, maker=MAKER, url=f"https://example.com/floortint-sample/{code}",
                                    last_used="2026-10-30", projects=[PROJECT], note=NOTE, **extra)
            w, h = _photo_px(kind, size)
            photos = [(f"{code}_{j + 1}.png", _encode(piece_photo(style, w, h, 100 * i + j, base))) for j in range(nimg)]
            ids.append(library.save_material(m, photos).id)
        dst.write_bytes(library.export_set(ids, SET_NAME))
    del os.environ["FLOORTINT_LIBRARY"]


def make_set_edge(dst: Path) -> None:
    from floortint import library

    def meta(mid: str, code: str, images: list[str], **kw) -> dict:
        j = asdict(library.LibMaterial(mid, code, product=code, maker=MAKER, note=NOTE, images=images, **kw))
        j["size_mm"] = list(j["size_mm"])
        return j

    files: dict[str, bytes] = {}
    metas = []
    # 表記違いの品番（norm_code で SMP-101・SMP-205 と同じになる）
    for mid, code, style, base, seed in (("SMP-101_zenkaku", "ＳＭＰ－１０１", "loop", (88, 104, 128), 900),
                                         ("SMP-205_space", "smp 205", "stripe", (176, 150, 116), 901)):
        metas.append(meta(mid, code, ["01_front.jpg"], color="#%02x%02x%02x" % base))
        files[f"{mid}/01_front.jpg"] = _encode(piece_photo(style, 600, 600, seed, base), "JPEG")
    # 写真 2 枚と書いて 1 枚だけ入れる
    metas.append(meta("SMX-701_sample", "SMX-701", ["01_front.jpg", "02_missing.jpg"], color="#8a7a6a"))
    files["SMX-701_sample/01_front.jpg"] = _encode(piece_photo("loop", 600, 600, 902, (138, 122, 106)), "JPEG")
    # 写真 7 枚（上限 6 枚を超える）。1 枚目は長辺 1600 の PNG（縮小と JPEG への変換を見る。ZIP を小さくするため粒は入れない）
    names = ["01_large.png"] + [f"{k:02d}_front.jpg" for k in range(2, 8)]
    metas.append(meta("SMX-702_sample", "SMX-702", names, kind="wood", pattern="流し", size_mm=(180.0, 1820.0), color="#9c7350"))
    files["SMX-702_sample/01_large.png"] = _encode(piece_photo("wood", 1600, 1200, 903, (156, 115, 80), noise=0.0))
    for k, nm in enumerate(names[1:]):
        files[f"SMX-702_sample/{nm}"] = _encode(piece_photo("wood", 600, 60, 904 + k, (156, 115, 80)), "JPEG")
    # id と名前だけ（他の項目は読み込み側の既定）。未知の項目は読み込みで捨てる
    metas.append({"id": "SMX-703_sample", "name": "SMX-703", "future_field": "未知の項目（読み込みで捨てる）"})
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("materials.json", json.dumps({"name": "材種セットの確認（境界）", "exported": time.strftime("%Y-%m-%d %H:%M"),
                                                 "materials": metas}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("material_set_3F.zip", make_set_3f), ("material_set_edge.zip", make_set_edge)):
        fn(out / name)
        print(f"{out / name}  ({(out / name).stat().st_size // 1024} KB)")
