"""床材の画像ライブラリと物件ごとの設定の保存。

共有フォルダ（既定: アプリの library/、環境変数 FLOORTINT_LIBRARY で変更）に置く。
  library/materials/<id>/meta.json + 画像ファイル   製品ごとの写真（1〜6 枚。1 枚 = 製品 1 枚の実物）
  library/projects/<名前>.json                       物件ごとの設定（材種の対応、部屋の上書き、見た目のパラメータ）
画像は製品 1 枚（タイル 1 枚・板 1 枚）を正面から撮ったものを想定し、貼るときに寸法に合わせて縮める。
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from floortint.legend import Material

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "library"
MAX_IMAGES = 6
MAX_SIDE = 1024          # 保存時に長辺をこの画素数まで縮める


def lib_dir() -> Path:
    d = Path(os.environ.get("FLOORTINT_LIBRARY", str(DEFAULT_DIR)))
    (d / "materials").mkdir(parents=True, exist_ok=True)
    (d / "projects").mkdir(parents=True, exist_ok=True)
    return d


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", s.strip(), flags=re.UNICODE).strip("_")
    return s[:60] or "item"


@dataclass
class LibMaterial:
    id: str
    name: str                              # 表示名（品番など）
    kind: str = "carpet"                   # carpet | vinyl | wood | tile | other
    pattern: str = "流し"
    size_mm: tuple[float, float] = (500.0, 500.0)
    color: str = "#b8b5b0"                 # 写真が無いときの色
    images: list[str] = field(default_factory=list)   # ファイル名
    saturation: float = 1.0
    brightness: float = 1.0
    joint: float = 0.5
    note: str = ""
    updated: float = 0.0
    product: str = ""                      # 品番（名前と別に持つ。裏で紐づけるキー）
    maker: str = ""
    url: str = ""                          # 製品ページ（PPT のリンクから）
    source: str = "manual"                 # manual | ppt | floor | import
    last_used: str = ""                    # 最後に案件で使った日 YYYY-MM-DD
    projects: list[str] = field(default_factory=list)   # 使った案件名
    discontinued: bool = False             # 廃盤（手で付ける。候補から外し、警告を出す）

    def to_material(self) -> Material:
        m = Material(self.id, None, kind=self.kind, product=self.name, size_mm=tuple(self.size_mm), pattern=self.pattern, note=self.note)
        return m

    @property
    def dir(self) -> Path:
        return lib_dir() / "materials" / self.id


def norm_code(code: str) -> str:
    """品番の正規化: 全角→半角、空白・ハイフンを除き大文字に（GA-144 / GA144 / ga 144 を同一視）。"""
    import unicodedata
    c = unicodedata.normalize("NFKC", code or "")
    return re.sub(r"[\s\-‐–_/]", "", c).upper()


def find_by_product(code: str, items: list[LibMaterial] | None = None) -> LibMaterial | None:
    k = norm_code(code)
    if not k:
        return None
    for m in (items if items is not None else list_materials()):
        if norm_code(m.product) == k or norm_code(m.name) == k:
            return m
    return None


def register_from_legend(legend: list, project: str = "", photos: bool = True) -> list[tuple[str, LibMaterial]]:
    """指示書（PPT）の凡例を裏でライブラリに蓄積する。品番が既にあれば最終使用日・案件名を更新し、写真が無ければ足す。
    無ければ新規（写真があれば写真、無ければ見本の色だけ）。戻り値 [(状態 "new"|"updated", 材種)]。"""
    out = []
    today = time.strftime("%Y-%m-%d")
    items = list_materials()
    for li in legend:
        code = getattr(li, "product", "") or ""
        if not code:
            continue
        m = find_by_product(code, items)
        photo = getattr(li, "photo", None) if photos else None
        color = "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in getattr(li, "color", (0.72, 0.71, 0.69)))
        if m is None:
            m = LibMaterial("", code, kind=getattr(li, "kind", "carpet") or "carpet", pattern=getattr(li, "pattern", "流し") or "流し",
                            size_mm=tuple(getattr(li, "size_mm", None) or (500.0, 500.0)), color=color, product=code,
                            maker=getattr(li, "maker", "") or "", url=getattr(li, "url", "") or "", source="ppt", last_used=today,
                            projects=[project] if project else [], note=(getattr(li, "text", "") or "")[:80])
            m = save_material(m, [("legend_photo.png", photo)] if photo else [])
            items.append(m); out.append(("new", m))
        else:
            changed = False
            if m.last_used != today:
                m.last_used = today; changed = True
            if project and project not in m.projects:
                m.projects.append(project); changed = True
            for attr in ("maker", "url"):
                v = getattr(li, attr, "") or ""
                if v and not getattr(m, attr):
                    setattr(m, attr, v); changed = True
            if not m.product:
                m.product = code; changed = True
            if photo and not m.images:
                save_material(m, [("legend_photo.png", photo)]); changed = False
            if changed:
                save_material(m)
            out.append(("updated", m))
    return out


def export_set(ids: list[str], name: str = "materials") -> bytes:
    """材種セット（メタ＋写真）を ZIP にして返す。案件ごとに保存・共有する用。"""
    import zipfile
    buf = io.BytesIO()
    metas = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for mid in ids:
            m = get_material(mid)
            if m is None:
                continue
            j = asdict(m); j["size_mm"] = list(m.size_mm); metas.append(j)
            for f in m.images:
                pth = m.dir / f
                if pth.exists():
                    z.write(pth, f"{m.id}/{f}")
        z.writestr("materials.json", json.dumps({"name": name, "exported": time.strftime("%Y-%m-%d %H:%M"), "materials": metas}, ensure_ascii=False, indent=1))
    return buf.getvalue()


def import_set(data: bytes, overwrite: bool = False) -> list[tuple[str, LibMaterial]]:
    """export_set の ZIP を読み込む。同じ品番（または id）があれば既定で飛ばす。"""
    import zipfile
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        meta = json.loads(z.read("materials.json").decode("utf-8"))
        items = list_materials()
        for j in meta.get("materials", []):
            j = {k: v for k, v in j.items() if k in LibMaterial.__dataclass_fields__}
            j["size_mm"] = tuple(j.get("size_mm") or (500.0, 500.0))
            m = LibMaterial(**j)
            exist = find_by_product(m.product or m.name, items) or get_material(m.id)
            if exist is not None and not overwrite:
                out.append(("skipped", exist)); continue
            imgs = []
            for f in list(m.images):
                try:
                    imgs.append((f, z.read(f"{m.id}/{f}")))
                except KeyError:
                    pass
            m.images = []
            m.source = m.source or "import"
            if exist is not None:
                m.id = exist.id
            m = save_material(m, imgs, replace_images=True)
            out.append(("imported", m))
    return out


def list_materials() -> list[LibMaterial]:
    out = []
    for d in sorted((lib_dir() / "materials").iterdir()):
        f = d / "meta.json"
        if f.exists():
            try:
                j = json.loads(f.read_text(encoding="utf-8"))
                j["size_mm"] = tuple(j.get("size_mm") or (500.0, 500.0))
                out.append(LibMaterial(**{k: v for k, v in j.items() if k in LibMaterial.__dataclass_fields__}))
            except Exception:
                continue
    return out


def get_material(mid: str) -> LibMaterial | None:
    return next((m for m in list_materials() if m.id == mid), None)


def save_material(m: LibMaterial, new_images: list[tuple[str, bytes]] | None = None, replace_images: bool = False) -> LibMaterial:
    if not m.id:
        m.id = f"{_slug(m.name)}_{int(time.time()) % 100000}"
    d = m.dir; d.mkdir(parents=True, exist_ok=True)
    if replace_images:
        for f in m.images:
            try:
                (d / f).unlink()
            except OSError:
                pass
        m.images = []
    for name, data in (new_images or []):
        if len(m.images) >= MAX_IMAGES:
            break
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if max(im.size) > MAX_SIDE:
            f = MAX_SIDE / max(im.size)
            im = im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))), Image.LANCZOS)
        fname = f"{len(m.images) + 1:02d}_{_slug(Path(name).stem)}.jpg"
        im.save(d / fname, "JPEG", quality=90)
        m.images.append(fname)
    m.updated = time.time()
    j = asdict(m); j["size_mm"] = list(m.size_mm)
    (d / "meta.json").write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    return m


def delete_material(mid: str) -> None:
    import shutil
    d = lib_dir() / "materials" / mid
    if d.exists():
        shutil.rmtree(d)


def load_images(m: LibMaterial) -> list[np.ndarray]:
    from floortint.textures import adjust_image
    out = []
    for f in m.images:
        p = m.dir / f
        if p.exists():
            out.append(adjust_image(np.array(Image.open(p).convert("RGB")), m.saturation, m.brightness))
    return out


# ---------------------------------------------------------------- 物件の設定
def list_projects() -> list[str]:
    return sorted(p.stem for p in (lib_dir() / "projects").glob("*.json"))


def save_project(name: str, settings: dict) -> Path:
    p = lib_dir() / "projects" / f"{_slug(name)}.json"
    settings = dict(settings); settings["saved"] = time.strftime("%Y-%m-%d %H:%M")
    p.write_text(json.dumps(settings, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return p


def load_project(name: str) -> dict | None:
    p = lib_dir() / "projects" / f"{_slug(name)}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def delete_project(name: str) -> None:
    p = lib_dir() / "projects" / f"{_slug(name)}.json"
    if p.exists():
        p.unlink()
