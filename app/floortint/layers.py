"""PDF のレイヤー（Optional Content Group）で図形を仕分ける。

CAD から出した平面図 PDF は、寸法・通り芯・ワークポイント・人物・什器・植栽・躯体がレイヤーに分かれている。
色で判定するより確実なので、レイヤーがあるときはこちらを優先し、色の規則は使わない。
レイヤー名は会社・担当者ごとに違うため、名前の語でカテゴリに分類し、既定の「落とす／残す」を決める。
画面で一覧を出し、個別に反転できる（議事録: 「ある程度自動で落とされるけど、復活もできる」）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from floortint.pdfdoc import touches

# カテゴリ: drop（注記類。既定で落とす）/ plant（植栽）/ structure（躯体）/ hatch（ハッチ。既定で落とす）/ keep（什器・建具など）
# 上から順に当てる。先に当たった規則が勝つ。
RULES: list[tuple[str, str, str]] = [
    # (カテゴリ, 表示名, 正規表現。大文字小文字を区別しない)
    ("plant", "植栽", r"植栽|植物|moss|plant|green|樹木"),
    ("drop", "文字・記号", r"text|moji|文字|室名|roomname|面積|注記|記号|kigo|hyouki|yaji|矢印|雲マーク|name_|名称|図面名|凡例"),
    ("drop", "寸法", r"dim\b|dim[_\-]|寸法|sunpou|(?<![一-龥])寸(?![一-龥])"),
    ("drop", "通り芯", r"center|通り?芯|通心|grid|axis|kijun|基準線|通芯|通リ芯"),
    ("drop", "割付・投影線", r"(?<![A-Za-z])OA(?![A-Za-z])|割付|わりつけ|投影|projector|プロジェクタ|ﾌﾟﾛｼﾞｪｸﾀ|タイル割|ﾀｲﾙ割|carpet.?grid"),
    ("drop", "ワークポイント・席マーカー", r"\bwp\b|wp[_\-]|okm_wp|ワークポイント|work.?point|point|ポイント|(?<![一-龥])席"),
    ("drop", "図枠・表題", r"waku|図枠|(?<![一-龥])枠|title|表題|okamura\b"),
    ("drop", "人物", r"人物|^人$|person|people|human"),
    ("drop", "設備・防災・セキュリティ", r"security|セキュリティ|消防|消火|防災|誘導灯|電灯|インターホン|入退出|fireline|カードリーダー|換気|ランプ|ﾗﾝﾌﾟ|電気|天井|ceiling|tenjo|天伏"),
    ("drop", "敷地・外構・隠線", r"敷地|境界|sikichi|外構|gaibu|hide|非表示|展開図|外部"),
    ("keep", "建具・階段・什器", r"建具|ドア|door|窓|window|階段|stair|手摺|furn|kagu|家具|什器|shikiri|panel|zousaku|造作|\bav\b|kiki|機器|floor|床上げ|スロープ|点字"),
    ("hatch", "ハッチ", r"hatch|ハッチ"),
    ("structure", "躯体", r"structure|躯体|kutai|\bstr\b|c-str|(?<![一-龥])柱|(?<![一-龥])壁|slab|スラブ|梁|brace|ブレース|間柱|_rc\b|rc_|rc・|rc\(|alc|lgs|wall|column|f1\d\d|f2\d\d|m1\d\d|m2\d\d|new_m2"),
]
_COMPILED = [(cat, label, re.compile(pat, re.IGNORECASE)) for cat, label, pat in RULES]
CATEGORY_LABEL = {"drop": "注記（既定で落とす）", "hatch": "ハッチ（既定で落とす）", "plant": "植栽", "structure": "躯体", "keep": "什器・建具・その他（残す）"}
CATEGORY_ORDER = ["keep", "structure", "plant", "hatch", "drop"]
_SUFFIX = re.compile(r"_ペン_番号__\d+|_番号__\d+|\(平面非表示\)|\.●|\s+$")


def normalize(name: str) -> str:
    """ARCHICAD の「_ペン_番号__46」などの装飾を外す。"""
    return _SUFFIX.sub("", name)


def categorize(name: str | None) -> tuple[str, str]:
    """レイヤー名 → (カテゴリ, 根拠の表示名)。名前が無い図形は keep。"""
    if not name:
        return "keep", "レイヤー無し"
    n = normalize(name)
    for cat, label, rx in _COMPILED:
        if rx.search(n):
            return cat, label
    return "keep", "その他"


@dataclass
class LayerInfo:
    name: str
    count: int
    category: str
    reason: str
    drop: bool                       # 既定（カテゴリから）または画面での上書き

    def to_dict(self) -> dict:
        return {"name": self.name, "count": self.count, "category": self.category, "reason": self.reason, "drop": self.drop}


@dataclass
class LayerPlan:
    """このページのレイヤー一覧と、落とす／植栽／躯体の集合。"""
    layers: list[LayerInfo] = field(default_factory=list)
    has_layers: bool = False

    @property
    def drop_set(self) -> set[str]:
        return {li.name for li in self.layers if li.drop}

    @property
    def plant_set(self) -> set[str]:
        return {li.name for li in self.layers if li.category == "plant" and not li.drop}

    @property
    def structure_set(self) -> set[str]:
        return {li.name for li in self.layers if li.category == "structure" and not li.drop}

    def summary(self) -> dict:
        by = {}
        for li in self.layers:
            by.setdefault(li.category, []).append(f"{li.name}({li.count})")
        return {"has_layers": self.has_layers, "dropped": sorted(self.drop_set), "by_category": by}


def inventory(drawings, overrides: dict[str, bool] | None = None, zone=None) -> LayerPlan:
    """図形の列からレイヤー一覧を作る。overrides は {レイヤー名: 落とすか} の上書き。zone があれば図面部内の図形だけ数える。"""
    counts: dict[str, int] = {}
    for d in drawings:
        if zone is not None and not touches(d.rect, zone):
            continue
        counts[d.layer or ""] = counts.get(d.layer or "", 0) + 1
    named = [k for k in counts if k]
    plan = LayerPlan(has_layers=len(named) >= 2 and sum(counts[k] for k in named) >= 300)   # 無名の図形が多くても、名前つきが十分あれば使う
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        cat, reason = categorize(name)
        drop = cat in ("drop", "hatch")
        if overrides and name in overrides:
            drop = bool(overrides[name])
        plan.layers.append(LayerInfo(name, n, cat, reason, drop))
    return plan
