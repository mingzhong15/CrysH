"""Level 5 — motif tokenizer（contracts.md §3.8，签名/token 格式冻结）。

把每个原子的局域环境编码为 L1–L4 可哈希字符串 token：

    l1: "Ti|6"                        Z_symbol | CN
    l2: "Ti|6|oct"                    + geometry label
    l3: "Ti|6|oct|O6"                 + 邻居元素直方图（符号字母序 + 计数拼接）
    l4: "Ti|6|oct|O6|edge|2D"         + sharing_label + d_star（"0D".."3D"）

依赖注入（duck typing，本模块不 import crysh.bond/crysh.coord/crysh.geometry）：
  * graph  — 契约 BondGraph（§3.2，v1.1：**双向邻接**，含 (i,j,S) 与 (j,i,-S) 两方向）：
    只需 .i/.j（int 数组，(E,)）与可选 .n_atoms；本模块用 set 对称化去重，双向图安全。
  * coord  — 契约 CoordResult（§3.6）：只需 .cn（(N,) int）
  * geometry_labels — list[str]，长度 N，逐原子 geometry label

契约疑义的保守解释（详见 level5-tokenizer/progress.md）：
  * sharing 的"中心-中心对"= 两端均为多面体中心且共享 ≥1 个配体的原子对（由配体端
    枚举，O(E) 图操作，符合 plan.md "两 polyhedra 共享配体数" 的定义）；共享 0 个
    配体的中心对不计入分母。n_shared=1→corner，2→edge，≥3→face。
    **中心门槛（v1.3-L5 放宽）**：CN≥4，或 CN==3 且该站点 geometry label 为
    "trigonal_planar"（确认的平面三角形中心，见 _PLANAR_CENTER_GEO）。物理动机：
    BO₃ 硼酸盐 / CO₃ 碳酸盐网络的 corner-sharing 在纯 CN≥4 门槛下**完全不可见**
    （B/C 的 CN 恒为 3 → 这类网络的 sharing 恒 isolated）；linear（CN=2）与 CN=3
    的 "other"/"ambiguous"（T 形等）仍排除——只有确认的平面三角形才算多面体中心。
  * d_star：每次调用 try import crysh.bond，可用则用 dimensionality_spectrum(atoms).d_star
    真值（v1.1：d_star = d(λ=1.20)）；任何 import/计算失败则保守取 3（"3D"，已文档化）。
  * h_neigh = −Σ x_a ln x_a（邻居元素比例熵，单元素=0，无邻居=0）。
  * 邻居直方图始终附计数（"F2O4"，即使计数=1 也写 "…1"），保证 token 可逆无歧义；
    无邻居时直方图为空串（"Ti|0|other"）。

代码可移植：不 import 未冻结的模块，仅标准库 + numpy。
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

__all__ = ["motif_tokens", "TOKEN_LEVELS", "SHARING_CLASSES"]

TOKEN_LEVELS = ("l1", "l2", "l3", "l4")
SHARING_CLASSES = ("corner", "edge", "face")
_MIXED_MARGIN = 0.10  # 前两档占比差 < 10% → "mixed"
_CENTER_CN_MIN = 4    # 中心原子 CN 下限（polyhedra 中心）
# v1.3-L5 平面三角形中心例外：CN==3 且 geometry label ∈ 此集合的站点也算
# 多面体中心。物理动机：BO₃/CO₃（硼酸盐/碳酸盐）网络的 corner-sharing 在纯
# CN≥4 门槛下完全不可见——B/C 的配位恒为平面三角形 CN=3，网络共享性此前恒
# isolated。只认 "trigonal_planar"（确认的平面三角形）；linear CN=2 与 CN=3 的
# "other"/"ambiguous"（T 形等）仍排除。不依赖 GEOMETRY_LABELS 全集（label 集由
# level4 独立演进，本模块只做字符串成员判断）。
_PLANAR_CENTER_GEO = frozenset({"trigonal_planar"})


def _neighbor_sets(graph, n_atoms: int) -> list[set[int]]:
    """邻接集合（去重；i==j 的跨周期自像保留为同元素邻居）。"""
    i = np.asarray(graph.i, dtype=np.int64).ravel()
    j = np.asarray(graph.j, dtype=np.int64).ravel()
    nbrs: list[set[int]] = [set() for _ in range(n_atoms)]
    for a, b in zip(i.tolist(), j.tolist()):
        nbrs[a].add(b)
        if b != a:
            nbrs[b].add(a)
    return nbrs


def _resolve_d_star(atoms) -> int:
    """d_star 真值（来自 :mod:`crysh.dimensionality`）。

    R2 之前这里是 `try: from crysh.dimensionality import ... except: return 3`：
    模块改名后 import 一直失败，于是**每个结构的 l4 token 都静默写成 "3D"**。
    现在直接调用同包实现，只在真正算不出来时抛错（不再吞异常）。
    """
    from crysh.dimensionality import dimensionality_spectrum

    return int(dimensionality_spectrum(atoms).d_star)


def motif_tokens(atoms, graph, coord, geometry_labels: list[str],
                 d_star: int | None = None) -> dict:
    """逐原子 L1–L4 token + h_neigh + sharing（contracts.md §3.8）。

    Parameters
    ----------
    atoms : ase.Atoms
        结构（只读 chemical symbols）。
    graph : BondGraph（§3.2，或同字段 mock）
        需要 .i/.j；可选 .n_atoms。
    coord : CoordResult（§3.6，或同字段 mock）
        需要 .cn（(N,) int）。
    geometry_labels : list[str]
        逐原子 geometry label，长度必须等于原子数。
    d_star : int | None
        预计算的维度真值（v1.1 集成者增补，向后兼容）。None 时内部调
        ``crysh.dimensionality.dimensionality_spectrum`` 计算（每结构多算整个 λ 谱，
        批量调用方如 pipeline 应传入已算好的值以避免重复计算）。

    Returns
    -------
    dict：
        {"l1": [...], "l2": [...], "l3": [...], "l4": [...],
         "h_neigh": np.ndarray(N,), "sharing": {"corner": f, "edge": f, "face": f},
         "sharing_label": str}   # corner|edge|face|isolated|mixed

    sharing 中心门槛（v1.3-L5）：CN≥4，或 CN==3 且 geometry label ==
    "trigonal_planar"（BO₃/CO₃ 平面三角形网络物理缺口修复，见模块 docstring /
    _PLANAR_CENTER_GEO）。含 CN=3 平面中心的结构中，这类中心参与共享配体计数。
    """
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    cn = np.asarray(coord.cn, dtype=np.int64).ravel()
    if cn.size != n_atoms:
        raise ValueError(f"coord.cn 长度 {cn.size} != n_atoms {n_atoms}")
    if len(geometry_labels) != n_atoms:
        raise ValueError(
            f"geometry_labels 长度 {len(geometry_labels)} != n_atoms {n_atoms}"
        )
    _ = int(getattr(graph, "n_atoms", n_atoms))
    if _ != n_atoms:
        raise ValueError(f"graph.n_atoms {_} != len(atoms) {n_atoms}")

    nbrs = _neighbor_sets(graph, n_atoms)

    # ---- 逐原子 token + h_neigh ----
    l1: list[str] = []
    l2: list[str] = []
    l3: list[str] = []
    h_neigh = np.zeros(n_atoms, dtype=np.float64)
    for a in range(n_atoms):
        sym = symbols[a]
        c = int(cn[a])
        geo = str(geometry_labels[a])
        hist = Counter(symbols[b] for b in nbrs[a])
        hist_str = "".join(f"{el}{hist[el]}" for el in sorted(hist))
        base = f"{sym}|{c}"
        l1.append(base)
        l2.append(f"{base}|{geo}")
        l3.append(f"{base}|{geo}|{hist_str}")
        total = sum(hist.values())
        if total > 0:
            h = 0.0
            for count in hist.values():
                x = count / total
                h -= x * math.log(x)
            h_neigh[a] = h
        else:
            h_neigh[a] = 0.0

    # ---- polyhedral sharing：中心对（多面体中心）× 共享配体数 ----
    # v1.3-L5 门槛放宽：CN≥4，或 CN==3 且 geometry label 为 "trigonal_planar"
    # （BO₃/CO₃ 平面三角形网络的中心，见 _PLANAR_CENTER_GEO 物理动机）。linear
    # （CN=2）与 CN=3 的 "other"/"ambiguous" 仍排除。含 CN=3 平面中心的结构中，
    # 这类中心现在参与共享配体计数（sharing_label/f_corner/edge/face 语义）。
    is_center = [
        c >= _CENTER_CN_MIN or (c == 3 and str(g) in _PLANAR_CENTER_GEO)
        for c, g in zip(cn.tolist(), geometry_labels)
    ]
    pair_shared: Counter[tuple[int, int]] = Counter()
    for lig in range(n_atoms):
        centers = [b for b in nbrs[lig] if b != lig and is_center[b]]
        if len(centers) < 2:
            continue
        for idx in range(len(centers)):
            for jdx in range(idx + 1, len(centers)):
                c1, c2 = centers[idx], centers[jdx]
                pair_shared[(min(c1, c2), max(c1, c2))] += 1

    n_corner = n_edge = n_face = 0
    for n_shared in pair_shared.values():
        if n_shared == 1:
            n_corner += 1
        elif n_shared == 2:
            n_edge += 1
        else:  # ≥3 共享配体
            n_face += 1
    n_total = n_corner + n_edge + n_face
    if n_total > 0:
        f_corner = n_corner / n_total
        f_edge = n_edge / n_total
        f_face = n_face / n_total
        ranked = sorted(
            ((f_corner, "corner"), (f_edge, "edge"), (f_face, "face")),
            reverse=True,
        )
        if ranked[0][0] - ranked[1][0] < _MIXED_MARGIN:
            sharing_label = "mixed"
        else:
            sharing_label = ranked[0][1]
    else:
        f_corner = f_edge = f_face = 0.0
        sharing_label = "isolated"

    # ---- L4：+ sharing_label + d_star ----
    if d_star is None:
        d_star = _resolve_d_star(atoms)
    else:
        d_star = int(d_star)
        if not 0 <= d_star <= 3:
            raise ValueError(f"d_star must be in [0, 3], got {d_star}")
    l4 = [f"{t}|{sharing_label}|{d_star}D" for t in l3]

    return {
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "l4": l4,
        "h_neigh": h_neigh,
        "sharing": {"corner": f_corner, "edge": f_edge, "face": f_face},
        "sharing_label": sharing_label,
    }
