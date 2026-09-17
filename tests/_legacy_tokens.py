"""参考实现：v1.3-L5 改造前 `motif_tokens` 的逐字快照（测试专用，不进库）。

来源：研究工程 `code/levels/level5-tokenizer/scripts/regress_2k_local.py`
（`_LEGACY_MIXED_MARGIN` / `_neighbor_sets_legacy` / `motif_tokens_legacy`）。
用途：`test_planar_sharing` 用它验证新实现与旧实现在退化情形下逐点一致。
等价于把"旧版本"钉在测试里，而不是让测试去 import 一个回归脚本。
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

_LEGACY_MIXED_MARGIN = 0.10
_LEGACY_CENTER_CN_MIN = 4



def _neighbor_sets_legacy(graph, n_atoms: int) -> list[set[int]]:
    """改造前 tokenize._neighbor_sets 的逐字拷贝。"""
    i = np.asarray(graph.i, dtype=np.int64).ravel()
    j = np.asarray(graph.j, dtype=np.int64).ravel()
    nbrs: list[set[int]] = [set() for _ in range(n_atoms)]
    for a, b in zip(i.tolist(), j.tolist()):
        nbrs[a].add(b)
        if b != a:
            nbrs[b].add(a)
    return nbrs


def motif_tokens_legacy(atoms, graph, coord, geometry_labels: list[str],
                        d_star: int | None = None) -> dict:
    """v1.3-L5 改造前 motif_tokens 的逐字快照（CN≥4 中心门槛原逻辑拷贝）。"""
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    cn = np.asarray(coord.cn, dtype=np.int64).ravel()
    if cn.size != n_atoms:
        raise ValueError(f"coord.cn 长度 {cn.size} != n_atoms {n_atoms}")
    if len(geometry_labels) != n_atoms:
        raise ValueError(
            f"geometry_labels 长度 {len(geometry_labels)} != n_atoms {n_atoms}")
    _ = int(getattr(graph, "n_atoms", n_atoms))
    if _ != n_atoms:
        raise ValueError(f"graph.n_atoms {_} != len(atoms) {n_atoms}")

    nbrs = _neighbor_sets_legacy(graph, n_atoms)

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

    # —— 旧中心门槛：纯 CN≥4（v1.3-L5 改造点）——
    is_center = cn >= _LEGACY_CENTER_CN_MIN
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
        else:
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
        if ranked[0][0] - ranked[1][0] < _LEGACY_MIXED_MARGIN:
            sharing_label = "mixed"
        else:
            sharing_label = ranked[0][1]
    else:
        f_corner = f_edge = f_face = 0.0
        sharing_label = "isolated"

    if d_star is None:
        d_star = 3
    else:
        d_star = int(d_star)
        if not 0 <= d_star <= 3:
            raise ValueError(f"d_star must be in [0, 3], got {d_star}")
    l4 = [f"{t}|{sharing_label}|{d_star}D" for t in l3]

    return {
        "l1": l1, "l2": l2, "l3": l3, "l4": l4, "h_neigh": h_neigh,
        "sharing": {"corner": f_corner, "edge": f_edge, "face": f_face},
        "sharing_label": sharing_label,
    }


