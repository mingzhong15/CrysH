"""共享内核：**一份邻居表，处处复用**。

动机（重构前实测）：同一个结构在一级流水线里被反复建邻居表——
`bond.build_bond_graph` 每被调一次就 `ase.neighbor_list` 一次，而 L1 的 d(λ) 谱要
8 个 λ（=8 次），L2/L3/L4/L5 各自还会再建。这份内核把"取邻居"与"按 λ 筛边"拆开：

    nl = neighbor_list(atoms, cfg)        # 一次，cutoff = λ_max · r0(pair)
    graph(λ) = masked_graph(nl, atoms, λ) # 纯数组筛选，O(E)，无 ASE 调用

并可在此基础上直接做壳层分析（L3 的 m_i 需要按归一化距离排序的邻居）。

设计约束
--------
- **数值必须与 `bond.build_bond_graph` 逐点一致**：同一 cutoff 表、同一
  `d < λ·r0` 严格判据、同样的去重与双向邻接语义（契约 §3.2）。
  `tests/test_kernels.py` 用 44 个 ground-truth 结构对拍钉住这一点。
- 本模块不做物理判断（不判定几何、不算 CN），只提供"邻居关系"这一层事实。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list as _ase_neighbor_list

from crysh.bond import BondGraph, default_covalent_table

__all__ = ["NeighborList", "neighbor_list", "masked_graph", "normalized_neighbor_table"]


@dataclass
class NeighborList:
    """某个结构在 `lam_max` 下的全部有向邻居对（ASE `ijdS` 语义）。

    `i/j/S/d` 与 :class:`crysh.bond.BondGraph` 同语义：`j` 位于
    `i + S·cell` 的周期像里；**双向都保留**（`(i,j,S)` 与 `(j,i,-S)` 是两条）。
    `r0` 是**未缩放**的 pair 参考键长（校准表值或共价半径和），因此
    "λ 筛边"就是一次纯数组比较 `d < lam * r0`。
    """

    i: np.ndarray      # (E,) int32
    j: np.ndarray      # (E,) int32
    S: np.ndarray      # (E,3) int32
    d: np.ndarray      # (E,) float64
    r0: np.ndarray     # (E,) float64 — pair 参考键长（未乘 λ）
    n_atoms: int
    lam_max: float
    cutoff_mode: str   # "covalent" | "table"
    pair_r0: dict      # {(Zi, Zj): r0}

    def __len__(self) -> int:
        return int(self.i.size)

    def masked(self, lam: float) -> np.ndarray:
        """`d < lam·r0` 的布尔掩码（严格小于，与契约 §3.2 一致）。"""
        return self.d < float(lam) * self.r0


def _resolve_r0(atoms: Atoms, cutoff_table: dict | None) -> tuple[dict, str]:
    """pair 参考键长表 `{(Zi, Zj): r0}`（sorted 键）与来源标记。"""
    cov = default_covalent_table(atoms)
    if cutoff_table is None:
        return cov, "covalent"
    norm: dict = {}
    for key, val in cutoff_table.items():
        zi, zj = int(key[0]), int(key[1])
        val = float(val)
        if not val > 0:
            raise ValueError(f"cutoff_table entry {key} must be > 0, got {val}")
        norm[(zi, zj) if zi <= zj else (zj, zi)] = val
    return {pair: norm.get(pair, cov[pair]) for pair in cov}, "table"


def neighbor_list(atoms: Atoms, cutoff_table: dict | None = None,
                  lam_max: float = 2.0) -> NeighborList:
    """一次建表：`cutoff = lam_max · r0(pair)`，保留 `(i, j, S, d, r0)`。

    λ 网格内的任意子图都可由 `masked_graph(nl, lam)` 得到，无需再调 ASE。

    Parameters
    ----------
    atoms:
        结构（周期性；`pbc=False` 时 ASE 只给团簇内邻居，语义自然退化）。
    cutoff_table:
        校准表 `{(Zi, Zj): r0}`；`None` 用共价半径和（契约 §3.1 的 fallback）。
    lam_max:
        建表时用的最大 λ（默认 2.0 = 契约 λ 网格上界）。所有要用的 λ 必须 ≤ 它。
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(f"atoms must be an ase.Atoms, got {type(atoms).__name__}")
    lam_max = float(lam_max)
    if not lam_max > 0:
        raise ValueError(f"lam_max must be > 0, got {lam_max}")

    r0_by_pair, mode = _resolve_r0(atoms, cutoff_table)
    r_cut = {pair: lam_max * r for pair, r in r0_by_pair.items()}
    i, j, d, S = _ase_neighbor_list("ijdS", atoms, r_cut)
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    d = np.asarray(d, dtype=np.float64)
    S = np.asarray(S, dtype=np.int64).reshape(-1, 3)

    # 排除 home-cell 自环（i == j 且 S == 0），与 bond.build_bond_graph 同规则
    keep = ~((i == j) & np.all(S == 0, axis=1))
    i, j, d, S = i[keep], j[keep], d[keep], S[keep]

    # 去重 EXACT (i, j, S)；双向邻接保留
    keys = np.stack([i, j, S[:, 0], S[:, 1], S[:, 2]], axis=1)
    keys, idx = np.unique(keys, axis=0, return_index=True)
    ii, jj, SS, dd = keys[:, 0], keys[:, 1], keys[:, 2:], d[idx]

    z = np.asarray(atoms.numbers, dtype=np.int64)
    zlo = np.minimum(z[ii], z[jj])
    zhi = np.maximum(z[ii], z[jj])
    r0 = np.array([r0_by_pair[(int(a), int(b))] for a, b in zip(zlo, zhi)])

    # 与 bond 一致的兜底：ASE 的边界比较可能是闭区间，这里按 λ_max 口径再筛一次
    keep2 = dd < lam_max * r0
    return NeighborList(
        i=ii[keep2].astype(np.int32), j=jj[keep2].astype(np.int32),
        S=SS[keep2].astype(np.int32), d=dd[keep2].astype(np.float64),
        r0=r0[keep2].astype(np.float64), n_atoms=len(atoms),
        lam_max=lam_max, cutoff_mode=mode, pair_r0=r0_by_pair,
    )


def masked_graph(nl: NeighborList, lam: float) -> BondGraph:
    """把共享邻居表按 λ 筛成一张 :class:`crysh.bond.BondGraph`（纯数组操作）。"""
    m = nl.masked(lam)
    return BondGraph(
        i=nl.i[m].copy(), j=nl.j[m].copy(), S=nl.S[m].copy(), d=nl.d[m].copy(),
        n_atoms=nl.n_atoms, lam=float(lam), cutoff_mode=nl.cutoff_mode,
        pair_r0=nl.pair_r0,
    )


def normalized_neighbor_table(nl: NeighborList, lam: float
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按距离排序的逐位点邻居表 + "是否在当前 λ 键集内"的掩码。

    为什么要带掩码、而不是只返回 λ 内的边：**壳层边界恰恰在最后一条键的外侧**。
    只给壳内邻居时，理想结构（壳内距离全相等）的相邻差分全是 0，找不到边界
    （实测：diamond Si / fcc Al / NaCl 的 `shell_gap` 都算成 0）。
    所以排序表覆盖到建表上界 `lam_max`，用掩码标出哪些属于当前 λ，
    调用方在 `mask` 的跳变处就能看到真实的第一壳边界。

    Returns
    -------
    (site, d, r_tilde, nbr, mask)
        五个等长数组，按 `(site, d)` 升序：

        - `site`：中心原子索引（"从 i 看 j"这一侧，有向）；
        - `d`：几何距离（Å）；
        - `r_tilde = d / r0`：无量纲距离，1.0 恰好是共价键长；
        - `nbr`：邻居原子索引（`j`），与前三者同序 —— `symbols[nbr]` 即邻居元素；
        - `mask`：该邻居是否在 λ 键集内（`d < λ·r0`）。

    `mask.sum()` 就是 λ 下的**有向邻居数**（`np.bincount(site[mask], minlength=n)`
    与 ``coord._count_cn`` 的去重口径一致；该函数对 i==j 的自像另有处理）。
    """
    m = nl.masked(lam)
    site = nl.i.astype(np.int64)
    nbr = nl.j.astype(np.int64)
    r_tilde = (nl.d / nl.r0).astype(np.float64)
    d = nl.d.astype(np.float64)
    order = np.lexsort((d, site))
    return site[order], d[order], r_tilde[order], nbr[order], m[order]
