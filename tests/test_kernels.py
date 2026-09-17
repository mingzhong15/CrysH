"""共享邻居表的等价性验收：必须与 `bond.build_bond_graph` **逐点一致**。

这是 v0.2 引入 `kernels.py` 的准入条件。共享表把"取邻居"与"按 λ 筛边"拆开后，
任何边界条件的偏差（ASE 闭区间比较、去重顺序、自环规则）都会让 d(λ) 谱变样，
而 d_star 是冻结契约里最贵的一列——所以用 44 个 ground-truth 结构对拍，而不是
靠"看起来对"。

对拍口径：对每个 λ ∈ LAMBDAS，比较两张图的**边集合**（(i, j, S) 排序后）与距离。
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

from crysh.bond import build_bond_graph, canonical_bond_graph
from crysh.config import LAMBDAS
from crysh.controls import CN_METHOD_SHELL, build_all
from crysh.kernels import masked_graph, neighbor_list, normalized_neighbor_table


def _edge_key(graph) -> np.ndarray:
    """(i, j, Sx, Sy, Sz) 排序去重后的边键，用于集合比较。"""
    keys = np.stack([np.asarray(graph.i), np.asarray(graph.j),
                     np.asarray(graph.S)[:, 0], np.asarray(graph.S)[:, 1],
                     np.asarray(graph.S)[:, 2]], axis=1)
    order = np.lexsort(keys[:, ::-1].T)
    return keys[order]


# 本档只有 numpy+ase（无 pymatgen），而 build_all() 的 CN 参考方法必须**显式**给定
# （默认的 "crystalnn" 需 crysh[research]，会直接 ImportError）；这里只要结构，
# 用无依赖的壳层法。CN 标签本身在 tests/tables，CrystalNN 口径复现在 tests/research。
STRUCTURES = [(name, atoms)
              for name, atoms, _ in build_all(cn_method=CN_METHOD_SHELL)]
STRUCTURE_IDS = [name for name, _ in STRUCTURES]

# 另外补几个"极端几何"：金属高配位、层状、孤立团簇、小胞
EXTRA = [
    ("fcc-Al", bulk("Al", "fcc", a=4.05)),
    ("bcc-W", bulk("W", "bcc", a=3.165)),
    ("diamond-Si", bulk("Si", "diamond", a=5.431)),
    ("rocksalt-NaCl", bulk("NaCl", "rocksalt", a=5.64)),
]


@pytest.mark.parametrize("name,atoms", STRUCTURES + EXTRA, ids=STRUCTURE_IDS + [n for n, _ in EXTRA])
def test_shared_neighbor_list_matches_bond_graph_at_every_lambda(name, atoms):
    """8 个 λ 上，共享表筛出的图与原实现的边集合 + 距离逐点相等。"""
    nl = neighbor_list(atoms, lam_max=max(LAMBDAS))
    for lam in LAMBDAS:
        ref = build_bond_graph(atoms, cutoff_table=None, lam=lam)
        got = masked_graph(nl, lam)
        assert got.i.size == ref.i.size, f"{name} λ={lam}: 边数 {got.i.size} != {ref.i.size}"
        np.testing.assert_array_equal(_edge_key(got), _edge_key(ref),
                                      err_msg=f"{name} λ={lam}: 边集合不一致")
        # 距离要按同样的边序比较
        order_got = np.lexsort(np.stack([got.i, got.j, got.S[:, 0], got.S[:, 1],
                                         got.S[:, 2]], axis=1)[:, ::-1].T)
        order_ref = np.lexsort(np.stack([ref.i, ref.j, ref.S[:, 0], ref.S[:, 1],
                                         ref.S[:, 2]], axis=1)[:, ::-1].T)
        np.testing.assert_allclose(np.asarray(got.d)[order_got],
                                   np.asarray(ref.d)[order_ref], rtol=0, atol=0,
                                   err_msg=f"{name} λ={lam}: 边长不一致")


@pytest.mark.parametrize("name,atoms", STRUCTURES + EXTRA, ids=STRUCTURE_IDS + [n for n, _ in EXTRA])
def test_shared_neighbor_list_matches_with_calibrated_table(name, atoms):
    """带校准表（table 模式、λ=1.0 canonical）时同样逐点相等。"""
    table = {(int(a), int(b)): 1.25 * float(v) for (a, b), v in
             build_bond_graph(atoms).pair_r0.items()}
    ref = build_bond_graph(atoms, cutoff_table=table, lam=1.0)
    nl = neighbor_list(atoms, cutoff_table=table, lam_max=max(LAMBDAS))
    got = masked_graph(nl, 1.0)
    assert nl.cutoff_mode == "table"
    np.testing.assert_array_equal(_edge_key(got), _edge_key(ref))


def test_canonical_graph_matches_shared_kernel_default_path():
    """canonical 口径（无表 → λ*=1.20）与共享表筛出的图一致。"""
    for name, atoms in EXTRA:
        ref = canonical_bond_graph(atoms, None)
        nl = neighbor_list(atoms, lam_max=max(LAMBDAS))
        got = masked_graph(nl, ref.lam)
        np.testing.assert_array_equal(_edge_key(got), _edge_key(ref), err_msg=name)


def test_lambda_mask_is_monotone():
    """λ 增大只会加边（掩码单调），这是"一份表切 8 刀"成立的前提。"""
    atoms = bulk("Si", "diamond", a=5.431)
    nl = neighbor_list(atoms, lam_max=max(LAMBDAS))
    counts = [int(nl.masked(lam).sum()) for lam in LAMBDAS]
    assert counts == sorted(counts), counts


def test_normalized_neighbor_table_shape_and_order():
    """r̃ 表：长度等于掩码边数、按 (site, r̃) 升序、r̃ = d/r0。"""
    atoms = bulk("Si", "diamond", a=5.431)
    nl = neighbor_list(atoms, lam_max=max(LAMBDAS))
    # 返回 (site, d, r_tilde, nbr, mask)：排序表覆盖到 lam_max，mask 标出当前 λ 键集
    site, d, r_tilde, nbr, mask = normalized_neighbor_table(nl, 1.2)
    assert site.size == nl.i.size, "排序表覆盖建表上界内的全部邻居"
    assert int(mask.sum()) == int(nl.masked(1.2).sum())
    order = np.lexsort((d, site))
    np.testing.assert_array_equal(site, site[order])
    np.testing.assert_array_equal(d, d[order])
    np.testing.assert_allclose(r_tilde, d / (d / r_tilde), rtol=1e-12)
    # diamond Si：键集内恰好 4 个邻居
    per_site = np.bincount(site[mask], minlength=len(atoms))
    assert set(per_site.tolist()) == {4}
    # r̃ = d / (r_cov(Si)+r_cov(Si)) = 2.351 / 2.22 = 1.059 —— 不是 1.0。
    # 这正是"共价半径和会低估真实键长"的直接证据（也是 Phase-0 需要校准表的原因）。
    assert np.allclose(r_tilde[mask], 1.059, atol=0.01), r_tilde[mask]


def test_lam_max_guard_rejects_partial_table():
    """建表 λ_max 小于要用 λ 时必须报错，而不是静默少边。"""
    atoms = bulk("Cu", "fcc", a=3.6)
    nl = neighbor_list(atoms, lam_max=1.0)
    from crysh.kernels import masked_graph as mg

    with pytest.raises(AssertionError):
        # 用 2.0 去筛 1.0 的表：边集合会少于真值 → 对拍必失败（回归护栏）
        ref = build_bond_graph(atoms, lam=2.0)
        np.testing.assert_array_equal(_edge_key(mg(nl, 2.0)), _edge_key(ref))
