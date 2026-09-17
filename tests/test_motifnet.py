"""L4 motif 超节点图：可核验的性质 + 分类的适用域。

这个模块的设计取舍比算法本身重要（见 `crysh/motifnet.py` 的 docstring）：

- **连接图与维数**是良定义的（图的连通性与平移环的整数秩），任何体系都能算，本文件钉死；
- **corner/edge/face** 只在"真桥"（配体恰接 2 个中心）时有意义。密堆金属与岩盐型
  离子晶体不满足前提 —— 岩盐相邻八面体的 4 个共享 Cl 在空间上张成四边形，
  既不是边也不是面。这类情形如实给 `"n/a"`，本文件同样钉死。
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from crysh.motifnet import (
    LIGAND_BRIDGE,
    LIGAND_SHARED,
    SHARING_NA,
    _integer_rank,
    motif_network,
)


# --------------------------------------------------------------------------- #
# 秩机器（精确整数行阶梯化）
# --------------------------------------------------------------------------- #
def test_integer_rank_basics():
    z = np.zeros(3, dtype=np.int64)
    e1 = np.array([1, 0, 0], dtype=np.int64)
    e2 = np.array([0, 1, 0], dtype=np.int64)
    assert _integer_rank([]) == 0
    assert _integer_rank([z, z]) == 0
    assert _integer_rank([e1, e1 * 3]) == 1
    assert _integer_rank([e1, e2]) == 2
    assert _integer_rank([e1, e2, e1 + e2]) == 2
    assert _integer_rank([e1, e2, np.array([0, 0, 1])]) == 3


def test_integer_rank_is_exact_at_large_magnitudes():
    """定点化后的平移可能到 1e5 量级；浮点 SVD 会给出无意义的秩（实测踩过 7）。"""
    v = np.array([100000, 100000, 0], dtype=np.int64)
    w = np.array([200000, 200000, 0], dtype=np.int64)
    assert _integer_rank([v, w]) == 1
    assert _integer_rank([v, np.array([0, 0, 100000])]) == 2


# --------------------------------------------------------------------------- #
# 连接图与维数（良定义，任何体系都该对）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,atoms", [
    ("diamond-Si", bulk("Si", "diamond", a=5.431)),
    ("fcc-Al", bulk("Al", "fcc", a=4.05)),
    ("bcc-W", bulk("W", "bcc", a=3.165)),
    ("rocksalt-NaCl", bulk("NaCl", "rocksalt", a=5.64)),
    ("zincblende-ZnS", bulk("ZnS", "zincblende", a=5.41)),
])
def test_dimension_agrees_with_the_frozen_L1_spectrum(name, atoms):
    """**跨模块一致性**：motifnet 的维数必须等于 L1 已冻结的 d_star。

    这是本模块最强的一条验收：L1 的 `dimensionality_spectrum` 是原胞上的整数秩
    （无边界截断），motifnet 的维数来自超胞节点图上的平移秩——两者独立实现却必须
    给出同一个数。实测五个体系全等。
    """
    from crysh.dimensionality import dimensionality_spectrum

    net = motif_network(atoms)
    assert net.dim == dimensionality_spectrum(atoms).d_star == 3, name
    assert net.nodes, "至少要认出一个多面体中心"


def test_component_count_is_a_local_view_not_the_periodic_truth():
    """分量数是**局部视角**：有限超胞会截断跨边界键，故可能比真实周期网络多。

    金刚石真实周期网络是 1 个连通分量，但任何有限超胞上都会在边界处断成 2 个
    （两个子晶格在边界不再相接）。这里如实钉住这个已知偏差——精确值要用
    :func:`crysh.dimensionality.dimensionality_spectrum`。
    """
    net = motif_network(bulk("Si", "diamond", a=5.431))
    from crysh.dimensionality import dimensionality_spectrum

    assert net.n_components == 2, "有限超胞的已知截断偏差"
    assert dimensionality_spectrum(bulk("Si", "diamond", a=5.431)).n_components == 1


def test_isolated_molecule_has_no_bridges():
    """孤立团簇：多面体之间没有共享配体 → 0 维、无桥。"""
    atoms = Atoms("Ti" + "O" * 6,
                  positions=[[0, 0, 0], [1.95, 0, 0], [-1.95, 0, 0], [0, 1.95, 0],
                             [0, -1.95, 0], [0, 0, 1.95], [0, 0, -1.95]],
                  cell=np.eye(3) * 20.0, pbc=True)
    # 显式不展开：真空盒里的孤立团簇，展开只会复制出互不相连的副本
    net = motif_network(atoms, cell_matrix=np.eye(3, dtype=int))
    assert len(net.nodes) == 1, "只有中心的 O6 八面体"
    assert not net.edges
    assert net.dim == 0
    assert net.n_components == 1


def test_family_dims_separate_sublattices():
    """per-family 维数：岩盐里 Na 亚晶格与 Cl 亚晶格各自都是 3D。"""
    net = motif_network(bulk("NaCl", "rocksalt", a=5.64))
    assert set(net.family_dims) == {"Na", "Cl"}
    assert all(v == 3 for v in net.family_dims.values())


def test_supercell_size_is_recorded_and_configurable():
    """超胞规模影响计数（不影响维数），必须记进 meta 可追溯。"""
    atoms = bulk("Si", "diamond", a=5.431)
    n1 = motif_network(atoms)                      # 默认 = 原胞
    n2 = motif_network(atoms, cell_matrix=np.diag([2, 2, 2]))
    n2x2 = motif_network(atoms, cell_matrix=np.eye(3, dtype=int))
    assert n2x2.meta["cell_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert n1.meta["auto_expanded"] is True, "2 原子胞应自动展开"
    assert n2.meta["cell_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert n2.meta["n_atoms_supercell"] == 8 * len(atoms)  # 2×2×2
    assert n1.dim == n2.dim == 3, "维数是拓扑量，不该随超胞变"
    assert n2x2.meta["n_atoms_supercell"] == len(atoms)


def test_input_atoms_is_not_mutated():
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    before = atoms.positions.copy()
    motif_network(atoms)
    np.testing.assert_array_equal(atoms.positions, before)


# --------------------------------------------------------------------------- #
# 分类的适用域（本模块的核心主张）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,atoms", [
    ("fcc-Al", bulk("Al", "fcc", a=4.05)),
    ("bcc-W", bulk("W", "bcc", a=3.165)),
    ("rocksalt-NaCl", bulk("NaCl", "rocksalt", a=5.64)),
])
def test_dense_solids_are_reported_as_not_applicable(name, atoms):
    """密堆体系：配体被多个多面体共用 → corner/edge/face 不适用，如实给 n/a。

    硬塞一个标签比给 n/a 更糟：岩盐相邻八面体共享 4 个 Cl，按计数会写成 "face"，
    而那 4 个 Cl 在空间上张成的是四边形（既不是边也不是面）。
    """
    net = motif_network(atoms)
    assert net.sharing_fractions, "应该有边"
    assert net.sharing_fractions.get(SHARING_NA, 0.0) > 0.9, net.sharing_fractions


def test_diamond_has_some_classifiable_bridges():
    """金刚石：至少有一部分边落在"真桥"上并被分类（其余如实 n/a）。"""
    net = motif_network(bulk("Si", "diamond", a=5.431))
    kinds = net.meta["ligand_kinds"]
    assert kinds.get(LIGAND_BRIDGE, 0) > 0
    assert kinds.get(LIGAND_SHARED, 0) > 0
    classified = {k: v for k, v in net.sharing_fractions.items() if k != SHARING_NA}
    assert classified, net.sharing_fractions


def test_edge_rows_carry_interface_span():
    """每条边都带界面跨度（共享配体张多开）：单配体为 0，多配体 > 0。"""
    net = motif_network(bulk("NaCl", "rocksalt", a=5.64))
    assert net.edges
    for e in net.edges:
        assert e.interface_span >= 0.0
        if e.n_shared == 1:
            assert e.interface_span == 0.0
        else:
            assert e.interface_span > 0.0


def test_edges_top_groups_by_family_pair():
    net = motif_network(bulk("NaCl", "rocksalt", a=5.64))
    top = net.edges_top(3)
    assert top and all("--" in key for key, _ in top)
    assert all(cnt > 0 for _, cnt in top)
