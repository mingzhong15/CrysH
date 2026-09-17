"""Level 3 小胞镜像混叠守卫单测（v1.3-L3 增补：cn_cell_warning / cn_cell_margin）。

契约 §3.6 冻结签名不动；本文件只测纯增补守卫字段的语义。核心断言：
**warning ≠ 错误**——CN 的 (j,S) 计数在混叠区仍严格正确（每个 (j,S) 是唯一
物理位置）。内联 fixture + mock BondGraph（§3.2 字段），不依赖 ckt.bond
与本地数据。

margin = 0.5·min(|a|,|b|,|c|) / max_pair_cutoff，max_pair_cutoff =
graph.lam × max(graph.pair_r0.values())；<1 → warning=True（≥1 无混叠）。
"""
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.neighborlist import neighbor_list

from crysh.coord import coordination


@dataclass
class MockBondGraph:
    """契约 §3.2 BondGraph 字段的测试 mock（level1 未合入时的替身）。"""
    i: np.ndarray
    j: np.ndarray
    S: np.ndarray
    d: np.ndarray
    n_atoms: int
    lam: float = 1.0
    cutoff_mode: str = "table"
    pair_r0: dict = field(default_factory=dict)


def make_graph(atoms, cutoff, lam=1.0, pair_r0=None, self_interaction=False):
    """用 ase.neighbor_list 造 mock 键图（"ijdS"）。

    cutoff 与 lam × max(pair_r0) 保持一致（守卫从 graph.lam/pair_r0 推
    max_pair_cutoff，与实际枚举 cutoff 相同才有可比性）。
    """
    i, j, d, S = neighbor_list(
        "ijdS", atoms, cutoff, self_interaction=self_interaction)
    mask = ~((i == j) & (S == 0).all(axis=1))  # 排除 i==j 且 S==0 的自环（§3.2）
    i, j, d, S = i[mask], j[mask], d[mask], S[mask]
    return MockBondGraph(
        i=i.astype(np.int32), j=j.astype(np.int32), S=S.astype(np.int32),
        d=d.astype(np.float64), n_atoms=len(atoms), lam=lam,
        cutoff_mode="table", pair_r0=dict(pair_r0) if pair_r0 else {})


def test_normal_cell_no_warning():
    """① 常规胞：Cu fcc 常规胞 2×2×2（a=7.2）cutoff 2.6 → margin>1、无警告、CN=12。

    注：a=3.6 的单胞本就在混叠区（0.5a=1.8 < 2.6，见
    test_single_cell_fcc_flagged），故用 2×2×2 超胞展示无混叠区：
    半胞 3.6 > cutoff 2.6 > 最近邻 2.546 > 次近邻被排除。
    """
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    graph = make_graph(atoms, 2.6, lam=1.0, pair_r0={(29, 29): 2.6})
    res = coordination(atoms, graph)
    assert res.cn_cell_warning is False
    assert res.cn_cell_margin == pytest.approx(0.5 * 7.2 / 2.6)  # ≈ 1.385
    assert res.cn_cell_margin > 1.0
    assert res.cn.shape == (32,)
    assert np.all(res.cn == 12)                 # fcc 最近邻 2.546 < 2.6 < 3.6
    assert res.mean_cn == pytest.approx(12.0)


def test_single_cell_fcc_flagged():
    """补充：Cu fcc 常规单胞（a=3.6）cutoff 2.6 在混叠区（0.5a=1.8 < 2.6）
    → warning=True、margin<1；但 CN 计数仍严格正确（12）——角原子的 12 个
    最近邻是 3 个面心指数 × 4 个周期镜像，ASE 全枚举、(j,S) 去重后恰为 12。
    warning ≠ 错误的直接证据。"""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    graph = make_graph(atoms, 2.6, lam=1.0, pair_r0={(29, 29): 2.6})
    res = coordination(atoms, graph)
    assert res.cn_cell_warning is True
    assert res.cn_cell_margin == pytest.approx(0.5 * 3.6 / 2.6)  # ≈ 0.692
    assert res.cn_cell_margin < 1.0
    assert np.all(res.cn == 12)                 # 计数不受警告影响


def test_aliasing_chain_warning_but_count_correct():
    """② 混叠胞：双原子 C 链 a=2.0、cov 表 λ=1.2（r0=1.52 → cutoff 1.824 ≥
    0.5a=1.0）→ warning=True、margin<1；CN 与手算 (j,S) 期望一致。

    手算：atom0(x=0) 的近邻 = (j=1, S=0)@1.0Å 与 (j=1, S=-x)@1.0Å → CN=2；
          atom1(x=1) 的近邻 = (j=0, S=0) 与 (j=0, S=+x) → CN=2。
    同一索引 j=1 的两个镜像（左/右邻）都是真实物理近邻（周期链间距 1.0），
    计数正确——warning 只标记"化学邻居数"解读进入混叠区。
    """
    atoms = Atoms("CC", positions=[[0, 0, 0], [1.0, 0, 0]],
                  cell=[[2.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], pbc=True)
    r0 = 0.76 + 0.76                    # C 共价和（cov 表 pair_r0，未乘 λ）
    lam = 1.2
    cutoff = lam * r0                    # 1.824
    graph = make_graph(atoms, {(6, 6): cutoff}, lam=lam, pair_r0={(6, 6): r0})
    res = coordination(atoms, graph)
    assert res.cn_cell_warning is True
    assert res.cn_cell_margin == pytest.approx(0.5 * 2.0 / cutoff)  # ≈ 0.548
    assert res.cn_cell_margin < 1.0
    assert res.cn.tolist() == [2, 2]     # 与手算 (j,S) 期望一致
    assert res.cn_min == 2 and res.cn_max == 2


def test_missing_pair_r0_conservative():
    """③ graph 缺 .pair_r0（或空 dict）→ margin=+inf、warning=False（保守不误报）。

    即便胞本身小（同 ② 的混叠链胞），cutoff 信息缺失时也不误报。
    """
    atoms = Atoms("CC", positions=[[0, 0, 0], [1.0, 0, 0]],
                  cell=[[2.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], pbc=True)
    # (a) 只有 .i/.j/.S/.n_atoms 的最小 duck-typing 对象（无 lam / pair_r0）
    minimal = SimpleNamespace(
        i=np.array([0, 1], dtype=np.int32), j=np.array([1, 0], dtype=np.int32),
        S=np.zeros((2, 3), dtype=np.int32), n_atoms=2)
    res = coordination(atoms, minimal)
    assert res.cn_cell_warning is False
    assert res.cn_cell_margin == float("inf")
    assert res.cn.tolist() == [1, 1]
    # (b) 有 lam 但 pair_r0 为空 dict（mock 默认值路径）
    graph = make_graph(atoms, {(6, 6): 1.824}, lam=1.2, pair_r0={})
    res2 = coordination(atoms, graph)
    assert res2.cn_cell_warning is False
    assert res2.cn_cell_margin == float("inf")


def test_degenerate_cell_warning():
    """④ 退化胞（某格矢为 0）→ warning=True、margin=0（病态胞，交 L0）。

    病态胞判定优先级最高：即便 graph 缺 pair_r0（保守路径）也标记。
    """
    atoms = Atoms("HH", positions=[[0, 0, 0], [1.0, 0, 0]],
                  cell=[[2.0, 0, 0], [0, 0.0, 0], [0, 0, 10.0]],
                  pbc=[True, False, False])
    graph = MockBondGraph(
        i=np.array([0, 1], dtype=np.int32), j=np.array([1, 0], dtype=np.int32),
        S=np.zeros((2, 3), dtype=np.int32), d=np.array([1.0, 1.0]),
        n_atoms=2, lam=1.2, pair_r0={(1, 1): 0.62})
    res = coordination(atoms, graph)
    assert res.cn_cell_warning is True
    assert res.cn_cell_margin == 0.0
    assert res.cn.tolist() == [1, 1]
    # 病态胞 × 保守 graph（缺 pair_r0）→ 仍标记（优先级：病态胞 > 保守路径）
    minimal = SimpleNamespace(
        i=np.array([0, 1], dtype=np.int32), j=np.array([1, 0], dtype=np.int32),
        S=np.zeros((2, 3), dtype=np.int32), n_atoms=2)
    res2 = coordination(atoms, minimal)
    assert res2.cn_cell_warning is True
    assert res2.cn_cell_margin == 0.0


def test_margin_boundary_equals_one_no_warning():
    """边界：margin == 1（cutoff 恰等于半最短格矢）→ 不警告。

    键判据为严格 d < cutoff：两镜像分离 ≥ min|cell| = 2·cutoff，若两者均
    d < cutoff 则分离 < 2·cutoff，矛盾 → 至多一个镜像，无混叠。
    （边放在 d=0.5 深入 cutoff 内，规避 ASE 边界枚举的浮点语义。）
    """
    atoms = Atoms("CC", positions=[[0, 0, 0], [0.5, 0, 0]],
                  cell=[[2.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], pbc=True)
    graph = make_graph(atoms, {(6, 6): 1.0}, lam=1.0, pair_r0={(6, 6): 1.0})
    res = coordination(atoms, graph)
    assert res.cn_cell_margin == pytest.approx(1.0)
    assert res.cn_cell_warning is False
    # 唯一近邻 (j=1, S=0)@0.5Å；镜像 ±2.0Å 均在 cutoff 外 → CN=[1,1]
    assert res.cn.tolist() == [1, 1]
