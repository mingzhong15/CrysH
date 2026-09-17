"""Level 3 coordination()/calibrate_cn_table() 单测（内联 fixture + mock BondGraph）。

契约 §6：level1 的 bond.py 可能未合入——测试内自造 mock graph（按 §3.2 字段），
不 import crysh.bond。已知 CN 的内联结构：NaCl=6、diamond Si=4、石墨=3、fcc Al=12。
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.build import bulk
from ase.lattice.hexagonal import Graphite
from ase.neighborlist import neighbor_list
from ase.spacegroup import crystal

from crysh.coord import (
    CoordResult,
    apply_cn_table,
    calibrate_cn_table,
    cn_histogram,
    coordination,
    load_cn_table,
    save_cn_table,
)


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


def make_graph(atoms, cutoff, self_interaction=False, lam=1.0,
               cutoff_mode="table"):
    """用 ase.neighbor_list 造 mock 键图（"ijdS" 顺序: i, j, d, S）。"""
    i, j, d, S = neighbor_list(
        "ijdS", atoms, cutoff, self_interaction=self_interaction)
    mask = ~((i == j) & (S == 0).all(axis=1))  # 排除 i==j 且 S==0 的自环（§3.2）
    i, j, d, S = i[mask], j[mask], d[mask], S[mask]
    return MockBondGraph(
        i=i.astype(np.int32), j=j.astype(np.int32), S=S.astype(np.int32),
        d=d.astype(np.float64), n_atoms=len(atoms), lam=lam,
        cutoff_mode=cutoff_mode, pair_r0={})


# ---------- 已知 CN 的内联结构 ----------

def test_nacl_cn_all_6():
    """NaCl 岩盐 2x2x2 超胞：Na-Cl 2.82 Å < 3.0，同种离子 3.99 Å > 3.0 → CN 全 6。"""
    atoms = crystal(
        ["Na", "Cl"], [(0, 0, 0), (0.5, 0.5, 0.5)], spacegroup=225,
        cellpar=[5.64, 5.64, 5.64, 90, 90, 90]) * (2, 2, 2)
    graph = make_graph(atoms, 3.0)
    res = coordination(atoms, graph)
    assert res.cn.shape == (64,)
    assert np.all(res.cn == 6)
    assert res.mean_cn == pytest.approx(6.0)
    assert res.cn_min == 6 and res.cn_max == 6
    assert np.all(np.isnan(res.cn_percentile))  # 无表
    assert res.low_cn_fraction == 0.0 and res.high_cn_fraction == 0.0


def test_diamond_si_cn_all_4():
    """金刚石 Si 原胞：Si-Si 2.35 Å < 2.6，次近邻 3.84 Å > 2.6 → CN 全 4。"""
    atoms = bulk("Si", "diamond", a=5.43)
    graph = make_graph(atoms, 2.6)
    res = coordination(atoms, graph)
    assert np.all(res.cn == 4)
    assert res.mean_cn == pytest.approx(4.0)


def test_graphite_cn_all_3():
    """石墨（λ=1.0 键图，C-C 共价 cutoff 2*0.76=1.52 Å）：面内 1.42 计入、
    层间 3.35 排除 → CN 全 3。"""
    atoms = Graphite("C", latticeconstant={"a": 2.46, "c": 6.71})
    graph = make_graph(atoms, {(6, 6): 1.52}, lam=1.0, cutoff_mode="covalent")
    graph.pair_r0 = {(6, 6): 1.52}
    res = coordination(atoms, graph)
    assert np.all(res.cn == 3)
    assert res.cn_min == 3 and res.cn_max == 3


def test_fcc_al_cn_12():
    """fcc Al 原胞（大 cutoff 3.0：12 个 2.86 Å 周期自像近邻，次近邻 4.05 排除）
    → CN 12。含 i==j 且 S!=0 的自像边，验证不重复计反向端点。"""
    atoms = bulk("Al", "fcc", a=4.05)
    graph = make_graph(atoms, 3.0, self_interaction=True)
    res = coordination(atoms, graph)
    assert res.cn.shape == (1,)
    assert res.cn[0] == 12
    assert res.cn_max == 12


# ---------- 手造 mock graph 的数值单测 ----------

def _hand_graph(atoms, edges):
    """手造非周期 mock 图。edges: [(i, j, [sx, sy, sz]), ...]（单向或双向均可）。"""
    i, j, S, d = [], [], [], []
    pos = atoms.positions
    for a, b, s in edges:
        i.append(a)
        j.append(b)
        S.append(s)
        d.append(float(np.linalg.norm(pos[b] - pos[a] + s @ atoms.cell)))
    return MockBondGraph(
        i=np.array(i, dtype=np.int32), j=np.array(j, dtype=np.int32),
        S=np.array(S, dtype=np.int32), d=np.array(d, dtype=np.float64),
        n_atoms=len(atoms))


def test_mock_graph_numerics():
    """4 原子链，单向存储边 (0,1),(1,2)：CN=[1,2,1,0]；验证汇总数值。"""
    atoms = Atoms("HHHH", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 0, 0]],
                  cell=[10, 10, 10], pbc=False)
    graph = _hand_graph(atoms, [(0, 1, [0, 0, 0]), (1, 2, [0, 0, 0])])
    table = {1: {"cdf": {0: 0.2, 1: 0.6, 2: 1.0}, "n": 100}}
    res = coordination(atoms, graph, cn_table=table)
    assert res.cn.tolist() == [1, 2, 1, 0]
    assert res.mean_cn == pytest.approx(1.0)
    assert res.cn_min == 0 and res.cn_max == 2
    np.testing.assert_allclose(
        res.cn_percentile, [0.6, 1.0, 0.6, 0.2], equal_nan=False)
    assert res.low_cn_fraction == 0.0          # 无 p<0.05
    assert res.high_cn_fraction == pytest.approx(0.25)  # 仅 p=1.0 的 1 站


def test_mock_graph_dedup_directions_and_duplicates():
    """重复边 / 双向存储 / 同 j 不同 S 的去重：CN 只数不同 (j, S)。"""
    atoms = Atoms("HHHH", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 0, 0]],
                  cell=[10, 10, 10], pbc=False)
    # 双向 + 重复 + 同 j 不同 S：atom0 有 2 个近邻 ((1,S=0) 与 (1,S=e_x))；
    # atom1 有 3 个 ((0,S=0)、(0,S=-e_x) 与 (2,S=0))；重复边与双向边被去重
    edges = [(0, 1, [0, 0, 0]), (1, 0, [0, 0, 0]), (0, 1, [0, 0, 0]),
             (0, 1, [1, 0, 0]), (1, 2, [0, 0, 0]), (2, 1, [0, 0, 0])]
    graph = _hand_graph(atoms, edges)
    res = coordination(atoms, graph)
    assert res.cn.tolist() == [2, 3, 1, 0]


def test_self_image_edges_count_once():
    """i==j 自像边：只计 (j,S) 一次，不虚构反向端点。"""
    atoms = Atoms("Al", positions=[[0, 0, 0]], cell=[4.05] * 3, pbc=True)
    vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -1, 0]]  # 4 个不同方向
    graph = _hand_graph(atoms, [(0, 0, v) for v in vecs])
    res = coordination(atoms, graph)
    assert res.cn[0] == 4


# ---------- percentile 边界 ----------

def test_percentile_no_table():
    atoms = Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10],
                  pbc=False)
    graph = _hand_graph(atoms, [(0, 1, [0, 0, 0])])
    res = coordination(atoms, graph, cn_table=None)
    assert np.all(np.isnan(res.cn_percentile))
    assert res.low_cn_fraction == 0.0 and res.high_cn_fraction == 0.0


def test_percentile_missing_z_and_unseen_cn():
    """表缺 Z 或缺该 cn 值 → NaN；NaN 站点计入分母但不计为异常（保守）。"""
    atoms = Atoms("HHHeHe", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 0, 0]],
                  cell=[10, 10, 10], pbc=False)
    graph = _hand_graph(atoms, [(0, 1, [0, 0, 0]), (1, 2, [0, 0, 0])])  # CN=[1,2,1,0]
    # 表只有 H(1) 且 cdf 只有 cn=1：H cn=2 站点、He(2) 站点 → NaN
    table = {1: {"cdf": {1: 0.5}}}
    res = coordination(atoms, graph, cn_table=table)
    p = res.cn_percentile
    assert p[0] == pytest.approx(0.5)   # H cn=1 有值
    assert np.isnan(p[1])               # H cn=2 表内无该 cn → NaN
    assert np.isnan(p[2])               # He 缺 Z → NaN
    assert np.isnan(p[3])               # He 缺 Z → NaN
    assert res.low_cn_fraction == 0.0 and res.high_cn_fraction == 0.0


def test_percentile_low_high_fractions():
    """低/高配位 fraction 数值：p<0.05 与 p>0.95，分母=N。"""
    atoms = Atoms("HHHH", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
                  cell=[10, 10, 10], pbc=False)
    graph = _hand_graph(atoms, [(0, 1, [0, 0, 0]), (1, 2, [0, 0, 0]),
                                (2, 3, [0, 0, 0])])  # CN=[1,2,2,1]
    table = {1: {"cdf": {1: 0.04, 2: 1.0}}}  # cn=1 → p=0.04(low)；cn=2 → 1.0(high)
    res = coordination(atoms, graph, cn_table=table)
    assert res.low_cn_fraction == pytest.approx(0.5)   # 2/4
    assert res.high_cn_fraction == pytest.approx(0.5)  # 2/4


def test_apply_cn_table_string_keys():
    """JSON 直接读入（str 键）也能查询。"""
    table = {"1": {"cdf": {"1": 0.5, "2": 1.0}}}
    p = apply_cn_table(np.array([1, 1]), np.array([1, 2]), table)
    np.testing.assert_allclose(p, [0.5, 1.0])


# ---------- calibrate_cn_table ----------

def test_calibrate_from_long_table():
    df = pd.DataFrame({
        "Z": [8, 8, 8, 8, 8, 14, 14],
        "cn": [2, 2, 2, 3, 4, 4, 5],
    })
    # min_sites=0：本测试验证 CDF 构造（门槛语义见 test_coord_cn_table_guard.py）
    table = calibrate_cn_table(df, min_sites=0)
    assert table[8]["n"] == 5
    assert table[8]["cdf"] == {2: pytest.approx(0.6), 3: pytest.approx(0.8),
                               4: pytest.approx(1.0)}
    assert table[14]["n"] == 2
    assert table[14]["cdf"] == {4: pytest.approx(0.5), 5: pytest.approx(1.0)}


def test_calibrate_from_manifest_style():
    """结构级降级模式：elements/n_atoms + cn_mean → 每站赋 round(cn_mean)。"""
    df = pd.DataFrame({
        "elements": [["Hf", "O", "Li"]],
        "n_atoms": [[2, 4, 2]],
        "cn_mean": [6.2],
    })
    table = calibrate_cn_table(df, min_sites=0)  # 门槛语义见 test_coord_cn_table_guard.py
    assert table[72]["n"] == 2 and table[72]["cdf"] == {6: pytest.approx(1.0)}
    assert table[8]["n"] == 4 and table[8]["cdf"] == {6: pytest.approx(1.0)}
    assert table[3]["n"] == 2 and table[3]["cdf"] == {6: pytest.approx(1.0)}


def test_calibrate_bad_columns_raises():
    with pytest.raises(ValueError):
        calibrate_cn_table(pd.DataFrame({"foo": [1], "bar": [2]}))


def test_calibrate_drops_invalid():
    df = pd.DataFrame({"Z": [0, 8, 8], "cn": [4, -1, 2]})
    table = calibrate_cn_table(df, min_sites=0)  # 门槛语义见 test_coord_cn_table_guard.py
    assert set(table) == {8}
    assert table[8]["n"] == 1
    assert table[8]["cdf"] == {2: pytest.approx(1.0)}


# ---------- 表 JSON 读写 ----------

def test_save_load_cn_table_roundtrip(tmp_path):
    table = {3: {"cdf": {0: 0.1, 6: 1.0}, "n": 10},
             8: {"cdf": {2: 0.6, 4: 1.0}, "n": 5}}
    out = save_cn_table(table, tmp_path / "cn_table_v1.json",
                        meta={"source": "test"})
    loaded = load_cn_table(out)
    assert set(loaded) == {3, 8}          # int 键恢复
    assert loaded[3]["n"] == 10
    assert loaded[3]["cdf"][0] == pytest.approx(0.1)
    assert loaded[8]["cdf"][2] == pytest.approx(0.6)


# ---------- 防御性检查 / 辅助 ----------

def test_empty_atoms_raises():
    with pytest.raises(ValueError):
        coordination(Atoms(), _hand_graph(Atoms(), []))


def test_graph_natoms_mismatch_raises():
    atoms = Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10],
                  pbc=False)
    graph = _hand_graph(atoms, [(0, 1, [0, 0, 0])])
    graph.n_atoms = 99
    with pytest.raises(ValueError):
        coordination(atoms, graph)


def test_empty_graph_cn_zero():
    atoms = Atoms("HHHH", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [5, 0, 0]],
                  cell=[10, 10, 10], pbc=False)
    graph = _hand_graph(atoms, [])
    res = coordination(atoms, graph)
    assert res.cn.tolist() == [0, 0, 0, 0]
    assert res.cn_min == 0 and res.cn_max == 0


def test_cn_histogram():
    hist = cn_histogram(np.array([0, 2, 2, 20, 3]))
    assert len(hist) == 17
    assert hist[0] == 1 and hist[2] == 2 and hist[3] == 1
    assert hist[16] == 1                      # CN=20 截断入末 bin
    assert sum(hist) == 5


def test_coordresult_fields():
    """CoordResult 字段类型符合契约。"""
    res = CoordResult(cn=np.array([1, 2]), cn_percentile=np.array([0.1, np.nan]),
                      mean_cn=1.5, low_cn_fraction=0.0, high_cn_fraction=0.5,
                      cn_min=1, cn_max=2)
    assert res.mean_cn == 1.5 and res.high_cn_fraction == 0.5
