"""v1.3-L5 sharing 中心门槛放宽单测（tokenize.py 平面三角形中心例外）。

物理动机（用户指令，主会话 L5 评审）：BO₃ 硼酸盐 / CO₃ 碳酸盐网络的
corner-sharing 在纯 CN≥4 门槛下完全不可见——B/C 的配位恒为平面三角形
CN=3，网络共享性此前恒 isolated。改造：多面体中心 = CN≥4，**或** CN==3
且 geometry label == "trigonal_planar"；linear（CN=2）与 CN=3 的
"other"/"ambiguous"（T 形等）仍排除。

"改造前"断言用 level5-tokenizer/scripts/regress_2k_local.py 内的
motif_tokens_legacy（改造前 tokenize.py 的逐字快照，纯参考实现）。

不依赖 GEOMETRY_LABELS 全集（level4 标签集独立演进中，本文件只做
字符串成员判断，不硬编码标签列表长度）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from crysh.tokens import motif_tokens

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from _legacy_tokens import motif_tokens_legacy  # noqa: E402  (本地快照，见该模块 docstring)


# ---- 契约 mock（contracts.md §3.2 / §3.6 字段，与 test_tokenize.py 同构）----
@dataclass
class MockBondGraph:
    i: np.ndarray
    j: np.ndarray
    S: np.ndarray
    d: np.ndarray
    n_atoms: int
    lam: float = 1.0
    cutoff_mode: str = "mock"
    pair_r0: dict = None


@dataclass
class MockCoordResult:
    cn: np.ndarray
    cn_percentile: np.ndarray = None
    mean_cn: float = 0.0
    low_cn_fraction: float = 0.0
    high_cn_fraction: float = 0.0
    cn_min: int = 0
    cn_max: int = 0


def make_graph(edges: list[tuple[int, int]], n_atoms: int) -> MockBondGraph:
    i = np.array([e[0] for e in edges], dtype=np.int32)
    j = np.array([e[1] for e in edges], dtype=np.int32)
    S = np.zeros((len(edges), 3), dtype=np.int32)
    d = np.full(len(edges), 2.0, dtype=np.float64)
    return MockBondGraph(i=i, j=j, S=S, d=d, n_atoms=n_atoms)


def mock_coord(cn: list[int]) -> MockCoordResult:
    arr = np.array(cn, dtype=np.int64)
    return MockCoordResult(cn=arr, mean_cn=float(arr.mean()),
                           cn_min=int(arr.min()), cn_max=int(arr.max()))


# ---- fixtures ----

def bo3_dimer(b_labels=("trigonal_planar", "trigonal_planar")):
    """两个 BO₃ 平面三角形共享 1 个桥氧（corner-sharing，硼酸盐网络基本连接）。

    B0(0): O_br(2) + 端氧 3,4；B1(1): O_br(2) + 端氧 5,6；
    桥氧 CN=2（连两 B），端氧 CN=1。b_labels 可注入其他 CN=3 几何标签
    （"other"/"ambiguous" → T 形对照）。
    """
    atoms = Atoms(
        symbols=["B", "B", "O", "O", "O", "O", "O"],
        positions=[[0, 0, 0], [3.0, 0, 0], [1.5, 0, 0],
                   [0, 1.4, 0], [0, -1.4, 0], [3.0, 1.4, 0], [3.0, -1.4, 0]],
        pbc=False,
    )
    edges = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 5), (1, 6)]
    graph = make_graph(edges, n_atoms=7)
    coord = mock_coord([3, 3, 2, 1, 1, 1, 1])
    labels = [str(b_labels[0]), str(b_labels[1])] + ["other"] * 5
    return atoms, graph, coord, labels


def co2_molecules():
    """CO₂ 类 linear CN=2 分子（两个分子；C label=linear，O CN=1）。"""
    atoms = Atoms(
        symbols=["C", "O", "O", "C", "O", "O"],
        positions=[[0, 0, 0], [1.2, 0, 0], [-1.2, 0, 0],
                   [5.0, 0, 0], [6.2, 0, 0], [3.8, 0, 0]],
        pbc=False,
    )
    graph = make_graph([(0, 1), (0, 2), (3, 4), (3, 5)], n_atoms=6)
    coord = mock_coord([2, 1, 1, 2, 1, 1])
    labels = ["linear", "other", "other", "linear", "other", "other"]
    return atoms, graph, coord, labels


def oct_plus_planar():
    """TiO6 八面体（CN=6 oct）与 BO₃ 平面（CN=3 trigonal_planar）共享 1 桥氧。

    跨门槛中心对（CN=6 × CN=3 平面）——新逻辑下计入 corner；旧逻辑下
    桥氧只挂 1 个 CN≥4 中心 → 无对 → isolated。
    """
    atoms = Atoms(
        symbols=["Ti", "B"] + ["O"] * 8,
        positions=[[0, 0, 0], [3.5, 0, 0], [1.75, 0, 0],
                   [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2], [2, 2, 0],
                   [3.5, 1.4, 0], [3.5, -1.4, 0]],
        pbc=False,
    )
    edges = [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
             (1, 2), (1, 8), (1, 9)]
    graph = make_graph(edges, n_atoms=10)
    coord = mock_coord([6, 3, 2, 1, 1, 1, 1, 1, 1, 1])
    labels = ["octahedral", "trigonal_planar"] + ["other"] * 8
    return atoms, graph, coord, labels


def isolated_bo3():
    """单个 BO₃ 基团（无桥氧）：B 计入中心，但无共享配体 → 仍 isolated。"""
    atoms = Atoms(symbols=["B", "O", "O", "O"],
                  positions=[[0, 0, 0], [1.4, 0, 0],
                             [-0.7, 1.21, 0], [-0.7, -1.21, 0]], pbc=False)
    graph = make_graph([(0, 1), (0, 2), (0, 3)], n_atoms=4)
    coord = mock_coord([3, 1, 1, 1])
    labels = ["trigonal_planar", "other", "other", "other"]
    return atoms, graph, coord, labels


# ---- ① BO₃ 网络：改造前 isolated → 改造后 corner ----

def test_bo3_network_corner_sharing():
    atoms, graph, coord, labels = bo3_dimer()
    # 改造前（legacy 逐字快照）：B CN=3 < 4 → 无中心对 → isolated / f=0
    legacy = motif_tokens_legacy(atoms, graph, coord, labels, d_star=3)
    assert legacy["sharing_label"] == "isolated"
    assert legacy["sharing"] == {"corner": 0.0, "edge": 0.0, "face": 0.0}
    # 改造后：两个平面三角形中心共享 1 个桥氧 → corner，f_corner > 0
    res = motif_tokens(atoms, graph, coord, labels, d_star=3)
    assert res["sharing_label"] == "corner"
    assert res["sharing"]["corner"] == 1.0
    assert res["sharing"]["corner"] > 0.0
    assert res["sharing"]["edge"] == 0.0
    assert res["sharing"]["face"] == 0.0
    # l4 token 携带新 sharing_label（B 位点）
    assert res["l4"][0].endswith("|corner|3D")
    assert res["l4"][1].endswith("|corner|3D")


def test_bo3_tokens_l1_l2_l3_unchanged():
    """放宽只动 sharing：l1/l2/l3 新旧逐位相同（2k 清单 (i) 的单测版）。"""
    atoms, graph, coord, labels = bo3_dimer()
    res = motif_tokens(atoms, graph, coord, labels, d_star=3)
    legacy = motif_tokens_legacy(atoms, graph, coord, labels, d_star=3)
    assert res["l1"] == legacy["l1"]
    assert res["l2"] == legacy["l2"]
    assert res["l3"] == legacy["l3"]
    # B 位点 l3 = "B|3|trigonal_planar|O3"（1 桥氧 + 2 端氧）
    assert res["l3"][0] == "B|3|trigonal_planar|O3"
    assert res["l3"][1] == "B|3|trigonal_planar|O3"
    # 桥氧 l3：邻居 2 个 B
    assert res["l3"][2] == "O|2|other|B2"


# ---- ② CO₂ 类 linear CN=2 → 仍 isolated（回归）----

def test_co2_linear_cn2_still_isolated():
    atoms, graph, coord, labels = co2_molecules()
    legacy = motif_tokens_legacy(atoms, graph, coord, labels, d_star=0)
    res = motif_tokens(atoms, graph, coord, labels, d_star=0)
    assert legacy["sharing_label"] == "isolated"
    assert res["sharing_label"] == "isolated"
    assert res["sharing"] == {"corner": 0.0, "edge": 0.0, "face": 0.0}
    assert res["l4"][0] == "C|2|linear|O2|isolated|0D"


# ---- ③ CN=3 但 geometry=other（T 形）→ 仍不计中心 ----

@pytest.mark.parametrize("bad_label", ["other", "ambiguous"])
def test_cn3_nonplanar_geometry_not_center(bad_label):
    atoms, graph, coord, labels = bo3_dimer(b_labels=(bad_label, bad_label))
    res = motif_tokens(atoms, graph, coord, labels, d_star=3)
    assert res["sharing_label"] == "isolated"
    assert res["sharing"] == {"corner": 0.0, "edge": 0.0, "face": 0.0}


def test_cn3_trigonal_planar_only_one_side():
    """只有一端是 trigonal_planar、另一端 other：单侧计入中心 → 无对 → isolated。"""
    atoms, graph, coord, labels = bo3_dimer(
        b_labels=("trigonal_planar", "other"))
    res = motif_tokens(atoms, graph, coord, labels, d_star=3)
    assert res["sharing_label"] == "isolated"
    assert res["sharing"] == {"corner": 0.0, "edge": 0.0, "face": 0.0}


# ---- 附加：跨门槛中心对（CN=6 八面体 × CN=3 平面）与孤立 BO₃ 回归 ----

def test_octahedral_planar_cross_pair_counts():
    atoms, graph, coord, labels = oct_plus_planar()
    legacy = motif_tokens_legacy(atoms, graph, coord, labels, d_star=3)
    assert legacy["sharing_label"] == "isolated"  # 旧：桥氧只挂 1 个 CN≥4 中心
    res = motif_tokens(atoms, graph, coord, labels, d_star=3)
    assert res["sharing_label"] == "corner"       # 新：跨门槛对计入
    assert res["sharing"]["corner"] == 1.0
    assert res["l4"][0].endswith("|corner|3D")    # Ti 位点
    assert res["l4"][1].endswith("|corner|3D")    # B 位点


def test_isolated_bo3_group_still_isolated():
    """无桥氧的 BO₃：B 是中心但没有共享配体 → isolated（分母为 0 语义不变）。"""
    atoms, graph, coord, labels = isolated_bo3()
    res = motif_tokens(atoms, graph, coord, labels, d_star=3)
    assert res["sharing_label"] == "isolated"
    assert res["sharing"] == {"corner": 0.0, "edge": 0.0, "face": 0.0}


def test_cn2_with_planar_label_not_center():
    """CN==2 即使 label 是 trigonal_planar（mock 不一致输入）也不计中心：
    门槛要求 CN==3 且 trigonal_planar（保守解释）。"""
    atoms = Atoms(symbols=["B", "O", "O"],
                  positions=[[0, 0, 0], [1.4, 0, 0], [-1.4, 0, 0]], pbc=False)
    graph = make_graph([(0, 1), (0, 2)], n_atoms=3)
    coord = mock_coord([2, 1, 1])
    labels = ["trigonal_planar", "other", "other"]
    res = motif_tokens(atoms, graph, coord, labels, d_star=3)
    assert res["sharing_label"] == "isolated"
