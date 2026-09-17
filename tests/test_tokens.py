"""motif_tokens 单测：契约字段 mock（§3.2 BondGraph / §3.6 CoordResult）+ TiO6 八面体真值。

所有 fixture 内联构造，不依赖 level1–4 真实现。
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest
from ase import Atoms

from crysh.tokens import motif_tokens


# ---- 契约 mock（contracts.md §3.2 / §3.6 字段）----
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


def make_graph(edges: list[tuple[int, int]], n_atoms: int, d: float = 2.0) -> MockBondGraph:
    i = np.array([e[0] for e in edges], dtype=np.int32)
    j = np.array([e[1] for e in edges], dtype=np.int32)
    S = np.zeros((len(edges), 3), dtype=np.int32)
    dd = np.full(len(edges), d, dtype=np.float64)
    return MockBondGraph(i=i, j=j, S=S, d=dd, n_atoms=n_atoms)


def mock_coord(cn: list[int]) -> MockCoordResult:
    arr = np.array(cn, dtype=np.int64)
    return MockCoordResult(cn=arr, mean_cn=float(arr.mean()),
                           cn_min=int(arr.min()), cn_max=int(arr.max()))


@pytest.fixture
def fake_dimension_module():
    """注入假 crysh.dimensionality（d_star=2），用后还原，与真实现无关。"""
    fake = types.ModuleType("crysh.dimensionality")

    class _Spec:
        d_star = 2
        persistence = 1.0
        n_components = 1
        translations_rank = 2
        d_by_lambda = {1.0: 2}

    def _spectrum(atoms, cutoff_table=None, lambdas=None):
        return _Spec()

    fake.dimensionality_spectrum = _spectrum
    prev = sys.modules.get("crysh.dimensionality")
    sys.modules["crysh.dimensionality"] = fake
    yield fake
    if prev is None:
        sys.modules.pop("crysh.dimensionality", None)
    else:
        sys.modules["crysh.dimensionality"] = prev


# ---- TiO6 八面体 mock（已知真值）----
def tio6_octahedron() -> tuple[Atoms, MockBondGraph, MockCoordResult, list[str]]:
    atoms = Atoms(
        symbols=["Ti"] + ["O"] * 6,
        positions=[[0, 0, 0], [2, 0, 0], [-2, 0, 0], [0, 2, 0],
                   [0, -2, 0], [0, 0, 2], [0, 0, -2]],
        pbc=False,
    )
    edges = [(0, k) for k in range(1, 7)]
    graph = make_graph(edges, n_atoms=7)
    coord = mock_coord([6, 1, 1, 1, 1, 1, 1])
    labels = ["oct"] + ["other"] * 6
    return atoms, graph, coord, labels


def test_tio6_octahedron_token_fields():
    atoms, graph, coord, labels = tio6_octahedron()
    res = motif_tokens(atoms, graph, coord, labels)

    assert res["l1"][0] == "Ti|6"
    assert res["l2"][0] == "Ti|6|oct"
    assert res["l3"][0] == "Ti|6|oct|O6"
    # d_star 由真实现算出：pbc=False 的孤立八面体是 0D 团簇 → "0D"
    # （R2 之前这里的期望是 "3D"，那是 import 失败后的保守兜底值，物理上是错的）
    assert res["l4"][0] == "Ti|6|oct|O6|isolated|0D"

    # O 位点（CN=1，邻居 Ti）
    assert res["l1"][1] == "O|1"
    assert res["l2"][1] == "O|1|other"
    assert res["l3"][1] == "O|1|other|Ti1"
    assert res["l4"][1] == "O|1|other|Ti1|isolated|0D"

    # h_neigh：Ti 单元素邻居 → 0；无邻居位点 → 0
    assert res["h_neigh"].shape == (7,)
    assert res["h_neigh"][0] == pytest.approx(0.0, abs=1e-12)
    assert all(res["h_neigh"][k] == pytest.approx(0.0, abs=1e-12) for k in range(1, 7))

    # sharing：无 CN≥4 中心对（O 均 CN=1）→ isolated
    assert res["sharing"] == {"corner": 0.0, "edge": 0.0, "face": 0.0}
    assert res["sharing_label"] == "isolated"


def test_d_star_from_dimension_module(fake_dimension_module):
    atoms, graph, coord, labels = tio6_octahedron()
    res = motif_tokens(atoms, graph, coord, labels)
    assert res["l4"][0] == "Ti|6|oct|O6|isolated|2D"
    assert res["l4"][1] == "O|1|other|Ti1|isolated|2D"


def test_mixed_neighbor_chemistry_and_h_neigh():
    # Ti 配 4 O + 2 F → l3 直方图按符号字母序 + 计数
    atoms = Atoms(
        symbols=["Ti"] + ["O"] * 4 + ["F"] * 2,
        positions=[[0, 0, 0]] + [[2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0],
                                [0, 0, 2], [0, 0, -2]],
        pbc=False,
    )
    graph = make_graph([(0, k) for k in range(1, 7)], n_atoms=7)
    coord = mock_coord([6, 1, 1, 1, 1, 1, 1])
    labels = ["oct"] + ["other"] * 6
    res = motif_tokens(atoms, graph, coord, labels)

    assert res["l3"][0] == "Ti|6|oct|F2O4"
    # H = -(2/3 ln 2/3 + 1/3 ln 1/3)
    expected = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    assert res["h_neigh"][0] == pytest.approx(expected, abs=1e-12)


def test_histogram_count_always_appended():
    # 每种邻居各 1 个也写计数（可逆无歧义）
    atoms = Atoms(symbols=["Ti", "O", "F"],
                  positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0]], pbc=False)
    graph = make_graph([(0, 1), (0, 2)], n_atoms=3)
    coord = mock_coord([2, 1, 1])
    labels = ["linear", "other", "other"]
    res = motif_tokens(atoms, graph, coord, labels)
    assert res["l3"][0] == "Ti|2|linear|F1O1"


def _two_octahedra(n_shared: int):
    """两个 TiO6 八面体共享 n_shared 个配体（1/2/3 → corner/edge/face）。"""
    n_atoms = 2 + n_shared + (6 - n_shared) * 2
    symbols = ["Ti", "Ti"] + ["O"] * (n_atoms - 2)
    positions = [[0, 0, 0], [4, 0, 0]] + [
        [2, 0, 0] for _ in range(n_shared)
    ] + [[1, 2 + 0.1 * k, 0] for k in range(6 - n_shared)] + [
        [3, 2 + 0.1 * k, 0] for k in range(6 - n_shared)
    ]
    atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
    edges = []
    shared = list(range(2, 2 + n_shared))
    for s in shared:
        edges += [(0, s), (1, s)]
    priv0 = list(range(2 + n_shared, 2 + n_shared + (6 - n_shared)))
    priv1 = list(range(2 + n_shared + (6 - n_shared), n_atoms))
    edges += [(0, p) for p in priv0]
    edges += [(1, p) for p in priv1]
    graph = make_graph(edges, n_atoms=n_atoms)
    cn = [6, 6] + [2] * n_shared + [1] * ((6 - n_shared) * 2)
    coord = mock_coord(cn)
    labels = ["oct", "oct"] + ["other"] * (n_atoms - 2)
    return atoms, graph, coord, labels


def test_sharing_corner():
    atoms, graph, coord, labels = _two_octahedra(1)
    res = motif_tokens(atoms, graph, coord, labels)
    assert res["sharing"] == {"corner": 1.0, "edge": 0.0, "face": 0.0}
    assert res["sharing_label"] == "corner"


def test_sharing_edge():
    atoms, graph, coord, labels = _two_octahedra(2)
    res = motif_tokens(atoms, graph, coord, labels)
    assert res["sharing"] == {"corner": 0.0, "edge": 1.0, "face": 0.0}
    assert res["sharing_label"] == "edge"


def test_sharing_face():
    atoms, graph, coord, labels = _two_octahedra(3)
    res = motif_tokens(atoms, graph, coord, labels)
    assert res["sharing"] == {"corner": 0.0, "edge": 0.0, "face": 1.0}
    assert res["sharing_label"] == "face"


def test_sharing_mixed():
    # 三个八面体链：pair(0,1) corner + pair(1,2) edge → 0.5/0.5 → mixed
    n_atoms = 18
    atoms = Atoms(
        symbols=["Ti"] * 3 + ["O"] * (n_atoms - 3),
        positions=[[0, 0, 0], [4, 0, 0], [8, 0, 0]]
        + [[2, 0, 0], [6, 0, 0], [6, 2, 0]]
        + [[1, 2, 0], [1, 2.4, 0], [1, 2.8, 0], [1, 3.2, 0], [1, 3.6, 0]]
        + [[5, 2, 0], [5, 2.4, 0], [5, 2.8, 0]]
        + [[9, 2, 0], [9, 2.4, 0], [9, 2.8, 0], [9, 3.2, 0]],
        pbc=False,
    )
    # Ti0(0): 共享 O3(与 Ti1) + 私 6..10；Ti1(1): O3 + O4,O5(与 Ti2) + 私 11..13
    # Ti2(2): O4,O5 + 私 14..17
    edges = [(0, 3), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10),
             (1, 3), (1, 4), (1, 5), (1, 11), (1, 12), (1, 13),
             (2, 4), (2, 5), (2, 14), (2, 15), (2, 16), (2, 17)]
    graph = make_graph(edges, n_atoms=n_atoms)
    cn = [6, 6, 6] + [2, 2, 2] + [1] * 12
    coord = mock_coord(cn)
    labels = ["oct"] * 3 + ["other"] * (n_atoms - 3)
    res = motif_tokens(atoms, graph, coord, labels)
    assert res["sharing"] == {"corner": 0.5, "edge": 0.5, "face": 0.0}
    assert res["sharing_label"] == "mixed"


def test_sharing_ignores_low_cn_centers():
    # 同一图但 CN 全 <4 → 不计任何中心对 → isolated（对比 corner 测试）。
    # v1.3-L5 修正注释：中心门槛放宽为 CN≥4 **或** CN==3 且 label ==
    # "trigonal_planar"。本 fixture 的 CN=3 中心带 mock 标签 "oct"（非
    # trigonal_planar）→ 仍被排除，断言不变；平面中心翻转行为由
    # tests/test_planar_sharing.py 覆盖（BO₃ 网络 corner 化）。
    atoms, graph, _, labels = _two_octahedra(1)
    coord = mock_coord([3, 3] + [1] * 11)
    res = motif_tokens(atoms, graph, coord, labels)
    assert res["sharing_label"] == "isolated"
    assert res["sharing"] == {"corner": 0.0, "edge": 0.0, "face": 0.0}


def test_length_mismatch_raises():
    atoms, graph, coord, labels = tio6_octahedron()
    with pytest.raises(ValueError):
        motif_tokens(atoms, graph, coord, labels[:-1])
    bad_cn = mock_coord([6, 1, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        motif_tokens(atoms, graph, bad_cn, labels)
