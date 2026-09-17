"""L3 局域环境（m 实例）的钉死测试。

本文件的期望值全部是**物理上可核对**的：每个结构的第一配位壳、径向距离比、
有效配位数都能从教科书晶格常数直接算出来。任何一条变了，都说明归一化口径
或壳层算法动了——那正是需要人来看的地方。
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from crysh.config import MapperConfig
from crysh.localenv import D_I_VERSION, local_environment, local_environments, site_table
from crysh.tokens import motif_tokens

# (名字, 结构, 契约 CN, 几何第一壳 CN, cn_eff, r̃_mean, 壳内距离是否全等)
# r̃ = d / (共价半径和)：金属 < 1（共价半径和**高估**了金属键长，故 λ*=1.2 才够），
# 主族/离子晶体 > 1（近邻比共价半径和远）。两种符号都出现在下表里，是本表的价值所在。
KNOWN = [
    ("diamond-Si", bulk("Si", "diamond", a=5.431), 4, 4, 4.0, 1.059, True),
    ("fcc-Al", bulk("Al", "fcc", a=4.05), 12, 12, 12.0, 1.183, True),
    ("bcc-W", bulk("W", "bcc", a=3.165), 14, 14, 10.57, 0.902, False),
    ("hcp-Mg", bulk("Mg", "hcp", a=3.21, c=5.21), 12, 12, 11.85, 1.136, False),
    ("rocksalt-NaCl", bulk("NaCl", "rocksalt", a=5.64), 6, 6, 6.0, 1.052, True),
    ("zincblende-ZnS", bulk("ZnS", "zincblende", a=5.41), 4, 4, 4.0, 1.032, True),
]


@pytest.mark.parametrize("name,atoms,cn,cn1,cn_eff,r_tilde,_equal", KNOWN,
                         ids=[k[0] for k in KNOWN])
def test_known_coordination_and_shells(name, atoms, cn, cn1, cn_eff, r_tilde, _equal):
    env = local_environment(atoms, 0)
    assert env.cn == cn, f"{name}: 契约 CN"
    assert env.cn1 == cn1, f"{name}: 几何第一壳"
    assert env.cn_eff == pytest.approx(cn_eff, abs=0.02), f"{name}: cn_eff"
    assert env.r_tilde_mean == pytest.approx(r_tilde, abs=0.005), f"{name}: r̃_mean"


def test_single_shell_crystals_have_no_radial_spread():
    """壳内距离全等的理想晶体 → σ_d = 0、D = 0（畸变量的零点）。

    bcc/hcp 不在这一列：它们的"配位壳"含两个距离（bcc 8+6、hcp 12+2 轴向），
    径向离散是真实存在的，D > 0 才对（下面单独测）。
    """
    for name, atoms, _cn, _cn1, _ce, _rt, equal in KNOWN:
        if not equal:
            continue
        env = local_environment(atoms, 0)
        assert env.r_tilde_std == pytest.approx(0.0, abs=1e-12), name
        assert env.bond_cv == pytest.approx(0.0, abs=1e-12), name
        assert env.d_i == pytest.approx(0.0, abs=1e-12), name


def test_two_shell_metals_have_positive_radial_spread():
    """bcc W / hcp Mg：配位壳里有两种距离 → σ_d > 0（且 hcp 的轴向畸变更小）。"""
    for name in ("bcc-W", "hcp-Mg"):
        atoms = dict((k[0], k[1]) for k in KNOWN)[name]
        env = local_environment(atoms, 0)
        assert env.r_tilde_std > 0, name
        assert env.bond_cv > 0, name


def test_diamond_silicon_full_radial_picture():
    """diamond Si 的完整径向图像：4 个近邻 + 12 个第二壳 = 16（著名事实）。"""
    env = local_environment(bulk("Si", "diamond", a=5.431), 0)
    assert env.cn == 4
    assert env.cn1 == 4
    assert env.cn2 == 12, "第二壳 12 个（diamond 结构的标准结果）"
    assert env.cn_species == "Si4"
    # 第一壳边界跳变：2.351 → 3.84 Å，相对跳变 0.633
    assert env.shell_gap == pytest.approx(0.633, abs=0.01)


def test_bcc_tungsten_two_shells_are_not_split_by_the_bond_criterion():
    """bcc W：共价半径和（2.74 Å 附近）落在第一壳与第二壳之间。

    λ*=1.2 的键判据 `d < λ·r0` 把**两壳都算进**键集 → cn = 14；几何上第一壳+第二壳
    也确实是 14（8 + 6），故 cn1 = 14 与 cn 一致。这是"金属键判据偏松"的直接体现，
    也是 `p_cn` / `shell_conf` 存在的理由。
    """
    env = local_environment(bulk("W", "bcc", a=3.165), 0)
    assert env.cn == 14
    assert env.cn1 == 14, "8 + 6 = 14：两壳都被键判据收进来，壳层边界在 14 之后"
    assert env.cn_eff == pytest.approx(10.57, abs=0.02), "8 个近邻 + 6 个稍远 → 权重和"
    assert env.shell_conf < 0.6, "键判据与几何壳层口径不一致时不该给高置信"


def test_molecular_fragment_has_no_vacuum_neighbours():
    """孤立团簇（pbc=False）：CN 与第一壳都由分子内几何决定，不出现幻影邻居。"""
    atoms = Atoms("Ti" + "O" * 6,
                  positions=[[0, 0, 0], [1.95, 0, 0], [-1.95, 0, 0], [0, 1.95, 0],
                             [0, -1.95, 0], [0, 0, 1.95], [0, 0, -1.95]],
                  pbc=False)
    envs = local_environments(atoms)
    ti = envs[0]
    assert ti.cn == 6
    assert ti.cn_species == "O6"
    assert ti.h_neigh == pytest.approx(0.0), "单一元素邻居 → 熵 0"
    # 六个 Ti–O 距离全等 → 径向分布里没有缝 → 壳层解析不可用（gap=0）：
    # `cn1/cn2` 如实记 NaN，且 `shell_conf == 0`（没有壳层证据，不该假装有）
    assert ti.shell_gap == pytest.approx(0.0)
    assert np.isnan(ti.cn1) and np.isnan(ti.cn2)
    assert ti.shell_conf == pytest.approx(0.0)
    # O 位点只有 1 个邻居（Ti）
    assert envs[1].cn == 1 and envs[1].cn_species == "Ti1"


def test_two_neighbour_counting_conventions_are_both_pinned():
    """**两套邻居口径并存是刻意的**，这里把两者都钉住，免得日后互相"修正"。

    - `localenv`（L3 site 表）：按**键数**计数 —— 岩盐里 Na 有 6 个 Cl 键，`cn_species='Cl6'`；
    - `tokens`（L5）：l3 化学串与 `h_neigh` 也按**键数**（`Cl6`），故两者对同一结构
      给出一致的邻居计数与熵——这条一致性是有意的，避免"同一结构两处报不同 CN"。

    注意 (2026-09-17)：**多面体共享判据另有问题**（按原子索引配对，周期结构里失准），
    见 `tests/test_site_api.py` 里 `xfail(strict=True)` 登记的两个已知失败。
    """
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    env = local_environment(atoms, 0)
    toks = motif_tokens(atoms, graph=_graph(atoms), coord=_coord(atoms),
                        geometry_labels=["octahedral"] * len(atoms), d_star=3)
    assert env.cn == 6 and env.cn_species == "Cl6", "localenv：按键数"
    assert toks["l3"][0].split("|")[1] == "6", "契约 CN 字段（来自 coord）"
    assert toks["l3"][0].split("|")[3] == "Cl6", "化学串同样是按键数口径"
    assert toks["h_neigh"][0] == pytest.approx(env.h_neigh, abs=1e-12), "熵也同口径"


def test_disorder_raises_distortion_monotonically():
    """把配位壳拉开 → bond_cv 与 D 单调上升（D 的零点与方向感）。"""
    perfect = bulk("NaCl", "rocksalt", a=5.64)
    d0 = local_environment(perfect, 0).d_i

    rng = np.random.default_rng(0)
    distorted = perfect.copy()
    distorted.positions += rng.normal(scale=0.12, size=distorted.positions.shape)
    d1 = local_environment(distorted, 0).d_i

    distorted2 = perfect.copy()
    distorted2.positions += rng.normal(scale=0.30, size=distorted2.positions.shape)
    d2 = local_environment(distorted2, 0).d_i

    assert d0 == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < d1 < d2, (d0, d1, d2)


def test_site_table_shape_and_columns():
    atoms = bulk("Al", "fcc", a=4.05)
    df = site_table(atoms)
    assert len(df) == len(atoms)
    assert list(df["site_idx"]) == list(range(len(atoms)))
    for col in ("cn", "cn_eff", "cn_species", "cn_by_lambda", "p_cn", "shell_gap",
                "cn_shell", "cn1", "cn2", "shell_conf", "neighbor_counts", "h_neigh",
                "r_tilde_mean", "r_tilde_std", "bond_cv", "d_i", "geometry"):
        assert col in df.columns, col
    assert len(df["cn_by_lambda"].iloc[0]) == len(MapperConfig().lambdas)


def test_cn_by_lambda_is_monotone_nondecreasing():
    """λ 增大只会加邻居 → CN(λ) 单调不减（CN(λ) 谱的物理约束）。"""
    for name, atoms, *_ in KNOWN:
        env = local_environment(atoms, 0)
        assert env.cn_by_lambda == sorted(env.cn_by_lambda), name


def test_p_cn_and_cn_eff_are_bounded():
    for name, atoms, cn, *_ in KNOWN:
        env = local_environment(atoms, 0)
        assert 0.0 < env.p_cn <= 1.0, name
        assert 0.0 <= env.cn_eff <= env.cn + 1e-9, (name, env.cn_eff, env.cn)
        assert 0.0 <= env.shell_conf <= 1.0, name
        assert 0.0 <= env.d_i <= 1.0, name


def test_d_i_version_is_declared():
    """D 是 interim 组合量：版本必须在代码里可查（换 CSM 时同名替换）。"""
    assert D_I_VERSION.startswith("interim")


# ---- 小工具：给跨模块一致性测试造 graph/coord（避免重复建表）----


def _nl(atoms):
    from crysh.kernels import neighbor_list

    return neighbor_list(atoms, lam_max=2.0)


def _graph(atoms):
    from crysh.kernels import masked_graph

    return masked_graph(_nl(atoms), 1.2)


def _coord(atoms):
    from crysh.coord import coordination

    return coordination(atoms, _graph(atoms))
