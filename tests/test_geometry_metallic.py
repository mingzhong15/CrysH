"""Level 4 geometry 金属环境扩展单测（v1.3-L4 前置扩展，集成者预先授权）。

覆盖（任务书）：
1. REF 实测值 vs 文献值 sanity（±0.005）——理想壳（fcc cuboctahedron 12 顶点 /
   hcp 理想 c/a 12 近邻壳 / bcc 8+6=14 近邻壳）经**本模块 _q_l** 实测；
2. 真实晶体：理想 fcc Al（常规 4 原子胞）/ hcp Mg（理想 c/a）/ bcc Fe（常规
   2 原子胞）经 level1 真键图（λ*=1.20）→ 全站点 cuboctahedral / hcp_like /
   bcc_like，conf≥0.9，且 CN 逐站点为 12/12/14；
3. icosahedral 既有 fixture 回归（CN=12 三候选下仍 icosahedral）；
4. 中等畸变 fcc（顶点随机抖动，d1 逼近 T_MATCH）→ label ∈ {cuboctahedral
   （conf 降低）, ambiguous}，**绝不**高置信 hcp_like/icosahedral（防错判）；
5. 路由/边界单元测试：_ROUTES 新条目、参考对距离 > T_MATCH、fcc–hcp 中点 →
   ambiguous、CN=7/9/10/11/13/15 fixture → other（确定性）；
6. bcc_like 的 CN=14 语义：键图 8+6（次近邻在 λ*=1.2 校准 cutoff 内），
   cutoff 收紧（λ=1.0，只含 8 近邻=立方体顶点）时走 CN=8 路由判 cubic。

fixture 全部内联（本文件 + test_geometry.py 的 FIXTURE_VECS），不依赖
controls/数据子集（contracts.md §1：单测先用自己内联构造的 fixture）。
"""
import math

import numpy as np
import pytest
from ase.build import bulk
from ase.data import covalent_radii

# 复用既有测试的 fixture 工具（中心+顶点 mock 图；双向邻接 v1.1）
from test_geometry import FIXTURE_VECS, classify_center, make_atoms

from crysh.geometry import (
    _ROUTES,
    GEOMETRY_LABELS,
    N_FEATURES,
    REF_GEOMETRY,
    T_MATCH,
    _q_l,
    build_graph,
    classify_geometry,
    geometry_features,
)

# 文献 (q4, q6)（Steinhardt 秩参量标准表；容差 ±0.005，任务书）
LITERATURE_QQ = {
    "cuboctahedral": (0.190945, 0.574524),   # fcc
    "hcp_like": (0.097221, 0.484761),        # hcp（理想 c/a）
    "bcc_like": (0.036370, 0.510688),        # bcc（8+6 等权）
}
EXPECTED_CN = {"cuboctahedral": 12, "hcp_like": 12, "bcc_like": 14}


def ideal_unit_vectors(name):
    """理想壳单位方向（fixture 向量归一化，与 REF 校准同源）。"""
    v = np.asarray(FIXTURE_VECS[name], dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _row(cn, q4, q6):
    """合成单站点特征行（仅 CN/q4/q6，路由与 q 距离判定只需这三列）。"""
    feats = np.zeros((1, N_FEATURES))
    feats[0, 0], feats[0, 12], feats[0, 13] = cn, q4, q6
    return feats


# ---------------------------------------------------------------------------
# 1. REF 实测 vs 文献 sanity（±0.005）+ REF 钉死
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(LITERATURE_QQ))
def test_ref_measured_vs_literature(name):
    """理想壳经本模块 _q_l 实测 == 文献值（±0.005），REF_GEOMETRY == 实测（1e-3）。"""
    u = ideal_unit_vectors(name)
    assert u.shape[0] == EXPECTED_CN[name]
    q4, q6 = _q_l(4, u), _q_l(6, u)
    l4, l6 = LITERATURE_QQ[name]
    assert q4 == pytest.approx(l4, abs=0.005), f"{name} q4={q4:.6f} vs lit {l4}"
    assert q6 == pytest.approx(l6, abs=0.005), f"{name} q6={q6:.6f} vs lit {l6}"
    # REF_GEOMETRY 钉在实测值上（4 位小数，防漂移）
    assert REF_GEOMETRY[name][0] == pytest.approx(q4, abs=1e-3)
    assert REF_GEOMETRY[name][1] == pytest.approx(q6, abs=1e-3)


def test_pairwise_ref_distances_support_margin():
    """新候选对参考距离必须 > T_MATCH，理想点 margin>0.09 → conf≥0.9。"""
    d = lambda a, b: math.dist(REF_GEOMETRY[a], REF_GEOMETRY[b])  # noqa: E731
    # CN=12 三候选两两距离（最近对 fcc–hcp ≈ 0.13 > 0.12）
    assert d("cuboctahedral", "hcp_like") > T_MATCH
    assert d("cuboctahedral", "icosahedral") > T_MATCH
    assert d("hcp_like", "icosahedral") > T_MATCH
    # 理想点 confidence（margin/(margin+LAMBDA)，LAMBDA=0.01）全部 ≥ 0.9
    for name in ("icosahedral", "cuboctahedral", "hcp_like"):
        _, conf = classify_geometry(_row(12, *REF_GEOMETRY[name]))[0]
        assert conf >= 0.9, f"{name} ideal conf={conf:.4f}"
    _, conf = classify_geometry(_row(14, *REF_GEOMETRY["bcc_like"]))[0]
    assert conf >= 0.9, f"bcc_like ideal conf={conf:.4f}"


# ---------------------------------------------------------------------------
# 2. 真实晶体（level1 真键图，λ*=1.20；契约 v1.1/v1.2 图标定）
# ---------------------------------------------------------------------------
REAL_CRYSTALS = [
    # (name, atoms 工厂, 期望 CN, 期望 label)
    ("Al_fcc", lambda: bulk("Al", "fcc", a=4.05, cubic=True), 12, "cuboctahedral"),
    ("Mg_hcp", lambda: bulk("Mg", "hcp", a=3.19, c=3.19 * math.sqrt(8 / 3)), 12, "hcp_like"),
    ("Mg_hcp_experimental_ca", lambda: bulk("Mg", "hcp", a=3.19, c=3.19 * 1.627), 12, "hcp_like"),
    ("Fe_bcc", lambda: bulk("Fe", "bcc", a=2.866, cubic=True), 14, "bcc_like"),
]


@pytest.mark.parametrize("name,make,cn_exp,label_exp", REAL_CRYSTALS)
def test_real_crystal_metallic(name, make, cn_exp, label_exp):
    """理想金属晶体全站点：CN=12→cuboctahedral/hcp_like、CN=14→bcc_like，conf≥0.9。"""
    atoms = make()
    g = build_graph(atoms, lam=1.2)          # level1 真实现（只读依赖）
    cn = np.bincount(g.i, minlength=len(atoms))
    assert (cn == cn_exp).all(), f"{name}: CN={cn.tolist()} != {cn_exp}"
    F = geometry_features(atoms, g)
    out = classify_geometry(F)
    for k, (lab, conf) in enumerate(out):
        assert lab == label_exp, f"{name} site {k}: {lab} (conf={conf:.3f})"
        assert conf >= 0.9, f"{name} site {k}: conf={conf:.3f} < 0.9"
    # 站点 q 落在理想参考上（理想晶体壳 == 理想壳，方向集相同；实验 c/a 的
    # Mg 偏离 0.0017 仍在 5e-3 内——c/a ±3% 鲁棒性，见 progress.md）
    r4, r6 = REF_GEOMETRY[label_exp]
    assert math.hypot(F[0, 12] - r4, F[0, 13] - r6) < 5e-3


def test_bcc_second_neighbors_inside_calibrated_cutoff():
    """bcc_like 的 CN=14 语义：λ*=1.20 校准 cutoff 把 6 个次近邻（轴向 a）圈进
    键图（d_2nd/(r_cov 和) ≈ 1.09 < 1.2），8 个最近邻为立方体顶点。"""
    atoms = bulk("Fe", "bcc", a=2.866, cubic=True)
    rc2 = 2 * covalent_radii[26]
    d_nn, d_2nd = 2.866 * math.sqrt(3) / 2, 2.866
    assert d_2nd / rc2 < 1.2                # 次近邻在 λ* cutoff 内 → 入键图
    assert d_nn / rc2 < d_2nd / rc2 < 1.2
    g = build_graph(atoms, lam=1.2)
    assert (np.bincount(g.i, minlength=len(atoms)) == 14).all()
    # cutoff 收紧到 λ=1.0（只含 8 最近邻 = 立方体顶点）→ CN=8 → cubic 路由，
    # 不是 bcc_like：证明 bcc_like 语义绑定"键图 8+6"而非教科书 CN=8
    g8 = build_graph(atoms, lam=1.0)
    assert (np.bincount(g8.i, minlength=len(atoms)) == 8).all()
    out8 = classify_geometry(geometry_features(atoms, g8))
    assert all(lab == "cubic" and conf >= 0.9 for lab, conf in out8)


# ---------------------------------------------------------------------------
# 3. icosahedral 回归（CN=12 从单候选 → 三候选）
# ---------------------------------------------------------------------------
def test_icosahedral_regression_three_candidate_route():
    """既有 icosahedral fixture 在三候选路由下仍 icosahedral（回归）。"""
    _, (label, conf) = classify_center(make_atoms(FIXTURE_VECS["icosahedral"]))
    assert label == "icosahedral"
    assert conf >= 0.9                      # d2=|ico−hcp|=0.203 → conf≈0.953


# ---------------------------------------------------------------------------
# 4. 中等畸变 fcc：d1 逼近 T_MATCH → cuboctahedral(conf 降低)/ambiguous，
#    绝不高置信 hcp_like/icosahedral（防错判断言）
# ---------------------------------------------------------------------------
DISTORTION_AMPS = (0.12, 0.16, 0.20)        # 顶点抖动（×bond）
DISTORTION_SEEDS = (20260909, 777, 12345)


def test_medium_distortion_fcc_no_misclassification():
    """随机抖动使 d1 逼近 T_MATCH：label ∈ {cuboctahedral, ambiguous}，conf 单调
    低于理想值，且不存在 conf≥0.7 的 hcp_like/icosahedral 站点。"""
    _, (lab0, conf0) = classify_center(make_atoms(FIXTURE_VECS["cuboctahedral"]))
    assert lab0 == "cuboctahedral" and conf0 >= 0.9
    cn12_refs = [REF_GEOMETRY[l] for l in _ROUTES[12]]
    d1_max, labels_seen = 0.0, set()
    for amp in DISTORTION_AMPS:
        for seed in DISTORTION_SEEDS:
            rng = np.random.default_rng(seed)
            atoms = make_atoms(FIXTURE_VECS["cuboctahedral"], perturb=amp, rng=rng)
            F, (label, conf) = classify_center(atoms)
            assert F[0] == 12                          # mock 图保证 CN=12
            d1 = min(math.hypot(F[12] - r4, F[13] - r6) for r4, r6 in cn12_refs)
            d1_max = max(d1_max, d1)
            labels_seen.add(label)
            # 中等畸变：只允许 cuboctahedral（conf 较理想下降）或 ambiguous
            assert label in ("cuboctahedral", "ambiguous"), \
                f"amp={amp} seed={seed}: {label} (conf={conf:.3f}, d1={d1:.4f})"
            assert conf < conf0, \
                f"amp={amp} seed={seed}: conf={conf:.4f} 未低于理想 {conf0:.4f}"
            # 防错判：绝不出现高置信 hcp_like / icosahedral
            assert not (label in ("hcp_like", "icosahedral") and conf >= 0.7)
    # 抖动确实把 d1 推到 T_MATCH 邻域（≥0.6·T_MATCH），而非浅尝辄止
    assert d1_max >= 0.6 * T_MATCH, f"d1_max={d1_max:.4f}"
    # 最大畸变档有 ambiguous 出现（margin 机制实际介入）
    assert "ambiguous" in labels_seen


# ---------------------------------------------------------------------------
# 5. 路由/边界单元测试
# ---------------------------------------------------------------------------
def test_routes_new_entries():
    assert _ROUTES[12] == ("icosahedral", "cuboctahedral", "hcp_like")
    assert _ROUTES[14] == ("bcc_like",)
    # 其余 CN 保持确定性 other 路由（无条目）
    for cn in (0, 1, 7, 9, 10, 11, 13, 15, 16, 20):
        assert _ROUTES.get(cn) is None


def test_fcc_hcp_midpoint_ambiguous():
    """(q4,q6) 落 fcc–hcp 参考中点：margin=0 → ambiguous（宁可不分类）。"""
    f4, f6 = REF_GEOMETRY["cuboctahedral"]
    h4, h6 = REF_GEOMETRY["hcp_like"]
    label, conf = classify_geometry(_row(12, (f4 + h4) / 2, (f6 + h6) / 2))[0]
    assert label == "ambiguous" and conf < 0.7
    # ico–cuboctahedral 中点同理（三候选的另两个边界）
    i4, i6 = REF_GEOMETRY["icosahedral"]
    label, conf = classify_geometry(_row(12, (i4 + f4) / 2, (i6 + f6) / 2))[0]
    assert label == "ambiguous" and conf < 0.7
    label, conf = classify_geometry(_row(12, (i4 + h4) / 2, (i6 + h6) / 2))[0]
    assert label == "ambiguous" and conf < 0.7


def test_cn14_far_from_bcc_ref_is_other():
    """CN=14 但 (q4,q6) 远离 bcc 参考（d1>T_MATCH）→ other。"""
    label, conf = classify_geometry(_row(14, 0.5, 0.5))[0]
    assert label == "other" and conf > 0.7


# CN=7/9/10/11/13/15 的真实（顶点 fixture）确定性路由 → other
_PENT_BIPYRAMID = [[0, 0, 1], [0, 0, -1]] + [
    [math.cos(math.radians(72 * k)), math.sin(math.radians(72 * k)), 0] for k in range(5)]
_TRICAPPED_PRISM = FIXTURE_VECS["trigonal_prismatic"] + [
    [0, 0, 1], [math.cos(math.radians(60)), math.sin(math.radians(60)), -1],
    [math.cos(math.radians(180)), math.sin(math.radians(180)), -1]]
_SQA_PLUS_CAP = FIXTURE_VECS["cubic"][:8] + [[0, 0, 1.5], [0, 0, -1.5]]  # 10 顶点
_ICO_MINUS_1 = FIXTURE_VECS["icosahedral"][:11]
_ICO_PLUS_1 = FIXTURE_VECS["icosahedral"] + [[0.5, 0.5, 0.5]]
_CUBOCTA_PLUS_3 = FIXTURE_VECS["cuboctahedral"] + [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5]]

UNROUTED_SHELLS = [
    (7, "pentagonal_bipyramidal", _PENT_BIPYRAMID),
    (9, "tricapped_trigonal_prism", _TRICAPPED_PRISM),
    (10, "capped_square_antiprism_like", _SQA_PLUS_CAP),
    (11, "icosahedron_minus_1", _ICO_MINUS_1),
    (13, "icosahedron_plus_1", _ICO_PLUS_1),
    (15, "cuboctahedron_plus_3", _CUBOCTA_PLUS_3),
]


@pytest.mark.parametrize("cn,name,vecs", UNROUTED_SHELLS)
def test_unrouted_cn_shells_are_other(cn, name, vecs):
    """CN=7/9/10/11/13/15 理想壳 fixture → other conf=1.0（确定性路由，回归）。"""
    assert len(vecs) == cn, name
    atoms = make_atoms(vecs)
    F, (label, conf) = classify_center(atoms)
    assert F[0] == cn
    assert (label, conf) == ("other", 1.0)


def test_labels_contract_15():
    assert len(GEOMETRY_LABELS) == 15
    assert GEOMETRY_LABELS[10:13] == ["cuboctahedral", "hcp_like", "bcc_like"]
