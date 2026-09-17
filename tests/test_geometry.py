"""Level 4 geometry 单测：理想 fixture 校准 + 规则分类断言。

Mock BondGraph（按 contracts.md §3.2 字段，SimpleNamespace）内联构造，
**双向邻接**（v1.1 契约：每根无向键保留 (i,j,S) 与 (j,i,-S) 两个方向）；
另有直接消费 level1 真实现 ckt.bond.build_bond_graph 的测试（只读依赖）。
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.data import covalent_radii

from crysh.geometry import (
    GEOMETRY_LABELS,
    N_FEATURES,
    REF_GEOMETRY,
    build_graph,
    classify_geometry,
    geometry_features,
)

# ---------------------------------------------------------------------------
# 理想多面体 fixture（中心 + 顶点，单位方向 × 键长）
# v1.3-L4 前置金属扩展：+cuboctahedral（fcc 12 顶点）/hcp_like（hcp 理想
# c/a=√(8/3) 12 近邻壳，上/下 3 近邻 ecliptic）/bcc_like（8+6=14 近邻壳）。
# ---------------------------------------------------------------------------
PHI = (1 + math.sqrt(5)) / 2
HCP_CA = math.sqrt(8.0 / 3.0)               # 理想 hcp c/a

FIXTURE_VECS = {
    "linear":              [[0, 0, 1], [0, 0, -1]],
    "trigonal_planar":     [[1, 0, 0], [-0.5, math.sqrt(3) / 2, 0], [-0.5, -math.sqrt(3) / 2, 0]],
    "tetrahedral":         [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
    "square_planar":       [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
    "trigonal_bipyramidal": [[0, 0, 1], [0, 0, -1], [1, 0, 0],
                             [-0.5, math.sqrt(3) / 2, 0], [-0.5, -math.sqrt(3) / 2, 0]],
    "square_pyramidal":    [[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
    "octahedral":          [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    "trigonal_prismatic":  [[1, 0, 0.8660254], [-0.5, 0.8660254, 0.8660254], [-0.5, -0.8660254, 0.8660254],
                            [1, 0, -0.8660254], [-0.5, 0.8660254, -0.8660254], [-0.5, -0.8660254, -0.8660254]],
    "cubic":               [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)],
    "icosahedral":         [[0, a, b] for a in (-1, 1) for b in (-PHI, PHI)]
                           + [[a, b, 0] for a in (-1, 1) for b in (-PHI, PHI)]
                           + [[a, 0, b] for a in (-PHI, PHI) for b in (-1, 1)],
    # --- v1.3-L4 金属环境（文献 q 值见 THEORY_QQ；REF 由 _q_l 实测校准） ---
    "cuboctahedral":       [[a, b, 0] for a in (-1, 1) for b in (-1, 1)]
                           + [[a, 0, b] for a in (-1, 1) for b in (-1, 1)]
                           + [[0, a, b] for a in (-1, 1) for b in (-1, 1)],
    "hcp_like":            [[math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k)), 0]
                            for k in range(6)]
                           + [[math.cos(math.radians(t)) / math.sqrt(3),
                               math.sin(math.radians(t)) / math.sqrt(3), s * HCP_CA / 2]
                              for s in (1, -1) for t in (90, 210, 330)],
    "bcc_like":            [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
                           + [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                              [0, 0, 1], [0, 0, -1]],
}

# 解析推导（球谐加法定理）的理论值 —— 校准基准，见 calibrate_ideals.py；
# 金属三项为文献值（fcc/hcp/bcc，容差 ±0.005，见 calibrate_metallic_ideals.py）
THEORY_QQ = {
    "linear": (1.0, 1.0),
    "trigonal_planar": (0.375000, 0.740829),
    "tetrahedral": (0.509175, 0.628539),
    "square_planar": (0.829156, 0.586302),
    "trigonal_bipyramidal": (0.625000, 0.455607),
    "square_pyramidal": (0.774597, 0.400000),
    "octahedral": (0.763763, 0.353553),
    "trigonal_prismatic": (0.428571, 0.126981),
    "cubic": (0.509175, 0.628539),
    "icosahedral": (0.000000, 0.663325),
    "cuboctahedral": (0.190945, 0.574524),   # fcc 文献值
    "hcp_like": (0.097221, 0.484761),        # hcp 文献值（理想 c/a）
    "bcc_like": (0.036370, 0.510688),        # bcc 文献值（8+6 等权）
}

BOND = 1.5
CENTER = np.array([5.0, 5.0, 5.0])


def make_atoms(vectors, bond=BOND, center=CENTER, element="C",
               perturb=0.0, rng=None):
    """中心 + 顶点分子（大盒）。perturb>0 时按 bond*perturb 幅值随机扰动顶点。"""
    pos = [center]
    for v in vectors:
        v = np.asarray(v, float)
        v = v / np.linalg.norm(v)
        p = center + v * bond
        if perturb > 0.0:
            r = rng.normal(size=3)
            p = p + r / np.linalg.norm(r) * bond * perturb
        pos.append(p)
    return Atoms(symbols=[element] * (len(vectors) + 1), positions=pos,
                 cell=[[10, 0, 0], [0, 10, 0], [0, 0, 10]], pbc=True)


def mock_graph(atoms, k):
    """中心原子 0 与 1..k 全部成键的 mock BondGraph（§3.2 字段，双向邻接 v1.1）。"""
    d = np.linalg.norm(atoms.positions[1:] - atoms.positions[0], axis=1)
    return SimpleNamespace(
        i=np.concatenate([np.zeros(k, dtype=np.int32),
                          np.arange(1, k + 1, dtype=np.int32)]),
        j=np.concatenate([np.arange(1, k + 1, dtype=np.int32),
                          np.zeros(k, dtype=np.int32)]),
        S=np.zeros((2 * k, 3), dtype=np.int32),
        d=np.concatenate([d, d]).astype(np.float64),
        n_atoms=len(atoms), lam=1.2, cutoff_mode="covalent",
        pair_r0={(6, 6): 2 * covalent_radii[6]})


def classify_center(atoms):
    feats = geometry_features(atoms, mock_graph(atoms, len(atoms) - 1))
    return feats[0], classify_geometry(feats)[0]


# ---------------------------------------------------------------------------
# 契约常量
# ---------------------------------------------------------------------------
def test_contract_constants():
    assert GEOMETRY_LABELS == ["linear", "trigonal_planar", "tetrahedral",
                               "square_planar", "trigonal_bipyramidal",
                               "square_pyramidal", "octahedral",
                               "trigonal_prismatic", "cubic", "icosahedral",
                               "cuboctahedral", "hcp_like", "bcc_like",
                               "other", "ambiguous"]
    assert len(GEOMETRY_LABELS) == 15
    assert N_FEATURES == 14
    assert set(REF_GEOMETRY) <= set(GEOMETRY_LABELS)


# ---------------------------------------------------------------------------
# 特征形状 / 顺序 / 语义（octahedral fixture）
# ---------------------------------------------------------------------------
def test_feature_shape_and_order():
    atoms = make_atoms(FIXTURE_VECS["octahedral"])
    F = geometry_features(atoms, mock_graph(atoms, 6))
    assert F.shape == (7, 14)
    assert F.dtype == np.float64
    # 中心原子
    assert F[0, 0] == 6                       # cn
    r_cov = covalent_radii[6]
    assert F[0, 1] == pytest.approx(BOND / (2 * r_cov), rel=1e-12)   # d_mean/r0
    assert F[0, 2] == 0.0                     # 全等键长 → std=0
    assert F[0, 3] == 1.0                     # d_min/d_mean
    # 角直方图：oct 有 3 对 180°(cos=-1 → bin0)、12 对 90°(cos=0 → bin4)
    assert F[0, 4] == pytest.approx(3 / 15)
    assert F[0, 8] == pytest.approx(12 / 15)
    assert F[0, 4:12].sum() == pytest.approx(1.0)
    # q4/q6
    assert F[0, 12] == pytest.approx(0.763763, abs=2e-3)
    assert F[0, 13] == pytest.approx(0.353553, abs=2e-3)
    # 非中心原子（双向图上 CN=1，只与中心成键）：单矢量 q4=q6=1
    assert (F[1:, 0] == 1).all()
    assert F[1, 12] == pytest.approx(1.0) and F[1, 13] == pytest.approx(1.0)
    assert F[1, 1] == pytest.approx(BOND / (2 * r_cov), rel=1e-12)
    assert F[1, 2] == 0.0 and F[1, 3] == 1.0
    assert (F[1:, 4:12] == 0.0).all()       # CN=1 无键角 → 直方图 0


# ---------------------------------------------------------------------------
# 校准：理想 fixture 实测 q4/q6 == 解析理论值，REF_GEOMETRY 被钉死
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(FIXTURE_VECS))
def test_ideal_q_calibration(name):
    atoms = make_atoms(FIXTURE_VECS[name])
    F, _ = classify_center(atoms)
    t4, t6 = THEORY_QQ[name]
    assert F[12] == pytest.approx(t4, abs=2e-3), f"{name} q4 实测 {F[12]:.6f} vs 理论 {t4}"
    assert F[13] == pytest.approx(t6, abs=2e-3), f"{name} q6 实测 {F[13]:.6f} vs 理论 {t6}"
    # REF_GEOMETRY 与解析理论一致（校准常数被钉死，防漂移）
    assert REF_GEOMETRY[name][0] == pytest.approx(t4, abs=1e-3)
    assert REF_GEOMETRY[name][1] == pytest.approx(t6, abs=1e-3)


# ---------------------------------------------------------------------------
# 分类：理想构型 label 正确且 conf>0.9
# ---------------------------------------------------------------------------
IDEAL_LABELS = {
    "linear": "linear", "trigonal_planar": "trigonal_planar",
    "tetrahedral": "tetrahedral", "square_planar": "square_planar",
    "trigonal_bipyramidal": "trigonal_bipyramidal",
    "square_pyramidal": "square_pyramidal", "octahedral": "octahedral",
    "trigonal_prismatic": "trigonal_prismatic", "cubic": "cubic",
    "icosahedral": "icosahedral",
    "cuboctahedral": "cuboctahedral", "hcp_like": "hcp_like",
    "bcc_like": "bcc_like",
}


@pytest.mark.parametrize("name", sorted(FIXTURE_VECS))
def test_ideal_classification(name):
    atoms = make_atoms(FIXTURE_VECS[name])
    _, (label, conf) = classify_center(atoms)
    assert label == IDEAL_LABELS[name], f"{name}: {label} (conf={conf:.3f})"
    assert conf > 0.9, f"{name} 理想构型 conf={conf:.3f} 应 >0.9"


# ---------------------------------------------------------------------------
# 扰动：5% → conf 下降但仍正确；20% → 只要求输出合法
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(FIXTURE_VECS))
def test_5pct_perturbation(name):
    rng = np.random.default_rng(20240607)
    ideal = make_atoms(FIXTURE_VECS[name])
    _, (label0, conf0) = classify_center(ideal)
    best_pert = None
    # 3 次独立扰动取中位行为，避免单次随机波动
    for _ in range(3):
        pert = make_atoms(FIXTURE_VECS[name], perturb=0.05, rng=rng)
        _, (label1, conf1) = classify_center(pert)
        if best_pert is None or conf1 < best_pert[1]:
            best_pert = (label1, conf1)
    label1, conf1 = best_pert
    assert label1 == label0, f"{name}: 5% 扰动后 label {label1} != {label0}"
    assert conf1 > 0.7, f"{name}: 5% 扰动后 conf={conf1:.3f} 应仍 >0.7"
    assert conf1 < conf0, f"{name}: conf 应下降（{conf0:.4f} -> {conf1:.4f}）"


@pytest.mark.parametrize("name", sorted(FIXTURE_VECS))
def test_20pct_perturbation_runs(name):
    rng = np.random.default_rng(7)
    atoms = make_atoms(FIXTURE_VECS[name], perturb=0.20, rng=rng)
    _, (label, conf) = classify_center(atoms)
    assert label in GEOMETRY_LABELS
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# 特例路由
# ---------------------------------------------------------------------------
def test_linear_vs_bent():
    # 理想 linear：bin0 gate + q 距离
    atoms = make_atoms(FIXTURE_VECS["linear"])
    _, (label, conf) = classify_center(atoms)
    assert label == "linear" and conf > 0.9
    # 120° 弯折 → other（bent 无标签，契约 GEOMETRY_LABELS 冻结）
    atoms = make_atoms([[1, 0, 0], [math.cos(math.radians(120)), math.sin(math.radians(120)), 0]])
    _, (label, conf) = classify_center(atoms)
    assert label == "other" and conf > 0.7


def test_cn0_cn1_other():
    # CN=0（孤原子）
    atoms = make_atoms([])
    F = geometry_features(atoms, mock_graph(atoms, 0))
    assert F.shape == (1, 14) and (F == 0).all()
    assert classify_geometry(F)[0] == ("other", 1.0)
    # CN=1：q4=q6=1 但路由 other
    atoms = make_atoms(FIXTURE_VECS["linear"][:1])
    F, (label, conf) = classify_center(atoms)
    assert F[0] == 1 and F[12] == pytest.approx(1.0) and F[13] == pytest.approx(1.0)
    assert label == "other" and conf == 1.0


def test_unknown_cn_other():
    feats = np.zeros((1, N_FEATURES))
    for cn in (7, 9, 10, 11, 13, 15, 16):
        feats[0, 0] = cn
        assert classify_geometry(feats)[0] == ("other", 1.0)


def test_ambiguous_midpoint():
    """(q4,q6) 落在 tbp/spy 理想中点：margin=0 → ambiguous。"""
    t4, t6 = REF_GEOMETRY["trigonal_bipyramidal"]
    s4, s6 = REF_GEOMETRY["square_pyramidal"]
    feats = np.zeros((1, N_FEATURES))
    feats[0, 0] = 5
    feats[0, 12] = (t4 + s4) / 2
    feats[0, 13] = (t6 + s6) / 2
    label, conf = classify_geometry(feats)[0]
    assert label == "ambiguous"
    assert conf < 0.7


def test_classify_single_row_and_batch():
    feats = np.zeros((3, N_FEATURES))
    feats[:, 0] = [0, 1, 7]
    out = classify_geometry(feats)
    assert len(out) == 3 and all(o == ("other", 1.0) for o in out)
    assert classify_geometry(feats[0]) == [("other", 1.0)]   # 单行 → 单元素 list


def test_octahedral_vs_trigonal_prismatic():
    for name in ("octahedral", "trigonal_prismatic"):
        atoms = make_atoms(FIXTURE_VECS[name])
        _, (label, conf) = classify_center(atoms)
        assert label == IDEAL_LABELS[name] and conf > 0.9


def test_cubic_and_icosahedral():
    for name in ("cubic", "icosahedral"):
        atoms = make_atoms(FIXTURE_VECS[name])
        _, (label, conf) = classify_center(atoms)
        assert label == IDEAL_LABELS[name] and conf > 0.9


def test_real_bidirectional_graph():
    """v1.1：消费 level1 真实现的双向 BondGraph——bincount(g.i) 语义下
    CN 全原子正确、中心特征/分类正确，且不因双向而双计。"""
    from crysh.bond import build_bond_graph

    atoms = make_atoms(FIXTURE_VECS["octahedral"])
    g = build_bond_graph(atoms, lam=1.2)
    assert np.bincount(g.i, minlength=7).tolist() == [6, 1, 1, 1, 1, 1, 1]
    pairs = set(zip(g.i.tolist(), g.j.tolist()))
    assert all((jj, ii) in pairs for ii, jj in pairs)   # 双向保留
    F = geometry_features(atoms, g)
    assert F[0, 0] == 6
    assert F[0, 12] == pytest.approx(0.763763, abs=2e-3)
    assert F[0, 13] == pytest.approx(0.353553, abs=2e-3)
    label, conf = classify_geometry(F)[0]
    assert label == "octahedral" and conf > 0.9
    # 邻居站点：CN=1、q=1（单矢量），features 与 mock 双向图一致
    Fm = geometry_features(atoms, mock_graph(atoms, 6))
    assert np.allclose(F, Fm, atol=1e-9)


def test_ase_fallback_graph_matches_real():
    """回退图（_ase_fallback_graph）双向且与 level1 真实现逐特征一致。"""
    import crysh.geometry as geom
    from crysh.bond import build_bond_graph

    atoms = make_atoms(FIXTURE_VECS["tetrahedral"])
    gf = geom._ase_fallback_graph(atoms, lam=1.2)
    gr = build_bond_graph(atoms, lam=1.2)
    assert np.bincount(gf.i, minlength=5).tolist() == \
        np.bincount(gr.i, minlength=5).tolist()
    assert np.allclose(geometry_features(atoms, gf),
                       geometry_features(atoms, gr), atol=1e-9)
    # 双向保留
    pairs = set(zip(gf.i.tolist(), gf.j.tolist()))
    assert all((jj, ii) in pairs for ii, jj in pairs)


def test_build_graph_default_lam_is_dstar():
    """build_graph/geometry_features 缺省 λ 取 v1.1 interim λ*。"""
    import crysh.geometry as geom
    from crysh.dimensionality import D_STAR_LAMBDA

    assert geom._D_STAR_LAMBDA == D_STAR_LAMBDA == 1.20
    atoms = make_atoms(FIXTURE_VECS["octahedral"])
    g = build_graph(atoms)              # 不传 lam → λ*
    assert g.lam == pytest.approx(1.20)
    F = geometry_features(atoms)        # graph=None → λ* 真实现路径
    assert F[0, 0] == 6
    label, conf = classify_geometry(F)[0]
    assert label == "octahedral" and conf > 0.9


def test_hist_bins_equal_width():
    """8 bins 等宽 [-1,1]，键角散开时占比合理（square_planar: 4×90°,2×180°）。"""
    atoms = make_atoms(FIXTURE_VECS["square_planar"])
    F, _ = classify_center(atoms)
    # sq-planar: 2 对 cos=-1（bin0）、4 对 cos=0（bin4）
    assert F[4] == pytest.approx(2 / 6)
    assert F[8] == pytest.approx(4 / 6)
    assert F[4:12].sum() == pytest.approx(1.0)
