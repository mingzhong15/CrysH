"""level0-validity 单元/冒烟测试（contracts.md §3.4、§5 验收）。

- 内联 fixture：人为 0.5 Å 重叠 → overlap_flag；盒子拉伸 20× → extreme_volume_flag；
  理想 bulk 不触发任何 flag。
- 2k 抽 10 个真实结构冒烟（不依赖批量产出，测试随时可绿）。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.build import bulk
from ase.io import read

from crysh.validity import (
    DEFAULT_ASPECT_RATIO_MAX,
    DEFAULT_CELL_KAPPA_MAX,
    DEFAULT_Q_C,
    DEFAULT_VOLUME_NORM_BOUNDS,
    FLAG_KEYS,
    METRIC_KEYS,
    apply_filters,
    calibrate_thresholds,
    default_thresholds,
    validity_metrics,
)

KT_ROOT = Path(__file__).resolve().parents[2]
SUBSET_DIR = KT_ROOT / "data" / "subset_2k"

NO_FLAGS = {"core_overlap_flag": False, "overlap_flag": False,
            "overpacked_flag": False, "sparse_flag": False,
            "extreme_volume_flag": False, "pathological_cell_flag": False}


# --------------------------------------------------------------------------- #
# 内联 fixture
# --------------------------------------------------------------------------- #
def test_overlap_fixture_triggers_overlap_flag():
    """两 C 原子 0.5 Å（q=0.5/1.52≈0.33 < 0.85，且 < C-C 芯 floor 1.376 Å）→
    overlap_flag + core_overlap_flag，其余不触发。"""
    atoms = Atoms("C2", positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                  cell=[3.0, 3.0, 3.0], pbc=True)
    m = validity_metrics(atoms)
    assert set(m.keys()) == set(METRIC_KEYS)
    assert m["natom"] == 2
    assert m["volume"] == pytest.approx(27.0)
    # q_min = 0.5 / (r_cov(C)+r_cov(C))，ASE r_cov(C)=0.76
    assert np.isfinite(m["q_min"]) and m["q_min"] < DEFAULT_Q_C
    assert m["n_overlap_pairs"] == 1  # 双向边去重后只有 1 个无序对
    # v1.3：0.5 Å < r_core(C)+r_core(C) ≈ 0.688 Å → OpenMX 冻芯重叠硬 floor 触发
    assert m["n_core_overlap_pairs"] == 1
    assert m["aspect_ratio"] == pytest.approx(1.0)
    assert m["cell_kappa"] == pytest.approx(1.0)
    # ν = 27 / (2·4π/3·0.76³) ≈ 7.34 ∈ [0.5, 10] → 不触发 extreme_volume
    assert 0.5 < m["volume_norm"] < 10.0
    flags = apply_filters(m, default_thresholds())
    assert flags == {"core_overlap_flag": True, "overlap_flag": True,
                     "overpacked_flag": False, "sparse_flag": False,
                     "extreme_volume_flag": False, "pathological_cell_flag": False}


def test_stretched_box_triggers_extreme_volume():
    """Si 金刚石 cell 各向同性放大 20×（体积×20）→ sparse（ν=69.8>10），无其他 flag。"""
    atoms = bulk("Si", "diamond", a=5.43)
    atoms.set_cell(atoms.cell * (20.0 ** (1.0 / 3.0)), scale_atoms=True)
    m = validity_metrics(atoms)
    assert m["volume_norm"] > 10.0
    assert m["aspect_ratio"] == pytest.approx(1.0)
    # primitive fcc 型 cell 的 cond=2（远低于病态阈值 100，物理正常）
    assert m["cell_kappa"] < DEFAULT_CELL_KAPPA_MAX
    # 各向同性拉伸后无 1.3×r0 内近邻 → q_min = inf → 不 flag overlap
    assert not np.isfinite(m["q_min"])
    # Si-Si 键 2.35 Å >> r_core(Si)+r_core(Si) ≈ 1.588 Å → 无冻芯重叠
    assert m["n_core_overlap_pairs"] == 0
    flags = apply_filters(m, default_thresholds())
    assert flags == {"core_overlap_flag": False, "overlap_flag": False,
                     "overpacked_flag": False, "sparse_flag": True,
                     "extreme_volume_flag": True, "pathological_cell_flag": False}


def test_ideal_bulk_no_flags():
    """理想 Si 金刚石 bulk 不触发任何 L0 flag。"""
    atoms = bulk("Si", "diamond", a=5.43)
    m = validity_metrics(atoms)
    flags = apply_filters(m, default_thresholds())
    assert flags == NO_FLAGS
    # 数值 sanity：Si-Si d=2.351 Å, r0=2.222 → q≈1.06
    assert np.isfinite(m["q_min"]) and m["q_min"] > DEFAULT_Q_C
    assert 0.5 < m["volume_norm"] < 10.0
    assert m["n_overlap_pairs"] == 0
    assert m["n_core_overlap_pairs"] == 0


def test_core_overlap_openmx_floor():
    """v1.3 硬 floor：d < r_core_i+r_core_j 触发 core_overlap；正常化学键不触发。

    物理锚点（v2 表规则：r_core = 0.5×最小伪化半径，H 无冻芯→0）：
    - bcc W 键长 2.741 Å，r_core(W)=0.344 Å → 芯和 0.688 Å ≪ 键长，纯 W 金属不触发
      （此前 q_c=0.85 会把 bcc W 误判重叠——q=2.741/3.24=0.846）；
    - 正常 C-H 1.09 Å > floor 0.344 Å、H₂ 0.74 Å > 0、C=C 1.34 Å > 0.688 Å 均不触发
      （v1 规则 min r_c 曾把这三类正常键误判为冻芯重叠）。"""
    from crysh.validity import load_rcore_table

    rcore = load_rcore_table()
    assert len(rcore) >= 77
    assert abs(rcore["W"] - 0.3440) < 1e-3
    assert rcore["H"] == 0.0
    # 正常 bcc W：无冻芯重叠、无 overlap（q=0.846 > 校准 q_c=0.643）
    w = bulk("W", "bcc", a=3.1652)
    m = validity_metrics(w)
    assert m["n_core_overlap_pairs"] == 0
    assert m["q_min"] == pytest.approx(0.846, abs=0.01)
    fl = apply_filters(m, {"q_c": 0.643})  # 校准阈值（v1.3 canonical）
    assert fl["core_overlap_flag"] is False and fl["overlap_flag"] is False
    # 正常化学键不触发：C-H 1.096 Å、H₂ 0.77 Å、C=C 1.35 Å
    ch = Atoms("CH4", positions=[[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                                 [0.63, -0.63, -0.63], [-0.63, 0.63, -0.63]],
               cell=[6, 6, 6], pbc=True)
    assert validity_metrics(ch)["n_core_overlap_pairs"] == 0  # C-H 1.096
    h2 = Atoms("H2", positions=[[0, 0, 0], [0.77, 0, 0]], cell=[6, 6, 6], pbc=True)
    assert validity_metrics(h2)["n_core_overlap_pairs"] == 0  # H₂
    cc = Atoms("C2", positions=[[0, 0, 0], [1.35, 0, 0]], cell=[6, 6, 6], pbc=True)
    assert validity_metrics(cc)["n_core_overlap_pairs"] == 0  # C=C
    # 人造冻芯重叠：两 W 相距 0.5 Å（< 0.688 Å floor）
    ww = Atoms("W2", positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
               cell=[4.0, 4.0, 4.0], pbc=True)
    m2 = validity_metrics(ww)
    assert m2["n_core_overlap_pairs"] == 1
    assert apply_filters(m2)["core_overlap_flag"] is True


def test_apply_filters_pathological_and_threshold_override():
    m = {"q_min": 0.90, "volume_norm": 1.0, "cell_kappa": 500.0, "aspect_ratio": 12.0}
    flags = apply_filters(m, default_thresholds())
    assert flags["overlap_flag"] is False
    assert flags["extreme_volume_flag"] is False
    assert flags["pathological_cell_flag"] is True  # kappa>100 或 aspect>10
    # 校准表可以替换阈值
    th2 = dict(default_thresholds(), q_c=0.95)
    assert apply_filters(m, th2)["overlap_flag"] is True
    th3 = dict(default_thresholds(), volume_norm_lo=0.8, volume_norm_hi=0.95)
    assert apply_filters(m, th3)["extreme_volume_flag"] is True  # ν=1.0 > 0.95 → sparse
    assert apply_filters(m, th3)["sparse_flag"] is True
    th4 = dict(default_thresholds(), volume_norm_lo=0.8, volume_norm_hi=1.2)
    assert apply_filters(m, th4)["extreme_volume_flag"] is False


def test_apply_filters_nan_is_conservative():
    """NaN 指标一律不 flag（保守：不因缺数据误杀）；None thresholds 用默认规则。"""
    m = {"q_min": np.nan, "volume_norm": np.nan, "cell_kappa": np.nan,
         "aspect_ratio": np.nan}
    flags = apply_filters(m)  # thresholds=None → 契约 §5 默认
    assert flags == NO_FLAGS
    assert set(flags.keys()) == set(FLAG_KEYS)
    # 奇异 cell（kappa=inf）是真病态 → flag
    m2 = {"q_min": 1.0, "volume_norm": 1.0, "cell_kappa": np.inf,
          "aspect_ratio": 1.0}
    assert apply_filters(m2)["pathological_cell_flag"] is True


def test_distribution_shift_report_smoke():
    """v1.3 分布一致性工具：ref（Alexandria）vs test（MatterGen）冒烟。"""
    from crysh.validity import distribution_shift_report

    rng = np.random.default_rng(7)
    ref = pd.DataFrame({
        "q_min": rng.uniform(0.8, 1.2, 2000),
        "volume_norm": rng.uniform(1.0, 6.0, 2000),
        "cell_kappa": rng.uniform(1.0, 30.0, 2000),
        "aspect_ratio": rng.uniform(1.0, 5.0, 2000),
        "n_core_overlap_pairs": 0,
    })
    test = pd.DataFrame({
        "q_min": np.concatenate([rng.uniform(0.8, 1.2, 900), rng.uniform(0.3, 0.6, 100)]),
        "volume_norm": rng.uniform(1.0, 6.0, 1000),
        "cell_kappa": rng.uniform(1.0, 30.0, 1000),
        "aspect_ratio": rng.uniform(1.0, 5.0, 1000),
        "n_core_overlap_pairs": [0] * 995 + [1] * 5,
    })
    rep = distribution_shift_report(ref, test)
    # test 的 q_min 有 10% 落在 ref Q0.001（≈0.8004）以下 → 尾巴明显
    assert rep["q_min"]["test_tail_fraction"]["below_ref_q001"] > 0.05
    assert "q0.999" in rep["q_min"]["ref_quantiles"]
    assert rep["q_min"]["ks_2samp"] is None or "statistic" in rep["q_min"]["ks_2samp"]
    assert rep["finite_rate_test"]["core_overlap_rate"] == 0.005
    assert rep["finite_rate_ref"]["core_overlap_rate"] == 0.0


def test_calibrate_thresholds_conservative_clamping():
    rng = np.random.default_rng(0)
    q = np.concatenate([rng.uniform(0.7, 1.2, 1998), [0.2, 0.3]])
    nu = np.concatenate([rng.uniform(1.0, 3.0, 1998), [0.05, 50.0]])
    df = pd.DataFrame({"q_min": q, "volume_norm": nu})
    th = calibrate_thresholds(df, q_quantile=0.001, vol_quantiles=(0.001, 0.999))
    # 保守钳制：q_c ≤ 0.85（校准只放宽）；volume 界是 [0.5,10] 与分位的并集
    assert th["q_c"] <= DEFAULT_Q_C
    assert th["q_c"] < 0.7  # Q0.001(2000 样本) ≈ 0.70
    assert th["volume_norm_lo"] <= DEFAULT_VOLUME_NORM_BOUNDS[0]
    assert th["volume_norm_hi"] >= DEFAULT_VOLUME_NORM_BOUNDS[1]
    assert th["cell_kappa_max"] == DEFAULT_CELL_KAPPA_MAX
    assert th["aspect_ratio_max"] == DEFAULT_ASPECT_RATIO_MAX
    for key in ("q_c", "volume_norm_lo", "volume_norm_hi", "cell_kappa_max",
                "aspect_ratio_max"):
        assert key in th
    # inf/NaN 自动剔除
    df2 = df.copy()
    df2.loc[0, "q_min"] = np.inf
    df2.loc[1, "volume_norm"] = np.nan
    th2 = calibrate_thresholds(df2)
    assert th2["n_q_min_used"] == 1999


def test_default_thresholds_matches_contract():
    th = default_thresholds()
    assert th["q_c"] == 0.85
    assert th["volume_norm_lo"] == 0.5 and th["volume_norm_hi"] == 10.0
    assert th["cell_kappa_max"] == 100.0 and th["aspect_ratio_max"] == 10.0


# --------------------------------------------------------------------------- #
# 2k 真实结构冒烟
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not SUBSET_DIR.is_dir(), reason="KT data/subset_2k not available (cluster-only)")
def test_smoke_10_real_structures():
    files = sorted(SUBSET_DIR.glob("*.vasp"))[:10]
    assert len(files) == 10
    for f in files:
        atoms = read(str(f))
        m = validity_metrics(atoms)
        assert set(m.keys()) == set(METRIC_KEYS)
        assert m["natom"] == len(atoms)
        assert m["volume"] == pytest.approx(atoms.get_volume(), rel=1e-9)
        assert m["volume"] > 0.0
        # 近邻集合由 d < 1.3·r0 定义 → q_min 必然 ≤ 1.3
        assert np.isfinite(m["q_min"])
        assert 0.0 < m["q_min"] <= 1.3 + 1e-9
        assert 0.0 < m["volume_norm"] < 200.0
        assert m["aspect_ratio"] >= 1.0
        assert m["cell_kappa"] >= 1.0
        assert isinstance(m["n_overlap_pairs"], (int, np.integer))
        assert m["n_overlap_pairs"] >= 0
        flags = apply_filters(m, default_thresholds())
        assert set(flags.keys()) == set(FLAG_KEYS)
        assert all(isinstance(v, bool) for v in flags.values())
