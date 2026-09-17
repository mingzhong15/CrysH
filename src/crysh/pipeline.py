"""CKT 全链 pipeline（postproc-atlas 所有）。

把 L0(validity) → L1(dimension) → L2(morphology) → L3(coord) → L4(geometry) → L5(token)
串成每结构一条 record，列严格按 `ckt.schema.RECORD_COLUMNS`（contracts.md §4）。

开发期处理：
- 每级 `try: from ckt.xxx import ...` 真模块；`except ImportError` 落到本文件
  "===== MOCK 层 =====" 段的等价实现（集中放置，真模块合入后整段替换）。
- mock 层优先读取各级子任务已产出的 parquet（按 structure_id 逐行 lookup）：
  level0-validity/data/validity_2k.parquet、level1-dimensionality/data/dimspectrum_2k.parquet、
  level2-morphology/data/morphology_2k.parquet、level3-coordination/data/coord_2k.parquet、
  level4-geometry/data/geometry_2k.parquet、level5-tokenizer/data/tokens_2k.parquet；
  读不到的用分布合理的 mock 值填充，并在 quality_flags 标 "mock_Lx"（x=级号）。
- L0 的 mock 实现按契约公式本地真算（q_min / volume_norm / cell_kappa / aspect_ratio），
  非合成随机数；因未经过真 validity 模块，provenance 上仍标 "mock_L0"。
- §4 record 无 per-element CN/geometry 与 l3 token 明细列（fig3/4/5 需要），
  run_batch / make_mock_records 在 records 旁写侧文件：
    <stem>_cn_counts.parquet    (element, cn, count)
    <stem>_geom_counts.parquet  (element, geometry, count)
    <stem>_l3_tokens.parquet    (structure_id, token)  去重后的 l3 token 长表
  record schema 保持冻结不动。
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import read as ase_read

from ckt.schema import LAMBDA_COLS, RECORD_COLUMNS

from .paths import KT_ROOT, LEVELS_DIR  # noqa: F401  (工作目录 / 各级数据根，见 paths.py)
LAMBDAS = tuple(LAMBDA_COLS.keys())

# ---------------------------------------------------------------------------
# 真模块探测（开发期大部分模块未合入 → ImportError → mock）
# ---------------------------------------------------------------------------
try:
    from ckt.validity import apply_filters as _real_apply_filters
    from ckt.validity import validity_metrics as _real_validity_metrics
    _HAVE_L0 = True
except ImportError:  # pragma: no cover - 真模块合入前的常态
    _HAVE_L0 = False

try:
    from ckt.bond import build_bond_graph as _real_build_bond_graph
    _HAVE_BOND = True
except ImportError:  # pragma: no cover
    _HAVE_BOND = False

try:
    from ckt.dimension import dimensionality_spectrum as _real_dimensionality_spectrum
    _HAVE_L1 = True
except ImportError:  # pragma: no cover
    _HAVE_L1 = False

try:
    from ckt.morphology import morphology as _real_morphology
    # v1.3-L2：诊断扩展键（span/n_layers/f_max/cn_std/vacuum_*/slab_*/mono_* score）
    from ckt.morphology import morphology_with_diagnostics as _real_morph_diag
    _HAVE_L2 = True
except ImportError:  # pragma: no cover
    _HAVE_L2 = False
    _real_morph_diag = None

try:
    from ckt.coord import coordination as _real_coordination
    _HAVE_L3 = True
except ImportError:  # pragma: no cover
    _HAVE_L3 = False

try:
    from ckt.geometry import classify_geometry as _real_classify_geometry
    from ckt.geometry import geometry_features as _real_geometry_features
    _HAVE_L4 = True
except ImportError:  # pragma: no cover
    _HAVE_L4 = False

try:
    from ckt.tokenize import motif_tokens as _real_motif_tokens
    _HAVE_L5 = True
except ImportError:  # pragma: no cover
    _HAVE_L5 = False

# L0 阈值 canonical 化（contracts.md v1.3）：pipeline 必须消费 level0 校准阈值
# 文件，缺文件/坏文件回退 validity 模块默认（契约 §5 pilot 默认）。校准后
# q_c≈0.643 取代默认 0.85——0.85 恰落在重过渡金属的金属键接触线上（bcc W
# q=0.846），会误拦大量正常金属结构（level0-validity/progress.md、contracts v1.3）。
_L0_THRESHOLDS_PATH = LEVELS_DIR / "level0-validity" / "data" / "thresholds_v1.json"


def _scale() -> str:
    """当前数据规模后缀（CKT_SCALE env，默认 2k）——集群 20k 链 sbatch 里导出。"""
    return os.environ.get("CKT_SCALE", "2k")


_CN_TABLE: Any = False  # False=未加载；None=无表；dict=已加载（fork worker 继承）


def _default_cn_table() -> dict | None:
    """L3 P(CN|Z) 校准表：CKT_CN_TABLE 显式 > CKT_SCALE 推导
    （level3-coordination/data/cn_table_{scale}.json；2k 契约名 cn_table_v1.json）。

    动机（bugfix 2026-09-10）：records 真模块路径原以 cn_table=None 计算
    percentile → 全 NaN → low/high_cn_fraction 恒 0（20k/全量 records 实证；
    L3 脚本写的 parquet 本身正确，但被真模块路径抢先覆盖）。"""
    global _CN_TABLE
    if _CN_TABLE is not False:
        return _CN_TABLE
    path_s = os.environ.get("CKT_CN_TABLE")
    if not path_s:
        s = _scale()
        name = "cn_table_v1.json" if s == "2k" else f"cn_table_{s}.json"
        path_s = str(LEVELS_DIR / "level3-coordination" / "data" / name)
    try:
        from ckt.coord import load_cn_table
        _CN_TABLE = load_cn_table(path_s)
        print(f"[pipeline] L3 cn_table 已加载: {path_s}", file=sys.stderr)
    except Exception:
        _CN_TABLE = None
        print(f"[pipeline] L3 cn_table 缺失（{path_s}）→ low/high_cn_fraction 将为 0",
              file=sys.stderr)
    return _CN_TABLE


def _l0_thresholds_path() -> Path:
    """L0 阈值表路径：CKT_L0_THRESHOLDS 显式覆盖 > CKT_SCALE 推导 > 2k 默认。"""
    env = os.environ.get("CKT_L0_THRESHOLDS")
    if env:
        return Path(env)
    s = _scale()
    if s == "2k":
        return _L0_THRESHOLDS_PATH
    return LEVELS_DIR / "level0-validity" / "data" / f"thresholds_v1_{s}.json"


def _load_l0_thresholds() -> dict | None:
    """校准 L0 阈值表（thresholds_v1[_scale].json）→ {key: float}；异常返回 None。"""
    try:
        d = json.loads(_l0_thresholds_path().read_text(encoding="utf-8"))
        keys = ("q_c", "volume_norm_lo", "volume_norm_hi",
                "cell_kappa_max", "aspect_ratio_max")
        out = {k: float(d[k]) for k in keys if k in d}
        return out or None
    except Exception:
        return None

# ===========================================================================
# ============================== MOCK 层 =====================================
# 真模块合入后整段替换（含下方 _mock_* 与 _parquet_lookup 之外的合成器均可删）
# ===========================================================================

# 各级子任务中间 parquet 的预期路径（存在才读；列名映射为 best-effort，
# 真 parquet 到位后按实际列名微调 _LEVEL_COL_MAP）。
# CKT_SCALE≠2k 时 _level_parquet_path() 把 "_2k" 后缀替换为 "_{scale}"
# （fork 多进程继承 env，worker 内调用一致）。
def _level_parquet_path(level: str) -> Path:
    s = _scale()
    if s == "2k":
        return _LEVEL_PARQUET_PATHS[level]
    return Path(str(_LEVEL_PARQUET_PATHS[level]).replace("_2k", f"_{s}"))


_LEVEL_PARQUET_PATHS = {
    "L0": LEVELS_DIR / "level0-validity" / "data" / "validity_2k.parquet",
    "L1": LEVELS_DIR / "level1-dimensionality" / "data" / "dimspectrum_2k.parquet",
    "L2": LEVELS_DIR / "level2-morphology" / "data" / "morphology_2k.parquet",
    "L3": LEVELS_DIR / "level3-coordination" / "data" / "coord_2k.parquet",
    "L4": LEVELS_DIR / "level4-geometry" / "data" / "geometry_2k.parquet",
    "L5": LEVELS_DIR / "level5-tokenizer" / "data" / "tokens_2k.parquet",
}

# record 列 -> 子任务 parquet 内候选列名（best-effort）
_LEVEL_COL_MAP = {
    "L0": {
        "q_min": ["q_min"], "n_overlap_pairs": ["n_overlap_pairs", "overlap_pairs"],
        "volume_norm": ["volume_norm", "nu"], "cell_kappa": ["cell_kappa", "kappa"],
        "aspect_ratio": ["aspect_ratio", "aspect"], "overlap_flag": ["overlap_flag"],
        "extreme_volume_flag": ["extreme_volume_flag"],
        "pathological_cell_flag": ["pathological_cell_flag"],
    },
    "L1": {
        "d_090": ["d_090"], "d_100": ["d_100"], "d_110": ["d_110"],
        "d_120": ["d_120"], "d_135": ["d_135"], "d_150": ["d_150"],
        "d_175": ["d_175"], "d_200": ["d_200"],
        "d_star": ["d_star"], "dim_persistence": ["dim_persistence", "persistence"],
        "n_components": ["n_components"], "dim_components": ["dim_components"],
        "ambiguous_dim_flag": ["ambiguous_dim_flag"],
    },
    "L2": {
        "vacuum_gap": ["vacuum_gap"], "vacuum_fraction": ["vacuum_fraction"],
        "surface_score": ["surface_score"], "porous_candidate": ["porous_candidate"],
        "morphology_class": ["morphology_class", "class_label"],
    },
    "L3": {
        "mean_cn": ["mean_cn"], "low_cn_fraction": ["low_cn_fraction"],
        "high_cn_fraction": ["high_cn_fraction"], "cn_min": ["cn_min"], "cn_max": ["cn_max"],
    },
    "L4": {
        "geom_top_label": ["geom_top_label"], "geom_ambiguous_fraction": ["geom_ambiguous_fraction"],
        "geom_label_counts": ["geom_label_counts"],
    },
    "L5": {
        "n_distinct_l3": ["n_distinct_l3"], "h_neigh_mean": ["h_neigh_mean"],
        "f_corner": ["f_corner"], "f_edge": ["f_edge"], "f_face": ["f_face"],
        "sharing_top_label": ["sharing_top_label"],
        "l3_tokens": ["l3_tokens"],  # 侧聚合用（非 record 列）
    },
}

_parquet_cache: dict[str, dict[str, dict[str, Any]]] = {}


def _as_py_dict(v: Any) -> dict[str, int]:
    """把 parquet 里读回的 dict 类列值（map/struct/dict）统一转 {str: int}。"""
    if v is None:
        return {}
    if isinstance(v, dict):
        d = v
    elif hasattr(v, "as_py"):
        d = dict(v.as_py())
    else:
        d = dict(v)
    return {str(k): int(x) for k, x in d.items()}


def _parquet_lookup(level: str, sid: str) -> dict[str, Any] | None:
    """按 structure_id 查该级子任务 parquet；无文件/无该行 → None。"""
    if not sid:
        return None
    cache_key = f"{level}@{_scale()}"
    if cache_key not in _parquet_cache:
        path = _level_parquet_path(level)
        table: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if "structure_id" not in df.columns:
                    df = df.reset_index().rename(columns={"index": "structure_id"})
                col_map = _LEVEL_COL_MAP[level]
                for rcol, cands in col_map.items():
                    src = next((c for c in cands if c in df.columns), None)
                    if src is not None:
                        df = df.rename(columns={src: rcol})
                for _, row in df.iterrows():
                    entry = {}
                    for rcol in col_map:
                        if rcol not in row or pd.isna(row[rcol]):
                            continue
                        v = row[rcol]
                        if rcol == "geom_label_counts":
                            entry[rcol] = _as_py_dict(v)
                        elif rcol == "l3_tokens":
                            entry[rcol] = [str(x) for x in (list(v) if v is not None else [])]
                        elif rcol in ("overlap_flag", "extreme_volume_flag",
                                      "pathological_cell_flag", "ambiguous_dim_flag",
                                      "porous_candidate"):
                            entry[rcol] = bool(v)
                        elif rcol in ("n_overlap_pairs", "d_090", "d_100", "d_110",
                                      "d_120", "d_135", "d_150", "d_175", "d_200",
                                      "d_star", "n_components",
                                      "cn_min", "cn_max", "n_distinct_l3"):
                            entry[rcol] = int(v)
                        else:
                            entry[rcol] = float(v)
                    if row.get("structure_id") is not None:
                        table[str(row["structure_id"])] = entry
            except Exception as exc:  # 坏 parquet → 当不存在，回落到合成 mock
                print(f"[pipeline] level {level} parquet 读取失败({exc})，回落到合成 mock",
                      file=sys.stderr)
        _parquet_cache[cache_key] = table
    return _parquet_cache[cache_key].get(sid)


# --- 合成 mock 的公共件 ---

# 各元素典型 CN（配位数基线；用于生成分布合理的 CN）
_TYPICAL_CN = {
    1: 1, 2: 0, 3: 4, 4: 4, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0,
    11: 4, 12: 6, 13: 6, 14: 4, 15: 3, 16: 2, 17: 1, 18: 0,
    19: 6, 20: 8, 21: 6, 22: 6, 23: 6, 24: 6, 25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 4,
    31: 4, 32: 4, 33: 3, 34: 2, 35: 1, 36: 0,
    37: 6, 38: 8, 39: 8, 40: 8, 41: 6, 42: 6, 43: 6, 44: 6, 45: 6, 46: 6, 47: 6, 48: 6,
    49: 4, 50: 6, 51: 3, 52: 6, 53: 1, 54: 0,
    55: 8, 56: 8, 57: 9, 58: 9, 59: 9, 60: 9, 61: 9, 62: 9, 63: 9, 64: 9, 65: 9, 66: 9,
    67: 9, 68: 9, 69: 9, 70: 9, 71: 9,
    72: 6, 73: 6, 74: 6, 75: 6, 76: 6, 77: 6, 78: 6, 79: 6, 80: 6, 81: 6, 82: 6, 83: 6,
    84: 6, 85: 0, 86: 0,
    87: 8, 88: 8, 89: 9, 90: 9, 91: 9, 92: 9, 93: 9, 94: 9, 95: 9, 96: 9,
}


def _seed_from_atoms(atoms: Atoms) -> int:
    """结构内容 -> 稳定随机种子（同一结构任何运行都得到相同 mock 值）。"""
    h = hashlib.md5()
    h.update(atoms.get_chemical_formula(mode="metal").encode())
    h.update(np.ascontiguousarray(atoms.cell.array).tobytes())
    h.update(np.ascontiguousarray(atoms.positions).round(6).tobytes())
    return int(h.hexdigest()[:8], 16)


# --- L0：按契约公式本地真算（非合成随机） ---

def _mock_level0(atoms: Atoms, rng: np.random.Generator) -> dict[str, Any]:
    zs = atoms.get_atomic_numbers()
    rc = covalent_radii[zs]
    natom = len(atoms)
    vol = float(atoms.get_volume())
    atom_vol = float((4.0 / 3.0) * np.pi * (rc ** 3).sum())
    nu = vol / atom_vol if atom_vol > 0 else np.nan

    cell = np.asarray(atoms.cell.array, dtype=float)
    norms = np.linalg.norm(cell, axis=1)
    aspect = float(norms.max() / norms.min()) if norms.min() > 0 else np.inf
    try:
        kappa = float(np.linalg.cond(cell))
    except Exception:
        kappa = np.inf

    q_min = np.nan
    n_overlap = 0
    if natom >= 2:
        try:  # ase neighbor list，λ=1.3 的 pair 表（含正反序，防 key 序歧义）
            from ase.neighborlist import neighbor_list
            cutoff = {(int(z1), int(z2)): 1.3 * (covalent_radii[z1] + covalent_radii[z2])
                      for z1 in set(zs.tolist()) for z2 in set(zs.tolist())}
            i, j, d = neighbor_list("ijd", atoms, cutoff=cutoff)
            if len(d):
                q = d / (rc[i] + rc[j])
                q_min = float(q.min())
                n_overlap = int((q < 0.85).sum())
        except Exception:
            # 兜底：最小镜像全对距离（N 小，O(N²) 可接受）
            from ase.geometry import get_distances
            dists, _ = get_distances(atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
            np.fill_diagonal(dists, np.inf)
            rcsum = rc[:, None] + rc[None, :]
            qmat = dists / rcsum
            mask = dists < 1.3 * rcsum
            if mask.any():
                q_min = float(qmat[mask].min())
                n_overlap = int((qmat[mask] < 0.85).sum())

    overlap = bool(np.isfinite(q_min) and q_min < 0.85)
    extreme_vol = bool(not (np.isfinite(nu) and 0.5 <= nu <= 10.0))
    pathological = bool(kappa > 100.0 or aspect > 10.0)
    return {
        "q_min": float(q_min) if np.isfinite(q_min) else q_min,
        "n_overlap_pairs": int(n_overlap),
        "volume_norm": float(nu) if np.isfinite(nu) else nu,
        "cell_kappa": float(kappa) if np.isfinite(kappa) else kappa,
        "aspect_ratio": float(aspect) if np.isfinite(aspect) else aspect,
        # v1.3 新列（mock 路径无 OpenMX r_core 表 → core_overlap 保守记 0/False）
        "n_core_overlap_pairs": 0,
        "core_overlap_flag": False,
        "overlap_flag": overlap,
        "overpacked_flag": bool(np.isfinite(nu) and nu < 0.5),
        "sparse_flag": bool(np.isfinite(nu) and nu > 10.0),
        "extreme_volume_flag": extreme_vol,
        "pathological_cell_flag": pathological,
    }


# --- L1：d(λ) 谱，90% 结构 d_star=3 ---

def _mock_level1(atoms: Atoms, rec: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    nu = rec.get("volume_norm", 2.0)
    aspect = rec.get("aspect_ratio", 1.0)
    if not np.isfinite(nu):
        nu = 2.0
    if not np.isfinite(aspect):
        aspect = 1.0
    u = rng.random()
    # 先用真实 L0 信号选维度族：巨真空→0D、高 aspect→1D、偏稀→2D，否则 90% 3D
    if nu > 8.0:
        fam = 0
    elif aspect > 4.0:
        fam = 1
    elif nu > 4.0:
        fam = 2
    elif u < 0.90:
        fam = 3
    elif u < 0.945:
        fam = 2
    elif u < 0.985:
        fam = 1
    else:
        fam = 0
    # v1.3-L1：specs 扩为 8 槽（网格 0.90..2.00）；λ=1.20 槽（索引 3）恒 = fam，
    # 保证 d_star==fam。低维族的"长 λ 升 3"（层间/链间/分子间弱键）落在 1.75 起。
    specs = {3: [3, 3, 3, 3, 3, 3, 3, 3], 2: [2, 2, 2, 2, 3, 3, 3, 3],
             1: [1, 1, 1, 1, 3, 3, 3, 3], 0: [0, 0, 0, 0, 3, 3, 3, 3]}
    dvals = list(specs[fam])
    # v1.1：d_star = d(λ=1.20)（D_STAR_LAMBDA）。specs 保证 λ=1.20 槽恒 = fam；
    # 3D 降维扰动只打 λ≠1.20 的档，保持 d_star==fam。
    idx_120 = list(LAMBDA_COLS.keys()).index(1.20)
    n_lam = len(LAMBDA_COLS)
    if fam == 3:
        v = rng.random()
        if v < 0.10:  # 少量 3D 结构在近程档降维（层内/链内弱连接）
            idx = int(rng.choice([i for i in range(n_lam) if i != idx_120]))
            dvals[idx] = int(rng.choice([2, 1]))
        elif v < 0.16:
            dvals[0] = 2
    persistence = sum(1 for x in dvals if x == fam) / float(n_lam)
    ambiguous = bool(persistence < 0.5)
    if not ambiguous and rng.random() < 0.01:  # 少量强制 ambiguous
        ambiguous = True
        persistence = float(rng.uniform(0.35, 0.49))
    n_components = int(rng.integers(1, 8)) if fam == 0 else 1
    out: dict[str, Any] = {
        "d_star": int(dvals[idx_120]),
        "dim_persistence": round(float(persistence), 4),
        "n_components": n_components,
        # v1.3-L1：mock 只给主分量（fam）一张单分量多标签，保持与真路径同构
        "dim_components": json.dumps(
            {str(fam): [n_components, len(atoms)]}, sort_keys=True),
        "ambiguous_dim_flag": ambiguous,
    }
    for lam, col in LAMBDA_COLS.items():
        out[col] = dvals[list(LAMBDA_COLS.keys()).index(lam)]
    return out


# --- L2：morphology 分类（与 L1 维度族自洽） ---

def _mock_level2(atoms: Atoms, rec: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    fam = int(rec.get("d_star", 3))
    u = rng.random()
    if fam == 3:
        if u < 0.85:
            cls = "dense_bulk"
        elif u < 0.90:
            cls = "layered_bulk"
        elif u < 0.95:
            cls = "porous_candidate"
        elif u < 0.97:
            cls = "surface_like"
        else:
            cls = "ambiguous"
    elif fam == 2:
        if u < 0.40:
            cls = "layered_bulk"
        elif u < 0.70:
            cls = "explicit_vacuum_slab"
        elif u < 0.92:
            cls = "intrinsic_2d_like"
        elif u < 0.97:
            cls = "surface_like"
        else:
            cls = "ambiguous"
    elif fam == 1:
        if u < 0.65:
            cls = "chain_solid"
        elif u < 0.95:
            cls = "isolated_chain"
        else:
            cls = "ambiguous"
    else:  # fam == 0
        if u < 0.55:
            cls = "molecular_crystal"
        elif u < 0.95:
            cls = "isolated_molecule"
        else:
            cls = "ambiguous"

    vac_ranges = {
        "dense_bulk": (0.0, 1.0), "layered_bulk": (0.5, 3.5),
        "porous_candidate": (1.0, 4.0), "surface_like": (4.0, 12.0),
        "explicit_vacuum_slab": (8.0, 20.0), "intrinsic_2d_like": (0.5, 3.0),
        "chain_solid": (0.0, 1.0), "isolated_chain": (6.0, 15.0),
        "molecular_crystal": (0.5, 3.0), "isolated_molecule": (8.0, 20.0),
        "ambiguous": (0.0, 8.0),
    }
    lo, hi = vac_ranges[cls]
    vacuum_gap = float(rng.uniform(lo, hi))
    vacuum_fraction = float(min(0.95, vacuum_gap / (vacuum_gap + 6.0)))
    if cls in ("surface_like", "explicit_vacuum_slab"):
        surface_score = float(rng.uniform(0.6, 0.95))
    elif cls in ("layered_bulk", "intrinsic_2d_like"):
        surface_score = float(rng.uniform(0.2, 0.6))
    elif cls == "dense_bulk":
        surface_score = float(rng.uniform(0.0, 0.25))
    else:
        surface_score = float(rng.uniform(0.0, 0.3))
    return {
        "vacuum_gap": round(vacuum_gap, 4),
        "vacuum_fraction": round(vacuum_fraction, 4),
        "surface_score": round(surface_score, 4),
        "porous_candidate": bool(cls == "porous_candidate"),
        "morphology_class": cls,
    }


# --- L3：元素典型 CN 基线 + 扰动 ---

def _mock_atom_cns(symbols: list[str], rng: np.random.Generator) -> np.ndarray:
    cns = []
    for s in symbols:
        z = int(atomic_numbers[s])
        base = _TYPICAL_CN.get(z, 6)
        c = base + int(rng.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]))
        if rng.random() < 0.05:
            c += int(rng.choice([-2, 2]))
        cns.append(int(min(12, max(0, c))))
    return np.asarray(cns, dtype=int)


def _mock_level3(atoms: Atoms, rng: np.random.Generator) -> tuple[dict[str, Any], np.ndarray]:
    syms = list(atoms.get_chemical_symbols())
    cns = _mock_atom_cns(syms, rng)
    mean_cn = float(cns.mean()) if len(cns) else 0.0
    u = rng.random()
    if u < 0.97:  # 常态：低/高配位占比都很小
        low, high = rng.uniform(0.0, 0.08), rng.uniform(0.0, 0.08)
    elif u < 0.985:  # 少量结构以低配位为主（表面/分子）
        low, high = rng.uniform(0.3, 0.6), rng.uniform(0.0, 0.08)
    else:  # 少量结构以高配位为主
        low, high = rng.uniform(0.0, 0.08), rng.uniform(0.3, 0.6)
    out = {
        "mean_cn": round(mean_cn, 4),
        "low_cn_fraction": round(float(low), 4),
        "high_cn_fraction": round(float(high), 4),
        "cn_min": int(cns.min()) if len(cns) else 0,
        "cn_max": int(cns.max()) if len(cns) else 0,
    }
    return out, cns


# --- L4：CN 路由规则几何（仿真模块 v0 规则版）+ 5% ambiguous ---

def _mock_geom_from_cn(cn: int, rng: np.random.Generator) -> str:
    if cn <= 1:
        return "other"
    if cn == 2:
        return "linear"
    if cn == 3:
        return "trigonal_planar"
    if cn == 4:
        return str(rng.choice(["tetrahedral", "square_planar"], p=[0.8, 0.2]))
    if cn == 5:
        return str(rng.choice(["trigonal_bipyramidal", "square_pyramidal"], p=[0.7, 0.3]))
    if cn == 6:
        return str(rng.choice(["octahedral", "trigonal_prismatic"], p=[0.85, 0.15]))
    if cn == 8:
        return "cubic"
    if cn == 12:  # v1.3-L4：金属标签（fcc 主导）
        return str(rng.choice(["cuboctahedral", "hcp_like", "icosahedral"],
                              p=[0.6, 0.2, 0.2]))
    if cn == 14:  # v1.3-L4：bcc 键图 CN=8+6
        return "bcc_like"
    return "other"


def _mock_level4(atoms: Atoms, cns: np.ndarray,
                 rng: np.random.Generator) -> tuple[dict[str, Any], list[str]]:
    labels = []
    for cn in cns:
        g = _mock_geom_from_cn(int(cn), rng)
        if rng.random() < 0.05:
            g = "ambiguous"
        labels.append(g)
    counts = Counter(labels) or Counter({"other": 0})
    top = counts.most_common(1)[0][0]
    out = {
        "geom_top_label": top,
        "geom_ambiguous_fraction": round(counts.get("ambiguous", 0) / len(labels), 4)
        if labels else 0.0,
        "geom_label_counts": dict(counts),
    }
    return out, labels


# --- L5：l3 token + 邻居化学熵 + sharing ---

def _mock_level5(atoms: Atoms, cns: np.ndarray, geoms: list[str], rec: dict[str, Any],
                 rng: np.random.Generator) -> tuple[dict[str, Any], list[str]]:
    syms = list(atoms.get_chemical_symbols())
    fam = int(rec.get("d_star", 3))
    toks, hvals = [], []
    for s, cn, g in zip(syms, cns, geoms):
        k = int(cn)
        nbr = list(rng.choice(syms, size=min(k, 16), replace=True)) if (k > 0 and syms) else []
        hist = Counter(nbr)
        hist_s = "".join(f"{sym}{c}" for sym, c in sorted(hist.items()))  # 符号字母序
        toks.append(f"{s}|{k}|{g}|{hist_s}")
        if k > 0:
            ps = np.asarray([v / k for v in hist.values()], dtype=float)
            hvals.append(float(-(ps * np.log(ps)).sum()))
        else:
            hvals.append(0.0)
    n_distinct = len(set(toks))
    h_neigh_mean = float(np.mean(hvals)) if hvals else 0.0
    if fam == 0:
        f_c, f_e, f_f = 0.0, 0.0, 0.0
    elif fam == 3:
        f_c, f_e, f_f = rng.uniform(0.55, 0.85), rng.uniform(0.10, 0.35), rng.uniform(0.02, 0.10)
    else:
        f_c, f_e, f_f = rng.uniform(0.30, 0.70), rng.uniform(0.15, 0.45), rng.uniform(0.0, 0.10)
    tot = f_c + f_e + f_f
    if tot > 0:
        f_c, f_e, f_f = f_c / tot, f_e / tot, f_f / tot
        top = max((("corner", f_c), ("edge", f_e), ("face", f_f)), key=lambda t: t[1])[0]
    else:
        top = "isolated"
    out = {
        "n_distinct_l3": int(n_distinct),
        "h_neigh_mean": round(h_neigh_mean, 4),
        "f_corner": round(float(f_c), 4),
        "f_edge": round(float(f_e), 4),
        "f_face": round(float(f_f), 4),
        "sharing_top_label": top,
    }
    return out, toks


# 合成结构用元素池（mock_records.parquet 的假结构）
_MOCK_ELEM_POOLS = {
    "metal": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb",
              "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
              "Au", "Hg", "Al", "Ga", "In", "Sn", "Pb", "Bi"],
    "anion": ["O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P", "As"],
    "light": ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba"],
}


def _synthetic_atoms(i: int, rng: np.random.Generator) -> Atoms:
    """造一个分布合理的假结构（准格子堆积 + 随机扰动），供 mock_records 使用。"""
    n_uni = int(rng.choice([2, 2, 3, 3, 4]))
    if n_uni == 2:
        elems = [str(rng.choice(_MOCK_ELEM_POOLS["metal"])),
                 str(rng.choice(_MOCK_ELEM_POOLS["anion"]))]
    elif n_uni == 3:
        elems = [str(rng.choice(_MOCK_ELEM_POOLS["metal"])),
                 str(rng.choice(_MOCK_ELEM_POOLS["anion"])),
                 str(rng.choice(_MOCK_ELEM_POOLS["light"]))]
    else:
        elems = [str(e) for e in rng.choice(_MOCK_ELEM_POOLS["metal"], size=2, replace=False)] + [
            str(rng.choice(_MOCK_ELEM_POOLS["anion"])),
            str(rng.choice(_MOCK_ELEM_POOLS["light"]))]
    counts = rng.integers(1, 5, size=len(elems)).astype(int)
    if counts.sum() < 2:
        counts[0] += 1
    symbols = []
    for e, c in zip(elems, counts):
        symbols += [e] * int(c)
    n = len(symbols)
    # 准格子堆积：把原子放上 jittered grid，避免随机重叠导致 overlap_flag 泛滥
    g = max(1, int(np.ceil(n ** (1 / 3))))
    idx = rng.choice(g ** 3, size=n, replace=False)
    coords = np.stack(np.unravel_index(idx, (g, g, g)), axis=1).astype(float)
    pos = (coords + rng.uniform(0.15, 0.85, size=(n, 3))) / g
    cell = np.diag(rng.uniform(3.5, 9.0, size=3))
    # 少量结构做成斜晶胞（k 偏高、aspect 偏大，覆盖 L0 flag 分支）
    if rng.random() < 0.15:
        cell = cell + rng.uniform(-1.2, 1.2, size=(3, 3)) * np.tril(np.ones((3, 3)))
    atoms = Atoms(symbols=symbols, scaled_positions=pos, cell=cell, pbc=True)
    atoms.info["structure_id"] = f"mock{i:06d}"
    atoms.info["source_line"] = int(i)
    atoms.info["elements"] = elems
    atoms.info["formula"] = atoms.get_chemical_formula(mode="metal")
    return atoms


# ===========================================================================
# ============================= /MOCK 层 =====================================
# ===========================================================================


def _fill_l0(rec: dict[str, Any], sid: str, atoms: Atoms, rng: np.random.Generator,
             flags: list[str]) -> None:
    if _HAVE_L0:
        try:
            # v1.3 canonical：优先消费校准阈值（q_c≈0.643），缺省回退模块默认；
            # n_overlap_pairs 与 overlap_flag 同尺度（均用校准 q_c 计数）
            thr = _load_l0_thresholds()
            m = _real_validity_metrics(atoms, q_c=thr.get("q_c", 0.85) if thr else 0.85)
            fl = _real_apply_filters(m, thr)
            rec.update({
                "q_min": float(m.get("q_min", np.nan)), "n_overlap_pairs": int(m.get("n_overlap_pairs", 0)),
                "volume_norm": float(m.get("volume_norm", np.nan)),
                "cell_kappa": float(m.get("cell_kappa", np.nan)),
                "aspect_ratio": float(m.get("aspect_ratio", np.nan)),
                "n_core_overlap_pairs": int(m.get("n_core_overlap_pairs", 0)),
                "core_overlap_flag": bool(fl.get("core_overlap_flag", False)),
                "overlap_flag": bool(fl.get("overlap_flag", False)),
                "overpacked_flag": bool(fl.get("overpacked_flag", False)),
                "sparse_flag": bool(fl.get("sparse_flag", False)),
                "extreme_volume_flag": bool(fl.get("extreme_volume_flag", False)),
                "pathological_cell_flag": bool(fl.get("pathological_cell_flag", False)),
            })
            return
        except Exception:  # 真模块异常 → 回落到 mock/parquet
            pass
    pv = _parquet_lookup("L0", sid)
    if pv is not None:
        rec.update(pv)
        return
    rec.update(_mock_level0(atoms, rng))
    flags.append("mock_L0")


def _run_levels(atoms: Atoms, cutoff_table: dict | None = None,
                cn_table: dict | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """链 L0→L5，返回 (record 按 RECORD_COLUMNS 顺序, side 聚合数据)。"""
    sid = str(atoms.info.get("structure_id", ""))
    rng = np.random.default_rng(_seed_from_atoms(atoms))
    rec: dict[str, Any] = {}
    flags: list[str] = []
    side: dict[str, Any] = {"l3_tokens": [], "cn_pairs": Counter(), "geom_pairs": Counter()}
    syms = list(atoms.get_chemical_symbols())

    # ---- 元信息（manifest 提供则用 manifest，否则从 atoms 推导）----
    if atoms.info.get("elements"):
        elements = [str(e) for e in atoms.info["elements"]]
    else:
        elements = sorted(set(syms), key=lambda s: atomic_numbers[s])
    rec["structure_id"] = sid
    rec["source_line"] = atoms.info.get("source_line")
    rec["formula"] = str(atoms.info.get("formula") or atoms.get_chemical_formula(mode="metal"))
    rec["elements"] = elements
    rec["natom"] = int(len(atoms))

    # ---- L0 ----
    _fill_l0(rec, sid, atoms, rng, flags)

    # ---- bond graph（L2/L3/L4/L5 真模块的公共依赖；mock 层不需要）----
    # v1.2: canonical_bond_graph —— Phase-0 校准表（$CKT_CUTOFF_TABLE）在场则
    # table@lam=1.0，否则 interim 共价×λ*=1.20（v1.1 语义，详见 contracts.md v1.2）。
    graph = None
    if _HAVE_BOND:
        try:
            from ckt.bond import canonical_bond_graph
            graph = canonical_bond_graph(atoms, cutoff_table)
        except Exception:
            graph = None

    # ---- L1 ----
    spectrum = None
    l1_ok = False
    if _HAVE_L1:
        try:
            spectrum = _real_dimensionality_spectrum(atoms, cutoff_table)
            rec["d_star"] = int(spectrum.d_star)
            rec["dim_persistence"] = float(spectrum.persistence)
            rec["n_components"] = int(spectrum.n_components)
            rec["ambiguous_dim_flag"] = bool(spectrum.persistence < 0.5)
            for lam, col in LAMBDA_COLS.items():
                rec[col] = int(spectrum.d_by_lambda.get(lam, spectrum.d_star))
            # v1.3-L1：逐分量维度多标签（dict[int,list[int]] → JSON 字符串列）
            rec["dim_components"] = json.dumps(
                {str(k): [int(x) for x in v]
                 for k, v in getattr(spectrum, "dim_components", {}).items()},
                sort_keys=True)
            l1_ok = True
        except Exception:
            l1_ok = False
    if not l1_ok:
        pv = _parquet_lookup("L1", sid)
        if pv is not None:
            rec.update(pv)
            # dimspectrum parquet 无 flag 列 → 按 §5 规则补
            rec.setdefault("ambiguous_dim_flag",
                           bool(rec.get("dim_persistence", 1.0) < 0.5))
        else:
            rec.update(_mock_level1(atoms, rec, rng))
            flags.append("mock_L1")

    # ---- L2 ----
    l2_ok = False
    if _HAVE_L2:
        try:
            res = _real_morphology(atoms, graph=graph, spectrum=spectrum, coord=None)
            rec.update({
                "vacuum_gap": float(res.vacuum_gap), "vacuum_fraction": float(res.vacuum_fraction),
                "surface_score": float(res.surface_score), "porous_candidate": bool(res.porous_candidate),
                "morphology_class": str(res.class_label),
            })
            # v1.3-L2：附加诊断键（additive 列，供 fingerprint/embed_qa 消费；
            # 诊断 dict 失败只丢附加键，不影响冻结字段）
            try:
                diag = _real_morph_diag(atoms, graph=graph, spectrum=spectrum, coord=None)
                for key in ("n_layers", "span", "f_max", "cn_std",
                            "vacuum_score", "slab_score", "mono_score"):
                    if key in diag:
                        rec[key] = float(diag[key]) if isinstance(diag[key], (int, float)) else diag[key]
            except Exception:
                pass
            l2_ok = True
        except Exception:
            l2_ok = False
    if not l2_ok:
        pv = _parquet_lookup("L2", sid)
        if pv is not None:
            rec.update(pv)
        else:
            rec.update(_mock_level2(atoms, rec, rng))
            flags.append("mock_L2")

    # ---- L3 ----
    coord_res = None
    cns: np.ndarray | None = None
    l3_ok = False
    if _HAVE_L3 and graph is not None:
        try:
            coord_res = _real_coordination(
                atoms, graph,
                cn_table if cn_table is not None else _default_cn_table())
            rec.update({
                "mean_cn": float(coord_res.mean_cn),
                "low_cn_fraction": float(coord_res.low_cn_fraction),
                "high_cn_fraction": float(coord_res.high_cn_fraction),
                "cn_min": int(coord_res.cn_min), "cn_max": int(coord_res.cn_max),
                # v1.3-L3：小胞镜像混叠守卫（additive）
                "cn_cell_warning": bool(getattr(coord_res, "cn_cell_warning", False)),
                "cn_cell_margin": float(getattr(coord_res, "cn_cell_margin", float("inf"))),
            })
            cns = np.asarray(coord_res.cn, dtype=int)
            l3_ok = True
        except Exception:
            l3_ok = False
    if not l3_ok:
        pv = _parquet_lookup("L3", sid)
        if pv is not None:
            rec.update(pv)
            cns = _mock_atom_cns(syms, rng)  # 仅用于 fig 侧聚合
        else:
            l3, cns = _mock_level3(atoms, rng)
            rec.update(l3)
            flags.append("mock_L3")
    side["cn_pairs"] = Counter(zip(syms, [int(c) for c in cns]))

    # ---- L4 ----
    geom_labels: list[str] | None = None
    l4_ok = False
    if _HAVE_L4 and graph is not None:
        try:
            feats = _real_geometry_features(atoms, graph)
            raw = _real_classify_geometry(feats)
            geom_labels = ["ambiguous" if (conf < 0.7 or lab == "ambiguous") else lab
                           for lab, conf in raw]
            counts = Counter(geom_labels) or Counter({"other": 0})
            rec.update({
                "geom_top_label": counts.most_common(1)[0][0],
                "geom_ambiguous_fraction": round(counts.get("ambiguous", 0) / len(geom_labels), 4)
                if geom_labels else 0.0,
                "geom_label_counts": dict(counts),
            })
            l4_ok = True
        except Exception:
            l4_ok = False
    if not l4_ok:
        pv = _parquet_lookup("L4", sid)
        if pv is not None:
            rec.update(pv)
            geom_labels = [_mock_geom_from_cn(int(c), rng) for c in cns]  # 仅用于 fig 侧聚合
        else:
            l4, geom_labels = _mock_level4(atoms, cns, rng)
            rec.update(l4)
            flags.append("mock_L4")
    side["geom_pairs"] = Counter(zip(syms, geom_labels))

    # ---- L5 ----
    l3_tokens: list[str] = []
    l5_ok = False
    if _HAVE_L5 and graph is not None and coord_res is not None and geom_labels is not None:
        try:
            toks = _real_motif_tokens(atoms, graph, coord_res, geom_labels)
            l3_tokens = list(toks.get("l3", []))
            h = np.asarray(toks.get("h_neigh", []), dtype=float)
            sh = toks.get("sharing", {})
            rec.update({
                "n_distinct_l3": int(len(set(l3_tokens))),
                "h_neigh_mean": round(float(h.mean()), 4) if len(h) else 0.0,
                "f_corner": round(float(sh.get("corner", 0.0)), 4),
                "f_edge": round(float(sh.get("edge", 0.0)), 4),
                "f_face": round(float(sh.get("face", 0.0)), 4),
                "sharing_top_label": str(toks.get("sharing_label", "")),
            })
            l5_ok = True
        except Exception:
            l5_ok = False
    if not l5_ok:
        pv = _parquet_lookup("L5", sid)
        if pv is not None:
            l3_tokens = [str(x) for x in (pv.pop("l3_tokens", None) or [])]
            rec.update(pv)
        else:
            l5, l3_tokens = _mock_level5(atoms, cns, geom_labels, rec, rng)
            rec.update(l5)
            flags.append("mock_L5")
    side["l3_tokens"] = sorted(set(l3_tokens))

    rec["quality_flags"] = sorted(set(flags))
    return {k: rec[k] for k in RECORD_COLUMNS}, side


def run_structure(atoms: Atoms, cutoff_table: dict | None = None,
                  cn_table: dict | None = None) -> dict[str, Any]:
    """链 L0→L5，返回严格按 schema.RECORD_COLUMNS 顺序的 record dict。"""
    rec, _side = _run_levels(atoms, cutoff_table=cutoff_table, cn_table=cn_table)
    return rec


def _write_outputs(records: list[dict[str, Any]], sides: list[dict[str, Any]],
                   out_parquet: Path) -> None:
    """records -> parquet + 三个 fig 侧文件（cn/geom 计数、l3 token 长表）。"""
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=RECORD_COLUMNS)
    df.to_parquet(out_parquet, index=False)
    stem = out_parquet.stem
    cn_counts: Counter = Counter()
    geom_counts: Counter = Counter()
    token_rows: list[tuple[str, str]] = []
    for rec, side in zip(records, sides):
        cn_counts.update(side["cn_pairs"])
        geom_counts.update(side["geom_pairs"])
        sid = rec["structure_id"]
        token_rows.extend((sid, t) for t in side["l3_tokens"])
    pd.DataFrame([(e, c, n) for (e, c), n in cn_counts.items()],
                 columns=["element", "cn", "count"]).to_parquet(
        out_parquet.parent / f"{stem}_cn_counts.parquet", index=False)
    pd.DataFrame([(e, g, n) for (e, g), n in geom_counts.items()],
                 columns=["element", "geometry", "count"]).to_parquet(
        out_parquet.parent / f"{stem}_geom_counts.parquet", index=False)
    pd.DataFrame(token_rows, columns=["structure_id", "token"]).to_parquet(
        out_parquet.parent / f"{stem}_l3_tokens.parquet", index=False)


def make_mock_records(out_parquet: Path, n: int = 2000, seed: int = 42) -> Path:
    """造 n 条全 mock 合成 record（结构与各 level 值全合成；走同一 _run_levels 代码路径）。"""
    out_parquet = Path(out_parquet)
    rng = np.random.default_rng(seed)
    records, sides = [], []
    for i in range(int(n)):
        atoms = _synthetic_atoms(i, rng)
        rec, side = _run_levels(atoms)
        records.append(rec)
        sides.append(side)
    _write_outputs(records, sides, out_parquet)
    return out_parquet


def _batch_worker(task: tuple[dict[str, Any], str | None]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    row, local_path = task
    try:
        sid = Path(row["path"]).stem
        atoms = ase_read(local_path if local_path else row["path"])
        atoms.info["structure_id"] = sid
        atoms.info["source_line"] = row.get("source_line")
        atoms.info["elements"] = [str(e) for e in (row.get("elements") or [])]
        atoms.info["formula"] = row.get("formula")
        return _run_levels(atoms)
    except Exception:
        return None


def run_batch(manifest: Path, out_parquet: Path, n_proc: int = 8,
              subset_dir: str | Path | None = None) -> None:
    """多进程全量：manifest(jsonl) → records parquet + 侧文件。

    manifest 的 path 字段指向原集群路径；这里按 structure_id（文件名去 .vasp）
    重定位到本地子集目录（subset_dir；默认 KT_ROOT/data/<manifest stem>，如
    data/subset_20k/），本地缺失才回落到 path 字段本身。集群上 CKT_SCALE=20k
    时 _parquet_lookup 自动读 _20k 各级 parquet。
    """
    manifest = Path(manifest)
    out_parquet = Path(out_parquet)
    sdir = Path(subset_dir) if subset_dir is not None else (
        manifest.parent / manifest.stem)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    tasks = []
    for r in rows:
        sid = Path(r["path"]).stem
        local = sdir / f"{sid}.vasp"
        tasks.append((r, str(local) if local.exists() else None))
    max_proc = max(8, int(os.environ.get("CKT_MAX_PROC", "8")))
    n_workers = max(1, min(int(n_proc), max_proc, len(tasks)))
    records: list[dict[str, Any]] = []
    sides: list[dict[str, Any]] = []
    n_failed = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(n_workers) as pool:
        for res in pool.map(_batch_worker, tasks, chunksize=16):
            if res is None:
                n_failed += 1
                continue
            rec, side = res
            records.append(rec)
            sides.append(side)
    if n_failed:
        print(f"[pipeline.run_batch] {n_failed}/{len(tasks)} 结构读取/处理失败，已跳过",
              file=sys.stderr)
    _write_outputs(records, sides, out_parquet)
