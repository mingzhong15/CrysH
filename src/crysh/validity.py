"""CKT Level 0 — Validity：q_min / ν / κ / core-overlap 指标、阈值校准、病态结构过滤。

契约：contracts.md §3.4（签名冻结，v1.3 增补键）、§5（flag 语义 v1.3）。
所有权：level0-validity 子任务。

约定：
- 近邻集合 = ASE natural_cutoffs × 1.3（per-atom 半径式 cutoff：包含当且仅当
  d_ij < c_i + c_j，即 d_ij < 1.3·(r_cov_i + r_cov_j)），与 build_bond_graph(lam=1.3)
  的边集合等价；本模块**不 import crysh.bond**（level1 并行开发中），直接使用
  ase.neighborlist。
- q_ij = d_ij / (r_cov_i + r_cov_j)，r_cov 取 ase.data.covalent_radii。
- 无近邻（孤立结构）时 q_min = +inf、n_overlap_pairs = 0（无重叠可判）。
- NaN 一律按"不 flag"处理（保守：不因缺数据而误杀结构）；仅 kappa=+inf（奇异
  cell）视为真病态并 flag。

v1.3 语义（contracts.md v1.3）：
1. **core_overlap（硬 floor，方法失效线）**：d_ij < r_core_i + r_core_j 的原子对，
   r_core 来自 OpenMX PBE19 模守恒赝势的最小伪化半径（src/ckt/data/rcore_openmx_v1.json，
   env CKT_RCORE_TABLE 可覆盖）。物理意义：冻芯区域重叠 → OpenMX/PAO 表示失效，
   与压力无关。这是 L0 唯一的 hard-kill 判据（MatterGen 侧；Alexandria 上记录，
   预期恒为 0）。r_core 缺元素的 pair 跳过（保守不误杀）。
2. **overlap_flag（软 flag，稀有接触）**：q_min < q_c，q_c 由 Alexandria 校准
   （Q_0.001 分位），语义 = "相对参考分布极端"，只记录不拦截（分布偏移指示器）。
3. **ν 拆分**：overpacked_flag（ν < ν_lo，比参考更致密）与 sparse_flag（ν > ν_hi，
   真空/分子/开骨架），均 record-only；extreme_volume_flag = overpacked ∨ sparse
   （schema 兼容别名）。
4. pathological_cell_flag：维持 record-only（合法 slab 形态会触发）。
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import natural_cutoffs, neighbor_list

# --- 契约 §5 pilot 默认阈值 -----------------------------------------------------
DEFAULT_Q_C = 0.85
DEFAULT_VOLUME_NORM_BOUNDS = (0.5, 10.0)
DEFAULT_CELL_KAPPA_MAX = 100.0
DEFAULT_ASPECT_RATIO_MAX = 10.0

# 近邻集合乘子（总计划 Level 0 / 任务书：natural_cutoffs × 1.3）
NEIGHBOR_MULT = 1.3

# core-overlap 搜索半径上界（Å）：r_core 最大约 0.95 Å → pair sum < 1.9 Å
CORE_SEARCH_CUTOFF = 2.0

# 内置 r_core 默认表（OpenMX DFT_DATA19 PBE19，ab22d 项目逐元素 VPS 选择）
_RCORE_DEFAULT_PATH = Path(__file__).resolve().parent / "tables" / "rcore_openmx_v1.json"
ENV_RCORE_TABLE = "CKT_RCORE_TABLE"

METRIC_KEYS = (
    "q_min", "n_overlap_pairs", "volume_norm", "cell_kappa",
    "aspect_ratio", "natom", "volume", "n_core_overlap_pairs",
)
FLAG_KEYS = ("overlap_flag", "extreme_volume_flag", "pathological_cell_flag",
             "overpacked_flag", "sparse_flag", "core_overlap_flag")


@lru_cache(maxsize=4)
def load_rcore_table(path: str | None = None) -> dict[str, float]:
    """逐元素 OpenMX 芯半径表 {symbol: r_core(Å)}。

    path=None → env CKT_RCORE_TABLE → 内置打包表
    （src/ckt/data/rcore_openmx_v1.json；溯源字段见该 JSON 的 source）。
    JSON 兼容两种容器：{"r_core_angstrom": {...}} 或裸 {sym: val}。
    """
    p = path or os.environ.get(ENV_RCORE_TABLE) or str(_RCORE_DEFAULT_PATH)
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("r_core_angstrom"), dict):
        data = data["r_core_angstrom"]
    if not isinstance(data, dict):
        raise ValueError(f"rcore 表格式错误: {p}")
    out: dict[str, float] = {}
    for sym, v in data.items():
        fv = float(v)
        if fv >= 0:  # H 的 r_core=0（无冻芯）是合法值，必须保留
            out[str(sym)] = fv
    return out


def default_thresholds() -> dict[str, float]:
    """契约 §5 的 pilot 默认阈值表（apply_filters 的缺省输入）。"""
    return {
        "q_c": DEFAULT_Q_C,
        "volume_norm_lo": DEFAULT_VOLUME_NORM_BOUNDS[0],
        "volume_norm_hi": DEFAULT_VOLUME_NORM_BOUNDS[1],
        "cell_kappa_max": DEFAULT_CELL_KAPPA_MAX,
        "aspect_ratio_max": DEFAULT_ASPECT_RATIO_MAX,
    }


def _radii(atoms: Atoms) -> np.ndarray:
    """每原子共价半径 (Å)。元素不在 ASE 表内（Z≥119）时给 NaN。"""
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    out = np.empty(len(z), dtype=float)
    for idx, zi in enumerate(z):
        out[idx] = covalent_radii[zi] if zi < len(covalent_radii) else np.nan
    return out


def _canonical_unique_edges(i: np.ndarray, j: np.ndarray, s: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """无序去重边：ASE 返回双向 (i,j,S)/(j,i,-S)，规范化为 (i≤j) 单边。

    返回 (unique_keys (E_u,5), inverse (E,))；keys 列为 [i, j, Sx, Sy, Sz]。
    i==j 的周期自像 (i,i,S)/(i,i,-S) 按 S 的首个非零分量符号规范化合并。
    """
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64).reshape(-1, 3)
    swap = i > j
    ii = np.where(swap, j, i)
    jj = np.where(swap, i, j)
    ss = np.where(swap[:, None], -s, s)
    # i==j：规范化平移符号
    same = ii == jj
    if same.any():
        nz = ss != 0
        first_nz = np.argmax(nz, axis=1)          # 全零行 → 0，值 0 不翻转
        vals = ss[np.arange(len(ss)), first_nz]
        flip = same & (vals < 0)
        ss = np.where(flip[:, None], -ss, ss)
    keys = np.column_stack([ii, jj, ss])
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    return unique, inverse


def _core_overlap_pairs(atoms: Atoms, rcore: dict[str, float] | None = None) -> int:
    """d_ij < r_core_i + r_core_j 的去重原子对数（OpenMX 冻芯重叠，方法失效线）。

    搜索半径 = CORE_SEARCH_CUTOFF（2.0 Å > 任何 pair 的 r_core 和上界 1.9 Å），
    标量 cutoff 的 ASE neighbor_list（O(N)）。r_core 表缺某元素 → 该 pair 跳过
    （保守：不因缺数据误杀）。i==j 周期自像同样计入（跨胞自重叠）。
    """
    if rcore is None:
        rcore = load_rcore_table()
    syms = atoms.get_chemical_symbols()
    if len(atoms) == 0:
        return 0
    i, j, d, s = neighbor_list("ijdS", atoms, CORE_SEARCH_CUTOFF,
                               self_interaction=False)
    if len(d) == 0:
        return 0
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    d = np.asarray(d, dtype=float)
    s = np.asarray(s, dtype=np.int64).reshape(-1, 3)
    si = np.array([rcore.get(s, np.nan) for s in syms], dtype=float)
    # 无向去重（与 q_min 的边集去重同一 helper）
    keys_unique, inverse = _canonical_unique_edges(i, j, s)
    # 每条无向边取最小 d 方向上的距离（对称，任意方向同值）
    d_unique = np.full(len(keys_unique), np.inf)
    np.minimum.at(d_unique, inverse, d)
    floor = si[keys_unique[:, 0].astype(int)] + si[keys_unique[:, 1].astype(int)]
    ok = np.isfinite(floor)
    return int(np.count_nonzero(ok & (d_unique < floor)))


def validity_metrics(atoms: Atoms, q_c: float = 0.85) -> dict[str, Any]:
    """Level 0 指标（contracts.md §3.4；v1.3 增补 n_core_overlap_pairs）。

    返回 keys（顺序固定）：q_min, n_overlap_pairs, volume_norm, cell_kappa,
    aspect_ratio, natom, volume, n_core_overlap_pairs。

    - q_min：近邻集合（natural_cutoffs×1.3）内 min_ij d_ij/(r_cov_i+r_cov_j)；
      空集合 → +inf。
    - n_overlap_pairs：近邻集合内 q < q_c 的**无序去重**原子对数。
    - volume_norm ν = V / Σ_i (4π/3) r_cov_i³。
    - cell_kappa = np.linalg.cond(cell 矩阵)（奇异 → +inf）。
    - aspect_ratio = 晶胞边长 a_max / a_min（a_min≤0 → +inf）。
    - n_core_overlap_pairs：d < r_core_i+r_core_j 对数（OpenMX 冻芯重叠，
      见 _core_overlap_pairs；v1.3）。
    """
    z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
    radii = _radii(atoms)

    # --- 近邻集合：per-atom 半径式 cutoff（等价 build_bond_graph(lam=1.3) 边集）--
    cutoffs = natural_cutoffs(atoms, mult=NEIGHBOR_MULT)  # [1.3·r_cov_i]
    i, j, d, s = neighbor_list("ijdS", atoms, list(cutoffs), self_interaction=False)
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        q_min = float("inf")
        n_overlap = 0
    else:
        q = d / (radii[i] + radii[j])  # 无序对，r_i+r_j 对称，方向无关
        keys_unique, inverse = _canonical_unique_edges(i, j, s)
        q_unique = np.full(len(keys_unique), np.inf)
        np.minimum.at(q_unique, inverse, q)          # 同一无向边取 q（重复方向同值）
        q_min = float(np.min(q_unique))
        n_overlap = int(np.count_nonzero(q_unique < q_c))

    # --- core overlap（v1.3：OpenMX 冻芯重叠硬 floor） ---------------------------
    n_core_overlap = _core_overlap_pairs(atoms)

    # --- 体积 / 归一化体积 -------------------------------------------------------
    volume = float(atoms.get_volume())
    sum_sphere = float(np.sum((4.0 * np.pi / 3.0) * radii ** 3))  # NaN 元素会污染
    if np.isnan(sum_sphere):
        volume_norm = float("nan")
    else:
        volume_norm = volume / sum_sphere if sum_sphere > 0 else float("inf")

    # --- cell 病态指标 -----------------------------------------------------------
    cell = np.asarray(atoms.get_cell(), dtype=float)
    if cell.shape == (3, 3):
        with np.errstate(all="ignore"):
            kappa = float(np.linalg.cond(cell))
        if not np.isfinite(kappa):
            kappa = float("inf")
    else:
        kappa = float("inf")
    lengths = np.linalg.norm(cell, axis=1)
    a_min, a_max = float(np.min(lengths)), float(np.max(lengths))
    aspect = a_max / a_min if a_min > 0 else float("inf")

    return {
        "q_min": q_min,
        "n_overlap_pairs": n_overlap,
        "volume_norm": volume_norm,
        "cell_kappa": kappa,
        "aspect_ratio": aspect,
        "natom": int(len(atoms)),
        "volume": volume,
        "n_core_overlap_pairs": n_core_overlap,
    }


def calibrate_thresholds(metrics_df, q_quantile: float = 0.001,
                         vol_quantiles: tuple[float, float] = (0.001, 0.999)
                         ) -> dict[str, Any]:
    """由指标分布自校准 L0 阈值（contracts.md §3.4）。

    保守解释（progress.md 已记录）：
    - q_c = min(0.85, Q(q_quantile)[q_min]) —— 校准值只可能**放宽**默认阈值，
      绝不多 flag 结构。
    - volume_norm 界 = [min(0.5, Q_lo), max(10, Q_hi)] —— 取默认区间与数据分位
      区间的**并集**（更宽 → 更少误杀）。
    - cell_kappa / aspect 无校准参数（签名内无对应 quantile），保持契约默认。

    metrics_df 需含列 q_min、volume_norm（±inf/NaN 自动剔除）。
    """
    q = np.asarray(metrics_df["q_min"], dtype=float)
    q = q[np.isfinite(q)]
    nu = np.asarray(metrics_df["volume_norm"], dtype=float)
    nu = nu[np.isfinite(nu)]

    q_quantile = float(np.clip(q_quantile, 0.0, 1.0))
    v_lo_q, v_hi_q = float(vol_quantiles[0]), float(vol_quantiles[1])

    if len(q) >= 2:
        q_raw = float(np.quantile(q, q_quantile))
        q_c = min(DEFAULT_Q_C, q_raw)
    else:
        q_raw, q_c = None, DEFAULT_Q_C
    if len(nu) >= 2:
        nu_raw_lo = float(np.quantile(nu, v_lo_q))
        nu_raw_hi = float(np.quantile(nu, v_hi_q))
        v_lo = min(DEFAULT_VOLUME_NORM_BOUNDS[0], nu_raw_lo)
        v_hi = max(DEFAULT_VOLUME_NORM_BOUNDS[1], nu_raw_hi)
    else:
        nu_raw_lo = nu_raw_hi = None
        v_lo, v_hi = DEFAULT_VOLUME_NORM_BOUNDS

    return {
        # 生效阈值（apply_filters 直接消费）
        "q_c": q_c,
        "volume_norm_lo": v_lo,
        "volume_norm_hi": v_hi,
        "cell_kappa_max": DEFAULT_CELL_KAPPA_MAX,
        "aspect_ratio_max": DEFAULT_ASPECT_RATIO_MAX,
        # 校准溯源
        "q_quantile": q_quantile,
        "vol_quantiles": [v_lo_q, v_hi_q],
        "q_min_at_quantile_raw": q_raw,
        "volume_norm_at_quantiles_raw": [nu_raw_lo, nu_raw_hi],
        "n_q_min_used": int(len(q)),
        "n_volume_norm_used": int(len(nu)),
        "n_total": int(len(metrics_df)),
        "defaults": default_thresholds(),
        "conservative_clamping": (
            "q_c = min(0.85, Q_0.001); volume bounds = union([0.5,10], quantiles) "
            "→ 校准只放宽、不收紧默认阈值（少误杀优先）"
        ),
    }


def apply_filters(metrics: dict[str, Any],
                  thresholds: dict[str, Any] | None = None) -> dict[str, bool]:
    """按阈值表输出 L0 flag（contracts.md §3.4 / §5，v1.3 语义）。

    thresholds 缺省（None 或缺 key）时使用契约 §5 默认规则：
      core_overlap_flag     = n_core_overlap_pairs > 0（OpenMX 冻芯重叠，硬 floor，
                              唯一 hard-kill 判据；Alexandria 上预期恒 False）
      overlap_flag          = q_min < q_c（默认 0.85；校准后 ≈0.64——稀有接触，只记录）
      overpacked_flag       = volume_norm < lo（默认 0.5——比参考更致密，只记录）
      sparse_flag           = volume_norm > hi（默认 10——真空/分子/开骨架，只记录，交 L2）
      extreme_volume_flag   = overpacked ∨ sparse（schema 兼容别名）
      pathological_cell_flag = cell_kappa > 100 或 aspect_ratio > 10（只记录）
    NaN 指标 → 不 flag（保守）；kappa=+inf（奇异 cell）→ pathological=True。
    """
    if thresholds is None:
        thresholds = {}
    q_c = float(thresholds.get("q_c", DEFAULT_Q_C))
    v_lo = float(thresholds.get("volume_norm_lo", DEFAULT_VOLUME_NORM_BOUNDS[0]))
    v_hi = float(thresholds.get("volume_norm_hi", DEFAULT_VOLUME_NORM_BOUNDS[1]))
    k_max = float(thresholds.get("cell_kappa_max", DEFAULT_CELL_KAPPA_MAX))
    a_max = float(thresholds.get("aspect_ratio_max", DEFAULT_ASPECT_RATIO_MAX))

    q_min = metrics.get("q_min", float("inf"))
    nu = metrics.get("volume_norm", float("nan"))
    kappa = metrics.get("cell_kappa", float("nan"))
    aspect = metrics.get("aspect_ratio", float("nan"))
    n_core = int(metrics.get("n_core_overlap_pairs", 0) or 0)

    overlap = bool(np.isfinite(q_min)) and float(q_min) < q_c
    overpacked = bool(np.isfinite(nu)) and float(nu) < v_lo
    sparse = bool(np.isfinite(nu)) and float(nu) > v_hi
    # NaN 与 inf 比较为 False；inf kappa 天然 > k_max → True（奇异 cell 真病态）
    pathological = (kappa > k_max) or (aspect > a_max)

    return {
        "core_overlap_flag": n_core > 0,
        "overlap_flag": overlap,
        "overpacked_flag": overpacked,
        "sparse_flag": sparse,
        "extreme_volume_flag": overpacked or sparse,
        "pathological_cell_flag": pathological,
    }


def distribution_shift_report(ref_df, test_df, thresholds: dict | None = None,
                              rcore: dict[str, float] | None = None) -> dict[str, Any]:
    """Alexandria（ref）vs MatterGen（test）L0 分布一致性报告（v1.3）。

    输入：ref_df / test_df 需含列 q_min、volume_norm、cell_kappa、aspect_ratio、
    n_core_overlap_pairs（NaN/inf 自动剔除，逐指标）。

    返回每指标：
      ref_quantiles  {q: {q001,q01,q05,q50,q95,q99,q999}}  参考分布分位（MatterGen 的判据）
      test_tail_fraction  P(X < ref_q001) 与 P(X > ref_q999)（落在参考极值外的比例）
      ks  scipy.stats.ks_2samp 的 {statistic, pvalue}（分布整体差异；scipy 缺则 None）
      flag_rate_ref / flag_rate_test（apply_filters 各 flag 占比）
    """
    cols = ["q_min", "volume_norm", "cell_kappa", "aspect_ratio"]
    quantile_grid = [0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999]
    rep: dict[str, Any] = {"quantile_grid": quantile_grid}
    ref_finite: dict[str, np.ndarray] = {}
    test_finite: dict[str, np.ndarray] = {}
    for c in cols:
        r = np.asarray(ref_df[c], dtype=float)
        t = np.asarray(test_df[c], dtype=float)
        ref_finite[c] = r[np.isfinite(r)]
        test_finite[c] = t[np.isfinite(t)]
        qs = [float(np.quantile(ref_finite[c], q)) for q in quantile_grid] \
            if len(ref_finite[c]) >= 2 else []
        lo, hi = (qs[0], qs[-1]) if qs else (np.nan, np.nan)
        n_t = len(test_finite[c])
        tail_lo = float(np.mean(test_finite[c] < lo)) if n_t and np.isfinite(lo) else np.nan
        tail_hi = float(np.mean(test_finite[c] > hi)) if n_t and np.isfinite(hi) else np.nan
        ks = None
        if len(ref_finite[c]) >= 2 and len(test_finite[c]) >= 2:
            try:
                from scipy.stats import ks_2samp  # type: ignore
                s, p = ks_2samp(ref_finite[c], test_finite[c])
                ks = {"statistic": float(s), "pvalue": float(p)}
            except Exception:
                pass
        rep[c] = {
            "ref_quantiles": {f"q{q}": v for q, v in zip(quantile_grid, qs)} if qs else {},
            "test_tail_fraction": {"below_ref_q001": tail_lo, "above_ref_q999": tail_hi},
            "ks_2samp": ks,
        }
    # flag 占比（两侧同阈值）
    thr = thresholds or {}
    for name, df in (("ref", ref_df), ("test", test_df)):
        rates: dict[str, float] = {}
        for c in cols:
            rates[c] = float(np.isfinite(np.asarray(df[c], dtype=float)).mean())
        rates["core_overlap_rate"] = float(
            (np.nan_to_num(np.asarray(df["n_core_overlap_pairs"], dtype=float),
                           nan=0.0) > 0).mean()
        ) if "n_core_overlap_pairs" in df.columns else np.nan
        rep[f"finite_rate_{name}"] = rates
    return rep
