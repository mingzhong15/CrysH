"""CrysH 配置与数据契约：**全库唯一的 λ 网格、列名与默认门槛来源**。

为什么要有这个模块
------------------
重构前 λ 网格在四个地方各写了一遍（`dimension` 的 8 点默认值、`morphology` 的 6 点、
`controls` 的 6 点、`figdata` 的 8 点、`schema` 的列名表），任何一处改动都可能让
"同一结构在不同层算出不同维数"。这里把它收敛为一份，其余模块只 import。

数据契约（**冻结**）
------------------
- `RECORD_COLUMNS`：每结构一行记录的全部列，顺序即 parquet 列序；
- `LAMBDA_COLS`：λ → 列名（`d_090` 这种定名列**故意不做成 list 列**，保住 parquet schema）；
- token 格式（`Ti|6|oct|O6`）由 `crysh.tokens` 定义，不在本模块。

修改本文件等于修改对外数据契约：列名只增不改，新增列追加在 `_NEW_COLUMNS` 之后。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "LAMBDAS",
    "LAMBDAS_STAR",
    "D_STAR_LAMBDA",
    "LAMBDA_COLS",
    "lambda_col",
    "RECORD_COLUMNS",
    "MapperConfig",
]

# ── λ 网格（唯一真源）───────────────────────────────────────────────────────
# 8 点网格：L1 的 d(λ) 谱与 L3 的 CN(λ) 谱共用同一网格（契约 v1.3-L1）。
LAMBDAS: tuple[float, ...] = (0.90, 1.00, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00)

# λ*=1.20：canonical 键图口径（d_star、CN*、token 都用它）。
D_STAR_LAMBDA: float = 1.20

# 6 点子网格：L2 morphological 分类只用到 ≤1.50 段（下游兼容保留）。
LAMBDAS_STAR: tuple[float, ...] = (0.90, 1.00, 1.10, 1.20, 1.35, 1.50)


def lambda_col(lam: float) -> str:
    """λ → records 列名（0.90 → ``d_090``）。"""
    return f"d_{int(round(lam * 100)):03d}"


LAMBDA_COLS: dict[float, str] = {lam: lambda_col(lam) for lam in LAMBDAS}

# ── 记录 schema（契约 v1.3，列名冻结）──────────────────────────────────────
RECORD_COLUMNS: list[str] = [
    # 元信息
    "structure_id", "source_line", "formula", "elements", "natom",
    # L0 — validity（v1.3：core_overlap 硬 floor + ν 拆分）
    "q_min", "n_overlap_pairs", "volume_norm", "cell_kappa", "aspect_ratio",
    "n_core_overlap_pairs", "core_overlap_flag",
    "overlap_flag", "overpacked_flag", "sparse_flag",
    "extreme_volume_flag", "pathological_cell_flag",
    # L1 — dimensionality（v1.3-L1：8 点网格 + 逐分量多标签）
    *[LAMBDA_COLS[lam] for lam in LAMBDAS],
    "d_star", "dim_persistence", "n_components", "dim_components", "ambiguous_dim_flag",
    # L2 — morphology（v1.3-L2：诊断列供 fingerprint / embed_qa）
    "vacuum_gap", "vacuum_fraction", "surface_score", "porous_candidate",
    "morphology_class", "n_layers", "span", "f_max", "cn_std",
    "vacuum_score", "slab_score", "mono_score",
    # L3 — coordination（v1.3-L3：小胞镜像混叠守卫）
    "mean_cn", "low_cn_fraction", "high_cn_fraction", "cn_min", "cn_max",
    "cn_cell_warning", "cn_cell_margin",
    # L4 — coordination geometry
    "geom_top_label", "geom_ambiguous_fraction", "geom_label_counts",
    # L5 — tokens / sharing
    "n_distinct_l3", "h_neigh_mean", "f_corner", "f_edge", "f_face",
    "sharing_top_label",
    # 汇总
    "quality_flags",
]


@dataclass(slots=True)
class MapperConfig:
    """一次映射运行的全部可调参数与落盘位置。

    设计原则：**库不认识工作区**。所有路径必须显式给出（或留空表示"不落盘/自动"），
    库内不读环境变量、不推测目录布局。研究工程侧若需要"约定式路径"，
    由调用方（如 `code/workspace.py`）构造好再传进来。

    Parameters
    ----------
    scale:
        数据规模标签（``"2k"``/``"20k"``/``"mgfull"``…），只用于产物命名与查表命中，
        不参与任何物理计算。
    lambdas:
        键图 λ 网格；默认 `LAMBDAS`（8 点，契约 v1.3）。
    d_star_lambda:
        canonical 口径的 λ（默认 1.20）。
    cutoff_table:
        pair cutoff 校准表（`{pair_key: r0}`）；``None`` 时用 interim 共价半径 × λ。
    env_cutoff_table_path:
        校准表 JSON 路径；``None`` 时回落到环境变量 ``CKT_CUTOFF_TABLE``（研究侧兼容）。
    rcore_table_path:
        冻芯半径冻结查表；``None`` 时用包内 `tables/rcore_openmx_v1.json`。
    cn_table:
        P(CN|Z) 分位表；``None`` 表示无表（percentile 记 NaN，不报错）。
    out_dir:
        产物根目录；``None`` 表示只算不落盘。
    level_paths:
        逐级 parquet 路径表（`{"L0": Path, ...}`）；供 records 组装时**按需回读**。
        Mapping 而非固定布局 —— 库不假设 `<root>/levelN-*/data/`。
    l0_thresholds_path / cn_table_path:
        L0 校准阈值与 L3 P(CN|Z) 表的 JSON 路径；``None`` 时用模块默认
        （阈值缺省即 `validity.default_thresholds()`，CN 表缺省即"无表"）。
    """

    scale: str = "2k"
    lambdas: tuple[float, ...] = LAMBDAS
    d_star_lambda: float = D_STAR_LAMBDA
    cutoff_table: dict | None = None
    env_cutoff_table_path: Path | str | None = None
    rcore_table_path: Path | str | None = None
    cn_table: dict | None = None
    l0_thresholds_path: Path | str | None = None
    cn_table_path: Path | str | None = None
    out_dir: Path | str | None = None
    level_paths: dict[str, Path | str] = field(default_factory=dict)

    # ── 便利访问器（避免调用方到处写 Path(...) / dict.get）──────────────────
    def out_path(self) -> Path | None:
        """产物根目录（Path 形式）；未配置则 None。"""
        return Path(self.out_dir) if self.out_dir is not None else None

    def level_path(self, level: str) -> Path | None:
        """某一级的 parquet 路径；未配置则 None。"""
        p = self.level_paths.get(level)
        return Path(p) if p is not None else None

    def lambda_cols(self) -> dict[float, str]:
        """本次配置的 λ → 列名（跟随 `lambdas`，不跟随全局 `LAMBDA_COLS`）。"""
        return {lam: lambda_col(lam) for lam in self.lambdas}

    def with_scale(self, scale: str) -> MapperConfig:
        """返回同参数、换规模标签的新配置（dataclass 不可变风格用法）。"""
        from dataclasses import replace

        return replace(self, scale=scale)
