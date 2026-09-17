"""figdata 导出层（集成者授权 postproc-atlas 所有）。

数据本地化策略配套（用户指令）：
- 原始结构文件与后处理 parquet 只存集群（tj.3k ~/zeng/kt/）；
- 本地只保留 代码 + figdata/（小体积画图聚合数据）+ fig/；
- 画图脚本只读 figdata/，保证本地秒级重调版式风格。

本模块把各级 2k（将来 20k/200M）parquet 聚合为 figdata 画图数据：

    figdata/
      level0/figs.json        q_min/volume_norm 直方图、L0 flag 率、L0 存活曲线、阈值
      level1/figs.json        λ×d 计数矩阵、d* 分布、persistence 直方图、ambiguous 率
      level2/figs.json        morphology 类分布、per-structure 真空散点、vacuum/score 直方图
      level3/figs.json        元素×CN 矩阵、cn_percentile 直方图、异常占比分组
      level4/figs.json        几何标签分布、元素×几何矩阵、confidence 直方图
      level5/figs.json        top-20 l3 token、richness/D_eff、accumulation 曲线
      oracle/figs.json        dim 混淆矩阵、CN 散点、CSM 值、ChemEnv top-20、λ 曲线
      data-subset/figs.json   registry 图 1-4 聚合
      atlas/records_lite.parquet  画图所需数值列（dict 列只留 top-3）
      atlas/cn_counts.json        元素×CN 计数（fig3）
      atlas/geom_counts.json      元素×几何计数（fig4）
      atlas/l3_token_counts.json  l3 token 计数 + 稀有 token 表（fig5/fig7 侧栏）

路径全部参数化：显式 --kt > 环境变量 CKT_ROOT/CKT_WORK > contracts.md 上溯（见 crysh.paths）。
集群上 KT=/thfs4/home/xuyong/zeng/kt 同样可跑。

CLI（KT 根或 --kt）：
    $CKT_PY -m crysh.figdata export-all [--kt DIR]
    $CKT_PY -m crysh.figdata export-level0 [--kt DIR] ...（逐级）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crysh.config import LAMBDA_COLS  # λ 网格唯一真源
from crysh.paths import resolve_kt_root as _resolve_kt_root  # 统一解析，见 paths.py

# 各级默认输入（相对 KT 根；显式参数可覆盖）
_LEVEL_PATHS = {
    "level0": ("level0-validity/data/validity_2k.parquet",
               "level0-validity/data/thresholds_v1.json"),
    "level1": ("level1-dimensionality/data/dimspectrum_2k.parquet",),
    "level2": ("level2-morphology/data/morphology_2k.parquet",),
    "level3": ("level3-coordination/data/cn_sites_2k.parquet",
               "level3-coordination/data/coord_2k.parquet",
               "data/subset_2k.jsonl"),
    "level4": ("level4-geometry/data/geometry_2k.parquet",
               "level4-geometry/data/geometry_sites_2k.parquet"),
    "level5": ("level5-tokenizer/data/token_counts_2k.json",
               "level5-tokenizer/data/metrics_summary.json",
               "level5-tokenizer/data/tokens_2k.parquet"),
    "oracle": ("oracle-bench/data/fast_merged_level1_level3.parquet",
               "oracle-bench/data/oracle_2k.parquet",
               "oracle-bench/data/oracle_2k_chemenv.parquet",
               "oracle-bench/data/cn_lambda_scan.json"),
    "data-subset": ("data/registry/subset_2k.parquet",
                    "data/registry/subset_20k.parquet"),
    "atlas": ("postproc-atlas/data/records_2k.parquet",),
}

LAMBDAS = tuple(LAMBDA_COLS)  # 来自 crysh.config，唯一真源
DIM_LABELS = ["0D", "1D", "2D", "3D"]
ACCUM_GRID = [100, 250, 500, 1000, 1500, 2000]
# scale≠2k 时 accumulation 网格扩展（accumulation 对超出结构数的档位钳制，安全）
ACCUM_GRID_EXT = [100, 250, 500, 1000, 1500, 2000, 4000, 8000, 12000, 16000, 20000]
CN_MAX_BIN = 16

# ---------------------------------------------------------------------------
# scale 维度（20k 规模化；scale="2k" 完全等价旧路径，向后兼容）：
# 其他 scale 按 _SCALE_TEMPLATES 推导输入路径，输出 figs_{scale}.json。
# ---------------------------------------------------------------------------
_SCALE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "level0": ("level0-validity/data/validity_{s}.parquet",
               "level0-validity/data/thresholds_v1_{s}.json"),
    "level1": ("level1-dimensionality/data/dimspectrum_{s}.parquet",),
    "level2": ("level2-morphology/data/morphology_{s}.parquet",),
    "level3": ("level3-coordination/data/cn_sites_{s}.parquet",
               "level3-coordination/data/coord_{s}.parquet",
               "data/subset_{s}.jsonl"),
    "level4": ("level4-geometry/data/geometry_{s}.parquet",
               "level4-geometry/data/geometry_sites_{s}.parquet"),
    "level5": ("level5-tokenizer/data/token_counts_{s}.json",
               "level5-tokenizer/data/metrics_summary_{s}.json",
               "level5-tokenizer/data/tokens_{s}.parquet"),
    "oracle": ("oracle-bench/data/fast_merged_level1_level3_{s}.parquet",
               "oracle-bench/data/oracle_{s}.parquet",
               "oracle-bench/data/oracle_{s}_chemenv.parquet",
               "oracle-bench/data/cn_lambda_scan_{s}.json"),
    # registry 对比恒以 2k 为基线 + 当前 scale（scale=20k 时与 2k 导出同对文件，幂等）
    "data-subset": ("data/registry/subset_2k.parquet",
                    "data/registry/subset_{s}.parquet"),
    "atlas": ("postproc-atlas/data/records_{s}.parquet",),
}


def _scale_paths(level: str, scale: str = "2k") -> tuple[str, ...]:
    """level 输入路径（相对 KT 根）：scale="2k" → 契约原路径；否则模板推导。"""
    if scale == "2k":
        return _LEVEL_PATHS[level]
    return tuple(t.format(s=scale) for t in _SCALE_TEMPLATES[level])


def _figs_json_name(scale: str = "2k") -> str:
    return "figs.json" if scale == "2k" else f"figs_{scale}.json"


def _accum_grid_for(scale: str, n_struct: int) -> list[int]:
    if scale == "2k":
        return ACCUM_GRID
    if n_struct <= ACCUM_GRID_EXT[-1]:
        grid = [g for g in ACCUM_GRID_EXT if g <= max(n_struct, ACCUM_GRID[-1])]
        return grid or ACCUM_GRID
    # 全量规模（>20k）：在扩展网格上按 ×1.6 对数补点到 n_struct
    grid = list(ACCUM_GRID_EXT)
    n = ACCUM_GRID_EXT[-1]
    while n < n_struct:
        n = min(int(n * 1.6), n_struct)
        grid.append(n)
        if n >= n_struct:
            break
    return grid

# atlas records_lite 保留列（画图所需 + 最小上下文；dict 列只留 top-3）
RECORDS_LITE_COLUMNS = [
    "structure_id", "formula", "elements", "natom",
    "q_min", "volume_norm",
    "overlap_flag", "extreme_volume_flag", "pathological_cell_flag",
    "d_star", "dim_persistence", "ambiguous_dim_flag",
    "morphology_class",
    "mean_cn", "low_cn_fraction", "high_cn_fraction",
    "geom_top_label", "geom_ambiguous_fraction", "geom_label_counts",
    "sharing_top_label", "quality_flags",
]


# ---------------------------------------------------------------------------
# 路径与序列化工具
# ---------------------------------------------------------------------------

def resolve_kt_root(kt: str | Path | None = None) -> Path:
    """显式 --kt > 环境变量 CKT_ROOT/CKT_WORK > 标记文件上溯 > 旧深度兜底。

    实际解析逻辑统一在 :mod:`crysh.paths`（全库唯一真相源）；本函数保留是因为
    figdata 的 CLI 约定一直是"``--kt`` 优先"，且有外部调用者用这个名字。
    """
    return _resolve_kt_root(kt).resolve()


def figdata_root(kt: str | Path | None = None) -> Path:
    """figdata 聚合层根目录解析（2026-09-17 E 组重组后四步）。

    重组把 figdata 从 `<KT>/figdata` 移到 `<repo>/01.working/figdata`（跨主题共享层），
    因此不能再简单地拼 `kt/figdata`。优先级：

        1. 显式 ``kt`` 参数（调用方明确指定某工作目录 → 沿用旧语义 `<kt>/figdata`）
        2. 环境变量 ``CKT_FIGDATA``（env.sh 会设；集群 slurm 亦设）
        3. ``<WORKING_DIR>/figdata``（= `01.working/figdata`）
        4. ``<KT>/figdata``（旧布局兜底，兼容集群侧未搬迁的目录）
    """

    if kt is not None:
        return resolve_kt_root(kt) / "figdata"
    env = os.environ.get("CKT_FIGDATA")
    if env:
        return Path(env).expanduser()
    from crysh.paths import WORKING_DIR
    cand = WORKING_DIR / "figdata"
    return cand if cand.is_dir() else resolve_kt_root(None) / "figdata"


def _jdef(o: Any) -> Any:
    """JSON 序列化兜底：非有限 float → None；numpy 标量 → python 标量。"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [_jdef(x) for x in o.tolist()]
    raise TypeError(f"not JSON serializable: {type(o)}")


def _write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=_jdef)
        + "\n",
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load(level: str, figdata_dir: str | Path | None = None,
         kt: str | Path | None = None, scale: str = "2k") -> dict:
    """读 figdata/<level>/figs.json（scale="2k"）/ figs_{scale}.json（画图脚本统一入口）。"""
    root = Path(figdata_dir) if figdata_dir is not None else figdata_root(kt)
    p = root / level / _figs_json_name(scale)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} 不存在 —— 先在数据侧运行 `python -m crysh.figdata export-{level} "
            f"(或 export-all)` 导出 figdata"
        )
    return _read_json(p)


def _require(path: Path, what: str) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{what} 不存在: {path}")
    return pd.read_parquet(path)


def _hist(arr: np.ndarray, bins: Any, rng: tuple[float, float] | None = None,
          density: bool = False) -> dict[str, list[float]]:
    counts, edges = np.histogram(arr, bins=bins, range=rng, density=density)
    return {"counts": [float(c) for c in counts], "edges": [float(e) for e in edges]}


def _round_rows(rows: list[list[float]], nd: int = 4) -> list[list[float]]:
    return [[round(v, nd) if isinstance(v, float) else v for v in r] for r in rows]


# ---------------------------------------------------------------------------
# level0 — validity
# ---------------------------------------------------------------------------

def export_level0(parquet_path: str | Path | None = None,
                  thresholds_path: str | Path | None = None,
                  figdata_dir: str | Path | None = None,
                  kt: str | Path | None = None,
                  scale: str = "2k") -> Path:
    root = resolve_kt_root(kt)
    _paths = _scale_paths("level0", scale)
    df = _require(parquet_path or root / _paths[0], "validity parquet")
    th_path = Path(thresholds_path or root / _paths[1])
    th = _read_json(th_path) if th_path.is_file() else {}

    q = df["q_min"].astype(float)
    q_fin = q[np.isfinite(q)]
    n_inf = int((~np.isfinite(q)).sum())
    qc = float(th.get("q_c", 0.85))
    n_below = int((q_fin < qc).sum())
    qmin_hist = _hist(q_fin.to_numpy(), bins=65, rng=(0.0, 1.3)) if len(q_fin) else \
        {"counts": [0.0] * 65, "edges": [0.0 + i * 1.3 / 65 for i in range(66)]}

    nu = df["volume_norm"].astype(float)
    nu_pos = nu[(np.isfinite(nu)) & (nu > 0)]
    vlo = float(th.get("volume_norm_lo", 0.5))
    vhi = float(th.get("volume_norm_hi", 10.0))
    n_out = int(((nu < vlo) | (nu > vhi)).sum())
    vol_hist: dict[str, Any]
    if len(nu_pos):
        lo_bin = max(float(nu_pos.min()) * 0.9, 1e-3)
        hi_bin = float(nu_pos.max()) * 1.1
        bins = np.logspace(np.log10(lo_bin), np.log10(hi_bin), 61)
        vol_hist = _hist(nu_pos.to_numpy(), bins=bins)
    else:
        vol_hist = {"counts": [0.0] * 60, "edges": [1.0] * 61}

    # v1.3：core_overlap（硬 floor）+ 拆分后的 record-only flag；极端体积别名保留
    flag_keys = ["core_overlap_flag", "overlap_flag", "overpacked_flag",
                 "sparse_flag", "pathological_cell_flag", "extreme_volume_flag"]
    flag_keys = [f for f in flag_keys if f in df.columns]
    rates = {f: float(df[f].fillna(False).astype(bool).mean()) for f in flag_keys}
    any_flag = pd.Series(False, index=df.index)
    for f in flag_keys:
        any_flag |= df[f].fillna(False).astype(bool)
    rates["any_l0_flag"] = float(any_flag.mean())

    # v1.3：存活曲线只对 hard-kill 判据（core_overlap）递减；record-only 不计入
    surv = [len(df)]
    mask = np.ones(len(df), dtype=bool)
    if "core_overlap_flag" in df.columns:
        mask &= ~df["core_overlap_flag"].fillna(False).astype(bool).to_numpy()
        surv.append(int(mask.sum()))

    out = {
        "N": int(len(df)),
        "scale": scale,
        "thresholds": {"q_c": qc, "volume_norm_lo": vlo, "volume_norm_hi": vhi},
        "qmin_hist": qmin_hist, "n_inf": n_inf, "n_below_qc": n_below,
        "vol_hist": vol_hist, "n_out_bounds": int(n_out),
        "flag_rates": rates, "survival": surv, "flag_keys": flag_keys,
    }
    return _write_json(out, Path(figdata_dir or figdata_root(root)) / "level0"
                       / _figs_json_name(scale))


# ---------------------------------------------------------------------------
# level1 — dimensionality
# ---------------------------------------------------------------------------

def export_level1(parquet_path: str | Path | None = None,
                  figdata_dir: str | Path | None = None,
                  kt: str | Path | None = None,
                  scale: str = "2k") -> Path:
    root = resolve_kt_root(kt)
    df = _require(parquet_path or root / _scale_paths("level1", scale)[0],
                  "dimspectrum parquet")
    lam_cols = [f"d_{int(round(lam * 100)):03d}" for lam in LAMBDAS]

    counts = np.zeros((len(LAMBDAS), 4), dtype=int)
    for r, col in enumerate(lam_cols):
        vc = df[col].value_counts()
        for d in range(4):
            counts[r, d] = int(vc.get(d, 0))

    vc_star = df["d_star"].value_counts().sort_index()
    dstar_counts = [int(vc_star.get(d, 0)) for d in range(4)]

    # v1.3-L1：persistence 分母 = 8（网格 8 点）
    n_lam = len(LAMBDAS)
    k = np.rint(df["dim_persistence"].astype(float) * n_lam).astype(int)
    values = np.array([kk / n_lam for kk in sorted(k.unique())])
    counts_p = np.array([int((k == kk).sum()) for kk in sorted(k.unique())])
    frac_ambiguous = float((df["dim_persistence"].astype(float) < 0.5).mean())

    out = {
        "N": int(len(df)),
        "lambdas": list(LAMBDAS),
        "lambda_dim_counts": counts.tolist(),
        "dstar_counts": dstar_counts,
        "persistence_hist": {"values": values.tolist(), "counts": counts_p.tolist()},
        "frac_ambiguous": frac_ambiguous,
    }
    return _write_json(out, Path(figdata_dir or figdata_root(root)) / "level1"
                       / _figs_json_name(scale))


# ---------------------------------------------------------------------------
# level2 — morphology
# ---------------------------------------------------------------------------

def export_level2(parquet_path: str | Path | None = None,
                  figdata_dir: str | Path | None = None,
                  kt: str | Path | None = None,
                  scale: str = "2k") -> Path:
    root = resolve_kt_root(kt)
    df = _require(parquet_path or root / _scale_paths("level2", scale)[0],
                  "morphology parquet")
    from crysh.morphology import MORPHOLOGY_CLASSES  # 只读常量

    counts = df["class_label"].value_counts().reindex(MORPHOLOGY_CLASSES).fillna(0).astype(int)
    present = counts[counts > 0].sort_values(ascending=False)
    class_counts = [[str(c), int(n)] for c, n in present.items()]

    sub = df[df["vacuum_fraction"].notna() & df["vacuum_gap"].notna()]
    scatter = _round_rows(
        [[str(c), float(f), float(g)] for c, f, g in
         zip(sub["class_label"], sub["vacuum_fraction"], sub["vacuum_gap"])]
    )

    # fig2 右侧：stacked vacuum_gap hist —— 导出公共 bins + 每类计数
    cls_present = [c for c in MORPHOLOGY_CLASSES if (df["class_label"] == c).any()]
    vg = df["vacuum_gap"].astype(float).dropna()
    edges = np.histogram_bin_edges(vg.to_numpy(), bins=40) if len(vg) else \
        np.linspace(0, 1, 41)
    per_class = {}
    for c in cls_present:
        vals = df.loc[df["class_label"] == c, "vacuum_gap"].astype(float).dropna()
        cnt, _ = np.histogram(vals.to_numpy(), bins=edges)
        per_class[str(c)] = [int(x) for x in cnt]

    ss = df["surface_score"].astype(float).dropna()
    surface_hist = _hist(ss.to_numpy(), bins=np.arange(0, 1.02, 0.02)) if len(ss) else \
        {"counts": [0.0] * 50, "edges": np.arange(0, 1.02, 0.02).tolist()}

    out = {
        "N": int(len(df)),
        "class_counts": class_counts,
        "scatter": scatter,
        "vacuum_hist": {"edges": [float(e) for e in edges],
                        "per_class": per_class, "order": [str(c) for c in cls_present]},
        "surface_hist": surface_hist,
    }
    return _write_json(out, Path(figdata_dir or figdata_root(root)) / "level2"
                       / _figs_json_name(scale))


# ---------------------------------------------------------------------------
# level3 — coordination
# ---------------------------------------------------------------------------

def export_level3(sites_parquet: str | Path | None = None,
                  records_parquet: str | Path | None = None,
                  manifest_path: str | Path | None = None,
                  figdata_dir: str | Path | None = None,
                  kt: str | Path | None = None,
                  scale: str = "2k") -> Path:
    root = resolve_kt_root(kt)
    _paths = _scale_paths("level3", scale)
    sites = _require(sites_parquet or root / _paths[0], "cn_sites parquet")
    records = _require(records_parquet or root / _paths[1], "coord parquet")
    mani_path = Path(manifest_path or root / _paths[2])

    counts = sites.groupby(["symbol", "cn"]).size()
    matrix = counts.unstack(fill_value=0).reindex(columns=range(CN_MAX_BIN + 1),
                                                  fill_value=0)
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).head(25).index]
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index]
    z_cn = {"rows": [str(s) for s in matrix.index],
            "cols": list(range(CN_MAX_BIN + 1)),
            "data": matrix.to_numpy(dtype=int).tolist()}

    p = sites["cn_percentile"].to_numpy(dtype=float)
    p = p[~np.isnan(p)]
    percentile_hist = _hist(p, bins=50, rng=(0.0, 1.0)) if len(p) else \
        {"counts": [0.0] * 50, "edges": np.linspace(0, 1, 51).tolist()}

    mani = pd.read_json(mani_path, lines=True)
    mani["structure_id"] = mani["path"].map(lambda s: Path(s).stem)
    merged = records.merge(mani[["structure_id", "n_unique_elements"]],
                           on="structure_id", how="left")
    groups = sorted(int(g) for g in merged["n_unique_elements"].dropna().unique())
    n_group = [int((merged["n_unique_elements"] == g).sum()) for g in groups]
    low_frac = [float((merged.loc[merged["n_unique_elements"] == g,
                                  "low_cn_fraction"] > 0.3).mean()) for g in groups]
    high_frac = [float((merged.loc[merged["n_unique_elements"] == g,
                                   "high_cn_fraction"] > 0.3).mean()) for g in groups]

    out = {
        "n_sites": int(len(sites)),
        "z_cn": z_cn,
        "percentile_hist": percentile_hist,
        "n_percentile_sites": int(len(p)),
        "anomaly": {"groups": groups, "n_group": n_group,
                    "low_frac": low_frac, "high_frac": high_frac},
        "N": int(len(merged)),
    }
    return _write_json(out, Path(figdata_dir or figdata_root(root)) / "level3"
                       / _figs_json_name(scale))


# ---------------------------------------------------------------------------
# level4 — geometry
# ---------------------------------------------------------------------------

def export_level4(records_parquet: str | Path | None = None,
                  sites_parquet: str | Path | None = None,
                  figdata_dir: str | Path | None = None,
                  kt: str | Path | None = None,
                  scale: str = "2k") -> Path:
    root = resolve_kt_root(kt)
    _paths = _scale_paths("level4", scale)
    df = _require(records_parquet or root / _paths[0], "geometry parquet")
    sdf = _require(sites_parquet or root / _paths[1], "geometry_sites parquet")
    from ase.data import chemical_symbols  # 只读

    from crysh.geometry import GEOMETRY_LABELS  # 只读常量

    c: Counter = Counter()
    for d in df["geom_label_counts"]:
        for k, v in d.items():
            c[str(k)] += int(v)
    label_counts = [[lab, int(n)] for lab, n in c.most_common()]

    sdf = sdf.copy()
    sdf["symbol"] = sdf["Z"].map(lambda z: chemical_symbols[int(z)])
    top = sdf["symbol"].value_counts().head(25).index.tolist()
    sub = sdf[sdf["symbol"].isin(top)]
    mat = pd.crosstab(sub["symbol"], sub["geom_label"]).reindex(
        index=top, columns=GEOMETRY_LABELS, fill_value=0)
    z_geom = {"rows": [str(s) for s in mat.index],
              "cols": list(GEOMETRY_LABELS),
              "data": mat.to_numpy(dtype=int).tolist()}

    conf = sdf["geom_conf"].astype(float)
    conf_hist = _hist(conf.to_numpy(), bins=50, rng=(0.0, 1.0))
    frac_lt_07 = float((conf < 0.7).mean())

    out = {
        "n_sites": int(len(sdf)),
        "label_counts": label_counts,
        "z_geom": z_geom,
        "conf_hist": conf_hist,
        "frac_conf_lt_07": frac_lt_07,
    }
    return _write_json(out, Path(figdata_dir or figdata_root(root)) / "level4"
                       / _figs_json_name(scale))


# ---------------------------------------------------------------------------
# level5 — tokenizer / metrics
# ---------------------------------------------------------------------------

def export_level5(token_counts_json: str | Path | None = None,
                  metrics_json: str | Path | None = None,
                  tokens_parquet: str | Path | None = None,
                  figdata_dir: str | Path | None = None,
                  kt: str | Path | None = None,
                  scale: str = "2k") -> Path:
    root = resolve_kt_root(kt)
    _paths = _scale_paths("level5", scale)
    counts_path = Path(token_counts_json or root / _paths[0])
    metrics_path = Path(metrics_json or root / _paths[1])
    toks_path = Path(tokens_parquet or root / _paths[2])
    if not counts_path.is_file():
        raise FileNotFoundError(f"token_counts 不存在: {counts_path}")
    counts = _read_json(counts_path)
    summary = _read_json(metrics_path) if metrics_path.is_file() else {}
    df = _require(toks_path, "tokens parquet")

    top_l3 = [[str(t), int(c)] for t, c in list(counts.get("l3", {}).items())[:20]]

    levels = {}
    for lev in ("l1", "l2", "l3", "l4"):
        block = (summary.get("levels") or {}).get(lev, {})
        levels[lev] = {
            "richness": block.get("richness_n_min_1"),
            "D_eff": block.get("D_eff"),
        }

    from crysh.metrics import accumulation  # 只读
    grid = _accum_grid_for(scale, int(len(df)))
    acc = {}
    for lev in ("l1", "l3", "l4"):
        if f"{lev}_tokens" in df.columns:
            lists = df[f"{lev}_tokens"].tolist()
            acc[lev] = [int(x) for x in accumulation(lists, grid)]
        else:
            acc[lev] = None

    out = {
        "N": int(len(df)),
        "scale": scale,
        "top_l3": top_l3,
        "levels": levels,
        "accumulation": acc,
        "grid": grid,
    }
    return _write_json(out, Path(figdata_dir or figdata_root(root)) / "level5"
                       / _figs_json_name(scale))


# ---------------------------------------------------------------------------
# oracle — alignment 聚合
# ---------------------------------------------------------------------------

def export_oracle(fast_parquet: str | Path | None = None,
                  oracle_parquet: str | Path | None = None,
                  chemenv_parquet: str | Path | None = None,
                  lambda_scan_json: str | Path | None = None,
                  figdata_dir: str | Path | None = None,
                  kt: str | Path | None = None,
                  scale: str = "2k") -> Path:
    root = resolve_kt_root(kt)
    _paths = _scale_paths("oracle", scale)
    fast_path = Path(fast_parquet or root / _paths[0])
    oracle_path = Path(oracle_parquet or root / _paths[1])
    if not fast_path.is_file():
        raise FileNotFoundError(
            f"fast parquet 不存在: {fast_path}（oracle 对齐图需 fast d_star/mean_cn）")
    if not oracle_path.is_file():
        raise FileNotFoundError(f"oracle parquet 不存在: {oracle_path}")

    fast = pd.read_parquet(fast_path)
    oracle = pd.read_parquet(oracle_path)
    merged = fast.merge(oracle, on="structure_id", how="inner")

    # --- dim 混淆矩阵（与 crysh.oracle.alignment_report 的 dim 块同式计算）---
    dim_block: dict[str, Any] = None
    if "d_star" in merged.columns and "larsen_dim" in merged.columns:
        dim = merged[["d_star", "larsen_dim"]].copy()
        if "ambiguous_dim_flag" in merged.columns:
            dim["ambiguous"] = merged["ambiguous_dim_flag"].fillna(False).astype(bool)
        else:
            dim["ambiguous"] = False
        dim = dim.dropna(subset=["d_star", "larsen_dim"])
        dim["d_fast"] = dim["d_star"].round().astype(int)
        dim["d_true"] = dim["larsen_dim"].round().astype(int)
        dim = dim[dim["d_fast"].isin((0, 1, 2, 3)) & dim["d_true"].isin((0, 1, 2, 3))]
        ambiguous = dim[dim["ambiguous"]]
        main = dim[~dim["ambiguous"]]
        cm = np.zeros((4, 4), dtype=int)
        for d_fast, d_true in zip(main["d_fast"], main["d_true"]):
            cm[int(d_fast), int(d_true)] += 1
        per_class_f1 = {}
        for c in range(4):
            tp = int(cm[c, c])
            fp = int(cm[c, :].sum()) - tp
            fn = int(cm[:, c].sum()) - tp
            denom = 2 * tp + fp + fn
            per_class_f1[DIM_LABELS[c]] = float(2 * tp / denom) if denom > 0 else 0.0
        dim_block = {
            "labels": DIM_LABELS,
            "confusion_4x4": cm.tolist(),
            "per_class_f1": per_class_f1,
            "macro_f1": float(np.mean(list(per_class_f1.values()))),
            "n_main": int(len(main)),
            "n_ambiguous_excluded": int(len(ambiguous)),
        }

    # --- CN 散点（per-structure mean_cn vs CrystalNN）---
    cn_scatter = None
    if "mean_cn" in merged.columns and "cnn_cn_mean" in merged.columns:
        cn = merged[["mean_cn", "cnn_cn_mean"]].dropna()
        cn_scatter = _round_rows(
            [[float(a), float(b)] for a, b in zip(cn["cnn_cn_mean"], cn["mean_cn"])])

    # --- ChemEnv CSM 值 + symbol top-20（前 500 pilot）---
    csm_block: dict[str, Any] = None
    chemenv_path = Path(chemenv_parquet or root / _paths[2])
    if chemenv_path.is_file():
        cdf = pd.read_parquet(chemenv_path)
        csms: list[float] = []
        counts: Counter = Counter()
        n_total = n_failed = 0
        for symbols, csm_list in zip(cdf["chemenv_symbols"], cdf["chemenv_csm"]):
            if symbols is None or csm_list is None:
                continue
            for sym, csm in zip(symbols, csm_list):
                n_total += 1
                if sym == "failed":
                    n_failed += 1
                    continue
                counts[str(sym)] += 1
                v = float(csm)
                if np.isfinite(v):
                    csms.append(round(v, 4))
        csm_block = {
            "values": csms,
            "n_sites_total": int(n_total),
            "n_sites_ok": int(n_total - n_failed),
            "n_sites_failed": int(n_failed),
            "top_symbols": [[s, int(n)] for s, n in counts.most_common(20)],
        }

    # --- CN λ 扫描曲线 ---
    lam_block: dict[str, Any] = None
    lam_path = Path(lambda_scan_json or root / _paths[3])
    if lam_path.is_file():
        scan = _read_json(lam_path)
        grid = [float(l) for l in scan.get("lambda_grid", [])]
        per = scan.get("per_lambda", {})
        lam_block = {
            "grid": grid,
            "exact_match": [per.get(str(l), {}).get("per_atom_exact_match") for l in grid],
            "mean_cn_mae": [per.get(str(l), {}).get("mean_cn_mae") for l in grid],
        }

    out = {
        "N_merged": int(len(merged)),
        "dim": dim_block,
        "cn_scatter": cn_scatter,
        "chemenv": csm_block,
        "lambda_curve": lam_block,
    }
    return _write_json(out, Path(figdata_dir or figdata_root(root)) / "oracle"
                       / _figs_json_name(scale))


# ---------------------------------------------------------------------------
# data-subset — registry 聚合
# ---------------------------------------------------------------------------

def export_registry(registry_2k_parquet: str | Path | None = None,
                    registry_20k_parquet: str | Path | None = None,
                    figdata_dir: str | Path | None = None,
                    kt: str | Path | None = None,
                    scale: str = "2k") -> Path:
    root = resolve_kt_root(kt)
    _paths = _scale_paths("data-subset", scale)
    reg2k = _require(registry_2k_parquet or root / _paths[0],
                     "registry 2k parquet")
    reg20k = _require(registry_20k_parquet or root / _paths[1],
                      f"registry {scale} parquet")
    ok2 = reg2k[reg2k["parse_ok"]]
    ok20 = reg20k[reg20k["parse_ok"]]

    def _presence(ok: pd.DataFrame) -> dict[str, float]:
        return (ok["elements"].explode().value_counts() / len(ok)).to_dict()

    # fig1: top-30 元素出现次数（2k）
    presence2 = ok2["elements"].explode().value_counts().head(30)
    fig1 = {"top": [[str(e), int(n)] for e, n in presence2.items()],
            "n_ok": int(len(ok2))}

    # fig2: n_atoms 直方图（公共 bins，密度归一）+ n_unique 分布
    n_max = int(max(ok2["n_atoms_actual"].max(), ok20["n_atoms_actual"].max()))
    bins = np.arange(0, n_max + 3, 3)
    h2, _ = np.histogram(ok2["n_atoms_actual"], bins=bins)
    h20, _ = np.histogram(ok20["n_atoms_actual"], bins=bins)
    u2 = ok2["elements"].apply(len).value_counts()
    u20 = ok20["elements"].apply(len).value_counts()
    idx = sorted(set(u2.index) | set(u20.index))
    fig2 = {
        "natom_hist": {"edges": [float(e) for e in bins],
                       "counts_2k": [int(x) for x in h2],
                       "counts_20k": [int(x) for x in h20],
                       "n_ok_2k": int(len(ok2)), "n_ok_20k": int(len(ok20))},
        "n_unique": {"values": [int(i) for i in idx],
                     "counts_2k": [int(u2.get(i, 0)) for i in idx],
                     "counts_20k": [int(u20.get(i, 0)) for i in idx]},
    }

    # fig3: 解析成功率 + volume vs n_atoms 散点（2k）
    fig3 = {
        "n_ok": int(reg2k["parse_ok"].sum()),
        "n_bad": int((~reg2k["parse_ok"]).sum()),
        "scatter": _round_rows(
            [[float(a), float(v)] for a, v in zip(ok2["n_atoms_actual"], ok2["volume"])],
            nd=2),
    }

    # fig4: 元素覆盖对比 top-30（按 20k 出现率）
    p2, p20 = _presence(ok2), _presence(ok20)
    top = sorted(p20.items(), key=lambda kv: -kv[1])[:30]
    fig4 = {"top": [[str(e), round(float(p2.get(e, 0.0)), 6), round(float(f), 6)]
                    for e, f in top]}

    out = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}
    return _write_json(out, Path(figdata_dir or figdata_root(root)) / "data-subset"
                       / _figs_json_name(scale))


# ---------------------------------------------------------------------------
# atlas — records_lite + 侧聚合
# ---------------------------------------------------------------------------

def _top_k_counts(d: Any, k: int = 3) -> dict[str, int]:
    """dict 列只留 top-k（按计数降序）；容忍 None 值（parquet map 往返可能产生）。"""
    if not isinstance(d, dict):
        return {}
    clean = {str(kk): int(vv) for kk, vv in d.items() if vv is not None}
    items = sorted(clean.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return dict(items)


def export_records_lite(records_parquet: str | Path | None = None,
                        figdata_dir: str | Path | None = None,
                        kt: str | Path | None = None,
                        scale: str = "2k") -> Path:
    root = resolve_kt_root(kt)
    src = Path(records_parquet or root / _scale_paths("atlas", scale)[0])
    df = _require(src, "records parquet")
    missing = [c for c in RECORDS_LITE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"records parquet 缺列: {missing}")
    lite = df[RECORDS_LITE_COLUMNS].copy()
    # dict 列只留 top-3 并序列化为 JSON 字符串：避免 parquet map 类型往返把
    # 全列 key 集合并入每行（缺失 key 补 None）的膨胀/歧义
    lite["geom_label_counts"] = lite["geom_label_counts"].map(
        lambda d: json.dumps(_top_k_counts(d), ensure_ascii=False, sort_keys=True))
    for c in ("q_min", "volume_norm", "dim_persistence", "mean_cn",
              "low_cn_fraction", "high_cn_fraction", "geom_ambiguous_fraction"):
        lite[c] = lite[c].astype(float)
        # +inf / NaN 保留原值（parquet 可存），画图侧再过滤
    _lite_name = "records_lite.parquet" if scale == "2k" else f"records_lite_{scale}.parquet"
    out = Path(figdata_dir or figdata_root(root)) / "atlas" / _lite_name
    out.parent.mkdir(parents=True, exist_ok=True)
    lite.to_parquet(out, index=False)
    return out


def export_atlas_side(records_parquet: str | Path | None = None,
                      figdata_dir: str | Path | None = None,
                      kt: str | Path | None = None,
                      scale: str = "2k") -> Path:
    """atlas fig3/4/5 + fig7 侧栏聚合 → figdata/atlas/side[_{scale}].json。"""
    root = resolve_kt_root(kt)
    src = Path(records_parquet or root / _scale_paths("atlas", scale)[0])
    stem = src.stem
    side_dir = src.parent

    cn_rows: list[list] = []
    cn_path = side_dir / f"{stem}_cn_counts.parquet"
    if cn_path.is_file():
        cn = pd.read_parquet(cn_path)
        cn_rows = [[str(e), int(c), int(n)] for e, c, n in
                   zip(cn["element"], cn["cn"], cn["count"])]
    geom_rows: list[list] = []
    geom_path = side_dir / f"{stem}_geom_counts.parquet"
    if geom_path.is_file():
        gm = pd.read_parquet(geom_path)
        geom_rows = [[str(e), str(g), int(n)] for e, g, n in
                     zip(gm["element"], gm["geometry"], gm["count"])]
    tok_counts: list[list] = []
    rare: list[list] = []
    n_rare_structs = 0
    tok_path = side_dir / f"{stem}_l3_tokens.parquet"
    if tok_path.is_file():
        tok = pd.read_parquet(tok_path)
        vc = tok["token"].value_counts()
        tok_counts = [[str(t), int(c)] for t, c in vc.items()]
        r = vc[vc <= 2]
        rare = [[str(t), int(c)] for t, c in r.sort_values(ascending=False).items()]
        if len(r):
            rare_set = set(r.index)
            n_rare_structs = int(
                tok.loc[tok["token"].isin(rare_set), "structure_id"].nunique()
            ) if "structure_id" in tok.columns else 0

    out = {
        "cn_counts": cn_rows,
        "geom_counts": geom_rows,
        "l3_token_counts": tok_counts,
        "rare_tokens": rare,
        "n_rare_structures": n_rare_structs,
    }
    return _write_json(out, Path(figdata_dir or figdata_root(root)) / "atlas"
                       / _figs_json_name(scale).replace("figs", "side"))


def export_atlas(records_parquet: str | Path | None = None,
                 figdata_dir: str | Path | None = None,
                 kt: str | Path | None = None,
                 scale: str = "2k") -> dict[str, Path]:
    """records_lite + 侧聚合（atlas 画图全部输入）。"""
    lite = export_records_lite(records_parquet, figdata_dir, kt, scale)
    side = export_atlas_side(records_parquet, figdata_dir, kt, scale)
    return {"records_lite": lite, "side": side}


# ---------------------------------------------------------------------------
# export-all + CLI
# ---------------------------------------------------------------------------

def export_all(kt: str | Path | None = None,
               figdata_dir: str | Path | None = None,
               scale: str = "2k") -> dict[str, Path]:
    root = resolve_kt_root(kt)
    fdd = Path(figdata_dir) if figdata_dir is not None else figdata_root(root)
    out: dict[str, Path] = {}
    out["level0"] = export_level0(figdata_dir=fdd, kt=root, scale=scale)
    out["level1"] = export_level1(figdata_dir=fdd, kt=root, scale=scale)
    out["level2"] = export_level2(figdata_dir=fdd, kt=root, scale=scale)
    out["level3"] = export_level3(figdata_dir=fdd, kt=root, scale=scale)
    out["level4"] = export_level4(figdata_dir=fdd, kt=root, scale=scale)
    out["level5"] = export_level5(figdata_dir=fdd, kt=root, scale=scale)
    try:
        out["data-subset"] = export_registry(figdata_dir=fdd, kt=root, scale=scale)
    except FileNotFoundError as exc:
        print(f"[figdata] 跳过 data-subset: {exc}", file=sys.stderr)
    try:
        out["oracle"] = export_oracle(figdata_dir=fdd, kt=root, scale=scale)
    except FileNotFoundError as exc:
        print(f"[figdata] 跳过 oracle: {exc}", file=sys.stderr)
    try:
        out["atlas_lite"] = export_records_lite(figdata_dir=fdd, kt=root, scale=scale)
        out["atlas_side"] = export_atlas_side(figdata_dir=fdd, kt=root, scale=scale)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[figdata] 跳过 atlas: {exc}", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kt", default=None, help="KT 根目录（默认环境变量 CKT_ROOT 或推导）")
    ap.add_argument("--scale", default="2k",
                    help="数据规模后缀（2k=默认旧路径；20k → figs_20k.json 等带后缀输出）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export-all", help="导出全部 figdata")
    for name in ("level0", "level1", "level2", "level3", "level4", "level5",
                 "oracle", "data-subset", "atlas"):
        sub.add_parser(f"export-{name}", help=f"只导出 {name}")
    args = ap.parse_args(argv)

    root = resolve_kt_root(args.kt)
    fdd = figdata_root(root)
    sc = args.scale
    cmds = {"export-all": lambda: export_all(kt=root, scale=sc),
            "export-level0": lambda: export_level0(figdata_dir=fdd, kt=root, scale=sc),
            "export-level1": lambda: export_level1(figdata_dir=fdd, kt=root, scale=sc),
            "export-level2": lambda: export_level2(figdata_dir=fdd, kt=root, scale=sc),
            "export-level3": lambda: export_level3(figdata_dir=fdd, kt=root, scale=sc),
            "export-level4": lambda: export_level4(figdata_dir=fdd, kt=root, scale=sc),
            "export-level5": lambda: export_level5(figdata_dir=fdd, kt=root, scale=sc),
            "export-oracle": lambda: export_oracle(figdata_dir=fdd, kt=root, scale=sc),
            "export-data-subset": lambda: export_registry(figdata_dir=fdd, kt=root, scale=sc),
            "export-atlas": lambda: export_atlas(figdata_dir=fdd, kt=root, scale=sc)}
    results = cmds[args.cmd]()
    paths = list(results.values()) if isinstance(results, dict) else [results]
    total = 0
    for p in paths:
        if isinstance(p, dict):
            for q in p.values():
                total += Path(q).stat().st_size
        else:
            total += Path(p).stat().st_size
        print(f"wrote {p}")
    print(f"figdata total: {total:,} bytes under {fdd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
