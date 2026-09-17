"""记录组装：把一个结构跑成一行 record（列序与 `config.RECORD_COLUMNS` 严格一致）。

设计取舍（R3 定案，2026-09-17）
------------------------------
重构前这个模块（原 `ckt/pipeline.py`，1005 行）同时承担四件事：真实现调度、mock
合成、逐级 parquet 回读、多进程批处理。后果是一条 record 可能来自**任意一个**来源，
而 8 个 `except ImportError` + 7 个 `_HAVE_*` 守卫让"算失败"与"模块没合入"无法区分——
任何异常都被静默吞掉再退回 mock，数据里出现 `mock_L3` 这种 flag 时没人会注意到。

现在只保留**一条路径**：真实现。失败就抛（由调用方决定跳过还是重跑），不再有 mock、
不再回读自己刚写的旧 parquet。开发用合成结构见 `crysh.dev`；逐级结果回读属于研究
工程的数据流，不在库里。

分层：
- `map_structure(atoms, cfg) -> (record, side)`  单结构（含聚合侧数据）
- `map_record(atoms, cfg) -> record`             单结构（只取 record，日常入口）
- `run_batch(manifest, out_parquet, cfg, ...)`   manifest(jsonl) → parquet + 侧文件
- `_write_outputs(...)`                          落盘（pandas 只在写出时需要）
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read as ase_read

from crysh.bond import canonical_bond_graph
from crysh.config import RECORD_COLUMNS, MapperConfig
from crysh.coord import coordination, load_cn_table
from crysh.dimensionality import dimensionality_spectrum
from crysh.geometry import classify_geometry, geometry_features
from crysh.morphology import morphology, morphology_with_diagnostics
from crysh.tokens import motif_tokens
from crysh.validity import apply_filters, default_thresholds, validity_metrics

__all__ = ["map_structure", "map_record", "run_structure", "map_sites",
           "run_batch", "SITE_SUMMARY_COLUMNS"]

_CN_TABLE_CACHE: dict[str, dict | None] = {}
_THRESHOLDS_CACHE: dict[str, dict | None] = {}


def _thresholds(cfg: MapperConfig) -> dict:
    """L0 校准阈值（q_c 等）；未配置路径 → 模块默认（契约 §5 pilot 默认 q_c=0.85）。"""
    if cfg.l0_thresholds_path is None:
        return default_thresholds()
    key = str(cfg.l0_thresholds_path)
    if key not in _THRESHOLDS_CACHE:
        try:
            _THRESHOLDS_CACHE[key] = json.loads(Path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"[crysh.records] L0 阈值不可用（{key}: {exc}）→ 用模块默认",
                  file=sys.stderr)
            _THRESHOLDS_CACHE[key] = None
    return _THRESHOLDS_CACHE[key] or default_thresholds()


def _cn_table(cfg: MapperConfig) -> dict | None:
    """L3 的 P(CN|Z) 分位表；无表时 percentile 记 NaN、low/high fraction 记为 0。"""
    if cfg.cn_table is not None:
        return cfg.cn_table
    if cfg.cn_table_path is None:
        return None
    key = str(cfg.cn_table_path)
    if key not in _CN_TABLE_CACHE:
        try:
            _CN_TABLE_CACHE[key] = load_cn_table(key)
        except (OSError, ValueError) as exc:
            print(f"[crysh.records] L3 cn_table 不可用（{key}: {exc}）"
                  "→ low/high_cn_fraction 记为 0", file=sys.stderr)
            _CN_TABLE_CACHE[key] = None
    return _CN_TABLE_CACHE[key]


def map_structure(atoms: Atoms, cfg: MapperConfig | None = None
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """链 L0→L5，返回 `(record, side)`。

    `record` 的键序 == `RECORD_COLUMNS`，可直接作为一行表数据。
    `side` 是 fig/聚合侧的明细（per-element CN/geometry 计数、去重 l3 token），
    不属冻结 schema，故不混进 record。

    Parameters
    ----------
    atoms:
        ASE `Atoms`；`info` 可带 `structure_id` / `source_line` / `formula` /
        `elements`（缺省从结构本身推导）。
    cfg:
        :class:`crysh.config.MapperConfig`；``None`` 等价默认配置。
    """
    cfg = cfg or MapperConfig()
    sid = str(atoms.info.get("structure_id", ""))
    syms = list(atoms.get_chemical_symbols())
    rec: dict[str, Any] = {}

    # ---- 元信息 ----
    if atoms.info.get("elements"):
        elements = [str(e) for e in atoms.info["elements"]]
    else:
        elements = sorted(set(syms), key=lambda s: atomic_numbers[s])
    rec["structure_id"] = sid
    rec["source_line"] = atoms.info.get("source_line")
    rec["formula"] = str(atoms.info.get("formula") or atoms.get_chemical_formula(mode="metal"))
    rec["elements"] = elements
    rec["natom"] = int(len(atoms))

    # ---- 公共键图（L1–L5 共用一份；v1.2 canonical：有校准表走表@λ=1.0，否则共价×λ*）----
    graph = canonical_bond_graph(atoms, cfg.cutoff_table)

    # ---- L0 validity ----
    thr = _thresholds(cfg)
    m = validity_metrics(atoms, q_c=float(thr.get("q_c", 0.85)))
    fl = apply_filters(m, thr)
    rec.update({
        "q_min": float(m.get("q_min", np.nan)),
        "n_overlap_pairs": int(m.get("n_overlap_pairs", 0)),
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

    # ---- L1 dimensionality ----
    spectrum = dimensionality_spectrum(atoms, cfg.cutoff_table, lambdas=cfg.lambdas)
    rec["d_star"] = int(spectrum.d_star)
    rec["dim_persistence"] = float(spectrum.persistence)
    rec["n_components"] = int(spectrum.n_components)
    rec["ambiguous_dim_flag"] = bool(spectrum.persistence < 0.5)
    for lam, col in cfg.lambda_cols().items():
        rec[col] = int(spectrum.d_by_lambda.get(lam, spectrum.d_star))
    rec["dim_components"] = json.dumps(
        {str(k): [int(x) for x in v]
         for k, v in getattr(spectrum, "dim_components", {}).items()},
        sort_keys=True)

    # ---- L2 morphology ----
    res = morphology(atoms, graph=graph, spectrum=spectrum, coord=None)
    rec.update({
        "vacuum_gap": float(res.vacuum_gap),
        "vacuum_fraction": float(res.vacuum_fraction),
        "surface_score": float(res.surface_score),
        "porous_candidate": bool(res.porous_candidate),
        "morphology_class": str(res.class_label),
    })
    diag = morphology_with_diagnostics(atoms, graph=graph, spectrum=spectrum, coord=None)
    for key in ("n_layers", "span", "f_max", "cn_std",
                "vacuum_score", "slab_score", "mono_score"):
        if key in diag:
            rec[key] = float(diag[key]) if isinstance(diag[key], (int, float)) else diag[key]

    # ---- L3 coordination ----
    coord_res = coordination(atoms, graph, _cn_table(cfg))
    cns = np.asarray(coord_res.cn, dtype=int)
    rec.update({
        "mean_cn": float(coord_res.mean_cn),
        "low_cn_fraction": float(coord_res.low_cn_fraction),
        "high_cn_fraction": float(coord_res.high_cn_fraction),
        "cn_min": int(coord_res.cn_min),
        "cn_max": int(coord_res.cn_max),
        "cn_cell_warning": bool(getattr(coord_res, "cn_cell_warning", False)),
        "cn_cell_margin": float(getattr(coord_res, "cn_cell_margin", float("inf"))),
    })

    # ---- L4 coordination geometry ----
    raw = classify_geometry(geometry_features(atoms, graph))
    geom_labels = ["ambiguous" if (conf < 0.7 or lab == "ambiguous") else lab for lab, conf in raw]
    counts = Counter(geom_labels) or Counter({"other": 0})
    rec.update({
        "geom_top_label": counts.most_common(1)[0][0],
        "geom_ambiguous_fraction": round(counts.get("ambiguous", 0) / len(geom_labels), 4)
        if geom_labels else 0.0,
        "geom_label_counts": dict(counts),
    })

    # ---- L5 tokens / sharing ----
    toks = motif_tokens(atoms, graph, coord_res, geom_labels)
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

    rec["quality_flags"] = []
    side = {
        "cn_pairs": Counter(zip(syms, [int(c) for c in cns])),
        "geom_pairs": Counter(zip(syms, geom_labels)),
        "l3_tokens": sorted(set(l3_tokens)),
    }
    ordered = {k: rec.get(k) for k in RECORD_COLUMNS}
    return ordered, side


def map_record(atoms: Atoms, cfg: MapperConfig | None = None) -> dict[str, Any]:
    """单结构 → 一行 record（不返回侧数据）。文档与日常用的就是这个入口。"""
    rec, _side = map_structure(atoms, cfg)
    return rec


#: 历史名（研究工程的脚本与测试仍按 run_structure 调用）；新代码用 map_record。
run_structure = map_record


def make_mock_records(*args, **kwargs):  # pragma: no cover - 兼容壳
    """历史名 → :func:`crysh.dev.make_synthetic_records`（见其 docstring 的语义说明）。"""
    from crysh.dev import make_synthetic_records

    return make_synthetic_records(*args, **kwargs)


#: site 级汇总列（**additive**：不并入冻结的 `RECORD_COLUMNS`，由调用方决定何时入库）
SITE_SUMMARY_COLUMNS = (
    "p_cn_mean", "shell_conf_mean", "shell_conf_low_frac", "cn_eff_mean",
    "cn_shell_agree_frac", "cn_species_top", "d_i_mean", "d_i_max",
)


def map_sites(atoms: Atoms, cfg: MapperConfig | None = None, *,
              graph=None, coord=None, geometry_labels: list[str] | None = None
              ) -> tuple[list, dict[str, Any]]:
    """L3 的 site 级结果（m 实例）+ 结构级汇总（**增量，不动冻结 schema**）。

    返回 `(sites, summary)`：

    - `sites`：:class:`crysh.localenv.SiteEnvironment` 列表（逐位点）；
    - `summary`：键见 `SITE_SUMMARY_COLUMNS`，可直接拼到 records 行的**新增列**上。

    为什么不直接塞进 record：`RECORD_COLUMNS` 是冻结契约（59 列，已写进集群上的
    全量 L0–L5 产物）。往契约里加列要连带改 schema 版本、reader 与下游 UI——那是
    计划里的 M4。这里先把数据算出来并以明确的键交付，由调用方决定何时并入。
    """
    cfg = cfg or MapperConfig()
    from crysh.localenv import local_environments

    envs = local_environments(atoms, cfg, graph=graph, coord=coord,
                              geometry_labels=geometry_labels)
    if not envs:
        return envs, {k: float("nan") for k in SITE_SUMMARY_COLUMNS}

    conf = np.array([e.shell_conf for e in envs], dtype=float)
    d_i = np.array([e.d_i for e in envs], dtype=float)
    agree = np.array([e.cn == e.cn_shell for e in envs], dtype=bool)
    species = Counter(e.cn_species for e in envs)
    summary = {
        "p_cn_mean": float(np.mean([e.p_cn for e in envs])),
        "shell_conf_mean": float(np.mean(conf)),
        # "低置信位点占比"：shell_conf < 0.3（几何清晰但 λ 持久度低的量级）
        "shell_conf_low_frac": float(np.mean(conf < 0.3)),
        "cn_eff_mean": float(np.mean([e.cn_eff for e in envs])),
        "cn_shell_agree_frac": float(np.mean(agree)),
        "cn_species_top": species.most_common(1)[0][0],
        "d_i_mean": float(np.mean(d_i)),
        "d_i_max": float(np.max(d_i)),
    }
    return envs, summary


def _write_outputs(records: list[dict[str, Any]], sides: list[dict[str, Any]],
                   out_parquet: Path) -> None:
    """records → parquet + 三个 fig 侧文件（cn/geom 计数、l3 token 长表）。"""
    import pandas as pd  # 只有写出路径需要（optional extra: tables）

    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=RECORD_COLUMNS).to_parquet(out_parquet, index=False)
    stem = out_parquet.stem
    cn_counts: Counter = Counter()
    geom_counts: Counter = Counter()
    token_rows: list[tuple[str, str]] = []
    for rec, side in zip(records, sides, strict=False):
        cn_counts.update(side["cn_pairs"])
        geom_counts.update(side["geom_pairs"])
        token_rows.extend((rec["structure_id"], t) for t in side["l3_tokens"])
    pd.DataFrame([(e, c, n) for (e, c), n in cn_counts.items()],
                 columns=["element", "cn", "count"]).to_parquet(
        out_parquet.parent / f"{stem}_cn_counts.parquet", index=False)
    pd.DataFrame([(e, g, n) for (e, g), n in geom_counts.items()],
                 columns=["element", "geometry", "count"]).to_parquet(
        out_parquet.parent / f"{stem}_geom_counts.parquet", index=False)
    pd.DataFrame(token_rows, columns=["structure_id", "token"]).to_parquet(
        out_parquet.parent / f"{stem}_l3_tokens.parquet", index=False)


def _batch_worker(task: tuple[dict[str, Any], str | None, MapperConfig]
                  ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    row, local_path, cfg = task
    try:
        sid = Path(row["path"]).stem
        atoms = ase_read(local_path if local_path else row["path"])
        atoms.info["structure_id"] = sid
        atoms.info["source_line"] = row.get("source_line")
        atoms.info["elements"] = [str(e) for e in (row.get("elements") or [])]
        atoms.info["formula"] = row.get("formula")
        return map_structure(atoms, cfg)
    except Exception as exc:  # 单结构失败不该中断整批；逐条打日志，由调用方处置
        print(f"[crysh.records] 跳过 {row.get('path')}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return None


def run_batch(manifest: Path | str, out_parquet: Path | str,
              cfg: MapperConfig | None = None, n_proc: int = 8,
              subset_dir: Path | str | None = None) -> dict[str, int]:
    """manifest(jsonl) → records parquet + 侧文件（多进程）。

    manifest 每行需含 `path`（原始结构路径）与可选 `source_line`/`elements`/`formula`。
    `subset_dir` 默认取 manifest 同级的同名目录；命中则读
    `<subset_dir>/<structure_id>.vasp`，否则读 `path` 本身。

    Returns
    -------
    dict
        `{"n_total", "n_ok", "n_failed"}`。
    """
    cfg = cfg or MapperConfig()
    manifest = Path(manifest)
    out_parquet = Path(out_parquet)
    sdir = Path(subset_dir) if subset_dir is not None else manifest.parent / manifest.stem
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    tasks = []
    for r in rows:
        sid = Path(r["path"]).stem
        local = sdir / f"{sid}.vasp"
        tasks.append((r, str(local) if local.exists() else None, cfg))
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
    print(f"[crysh.records] {len(records)}/{len(tasks)} 成功，{n_failed} 失败", file=sys.stderr)
    _write_outputs(records, sides, out_parquet)
    return {"n_total": len(tasks), "n_ok": len(records), "n_failed": n_failed}
