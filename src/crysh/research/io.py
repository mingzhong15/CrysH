"""CKT 数据基础设施 io —— POSCAR 子集解析、manifest 读取、registry 构建。

所有权：data-subset 子任务（contracts.md §2）；接口签名冻结于 contracts.md §3.1。

设计要点（含契约疑义的最保守解释，见 data-subset/progress.md）
------------------------------------------------------------------
* ``structure_id`` = POSCAR 文件名去 ``.vasp`` 后缀（全局唯一）。
* registry 列（frozen，§3.1）：``structure_id, source_line, formula, elements,
  n_atoms_manifest, n_atoms_actual, volume, parse_ok, parse_error`` —— 本模块不增不减。
* manifest 的 ``n_atoms`` 是逐元素计数的 list；registry 的 ``n_atoms_manifest`` 取其**总和**
  （标量），``n_atoms_actual`` = 解析所得原子总数。
* manifest 的 ``formula`` 是**约化式（reduced）**（对 Z>1 晶胞，其系数和为
  n_atoms_manifest 的真子集倍数）：formula 一致性校验按 **gcd 约化后的元素计数 dict**
  比较（解析侧 = Counter(symbols) 约化；manifest 侧 = 正则解析 formula 字符串），
  不做字符串/顺序比较。
* registry 的 ``formula`` 列 = 解析结构导出的规范约化式：元素按原子序数升序、计数
  显式（含 1）、空格分隔，如 ``"Li1 O2 Hf1"``。
* registry 的 ``elements`` 列 = 解析结构的去重元素符号 list（原子序数升序）；
  解析失败行为空 list ``[]``（保持 parquet list 类型稳定，失败由 parse_ok 表达）。
* 解析失败行：``parse_ok=False``、``parse_error="<ExceptionName>: <msg>"``，
  其余解析列 NaN/None；manifest 侧字段（source_line / n_atoms_manifest）仍填充。
* ``build_registry`` 以 **manifest 为权威清单** 生成任务（structure_id 取 manifest
  path 的文件名 stem）；文件缺失记 ``FileNotFoundError`` 失败行而不中断；
  subset_dir 中不在 manifest 的文件只记 warning 忽略。
* 校验（n_atoms 一致性、formula 一致性）只报告、不改数据：不一致行写
  ``registry_checks.csv``；解析失败行写 ``parse_failures.csv``；统计摘要写
  ``registry_summary.json``。三个落盘路径默认为
  ``<KT>/data-subset/data/``（由本模块位置推导，不硬编码），均可选参数覆盖。
* 多进程并行度硬上限 8（contracts.md §0/任务书）；默认 n_proc=8，
  实际 workers = min(n_proc, 8, 任务数)。
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import re
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read as _ase_read

logger = logging.getLogger("ckt.io")

# KT 工作目录：统一由 crysh.paths 解析（CKT_ROOT 环境变量 > contracts.md 上溯 > 旧深度兜底）
# 路径常量：既服务本模块默认值，也作为历史入口再导出（脚本/测试按 ckt.io.KT_ROOT 引用过）
from .paths import KT_ROOT, LEVELS_DIR  # noqa: F401

# 契约/任务书：多进程并行度 ≤ 8
MAX_N_PROC = 8

MANIFEST_COLUMNS = ["path", "elements", "n_atoms", "n_unique_elements", "formula", "source_line"]

# registry 列（frozen，contracts.md §3.1）——顺序即 parquet 列顺序
REGISTRY_COLUMNS = [
    "structure_id", "source_line", "formula", "elements",
    "n_atoms_manifest", "n_atoms_actual", "volume",
    "parse_ok", "parse_error",
]

# manifest formula 形如 "Li1 O2 Hf1" / "Y6 Ru Te2"（计数 1 可能省略）
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def iter_poscars(folder: Path) -> Iterator[tuple[str, Atoms]]:
    """遍历 folder 下 ``*.vasp``，yield ``(structure_id, atoms)``。

    解析异常优雅跳过：记 warning（logger ``ckt.io``）后继续，不让整个遍历崩溃。
    structure_id = 文件名去 ``.vasp`` 后缀；文件按名称排序保证确定性。
    """
    folder = Path(folder)
    files = sorted(p for p in folder.glob("*.vasp") if p.is_file())
    n_skipped = 0
    for path in files:
        structure_id = path.name[: -len(".vasp")]
        try:
            atoms = _ase_read(path, format="vasp")
        except Exception as exc:  # 任何解析异常都跳过并记录
            n_skipped += 1
            logger.warning(
                "iter_poscars: skip %s (%s: %s)", path.name, type(exc).__name__, exc
            )
            continue
        yield structure_id, atoms
    if n_skipped:
        logger.warning("iter_poscars: %d/%d 文件解析失败被跳过", n_skipped, len(files))


def load_manifest(jsonl_path: Path) -> pd.DataFrame:
    """逐行读取 manifest jsonl → DataFrame（原始列，不做重命名）。

    - 路径不存在 → ``FileNotFoundError``（输入缺失应显式失败）。
    - 空行跳过；坏 JSON 行 / 非 dict 行记 warning 跳过。
    - 空文件 → 空 DataFrame（带 MANIFEST_COLUMNS 列）。
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"manifest not found: {jsonl_path}")
    rows: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "load_manifest: skip bad line %s:%d (%s)", jsonl_path, line_no, exc
                )
                continue
            if not isinstance(rec, dict):
                logger.warning(
                    "load_manifest: skip non-dict line %s:%d", jsonl_path, line_no
                )
                continue
            rows.append(rec)
    if rows:
        df = pd.DataFrame(rows)
        for col in MANIFEST_COLUMNS:  # 个别行缺 key 时补空列，保证下游可用
            if col not in df.columns:
                df[col] = None
    else:
        df = pd.DataFrame(columns=MANIFEST_COLUMNS)
    return df


# --------------------------------------------------------------------------
# registry 构建
# --------------------------------------------------------------------------

def _composition_of_formula(formula: str) -> dict[str, int]:
    """把 manifest 式 formula 字符串解析为 {元素符号: 计数}（计数省略视为 1）。"""
    comp: dict[str, int] = {}
    for m in _FORMULA_TOKEN_RE.finditer(formula):
        sym, cnt = m.group(1), m.group(2)
        comp[sym] = comp.get(sym, 0) + (int(cnt) if cnt else 1)
    return comp


def _reduced_composition(counts: dict[str, int]) -> dict[str, int]:
    """元素计数 dict 除以 gcd → 约化组成（与 manifest formula 语义对齐）。"""
    g = 0
    for c in counts.values():
        g = math.gcd(g, int(c))
    if g <= 1:
        return dict(counts)
    return {k: int(v) // g for k, v in counts.items()}


def _canonical_formula(counts: dict[str, int]) -> str:
    """规范约化式：元素按原子序数升序、计数显式、空格分隔，如 "Li1 O2 Hf1"。"""
    syms = sorted(counts, key=lambda s: atomic_numbers.get(s, 0))
    return " ".join(f"{s}{counts[s]}" for s in syms)


def _parse_one(job: tuple[str, Path]) -> dict:
    """进程池 worker（模块级函数以可 pickle）：解析单个 POSCAR → 单行 dict。

    ``_symbol_counts`` 为内部校验列（约化组成比较用），写盘前丢弃。
    """
    structure_id, path = job
    row: dict = {
        "structure_id": structure_id,
        "formula": None,
        "elements": [],
        "n_atoms_actual": None,
        "volume": None,
        "parse_ok": False,
        "parse_error": None,
        "_symbol_counts": {},
    }
    try:
        atoms = _ase_read(path, format="vasp")
    except Exception as exc:
        row["parse_error"] = f"{type(exc).__name__}: {exc}"
        return row
    try:
        symbols = list(atoms.get_chemical_symbols())
        counts = Counter(symbols)
        row["_symbol_counts"] = dict(counts)
        row["n_atoms_actual"] = len(symbols)
        row["volume"] = float(atoms.get_volume())
        row["formula"] = _canonical_formula(_reduced_composition(counts))
        row["elements"] = sorted(counts, key=lambda s: atomic_numbers.get(s, 0))
        row["parse_ok"] = True
    except Exception as exc:
        row["parse_error"] = f"{type(exc).__name__}: {exc}"
    return row


def _default_side_paths(out_stem: str) -> tuple[Path, Path, Path]:
    """默认落盘位置：<KT>/data-subset/data/（由模块位置推导，可移植）。

    侧文件名带 registry stem 以防不同子集互相覆盖；仅 subset_2k 保留任务书固定名
    （parse_failures.csv / registry_checks.csv / registry_summary.json）。
    """
    # 2026-09-17：data-subset 搬到 code/levels/data-subset/
    base = LEVELS_DIR / "data-subset" / "data"
    if out_stem == "subset_2k":  # 任务书指定的固定文件名（2k 历史产物名，保持兼容）
        return base / "parse_failures.csv", base / "registry_checks.csv", base / "registry_summary.json"
    return (
        base / f"parse_failures_{out_stem}.csv",
        base / f"registry_checks_{out_stem}.csv",
        base / f"registry_summary_{out_stem}.json",
    )


def build_registry(
    subset_dir: Path,
    manifest_path: Path,
    out_parquet: Path,
    n_proc: int = MAX_N_PROC,
    failures_csv: Path | None = None,
    checks_csv: Path | None = None,
    summary_json: Path | None = None,
) -> pd.DataFrame:
    """对 manifest 全量结构解析 → registry DataFrame，并写 parquet + 侧文件。

    参数
    ----
    subset_dir : 含 ``*.vasp`` 的目录（文件名为 structure_id + ".vasp"）。
    manifest_path : manifest jsonl。
    out_parquet : registry parquet 输出路径。
    n_proc : 进程数（硬上限 8；默认 8；任务数少时自动缩减）。
    failures_csv / checks_csv / summary_json : 侧文件路径覆盖（默认 <KT>/data-subset/data/…）。

    返回
    ----
    pd.DataFrame：与 parquet 内容一致（列顺序 = REGISTRY_COLUMNS）。

    行为
    ----
    - manifest 为权威清单：逐行生成 (structure_id, 文件) 任务；文件缺失记
      ``FileNotFoundError`` 失败行，不中断；subset_dir 中不在 manifest 的
      ``*.vasp`` 记 warning 忽略。
    - 校验（只报告不改数据）：n_atoms_manifest（=manifest n_atoms list 总和）vs
      n_atoms_actual；formula（gcd 约化组成）vs manifest formula 组成。
      不一致行写 checks_csv；解析失败行写 failures_csv（无失败时仅表头）；
      统计摘要写 summary_json。
    """
    t0 = time.time()
    subset_dir = Path(subset_dir)
    manifest = load_manifest(manifest_path)
    if failures_csv is None or checks_csv is None or summary_json is None:
        f0, c0, s0 = _default_side_paths(Path(out_parquet).stem)
        failures_csv = Path(failures_csv) if failures_csv is not None else f0
        checks_csv = Path(checks_csv) if checks_csv is not None else c0
        summary_json = Path(summary_json) if summary_json is not None else s0

    # ---- 任务构建（manifest 权威） ----
    jobs: list[tuple[str, Path]] = []
    man_rows: list[dict] = []
    seen: set[str] = set()
    for rec in manifest.to_dict("records"):
        path_str = str(rec.get("path") or "")
        if not path_str:
            logger.warning("build_registry: manifest 行缺 path，跳过")
            continue
        sid = Path(path_str).stem
        if sid in seen:
            logger.warning("build_registry: manifest path 重复，跳过 %s", path_str)
            continue
        seen.add(sid)
        jobs.append((sid, subset_dir / Path(path_str).name))
        try:
            natoms_manifest = int(sum(rec.get("n_atoms") or []))
        except (TypeError, ValueError) as exc:
            logger.warning("build_registry: %s 的 n_atoms 非法（%s），记 None", sid, exc)
            natoms_manifest = None
        man_rows.append({
            "structure_id": sid,
            "source_line": rec.get("source_line"),
            "formula_manifest": rec.get("formula"),
            "n_atoms_manifest": natoms_manifest,
        })

    folder_sids = {p.stem for p in subset_dir.glob("*.vasp") if p.is_file()}
    extra = folder_sids - seen
    if extra:
        logger.warning(
            "build_registry: %d 个文件不在 manifest 中，忽略（例：%s）",
            len(extra), sorted(extra)[0],
        )

    if not jobs:
        logger.warning("build_registry: 无解析任务（manifest 为空）→ 输出空 registry")
        out = pd.DataFrame(columns=REGISTRY_COLUMNS)
        Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(Path(out_parquet), index=False)
        Path(failures_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["structure_id", "source_line", "parse_error"]).to_csv(failures_csv, index=False)
        pd.DataFrame(columns=["structure_id", "check", "manifest_value", "actual_value"]).to_csv(checks_csv, index=False)
        Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_json).write_text(json.dumps({
            "n_manifest": 0, "n_total_rows": 0, "n_parsed_ok": 0, "n_parse_failed": 0,
            "n_missing_files": 0, "n_natom_mismatch": 0, "n_formula_mismatch": 0,
            "n_workers": 0, "wall_seconds": round(time.time() - t0, 3),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    # ---- 并行解析（≤8 进程） ----
    n_workers = max(1, min(int(n_proc), MAX_N_PROC, len(jobs)))
    if n_workers > 1:
        try:
            ctx = mp.get_context("fork")
        except ValueError:  # 平台无 fork（Windows）→ 平台默认（spawn）
            ctx = mp.get_context()
        with ctx.Pool(n_workers) as pool:
            parsed = pool.map(
                _parse_one, jobs, chunksize=max(1, len(jobs) // (n_workers * 8))
            )
    else:
        parsed = [_parse_one(job) for job in jobs]

    df = pd.DataFrame(parsed)
    man_df = pd.DataFrame(man_rows)
    df = df.merge(man_df, on="structure_id", how="left", validate="one_to_one")

    # ---- 校验（只报告） ----
    ok_mask = df["parse_ok"].fillna(False).astype(bool)
    natom_ok = ok_mask & (df["n_atoms_manifest"].astype("Int64") == df["n_atoms_actual"].astype("Int64"))
    df["_manifest_comp"] = [
        _composition_of_formula(f) if isinstance(f, str) else None
        for f in df["formula_manifest"]
    ]
    formula_ok = pd.Series([
        bool(ok) and (_reduced_composition(sc or {}) == mc)
        for ok, sc, mc in zip(ok_mask, df["_symbol_counts"], df["_manifest_comp"])
    ], index=df.index)
    natom_bad = df[ok_mask & ~natom_ok]
    formula_bad = df[ok_mask & ~formula_ok]

    # ---- 落盘 ----
    out = df[REGISTRY_COLUMNS].copy()
    out["n_atoms_manifest"] = pd.to_numeric(out["n_atoms_manifest"], errors="coerce").astype("Int64")
    out["n_atoms_actual"] = pd.to_numeric(out["n_atoms_actual"], errors="coerce").astype("Int64")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["parse_ok"] = out["parse_ok"].fillna(False).astype(bool)

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(out_parquet), index=False)

    Path(failures_csv).parent.mkdir(parents=True, exist_ok=True)
    fails = out[~out["parse_ok"]]
    pd.DataFrame({
        "structure_id": fails["structure_id"],
        "source_line": fails["source_line"],
        "parse_error": fails["parse_error"],
    }).to_csv(failures_csv, index=False)

    checks = []
    for r in natom_bad.to_dict("records"):
        checks.append({
            "structure_id": r["structure_id"], "check": "n_atoms",
            "manifest_value": r["n_atoms_manifest"], "actual_value": r["n_atoms_actual"],
        })
    for r in formula_bad.to_dict("records"):
        checks.append({
            "structure_id": r["structure_id"], "check": "formula",
            "manifest_value": r["formula_manifest"], "actual_value": r["formula"],
        })
    pd.DataFrame(checks, columns=["structure_id", "check", "manifest_value", "actual_value"]).to_csv(
        checks_csv, index=False
    )

    n_missing = int(
        fails["parse_error"].astype(str).str.startswith("FileNotFoundError", na=False).sum()
    )
    summary = {
        "subset_dir": str(subset_dir),
        "manifest_path": str(manifest_path),
        "out_parquet": str(out_parquet),
        "n_manifest": len(man_df),
        "n_total_rows": len(out),
        "n_parsed_ok": int(out["parse_ok"].sum()),
        "n_parse_failed": int((~out["parse_ok"]).sum()),
        "n_missing_files": n_missing,
        "n_natom_mismatch": int(len(natom_bad)),
        "n_formula_mismatch": int(len(formula_bad)),
        "n_workers": n_workers,
        "wall_seconds": round(time.time() - t0, 3),
    }
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "build_registry: %d/%d parsed OK, %d failed, %d natom mismatch, %d formula mismatch (%.1fs, %d workers)",
        summary["n_parsed_ok"], summary["n_total_rows"], summary["n_parse_failed"],
        summary["n_natom_mismatch"], summary["n_formula_mismatch"],
        summary["wall_seconds"], n_workers,
    )
    return out
