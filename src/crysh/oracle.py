"""CKT oracle layer: slow pymatgen ground truth + fast-vs-oracle alignment report.

Contract: contracts.md §3.9. This module owns the "oracle-bench" subtask.

- ``run_larsen`` / ``run_crystalnn_cn`` / ``run_chemenv``: per-structure slow oracles.
- ``batch_oracle``: multiprocess batch over a manifest jsonl or a folder of POSCARs.
- ``alignment_report``: fast-layer vs oracle metrics (dim confusion / macro-F1,
  CN MAE / exact-match, ChemEnv CSM summary) with json + PNG outputs.

Design notes
------------
* pymatgen / matplotlib are imported lazily inside functions: ``import crysh.oracle``
  stays cheap, and workers only pay the import cost once per process.
* A single bad structure can never kill a batch: every component is wrapped in
  try/except and recorded in the ``oracle_error`` column (per §3.9).
* ChemEnv uses ``LocalGeometryFinder`` + ``SimplestChemenvStrategy`` (pymatgen
  2026.5.4 API); per site the lowest-CSM environment is kept, failed sites are
  reported as ``("failed", nan)`` and a fully failed structure returns all
  ``("failed", nan)``.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from crysh.paths import KT_ROOT  # noqa: F401  (KT 工作目录，见 paths.py)

DIM_LABELS = ["0D", "1D", "2D", "3D"]
DIM_VALUES = (0, 1, 2, 3)

# Frozen oracle parquet columns (contracts.md §3.9)
ORACLE_COLUMNS = [
    "structure_id",
    "larsen_dim",
    "cnn_cn_mean",
    "cnn_cn",
    "chemenv_symbols",
    "chemenv_csm",
    "oracle_error",
]


# ---------------------------------------------------------------------------
# per-structure oracles
# ---------------------------------------------------------------------------

def _ase_to_structure(atoms):
    """ase.Atoms -> pymatgen Structure (contract: use AseAtomsAdaptor)."""
    from pymatgen.io.ase import AseAtomsAdaptor

    return AseAtomsAdaptor.get_structure(atoms)


def _crystalnn():
    from pymatgen.analysis.local_env import CrystalNN

    return CrystalNN()


def _quiet(fn, *args, **kwargs):
    """Run ``fn`` with all warnings suppressed (pymatgen is very chatty here)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def run_larsen(atoms) -> int:
    """Periodic dimensionality 0/1/2/3 via pymatgen CrystalNN + Larsen method.

    Contract §3.9: CrystalNN().get_bonded_structure() + get_dimensionality_larsen().
    """
    from pymatgen.analysis.dimensionality import get_dimensionality_larsen

    structure = _ase_to_structure(atoms)
    bonded = _quiet(_crystalnn().get_bonded_structure, structure)
    return int(_quiet(get_dimensionality_larsen, bonded))


def run_crystalnn_cn(atoms) -> np.ndarray:
    """(N,) int64 array: per-atom coordination number (CrystalNN neighbor count)."""
    structure = _ase_to_structure(atoms)
    nn_lists = _quiet(_crystalnn().get_all_nn_info, structure)
    return np.asarray([len(nn) for nn in nn_lists], dtype=np.int64)


def run_chemenv(atoms) -> list[tuple[str, float]]:
    """Per-site best ChemEnv environment symbol + CSM.

    Contract §3.9: simplified ChemEnv via LocalGeometryFinder +
    SimplestChemenvStrategy (pymatgen 2026.5.4 API: the strategy is applied with
    ``strategy.set_structure_environments(se)`` and per-site
    ``get_site_coordination_environments(site=..., isite=...)``). For each site the
    environment with the lowest continuous symmetry measure is kept. A site
    without any environment is ``("failed", nan)``; if the whole structure fails,
    every site is reported as ``("failed", nan)`` (never raises).
    """
    n_sites = len(atoms)
    failed = [("failed", float("nan"))] * n_sites
    try:
        structure = _ase_to_structure(atoms)
        if len(structure) != n_sites:  # adaptor sanity (should never happen)
            return failed
        from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
            SimplestChemenvStrategy,
        )
        from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
            LocalGeometryFinder,
        )

        lgf = LocalGeometryFinder()
        lgf.setup_parameters(centering_type="standard")
        lgf.setup_structure(structure)
        se = _quiet(
            lgf.compute_structure_environments,
            maximum_distance_factor=1.41,
            only_cations=False,
            only_indices=None,
        )
        strategy = SimplestChemenvStrategy(distance_cutoff=1.4, angle_cutoff=0.3)
        strategy.set_structure_environments(se)

        out: list[tuple[str, float]] = []
        for isite in range(len(structure)):
            try:
                envs = strategy.get_site_coordination_environments(
                    site=structure[isite], isite=isite
                )
                best_symbol, best_csm = "failed", float("nan")
                for env in envs or []:
                    if env is None:  # strategy found no environment
                        continue
                    symbol, item = env
                    csm = float(item.get("symmetry_measure", math.nan))
                    if math.isnan(csm):
                        continue
                    if best_symbol == "failed" or csm < best_csm:
                        best_symbol, best_csm = str(symbol), csm
                out.append((best_symbol, best_csm))
            except Exception:
                out.append(("failed", float("nan")))
        return out
    except Exception:
        return failed


# ---------------------------------------------------------------------------
# oracle-internal consistency utilities (placeholder for the dim-confusion figure
# until level1 produces its dimspectrum parquet)
# ---------------------------------------------------------------------------

def _graph_looprank_dim(bonded_structure) -> int:
    """0/1/2/3: max over connected components of rank{loop translations}.

    BFS assigns each site a cumulative lattice translation ``t``; every edge
    (u -> v, image S) visited while v is already known yields a loop vector
    ``t_u + S - t_v``. This is an independent re-implementation of the idea
    behind the Larsen score (and of the fast layer's §3.3 algorithm) used for
    the oracle self-consistency placeholder figure.

    The StructureGraph is a *directed* graph: an edge stored as (u -> v,
    to_jimage=S) means v sits at image S relative to u, so reverse traversal
    uses -S. The rank of the loop lattice is computed exactly as the real
    matrix rank of the loop vectors (for a finitely generated subgroup of
    Z^3, rank_Z = dim_Q of its Q-span = real rank).
    """
    graph = bonded_structure.graph
    n = len(bonded_structure.structure)
    seen: list[np.ndarray | None] = [None] * n
    best_rank = 0
    for root in range(n):
        if seen[root] is not None:
            continue
        seen[root] = np.zeros(3, dtype=np.int64)
        stack = [root]
        loops: list[np.ndarray] = []

        def relax(u, v, s):
            """Walk u -> v where v sits at image s relative to u."""
            if seen[v] is None:
                seen[v] = seen[u] + s
                stack.append(v)
            else:
                loops.append(seen[u] + s - seen[v])

        while stack:
            u = stack.pop()
            for v, edge_map in graph[u].items():  # out-edges: v at +S rel u
                for data in edge_map.values():
                    relax(u, v, np.asarray(data.get("to_jimage", (0, 0, 0)),
                                            dtype=np.int64))
            for w, edge_map in graph.pred[u].items():  # in-edges: w at -S rel u
                for data in edge_map.values():
                    relax(u, w, -np.asarray(data.get("to_jimage", (0, 0, 0)),
                                             dtype=np.int64))
        rank = (
            int(np.linalg.matrix_rank(np.asarray(loops, dtype=float)))
            if loops
            else 0
        )
        best_rank = max(best_rank, rank)
    return best_rank


def _looprank_one(item: tuple[str, Path | None]) -> dict:
    """(sid, looprank_dim|None, error|None) for one structure."""
    sid, path = item
    if path is None:
        return {"structure_id": sid, "looprank_dim": None, "looprank_error": "file not found"}
    try:
        from ase.io import read

        atoms = read(path)
        bonded = _quiet(_crystalnn().get_bonded_structure, _ase_to_structure(atoms))
        return {"structure_id": sid, "looprank_dim": int(_graph_looprank_dim(bonded)),
                "looprank_error": None}
    except Exception as exc:
        return {"structure_id": sid, "looprank_dim": None,
                "looprank_error": f"{type(exc).__name__}:{exc}"}


def _looprank_worker_chunk(chunk):
    out = []
    for item in chunk:
        try:
            out.append(_looprank_one(item))
        except Exception as exc:
            out.append({"structure_id": item[0], "looprank_dim": None,
                        "looprank_error": f"unhandled:{type(exc).__name__}:{exc}"})
    return out


def _looprank_batch(manifest_or_folder, n_proc: int = 8) -> pd.DataFrame:
    """Multiprocess loop-translation-rank dim over CrystalNN bonded graphs.

    Private helper for the oracle self-consistency placeholder figure
    (fig1 until level1 ships its dimspectrum parquet).
    """
    items = list(_iter_structure_items(manifest_or_folder))
    if not items:
        raise ValueError(f"未找到任何结构: {manifest_or_folder}")
    n_proc = max(1, min(int(n_proc), max(8, int(os.environ.get("CKT_MAX_PROC", "8")))))
    chunks = [items[i : i + 16] for i in range(0, len(items), 16)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        futures = {pool.submit(_looprank_worker_chunk, ch): ch for ch in chunks}
        for fut in as_completed(futures):
            try:
                rows.extend(fut.result())
            except Exception as exc:
                for sid, _p in futures[fut]:
                    rows.append({"structure_id": sid, "looprank_dim": None,
                                 "looprank_error": f"worker_crashed:{type(exc).__name__}"})
    df = pd.DataFrame(rows, columns=["structure_id", "looprank_dim", "looprank_error"])
    order = {sid: i for i, (sid, _p) in enumerate(items)}
    df["_order"] = df["structure_id"].map(order)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------

def _resolve_local_file(sid: str, manifest_path: Path) -> Path | None:
    """Portable resolution: manifest path as-is, else KT_ROOT/data/{subset_2k,subset_20k}."""
    candidates = [manifest_path]
    for subset in ("subset_2k", "subset_20k"):
        candidates.append(KT_ROOT / "data" / subset / f"{sid}.vasp")
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _iter_structure_items(manifest_or_folder):
    """Yield (structure_id, Path|None) in input order.

    ``manifest_or_folder`` is either a folder of ``*.vasp`` files (structure_id =
    stem) or a manifest jsonl whose ``path`` field gives the source filename.
    """
    src = Path(manifest_or_folder)
    if src.is_dir():
        for f in sorted(src.glob("*.vasp")):
            yield f.stem, f
        return
    if src.is_file():
        seen: set[str] = set()
        with src.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                p = Path(rec["path"])
                sid = p.stem
                if sid in seen:
                    continue
                seen.add(sid)
                yield sid, _resolve_local_file(sid, p)
        return
    raise FileNotFoundError(
        f"manifest 或文件夹不存在: {src}（相对 KT 根: {src.resolve()})"
    )


def _oracle_one(item: tuple[str, Path | None], chemenv: bool) -> dict:
    sid, path = item
    row = {
        "structure_id": sid,
        "larsen_dim": None,
        "cnn_cn_mean": None,
        "cnn_cn": None,
        "chemenv_symbols": None,
        "chemenv_csm": None,
        "oracle_error": None,
    }
    errors: list[str] = []
    if path is None:
        row["oracle_error"] = "input file not found locally (not in data/subset_2k|20k)"
        return row
    try:
        from ase.io import read

        atoms = read(path)
    except Exception as exc:  # unparseable structure -> error row, batch continues
        row["oracle_error"] = f"read:{type(exc).__name__}:{exc}"
        return row
    try:
        row["larsen_dim"] = int(run_larsen(atoms))
    except Exception as exc:
        errors.append(f"larsen:{type(exc).__name__}:{exc}")
    try:
        cn = run_crystalnn_cn(atoms)
        row["cnn_cn"] = [int(x) for x in cn]
        row["cnn_cn_mean"] = float(np.mean(cn))
    except Exception as exc:
        errors.append(f"crystalnn:{type(exc).__name__}:{exc}")
    if chemenv:
        pairs = run_chemenv(atoms)  # never raises
        row["chemenv_symbols"] = [p[0] for p in pairs]
        row["chemenv_csm"] = [float(p[1]) for p in pairs]
        if all(sym == "failed" for sym in row["chemenv_symbols"]):
            errors.append("chemenv:all_sites_failed")
    row["oracle_error"] = "; ".join(errors) if errors else None
    return row


def _worker_chunk(chunk, chemenv: bool) -> list[dict]:
    """Worker: process one chunk of (sid, path); never raises for a single row."""
    rows = []
    for item in chunk:
        try:
            rows.append(_oracle_one(item, chemenv))
        except Exception as exc:  # last-resort guard: batch must survive
            rows.append(
                {
                    "structure_id": item[0],
                    "larsen_dim": None,
                    "cnn_cn_mean": None,
                    "cnn_cn": None,
                    "chemenv_symbols": None,
                    "chemenv_csm": None,
                    "oracle_error": f"unhandled:{type(exc).__name__}:{exc}",
                }
            )
    return rows


def batch_oracle(manifest_or_folder, out_parquet: Path, n_proc: int = 8,
                 chemenv: bool = False) -> None:
    """Multiprocess oracle batch -> parquet with the frozen §3.9 columns.

    ``manifest_or_folder``: jsonl manifest or a folder of ``*.vasp`` POSCARs.
    A failing structure lands in the ``oracle_error`` column and never aborts the
    batch. n_proc is capped at 8 (project rule).
    """
    out_parquet = Path(out_parquet)
    items = list(_iter_structure_items(manifest_or_folder))
    if not items:
        raise ValueError(f"未找到任何结构: {manifest_or_folder}")
    n_proc = max(1, min(int(n_proc), max(8, int(os.environ.get("CKT_MAX_PROC", "8")))))
    chunk_size = 4 if chemenv else 16  # ChemEnv is the slow oracle
    chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        futures = {pool.submit(_worker_chunk, ch, chemenv): ch for ch in chunks}
        for fut in as_completed(futures):
            chunk = futures[fut]
            try:
                rows.extend(fut.result())
            except Exception as exc:  # worker-level crash: keep the batch alive
                for sid, _path in chunk:
                    rows.append(
                        {
                            "structure_id": sid,
                            "larsen_dim": None,
                            "cnn_cn_mean": None,
                            "cnn_cn": None,
                            "chemenv_symbols": None,
                            "chemenv_csm": None,
                            "oracle_error": f"worker_crashed:{type(exc).__name__}",
                        }
                    )

    df = pd.DataFrame(rows, columns=ORACLE_COLUMNS)
    order = {sid: i for i, (sid, _p) in enumerate(items)}
    df["_order"] = df["structure_id"].map(order)
    df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)


# ---------------------------------------------------------------------------
# fast-CN lambda sensitivity scan (contract v1.1: bidirectional BondGraph)
# ---------------------------------------------------------------------------

def _fast_cn_from_graph(graph, n_atoms: int) -> np.ndarray:
    """Undirected degree per atom from the (bidirectional) v1.1 BondGraph.

    Distinct (j, S) neighbour count per atom (mirrors coord._count_cn on a
    bidirectional graph: the forward half already carries every neighbour;
    self-bonds at i keep both images, consistent with level3 semantics).
    """
    keys = np.unique(
        np.column_stack([np.asarray(graph.i, dtype=np.int64),
                         np.asarray(graph.j, dtype=np.int64),
                         np.asarray(graph.S, dtype=np.int64).reshape(-1, 3)]),
        axis=0,
    )
    if keys.size == 0:
        return np.zeros(n_atoms, dtype=np.int64)
    return np.bincount(keys[:, 0], minlength=n_atoms).astype(np.int64)


def _cn_lambda_worker(chunk, lambdas) -> list[dict]:
    """Worker: per structure per lambda -> {structure_id, lam, fast_cn(array),
    symbols(list)}. A failing structure/lambda is skipped (recorded as error)."""
    from ase.io import read

    from crysh.bond import build_bond_graph

    rows = []
    for sid, path in chunk:
        if path is None:
            continue
        try:
            atoms = read(path)
        except Exception as exc:
            rows.append({"structure_id": sid, "lam": None,
                         "error": f"read:{type(exc).__name__}:{exc}"})
            continue
        symbols = list(atoms.get_chemical_symbols())
        n = len(atoms)
        for lam in lambdas:
            try:
                graph = build_bond_graph(atoms, cutoff_table=None, lam=float(lam))
                deg = _fast_cn_from_graph(graph, n)
            except Exception as exc:
                rows.append({"structure_id": sid, "lam": float(lam),
                             "error": f"{type(exc).__name__}:{exc}"})
                continue
            rows.append({"structure_id": sid, "lam": float(lam), "fast_cn": deg,
                         "symbols": symbols})
    return rows


def _cn_lambda_scan(manifest_or_folder, oracle_parquet: Path,
                    lambdas=(1.1, 1.2, 1.3), n_proc: int = 8) -> pd.DataFrame:
    """Per-atom fast-CN vs CrystalNN-CN table for a lambda grid.

    Long-form DataFrame: structure_id, lam, larsen_dim, symbol, fast_cn,
    oracle_cn (one row per atom per lambda, site-aligned). Uses crysh.dimensionality's
    v1.1 bidirectional graph (default covalent table) for the fast CN; the
    oracle CN comes from an oracle parquet produced by batch_oracle.
    """
    oracle_parquet = Path(oracle_parquet)
    if not oracle_parquet.is_file():
        raise FileNotFoundError(f"oracle parquet 不存在: {oracle_parquet}")
    oracle = pd.read_parquet(oracle_parquet)
    _require_columns(oracle, ["structure_id", "cnn_cn", "larsen_dim"], "oracle")
    oracle_map = {
        r.structure_id: (np.asarray(r.cnn_cn, dtype=np.int64), int(r.larsen_dim))
        for r in oracle.itertuples()
        if r.cnn_cn is not None and r.larsen_dim is not None
    }
    items = list(_iter_structure_items(manifest_or_folder))
    if not items:
        raise ValueError(f"未找到任何结构: {manifest_or_folder}")
    n_proc = max(1, min(int(n_proc), max(8, int(os.environ.get("CKT_MAX_PROC", "8")))))
    chunks = [items[i : i + 32] for i in range(0, len(items), 32)]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        futures = {pool.submit(_cn_lambda_worker, ch, list(lambdas)): ch
                   for ch in chunks}
        for fut in as_completed(futures):
            try:
                rows.extend(fut.result())
            except Exception as exc:  # worker crash: keep going
                for sid, _p in futures[fut]:
                    rows.append({"structure_id": sid, "lam": None,
                                 "error": f"worker_crashed:{type(exc).__name__}"})
    records = []
    for row in rows:
        sid, lam = row["structure_id"], row["lam"]
        if lam is None:
            continue
        if sid not in oracle_map:
            continue
        oracle_cn, larsen_dim = oracle_map[sid]
        fast_cn = np.asarray(row["fast_cn"], dtype=np.int64)
        if len(fast_cn) != len(oracle_cn):
            continue
        for sym, fc, oc in zip(row["symbols"], fast_cn, oracle_cn):
            records.append({"structure_id": sid, "lam": float(lam),
                            "larsen_dim": int(larsen_dim), "symbol": str(sym),
                            "fast_cn": int(fc), "oracle_cn": int(oc)})
    df = pd.DataFrame(records,
                      columns=["structure_id", "lam", "larsen_dim", "symbol",
                               "fast_cn", "oracle_cn"])
    df["exact"] = df["fast_cn"] == df["oracle_cn"]
    df["abs_err"] = (df["fast_cn"] - df["oracle_cn"]).abs()
    return df


# ---------------------------------------------------------------------------
# alignment report
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, cols, tag: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{tag} parquet 缺少列: {missing}（实际列: {list(df.columns)}）")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    raise TypeError(f"不可序列化: {type(obj)}")


def _expand_hist(hist) -> list[int] | None:
    """cn_hist -> sorted per-atom CN list.

    Accepted formats (fast-layer intermediates):
    * dict ``{cn: count}`` (or its JSON string),
    * 1-D array indexed by CN (level3 coord_2k format: value = count of atoms
      with that CN, bins 0..len-1).
    """
    if hist is None:
        return None
    if isinstance(hist, str):
        try:
            hist = json.loads(hist)
        except Exception:
            return None
    try:
        if isinstance(hist, dict):
            out: list[int] = []
            for cn, count in hist.items():
                out.extend([int(cn)] * int(count))
            return sorted(out)
        arr = np.asarray(hist)
        if arr.ndim == 1:
            out = []
            for cn, count in enumerate(arr):
                out.extend([int(cn)] * int(count))
            return sorted(out)
        return None
    except Exception:
        return None


def alignment_report(fast_parquet: Path, oracle_parquet: Path, out_dir: Path) -> dict:
    """fast vs oracle alignment metrics; writes json + PNGs under out_dir.

    Outputs: 4x4 dimensionality confusion matrix (rows = fast d_star, cols =
    oracle larsen_dim) + per-class F1 + macro-F1 (structures flagged by
    ``ambiguous_dim_flag`` are excluded from the main F1 and reported
    separately), CN MAE / exact-match (per-atom via ``cn_hist`` when present,
    otherwise structure-level), ChemEnv CSM summary when available.

    ``out_dir``: base directory; json goes to ``out_dir/data/alignment_report.json``,
    figures to ``out_dir/fig/``. Raises FileNotFoundError with a clear message
    when an input does not exist.
    """
    fast_parquet = Path(fast_parquet)
    oracle_parquet = Path(oracle_parquet)
    out_dir = Path(out_dir)
    if not fast_parquet.is_file():
        raise FileNotFoundError(
            f"fast parquet 不存在: {fast_parquet}（等 level1/level3 产出后重跑）"
        )
    if not oracle_parquet.is_file():
        raise FileNotFoundError(
            f"oracle parquet 不存在: {oracle_parquet}（先跑 batch_oracle 产出）"
        )

    fast = pd.read_parquet(fast_parquet)
    oracle = pd.read_parquet(oracle_parquet)
    _require_columns(fast, ["structure_id"], "fast")
    _require_columns(oracle, ["structure_id"], "oracle")
    has_dim = "d_star" in fast.columns and "larsen_dim" in oracle.columns
    has_cn = "mean_cn" in fast.columns and "cnn_cn_mean" in oracle.columns
    if not has_dim and not has_cn:
        raise ValueError(
            "fast parquet 需含 d_star（level1）和/或 mean_cn（level3）列以对齐，"
            f"实际列: {list(fast.columns)}"
        )

    merged = fast.merge(oracle, on="structure_id", how="inner",
                        suffixes=("", "_oracle"))
    report: dict = {
        "fast_parquet": str(fast_parquet),
        "oracle_parquet": str(oracle_parquet),
        "n_fast": int(len(fast)),
        "n_oracle": int(len(oracle)),
        "n_merged": int(len(merged)),
    }

    # ---------------- dimensionality ----------------
    if has_dim:
        dim = merged[["structure_id", "d_star", "larsen_dim"]].copy()
        if "ambiguous_dim_flag" in merged.columns:
            dim["ambiguous"] = (
                merged["ambiguous_dim_flag"].fillna(False).astype(bool)
            )
        else:
            dim["ambiguous"] = False
        n_raw = len(dim)
        dim = dim.dropna(subset=["d_star", "larsen_dim"])
        dim["d_fast"] = dim["d_star"].round().astype(int)
        dim["d_true"] = dim["larsen_dim"].round().astype(int)
        dim = dim[
            dim["d_fast"].isin(DIM_VALUES) & dim["d_true"].isin(DIM_VALUES)
        ]
        ambiguous = dim[dim["ambiguous"]]
        main = dim[~dim["ambiguous"]]
        cm = np.zeros((4, 4), dtype=int)
        for d_fast, d_true in zip(main["d_fast"], main["d_true"]):
            cm[int(d_fast), int(d_true)] += 1
        per_class_f1: dict[str, float] = {}
        for c in range(4):
            tp = int(cm[c, c])
            fp = int(cm[c, :].sum()) - tp
            fn = int(cm[:, c].sum()) - tp
            denom = 2 * tp + fp + fn
            per_class_f1[DIM_LABELS[c]] = float(2 * tp / denom) if denom > 0 else 0.0
        macro_f1 = float(np.mean(list(per_class_f1.values())))
        amb_counts = {
            int(k): int(v)
            for k, v in ambiguous["d_true"].value_counts().items()
        }
        report["dim"] = {
            "labels": DIM_LABELS,
            "confusion_4x4_fast_row_oracle_col": cm.tolist(),
            "per_class_f1": per_class_f1,
            "macro_f1": macro_f1,
            "accuracy": float(np.trace(cm) / max(1, int(cm.sum()))),
            "n_main": int(len(main)),
            "n_ambiguous_excluded": int(len(ambiguous)),
            "ambiguous_larsen_dim_counts": amb_counts,
            "n_dropped_missing_or_out_of_range": int(n_raw - len(main) - len(ambiguous)),
        }
    else:
        report["dim"] = None
        cm = None

    # ---------------- coordination number ----------------
    if has_cn:
        cn = merged[["structure_id", "mean_cn", "cnn_cn_mean"]].dropna()
        cn_result: dict = {"n_structures": int(len(cn))}
        if len(cn) > 0:
            cn_result["mae_mean_cn"] = float(
                (cn["mean_cn"] - cn["cnn_cn_mean"]).abs().mean()
            )
            cn_result["rmse_mean_cn"] = float(
                np.sqrt(((cn["mean_cn"] - cn["cnn_cn_mean"]) ** 2).mean())
            )
            cn_result["pearson"] = (
                float(cn["mean_cn"].corr(cn["cnn_cn_mean"])) if len(cn) > 1 else None
            )
        else:
            cn_result["mae_mean_cn"] = None
            cn_result["rmse_mean_cn"] = None
            cn_result["pearson"] = None
        if "cn_hist" in merged.columns and "cnn_cn" in oracle.columns:
            hit = total = skipped = 0
            for _, r in merged.iterrows():
                fast_cns = _expand_hist(r.get("cn_hist"))
                oracle_cns = r.get("cnn_cn")
                if fast_cns is None or oracle_cns is None:
                    skipped += 1
                    continue
                oracle_cns = sorted(int(x) for x in oracle_cns)
                if len(fast_cns) != len(oracle_cns):
                    skipped += 1
                    continue
                hit += sum(a == b for a, b in zip(fast_cns, oracle_cns))
                total += len(fast_cns)
            cn_result["per_atom_exact_match"] = (
                float(hit / total) if total else None
            )
            cn_result["n_atoms_compared"] = total
            cn_result["n_structures_skipped_cn_hist"] = skipped
        else:
            cn_result["per_atom_exact_match"] = None
            cn_result["structure_level_exact_match"] = (
                float(
                    (
                        cn["mean_cn"].round() == cn["cnn_cn_mean"].round()
                    ).mean()
                )
                if len(cn) > 0
                else None
            )
        report["cn"] = cn_result
    else:
        report["cn"] = None

    # ---------------- ChemEnv summary ----------------
    if "chemenv_csm" in oracle.columns:
        csms = []
        symbols = []
        for lst in oracle["chemenv_csm"].dropna():
            for v in lst:
                if v is not None and not math.isnan(v):
                    csms.append(float(v))
        n_sites_total = 0
        for lst in oracle["chemenv_symbols"].dropna():
            for s in lst:
                n_sites_total += 1
                if s != "failed":
                    symbols.append(str(s))
        csms_arr = np.asarray(csms, dtype=float)
        chemenv_summary = {
            "n_sites_total": int(n_sites_total),
            "n_sites_ok": int(len(symbols)),
            "n_sites_failed": int(n_sites_total - len(symbols)),
        }
        if len(csms_arr):
            chemenv_summary["csm_mean"] = float(csms_arr.mean())
            chemenv_summary["csm_median"] = float(np.median(csms_arr))
            chemenv_summary["csm_p90"] = float(np.percentile(csms_arr, 90))
        else:
            chemenv_summary["csm_mean"] = chemenv_summary["csm_median"] = None
            chemenv_summary["csm_p90"] = None
        from collections import Counter

        chemenv_summary["top_symbols"] = Counter(symbols).most_common(20)
        report["chemenv"] = chemenv_summary
        csm_values = csms_arr
        symbol_counts = dict(Counter(symbols))
    else:
        report["chemenv"] = None
        csm_values = symbol_counts = None

    # ---------------- outputs ----------------
    data_dir = out_dir / "data"
    fig_dir = out_dir / "fig"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if cm is not None:
        _plot_dim_confusion(
            cm,
            fig_dir / "alignment_dim_confusion.png",
            title=f"fast d_star vs Larsen dim  (macro-F1 = {macro_f1:.3f}, "
                  f"n = {int(cm.sum())}, ambiguous excluded)",
        )
    if has_cn and len(cn) > 0:
        _plot_cn_scatter(cn, fig_dir / "alignment_cn_scatter.png")
    if csm_values is not None and len(csm_values):
        _plot_csm_hist(csm_values, fig_dir / "alignment_csm_dist.png",
                       title="ChemEnv CSM distribution (fast-oracle alignment run)")
    if symbol_counts:
        _plot_label_topk(symbol_counts, fig_dir / "alignment_chemenv_label_dist.png",
                         k=20,
                         title="ChemEnv environment symbols, top-20 (alignment run)")

    json_path = data_dir / "alignment_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    report["outputs"] = {
        "json": str(json_path),
        "fig_dir": str(fig_dir),
    }
    return report


# ---------------------------------------------------------------------------
# figure helpers (Agg backend, PNG, 300 dpi, labels + title — project rules)
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _plot_dim_confusion(cm: np.ndarray, path: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(4), labels=[f"true {l}" for l in DIM_LABELS])
    ax.set_yticks(range(4), labels=[f"pred {l}" for l in DIM_LABELS])
    ax.set_xlabel("Oracle (Larsen) dimensionality")
    ax.set_ylabel("Fast d_star")
    ax.set_title(title, fontsize=10)
    vmax = max(1, int(cm.max()))
    for i in range(4):
        for j in range(4):
            v = int(cm[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > 0.6 * vmax else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_cn_scatter(cn: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(cn["cnn_cn_mean"], cn["mean_cn"], s=12, alpha=0.5)
    lo = min(cn["cnn_cn_mean"].min(), cn["mean_cn"].min())
    hi = max(cn["cnn_cn_mean"].max(), cn["mean_cn"].max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y = x")
    ax.set_xlabel("Oracle mean CN (CrystalNN)")
    ax.set_ylabel("Fast mean_cn")
    ax.set_title("Fast mean_cn vs CrystalNN mean CN")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_csm_hist(values: np.ndarray, path: Path, title: str) -> None:
    plt = _plt()
    finite = np.asarray(values, dtype=float)
    finite = finite[~np.isnan(finite)]
    if finite.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    xmax = min(10.0, float(np.percentile(finite, 99)) * 1.2) or 10.0
    ax.hist(finite, bins=40, range=(0.0, xmax), color="#4C72B0", edgecolor="white")
    ax.set_xlabel("CSM (continuous symmetry measure)")
    ax.set_ylabel("site count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_label_topk(counts: dict, path: Path, k: int = 20, title: str = "") -> None:
    plt = _plt()
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:k]
    if not top:
        return
    labels = [str(sym) for sym, _ in top]
    values = [int(n) for _, n in top]
    fig, ax = plt.subplots(figsize=(6.4, 0.35 * k + 1.6))
    ax.barh(range(len(labels)), values, color="#55A868")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("site count")
    ax.set_ylabel("ChemEnv environment symbol")
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
