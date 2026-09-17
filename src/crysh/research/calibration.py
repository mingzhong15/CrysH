"""Phase-0 per-pair cutoff calibration: CrystalNN bond-length statistics -> the
``crysh.dimensionality`` ``cutoff_table`` format.

Ownership: phase0-calibration. Read-only dependencies: ``crysh.dimensionality`` /
``crysh.dimensionality`` (public API only; their private helpers are deliberately
re-implemented here so this module stays self-contained). pymatgen
(CrystalNN) and matplotlib are imported lazily inside functions.

Table semantics (frozen in contracts.md §3.2): an entry ``{(Zi, Zj): r0}`` is
a per-pair reference bond length in Angstrom, NOT yet scaled by the lambda
multiplier; an edge exists iff ``d_ij < lam * r0_ij`` (strict inequality,
enforced in bond.py). Pairs missing from the table fall back to the
covalent-radii sum inside bond.py.

Calibration rule (plan.md §目标):
* ``r0_ij = d_q95(pair)`` — the 95th percentile of CrystalNN bond lengths of
  the sorted element pair, so >= 95% of the reference CrystalNN bonds of that
  pair are captured by the fast bond graph at ``lam = 1.0``;
* pairs with fewer than ``min_samples`` reference bonds fall back to
  ``FALLBACK_COV_LAM (= 1.20) * (r_cov_i + r_cov_j)`` — the interim v1.1 global
  scale, so rare pairs behave exactly like the lambda* = 1.20 baseline;
* tie guard: because bond.py enforces the STRICT inequality ``d < lam*r0``,
  ``r0`` is inflated by the relative factor ``R0_TIE_GUARD = 1e-6`` so that
  bonds sitting exactly at ``d_q95`` (ubiquitous in symmetric crystals, where
  many distances are bit-identical) are not dropped. This keeps the
  ">= 95% captured" promise inclusive and is physically negligible
  (sub-femtometre for typical bond lengths).

Pipeline:

    extract_cnn_bonds(...)            -> bonds_2k.parquet   (structure_id, i, j, zi, zj, d)
    table_statistics(bonds_parquet)   -> pair_stats parquet (per-pair n / q50 / q95 / d_max / ...)
    build_cutoff_table(bonds_parquet) -> {(Zi, Zj): r0} dict (bond.py compatible)
    validate_cutoff_table(...)        -> validation_v1.json (CN site-aligned exact-match
                                        + d_star vs Larsen, baseline cov+lam=1.2 vs
                                        table+lam=1.0)
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
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.io import read

from crysh.bond import build_bond_graph
from crysh.dimensionality import dimensionality_spectrum
from .paths import KT_ROOT  # noqa: F401  (KT 工作目录，见 paths.py)

# Fallback scale for uncalibrated pairs: the interim v1.1 global lambda.
FALLBACK_COV_LAM = 1.20
# Relative tie guard on calibrated r0 (see module docstring).
R0_TIE_GUARD = 1e-6
# Project rule: never use more than 8 worker processes.
MAX_PROCS = max(8, int(os.environ.get("CKT_MAX_PROC", "8")))  # 集群可经 CKT_MAX_PROC 提升
# Element-equal pairs in CrystalNN also follow the same statistics.
_DEFAULT_MIN_SAMPLES = 30
_DEFAULT_QUANTILE = 0.95
# Frozen bonds parquet columns (per-structure intermediates may add more, but
# every bond row has exactly these).
BONDS_COLUMNS = ["structure_id", "i", "j", "zi", "zj", "d"]

# Covalent-radius fallback for elements without a tabulated radius (mirror of
# bond._pair_r0, kept local so this module is self-contained).
_COV_MEAN_FALLBACK = float(np.mean([r for r in covalent_radii if r > 0.0]) or 1.60)


# ---------------------------------------------------------------------------
# covalent-sum helpers (local mirror of the bond.py fallback semantics)
# ---------------------------------------------------------------------------

def _cov_radius(z: int) -> float:
    """Covalent radius of element Z in Angstrom (with mean fallback)."""
    if 0 < z < len(covalent_radii) and covalent_radii[z] > 0.0:
        return float(covalent_radii[z])
    return _COV_MEAN_FALLBACK


def _cov_sum(zi: int, zj: int) -> float:
    """r_cov(zi) + r_cov(zj) — the bond.py fallback r0 for an unknown pair."""
    return _cov_radius(int(zi)) + _cov_radius(int(zj))


# ---------------------------------------------------------------------------
# per-structure CrystalNN bond extraction
# ---------------------------------------------------------------------------

def extract_structure_cnn_bonds(atoms, mode: str = "union") -> list:
    """CrystalNN reference bonds of one structure -> list of bonds.

    Returns a list of 5-tuples ``(i, j, zi, zj, d)`` — one entry per
    *undirected* reference bond (bidirectional duplicates removed; each
    CrystalNN neighbour pair stored once, canonicalised ``i <= j``).
    ``i``/``j`` are site indices (identical to the ase atom order), ``zi``/
    ``zj`` atomic numbers, ``d`` the bond length in Angstrom.

    The neighbour set is ``CrystalNN.get_all_nn_info`` — the exact same call
    behind the oracle ``cnn_cn`` (oracle.run_crystalnn_cn). Deliberately NOT
    ``get_bonded_structure`` as the primary source: its StructureGraph edge
    set differs from the CN-defining neighbour set for some structures (e.g.
    agm001006038: In sites get degree 6 from the graph vs CN 3 from
    get_all_nn_info). Distances are computed with ``Structure.get_distance``
    (the nn-info ``weight`` is NOT a distance in pymatgen 2026.5.4 — it is 1
    for the unweighted CrystalNN).

    CrystalNN's neighbour relation is NOT symmetric (per-site weighted
    Voronoi): a site can list a neighbour that does not list it back
    (affects ~19% of sites on subset_2k; electropositive metals are listed
    by far more neighbours than they list). ``mode`` selects the symmetric
    projection used for training:

    * ``"union"`` (default): a bond is kept if EITHER endpoint lists it —
      identical to the CrystalNN bonded StructureGraph edge set;
    * ``"intersection"``: a bond is kept only if BOTH endpoints list it
      (drops one-sided entries; the conservative core of the reference set).
    """
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.io.ase import AseAtomsAdaptor

    if mode not in ("union", "intersection"):
        raise ValueError(f"mode 必须是 'union' 或 'intersection'，got {mode}")
    if not isinstance(atoms, Atoms):
        raise TypeError(f"atoms must be an ase.Atoms, got {type(atoms).__name__}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = AseAtomsAdaptor.get_structure(atoms)
        nn_lists = CrystalNN().get_all_nn_info(structure)

    z = np.asarray(atoms.numbers, dtype=np.int64)
    directed: set = set()
    for a, nns in enumerate(nn_lists):
        for nn in nns:
            b = int(nn["site_index"])
            s = tuple(int(round(float(x))) for x in nn["image"])
            directed.add((a, b, s))
    if mode == "intersection":
        directed = {t for t in directed
                    if (t[1], t[0], tuple(-np.asarray(t[2]))) in directed}

    seen: set = set()
    out: list = []
    for (a, b, s) in directed:
        if a > b:
            a_c, b_c, s_c = b, a, tuple(-np.asarray(s))
        else:
            a_c, b_c, s_c = a, b, s
        ck = (a_c, b_c, int(s_c[0]), int(s_c[1]), int(s_c[2]))
        if ck in seen:
            continue
        seen.add(ck)
        d = float(structure.get_distance(a_c, b_c, jimage=s_c))
        out.append((a_c, b_c, int(z[a_c]), int(z[b_c]), d))
    return out


def _extract_worker_chunk(chunk, mode: str = "union") -> list:
    """Worker: extract CrystalNN bonds for one chunk of (sid, path)."""
    rows = []
    for sid, path in chunk:
        if path is None:
            rows.append({"structure_id": sid, "error": "file not found", "bonds": None})
            continue
        try:
            atoms = read(path)
            bonds = extract_structure_cnn_bonds(atoms, mode=mode)
            rows.append({"structure_id": sid, "error": None, "bonds": bonds})
        except Exception as exc:  # a bad structure must never kill the batch
            rows.append({"structure_id": sid, "error": f"{type(exc).__name__}:{exc}",
                         "bonds": None})
    return rows


def _iter_structure_items(manifest_or_folder):
    """Yield (structure_id, Path|None) in input order.

    ``manifest_or_folder`` is a folder of ``*.vasp`` (structure_id = stem) or a
    manifest jsonl whose ``path`` field gives the source filename (resolution:
    path as-is, else KT_ROOT/data/{subset_2k,subset_20k}/{sid}.vasp). Local
    mirror of the oracle-bench iterator (private helpers not imported).
    """
    src = Path(manifest_or_folder)
    if src.is_dir():
        for f in sorted(src.glob("*.vasp")):
            yield f.stem, f
        return
    if src.is_file():
        seen: set = set()
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
                candidates = [p]
                for subset in ("subset_2k", "subset_20k"):
                    candidates.append(KT_ROOT / "data" / subset / f"{sid}.vasp")
                yield sid, next((c for c in candidates if c.is_file()), None)
        return
    raise FileNotFoundError(
        f"manifest 或文件夹不存在: {src}（相对 KT 根: {src.resolve()}）"
    )


def extract_cnn_bonds(folder_or_manifest, out_parquet: Path, n_proc: int = 8,
                      oracle_parquet=None, n_check: int = 10,
                      mode: str = "union") -> dict | None:
    """Batch-extract CrystalNN reference bonds -> parquet with BONDS_COLUMNS.

    ``folder_or_manifest``: folder of ``*.vasp`` or manifest jsonl.
    ``n_proc`` is capped at 8 (project rule). ``mode``: "union" (bond kept if
    either endpoint lists it — the CrystalNN bonded-graph edge set) or
    "intersection" (both endpoints must list it); see
    extract_structure_cnn_bonds. Failing structures are skipped from the
    parquet and reported in the returned summary.

    If ``oracle_parquet`` (an oracle_2k-style parquet with ``cnn_cn``) is
    given, a CN consistency spot check over the first ``n_check`` structures
    is performed: the per-atom neighbour count derived from the extracted
    bonds (``bincount`` over incident bond ends) is compared against the
    oracle ``cnn_cn``. CrystalNN's neighbour relation is asymmetric, so the
    union/intersection projections can deviate from ``cnn_cn`` on some
    sites; the check reports the per-structure mismatch counts (structures
    with fully symmetric relations must match exactly). The check report is
    merged into the returned summary dict; without ``oracle_parquet`` the
    return value is ``None``.
    """
    out_parquet = Path(out_parquet)
    items = list(_iter_structure_items(folder_or_manifest))
    if not items:
        raise ValueError(f"未找到任何结构: {folder_or_manifest}")
    n_proc = max(1, min(int(n_proc), MAX_PROCS))
    chunks = [items[i : i + 16] for i in range(0, len(items), 16)]

    bonds_rows: list = []
    failures: list = []
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        futures = {pool.submit(_extract_worker_chunk, ch, mode): ch for ch in chunks}
        for fut in as_completed(futures):
            chunk = futures[fut]
            try:
                rows = fut.result()
            except Exception as exc:  # worker crash: keep the batch alive
                for sid, _p in chunk:
                    failures.append((sid, f"worker_crashed:{type(exc).__name__}"))
                continue
            for row in rows:
                if row["error"] is not None:
                    failures.append((row["structure_id"], row["error"]))
                elif row["bonds"]:
                    sid = row["structure_id"]
                    bonds_rows.extend([(sid, *t) for t in row["bonds"]])

    df = pd.DataFrame(bonds_rows, columns=BONDS_COLUMNS)
    for col in ("i", "j", "zi", "zj"):
        df[col] = df[col].astype(np.int64)
    df["d"] = df["d"].astype(np.float64)
    order = {sid: k for k, (sid, _p) in enumerate(items)}
    df["_order"] = df["structure_id"].map(order)
    df = df.sort_values(["_order", "i", "j"]).drop(columns="_order").reset_index(drop=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    summary: dict = {
        "out_parquet": str(out_parquet),
        "n_structures": int(len(items)),
        "n_structures_with_bonds": int(df["structure_id"].nunique()),
        "n_bonds": int(len(df)),
        "n_failed": int(len(failures)),
        "failures": [{"structure_id": sid, "error": err} for sid, err in failures[:20]],
    }
    if oracle_parquet is not None:
        summary["cn_consistency_check"] = check_cn_consistency(
            out_parquet, Path(oracle_parquet), n_structures=int(n_check)
        )
    return summary


# ---------------------------------------------------------------------------
# CN consistency spot check
# ---------------------------------------------------------------------------

def check_cn_consistency(bonds_parquet: Path, oracle_parquet: Path,
                         n_structures: int = 10) -> dict:
    """Per-atom CN from extracted bonds vs oracle ``cnn_cn`` (spot check).

    For the first ``n_structures`` structures present in both inputs, the
    undirected neighbour count per atom (each bond stored once; atom ``i``
    gains one neighbour for every bond row with ``i`` or ``j`` equal to it)
    is compared against the oracle CrystalNN CN list. Site alignment relies on
    both sides using the same atom order (ase read order == pymatgen site
    order, as in the oracle batch).
    """
    bonds = pd.read_parquet(Path(bonds_parquet))
    oracle = pd.read_parquet(Path(oracle_parquet))
    if "cnn_cn" not in oracle.columns:
        raise ValueError(f"oracle parquet 缺少 cnn_cn 列: {list(oracle.columns)}")
    oracle_map = {
        r.structure_id: np.asarray(r.cnn_cn, dtype=np.int64)
        for r in oracle.itertuples()
        if r.cnn_cn is not None
    }
    per_structure = []
    for sid, g in bonds.groupby("structure_id", sort=False):
        if sid not in oracle_map:
            continue
        if len(per_structure) >= n_structures:
            break
        cnn = oracle_map[sid]
        n_atoms = int(len(cnn))
        ends = np.concatenate([np.asarray(g["i"]), np.asarray(g["j"])])
        cn = np.bincount(ends, minlength=n_atoms).astype(np.int64)
        n_mismatch = int((cn != cnn).sum())
        per_structure.append({
            "structure_id": str(sid),
            "n_atoms": n_atoms,
            "n_mismatch_sites": n_mismatch,
            "cn_match": bool(n_mismatch == 0),
        })
    return {
        "n_checked": int(len(per_structure)),
        "all_match": bool(per_structure and all(p["cn_match"] for p in per_structure)),
        "per_structure": per_structure,
    }


# ---------------------------------------------------------------------------
# per-pair statistics + cutoff table
# ---------------------------------------------------------------------------

def pair_statistics(bonds_parquet, quantile: float = _DEFAULT_QUANTILE) -> pd.DataFrame:
    """Per sorted element pair (zi <= zj): n, d_q50, d_q{quantile}, d_max,
    cov_sum and ratios.

    Columns: zi, zj, sym_i, sym_j, n, d_q50, d_q95 (or d_q{int(q*100)}),
    d_max, cov_sum, ratio_q50, ratio_q95.
    """
    if not 0.0 < float(quantile) <= 1.0:
        raise ValueError(f"quantile 必须在 (0,1]，got {quantile}")
    df = pd.read_parquet(Path(bonds_parquet))
    missing = [c for c in ("zi", "zj", "d") if c not in df.columns]
    if missing:
        raise ValueError(f"bonds parquet 缺少列: {missing}（实际: {list(df.columns)}）")
    zi = np.minimum(df["zi"].astype(np.int64), df["zj"].astype(np.int64))
    zj = np.maximum(df["zi"].astype(np.int64), df["zj"].astype(np.int64))
    q_col = f"d_q{int(round(float(quantile) * 100))}"
    stats = (
        df.assign(_zi=zi, _zj=zj)
        .groupby(["_zi", "_zj"], sort=True)["d"]
        .agg(
            n="count",
            d_q50=lambda s: float(np.percentile(s, 50)),
            **{q_col: lambda s: float(np.percentile(s, float(quantile) * 100))},
            d_max="max",
        )
        .reset_index()
        .rename(columns={"_zi": "zi", "_zj": "zj"})
    )
    stats["cov_sum"] = [_cov_sum(a, b) for a, b in zip(stats["zi"], stats["zj"])]
    stats["ratio_q50"] = stats["d_q50"] / stats["cov_sum"]
    stats["ratio_q95"] = stats[q_col] / stats["cov_sum"]
    stats["sym_i"] = [chemical_symbols[z] for z in stats["zi"]]
    stats["sym_j"] = [chemical_symbols[z] for z in stats["zj"]]
    stats["pair"] = stats["sym_i"] + "-" + stats["sym_j"]
    return stats.sort_values("n", ascending=False).reset_index(drop=True)


def table_statistics(bonds_parquet, min_samples: int = _DEFAULT_MIN_SAMPLES,
                     quantile: float = _DEFAULT_QUANTILE) -> pd.DataFrame:
    """Enriched per-pair statistics incl. the final table value.

    Columns: pair_statistics() columns + ``calibrated`` (n >= min_samples),
    ``r0_raw`` (the q95 bond length before the tie guard), ``r0`` (final table
    value: q95 x (1 + R0_TIE_GUARD) when calibrated, else
    FALLBACK_COV_LAM x cov_sum) and ``source`` ("calibrated" | "fallback").
    """
    stats = pair_statistics(bonds_parquet, quantile=quantile)
    q_col = f"d_q{int(round(float(quantile) * 100))}"
    min_samples = int(min_samples)
    cal = stats["n"] >= min_samples
    stats["calibrated"] = cal
    stats["r0_raw"] = stats[q_col]
    stats["r0"] = np.where(
        cal,
        stats[q_col] * (1.0 + R0_TIE_GUARD),
        stats["cov_sum"] * FALLBACK_COV_LAM,
    )
    stats["source"] = np.where(cal, "calibrated", "fallback")
    return stats


def build_cutoff_table(bonds_parquet, min_samples: int = _DEFAULT_MIN_SAMPLES,
                       quantile: float = _DEFAULT_QUANTILE) -> dict:
    """{(Zi, Zj): r0} cutoff table compatible with crysh.dimensionality.build_bond_graph.

    ``r0 = d_q95 x (1 + R0_TIE_GUARD)`` for pairs with ``n >= min_samples``
    reference CrystalNN bonds; otherwise the fallback
    ``FALLBACK_COV_LAM * (r_cov_i + r_cov_j)``. Keys are sorted int tuples
    ``(min(Zi, Zj), max(Zi, Zj))``.
    """
    stats = table_statistics(bonds_parquet, min_samples=min_samples, quantile=quantile)
    return {
        (int(row.zi), int(row.zj)): float(row.r0)
        for row in stats.itertuples()
    }


def coverage_metrics(bonds_parquet, min_samples: int = _DEFAULT_MIN_SAMPLES,
                     quantile: float = _DEFAULT_QUANTILE) -> dict:
    """Coverage documentation of the table over the reference bonds.

    Returns bond coverage (fraction of CrystalNN reference bonds whose pair is
    calibrated), fallback pair fraction (pairs with n < min_samples over all
    pairs), and the fraction of structures that contain at least one fallback
    pair among their extracted bonds ("structures involved").
    """
    df = pd.read_parquet(Path(bonds_parquet))
    stats = table_statistics(bonds_parquet, min_samples=min_samples, quantile=quantile)
    cal_pairs = {
        (int(r.zi), int(r.zj)) for r in stats.itertuples() if r.calibrated
    }
    zi = np.minimum(df["zi"].astype(np.int64), df["zj"].astype(np.int64))
    zj = np.maximum(df["zi"].astype(np.int64), df["zj"].astype(np.int64))
    bond_cal = np.fromiter(((int(a), int(b)) in cal_pairs for a, b in zip(zi, zj)),
                           dtype=bool, count=len(df))
    n_bonds = int(len(df))
    n_pairs = int(len(stats))
    n_fallback_pairs = int((~stats["calibrated"]).sum())
    involved = df.assign(_zi=zi, _zj=zj)
    involved["_cal"] = bond_cal
    n_struct_involved = int(
        involved.groupby("structure_id")["_cal"].apply(lambda s: not s.all()).sum()
    )
    return {
        "n_bonds_total": n_bonds,
        "n_bonds_covered": int(bond_cal.sum()),
        "bond_coverage": float(bond_cal.mean()) if n_bonds else 0.0,
        "n_pairs_total": n_pairs,
        "n_pairs_calibrated": n_pairs - n_fallback_pairs,
        "n_pairs_fallback": n_fallback_pairs,
        "fallback_pair_fraction": float(n_fallback_pairs / n_pairs) if n_pairs else 0.0,
        "n_structures_involved_fallback": n_struct_involved,
        "n_structures_total": int(df["structure_id"].nunique()),
        "structure_involvement_fraction": (
            float(n_struct_involved / df["structure_id"].nunique())
            if df["structure_id"].nunique() else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# table (de)serialisation
# ---------------------------------------------------------------------------

def save_cutoff_table(table: dict, json_path: Path, meta: dict | None = None) -> None:
    """Write the table to JSON: {"r0": {"Zi,Zj": r0, ...}, "meta": {...}}.

    Keys are normalised to sorted int tuples and serialised as "lo,hi" strings
    (JSON requires string keys); loaded back with load_table().
    """
    payload: dict = {
        "r0": {
            f"{min(int(a), int(b))},{max(int(a), int(b))}": float(v)
            for (a, b), v in sorted(table.items())
        }
    }
    if meta is not None:
        payload["meta"] = meta
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def load_table(json_path: Path) -> dict:
    """Load a calibration json -> plain {(Zi, Zj): r0} dict (sorted int keys).

    Accepts both the {"r0": {...}, "meta": ...} container produced by
    save_cutoff_table and a bare string-keyed mapping.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "r0" in data and isinstance(data["r0"], dict):
        data = data["r0"]
    if not isinstance(data, dict):
        raise ValueError(f"cutoff 表 json 格式错误: {json_path}")
    out: dict = {}
    for key, val in data.items():
        if isinstance(key, str):
            a, b = key.split(",")
            zi, zj = int(a), int(b)
        else:
            zi, zj = int(key[0]), int(key[1])
        out[(min(zi, zj), max(zi, zj))] = float(val)
    return out


# ---------------------------------------------------------------------------
# self-consistency ceiling: what any symmetric pair-cutoff graph can achieve
# ---------------------------------------------------------------------------

def self_consistency_ceiling(bonds_parquet: Path, oracle_parquet: Path) -> dict:
    """Upper bound for ANY symmetric pair-cutoff bond graph vs the oracle CN.

    A symmetric graph (any set of per-pair cutoffs) can at best reproduce the
    *per-structure union* of CrystalNN's (asymmetric) neighbour relation —
    the extracted undirected bond set. This function reports the site-aligned
    exact-match of the union-derived CN against the oracle ``cnn_cn``, plus
    the delta distribution and the fraction of mismatching structures.
    CrystalNN's neighbour relation is asymmetric for ~19% of the 2k sites
    (electropositive metals are listed by more neighbours than they list
    themselves), so this ceiling sits well below 100%.
    """
    bonds = pd.read_parquet(Path(bonds_parquet))
    oracle = pd.read_parquet(Path(oracle_parquet))
    if "cnn_cn" not in oracle.columns:
        raise ValueError(f"oracle parquet 缺少 cnn_cn 列: {list(oracle.columns)}")
    oracle_map = {
        r.structure_id: np.asarray(r.cnn_cn, dtype=np.int64)
        for r in oracle.itertuples()
        if r.cnn_cn is not None
    }
    n_struct = n_atoms = n_exact = n_struct_mismatch = 0
    deltas: list = []
    for sid, g in bonds.groupby("structure_id", sort=False):
        if sid not in oracle_map:
            continue
        cnn = oracle_map[sid]
        ends = np.concatenate([np.asarray(g["i"]), np.asarray(g["j"])])
        cn = np.bincount(ends, minlength=len(cnn))
        d = cn - cnn
        n_struct += 1
        n_atoms += len(cnn)
        n_exact += int((d == 0).sum())
        if (d != 0).any():
            n_struct_mismatch += 1
        deltas.extend(d.tolist())
    deltas = np.asarray(deltas, dtype=np.int64)
    hist = np.bincount(np.abs(deltas), minlength=16).tolist()
    return {
        "n_structures": int(n_struct),
        "n_atoms": int(n_atoms),
        "site_exact_match_ceiling": float(n_exact / n_atoms) if n_atoms else None,
        "n_structures_with_mismatch": int(n_struct_mismatch),
        "structure_mismatch_fraction": (
            float(n_struct_mismatch / n_struct) if n_struct else 0.0),
        "mean_delta_union_minus_cnn": (
            float(deltas.mean()) if len(deltas) else 0.0),
        "abs_delta_histogram": hist,
    }


# ---------------------------------------------------------------------------
# validation: table + lam=1.0 vs baseline cov + lam=1.2 (CN + dimensionality)
# ---------------------------------------------------------------------------

def cn_from_graph(graph, n_atoms: int) -> np.ndarray:
    """Per-atom undirected CN from a (bidirectional) BondGraph.

    Counts distinct (j, S) neighbours per atom: unique (i, j, S) triples
    bincounted over the tail index. Mirrors coord._count_cn /
    oracle._fast_cn_from_graph semantics (private helpers not imported).
    """
    if graph is None or len(graph.i) == 0:
        return np.zeros(int(n_atoms), dtype=np.int64)
    keys = np.unique(
        np.column_stack([
            np.asarray(graph.i, dtype=np.int64),
            np.asarray(graph.j, dtype=np.int64),
            np.asarray(graph.S, dtype=np.int64).reshape(-1, 3),
        ]),
        axis=0,
    )
    if keys.size == 0:
        return np.zeros(int(n_atoms), dtype=np.int64)
    return np.bincount(keys[:, 0], minlength=int(n_atoms)).astype(np.int64)


def _validation_worker(chunk, table: dict, oracle_map: dict) -> list:
    """Worker: per structure compute CN (base + calibrated) and d_star spectra."""
    from ase.io import read as _read

    rows = []
    for sid, path in chunk:
        if path is None or sid not in oracle_map:
            continue
        try:
            atoms = _read(path)
            symbols = list(atoms.get_chemical_symbols())
            n = int(len(atoms))
            zs = sorted({int(x) for x in atoms.numbers})
            pairs = [(zs[a], zs[b]) for a in range(len(zs)) for b in range(a, len(zs))]
            oracle_cn, larsen_dim = oracle_map[sid]
            if len(oracle_cn) != n:
                continue
        except Exception:
            continue
        try:
            g_base = build_bond_graph(atoms, cutoff_table=None, lam=1.20)
            cn_base = cn_from_graph(g_base, n)
            spec_base = dimensionality_spectrum(atoms, cutoff_table=None,
                                                d_star_lambda=1.20)
        except Exception as exc:
            rows.append({"structure_id": sid, "error": f"baseline:{type(exc).__name__}"})
            continue
        try:
            g_new = build_bond_graph(atoms, cutoff_table=table, lam=1.0)
            cn_new = cn_from_graph(g_new, n)
            spec_new = dimensionality_spectrum(atoms, cutoff_table=table,
                                               d_star_lambda=1.0)
        except Exception as exc:
            rows.append({"structure_id": sid, "error": f"calibrated:{type(exc).__name__}"})
            continue
        rows.append({
            "structure_id": sid,
            "symbols": symbols,
            "cn_base": cn_base,
            "cn_new": cn_new,
            "oracle_cn": oracle_cn,
            "larsen_dim": int(larsen_dim),
            "d_base": int(spec_base.d_star),
            "pers_base": float(spec_base.persistence),
            "d_new": int(spec_new.d_star),
            "pers_new": float(spec_new.persistence),
            "element_pairs": pairs,
            "error": None,
        })
    return rows


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Series,)):
        return obj.tolist()
    raise TypeError(f"不可序列化: {type(obj)}")


def _cn_block(tag: str, records: list) -> dict:
    """Aggregate per-atom CN metrics over the collected per-structure records."""
    syms, f_base, f_new, o = [], [], [], []
    for r in records:
        for s, b, nw, oc in zip(r["symbols"], r["cn_base"], r["cn_new"], r["oracle_cn"]):
            syms.append(s)
            f_base.append(int(b))
            f_new.append(int(nw))
            o.append(int(oc))
    f_base = np.asarray(f_base, dtype=np.int64)
    f_new = np.asarray(f_new, dtype=np.int64)
    o = np.asarray(o, dtype=np.int64)
    syms = np.asarray(syms)

    def _agg(fast):
        m = fast == o
        return {
            "n_atoms": int(len(fast)),
            "exact_match": float(m.mean()) if len(m) else None,
            "mae": float(np.abs(fast - o).mean()) if len(fast) else None,
            "mean_delta_fast_minus_oracle": (
                float((fast - o).mean()) if len(fast) else None),
        }

    by_element = {}
    for sym in sorted(set(syms)):
        idx = syms == sym
        m = f_new[idx] == o[idx]
        by_element[str(sym)] = {
            "symbol": str(sym),
            "n_sites": int(idx.sum()),
            "base_exact": float((f_base[idx] == o[idx]).mean()) if idx.any() else None,
            "new_exact": float(m.mean()) if idx.any() else None,
            "new_mae": float(np.abs(f_new[idx] - o[idx]).mean()) if idx.any() else None,
            "new_mean_delta": float((f_new[idx] - o[idx]).mean()) if idx.any() else None,
        }
    dims_atom = np.asarray(
        [d for r in records for d in [int(r["larsen_dim"])] * len(r["symbols"])],
        dtype=np.int64,
    )
    by_dim_new = {}
    for d in range(4):
        idx = dims_atom == d
        if not idx.any():
            continue
        sub_f = f_new[idx]
        sub_o = o[idx]
        by_dim_new[str(d)] = {
            "n_atoms": int(len(sub_f)),
            "exact_match": float((sub_f == sub_o).mean()),
            "mae": float(np.abs(sub_f - sub_o).mean()),
        }
    worst = sorted(
        (v for v in by_element.values() if v["n_sites"] >= 50),
        key=lambda v: v["new_exact"] if v["new_exact"] is not None else 1.0,
    )[:10]
    return {
        "baseline": _agg(f_base),
        "calibrated": _agg(f_new),
        "delta_pp_exact": float(
            100.0 * (_agg(f_new)["exact_match"] - _agg(f_base)["exact_match"])),
        "by_element": by_element,
        "by_larsen_dim_new": by_dim_new,
        "residual_worst_by_element": worst,
    }


def _dim_block(tag: str, ds, ls, pers) -> dict:
    """Dimensionality accuracy vs Larsen with persistence < 0.5 excluded."""
    ds = np.asarray(ds, dtype=np.int64)
    ls = np.asarray(ls, dtype=np.int64)
    pers = np.asarray(pers, dtype=float)
    main = pers >= 0.5
    n_amb = int((~main).sum())
    cm = np.zeros((4, 4), dtype=int)
    for d_f, d_t in zip(ds[main], ls[main]):
        cm[int(d_f), int(d_t)] += 1
    per_class_f1 = {}
    for c in range(4):
        tp = int(cm[c, c])
        fp = int(cm[c, :].sum()) - tp
        fn = int(cm[:, c].sum()) - tp
        denom = 2 * tp + fp + fn
        per_class_f1[f"{c}D"] = float(2 * tp / denom) if denom > 0 else 0.0
    return {
        "accuracy": float(np.trace(cm) / max(1, int(cm.sum()))),
        "macro_f1": float(np.mean(list(per_class_f1.values()))),
        "per_class_f1": per_class_f1,
        "confusion_4x4_fast_row_oracle_col": cm.tolist(),
        "n_main": int(main.sum()),
        "n_ambiguous_excluded": n_amb,
    }


def validate_cutoff_table(folder_or_manifest, oracle_parquet: Path, table,
                          out_json: Path, n_proc: int = 8) -> dict:
    """Core validation: calibrated table + lam=1.0 vs interim baseline
    cov + lam=1.2, on the 2k subset.

    * CN: per-atom site-aligned exact-match / MAE vs oracle ``cnn_cn`` (plus
      per-element and per-Larsen-dim splits, residual worst elements);
    * dimensionality: ``crysh.dimensionality.d_star`` vs oracle ``larsen_dim`` with
      the oracle-bench ambiguous rule (``dim_persistence < 0.5`` excluded);
    * missing-pair analysis (structure element pairs absent from the table,
      which bond.py would resolve to the bare covalent sum).

    ``table`` is either a dict or a path to a calibration json (load_table).
    Writes the full report to ``out_json`` and returns it.
    """
    oracle_parquet = Path(oracle_parquet)
    out_json = Path(out_json)
    if not oracle_parquet.is_file():
        raise FileNotFoundError(f"oracle parquet 不存在: {oracle_parquet}")
    oracle = pd.read_parquet(oracle_parquet)
    for col in ("cnn_cn", "larsen_dim"):
        if col not in oracle.columns:
            raise ValueError(f"oracle parquet 缺少列 {col}: {list(oracle.columns)}")
    oracle_map = {
        r.structure_id: (np.asarray(r.cnn_cn, dtype=np.int64), int(r.larsen_dim))
        for r in oracle.itertuples()
        if r.cnn_cn is not None and r.larsen_dim is not None
    }
    if isinstance(table, (str, Path)):
        table = load_table(Path(table))
    if not isinstance(table, dict) or not table:
        raise ValueError("table 必须是非空 {(Zi,Zj): r0} dict 或 json 路径")
    table = {(int(a), int(b)): float(v) for (a, b), v in table.items()}

    items = list(_iter_structure_items(folder_or_manifest))
    if not items:
        raise ValueError(f"未找到任何结构: {folder_or_manifest}")
    n_proc = max(1, min(int(n_proc), MAX_PROCS))
    chunks = [items[i : i + 32] for i in range(0, len(items), 32)]
    records: list = []
    errors: list = []
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        futures = {pool.submit(_validation_worker, ch, table, oracle_map): ch
                   for ch in chunks}
        for fut in as_completed(futures):
            try:
                rows = fut.result()
            except Exception as exc:
                for sid, _p in futures[fut]:
                    errors.append((sid, f"worker_crashed:{type(exc).__name__}"))
                continue
            for row in rows:
                if row.get("error"):
                    errors.append((row["structure_id"], row["error"]))
                else:
                    records.append(row)

    n_struct = len(records)
    report = {
        "generated_by": "crysh.calibration.validate_cutoff_table",
        "n_structures": int(n_struct),
        "n_failed": int(len(errors)),
        "errors": [{"structure_id": s, "error": e} for s, e in errors[:20]],
        "cn": _cn_block("cn", records),
        "dim": {
            "baseline": _dim_block("base", [r["d_base"] for r in records],
                                   [r["larsen_dim"] for r in records],
                                   [r["pers_base"] for r in records]),
            "calibrated": _dim_block("new", [r["d_new"] for r in records],
                                     [r["larsen_dim"] for r in records],
                                     [r["pers_new"] for r in records]),
        },
    }
    report["dim"]["delta_pp_accuracy"] = float(
        100.0 * (report["dim"]["calibrated"]["accuracy"]
                 - report["dim"]["baseline"]["accuracy"])
    )

    # missing-pair analysis (bond.py resolves such pairs to bare cov sum)
    missing: dict = {}
    n_struct_missing = 0
    for r in records:
        miss = [p for p in r["element_pairs"]
                if (int(p[0]), int(p[1])) not in table]
        if miss:
            n_struct_missing += 1
            for p in miss:
                key = f"{p[0]},{p[1]}"
                missing[key] = missing.get(key, 0) + 1
    top_missing = sorted(missing.items(), key=lambda kv: -kv[1])[:10]
    report["missing_pairs"] = {
        "n_structures_with_missing_pairs": int(n_struct_missing),
        "fraction": float(n_struct_missing / n_struct) if n_struct else 0.0,
        "top_missing_pairs": [{"pair": k, "n_structures": v} for k, v in top_missing],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return report
