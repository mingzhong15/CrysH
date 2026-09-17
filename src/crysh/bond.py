"""Periodic bond graph infrastructure — CKT Level 1 (shared by L1-L5).

A *periodic bond graph* encodes the neighbour network of a periodic
structure. Every edge carries a periodic translation vector ``S`` in Z^3:
the head atom ``j`` sits in the cell image ``j + S . cell`` relative to the
tail atom ``i``. Summing ``S`` along any graph cycle yields the cycle's net
lattice translation ``T``; the rank over Q of the set of cycle translations
is the network dimensionality (computed in ``crysh.dimensionality``).

Ownership: level1-dimensionality (contracts.md §2). Signatures frozen in
contracts.md §3.2 — do not change without the integrator.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

# v1.2: env var pointing to the Phase-0 calibration table json. When set,
# `canonical_bond_graph` (and dimensionality_spectrum's auto path) use the
# per-pair table at lam=1.0 instead of the interim covalent*lambda* scale.
ENV_CUTOFF_TABLE = "CKT_CUTOFF_TABLE"

# Fallback radius (Angstrom) for elements whose covalent radius is unknown
# (<= 0) in the tabulated values. ASE 3.29's covalent_radii has no such
# entries, but the guard is kept for robustness against older tables
# (noble gases / Fr / Ra historically missing). Conservative choice: the
# mean covalent radius over all known elements (~1.60 A in ASE 3.29).
_COV_MEAN_FALLBACK = float(np.mean([r for r in covalent_radii if r > 0.0]) or 1.60)


def _pair_r0(zi: int, zj: int) -> float:
    """Covalent bond-length estimate r0 = r_cov(zi) + r_cov(zj) in Angstrom.

    Elements without a tabulated (positive) covalent radius fall back to the
    mean known covalent radius (documented conservative choice).
    """
    def _r(z: int) -> float:
        if 0 < z < len(covalent_radii) and covalent_radii[z] > 0.0:
            return float(covalent_radii[z])
        return _COV_MEAN_FALLBACK

    return _r(zi) + _r(zj)


@dataclass
class BondGraph:
    """Periodic bond graph with per-edge lattice translations (contract §3.2)."""
    i: np.ndarray          # (E,) int32 — tail atom index
    j: np.ndarray          # (E,) int32 — head atom index (j in cell image i + S.cell)
    S: np.ndarray          # (E,3) int32 — cell translation of j w.r.t. i (units of cell vectors)
    d: np.ndarray          # (E,) float64 — bond distance in Angstrom
    n_atoms: int
    lam: float             # cutoff multiplier actually used
    cutoff_mode: str       # "covalent" (default table) | "table" (user table)
    pair_r0: dict          # sorted (Zi, Zj) -> r0 in Angstrom (BEFORE lam scaling)


def default_covalent_table(atoms) -> dict:
    """Pairwise covalent cutoff radii for all (sorted) element pairs present.

    Returns ``{(Zi, Zj): r0}`` with ``Zi <= Zj`` and ``r0 = r_cov(Zi) +
    r_cov(Zj)`` in Angstrom (ase.data.covalent_radii is in Angstrom). Pairs
    involving elements without a tabulated radius fall back to the mean
    known covalent radius (see ``_pair_r0``).
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(f"atoms must be an ase.Atoms, got {type(atoms).__name__}")
    zs = sorted({int(z) for z in atoms.numbers})
    return {(zi, zj): _pair_r0(zi, zj) for a, zi in enumerate(zs) for zj in zs[a:]}


def build_bond_graph(atoms, cutoff_table=None, lam=1.0) -> BondGraph:
    """Build the periodic bond graph with bond criterion ``d_ij < lam * r0_ij``.

    Parameters
    ----------
    atoms : ase.Atoms
    cutoff_table : dict | None
        Optional user table ``{(Zi, Zj): r0}`` (unsorted keys accepted). Pairs
        missing from the table fall back to covalent radii (contracts plan
        §3.1 "unknown pair fallback").
    lam : float > 0
        Cutoff multiplier: an edge exists iff ``d_ij < lam * r0_ij``.

    Implementation uses ``ase.neighborlist.neighbor_list("ijdS", atoms,
    r_cut_dict)`` with ``r_cut_dict[(Zi, Zj)] = lam * r0`` (sorted keys).
    Home-cell self-loops (``i == j`` and ``S == (0,0,0)``) are excluded;
    exact duplicate ``(i, j, S)`` triples are removed. **Both directions of
    a pair are kept** (ASE-compatible adjacency): ``(i, j, S)`` and
    ``(j, i, -S)`` are distinct triples, so ``np.bincount(g.i)`` yields the
    per-atom directed edge count and undirected consumers must symmetrise
    (see ``coord._count_cn`` for the reference dedup pattern).
    """
    if not isinstance(atoms, Atoms):
        raise TypeError(f"atoms must be an ase.Atoms, got {type(atoms).__name__}")
    lam = float(lam)
    if not lam > 0:
        raise ValueError(f"lam must be > 0, got {lam}")

    cov = default_covalent_table(atoms)
    if cutoff_table is None:
        r0 = cov
        cutoff_mode = "covalent"
    else:
        norm: dict = {}
        for key, val in cutoff_table.items():
            zi, zj = int(key[0]), int(key[1])
            val = float(val)
            if not val > 0:
                raise ValueError(f"cutoff_table entry {key} must be > 0, got {val}")
            norm[(zi, zj) if zi <= zj else (zj, zi)] = val
        r0 = {pair: norm.get(pair, cov[pair]) for pair in cov}
        cutoff_mode = "table"

    r_cut = {pair: lam * r for pair, r in r0.items()}
    i, j, d, S = neighbor_list("ijdS", atoms, r_cut)
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    d = np.asarray(d, dtype=np.float64)
    S = np.asarray(S, dtype=np.int64).reshape(-1, 3)

    # Exclude home-cell self-loops (i == j and S == 0).
    keep = ~((i == j) & np.all(S == 0, axis=1))
    i, j, d, S = i[keep], j[keep], d[keep], S[keep]

    # De-duplicate EXACT (i, j, S) triples only — both directions of a pair
    # are kept (bidirectional adjacency, ASE-compatible). Integrator change
    # v1.1: the former one-directional canonicalisation broke bincount(g.i)
    # consumers (geometry CN undercount for high-index atoms).
    keys = np.stack([i, j, S[:, 0], S[:, 1], S[:, 2]], axis=1)
    keys, idx = np.unique(keys, axis=0, return_index=True)
    ii, jj, SS, d = keys[:, 0], keys[:, 1], keys[:, 2:], d[idx]

    # Belt-and-braces: enforce the strict bond criterion d < lam * r0 per pair
    # (ASE's internal comparison may be inclusive at the boundary).
    z = np.asarray(atoms.numbers, dtype=np.int64)
    zlo = np.minimum(z[ii], z[jj])
    zhi = np.maximum(z[ii], z[jj])
    lam_r0 = np.array([lam * r0[(int(a), int(b))] for a, b in zip(zlo, zhi)])
    keep2 = d < lam_r0
    ii, jj, SS, d = ii[keep2], jj[keep2], SS[keep2], d[keep2]

    return BondGraph(
        i=ii.astype(np.int32),
        j=jj.astype(np.int32),
        S=SS.astype(np.int32),
        d=d.astype(np.float64),
        n_atoms=len(atoms),
        lam=lam,
        cutoff_mode=cutoff_mode,
        pair_r0=r0,
    )


# --------------------------------------------------------------------------- #
# v1.2 canonical graph policy (integrator; contracts.md v1.2)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _load_table_json(path_str: str) -> dict:
    """Load a calibration table json -> {(Zi, Zj): r0} (sorted int keys).

    Accepts both the {"r0": {...}} container (phase0-calibration format) and
    a bare string-keyed mapping. Cached per path.
    """
    data = json.loads(Path(path_str).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("r0"), dict):
        data = data["r0"]
    if not isinstance(data, dict):
        raise ValueError(f"cutoff table json 格式错误: {path_str}")
    out: dict = {}
    for key, val in data.items():
        if isinstance(key, str):
            a, b = key.split(",")
            zi, zj = int(a), int(b)
        else:
            zi, zj = int(key[0]), int(key[1])
        out[(min(zi, zj), max(zi, zj))] = float(val)
    return out


def load_env_table() -> dict | None:
    """The Phase-0 calibration table from ``$CKT_CUTOFF_TABLE`` (None if unset
    or unreadable — the interim covalent*lambda* scale applies then)."""
    path = os.environ.get(ENV_CUTOFF_TABLE)
    if not path:
        return None
    try:
        return _load_table_json(path)
    except Exception:
        return None


def canonical_bond_graph(atoms, cutoff_table: dict | None = None) -> BondGraph:
    """v1.2 canonical periodic bond graph — single policy for pipeline/levels.

    * table available (explicit ``cutoff_table`` or ``$CKT_CUTOFF_TABLE``):
      ``build_bond_graph(atoms, table, lam=1.0)`` (table values are already
      at the CrystalNN-calibrated scale; phase0 tie-guarded);
    * else: interim fallback ``build_bond_graph(atoms, None,
      lam=D_STAR_LAMBDA)`` (covalent sum x 1.20).

    ``d_star`` convention follows automatically in
    ``crysh.dimensionality.dimensionality_spectrum`` (1.00 with table, D_STAR_LAMBDA
    otherwise).
    """
    table = cutoff_table if cutoff_table is not None else load_env_table()
    if table:
        return build_bond_graph(atoms, cutoff_table=table, lam=1.0)
    from crysh.config import D_STAR_LAMBDA
    return build_bond_graph(atoms, cutoff_table=None, lam=D_STAR_LAMBDA)
