"""Periodic dimensionality spectrum d(lambda) — CKT Level 1.

For every cutoff multiplier lambda, the periodic bond graph is built
(``ckt.bond``). Each connected component is traversed by BFS and every node
is assigned a cumulative lattice translation ``t`` in Z^3 (``t[root] = 0``,
``t[v] = t[u] + S`` along a tree edge ``(u, v, S)``). A non-tree edge
``(u, v, S)`` yields the cycle translation ``T = t[u] + S - t[v]``.

The dimensionality of a component is the rank over Q (rationals) of its
cycle translations — computed exactly with a greedy integer basis using
cross/dot products (NOT mod-2). The structure dimensionality is the
*maximum* over connected components (contracts.md §3.3 "多连通分量取 max";
consistent with the Larsen definition evaluated per connected component).
Rank cannot exceed 3 (Z^3).

Default lambda grid — extended to (0.90, 1.00, 1.10, 1.20, 1.35, 1.50,
1.75, 2.00). Motivation: van der Waals interlayer contacts sit at
q = d_interlayer / r0 of roughly 1.6-2.2 (e.g. graphite: 3.35 A vs
2*r_cov(C) = 1.52 A, q ~ 2.2). With the former ceiling lambda = 1.50 such
contacts never bond, so the layered-bulk signature "2,2,2,2,3,3" (a 2->3
transition inside the grid) was invisible for real vdW materials — their
spectra saturated at d = 2. The 1.75 / 2.00 points make the 2->3 transition
observable for typical vdW stackings (q <~ 2.0); the persistence
denominator is len(lambdas) and thus naturally becomes 8 on the default
grid. All other v1.2 semantics are unchanged: table mode (explicit
``cutoff_table`` or $CKT_CUTOFF_TABLE) keeps d_star at lambda = 1.00, the
no-table interim keeps D_STAR_LAMBDA = 1.20, and an explicit
``d_star_lambda`` must be contained in the grid.

Multi-label output: besides the max-rank ``d_star``, the spectrum carries
``dim_components`` — per-connected-component ranks on the d_star graph
(key = rank, value = [n_components of that rank, atoms in them]) — so
mixed-dimensionality structures (e.g. a 2D sheet plus 0D molecules in one
cell) keep their low-dimensional information instead of collapsing to the
max.

Ownership: level1-dimensionality (contracts.md §2). Signatures frozen in
contracts.md §3.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ckt.bond import BondGraph, build_bond_graph

from .paths import KT_ROOT  # noqa: F401  (KT 工作目录，见 paths.py)

# Integrator decision v1.1: interim canonical bond scale. The bare covalent
# sum (lam=1.00) systematically under-bonds (diamond Si q=1.06; 40% zero-CN
# sites on subset_2k; oracle: 131 metallic structures fast-0D vs Larsen-3D).
# Superseded by the Phase-0 pair-cutoff calibration table when it lands.
D_STAR_LAMBDA = 1.20


@dataclass
class DimensionSpectrum:
    """d(lambda) spectrum of one structure (contract §3.3)."""
    d_by_lambda: dict     # {lambda: d} with d in {0, 1, 2, 3}
    d_star: int           # d at lambda = D_STAR_LAMBDA (interim 1.20, v1.1)
    persistence: float    # fraction of lambdas with d(lambda) == d_star
    n_components: int     # connected components of the lambda = D_STAR_LAMBDA graph (v1.1)
    translations_rank: int  # cycle-translation rank at lambda = D_STAR_LAMBDA (== d_star)
    # Multi-label per-component dimensionality, collected on the SAME graph
    # where n_components is counted (lambda = d_star_lambda):
    #   {rank in {0,1,2,3}: [n_components of that cycle-translation rank,
    #                        total atoms in those components]}
    # Invariants: sum(n_components) == n_components,
    #             sum(atoms) == n_atoms of the d_star graph,
    #             max(ranks present) == d_star.
    # Empty dict iff the d_star graph has no atoms (n_atoms == 0).
    dim_components: dict[int, list[int]] = field(default_factory=dict)


def _independent(t: np.ndarray, basis: list) -> bool:
    """True iff integer 3-vector ``t`` lies outside the Q-span of ``basis``.

    Exact integer arithmetic: parallelism via cross product (rank-1 basis),
    coplanarity via scalar triple product (rank-2 basis). ``len(basis) <= 2``
    is guaranteed by the caller.
    """
    k = len(basis)
    if k == 0:
        return True
    if k == 1:
        return bool(np.any(np.cross(basis[0], t) != 0))
    if k == 2:
        return bool(np.dot(np.cross(basis[0], basis[1]), t) != 0)
    return False  # rank already saturated at 3


def _graph_dim_and_components(g: BondGraph, component_stats: bool = False) -> tuple:
    """Return ``(d, n_components)`` for a periodic bond graph.

    ``d`` = max over connected components of the cycle-translation rank;
    ``n_components`` = number of connected components (isolated atoms count).

    With ``component_stats=True`` a third element is appended: a list of
    ``(rank, size)`` per connected component, where ``rank`` is that
    component's cycle-translation rank and ``size`` its atom count. The
    stats are collected inside the same single BFS pass (O(V+E)); callers
    that only need the rank pay no extra cost.
    """
    i, j, S, n_atoms = g.i, g.j, g.S, g.n_atoms
    # Undirected adjacency (CSR) with signed shifts: t[v] = t[u] + shift.
    # v1.1: BondGraph is already bidirectional (both (i,j,S) and (j,i,-S));
    # mirroring again is harmless (consistent duplicates) and keeps the
    # helper robust to one-directional inputs, so it is retained.
    u = np.concatenate([i, j])
    v = np.concatenate([j, i])
    s = np.concatenate([S, -S])
    order = np.argsort(u, kind="stable")
    u, v, s = u[order], v[order], s[order]
    ptr = np.searchsorted(u, np.arange(n_atoms + 1))

    visited = np.zeros(n_atoms, dtype=bool)
    t = np.zeros((n_atoms, 3), dtype=np.int64)
    max_rank = 0
    n_comp = 0
    stats: list = []  # (rank, size) per component, only if component_stats
    for root in range(n_atoms):
        if visited[root]:
            continue
        n_comp += 1
        visited[root] = True
        t[root] = 0
        stack = [root]
        basis: list = []
        size = 0
        while stack:
            a = stack.pop()
            size += 1
            for e in range(ptr[a], ptr[a + 1]):
                b = v[e]
                if not visited[b]:
                    visited[b] = True
                    t[b] = t[a] + s[e]
                    stack.append(b)
                elif len(basis) < 3:
                    T = t[a] + s[e] - t[b]
                    if np.any(T) and _independent(T, basis):
                        basis.append(T)
        if len(basis) > max_rank:
            max_rank = len(basis)
        if component_stats:
            stats.append((len(basis), size))
    if component_stats:
        return max_rank, n_comp, stats
    return max_rank, n_comp


def dimensionality_spectrum(atoms, cutoff_table=None,
                            lambdas=(0.90, 1.00, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00),
                            d_star_lambda: float | None = None) -> DimensionSpectrum:
    """Compute the d(lambda) dimensionality spectrum of a structure.

    For each ``lambda`` the periodic bond graph is rebuilt with the bond
    criterion ``d_ij < lambda * r0_ij`` and the cycle-translation rank per
    connected component is computed (max over components). ``d_star`` is
    taken at ``lambda = d_star_lambda``; ``persistence`` is the fraction of
    lambdas with ``d(lambda) == d_star`` (denominator ``len(lambdas)`` — 8
    on the default grid); ``n_components`` and ``dim_components`` are
    evaluated on the ``lambda = d_star_lambda`` graph.

    Default grid (extended): ``(0.90, 1.00, 1.10, 1.20, 1.35, 1.50, 1.75,
    2.00)`` — vdW interlayer contacts (q = d/r0 ~ 1.6-2.2, e.g. graphite
    3.35 A vs 2*r_cov(C) = 1.52 A) never bond below the former 1.50
    ceiling, hiding the 2->3 layered-bulk transition for real vdW
    materials. See the module docstring for details.

    ``dim_components`` (new field): on the same d_star graph where
    ``n_components`` is counted, every connected component contributes its
    cycle-translation rank and atom count; the aggregate maps
    ``rank -> [n_components of that rank, atoms in those components]``.
    Isolated atoms are rank-0 size-1 components; the field is ``{}`` iff
    the d_star graph has no atoms.

    v1.2 auto policy (``cutoff_table=None, d_star_lambda=None``):
    * if ``$CKT_CUTOFF_TABLE`` (Phase-0 calibration table) is set, the table
      is used and ``d_star_lambda`` defaults to 1.00 (table values are
      already at the CrystalNN-calibrated scale);
    * otherwise the interim covalent scale applies with
      ``d_star_lambda = D_STAR_LAMBDA`` (1.20 — the bare covalent sum at
      lambda=1.00 systematically under-bonds, e.g. diamond Si q=1.06>1,
      40% zero-CN sites on subset_2k; Phase-0 table supersedes).
    An explicitly passed ``d_star_lambda`` must be contained in ``lambdas``.
    """
    from ckt.bond import load_env_table
    if cutoff_table is None:
        cutoff_table = load_env_table()
    if d_star_lambda is None:
        d_star_lambda = 1.00 if cutoff_table is not None else D_STAR_LAMBDA
    if not lambdas:
        raise ValueError("lambdas must be non-empty")
    lambdas = tuple(float(x) for x in lambdas)
    d_star_lambda = float(d_star_lambda)

    d_by_lambda: dict = {}
    n_components = None
    dim_components: dict[int, list[int]] = {}
    lam_key = None
    for lam in lambdas:
        g = build_bond_graph(atoms, cutoff_table=cutoff_table, lam=lam)
        if abs(lam - d_star_lambda) < 1e-12:
            # d_star graph: collect per-component (rank, size) inside the
            # same single BFS pass that yields the rank and component count.
            rank, ncomp, comp_stats = _graph_dim_and_components(
                g, component_stats=True)
            n_components = int(ncomp)
            lam_key = lam
            for comp_rank, comp_size in comp_stats:
                entry = dim_components.setdefault(int(comp_rank), [0, 0])
                entry[0] += 1
                entry[1] += int(comp_size)
        else:
            rank, ncomp = _graph_dim_and_components(g)
        d_by_lambda[lam] = int(rank)
    if lam_key is None:
        raise ValueError(
            f"lambdas must include {d_star_lambda}: d_star is defined there (v1.1)")

    d_star = int(d_by_lambda[lam_key])
    persistence = float(
        sum(1 for lam in lambdas if d_by_lambda[lam] == d_star) / len(lambdas)
    )
    return DimensionSpectrum(
        d_by_lambda=d_by_lambda,
        d_star=d_star,
        persistence=persistence,
        n_components=n_components,
        translations_rank=d_star,
        dim_components=dim_components,
    )
