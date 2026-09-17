"""controls — constructed ground-truth structures for CKT level tests.

Ownership: controls subtask (contracts.md §2). Read contracts.md §2/§3/§5
before changing anything here.

What this module provides
-------------------------
- ``build_all()``: deterministic, idempotent construction of ~40-60 control
  structures with hand-set physical ground-truth labels + reference
  expectations computed with contract-literal algorithms (default covalent
  table d(lambda) spectra, L0 validity quantities, CN reference, vacuum-gap
  estimate).
- ``load_labels()``: read ``controls/data/labels.yaml``.
- CLI (``python src/ckt/controls.py``): (re)generate
  ``controls/data/structures/*.vasp``, ``controls/data/labels.yaml`` and the
  three mandatory figures in ``controls/fig/``.

Label semantics (see labels.yaml ``definitions``; mirrored here)
----------------------------------------------------------------
- ``expected_d_star``: PHYSICAL ground-truth periodic bonding-network
  dimensionality of the material (0/1/2/3).  This is the taxonomy answer a
  correct classifier must give.  It is NOT necessarily the output of the
  frozen default-covalent-table algorithm at lambda=1.0 (see notes).
- ``expected_short_lambda_dim``: the d(lambda) values the frozen contract
  algorithm (contracts.md §3.2/§3.3: bond iff d_ij < lambda*(r_cov_i+r_cov_j)
  with ASE Cordero covalent radii, self-loops excluded, d = rank of loop
  translation set per component, max over components) produces for
  lambda in {0.90,1.00,1.10,1.20,1.35,1.50}.  Computed here by
  ``_ref_dim_spectrum`` and verified against analytic hand derivations.
  Structures where ``expected_short_lambda_dim["1.00"] != expected_d_star``
  are documented Cordero-under-bonding cases (Si, Al, NaCl, CaF2, ...):
  the pilot's Phase-0 pair-calibrated cutoff table is expected to fix them.
- ``expected_mean_cn``/``expected_cn_min``/``expected_cn_max``/
  ``expected_cn_hist``: physical coordination numbers.  Primary reference is
  pymatgen CrystalNN; when CrystalNN raises or disagrees with the hand-set
  histogram, a documented shell rule (neighbours with d < 1.25*(r_i+r_j)) is
  used (``cn_method`` field records which).
- ``expected_validity``: L0 quantities per contracts.md §3.4/§5 (q_min over
  the lambda=1.3 bond set, volume_norm, cell_kappa, aspect_ratio, flags).
- ``expected_vacuum_gap``: pilot-simple estimate per contracts.md §3.5
  wording: for each of the three lattice-axis projections take the largest
  circular gap in the fractional coordinate minus the covalent radii of the
  two flanking atoms; take the max over axes, clamped at 0.
- ``expected_site_geometries``: hand-set coordination-geometry labels for the
  inequivalent sites, using the frozen GEOMETRY_LABELS of contracts.md §3.7
  ("other" is used for shapes not in that list, with explanation in notes).

All construction is deterministic (no RNG).  The module is self-contained and
portable: the KT root is derived from ``__file__``; no absolute paths.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, surface
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

# ---------------------------------------------------------------------------
# Paths (portable: derived from this file's location)
# ---------------------------------------------------------------------------
from crysh.config import (
    LAMBDAS_STAR as LAMBDAS,  # 6 点子网格（见 config 与 labels.yaml 的 expected_short_lambda_dim）
)

# 2026-09-17：controls 整体搬到 code/levels/controls/ → 数据随模块走，不按工作目录拼
#: ground-truth 资产随库分发（labels.yaml + 44 个 POSCAR），不再依赖研究工作区布局
DATA_DIR = Path(__file__).resolve().parent / "controls_data"
STRUCTURES_DIR = DATA_DIR / "structures"
LABELS_PATH = DATA_DIR / "labels.yaml"
#: 默认出图目录（在包内，避免依赖任何研究工作区布局；调用方可用 out_dir 覆盖）
FIG_DIR = Path(__file__).resolve().parent / "figures"

# ---------------------------------------------------------------------------
# Frozen constants (mirrors of the frozen contract; keep in sync, do NOT edit
# the contract from here)
# ---------------------------------------------------------------------------
# contracts.md §3.5
MORPHOLOGY_CLASSES = [
    "dense_bulk", "layered_bulk", "explicit_vacuum_slab", "surface_like",
    "intrinsic_2d_like", "chain_solid", "isolated_chain", "molecular_crystal",
    "isolated_molecule", "porous_candidate", "ambiguous",
]
# contracts.md §3.7（v1.3-L4：+金属标签）
GEOMETRY_LABELS = [
    "linear", "trigonal_planar", "tetrahedral", "square_planar",
    "trigonal_bipyramidal", "square_pyramidal", "octahedral",
    "trigonal_prismatic", "cubic", "icosahedral",
    "cuboctahedral", "hcp_like", "bcc_like",
    "other", "ambiguous",
]
# contracts.md §3.3
# contracts.md §5
Q_C = 0.85
VOL_NORM_RANGE = (0.5, 10.0)
KAPPA_MAX = 100.0
ASPECT_MAX = 10.0

FAMILIES = [
    "dense_bulk",
    "layered_bulk",
    "explicit_vacuum_slab",
    "intrinsic_2d_like",
    "chain_solid",
    "isolated_chain",
    "molecular_crystal",
    "isolated_molecule",
    "stretched_layer",
    "high_coordination",
    "low_coordination",
]


# ---------------------------------------------------------------------------
# Reference computations (contract-literal; used to fill the labels and
# documented in labels.yaml ``definitions``)
# ---------------------------------------------------------------------------
def _cov(z: int) -> float:
    return float(covalent_radii[z])


def _pair_table(atoms: Atoms, lam: float) -> dict[tuple[int, int], float]:
    """All sorted (Zi,Zj) pairs -> lam*(r_cov_i + r_cov_j) (contracts §3.2)."""
    zs = [int(z) for z in atoms.numbers]
    pairs = sorted({(min(z1, z2), max(z1, z2)) for z1 in zs for z2 in zs})
    return {(p1, p2): lam * (_cov(p1) + _cov(p2)) for p1, p2 in pairs}


def _edges(atoms: Atoms, lam: float):
    """Bond edges at cutoff lam (i, j, d, S), self-loops excluded (§3.2)."""
    r_cut = _pair_table(atoms, lam)
    i, j, d, S = neighbor_list("ijdS", atoms, r_cut)
    keep = ~np.logical_and(i == j, np.all(S == 0, axis=1))
    return i[keep], j[keep], d[keep], S[keep]


def _integer_rank(vecs) -> int:
    """Rank of a set of integer vectors (dimension of their real span).

    Implementation: greedy integer basis reduction carried to a fixpoint
    (every basis vector is repeatedly reduced against all others), i.e. the
    contract's "贪心整数基约简" done correctly.  A naive single-pass greedy
    can overestimate the rank (e.g. [(2,0,0),(3,0,0)]); the fixpoint reduction
    below does not.  Result is cross-checked against
    ``np.linalg.matrix_rank`` (real rank == rational rank for integer
    vectors).
    """
    basis = []
    for v in vecs:
        v = np.asarray(v, dtype=int).reshape(-1).copy()
        if np.all(v == 0):
            continue
        # reduce v against current basis to a fixpoint
        changed = True
        while changed and np.any(v != 0):
            changed = False
            for b in basis:
                den = int(np.dot(b, b))
                if den == 0:
                    continue
                q = int(round(np.dot(b, v) / den))
                if q != 0:
                    v = v - q * np.asarray(b)
                    changed = True
                    if np.all(v == 0):
                        break
        if np.all(v == 0):
            continue
        basis.append(v)
        # re-reduce the whole basis to a fixpoint
        changed = True
        while changed:
            changed = False
            for a in range(len(basis)):
                for b in range(len(basis)):
                    if a == b:
                        continue
                    den = int(np.dot(basis[b], basis[b]))
                    if den == 0:
                        continue
                    q = int(round(np.dot(basis[b], basis[a]) / den))
                    if q != 0:
                        basis[a] = basis[a] - q * basis[b]
                        changed = True
            basis = [b for b in basis if np.any(b != 0)]
    if basis:
        mr = int(np.linalg.matrix_rank(np.array(basis, dtype=float)))
        if mr != len(basis):
            # defensive: real rank is authoritative for integer vectors
            return mr
    return len(basis)


def _ref_dim(atoms: Atoms, lam: float) -> int:
    """d(lambda) per contracts.md §3.3 (max component rank of loop shifts)."""
    i, j, d, S = _edges(atoms, lam)
    n = len(atoms)
    adj: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n)]
    for e in range(len(i)):
        si = S[e].astype(int)
        adj[i[e]].append((j[e], si))
        adj[j[e]].append((i[e], -si))
    visited = np.zeros(n, dtype=bool)
    t = np.zeros((n, 3), dtype=int)
    best = 0
    for root in range(n):
        if visited[root]:
            continue
        visited[root] = True
        stack = [root]
        loops: list[np.ndarray] = []
        while stack:
            u = stack.pop()
            for v, suv in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    t[v] = t[u] + suv
                    stack.append(v)
                else:
                    loops.append(t[u] + suv - t[v])
        best = max(best, _integer_rank(loops))
    return best


def _ref_dim_spectrum(atoms: Atoms) -> dict[str, int]:
    """Six-lambda d spectrum under the default covalent table."""
    return {f"{lam:.2f}": _ref_dim(atoms, lam) for lam in LAMBDAS}


def _shell_cn(atoms: Atoms) -> np.ndarray:
    """Documented fallback CN: neighbours with d < 1.25*(r_cov_i + r_cov_j)."""
    r_cut = {(p1, p2): 1.25 * (_cov(p1) + _cov(p2))
             for (p1, p2) in _pair_table(atoms, 1.0)}
    i, j, d = neighbor_list("ijd", atoms, r_cut)
    keep = ~(i == j)
    i, j = i[keep], j[keep]
    cn = np.zeros(len(atoms), dtype=int)
    for k in i:
        cn[k] += 1
    return cn


def _crystalnn_cn(atoms: Atoms):
    """CrystalNN CN per site; None on failure (vacuum boxes can break it)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pymatgen.analysis.local_env import CrystalNN
            from pymatgen.io.ase import AseAtomsAdaptor
            struct = AseAtomsAdaptor.get_structure(atoms)
            cnn = CrystalNN()
            cns = [int(round(cnn.get_cn(struct, k)))
                   for k in range(len(struct))]
        return np.array(cns, dtype=int)
    except Exception:
        return None


def _cn_reference(atoms: Atoms, hand_hist: dict[int, int]):
    """Physical CN reference + the method that produced it.

    Decision rule (documented): trust CrystalNN when it succeeds AND matches
    the hand-set histogram; otherwise fall back to the 1.25x covalent shell
    rule when that matches the hand histogram; otherwise keep the CrystalNN
    value and flag it as unverified.
    """
    hist = dict(hand_hist)
    cns = _crystalnn_cn(atoms)
    if cns is not None:
        got = {int(k): int(v) for k, v in zip(*np.unique(cns, return_counts=True))}
        if got == hist:
            return cns, "CrystalNN", None
        shell = _shell_cn(atoms)
        got_shell = {int(k): int(v) for k, v in zip(*np.unique(shell, return_counts=True))}
        if got_shell == hist:
            return shell, "shell_1.25_fallback", (
                f"CrystalNN gave {got} which disagrees with the hand-set "
                f"physical histogram {hist}; documented 1.25x-covalent shell "
                "rule used instead (CrystalNN is unreliable in vacuum boxes)."
            )
        return cns, "CrystalNN_unverified", (
            f"CrystalNN gave {got}; hand-set physical histogram is {hist}; "
            "value kept but flagged as unverified."
        )
    shell = _shell_cn(atoms)
    got_shell = {int(k): int(v) for k, v in zip(*np.unique(shell, return_counts=True))}
    if got_shell == hist:
        return shell, "shell_1.25_fallback", (
            "CrystalNN failed on this cell; documented 1.25x-covalent shell "
            "rule used instead."
        )
    return shell, "shell_1.25_unverified", (
        f"CrystalNN failed and the shell rule gave {got_shell} != hand-set "
        f"{hist}; value kept but flagged as unverified."
    )


def _vacuum_gap_estimate(atoms: Atoms) -> float:
    """Pilot-simple vacuum gap (documented in module docstring).

    Contract-literal (§3.5: circular interval of the projected coordinate):
    for each lattice-axis projection the largest circular gap between
    consecutive atom images is taken, minus the covalent radii of the two
    flanking atoms; max over axes, clamped at 0.  NOTE: for a symmetric slab
    (vacuum on both sides) the circular interval merges the two vacuum
    regions across the periodic boundary, so the estimate reports the TOTAL
    vacuum (2x the per-side value).  This mirrors the literal contract
    formula; per-side values are quoted in the per-structure notes.
    """
    s = atoms.get_scaled_positions(wrap=False) % 1.0
    lens = np.linalg.norm(atoms.cell[:], axis=1)
    best = 0.0
    for axis in range(3):
        h = np.sort(s[:, axis])
        gaps = np.diff(np.concatenate([h, [h[0] + 1.0]]))
        order = np.argsort(s[:, axis])
        gap = gaps.max()
        idx = int(np.argmax(gaps))
        if len(atoms) > 1:
            ia = order[idx]
            ib = order[(idx + 1) % len(atoms)]
            r = _cov(int(atoms.numbers[ia])) + _cov(int(atoms.numbers[ib]))
        else:
            r = 2.0 * _cov(int(atoms.numbers[0]))
        best = max(best, gap * lens[axis] - r)
    return max(0.0, float(best))


def _validity_reference(atoms: Atoms) -> dict:
    """L0 quantities per contracts.md §3.4/§5 (pilot defaults)."""
    zs = atoms.numbers
    vol = float(atoms.get_volume())
    sum_sphere = sum((4.0 / 3.0) * math.pi * _cov(int(z)) ** 3 for z in zs)
    volume_norm = vol / sum_sphere if sum_sphere > 0 else float("nan")
    # q_min over the lambda=1.3 nearest-neighbour set (§3.4); fallback 1.5
    q_min = None
    n_overlap = 0
    for lam in (1.3, 1.5):
        i, j, d, S = _edges(atoms, lam)
        if len(d):
            q = np.fromiter(
                (d[e] / (_cov(int(zs[i[e]])) + _cov(int(zs[j[e]])))
                 for e in range(len(d))), dtype=float)
            q_min = float(q.min())
            n_overlap = int(np.sum(q < Q_C))
            break
    cell = atoms.cell[:].astype(float)
    kappa = float(np.linalg.cond(cell)) if np.linalg.matrix_rank(cell) == 3 else float("inf")
    lens = np.linalg.norm(cell, axis=1)
    aspect = float(lens.max() / lens.min()) if lens.min() > 0 else float("inf")
    return {
        "q_min": None if q_min is None else round(q_min, 4),
        "n_overlap_pairs": n_overlap,
        "volume_norm": round(volume_norm, 4),
        "cell_kappa": round(kappa, 3),
        "aspect_ratio": round(aspect, 3),
        "overlap_flag": bool(q_min is not None and q_min < Q_C),
        "extreme_volume_flag": bool(
            not (VOL_NORM_RANGE[0] <= volume_norm <= VOL_NORM_RANGE[1])
        ),
        "pathological_cell_flag": bool(kappa > KAPPA_MAX or aspect > ASPECT_MAX),
    }


def _hist(cn: np.ndarray) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in zip(*np.unique(cn, return_counts=True))}


# ---------------------------------------------------------------------------
# Builders (hand-set ground truth + construction notes)
# ---------------------------------------------------------------------------
def _mk(family: str, n: int, atoms: Atoms, d_star: int, morph: str,
        hand_hist: dict[int, int], sites: list[dict], note: str) -> dict:
    atoms = atoms.copy()
    atoms.pbc = True
    atoms.wrap()
    return {
        "_id": f"control_{family}_{n:02d}",
        "family": family,
        "atoms": atoms,
        "expected_d_star": int(d_star),
        "expected_morphology_class": morph,
        "hand_cn_hist": {int(k): int(v) for k, v in hand_hist.items()},
        "expected_site_geometries": [dict(s) for s in sites],
        "note": note,
    }


def _site(symbol: str, cn: int, geometry: str) -> dict:
    return {"site": symbol, "cn": int(cn), "geometry": geometry}


def _build_dense_bulk() -> list[dict]:
    out = []
    si = bulk("Si", "diamond", a=5.431, cubic=True)
    out.append(_mk("dense_bulk", 1, si, 3, "dense_bulk", {4: 8},
                   [_site("Si", 4, "tetrahedral")],
                   "Diamond Si (a=5.431 A). Physical 3D covalent network. "
                   "Cordero Si-Si cutoff 2.22 A < 2.35 A bond, so the default "
                   "table under-bonds at lambda<=1.0 (see spectrum)."))
    al = bulk("Al", "fcc", a=4.05, cubic=True)
    out.append(_mk("dense_bulk", 2, al, 3, "dense_bulk", {12: 4},
                   [_site("Al", 12, "cuboctahedral")],
                   "fcc Al (a=4.05 A), CN=12 cuboctahedral (v1.3-L4 label). "
                   "Metallic: default covalent table bonds only at lambda>=1.2."))
    fe = bulk("Fe", "bcc", a=2.866, cubic=True)
    out.append(_mk("dense_bulk", 3, fe, 3, "dense_bulk", {8: 2},
                   [_site("Fe", 8, "cubic")],
                   "bcc Fe (a=2.866 A); first shell (8, cubic) bonds under "
                   "the default covalent table already at lambda=1.0."))
    nacl = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    out.append(_mk("dense_bulk", 4, nacl, 3, "dense_bulk", {6: 8},
                   [_site("Na", 6, "octahedral"), _site("Cl", 6, "octahedral")],
                   "Rocksalt NaCl (a=5.64 A). Ionic: Na-Cl 2.82 A vs Cordero "
                   "sum 2.68 A -> default table bonds only at lambda>=1.1."))
    ru = Atoms("Ti2O4",
               scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5),
                                 (0.3048, 0.3048, 0), (-0.3048, -0.3048, 0),
                                 (0.8048, 0.1952, 0.5), (0.1952, 0.8048, 0.5)],
               cell=[4.594, 4.594, 2.959, 90, 90, 90])
    out.append(_mk("dense_bulk", 5, ru, 3, "dense_bulk", {6: 2, 3: 4},
                   [_site("Ti", 6, "octahedral"), _site("O", 3, "trigonal_planar")],
                   "Rutile TiO2 (a=4.594, c=2.959, u=0.3048): Ti-O 1.95/1.98 A. "
                   "Bonds at lambda=1.0 under the default table."))
    caf2 = bulk("CaF2", "fluorite", a=5.463, cubic=True)
    out.append(_mk("dense_bulk", 6, caf2, 3, "dense_bulk", {8: 4, 4: 8},
                   [_site("Ca", 8, "cubic"), _site("F", 4, "tetrahedral")],
                   "Fluorite CaF2 (a=5.463 A). Ionic: Ca-F 2.37 A vs Cordero "
                   "sum 2.33 A -> default table bonds only at lambda>=1.1."))
    return out


def _build_layered_bulk() -> list[dict]:
    out = []
    a, d_lay = 2.46, 3.35
    for nl, tag in ((2, "2L"), (4, "4L")):
        c = nl * 2 * d_lay
        z = [0.25 + k * 0.5 / nl for k in range(nl)] if nl == 2 else \
            [0.125 + k * 0.25 for k in range(nl)]
        pos, sym = [], []
        for k in range(nl):
            if k % 2 == 0:
                pos += [(0, 0, z[k]), (1 / 3, 2 / 3, z[k])]
            else:
                pos += [(2 / 3, 1 / 3, z[k]), (0, 0, z[k])]
            sym += ["C", "C"]
        g = Atoms(sym, scaled_positions=pos, cell=[a, a, c, 90, 90, 120])
        out.append(_mk("layered_bulk", len(out) + 1, g, 2, "layered_bulk",
                       {3: 2 * nl},
                       [_site("C", 3, "trigonal_planar")],
                       f"Graphite AB {tag} (a=2.46, c={c:.2f}): in-plane "
                       "C-C 1.42 A; interlayer 3.35 A never bonds at "
                       "lambda<=1.5 under the default table (ratio 2.2)."))
    a, c, dz = 3.16, 12.30, 1.586  # dz = vertical Mo-S offset in A
    for nl, tag in ((2, "2L"), (4, "4L")):
        c_full = (nl / 2) * c
        dz_frac = dz / c_full
        zc = [0.125 + 0.5 * k / (nl / 2) for k in range(nl)] if nl == 2 else \
             [0.125 + 0.25 * k for k in range(nl)]
        pos, sym = [], []
        for k in range(nl):
            if k % 2 == 0:
                pos += [(0, 0, zc[k]), (1 / 3, 2 / 3, zc[k] + dz_frac),
                        (1 / 3, 2 / 3, zc[k] - dz_frac)]
                sym += ["Mo", "S", "S"]
            else:
                pos += [(2 / 3, 1 / 3, zc[k]), (0, 0, zc[k] + dz_frac),
                        (0, 0, zc[k] - dz_frac)]
                sym += ["Mo", "S", "S"]
        m = Atoms(sym, scaled_positions=pos, cell=[a, a, c_full, 90, 90, 120])
        out.append(_mk("layered_bulk", len(out) + 1, m, 2, "layered_bulk",
                       {6: 1 * nl, 3: 2 * nl},
                       [_site("Mo", 6, "trigonal_prismatic"),
                        _site("S", 3, "other")],
                       f"MoS2 2H-type {tag} (a=3.16, c={c_full:.2f}): Mo-S "
                       "2.42 A; interlayer S-S ~3.5 A (vdW, no bond at "
                       "lambda<=1.5). S is trigonal-pyramidal (CN=3) -> "
                       "'other' (not in frozen GEOMETRY_LABELS)."))
    return out


def _build_explicit_vacuum_slab() -> list[dict]:
    out = []
    for n, (layers, vac) in enumerate(
            ((4, 12.0), (5, 12.0), (6, 15.0), (7, 15.0), (8, 18.0)), start=1):
        s = surface("Si", (1, 1, 1), layers=layers, vacuum=vac)
        s.pbc = True
        nat = len(s)
        out.append(_mk("explicit_vacuum_slab", n, s, 2, "explicit_vacuum_slab",
                       {1: 8, 4: nat - 8},
                       [_site("Si", 4, "tetrahedral"), _site("Si", 1, "other")],
                       f"Si(111) ideal 1x1 slab, ASE layers={layers} bilayers "
                       f"({nat} atoms), per-side vacuum {vac:.0f} A (the "
                       "expected_vacuum_gap below reports the merged total "
                       "2x{vac:.0f} A per the circular-interval formula). "
                       "Surface planes CN=1 (dangling bond), interior CN=4; "
                       "no CN=3 plane in this cut. d(1.0)=0 because Cordero "
                       "Si under-bonds (Si-Si 2.35 A vs 2.22 A); in-plane "
                       "network appears at lambda>=1.1 -> d=2 at all higher "
                       "lambda."))
    return out


def _build_intrinsic_2d_like() -> list[dict]:
    out = []
    for n, c in enumerate((15.0, 20.0), start=1):
        gr = Atoms("C2", scaled_positions=[(0, 0, 0.5), (1 / 3, 2 / 3, 0.5)],
                   cell=[2.46, 2.46, c, 90, 90, 120])
        out.append(_mk("intrinsic_2d_like", n, gr, 2, "intrinsic_2d_like",
                       {3: 2}, [_site("C", 3, "trigonal_planar")],
                       f"Graphene monolayer, vacuum {c:.0f} A. True 2D "
                       "material: no surface CN gradient, no bulk core."))
    hbn = Atoms("BN", scaled_positions=[(0, 0, 0.5), (1 / 3, 2 / 3, 0.5)],
                cell=[2.504, 2.504, 15.0, 90, 90, 120])
    out.append(_mk("intrinsic_2d_like", 3, hbn, 2, "intrinsic_2d_like",
                   {3: 2}, [_site("B", 3, "trigonal_planar"),
                            _site("N", 3, "trigonal_planar")],
                   "hBN monolayer, vacuum 15 A (B-N 1.446 A)."))
    mos = Atoms("MoS2", scaled_positions=[(0, 0, 0.5), (1 / 3, 2 / 3, 0.5 + 1.586 / 15.0),
                                          (1 / 3, 2 / 3, 0.5 - 1.586 / 15.0)],
                cell=[3.16, 3.16, 15.0, 90, 90, 120])
    out.append(_mk("intrinsic_2d_like", 4, mos, 2, "intrinsic_2d_like",
                   {6: 1, 3: 2}, [_site("Mo", 6, "trigonal_prismatic"),
                                  _site("S", 3, "other")],
                   "MoS2 monolayer, vacuum 15 A."))
    return out


def _build_chain_solid() -> list[dict]:
    out = []
    specs = [
        # (a, atoms per chain, id, phases staggered?, note)
        (5.958, 3, 1, True, "Se trigonal-type chain solid: a=5.958 A -> "
                            "interchain ~3.5 A (vdW contact, bonds at "
                            "lambda=1.5 -> d:1->3); intrachain Se-Se 2.373 A."),
        (5.958, 6, 2, True, "Se chain solid, 6 atoms per chain (c doubled)."),
        (5.543, 3, 3, False, "Se chain solid with closer, phase-aligned "
                             "interchain contact 3.20 A (bonds at "
                             "lambda=1.35)."),
    ]
    for a, nrep, n, staggered, note in specs:
        c = nrep * 2.373
        pos = []
        for i, (fx, fy) in enumerate(
                [(0.0, 0.0), (1 / 3, 2 / 3), (2 / 3, 1 / 3)]):
            phase = i / 3 if (staggered and nrep == 3) else \
                    i / 6 if staggered else 0.0
            for k in range(nrep):
                pos.append((fx, fy, (k + phase) / nrep))
        se = Atoms(f"Se{3 * nrep}", scaled_positions=pos,
                   cell=[a, a, c, 90, 90, 120])
        out.append(_mk("chain_solid", n, se, 1, "chain_solid",
                       {2: 3 * nrep}, [_site("Se", 2, "other")],
                       note + " Se CN=2, bond angle ~103 deg -> 'other'."))
    return out


def _build_isolated_chain() -> list[dict]:
    out = []
    se_f = Atoms("Se8", positions=[(7.5, 7.5, 2.5 + k * 2.373) for k in range(8)],
                 cell=[15, 15, 25])
    out.append(_mk("isolated_chain", 1, se_f, 0, "isolated_chain",
                   {2: 6, 1: 2},
                   [_site("Se", 2, "other"), _site("Se", 1, "other")],
                   "FINITE Se chain (8 atoms) in a 15x15x25 A box. Contract "
                   "defines finite clusters as d=0; the 1D character is "
                   "carried by the morphology class, not by d(lambda). "
                   "End-to-image distance is 8.9 A so the chain never bonds "
                   "to its periodic image at lambda<=1.5."))
    c_wrap = 8 * 2.373
    se_w = Atoms("Se8", positions=[(7.5, 7.5, k * 2.373) for k in range(8)],
                 cell=[15, 15, c_wrap])
    out.append(_mk("isolated_chain", 2, se_w, 1, "isolated_chain",
                   {2: 8}, [_site("Se", 2, "other")],
                   "PERIODICALLY WRAPPED Se chain (8 atoms span the cell): "
                   "closes on itself through the periodic boundary -> d=1. "
                   "Contrast with control_isolated_chain_01 (finite, d=0)."))
    c_f = Atoms("C8", positions=[(7.5, 7.5, 1.0 + k * 1.30) for k in range(8)],
                cell=[15, 15, 15])
    out.append(_mk("isolated_chain", 3, c_f, 0, "isolated_chain",
                   {2: 6, 1: 2},
                   [_site("C", 2, "linear"), _site("C", 1, "other")],
                   "FINITE C cumulene chain (8 atoms, d_CC=1.30 A) in a "
                   "15 A box. Bonds at lambda=1.0 (ratio 0.855). d=0 "
                   "(finite cluster)."))
    c_w = Atoms("C12", positions=[(9.0, 9.0, k * 1.30) for k in range(12)],
                cell=[18, 18, 12 * 1.30])
    out.append(_mk("isolated_chain", 4, c_w, 1, "isolated_chain",
                   {2: 12}, [_site("C", 2, "linear")],
                   "PERIODICALLY WRAPPED C cumulene chain (12 atoms) -> d=1 "
                   "at every lambda."))
    return out


def _build_molecular_crystal() -> list[dict]:
    out = []
    # H2 fcc, 4 molecules
    a = 5.3
    sites = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    pos = []
    for s in sites:
        pos.append([x * a for x in (s[0] + 0.37 / a, s[1], s[2])])
        pos.append([x * a for x in (s[0] - 0.37 / a, s[1], s[2])])
    h2 = Atoms("H8", positions=pos, cell=[a] * 3)
    out.append(_mk("molecular_crystal", 1, h2, 0, "molecular_crystal",
                   {1: 8}, [_site("H", 1, "other")],
                   "H2 fcc crystal, 4 molecules (a=5.3 A). Intramolecular "
                   "H-H 0.74 A bonds at lambda>=1.2; intermolecular >=3.7 A "
                   "never bonds -> d=0 at all lambda. Large volume_norm "
                   "expected (vdW crystal)."))
    # N2 fcc
    a = 5.66
    pos = []
    for s in sites:
        pos.append([x * a for x in (s[0] + 0.55 / a, s[1], s[2])])
        pos.append([x * a for x in (s[0] - 0.55 / a, s[1], s[2])])
    n2 = Atoms("N8", positions=pos, cell=[a] * 3)
    out.append(_mk("molecular_crystal", 2, n2, 0, "molecular_crystal",
                   {1: 8}, [_site("N", 1, "other")],
                   "N2 fcc crystal (a=5.66 A). N-N 1.10 A bonds at lambda=0.9."))
    # H2O fcc, 4 molecules, fixed orientation
    a = 5.6
    hx, hy = 0.759, 0.588  # O-H 0.96 A, H-O-H 104.5 deg
    pos = []
    for s in sites:
        pos.append([x * a for x in s])
        pos.append([x * a for x in (s[0] + hx / a, s[1] + hy / a, s[2])])
        pos.append([x * a for x in (s[0] - hx / a, s[1] + hy / a, s[2])])
    h2o = Atoms("H8O4", positions=pos, cell=[a] * 3)
    out.append(_mk("molecular_crystal", 3, h2o, 0, "molecular_crystal",
                   {2: 4, 1: 8}, [_site("O", 2, "other"), _site("H", 1, "other")],
                   "Idealized H2O fcc crystal, 4 molecules (a=5.6 A), fixed "
                   "orientation. O is bent CN=2 -> 'other'. Intermolecular "
                   "H...O >=3.0 A never bonds at lambda<=1.5."))
    # H2O simple cubic, 1 molecule
    a = 6.0
    h2o_sc = Atoms("H2O", positions=[(3.0, 3.0, 3.0),
                                     (3.0 + hx, 3.0 + hy, 3.0),
                                     (3.0 - hx, 3.0 + hy, 3.0)],
                   cell=[a] * 3)
    out.append(_mk("molecular_crystal", 4, h2o_sc, 0, "molecular_crystal",
                   {2: 1, 1: 2}, [_site("O", 2, "other"), _site("H", 1, "other")],
                   "Idealized H2O simple-cubic crystal, 1 molecule per cell "
                   "(a=6.0 A)."))
    return out


def _build_isolated_molecule() -> list[dict]:
    out = []
    h2 = Atoms("H2", positions=[(7.5, 7.5, 7.5), (8.24, 7.5, 7.5)],
               cell=[15] * 3)
    out.append(_mk("isolated_molecule", 1, h2, 0, "isolated_molecule",
                   {1: 2}, [_site("H", 1, "other")],
                   "Single H2 in a 15 A box. H-H 0.74 A bonds only at "
                   "lambda>=1.2 (Cordero 0.62 A); d=0 either way (tree)."))
    h2o = Atoms("H2O", positions=[(7.5, 7.5, 7.5),
                                  (7.5 + 0.759, 7.5 + 0.588, 7.5),
                                  (7.5 - 0.759, 7.5 + 0.588, 7.5)],
                cell=[15] * 3)
    out.append(_mk("isolated_molecule", 2, h2o, 0, "isolated_molecule",
                   {2: 1, 1: 2}, [_site("O", 2, "other"), _site("H", 1, "other")],
                   "Single H2O in a 15 A box. O bent (104.5 deg) -> 'other'."))
    co2 = Atoms("CO2", positions=[(7.5, 7.5, 6.34), (7.5, 7.5, 7.5),
                                  (7.5, 7.5, 8.66)],
                cell=[15] * 3)
    out.append(_mk("isolated_molecule", 3, co2, 0, "isolated_molecule",
                   {2: 1, 1: 2}, [_site("C", 2, "linear"), _site("O", 1, "other")],
                   "Single CO2 in a 15 A box. C-O 1.16 A: q_min=0.817<0.85 "
                   "-> the L0 overlap_flag FIRES on a perfectly valid "
                   "double-bonded molecule (expected, documented)."))
    c2h2 = Atoms("C2H2", positions=[(7.5, 7.5, 5.835), (7.5, 7.5, 7.045),
                                    (7.5, 7.5, 8.255), (7.5, 7.5, 9.465)],
                 cell=[15] * 3)
    out.append(_mk("isolated_molecule", 4, c2h2, 0, "isolated_molecule",
                   {2: 2, 1: 2}, [_site("C", 2, "linear"), _site("H", 1, "other")],
                   "Single acetylene in a 15 A box (C-C 1.21, C-H 1.06 A)."))
    ch4 = Atoms("CH4", positions=[(7.5, 7.5, 7.5),
                                  (8.129, 8.129, 8.129), (6.871, 6.871, 8.129),
                                  (6.871, 8.129, 6.871), (8.129, 6.871, 6.871)],
                cell=[15] * 3)
    out.append(_mk("isolated_molecule", 5, ch4, 0, "isolated_molecule",
                   {4: 1, 1: 4}, [_site("C", 4, "tetrahedral"), _site("H", 1, "other")],
                   "Single CH4 in a 15 A box. C-H 1.09 A vs Cordero 1.07 A: "
                   "under-bonds at lambda=1.0, molecule forms at lambda>=1.1."))
    return out


def _build_stretched_layer() -> list[dict]:
    out = []
    a = 2.46
    specs = [(1.5, 10.05, 1), (2.0, 13.40, 2)]
    for f, c, n in specs:
        g = Atoms("C4", scaled_positions=[(0, 0, 0.25), (1 / 3, 2 / 3, 0.25),
                                          (2 / 3, 1 / 3, 0.75), (0, 0, 0.75)],
                  cell=[a, a, c, 90, 90, 120])
        gap = c / 2 - 1.52
        out.append(_mk("stretched_layer", n, g, 2, "layered_bulk",
                       {3: 4}, [_site("C", 3, "trigonal_planar")],
                       f"Graphite AB 2L with c stretched {f:.1f}x "
                       f"(interlayer {c/2:.2f} A). Expected layered_bulk: no "
                       f"explicit vacuum (gap {gap:.2f} A after radii << slab "
                       "vacuum >=12 A), no surface CN gradient, layers repeat "
                       "in a periodic cell. Discriminates layered bulk from "
                       "explicit-vacuum slab."))
    c = 18.45
    dz = 1.586 / c
    m = Atoms("Mo2S4", scaled_positions=[(0, 0, 0.25), (2 / 3, 1 / 3, 0.75),
                                         (1 / 3, 2 / 3, 0.25 + dz),
                                         (1 / 3, 2 / 3, 0.25 - dz),
                                         (0, 0, 0.75 + dz), (0, 0, 0.75 - dz)],
              cell=[3.16, 3.16, c, 90, 90, 120])
    out.append(_mk("stretched_layer", 3, m, 2, "layered_bulk",
                   {6: 2, 3: 4},
                   [_site("Mo", 6, "trigonal_prismatic"), _site("S", 3, "other")],
                   "MoS2 2H-type 2L with c stretched 1.5x (18.45 A). "
                   "Interlayer S-S gap grows to ~6.1 A; still layered_bulk "
                   "(no true vacuum, repeating layers)."))
    return out


def _build_high_coordination() -> list[dict]:
    out = []
    for n, (sym, a) in enumerate((("Cu", 3.615), ("Pd", 3.89), ("Ag", 4.085)), start=1):
        m = bulk(sym, "fcc", a=a, cubic=True)
        out.append(_mk("high_coordination", n, m, 3, "dense_bulk", {12: 4},
                       [_site(sym, 12, "cuboctahedral")],
                       f"fcc {sym} (a={a} A), CN=12 cuboctahedral (v1.3-L4 label). Metallic "
                       "dense bulk used as the high-CN control."))
    return out


def _build_low_coordination() -> list[dict]:
    out = []
    for n, f, a in ((1, 1.25, 5.431 * 1.25), (2, 1.40, 5.431 * 1.40)):
        si = bulk("Si", "diamond", a=a, cubic=True)
        out.append(_mk("low_coordination", n, si, 3, "dense_bulk",
                       {4: 8} if n == 1 else {0: 8},
                       [_site("Si", 4, "tetrahedral")] if n == 1
                       else [_site("Si", 0, "other")],
                       f"Diamond Si lattice expanded {f:.2f}x (Si-Si "
                       f"{2.351*f:.2f} A): sparse, bond-diluted covalent net. "
                       f"Default table bonds only at lambda>=1.35"
                       + ("" if n == 1 else " (or >=1.5)")
                       + f"; CrystalNN CN={'4' if n == 1 else '0'}. Tests the "
                       "low-CN / sparse regime."))
    d = 2.35 / math.sqrt(3.0)
    si4 = Atoms("Si4", positions=[(1.0, 1.0, 1.0), (1.0 + 2 * d, 1.0, 1.0),
                                  (1.0 + d, 1.0 + d, 1.0), (1.0 + d, 1.0, 1.0 + d)],
                cell=[15] * 3)
    out.append(_mk("low_coordination", 3, si4, 0, "isolated_molecule",
                   {3: 4}, [_site("Si", 3, "other")],
                   "Si4 tetrahedral cluster in a 15 A box: genuinely low CN "
                   "(3 vs bulk 4), 0D finite cluster. Morphologically an "
                   "isolated cluster (mapped to isolated_molecule)."))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_all() -> list[tuple[str, Atoms, dict]]:
    """Deterministically build all control structures with full labels.

    Returns list[(structure_id, atoms, expected)] where ``expected`` mirrors
    the per-structure entry written to ``controls/data/labels.yaml``
    (identical content; floats rounded identically).  Idempotent and
    reproducible: no RNG, no external data.
    """
    records: list[dict] = []
    for builder in (
            _build_dense_bulk, _build_layered_bulk, _build_explicit_vacuum_slab,
            _build_intrinsic_2d_like, _build_chain_solid, _build_isolated_chain,
            _build_molecular_crystal, _build_isolated_molecule,
            _build_stretched_layer, _build_high_coordination,
            _build_low_coordination):
        records.extend(builder())

    out = []
    for rec in records:
        atoms: Atoms = rec["atoms"]
        spectrum = _ref_dim_spectrum(atoms)
        cn, cn_method, cn_warn = _cn_reference(atoms, rec["hand_cn_hist"])
        validity = _validity_reference(atoms)
        vacuum = _vacuum_gap_estimate(atoms)
        notes = [rec["note"]]
        if cn_warn:
            notes.append("CN note: " + cn_warn)
        if spectrum["1.00"] != rec["expected_d_star"]:
            notes.append(
                "fast_default: d(lambda=1.00)="
                f"{spectrum['1.00']} != physical expected_d_star="
                f"{rec['expected_d_star']} (Cordero covalent radii under-bond; "
                "Phase-0 calibrated pair table expected to fix)."
            )
        if validity["extreme_volume_flag"]:
            notes.append("extreme_volume_flag expected (boxed / vdW species).")
        if validity["overlap_flag"]:
            notes.append("overlap_flag expected (short multiple bond).")
        expected = {
            "family": rec["family"],
            "formula": atoms.get_chemical_formula(),
            "elements": sorted(set(atoms.get_chemical_symbols())),
            "natom": len(atoms),
            "expected_d_star": rec["expected_d_star"],
            "expected_morphology_class": rec["expected_morphology_class"],
            "expected_short_lambda_dim": dict(spectrum),
            "expected_mean_cn": round(float(cn.mean()), 4),
            "expected_cn_min": int(cn.min()),
            "expected_cn_max": int(cn.max()),
            "expected_cn_hist": _hist(cn),
            "cn_method": cn_method,
            "expected_site_geometries": rec["expected_site_geometries"],
            "expected_validity": dict(validity),
            "expected_vacuum_gap": round(vacuum, 3),
            "notes": " ".join(notes),
        }
        out.append((rec["_id"], atoms, expected))
    return out


def _labels_definitions() -> dict:
    return {
        "version": 1,
        "n_structures_hint": None,
        "families": list(FAMILIES),
        "morphology_classes_source": "contracts.md §3.5 (frozen mirror in "
                                     "src/ckt/controls.py)",
        "definitions": {
            "expected_d_star":
                "Physical ground-truth periodic bonding-network "
                "dimensionality of the material (0/1/2/3), independent of any "
                "cutoff choice. Finite clusters (incl. finite chains and "
                "molecules in boxes) are 0 per contracts.md §3.3.",
            "expected_short_lambda_dim":
                "d(lambda) produced by the frozen default-covalent-table "
                "algorithm (contracts.md §3.2/§3.3: edge iff "
                "d_ij < lambda*(r_cov_i+r_cov_j) with ASE Cordero covalent "
                "radii; self-loops excluded; d = rank of loop-translation "
                "vectors per connected component, max over components; "
                "rank computed by fixpoint greedy integer basis reduction, "
                "verified == real rank). Keys are lambda strings. "
                "Where ['1.00'] != expected_d_star the structure is a "
                "documented Cordero under-bonding case.",
            "expected_mean_cn":
                "Physical CN reference: pymatgen CrystalNN when it succeeds "
                "and matches the hand-set histogram; else documented "
                "1.25x-covalent shell rule (d < 1.25*(r_i+r_j)); cn_method "
                "records which.",
            "expected_validity":
                "L0 quantities per contracts.md §3.4/§5: q_min over the "
                "lambda=1.3 bond set (fallback 1.5), volume_norm "
                "V/sum(4pi/3 r_cov^3), cell_kappa = cond(cell), aspect = "
                "max/min cell length; flags with pilot defaults "
                "(q_c=0.85, nu in [0.5,10], kappa<=100, aspect<=10).",
            "expected_vacuum_gap":
                "Pilot-simple vacuum estimate per contracts.md §3.5 wording: "
                "for each lattice-axis projection, largest CIRCULAR gap in "
                "fractional coordinate minus covalent radii of the flanking "
                "atoms; max over axes, clamped at 0. For a symmetric slab "
                "(vacuum on both sides) the circular interval merges the two "
                "vacuum regions across the periodic boundary, so the value "
                "reports the TOTAL vacuum (2x per-side; per-side values are "
                "quoted in the structure notes). Reference magnitude for "
                "tests/figures, not the level2 implementation.",
            "expected_site_geometries":
                "Hand-set coordination geometry for inequivalent sites using "
                "the frozen GEOMETRY_LABELS of contracts.md §3.7 ('other' = "
                "shape not in the list; see notes).",
            "notes":
                "Construction details + documented expected-flag / "
                "under-bonding behaviour.",
        },
    }


def labels_payload(records: list | None = None) -> dict:
    """Full labels.yaml payload (meta + definitions + structures)."""
    if records is None:
        records = build_all()
    structures = {}
    for sid, atoms, expected in records:
        structures[sid] = expected
    meta = _labels_definitions()
    meta["n_structures_hint"] = len(structures)
    return {"meta": meta, "structures": structures}


def load_labels() -> dict:
    """Load controls/data/labels.yaml (returns the full dict)."""
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    with open(LABELS_PATH, encoding="utf-8") as fh:
        return dict(yaml.load(fh))


def write_assets(force: bool = False) -> dict:
    """Write structures/*.vasp + labels.yaml. Returns a small summary dict."""
    import ase.io

    records = build_all()
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for sid, atoms, expected in records:
        path = STRUCTURES_DIR / f"{sid}.vasp"
        if path.exists() and not force:
            raise FileExistsError(f"{path} exists; pass force=True to overwrite")
        ase.io.write(path, atoms, format="vasp", direct=True)
        written.append(path.name)
    payload = labels_payload(records)
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 4096
    with open(LABELS_PATH, "w", encoding="utf-8") as fh:
        yaml.dump(payload, fh)
    return {"n_structures": len(written), "n_vasp": len(written),
            "structures_dir": str(STRUCTURES_DIR), "labels_path": str(LABELS_PATH)}


# ---------------------------------------------------------------------------
# Figures (mandatory >=3, PNG 300 dpi, axis labels + titles)
# ---------------------------------------------------------------------------
def make_figs(out_dir: Path | None = None) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) if out_dir is not None else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    records = build_all()
    fams = [e["family"] for _, _, e in records]
    natom = np.array([e["natom"] for _, _, e in records], dtype=float)
    vol_per_atom = np.array(
        [a.get_volume() / len(a) for _, a, _ in records], dtype=float)
    vac = np.array([e["expected_vacuum_gap"] for _, _, e in records])
    min_len = np.array(
        [np.linalg.norm(a.cell[:], axis=1).min() for _, a, _ in records])
    dstar = np.array([e["expected_d_star"] for _, _, e in records])

    colors = {f: plt.cm.tab20(i) for i, f in enumerate(FAMILIES)}
    markers = {0: "o", 1: "^", 2: "s", 3: "D"}

    # fig1: family counts
    counts = {f: fams.count(f) for f in FAMILIES}
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(list(counts), list(counts.values()),
                  color=[colors[f] for f in counts])
    ax.bar_label(bars)
    ax.set_title(f"Controls: structure counts per family (N={len(records)})")
    ax.set_xlabel("family")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    p1 = out_dir / "fig1_family_counts.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)

    # fig2: vacuum gap + min cell dimension by family
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    rng = np.random.default_rng(0)
    for ax, vals, ylab, ttl in (
            (ax1, vac, "vacuum-gap estimate (A)", "vacuum gap vs slab/bulk/molecule"),
            (ax2, min_len, "min cell dimension (A)", "min cell dimension")):
        for k, f in enumerate(FAMILIES):
            idx = [i for i, x in enumerate(fams) if x == f]
            if not idx:
                continue
            ax.scatter(np.full(len(idx), k) + rng.uniform(-0.25, 0.25, len(idx)),
                       vals[idx], s=36, color=colors[f], alpha=0.75,
                       label=f if ax is ax1 else None)
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(FAMILIES, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
    ax1.axhline(12.0, ls="--", color="grey", lw=1)
    ax1.text(0.02, 12.1, "slab per-side vacuum lower bound (12 A)", fontsize=8,
             color="grey")
    ax1.legend(fontsize=7, loc="upper left")
    fig.text(0.5, 0.01,
             "vacuum-gap estimate = contract-literal circular projection "
             "(symmetric slabs report merged top+bottom vacuum, 2x per-side)",
             ha="center", fontsize=8, color="grey")
    fig.suptitle(f"Controls: void / cell-size distribution by family (N={len(records)})")
    fig.tight_layout()
    p2 = out_dir / "fig2_vacuum_dist.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)

    # fig3: overview scatter
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, f in enumerate(FAMILIES):
        idx = [i for i, x in enumerate(fams) if x == f]
        if not idx:
            continue
        for ds in sorted(set(dstar[idx].tolist())):
            sel = [i for i in idx if dstar[i] == ds]
            ax.scatter(natom[sel], vol_per_atom[sel], s=48,
                       color=colors[f], marker=markers[ds])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("natom (log)")
    ax.set_ylabel("volume per atom (A^3, log)")
    ax.set_title("Controls overview: natom vs volume/atom (color=family, "
                 "marker=expected_d_star)")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=markers[d], color="grey", ls="",
                      label=f"d_star={d}") for d in (0, 1, 2, 3)] + \
              [Line2D([0], [0], marker="o", color=colors[f], ls="", label=f)
               for f in FAMILIES]
    ax.legend(handles=handles, fontsize=7, ncol=2, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    p3 = out_dir / "fig3_controls_overview.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    return [p1, p2, p3]


def _main() -> None:
    summary = write_assets(force=True)
    figs = make_figs()
    print(f"controls: wrote {summary['n_structures']} structures -> "
          f"{summary['structures_dir']}")
    print(f"controls: wrote labels -> {summary['labels_path']}")
    print("controls: figures -> " + ", ".join(str(p.name) for p in figs))


if __name__ == "__main__":
    _main()
