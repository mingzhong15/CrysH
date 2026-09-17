"""Level 2 — Morphology: vacuum detection, surface-likeness score, porous pre-screen,
and 0D/1D/2D/3D morphological classification.

Frozen contract: `contracts.md` §3.5 (``MorphologyResult`` / ``morphology`` /
``MORPHOLOGY_CLASSES``).  Owner: `level2-morphology`.

Inputs: `graph` / `spectrum` / `coord` are ALWAYS honored as-is when supplied (the
pipeline passes the crysh.dimensionality graph at D_STAR_LAMBDA=1.2 + the crysh.dimensionality
spectrum).  When they are `None`, level-1 is used if importable (``_PREFER_L1``);
the module's own deterministic fallback (note 1) is the last-resort stand-alone
path and never blocks on level-1.

------------------------------------------------------------------------------
Implementer's notes — pilot simplifications (all documented)
------------------------------------------------------------------------------

1. **Internal fallback bonding table** — `r0_ij = TOL * (r_cov_i + r_cov_j)` with
   `TOL = 1.15`, scaled by λ for the spectrum (ase natural cutoffs × λ plus a
   documented ionic/metallic tolerance; pure covalent sums underbond NaCl/Si-Si).
   Only used when level-1 is unavailable; callers pass `graph`/`spectrum`
   explicitly in the pipeline.

2. **Vacuum (v3: direction search over lattice axes + low-index plane normals)** —
   candidate directions are the 3 lattice axes PLUS the low-index plane normals
   G = h·b1 + k·b2 + l·b3 (b_i = reciprocal basis, A⁻¹ᵀ rows; h,k,l ∈ {−1,0,1},
   not all zero, ±-deduplicated → 13 directions in a deterministic order).
   * axis k (legacy metric, kept for the `per_axis_gaps` diagnostics and for
     backward compatibility): project the k-th FRACTIONAL coordinate onto the
     circular interval [0, 1); longest empty interval g_frac; raw gap (Å) =
     g_frac × |cell[k]| minus the covalent radii of the two bounding atoms.
     For non-orthogonal cells this scales the fractional gap by the lattice
     vector length — pilot approximation (overestimates the perpendicular gap).
   * plane normal (hkl) (physically correct metric): project z_i = r_i·(G/|G|)
     mod d with the interplanar spacing d = 1/|G|; the circular-interval /
     subtract-bounding-radii algorithm is isomorphic to the axis one, but the
     metric is the true perpendicular distance between (hkl) planes.
   The direction with the largest net gap wins (strict >, axes first, so
   orthogonal cells keep their axis result exactly).  `vacuum_fraction =
   gap / h` and `span = h − g_raw` use the period h of the winning direction
   (|a_k| for an axis, d_hkl for a plane normal) — the semantics stay
   "thickness / layer profile along the vacuum normal".  Diagnostics:
   `per_axis_gaps` (3 axis gaps), `vacuum_axis` (0/1/2 = axis; ≥3 = index into
   `_plane_normal_candidates` order), `vacuum_normal` (unit vector, JSON-safe
   list), `vacuum_miller` ([h,k,l] or None).  Non-orthogonal cells where the
   vacuum normal is not parallel to any reciprocal axis (tilted sheets whose
   in-plane lattice vectors are not cell vectors, e.g. spanning (a1, a2+a3))
   are now detected instead of missed.  `_USE_PLANE_NORMALS = False` restores
   the legacy 3-axis search (diagnostic/regression hook).

3. **surface_score (pilot, simplified per contract §3.5)** —
   `S = 0.30·s_vac + 0.60·s_dcn + 0.10·s_lay`, clipped to [0, 1]:
   * `s_vac = min(1, vacuum_fraction / 0.30)` (vacuum presence);
   * `s_dcn = min(1, ΔCN / 4.0)` with ΔCN = ⟨CN⟩_interior − ⟨CN⟩_outer where
     outer = atoms in the outermost 12% of the projected span along the vacuum
     normal, interior = the middle 76% (contract: "无层标定时用投影 z 上 CN 差").
     CN = `coord.cn` when supplied, else the bond-graph degree.  ΔCN = 0 when the
     material is monolayer-thin (span ≤ 1.6 × median bond) — no interior exists.
     Calibrated so that an ideal cleaved slab (ΔCN ≈ 1, e.g. Si(111) surface CN 3
     vs bulk 4) scores ≈ 0.55 < 0.6 → `explicit_vacuum_slab`; only stronger
     surface signatures (ΔCN ≥ 2) cross the 0.6 gate → `surface_like`.  (The
     earlier std(CN)-proxy conflated elemental CN diversity — MoS₂ std ≈ 4.2 — with
     surface gradients and was dropped.)
   * `s_lay`: 1.0 if d(λ=0.90) == 2 or d(λ=1.00) == 2 (short-λ 2D sheet), 0.5 if
     d(1.10) == 2, else 0.

4. **class_label decision order** (documented; routing on the effective
   dimensionality, see the Cordero under-bonding rescue):
   * **soft gates (v3, decision-equivalent reformulation of the v2 hard AND/OR
     gates)** — the three hard gates became bounded scores with the decision
     rule "score ≥ `_SCORE_GATE` (0.5)"; min-composite scoring reproduces the
     AND gate exactly because s(x; c) = x/(x+c) is strictly increasing with
     s(c) = 0.5:
     - `vacuum_score = min(s(gap; 5), s(frac; 0.25))` ≥ 0.5 ⟺ gap ≥ 5 Å AND
       frac ≥ 0.25 (the v2 `big_vacuum`);
     - `slab_score = s(gap; 12)` ≥ 0.5 ⟺ gap ≥ 12 Å;
     - `mono_score = 1/(1 + span/(1.6·med_bond))` ≥ 0.5 ⟺ span ≤ 1.6·med_bond.
     Equivalence is exact in real arithmetic (grid-tested in
     test_morphology_softgates.py); IEEE rounding can only differ within ~1 ulp
     of a gate boundary, which no physical input hits.  The margin
     `2·|score − 0.5|` ∈ [0, 1] ("distance from the decision boundary" in
     score units) is exported for every gate.  The monolayer vacuum
     requirement (v2 `_MONO_VAC_GAP` = 5 Å) is expressed through the same
     s(gap; 5) component as the vacuum gate (constants kept equal by
     construction — recalibrate together).  Gate constants 5/0.25/12/1.6 are
     pilot values pending Phase-0/1 calibration.
   * **Cordero under-bonding rescue** — the frozen default table (pure covalent
     sums) gives d(λ=1.0) = 0 for many bulk solids (Si, Al, Fe, NaCl, CaF₂, ...;
     the 44 controls document this in `expected_short_lambda_dim`).  If
     d_star == 0, route on the effective dimension instead:
     max(d(1.10), d(1.20), d(1.35)) == 3 → treat as d=3; elif d(1.20) == 2 →
     treat as d=2; elif d(1.20) == 1 → treat as d=1; else d=0.
   * eff d == 0: vacuum_gap ≥ 7.0 Å (`_VAC_ISOLATED_GAP`, not softened — a
     distinct bar from the three gates above) → isolated regime: largest connected
     component with ≥ 5 atoms and gyration-tensor aspect λ₁/λ₂ ≥ 3.0 →
     `isolated_chain`; otherwise `isolated_molecule` (component size / aspect
     gate keeps CO₂/acetylene/CH₄ as molecules).  Else (packed) →
     `molecular_crystal` (covers one-molecule-per-cell crystals and partially
     bonded molecules; controls: molecular_crystal_04, isolated_molecule_02/04).
   * eff d == 1: big_vacuum (vacuum_score gate) → `isolated_chain`; else
     `chain_solid`.
   * eff d == 2: max(d(λ)) ≥ 3 → `layered_bulk` (2→3 interlayer-bonding
     signature, total plan §Level2); else (all λ = 2): monolayer (mono_score
     gate) with s(gap; 5) ≥ 0.5 → `intrinsic_2d_like`; else slab_score gate →
     `surface_like` if surface_score > 0.6 else
     `explicit_vacuum_slab`; else → `layered_bulk` (multi-layer 2D materials:
     graphite/MoS₂ few-layers, stretched layers).
   * eff d == 3: big_vacuum (vacuum_score gate; slab of a 3D material) →
     `surface_like` if
     surface_score > 0.6 else `explicit_vacuum_slab`; else porous_candidate
     (eff-d 3D, ν > 4.0, no big vacuum AND the λ=1.2 graph is one connected
     network — a bond-diluted/disconnected sparse solid is not a framework) →
     `porous_candidate`; else `dense_bulk`.
   * anything else → `ambiguous`.

   Contract ambiguities resolved conservatively (recorded in progress.md):
   (a) "porous" of the task text is emitted as `porous_candidate` because the frozen
   `MORPHOLOGY_CLASSES` list contains only the latter; (b) `layered_bulk` is emitted
   here on the 2→3 spectrum signature and on multi-layer all-λ=2 stacks (the only
   way the frozen class is reachable; total plan §Level2 assigns it to morphology);
   routing still uses d_star per "这里按 d_star 判", with the Cordero rescue as the
   documented deviation forced by the frozen default table; (c) the earlier
   "≤ 2 atomic layers ⇒ intrinsic 2D" criterion was replaced by the span-based
   monolayer criterion (it mislabeled 2-layer graphite AB; controls).
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from ase.neighborlist import natural_cutoffs, neighbor_list

# level-1（键图/维数谱）与本模块同包，直接依赖（R2 去掉了"未合入就降级"的分支）
from crysh.bond import build_bond_graph as _l1_build_bond_graph
from crysh.config import D_STAR_LAMBDA as _L1_GRAPH_LAM
from crysh.config import LAMBDAS
from crysh.dimensionality import dimensionality_spectrum as _l1_dimension

MORPHOLOGY_CLASSES = [
    "dense_bulk",
    "layered_bulk",
    "explicit_vacuum_slab",
    "surface_like",
    "intrinsic_2d_like",
    "chain_solid",
    "isolated_chain",
    "molecular_crystal",
    "isolated_molecule",
    "porous_candidate",
    "ambiguous",
]


# ---- pilot thresholds (all documented above; calibratable later) ----
# Gate constants 5 / 0.25 / 12 / 1.6 are PILOT values pending Phase-0/1
# calibration (v3: consumed through the soft scores below, not raw comparisons).
_PREFER_L1 = True          # level-1 is merged and brute-force-verified (44/44 controls)
_FALLBACK_TOL = 1.15      # ionic/metallic tolerance on the fallback pair table
_VAC_GAP_MIN_A = 5.0      # Å, AND-combined with ... (score scale: s_gap = gap/(gap+5))
_VAC_FRAC_MIN = 0.25      # ... fractional vacuum (score scale: s_frac = frac/(frac+0.25))
_VAC_ISOLATED_GAP = 7.0   # Å bar for the 0D isolated-molecule/chain regime (NOT softened)
_SLAB_GAP_MIN_A = 12.0    # Å bar for the 2D multi-layer slab gate (slab_score = gap/(gap+12))
_SPAN_FRAC = 1.6          # monolayer: material span <= _SPAN_FRAC x median bond
                           #   (mono_score = 1/(1 + span/(1.6*med)))
_MONO_VAC_GAP = 5.0       # Å vacuum required for a monolayer to be intrinsic_2d_like
                           #   (v3: expressed as s(gap; _VAC_GAP_MIN_A) >= _SCORE_GATE;
                           #    keep equal to _VAC_GAP_MIN_A — recalibrate together)
_SCORE_GATE = 0.5         # soft-gate decision threshold (see docstring note 4:
                           #   min-composite score >= 0.5 <=> the v2 hard AND gate)
_USE_PLANE_NORMALS = True # v3 vacuum search: include low-index plane-normal
                           #   candidates (False = legacy 3-axis search; used by
                           #   regress_2k_local.py to isolate the soft-gate effect)
_NU_POROUS = 4.0          # volume_norm ν threshold for the porous pre-screen
                          #   (task text: ν>3; raised to 4.0 as the most-conservative
                          #    reading of total plan §Level2 "φ ≪ normal": ν>3 flags
                          #    11.1% of subset_2k incl. diamond-Si (ν=3.5, a dense
                          #    covalent network); ν>4 gives 3.1% ≈ the plan's 1-5%
                          #    candidate expectation; zeolites ν>8 stay captured)
_S_THRESH = 0.6           # surface_score threshold (contracts.md §5)
_W_VAC, _W_DCN, _W_LAY = 0.30, 0.60, 0.10
_VF_REF = 0.30            # vacuum_fraction giving full s_vac credit
_DCN_REF = 4.0            # ΔCN giving full s_dcn credit (ΔCN=1 slab -> 0.55 < 0.6)
_LAYER_GAP_FRAC = 0.7     # × median bond length = layer clustering threshold
_LAYER_GAP_MIN = 1.0      # Å floor for the layer threshold
_OUTER_FRAC = 0.12        # outermost fraction of the span = "outer" atoms for ΔCN
_CHAIN_MIN_ATOMS = 5      # min component size for the inertia chain detector
_CHAIN_ASPECT = 3.0       # gyration eigenvalue ratio lambda1/lambda2 chain gate


@dataclass
class MorphologyResult:
    """Frozen per contracts.md §3.5."""

    vacuum_gap: float        # Å (max empty interval along an axis, minus atom radii)
    vacuum_fraction: float
    surface_score: float     # 0..1
    porous_candidate: bool
    n_components: int
    f_max: float             # fraction of atoms in the largest connected component
    class_label: str


# ---------------------------------------------------------------------------
# internal fallback containers (duck-type compatible with crysh.dimensionality/dimension)
# ---------------------------------------------------------------------------

@dataclass
class _Graph:
    i: np.ndarray
    j: np.ndarray
    S: np.ndarray
    d: np.ndarray
    n_atoms: int
    lam: float


@dataclass
class _Spectrum:
    d_by_lambda: dict
    d_star: int
    persistence: float
    n_components: int
    translations_rank: int


def _atom_rcov(atoms) -> np.ndarray:
    """Per-atom covalent radii (ase natural cutoffs, mult=1)."""
    return np.asarray(natural_cutoffs(atoms, mult=1), dtype=np.float64)


_RCOV_BY_Z: dict[int, float] = {}


def _rcov_by_z(z: int) -> float:
    if z not in _RCOV_BY_Z:
        from ase.data import covalent_radii

        _RCOV_BY_Z[z] = float(covalent_radii[z])
    return _RCOV_BY_Z[z]


def _pair_cutoffs_simple(atoms, lam: float) -> dict:
    zset = sorted(set(int(z) for z in atoms.get_atomic_numbers()))
    return {
        (a, b): _FALLBACK_TOL * lam * (_rcov_by_z(a) + _rcov_by_z(b))
        for a in zset
        for b in zset
    }


def _build_graph(atoms, lam: float) -> _Graph:
    """Fallback periodic bond graph at λ (ase neighbor_list 'ijdS').

    Implementation note (integration review v2): the query uses the single float
    max cutoff and the per-pair bond criterion ``d_ij < lam * TOL * (r_cov_i +
    r_cov_j)`` is applied in numpy afterwards — exactly the contract criterion.
    This is an equivalent-implementation choice, NOT a workaround: the earlier
    claim that ASE's dict-cutoff path drops pairs was RETRACTED (the repro was an
    artifact of a misbuilt test fixture; the integrator brute-force-verified the
    dict path 44/44 on controls, and control_isolated_molecule_02's "missing"
    pair is an H-H contact correctly excluded by the H-H cutoff).
    """
    cut = _pair_cutoffs_simple(atoms, lam)
    max_cut = max(cut.values())
    i, j, d, S = neighbor_list("ijdS", atoms, max_cut)
    z = np.asarray(atoms.numbers, dtype=np.int64)
    zlo = np.minimum(z[i], z[j])
    zhi = np.maximum(z[i], z[j])
    pc = np.array([cut[(int(a), int(b))] for a, b in zip(zlo, zhi)])
    keep = (d < pc) & ~((i == j) & (S == 0).all(axis=1))  # pair cutoff + self loops
    return _Graph(
        i=i[keep].astype(np.int64),
        j=j[keep].astype(np.int64),
        S=S[keep].astype(np.int64),
        d=d[keep].astype(np.float64),
        n_atoms=len(atoms),
        lam=float(lam),
    )


def _union_find(n: int, i, j):
    parent = np.arange(n, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(i, j):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    comp_of = np.array([find(x) for x in range(n)], dtype=np.int64)
    return comp_of


def _components_fmax(graph: _Graph):
    """Periodic connected components (boundary-crossing edges merge components)."""
    n = graph.n_atoms
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0, 0.0
    comp_of = _union_find(n, graph.i, graph.j)
    sizes = np.bincount(comp_of, minlength=n)
    n_components = int(np.count_nonzero(sizes))
    f_max = float(sizes.max() / n)
    return comp_of, n_components, f_max


def _int_rank(vecs) -> int:
    """Exact rank of a set of integer 3-vectors over Q (fractional row-echelon)."""
    pivots: list[tuple[int, list[Fraction]]] = []
    for vec in vecs:
        row = [Fraction(int(x), 1) for x in vec]
        for col, prow in pivots:
            if row[col] != 0:
                f = row[col] / prow[col]
                row = [x - f * y for x, y in zip(row, prow)]
        for c in range(3):
            if row[c] != 0:
                pivots.append((c, [x / row[c] for x in row]))
                break
    return len(pivots)


def _loop_rank_max(graph: _Graph, comp_of) -> int:
    """d = max over components of rank{T_loop}, T_loop = t_i + S_ij − t_j (BFS)."""
    n = graph.n_atoms
    adj: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(n)]
    for a, b, s in zip(graph.i.tolist(), graph.j.tolist(), graph.S.tolist()):
        s = np.asarray(s, dtype=np.int64)
        adj[a].append((b, s))
        adj[b].append((a, -s))
    t: list[np.ndarray | None] = [None] * n
    best = 0
    for root in range(n):
        if t[root] is not None:
            continue
        t[root] = np.zeros(3, dtype=np.int64)
        vecs: list[np.ndarray] = []
        q = deque([root])
        while q:
            u = q.popleft()
            for v, s in adj[u]:
                if t[v] is None:
                    t[v] = t[u] + s
                    q.append(v)
                else:
                    w = t[u] + s - t[v]
                    if w.any():
                        vecs.append(w)
        best = max(best, _int_rank(vecs))
    return best


def _spectrum_fallback(atoms) -> _Spectrum:
    """d(λ) spectrum with the fallback table (same λ grid as contract §3.3)."""
    d_by_lambda: dict[float, int] = {}
    g1: _Graph | None = None
    for lam in LAMBDAS:
        g = _build_graph(atoms, lam)
        comp_of, _, _ = _components_fmax(g)
        d = _loop_rank_max(g, comp_of)
        d_by_lambda[float(lam)] = int(d)
        if abs(lam - 1.00) < 1e-9:
            g1 = g
    d_star = int(d_by_lambda[1.00])
    persistence = float(
        sum(1 for lam in LAMBDAS if d_by_lambda[float(lam)] == d_star) / len(LAMBDAS)
    )
    if g1 is not None:
        _, n_components, _ = _components_fmax(g1)
    else:  # pragma: no cover - LAMBDAS always contains 1.00
        n_components = 0
    return _Spectrum(
        d_by_lambda=d_by_lambda,
        d_star=d_star,
        persistence=persistence,
        n_components=n_components,
        translations_rank=d_star,
    )


def _compute_internals(atoms):
    """Internal self-computation for the None path (contract: "内部以 λ=1.0 自算").

    Level-1 (merged, brute-force-verified) is preferred when importable
    (``_PREFER_L1``); this module's own fallback (TOL = 1.15 table) is the
    stand-alone last resort and never blocks on level-1.  Callers may pass
    `graph`/`spectrum` explicitly to bypass this path entirely (the pipeline
    does).
    """
    if _PREFER_L1:
        try:
            spec = _l1_dimension(atoms)
            gr = _l1_build_bond_graph(atoms, lam=_L1_GRAPH_LAM)  # pipeline-consistent
            if spec is not None and gr is not None and spec.d_star is not None:
                return gr, spec, "level1"
        except Exception:
            pass
    spec = _spectrum_fallback(atoms)
    gr = _build_graph(atoms, 1.0)
    return gr, spec, "fallback"


# ---------------------------------------------------------------------------
# vacuum
# ---------------------------------------------------------------------------

def _gap_score(x: float, c: float) -> float:
    """s(x; c) = x/(x+c): strictly increasing, s(0)=0, s(c)=0.5, s(∞)=1.

    The v3 soft gates evaluate ``s(x; c) >= 0.5`` which is exactly ``x >= c``
    in real arithmetic (x ≥ 0, c > 0) — the decision-equivalence proof of the
    gate reformulation (docstring note 4).
    """
    return float(x) / (float(x) + float(c))


def _vacuum_score(gap: float, frac: float) -> tuple[float, float, float]:
    """(vacuum_score, s_gap, s_frac): min-composite of the two AND-ed v2 gates.

    ``min(s_gap, s_frac) >= _SCORE_GATE`` ⟺ ``gap >= _VAC_GAP_MIN_A`` AND
    ``frac >= _VAC_FRAC_MIN`` (both components must pass because s is strictly
    increasing with s(c) = 0.5).
    """
    s_gap = _gap_score(gap, _VAC_GAP_MIN_A)
    s_frac = _gap_score(frac, _VAC_FRAC_MIN)
    return min(s_gap, s_frac), s_gap, s_frac


def _slab_score(gap: float) -> float:
    """slab_score = gap/(gap+12) — 0.5 gate ⟺ gap >= _SLAB_GAP_MIN_A."""
    return _gap_score(gap, _SLAB_GAP_MIN_A)


def _mono_score(span: float, med_bond: float) -> float:
    """mono_score = 1/(1 + span/(1.6*med)) — 0.5 gate ⟺ span <= _SPAN_FRAC*med."""
    denom = _SPAN_FRAC * float(med_bond)
    if denom <= 0.0:
        return 1.0 if float(span) <= 0.0 else 0.0
    return 1.0 / (1.0 + float(span) / denom)


def _score_margin(score: float) -> float:
    """2*|score - 0.5| in [0, 1]: distance from the decision boundary."""
    return 2.0 * abs(float(score) - _SCORE_GATE)


def _plane_normal_candidates(cell) -> list[tuple[np.ndarray, float, tuple[int, int, int]]]:
    """Low-index plane-normal candidates (hkl), h,k,l in {-1,0,1}, not all zero.

    G = h*b1 + k*b2 + l*b3 with the reciprocal basis B = inv(A).T (rows b_i,
    b_i·a_j = δ_ij); interplanar spacing d = 1/|G|.  Antiparallel duplicates are
    removed by the canonical-sign rule (first nonzero component positive), which
    is exact for components in {-1,0,1} (parallel integer vectors here are ±
    equal).  Deterministic order: itertools.product((-1,0,1), repeat=3).
    Returns [(G, d_hkl, (h, k, l)), ...] — candidate index m (diagnostic
    ``vacuum_axis == 3 + m``) refers to this list order.
    """
    A = np.asarray(cell, dtype=np.float64).reshape(3, 3)
    try:
        B = np.linalg.inv(A).T
    except np.linalg.LinAlgError:  # degenerate cell — axes only
        return []
    out: list[tuple[np.ndarray, float, tuple[int, int, int]]] = []
    for h, k, l in itertools.product((-1, 0, 1), repeat=3):
        if (h, k, l) == (0, 0, 0):
            continue
        if next(v for v in (h, k, l) if v != 0) < 0:  # antiparallel duplicate
            continue
        G = np.asarray((h, k, l), dtype=np.float64) @ B
        norm = float(np.linalg.norm(G))
        if norm <= 1e-12:
            continue
        out.append((G, 1.0 / norm, (h, k, l)))
    return out


def _circular_gap(z: np.ndarray, h: float, rc: np.ndarray):
    """Largest empty circular interval on the phase z in [0,1), period h (Å).

    Isomorphic core of both the axis and the plane-normal algorithms: sort the
    phases, take the circular differences, scale by the period and subtract the
    covalent radii of the two bounding atoms.  Returns
    (net_gap, g_raw, idx, order, zs) — idx = position of the left bounding atom
    of the max gap within the sorted array zs = z[order].
    """
    n = len(z)
    order = np.argsort(z, kind="stable")
    zs = z[order]
    gaps_frac = np.empty(n, dtype=np.float64)
    if n > 1:
        gaps_frac[:-1] = zs[1:] - zs[:-1]
    gaps_frac[-1] = zs[0] + 1.0 - zs[-1]
    r_left = rc[order]
    r_right = rc[np.roll(order, -1)]
    net = gaps_frac * h - (r_left + r_right)
    idx = int(np.argmax(net))
    return max(0.0, float(net[idx])), float(gaps_frac[idx] * h), idx, order, zs


def _vacuum(atoms) -> dict:
    """Largest empty circular interval along the candidate vacuum directions.

    v3: candidates = 3 lattice axes (legacy fractional-projection metric, kept
    verbatim for compatibility + the ``per_axis_gaps`` diagnostics) + 13
    low-index plane normals (physically correct perpendicular metric,
    docstring note 2).  Best = largest net gap; strict ``>`` with axes first
    keeps orthogonal-cell results bit-identical to v2.
    """
    n = len(atoms)
    out = {
        "vacuum_gap": 0.0,
        "vacuum_fraction": 0.0,
        "axis": 0,
        "g_raw": 0.0,
        "z_shift": np.zeros(n) if n else np.zeros(0),
        "order": np.zeros(0, dtype=np.int64),
        "h": 0.0,                       # period (Å) of the winning direction
        "per_axis_gaps": [0.0, 0.0, 0.0],
        "normal": [0.0, 0.0, 0.0],      # unit vector of the winning direction
        "miller": None,                 # (h,k,l) when a plane normal won
    }
    if n == 0:
        return out
    frac = atoms.get_scaled_positions(wrap=True)
    cell = np.asarray(atoms.cell, dtype=np.float64)
    rc = _atom_rcov(atoms)
    best_gap = -1.0

    # --- 3 lattice axes (legacy metric; identical to the v2 implementation) ---
    for k in range(3):
        h = float(np.linalg.norm(cell[k]))
        if h <= 0:
            out["per_axis_gaps"][k] = 0.0
            continue
        z = np.mod(frac[:, k], 1.0)
        gap, g_raw, idx, order, zs = _circular_gap(z, h, rc)
        out["per_axis_gaps"][k] = gap
        if gap > best_gap:
            best_gap = gap
            out["vacuum_gap"] = gap
            out["vacuum_fraction"] = gap / h
            out["axis"] = k
            out["g_raw"] = g_raw
            out["h"] = h
            # re-anchor the circular coordinate so the max gap sits at the wrap
            out["z_shift"] = np.mod(frac[:, k] - zs[idx], 1.0)
            out["order"] = order
            out["normal"] = [float(v) for v in (cell[k] / h)]
            out["miller"] = None

    # --- low-index plane normals (v3; perpendicular metric) ---
    if _USE_PLANE_NORMALS:
        positions = np.asarray(atoms.positions, dtype=np.float64)
        for m_idx, (G, d_hkl, hkl) in enumerate(_plane_normal_candidates(cell)):
            z = np.mod(positions @ G, 1.0)
            gap, g_raw, idx, order, zs = _circular_gap(z, d_hkl, rc)
            if gap > best_gap:
                best_gap = gap
                out["vacuum_gap"] = gap
                out["vacuum_fraction"] = gap / d_hkl
                out["axis"] = 3 + m_idx
                out["g_raw"] = g_raw
                out["h"] = d_hkl
                out["z_shift"] = np.mod(z - zs[idx], 1.0)
                out["order"] = order
                out["normal"] = [float(v) for v in (G * d_hkl)]  # G/|G|
                out["miller"] = list(hkl)
    return out


def _volume_norm(atoms) -> float:
    V = abs(float(np.linalg.det(atoms.cell)))
    rc = _atom_rcov(atoms)
    s = float(np.sum((4.0 / 3.0) * np.pi * rc ** 3))
    return V / s if s > 0 else np.inf


def _degree(graph: _Graph, n: int) -> np.ndarray:
    deg = np.zeros(n, dtype=np.float64)
    np.add.at(deg, graph.i, 1.0)
    return deg


def _n_layers(z_shift: np.ndarray, graph: _Graph, h: float) -> int:
    """Clustered atomic layers along the vacuum normal (wrap sits inside the vacuum)."""
    n = len(z_shift)
    if n == 0:
        return 0
    med = float(np.median(graph.d)) if len(graph.d) else 2.0
    thr = max(_LAYER_GAP_FRAC * med, _LAYER_GAP_MIN)
    zs = np.sort(z_shift)
    gaps = np.diff(zs) * h
    return int(np.count_nonzero(gaps > thr) + 1)


def _component_shape(atoms, comp_of) -> tuple[int, float]:
    """Largest connected component: (size, gyration aspect λ1/λ2)."""
    if len(comp_of) == 0:
        return 0, 1.0
    sizes = np.bincount(comp_of, minlength=len(comp_of))
    cid = int(sizes.argmax())
    idx = np.where(comp_of == cid)[0]
    pos = np.asarray(atoms.positions, dtype=np.float64)[idx]
    pos = pos - pos.mean(axis=0)
    cov = (pos.T @ pos) / max(len(pos), 1)
    ev = np.linalg.eigvalsh(cov)[::-1]  # descending λ1 ≥ λ2 ≥ λ3 ≥ 0
    if ev[1] > 1e-12:
        aspect = float(ev[0] / ev[1])
    elif ev[0] > 1e-12:
        aspect = float(ev[0] / max(ev[2], 1e-12))
    else:
        aspect = 1.0
    return int(len(idx)), aspect


def _surface_score(cn: np.ndarray, z_shift: np.ndarray, spectrum, vac: dict,
                   monolayer: bool) -> float:
    """S = 0.30·s_vac + 0.60·s_dcn + 0.10·s_lay (documented pilot weights).

    s_dcn uses the layer-resolved CN deficit along the vacuum normal
    (⟨CN⟩_interior − ⟨CN⟩_outer, outer = outermost 12% of the span), not the
    global CN std: the std proxy conflated elemental CN diversity (MoS2 ≈ 4.2)
    with surface gradients.  ΔCN = 0 for monolayers (no interior).
    """
    vf = float(vac["vacuum_fraction"])
    s_vac = min(1.0, vf / _VF_REF)
    z = np.asarray(z_shift, dtype=np.float64)
    dcn = 0.0
    if (not monolayer) and len(cn) == len(z) and len(z) > 0:
        outer = (z < _OUTER_FRAC) | (z > 1.0 - _OUTER_FRAC)
        core = ~outer
        if core.any() and outer.any():
            dcn = float(cn[core].mean() - cn[outer].mean())
    s_dcn = min(1.0, max(0.0, dcn / _DCN_REF))
    db = getattr(spectrum, "d_by_lambda", {}) or {}
    if db.get(0.90) == 2 or db.get(1.00) == 2:
        s_lay = 1.0
    elif db.get(1.10) == 2:
        s_lay = 0.5
    else:
        s_lay = 0.0
    return float(min(1.0, max(0.0, _W_VAC * s_vac + _W_DCN * s_dcn + _W_LAY * s_lay)))


def _classify(eff_d: int, big_vac: bool, vac_gap: float, s_gap: float,
              slab_sc: float, monolayer: bool, chain_like: bool, score: float,
              porous: bool, db_max: int) -> str:
    """Route on the soft-gate decisions (docstring note 4; branch structure
    unchanged from v2 — every gate comparison below is decision-equivalent to
    the v2 hard threshold it replaces)."""
    if eff_d == 0:
        if vac_gap >= _VAC_ISOLATED_GAP:  # distinct 7 Å bar (not softened)
            return "isolated_chain" if chain_like else "isolated_molecule"
        return "molecular_crystal"
    if eff_d == 1:
        return "isolated_chain" if big_vac else "chain_solid"
    if eff_d == 2:
        if db_max >= 3:
            return "layered_bulk"
        if monolayer and s_gap >= _SCORE_GATE:  # ⟺ vac_gap >= _MONO_VAC_GAP
            return "intrinsic_2d_like"
        if slab_sc >= _SCORE_GATE:  # ⟺ vac_gap >= _SLAB_GAP_MIN_A
            return "surface_like" if score > _S_THRESH else "explicit_vacuum_slab"
        return "layered_bulk"
    if eff_d == 3:
        if big_vac:
            return "surface_like" if score > _S_THRESH else "explicit_vacuum_slab"
        if porous:
            return "porous_candidate"
        return "dense_bulk"
    return "ambiguous"


# ---------------------------------------------------------------------------
# public API (frozen) + pilot diagnostics helper
# ---------------------------------------------------------------------------

def morphology_with_diagnostics(atoms, graph=None, spectrum=None, coord=None) -> dict:
    """`morphology()` fields plus pilot diagnostics (d_star, eff_d_star, volume_norm,
    n_layers, big_vacuum, vacuum_axis, cn_std, span, comp_size, comp_aspect,
    spectrum_source + the v3 soft-gate keys vacuum_score/slab_score/mono_score
    with their margins + per_axis_gaps/vacuum_normal/vacuum_miller from the
    extended vacuum search).  The extra keys are NOT part of the frozen
    contract; the 2k batch script uses them to enrich the parquet."""
    zero = dict(
        vacuum_gap=0.0,
        vacuum_fraction=0.0,
        surface_score=0.0,
        porous_candidate=False,
        n_components=0,
        f_max=0.0,
        class_label="ambiguous",
        d_star=-1,
        eff_d_star=-1,
        volume_norm=np.nan,
        n_layers=0,
        big_vacuum=False,
        vacuum_axis=0,
        cn_std=np.nan,
        span=np.nan,
        comp_size=0,
        comp_aspect=np.nan,
        spectrum_source="empty",
        vacuum_score=0.0,
        slab_score=0.0,
        mono_score=0.0,
        vacuum_margin=1.0,
        slab_margin=1.0,
        mono_margin=1.0,
        per_axis_gaps=[0.0, 0.0, 0.0],
        vacuum_normal=[0.0, 0.0, 0.0],
        vacuum_miller=None,
    )
    n = len(atoms)
    if n == 0:
        return zero

    source = "caller"
    if graph is None and spectrum is None:
        graph, spectrum, source = _compute_internals(atoms)
    elif graph is None:
        if _PREFER_L1:
            try:
                graph = _l1_build_bond_graph(atoms, lam=_L1_GRAPH_LAM)
                source = "level1-graph"
            except Exception:
                graph, source = _build_graph(atoms, 1.0), "fallback-graph"
        else:
            graph, source = _build_graph(atoms, 1.0), "fallback-graph"
    elif spectrum is None:
        if _PREFER_L1:
            try:
                spectrum = _l1_dimension(atoms)
                source = "level1-spectrum"
            except Exception:
                spectrum, source = _spectrum_fallback(atoms), "fallback-spectrum"
        else:
            spectrum, source = _spectrum_fallback(atoms), "fallback-spectrum"

    i = np.asarray(graph.i, dtype=np.int64).reshape(-1)
    j = np.asarray(graph.j, dtype=np.int64).reshape(-1)
    S = np.asarray(graph.S, dtype=np.int64).reshape(-1, 3)
    d_edges = np.asarray(getattr(graph, "d", np.zeros(0)), dtype=np.float64).reshape(-1)
    n_atoms = int(getattr(graph, "n_atoms", n))
    gn = _Graph(i=i, j=j, S=S, d=d_edges, n_atoms=n_atoms,
                lam=float(getattr(graph, "lam", 1.0)))

    comp_of, n_components, f_max = _components_fmax(gn)
    vac = _vacuum(atoms)
    nu = _volume_norm(atoms)
    # v3 soft gates (decision-equivalent to the v2 hard gates; docstring note 4)
    vac_score, s_gap, s_frac = _vacuum_score(vac["vacuum_gap"],
                                             vac["vacuum_fraction"])
    big_vac = bool(vac_score >= _SCORE_GATE)
    slab_sc = _slab_score(vac["vacuum_gap"])
    h_dir = float(vac["h"])  # period of the winning direction (axis len or d_hkl)
    n_layers = _n_layers(vac["z_shift"], gn, h_dir)
    med_bond = float(np.median(d_edges)) if len(d_edges) else 2.0
    span = float(max(0.0, h_dir - vac["g_raw"]))  # thickness along vacuum normal
    mono_sc = _mono_score(span, med_bond)
    monolayer = bool(mono_sc >= _SCORE_GATE)

    d_star = int(getattr(spectrum, "d_star", -1))
    db = dict(getattr(spectrum, "d_by_lambda", {}) or {})
    db_max = max((int(v) for v in db.values()), default=-1)

    # Cordero under-bonding rescue (documented): the frozen pure-covalent table
    # gives d(1.0) = 0 for many bulk solids; route on the effective dimension.
    if d_star == 0:
        d_rescue = max(int(db.get(1.10, 0)), int(db.get(1.20, 0)), int(db.get(1.35, 0)))
        if d_rescue >= 3:
            eff_d = 3
        elif int(db.get(1.20, 0)) == 2:
            eff_d = 2
        elif int(db.get(1.20, 0)) == 1:
            eff_d = 1
        else:
            eff_d = 0
    else:
        eff_d = d_star

    comp_size, comp_aspect = _component_shape(atoms, comp_of)
    chain_like = bool(comp_size >= _CHAIN_MIN_ATOMS and comp_aspect >= _CHAIN_ASPECT)

    cn = (np.asarray(coord.cn, dtype=np.float64)
          if coord is not None and getattr(coord, "cn", None) is not None
          else _degree(gn, n_atoms))
    score = _surface_score(cn, vac["z_shift"], spectrum, vac, monolayer)

    porous = bool(eff_d == 3 and nu > _NU_POROUS and not big_vac and n_components == 1)
    label = _classify(eff_d, big_vac, float(vac["vacuum_gap"]), s_gap, slab_sc,
                      monolayer, chain_like, score, porous, db_max)

    deg = _degree(gn, n_atoms)
    return dict(
        vacuum_gap=float(vac["vacuum_gap"]),
        vacuum_fraction=float(vac["vacuum_fraction"]),
        surface_score=float(score),
        porous_candidate=bool(porous),
        n_components=int(n_components),
        f_max=float(f_max),
        class_label=label,
        d_star=int(d_star),
        eff_d_star=int(eff_d),
        volume_norm=float(nu),
        n_layers=int(n_layers),
        big_vacuum=bool(big_vac),
        vacuum_axis=int(vac["axis"]),
        cn_std=float(np.std(deg)) if len(deg) else np.nan,
        span=float(span),
        comp_size=int(comp_size),
        comp_aspect=float(comp_aspect),
        spectrum_source=source,
        d_by_lambda={float(k): int(v) for k, v in db.items()},
        dim_persistence=float(getattr(spectrum, "persistence", np.nan)),
        # v3 soft gates + margins (non-frozen diagnostics)
        vacuum_score=float(vac_score),
        slab_score=float(slab_sc),
        mono_score=float(mono_sc),
        vacuum_margin=float(_score_margin(vac_score)),
        slab_margin=float(_score_margin(slab_sc)),
        mono_margin=float(_score_margin(mono_sc)),
        # v3 extended vacuum search (non-frozen diagnostics)
        per_axis_gaps=[float(g) for g in vac["per_axis_gaps"]],
        vacuum_normal=[float(v) for v in vac["normal"]],
        vacuum_miller=(list(vac["miller"]) if vac["miller"] is not None else None),
    )


def morphology(atoms, graph=None, spectrum=None, coord=None) -> MorphologyResult:
    """Level-2 morphology classification.  Signature frozen in contracts.md §3.5.

    `graph`/`spectrum`/`coord` may be `None`; missing pieces are computed internally
    (level-1 implementation when importable, documented fallback otherwise).
    """
    d = morphology_with_diagnostics(atoms, graph=graph, spectrum=spectrum, coord=coord)
    return MorphologyResult(
        vacuum_gap=d["vacuum_gap"],
        vacuum_fraction=d["vacuum_fraction"],
        surface_score=d["surface_score"],
        porous_candidate=d["porous_candidate"],
        n_components=d["n_components"],
        f_max=d["f_max"],
        class_label=d["class_label"],
    )
