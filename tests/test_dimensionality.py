"""Dimensionality-spectrum tests with inline fixtures (no controls dependency).

Note on cutoff tables: the frozen bond criterion is
``d_ij < lambda * (r_cov_i + r_cov_j)``. ASE covalent radii (C 0.76,
Si 1.11, Na 1.66, Cl 1.02 A) place the canonical bond lengths of several
textbook structures outside the pure covalent window (e.g. Si-Si 2.35 vs
r0 2.22; graphite interlayer 3.35 vs r0 1.52), so fixtures pass explicit
``cutoff_table`` entries to probe the graph/rank algorithm at the exact
d(lambda) patterns the contract specifies. In production the calibrated
pair table serves the same role.

Default grid (extended): LAMBDAS below mirrors the production default
(0.90 .. 2.00, 8 points) — the 1.75/2.00 points expose the 2->3
transition of vdW-layered structures (interlayer q = d/r0 ~ 1.6-2.2,
invisible under the former 1.50 ceiling). Passing an explicit
``cutoff_table`` activates v1.2 table mode (d_star at lambda = 1.00);
without a table d_star stays at D_STAR_LAMBDA = 1.20.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, graphene

from crysh.dimensionality import D_STAR_LAMBDA, DimensionSpectrum, dimensionality_spectrum

LAMBDAS = (0.90, 1.00, 1.10, 1.20, 1.35, 1.50, 1.75, 2.00)


def test_diamond_si_all_lambda_3d():
    atoms = bulk("Si", "diamond", a=5.431)
    spec = dimensionality_spectrum(atoms, cutoff_table={(14, 14): 2.70})
    assert isinstance(spec, DimensionSpectrum)
    assert spec.d_by_lambda == {lam: 3 for lam in LAMBDAS}
    assert spec.d_star == 3
    assert spec.translations_rank == 3
    assert spec.persistence == 1.0
    assert spec.n_components == 1


def test_graphene_monolayer_vacuum_all_lambda_2d():
    atoms = graphene(a=2.46, size=(2, 2, 1), vacuum=15.0)
    atoms.pbc = True  # graphene() returns pbc z=False; POSCAR-like input is fully periodic
    spec = dimensionality_spectrum(atoms, cutoff_table={(6, 6): 1.70})
    assert all(d == 2 for d in spec.d_by_lambda.values())
    assert spec.d_star == 2
    assert spec.persistence == 1.0
    assert spec.n_components == 1


def test_water_molecule_big_box_all_lambda_0d():
    # default covalent table: O-H 0.957 A < r0(O,H) = 0.97 A at lambda >= 1.0
    L = 15.0
    atoms = Atoms(
        "OH2",
        positions=[(L / 2, L / 2, L / 2),
                   (L / 2 + 0.757, L / 2 + 0.586, L / 2),
                   (L / 2 - 0.757, L / 2 + 0.586, L / 2)],
        cell=[L, L, L], pbc=True,
    )
    spec = dimensionality_spectrum(atoms)
    assert all(d == 0 for d in spec.d_by_lambda.values())
    assert spec.d_star == 0
    assert spec.persistence == 1.0
    assert spec.n_components == 1  # intact molecule at lambda = 1.0


def test_linear_carbon_chain_all_lambda_1d():
    atoms = Atoms(
        "C3",
        positions=[(7.5, 7.5, 0.0), (7.5, 7.5, 1.40), (7.5, 7.5, 2.80)],
        cell=[15.0, 15.0, 4.2], pbc=True,
    )
    spec = dimensionality_spectrum(atoms, cutoff_table={(6, 6): 1.70})
    assert all(d == 1 for d in spec.d_by_lambda.values())
    assert spec.d_star == 1
    assert spec.persistence == 1.0
    assert spec.n_components == 1


def test_graphite_ab_short_lambda_2d_long_lambda_3d():
    # Graphite AB (P6_3/mmc): A at (0,0,1/4),(1/3,2/3,1/4); B at (0,0,3/4),(2/3,1/3,3/4).
    # Intralayer C-C = 1.42 A, interlayer 3.35 A. With r0(C,C)=2.60 the
    # criterion lambda*r0 crosses the interlayer gap for lambda in (1.20, 1.35].
    a, c = 2.46, 6.70
    cell = [[a, 0.0, 0.0],
            [-a / 2, a * np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, c]]
    atoms = Atoms(
        "C4",
        scaled_positions=[[0.0, 0.0, 0.25], [1 / 3, 2 / 3, 0.25],
                          [0.0, 0.0, 0.75], [2 / 3, 1 / 3, 0.75]],
        cell=cell, pbc=True,
    )
    spec = dimensionality_spectrum(atoms, cutoff_table={(6, 6): 2.60})
    assert spec.d_by_lambda[0.90] == 2
    assert spec.d_by_lambda[1.00] == 2
    assert spec.d_by_lambda[1.10] == 2
    assert spec.d_by_lambda[1.20] == 2
    assert spec.d_by_lambda[1.35] == 3
    assert spec.d_by_lambda[1.50] == 3
    assert spec.d_by_lambda[1.75] == 3  # extended grid: stays connected
    assert spec.d_by_lambda[2.00] == 3
    assert spec.d_star == 2
    assert spec.persistence == pytest.approx(4 / 8)  # denominator = 8 lambdas
    # at lambda = 1.0 (cutoff 2.60 A) the two layers are still disconnected
    assert spec.n_components == 2
    assert spec.translations_rank == 2
    # both components are rank-2 layers of 2 atoms each (table-mode d* graph)
    assert spec.dim_components == {2: [2, 4]}


def test_nacl_all_lambda_3d():
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    spec = dimensionality_spectrum(atoms, cutoff_table={(11, 17): 3.20})
    assert all(d == 3 for d in spec.d_by_lambda.values())
    assert spec.d_star == 3
    assert spec.n_components == 1


def test_single_atom_cell_no_bonds_0d():
    atoms = Atoms("Si", positions=[(2.5, 2.5, 2.5)], cell=[5, 5, 5], pbc=True)
    spec = dimensionality_spectrum(atoms)
    assert all(d == 0 for d in spec.d_by_lambda.values())
    assert spec.d_star == 0
    assert spec.n_components == 1


def test_single_atom_periodic_self_bonds_3d():
    # atom bonded to its own periodic images: loop translations span 3D
    atoms = Atoms("C", positions=[(0.75, 0.75, 0.75)], cell=[1.5, 1.5, 1.5], pbc=True)
    spec = dimensionality_spectrum(atoms, cutoff_table={(6, 6): 1.70})
    assert all(d == 3 for d in spec.d_by_lambda.values())
    assert spec.n_components == 1


def test_disconnected_chains_dim_is_max_over_components():
    # Two chains in different directions without inter-chain bonds: each
    # component has rank 1, so d = max over components = 1 (contracts.md
    # §3.3 "多连通分量取 max" — documented interpretation in progress.md;
    # a merged-across-components rank would give 2 here).
    atoms = Atoms(
        "C8",
        positions=[(2.8, 10.0, 0.0), (2.8, 10.0, 1.4), (2.8, 10.0, 2.8), (2.8, 10.0, 4.2),
                   (0.0, 18.0, 2.8), (1.4, 18.0, 2.8), (2.8, 18.0, 2.8), (4.2, 18.0, 2.8)],
        cell=[5.6, 20.0, 5.6], pbc=True,
    )
    spec = dimensionality_spectrum(atoms, cutoff_table={(6, 6): 1.70})
    assert all(d == 1 for d in spec.d_by_lambda.values())
    assert spec.n_components == 2
    assert spec.dim_components == {1: [2, 8]}  # two rank-1 chains, 4+4 atoms


def test_spectrum_requires_d_star_lambda_in_grid():
    atoms = Atoms("C", positions=[(0, 0, 0)], cell=[10, 10, 10], pbc=True)
    with pytest.raises(ValueError):
        dimensionality_spectrum(atoms, lambdas=(0.9, 1.1))  # default d_star_lambda=1.20 missing
    with pytest.raises(ValueError):
        dimensionality_spectrum(atoms, lambdas=())
    # explicit v1.0-style grid still supported via d_star_lambda override
    spec = dimensionality_spectrum(atoms, lambdas=(0.9, 1.0, 1.1), d_star_lambda=1.00)
    assert spec.d_star == spec.d_by_lambda[1.00] == 0


def test_v11_d_star_lambda_default_120():
    # v1.1 integrator revision: d_star = d(lambda=1.20), n_components on the
    # lambda=1.20 graph. NaCl rocksalt (r0(Na,Cl)=2.68 vs 2.82 A bond) has NO
    # bonds at lambda=1.00 and full 3D connectivity from lambda=1.10 on, so
    # the two scales give different answers — locking the semantics.
    assert D_STAR_LAMBDA == 1.20
    atoms = bulk("NaCl", "rocksalt", a=5.64)  # 2-atom primitive cell
    spec100 = dimensionality_spectrum(atoms, d_star_lambda=1.00)
    assert spec100.d_by_lambda[1.00] == 0
    assert spec100.d_star == 0
    assert spec100.translations_rank == 0
    assert spec100.n_components == 2  # both atoms isolated at lambda=1.00
    assert spec100.dim_components == {0: [2, 2]}  # two rank-0 singletons
    spec = dimensionality_spectrum(atoms)  # default D_STAR_LAMBDA = 1.20
    assert spec.d_by_lambda[1.20] == 3
    assert spec.d_star == 3
    assert spec.translations_rank == 3
    assert spec.n_components == 1  # connected at lambda=1.20
    assert spec.dim_components == {3: [1, 2]}
    # d(0.90)=d(1.00)=0 (no bonds), d(1.10..2.00)=3 -> 6/8 agree with d_star=3
    assert spec.persistence == 6 / 8


def test_spectrum_d_by_lambda_keys_are_ints_and_floats():
    atoms = bulk("Si", "diamond", a=5.431)
    spec = dimensionality_spectrum(atoms, cutoff_table={(14, 14): 2.7})
    assert all(isinstance(d, int) and 0 <= d <= 3 for d in spec.d_by_lambda.values())
    assert spec.d_by_lambda[D_STAR_LAMBDA] == spec.d_star  # v1.1: d* at lambda=1.20
    assert 0.0 < spec.persistence <= 1.0
    # dim_components typing: int keys in {0..3}, [int, int] values
    assert all(isinstance(r, int) and 0 <= r <= 3 for r in spec.dim_components)
    assert all(len(v) == 2 and all(isinstance(x, int) for x in v)
               for v in spec.dim_components.values())


# --------------------------------------------------------------------------- #
# extended default grid + dim_components (vdW visibility / multi-label output)
# --------------------------------------------------------------------------- #

def test_default_grid_contains_175_and_200():
    # Default grid = 8 ascending points incl. the two new vdW points;
    # persistence is k/8 (denominator len(lambdas)).
    atoms = Atoms("C", positions=[(0.75, 0.75, 0.75)], cell=[1.5, 1.5, 1.5], pbc=True)
    spec = dimensionality_spectrum(atoms, cutoff_table={(6, 6): 1.70})
    assert tuple(spec.d_by_lambda.keys()) == LAMBDAS
    assert 1.75 in spec.d_by_lambda and 2.00 in spec.d_by_lambda
    assert spec.persistence == pytest.approx(8 / 8)


def _mixed_2d_sheet_1d_chain():
    """One cell with a 2D square-grid sheet (ab plane) + a 1D chain along c.

    Geometry (r0 = 1.0 A for self pairs): sheet = 3x3 C grid, spacing
    t = 0.85 < 0.90*r0 (bonded at every grid lambda), z = 0, a = b = 2.55;
    chain = 4 Si atoms along c, spacing 0.85, c = 3.4, sitting at the
    centre of a sheet mesh column (in-plane distance to the sheet
    t/sqrt(2) = 0.601 A) and half a chain spacing above it (min |dz| =
    0.425 A) -> minimum sheet-chain distance sqrt(0.601^2 + 0.425^2) =
    0.736 A.

    Why the cross pair r0 is small: a periodic square sheet tiles the ab
    torus with covering radius t/sqrt(2) and a c-axis chain covers z mod c
    at spacing s, so the minimal sheet-chain distance is bounded by
    sqrt(t^2/2 + s^2/4) <= sqrt(3)/2 * lambda_dstar * r0 < 2.0*r0 for any
    t, s that bond at lambda_dstar — with a single shared r0 the two
    sublattices can NEVER stay unbonded up to lambda = 2.00. The pair
    cutoff table (contract §3.2) is the sanctioned pair-specific mechanism,
    so the cross pair gets r0 = 0.15 A (max cross cutoff 0.30 A at
    lambda = 2.00) against an actual minimum distance of 0.736 A.
    """
    r0, t, s = 1.0, 0.85, 0.85
    sheet = [(i * t, j * t, 0.0) for i in range(3) for j in range(3)]
    chain = [(1.5 * t, 1.5 * t, s / 2 + k * s) for k in range(4)]
    atoms = Atoms("C9Si4", positions=sheet + chain,
                  cell=[3 * t, 3 * t, 4 * s], pbc=True)
    table = {(6, 6): r0, (14, 14): r0, (6, 14): 0.15}
    return atoms, table


def test_mixed_2d_sheet_plus_1d_chain_dim_components():
    atoms, table = _mixed_2d_sheet_1d_chain()
    spec = dimensionality_spectrum(atoms, cutoff_table=table)
    # rank-2 sheet + rank-1 chain coexist in one cell and never merge:
    # d stays 2 at every grid lambda (a merge would rank 2+1 = 3).
    assert all(d == 2 for d in spec.d_by_lambda.values())
    assert spec.d_star == 2
    assert spec.persistence == 1.0
    assert spec.n_components == 2
    assert spec.dim_components == {1: [1, 4], 2: [1, 9]}


def _vdw_bilayer():
    """Two parallel 2x2 square-grid sheets, AA stacked, gap = 1.7*r0.

    In-sheet spacing 0.85*r0 bonds at every grid lambda (d = 2); the
    interlayer distance 1.70 A satisfies 1.50 <= 1.70 < 1.75, so the
    layers first bond at lambda = 1.75 — the 2->3 vdW layered-bulk
    signature the former 1.50 ceiling could not see (c = 2*gap closes the
    stacking through the periodic image, cycle translation (0,0,1)).
    """
    r0, t, gap = 1.0, 0.85, 1.70
    pos = [(i * t, j * t, 0.0) for i in range(2) for j in range(2)]
    pos += [(i * t, j * t, gap) for i in range(2) for j in range(2)]
    atoms = Atoms("C8", positions=pos, cell=[2 * t, 2 * t, 2 * gap], pbc=True)
    return atoms, {(6, 6): r0}


def test_vdw_bilayer_first_3d_at_lambda_175():
    atoms, table = _vdw_bilayer()
    spec = dimensionality_spectrum(atoms, cutoff_table=table)
    # the promised layered-bulk pattern on the extended grid: 2,2,2,2,2,2,3,3
    assert [spec.d_by_lambda[lam] for lam in LAMBDAS] == [2, 2, 2, 2, 2, 2, 3, 3]
    assert min(lam for lam in LAMBDAS if spec.d_by_lambda[lam] == 3) == 1.75
    # d* graph (lambda = 1.00, table mode): two disconnected rank-2 layers
    assert spec.d_star == 2
    assert spec.n_components == 2
    assert spec.dim_components == {2: [2, 8]}
    assert spec.persistence == pytest.approx(6 / 8)


def test_dim_components_single_atom_and_empty_cell():
    atoms = Atoms("Si", positions=[(2.5, 2.5, 2.5)], cell=[5, 5, 5], pbc=True)
    spec = dimensionality_spectrum(atoms)
    assert spec.dim_components == {0: [1, 1]}  # one rank-0 singleton

    periodic = Atoms("C", positions=[(0.75, 0.75, 0.75)],
                     cell=[1.5, 1.5, 1.5], pbc=True)
    spec3 = dimensionality_spectrum(periodic, cutoff_table={(6, 6): 1.70})
    assert spec3.dim_components == {3: [1, 1]}  # self-image bonds span 3D

    # n_atoms = 0 boundary: the d* graph has no atoms -> empty dict
    spec0 = dimensionality_spectrum(Atoms())
    assert spec0.dim_components == {}
    assert spec0.n_components == 0
    assert spec0.d_star == 0


def _invariant_cases():
    mixed, mixed_table = _mixed_2d_sheet_1d_chain()
    bilayer, bilayer_table = _vdw_bilayer()
    graphite = Atoms(
        "C4",
        scaled_positions=[[0.0, 0.0, 0.25], [1 / 3, 2 / 3, 0.25],
                          [0.0, 0.0, 0.75], [2 / 3, 1 / 3, 0.75]],
        cell=[[2.46, 0.0, 0.0], [-1.23, 2.13, 0.0], [0.0, 0.0, 6.70]], pbc=True)
    water = Atoms(
        "OH2",
        positions=[(7.5, 7.5, 7.5), (8.257, 8.086, 7.5), (6.743, 8.086, 7.5)],
        cell=[15.0, 15.0, 15.0], pbc=True)
    chains = Atoms(
        "C8",
        positions=[(2.8, 10.0, 0.0), (2.8, 10.0, 1.4), (2.8, 10.0, 2.8), (2.8, 10.0, 4.2),
                   (0.0, 18.0, 2.8), (1.4, 18.0, 2.8), (2.8, 18.0, 2.8), (4.2, 18.0, 2.8)],
        cell=[5.6, 20.0, 5.6], pbc=True)
    return [
        ("mixed-sheet-chain", mixed, {"cutoff_table": mixed_table}),
        ("vdw-bilayer", bilayer, {"cutoff_table": bilayer_table}),
        ("graphite-ab", graphite, {"cutoff_table": {(6, 6): 2.60}}),
        ("water-default-covalent", water, {}),
        ("nacl-default-covalent", bulk("NaCl", "rocksalt", a=5.64), {}),
        ("two-chains", chains, {"cutoff_table": {(6, 6): 1.70}}),
        ("empty-cell", Atoms(), {}),
    ]


@pytest.mark.parametrize("name,atoms,kwargs", _invariant_cases(),
                         ids=[c[0] for c in _invariant_cases()])
def test_dim_components_invariants(name, atoms, kwargs):
    spec = dimensionality_spectrum(atoms, **kwargs)
    dc = spec.dim_components
    # invariant 1: component counts sum to n_components
    assert sum(cnt for cnt, _ in dc.values()) == spec.n_components
    # invariant 2: atom counts sum to the number of atoms
    assert sum(n for _, n in dc.values()) == len(atoms)
    if len(atoms):
        # invariant 3: max per-component rank equals d_star
        assert max(dc) == spec.d_star
        assert all(r in (0, 1, 2, 3) for r in dc)
    else:
        assert dc == {}
