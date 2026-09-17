"""Level 2 morphology tests — inline fixtures (contracts.md §6: L2 builds its own
fixtures; level-1 may not be merged / calibrated yet).

Acceptance fixtures (task brief):
- Si slab (6 layers + 15 Å vacuum) -> explicit_vacuum_slab or surface_like
- graphene monolayer + vacuum     -> intrinsic_2d_like
- H2O in a box                    -> isolated_molecule
- NaCl rocksalt                   -> dense_bulk
- hand-made sparse framework      -> porous_candidate
plus molecular crystal / chain solid / isolated chain / layered bulk / dense Si.
"""

import itertools

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, diamond111, graphene, molecule

from crysh.morphology import (
    MORPHOLOGY_CLASSES,
    MorphologyResult,
    morphology,
    morphology_with_diagnostics,
)

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def si_slab():
    """Si(111), 6 atomic layers, 15 A vacuum (3D-bonded slab: d_star == 2)."""
    atoms = diamond111("Si", size=(1, 1, 6), a=5.431, vacuum=15.0)
    atoms.pbc = True
    return atoms


@pytest.fixture(scope="module")
def graphene_monolayer():
    atoms = graphene(a=2.46, size=(1, 1, 1), vacuum=15.0)
    atoms.pbc = True
    return atoms


@pytest.fixture(scope="module")
def h2o_box():
    return Atoms("OH2", positions=[[5, 5, 5], [4.2, 5, 5], [5.8, 5, 5]],
                 cell=[10, 10, 10], pbc=True)


@pytest.fixture(scope="module")
def nacl():
    return bulk("NaCl", "rocksalt", a=5.64)


@pytest.fixture(scope="module")
def sparse_framework():
    """Si diamond 2x2x2 supercell (primitive cell) minus 3 atoms on a
    (i+j+k) % 4 == 0 pattern: still a 3D-periodic network, nu ~ 4.3 > 4,
    no big vacuum (documented pilot porous candidate)."""
    atoms = bulk("Si", "diamond", a=5.431) * (2, 2, 2)
    sc = atoms.get_scaled_positions()
    sub = np.floor(sc * 2).astype(int)
    pat = (sub[:, 0] + sub[:, 1] + sub[:, 2]) % 4
    del atoms[pat == 0]
    return atoms


@pytest.fixture(scope="module")
def molecular_crystal():
    """8 methane molecules on a 2x2x2 grid in a 10 A box (vdW packing)."""
    mol = molecule("CH4")
    symbols = ["C", "H", "H", "H", "H"] * 8
    positions = []
    for a, b, c in itertools.product([2.5, 7.5], repeat=3):
        m = mol.copy()
        m.positions += [a, b, c]
        positions.extend(m.positions)
    return Atoms(symbols, positions=positions, cell=[10, 10, 10], pbc=True)


def _zigzag_chain():
    """8-carbon zigzag chain along x (bond 1.54 A, period 8*0.885)."""
    xs = np.arange(8) * 0.885
    zs = np.where(np.arange(8) % 2 == 0, 0.63, -0.63)
    return Atoms("C8", positions=np.column_stack([xs, np.full(8, 2.0), zs]),
                 cell=[7.08, 4, 4], pbc=True)


@pytest.fixture(scope="module")
def chain_solid():
    return _zigzag_chain()


@pytest.fixture(scope="module")
def isolated_chain():
    atoms = _zigzag_chain()
    atoms.set_cell([7.08, 12, 12])
    return atoms


@pytest.fixture(scope="module")
def layered_bulk():
    """3 graphene sheets bonded at 2.0 A interlayer spacing (d(l=1.0)=2,
    d(l=1.2)=3 -> the 2->3 signature of layered bulk; c=6 has no big vacuum)."""
    g1 = graphene(a=2.46, size=(1, 1, 1), vacuum=0.0)
    positions = []
    for z in (0.0, 2.0, 4.0):
        positions.append([g1.positions[0, 0], g1.positions[0, 1], z])
        positions.append([g1.positions[1, 0], g1.positions[1, 1], z])
    return Atoms("C6", positions=positions,
                 cell=[[2.46, 0, 0], [-1.23, 2.13, 0], [0, 0, 6.0]], pbc=True)


@pytest.fixture(scope="module")
def si_bulk():
    return bulk("Si", "diamond", a=5.431)


# --------------------------------------------------------------------------- #
# contract shape
# --------------------------------------------------------------------------- #

def test_morphology_classes_frozen():
    assert MORPHOLOGY_CLASSES == [
        "dense_bulk", "layered_bulk", "explicit_vacuum_slab", "surface_like",
        "intrinsic_2d_like", "chain_solid", "isolated_chain", "molecular_crystal",
        "isolated_molecule", "porous_candidate", "ambiguous",
    ]


def test_morphology_result_fields_frozen():
    fields = [f.name for f in __import__("dataclasses").fields(MorphologyResult)]
    assert fields == ["vacuum_gap", "vacuum_fraction", "surface_score",
                      "porous_candidate", "n_components", "f_max", "class_label"]


def test_labels_valid_and_score_in_range(h2o_box):
    r = morphology(h2o_box)
    assert r.class_label in MORPHOLOGY_CLASSES
    assert 0.0 <= r.surface_score <= 1.0
    assert 0.0 <= r.vacuum_fraction <= 1.0


# --------------------------------------------------------------------------- #
# acceptance fixtures
# --------------------------------------------------------------------------- #

def test_si_slab(si_slab):
    r = morphology(si_slab)
    assert r.class_label in {"explicit_vacuum_slab", "surface_like"}
    assert r.vacuum_gap >= 12.0
    assert r.porous_candidate is False


def test_graphene_monolayer(graphene_monolayer):
    r = morphology(graphene_monolayer)
    d = morphology_with_diagnostics(graphene_monolayer)
    assert r.class_label == "intrinsic_2d_like"
    # monolayer: vacuum but no CN gradient -> score below the 0.6 surface gate
    assert r.surface_score < 0.6
    assert d["d_star"] == 2
    assert d["n_layers"] <= 2


def test_h2o_isolated_molecule(h2o_box):
    r = morphology(h2o_box)
    assert r.class_label == "isolated_molecule"
    assert r.n_components == 1
    assert r.vacuum_gap > 5.0


def test_nacl_dense_bulk(nacl):
    r = morphology(nacl)
    assert r.class_label == "dense_bulk"
    assert r.porous_candidate is False


def test_sparse_framework_porous(sparse_framework):
    r = morphology(sparse_framework)
    d = morphology_with_diagnostics(sparse_framework)
    assert r.porous_candidate is True
    assert r.class_label == "porous_candidate"
    # level-1's pure-Cordero table underbonds Si-Si (2.35 vs 2.22 A), so the raw
    # d_star is 0 and the documented Cordero rescue gives the effective d = 3.
    assert d["eff_d_star"] == 3
    assert d["volume_norm"] > 4.0
    assert d["big_vacuum"] is False


# --------------------------------------------------------------------------- #
# the other classes
# --------------------------------------------------------------------------- #

def test_molecular_crystal(molecular_crystal):
    r = morphology(molecular_crystal)
    assert r.class_label == "molecular_crystal"
    assert r.n_components > 1
    assert r.f_max < 0.5


def test_chain_solid(chain_solid):
    r = morphology(chain_solid)
    assert r.class_label == "chain_solid"
    assert r.vacuum_gap < 5.0


def test_isolated_chain(isolated_chain):
    r = morphology(isolated_chain)
    assert r.class_label == "isolated_chain"
    assert r.vacuum_gap > 5.0


def test_layered_bulk(layered_bulk):
    r = morphology(layered_bulk)
    d = morphology_with_diagnostics(layered_bulk)
    assert r.class_label == "layered_bulk"
    assert d["d_star"] == 2
    assert max(d["d_by_lambda"].values()) >= 3  # 2 -> 3 interlayer signature


def test_si_bulk_dense(si_bulk):
    r = morphology(si_bulk)
    assert r.class_label == "dense_bulk"


# --------------------------------------------------------------------------- #
# caller-supplied graph / spectrum / coord (duck-typed, contracts §3.5)
# --------------------------------------------------------------------------- #

def test_caller_supplied_graph_spectrum_consistent(h2o_box):
    from crysh.morphology import _build_graph, _spectrum_fallback  # internal helpers

    g = _build_graph(h2o_box, 1.0)
    spec = _spectrum_fallback(h2o_box)
    r0 = morphology(h2o_box)
    r1 = morphology(h2o_box, graph=g, spectrum=spec)
    assert r1.class_label == r0.class_label == "isolated_molecule"
    assert r1.n_components == r0.n_components


def test_coord_input_accepted(si_slab):
    class FakeCoord:
        def __init__(self, cn):
            self.cn = cn

    cn = np.array([3, 3, 3, 4, 4, 4], dtype=int)[: len(si_slab)]
    r = morphology(si_slab, coord=FakeCoord(cn))
    assert r.class_label in {"explicit_vacuum_slab", "surface_like"}
    assert 0.0 <= r.surface_score <= 1.0


def test_diagnostics_keys(h2o_box):
    d = morphology_with_diagnostics(h2o_box)
    for key in ["structure_id" if False else "vacuum_gap", "vacuum_fraction",
                "surface_score", "porous_candidate", "n_components", "f_max",
                "class_label", "d_star", "volume_norm", "n_layers", "big_vacuum",
                "vacuum_axis", "cn_std", "spectrum_source"]:
        assert key in d
    assert d["class_label"] in MORPHOLOGY_CLASSES


def test_empty_atoms():
    r = morphology(Atoms())
    assert r.class_label == "ambiguous"
    assert r.n_components == 0
