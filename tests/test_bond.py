"""Unit tests for ckt.bond — periodic bond graph construction.

Fixtures are constructed inline (no dependency on the controls subtask).
"""
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.data import covalent_radii

from crysh import bond
from crysh.bond import BondGraph, build_bond_graph, default_covalent_table


def test_default_covalent_table_sorted_keys_and_sums():
    atoms = Atoms("CHO", positions=[(0, 0, 0), (0, 0, 1), (0, 0, 2)])
    t = default_covalent_table(atoms)
    assert set(t) == {(1, 1), (1, 6), (1, 8), (6, 6), (6, 8), (8, 8)}
    assert t[(1, 6)] == pytest.approx(covalent_radii[1] + covalent_radii[6])
    assert t[(6, 6)] == pytest.approx(2 * covalent_radii[6])
    assert all(v > 0 for v in t.values())


def test_zero_radius_element_fallback(monkeypatch):
    radii = list(covalent_radii)
    radii[10] = 0.0  # pretend Ne has no tabulated covalent radius
    monkeypatch.setattr(bond, "covalent_radii", radii)
    atoms = Atoms("Ne2", positions=[(0, 0, 0), (3, 0, 0)], cell=[10, 10, 10])
    t = default_covalent_table(atoms)
    assert t[(10, 10)] > 0
    assert t[(10, 10)] == pytest.approx(2 * bond._COV_MEAN_FALLBACK)


def test_build_bond_graph_diamond_si_fields_and_dedupe():
    atoms = bulk("Si", "diamond", a=5.431)  # 2-atom primitive cell
    g = build_bond_graph(atoms, cutoff_table={(14, 14): 2.70}, lam=1.0)
    assert isinstance(g, BondGraph)
    assert g.n_atoms == 2
    assert g.cutoff_mode == "table"
    assert g.lam == 1.0
    assert g.pair_r0[(14, 14)] == 2.70
    assert g.i.dtype == np.int32 and g.j.dtype == np.int32
    assert g.S.dtype == np.int32 and g.S.shape[1] == 3
    assert g.d.dtype == np.float64
    # v1.1 (integrator): bidirectional adjacency — 4 unique periodic bonds,
    # each stored in BOTH directions (ASE-compatible) = 8 directed edges.
    assert len(g.i) == 8
    assert not ((g.i == g.j) & np.all(g.S == 0, axis=1)).any()
    assert np.allclose(g.d, 2.351, atol=1e-3)
    keys = np.stack([g.i, g.j, g.S[:, 0], g.S[:, 1], g.S[:, 2]], axis=1)
    assert len(np.unique(keys, axis=0)) == len(g.i)
    # v1.1: every edge has its reverse (j, i, -S) present
    rev = set(map(tuple, np.stack([g.j, g.i, -g.S[:, 0], -g.S[:, 1], -g.S[:, 2]], axis=1).tolist()))
    fwd = set(map(tuple, keys.tolist()))
    assert fwd == rev


def test_build_bond_graph_covalent_default_no_bonds_for_si():
    # Default covalent radii: r0(Si,Si)=2.22 A < 2.351 A bond -> no edges at lam=1.0.
    # (Documents why fixture tests pass explicit tables for the d(lambda) patterns.)
    atoms = bulk("Si", "diamond", a=5.431)
    g = build_bond_graph(atoms, lam=1.0)
    assert g.cutoff_mode == "covalent"
    assert g.pair_r0[(14, 14)] == pytest.approx(2 * covalent_radii[14])
    assert len(g.i) == 0


def test_build_bond_graph_table_fallback_for_missing_pairs():
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    g = build_bond_graph(atoms, cutoff_table={(11, 17): 3.20}, lam=0.9)
    # missing (11,11)/(17,17) pairs fall back to covalent radii
    assert g.pair_r0[(11, 11)] == pytest.approx(2 * covalent_radii[11])
    assert g.pair_r0[(17, 17)] == pytest.approx(2 * covalent_radii[17])
    assert g.pair_r0[(11, 17)] == pytest.approx(3.20)
    # rocksalt 2-atom primitive cell: 6 unique Na-Cl bonds at lam=0.9,
    # stored bidirectionally (v1.1) = 12 directed edges
    assert len(g.i) == 12
    z = atoms.numbers
    for e in range(len(g.i)):
        a, b = int(z[g.i[e]]), int(z[g.j[e]])
        key = (a, b) if a <= b else (b, a)
        assert g.d[e] < g.lam * g.pair_r0[key]


def test_build_bond_graph_periodic_self_edges():
    # atom bonded to its own periodic images -> self-edges with S != 0
    atoms = Atoms("C", positions=[(0.75, 0.75, 0.75)], cell=[1.5, 1.5, 1.5], pbc=True)
    g = build_bond_graph(atoms, cutoff_table={(6, 6): 1.70}, lam=1.0)
    assert len(g.i) == 6
    assert (g.i == g.j).all()
    assert not np.any(np.all(g.S == 0, axis=1))  # every self-edge is a periodic image
    assert np.allclose(g.d, 1.5)


def test_build_bond_graph_home_cell_self_loop_excluded():
    # two coincident atoms: one (i!=j, S=0) bond stored in both directions
    # (v1.1 bidirectional); i==j,S=0 never emitted
    atoms = Atoms("C2", positions=[(0, 0, 0), (0, 0, 0)], cell=[10, 10, 10], pbc=True)
    g = build_bond_graph(atoms, cutoff_table={(6, 6): 1.7})
    assert len(g.i) == 2
    assert g.i[0] != g.j[0]
    assert np.all(g.S[0] == 0)
    assert g.d[0] == pytest.approx(0.0)
    assert sorted(zip(g.i.tolist(), g.j.tolist())) == [(0, 1), (1, 0)]


def test_build_bond_graph_rejects_invalid_inputs():
    atoms = Atoms("C", positions=[(0, 0, 0)])
    with pytest.raises(ValueError):
        build_bond_graph(atoms, lam=0.0)
    with pytest.raises(ValueError):
        build_bond_graph(atoms, lam=-1.0)
    with pytest.raises(TypeError):
        build_bond_graph("not atoms")
    with pytest.raises(ValueError):
        build_bond_graph(atoms, cutoff_table={(6, 6): 0.0})


def test_build_bond_graph_unsorted_table_keys_normalized():
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    g = build_bond_graph(atoms, cutoff_table={(17, 11): 3.20}, lam=1.0)
    assert g.pair_r0[(11, 17)] == pytest.approx(3.20)
    assert g.cutoff_mode == "table"
