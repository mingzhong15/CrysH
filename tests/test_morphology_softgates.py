"""Level 2 v3 soft gates — decision-equivalence proofs (task B).

The three v2 hard gates were reformulated as bounded scores with the decision
rule "score >= _SCORE_GATE (0.5)".  Because s(x; c) = x/(x+c) is strictly
increasing with s(c) = 0.5, and the AND gate is reproduced by a min-composite:

    min(s(gap; 5), s(frac; 0.25)) >= 0.5  <=>  gap >= 5 AND frac >= 0.25
    s(gap; 12) >= 0.5                     <=>  gap >= 12
    1/(1 + span/(1.6*med)) >= 0.5         <=>  span <= 1.6*med

These tests verify the equivalence POINTWISE on dense grids (including the
exact boundary values), check the margin definition, and pin the class labels
of the inline acceptance fixtures to the pre-refactor (v2) values so the
routing change is proven label-invariant on real fixtures.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, diamond111, graphene, molecule

import crysh.morphology as morph
from crysh.morphology import (
    MORPHOLOGY_CLASSES,
    morphology,
    morphology_with_diagnostics,
)

# grids avoid landing within ~1e-9 of a boundary except at the exact boundary
# (IEEE rounding may flip a comparison only within ~1 ulp of a boundary —
# documented in morphology.py note 4; no physical input hits that).
GAPS = np.concatenate([np.linspace(0.0, 20.0, 401), [5.0]])
FRACS = np.concatenate([np.linspace(0.0, 1.0, 201), [0.25]])
SLAB_GAPS = np.concatenate([np.linspace(0.0, 30.0, 301), [12.0]])
SPANS = np.concatenate([np.linspace(0.0, 10.0, 201), [1.6 * m for m in (1.0, 2.0, 2.2)]])


# --------------------------------------------------------------------------- #
# pointwise gate equivalence
# --------------------------------------------------------------------------- #

def test_vacuum_gate_grid_equivalence():
    """min-composite score gate == v2 AND gate, pointwise on (gap, frac) grid."""
    n_checked = 0
    for gap in GAPS:
        for frac in FRACS:
            score, s_gap, s_frac = morph._vacuum_score(gap, frac)
            hard = (gap >= morph._VAC_GAP_MIN_A) and (frac >= morph._VAC_FRAC_MIN)
            assert (score >= morph._SCORE_GATE) == hard, (gap, frac, score)
            # component form of the same statement
            assert (s_gap >= morph._SCORE_GATE) == (gap >= morph._VAC_GAP_MIN_A)
            assert (s_frac >= morph._SCORE_GATE) == (frac >= morph._VAC_FRAC_MIN)
            n_checked += 1
    assert n_checked > 80_000


def test_vacuum_gate_boundary_values():
    # exact boundary: both sides of the AND gate sit exactly at 0.5
    score, s_gap, s_frac = morph._vacuum_score(5.0, 0.25)
    assert s_gap == pytest.approx(0.5, abs=0.0)
    assert s_frac == pytest.approx(0.5, abs=0.0)
    assert score >= morph._SCORE_GATE  # v2: 5>=5 AND 0.25>=0.25 -> True
    # one ulp-scale offset away from the boundary the agreement persists
    for d in (1e-9, -1e-9):
        score, _, _ = morph._vacuum_score(5.0 + d, 0.25 + d)
        hard = (5.0 + d >= 5.0) and (0.25 + d >= 0.25)
        assert (score >= morph._SCORE_GATE) == hard
    # far-side sanity
    assert morph._vacuum_score(4.999, 0.9)[0] < morph._SCORE_GATE  # gap fails
    assert morph._vacuum_score(9.0, 0.249)[0] < morph._SCORE_GATE  # frac fails
    assert morph._vacuum_score(9.0, 0.9)[0] > morph._SCORE_GATE


def test_slab_gate_grid_equivalence():
    for gap in SLAB_GAPS:
        score = morph._slab_score(gap)
        assert (score >= morph._SCORE_GATE) == (gap >= morph._SLAB_GAP_MIN_A), gap
    assert morph._slab_score(12.0) == pytest.approx(0.5)
    assert morph._slab_score(0.0) == 0.0


def test_mono_gate_grid_equivalence():
    for med in (1.0, 2.0, 2.2, 2.82):
        for span in SPANS:
            score = morph._mono_score(span, med)
            hard = span <= morph._SPAN_FRAC * med
            assert (score >= morph._SCORE_GATE) == hard, (span, med, score)
        # exact boundary
        edge = morph._SPAN_FRAC * med
        assert morph._mono_score(edge, med) == pytest.approx(0.5)
    # degenerate: no edges -> med=0 guard keeps the v2 semantics
    assert morph._mono_score(0.0, 0.0) >= morph._SCORE_GATE
    assert morph._mono_score(1.0, 0.0) < morph._SCORE_GATE


def test_scores_bounded_and_margin_definition():
    """Scores live in [0, 1]; margin = 2|score-0.5| in [0, 1]."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        gap = rng.uniform(0, 40)
        frac = rng.uniform(0, 1)
        v, _, _ = morph._vacuum_score(gap, frac)
        s = morph._slab_score(gap)
        m = morph._mono_score(rng.uniform(0, 12), rng.uniform(0.5, 3))
        for sc in (v, s, m):
            assert 0.0 <= sc <= 1.0
            marg = morph._score_margin(sc)
            assert 0.0 <= marg <= 1.0
            assert marg == pytest.approx(2 * abs(sc - 0.5))
    assert morph._score_margin(0.5) == 0.0   # on the boundary
    assert morph._score_margin(0.0) == 1.0   # farthest below
    assert morph._score_margin(1.0) == 1.0   # farthest above


# --------------------------------------------------------------------------- #
# class labels unchanged on the inline fixtures (pinned to the v2 values)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def si_slab():
    atoms = diamond111("Si", size=(1, 1, 6), a=5.431, vacuum=15.0)
    atoms.pbc = True
    return atoms


@pytest.fixture(scope="module")
def molecular_crystal():
    import itertools

    mol = molecule("CH4")
    symbols = ["C", "H", "H", "H", "H"] * 8
    positions = []
    for a, b, c in itertools.product([2.5, 7.5], repeat=3):
        m = mol.copy()
        m.positions += [a, b, c]
        positions.extend(m.positions)
    return Atoms(symbols, positions=positions, cell=[10, 10, 10], pbc=True)


def _h2o_box():
    return Atoms("OH2", positions=[[5, 5, 5], [4.2, 5, 5], [5.8, 5, 5]],
                 cell=[10, 10, 10], pbc=True)


def test_softgate_diagnostics_keys():
    d = morphology_with_diagnostics(_h2o_box())
    for key in ("vacuum_score", "slab_score", "mono_score",
                "vacuum_margin", "slab_margin", "mono_margin",
                "per_axis_gaps", "vacuum_normal", "vacuum_miller"):
        assert key in d, key
    assert 0.0 <= d["vacuum_score"] <= 1.0
    assert d["vacuum_margin"] == pytest.approx(2 * abs(d["vacuum_score"] - 0.5))
    assert d["big_vacuum"] == (d["vacuum_score"] >= morph._SCORE_GATE)
    assert isinstance(d["vacuum_normal"], list) and len(d["vacuum_normal"]) == 3


def test_fixture_labels_pinned_to_v2(si_slab, molecular_crystal):
    """Labels identical to the pre-soft-gate (v2) implementation."""
    h2o = _h2o_box()
    graphene_monolayer = graphene(a=2.46, size=(1, 1, 1), vacuum=15.0)
    graphene_monolayer.pbc = True
    # dense ionic bulk
    r = morphology(bulk("NaCl", "rocksalt", a=5.64))
    assert r.class_label == "dense_bulk"
    # dense covalent bulk
    r = morphology(bulk("Si", "diamond", a=5.431))
    assert r.class_label == "dense_bulk"
    # 3D-bonded slab with 15 A vacuum
    r = morphology(si_slab)
    assert r.class_label == "explicit_vacuum_slab"
    assert r.vacuum_gap >= 12.0
    # monolayer with vacuum
    assert morphology(graphene_monolayer).class_label == "intrinsic_2d_like"
    # isolated molecule
    assert morphology(h2o).class_label == "isolated_molecule"
    # molecular crystal
    r = morphology(molecular_crystal)
    assert r.class_label == "molecular_crystal"
    # all labels stay in the frozen class set
    for at in (si_slab, molecular_crystal, h2o, graphene_monolayer):
        assert morphology(at).class_label in MORPHOLOGY_CLASSES


def test_mono_vac_subgate_equivalence():
    """The intrinsic_2d vacuum requirement via s(gap;5) == the v2 gap>=5 test."""
    for gap in GAPS:
        s_gap = morph._gap_score(gap, morph._VAC_GAP_MIN_A)
        assert (s_gap >= morph._SCORE_GATE) == (gap >= morph._MONO_VAC_GAP)
    # _MONO_VAC_GAP is kept equal to the vacuum-gap scale by construction
    assert morph._MONO_VAC_GAP == morph._VAC_GAP_MIN_A


def test_empty_atoms_softgate_defaults():
    d = morphology_with_diagnostics(Atoms())
    assert d["class_label"] == "ambiguous"
    assert d["vacuum_score"] == 0.0 and d["slab_score"] == 0.0
    assert d["vacuum_margin"] == 1.0 and d["slab_margin"] == 1.0
    assert d["per_axis_gaps"] == [0.0, 0.0, 0.0]
    assert d["vacuum_miller"] is None
