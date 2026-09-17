"""Level 2 v3 vacuum direction search (task D) + puckered monolayer (task E).

D: the vacuum search candidates were extended from the 3 lattice axes to
axes + low-index plane normals G = h·b1 + k·b2 + l·b3 (h,k,l ∈ {−1,0,1}).
The headline case is a triclinic cell hosting a 2D sheet whose vacuum normal
is not parallel to any reciprocal axis: the sheet's in-plane lattice vectors
are (a1, a2+a3), so the normal is the (1,1,1) plane normal.  The legacy
3-axis search smears the sheet over every fractional coordinate and misses
the vacuum entirely (max axis gap ≈ 2.2 Å); the (1,1,1) candidate sees the
full 9.8 Å layer spacing.  Orthogonal fixtures (slab / monolayer / dense)
keep their axis result exactly (axes are scanned first, strict >).

E: a phosphorene-type puckered monolayer (rectangular cell, 4 P atoms in two
sub-planes, all in-plane bonds 2.2 Å, sub-plane separation = 0.72 × bond,
18 Å of c vacuum) must classify as intrinsic_2d_like through the span-based
monolayer criterion, while the AB-stacked 2-layer graphite stays
layered_bulk (the regression that motivated the span criterion in v2).
"""

import numpy as np
import pytest
from ase import Atoms
from ase.build import diamond111, graphene

import crysh.morphology as morph
from crysh.morphology import morphology, morphology_with_diagnostics

# --------------------------------------------------------------------------- #
# fixture builders
# --------------------------------------------------------------------------- #

def tilted_sheet():
    """Square Si sheet (bond 2.2 A) in a triclinic cell whose (1,1,1) plane
    normal is the sheet normal; 12 A layer spacing along that normal.

    Cell: a1 = 2q·e1 + D·n, a2 = 2q·e2 + D·n, a3 = −2q·(e1+e2) + D·n with the
    orthonormal in-plane basis (e1, e2) and n = (1,1,1)/√3 — every cell vector
    has an in-plane component, so no reciprocal axis is parallel to n.
    """
    q, D = 2.2, 12.0
    e1 = np.array([1, -1, 0]) / np.sqrt(2)
    e2 = np.array([1, 1, -2]) / np.sqrt(6)
    n = np.array([1, 1, 1]) / np.sqrt(3)
    m = 2
    cell = np.array([q * m * e1 + D * n,
                     q * m * e2 + D * n,
                     -q * m * (e1 + e2) + D * n])
    inv = np.linalg.inv(cell)
    pos = []
    for k in range(-2, 5):
        for i in range(-6, 7):
            for j in range(-6, 7):
                r = i * q * e1 + j * q * e2 + k * D * n
                s = r @ inv
                if np.all(s >= -1e-9) and np.all(s < 1 - 1e-9):
                    pos.append(r)
    return Atoms(f"Si{len(pos)}", positions=pos, cell=cell, pbc=True)


def puckered_monolayer():
    """Phosphorene-type puckered P monolayer (buckled rectangular honeycomb).

    4 P atoms in two sub-planes at z = ±0.792 Å (separation 1.584 Å =
    0.72 × bond 2.2 Å, inside the 0.7–1.0 × bond band of real puckered
    monolayers); every bond connects the sub-planes and is exactly 2.2 Å;
    c = 18 Å vacuum.  Rectangular cell a = √3·ℓ, b = 3ℓ with the in-plane
    bond projection ℓ = sqrt(2.2² − 1.584²).
    """
    bond = 2.2
    t = 0.72 * bond          # sub-plane separation (thickness)
    delt = t / 2.0
    ell = np.sqrt(bond ** 2 - t ** 2)
    a, b, c = np.sqrt(3) * ell, 3 * ell, 18.0
    pos = [
        [0.0, 0.0, +delt],
        [0.0, ell, -delt],
        [np.sqrt(3) * ell / 2, 1.5 * ell, +delt],
        [np.sqrt(3) * ell / 2, 2.5 * ell, -delt],
    ]
    return Atoms("P4", positions=pos, cell=[a, b, c], pbc=True)


def ab_bilayer_graphite():
    """2-layer AB (Bernal) graphite, c = 2 x 3.35 A (no added vacuum)."""
    ag = 2.46
    a1 = np.array([ag, 0.0, 0.0])
    a2 = np.array([ag / 2, ag * np.sqrt(3) / 2, 0.0])
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([ag / 2, ag / (2 * np.sqrt(3)), 0.0])
    shift = (a1 + a2) / 3.0
    z2 = 3.35
    return Atoms("C4",
                 positions=[A, B, A + shift + [0, 0, z2], B + shift + [0, 0, z2]],
                 cell=[a1, a2, [0.0, 0.0, 2 * z2]], pbc=True)


# --------------------------------------------------------------------------- #
# D — tilted sheet detection + no-regression on orthogonal fixtures
# --------------------------------------------------------------------------- #

def test_tilted_sheet_detected_by_plane_normal():
    d = morphology_with_diagnostics(tilted_sheet())
    # the (1,1,1) plane-normal candidate wins: big vacuum is now detected
    assert d["big_vacuum"] is True
    assert d["vacuum_axis"] >= 3
    assert d["vacuum_miller"] == [1, 1, 1]
    assert d["vacuum_gap"] == pytest.approx(12.0 - 2 * 1.11, abs=0.05)
    assert d["vacuum_fraction"] > 0.25
    # the winning unit normal is (1,1,1)/sqrt(3)
    n123 = np.array([1, 1, 1]) / np.sqrt(3)
    assert np.allclose(np.asarray(d["vacuum_normal"]), n123, atol=1e-9)
    # single atomic plane along the normal -> monolayer-thin span
    assert d["span"] == pytest.approx(0.0, abs=1e-9)
    assert d["mono_score"] >= morph._SCORE_GATE
    # 2D-bonded sheet with vacuum -> intrinsic_2d_like
    assert d["eff_d_star"] == 2
    assert d["class_label"] == "intrinsic_2d_like"


def test_tilted_sheet_legacy_axis_search_misses():
    """The v2 3-axis search cannot see the tilted-sheet vacuum (the fix's target).

    `per_axis_gaps` are exactly the legacy candidates' results: all three fall
    far below the 5 Å bar, so the legacy AND gate never fired on this cell.
    """
    d = morphology_with_diagnostics(tilted_sheet())
    assert all(g < morph._VAC_GAP_MIN_A for g in d["per_axis_gaps"])
    # and disabling the extension reproduces the v2 outcome verbatim
    at = tilted_sheet()
    try:
        morph._USE_PLANE_NORMALS = False
        legacy = morphology_with_diagnostics(at)
    finally:
        morph._USE_PLANE_NORMALS = True
    assert legacy["big_vacuum"] is False
    assert legacy["class_label"] == "layered_bulk"  # the v2 misroute (documented)
    assert legacy["vacuum_gap"] < morph._VAC_GAP_MIN_A


def test_plane_normal_candidate_set():
    """13 unique directions, deterministic order, correct interplanar metric."""
    at = tilted_sheet()
    cands = morph._plane_normal_candidates(np.asarray(at.cell))
    assert len(cands) == 13
    millers = [tuple(h[2]) for h in cands]
    assert len(set(millers)) == 13  # no ± duplicates
    # cubic sanity: d_hkl = 1/|G| reproduces the textbook spacings
    cubic = np.diag([4.0, 4.0, 4.0])
    cc = {tuple(h[2]): h[1] for h in morph._plane_normal_candidates(cubic)}
    assert cc[(1, 0, 0)] == pytest.approx(4.0)
    assert cc[(1, 1, 0)] == pytest.approx(4.0 / np.sqrt(2))
    assert cc[(1, 1, 1)] == pytest.approx(4.0 / np.sqrt(3))
    # degenerate cell -> no candidates (axes-only fallback)
    assert morph._plane_normal_candidates(np.zeros((3, 3))) == []


def test_orthogonal_fixtures_keep_axis_result():
    """Orthogonal cells: the axis candidates win (scanned first, strict >),
    so vacuum_gap/axis/class are bit-identical to v2."""
    si_slab = diamond111("Si", size=(1, 1, 6), a=5.431, vacuum=15.0)
    si_slab.pbc = True
    mono = graphene(a=2.46, size=(1, 1, 1), vacuum=15.0)
    mono.pbc = True
    h2o = Atoms("OH2", positions=[[5, 5, 5], [4.2, 5, 5], [5.8, 5, 5]],
                cell=[10, 10, 10], pbc=True)
    for at, label, axis in ((si_slab, "explicit_vacuum_slab", 2),
                            (mono, "intrinsic_2d_like", 2),
                            (h2o, "isolated_molecule", None)):
        d = morphology_with_diagnostics(at)
        assert d["class_label"] == label
        assert d["vacuum_axis"] <= 2  # an axis won
        assert d["vacuum_miller"] is None
        assert d["per_axis_gaps"][d["vacuum_axis"]] == pytest.approx(
            d["vacuum_gap"], abs=1e-12)
        if axis is not None:
            assert d["vacuum_axis"] == axis


def test_per_axis_gaps_diagnostic_shape():
    d = morphology_with_diagnostics(tilted_sheet())
    assert len(d["per_axis_gaps"]) == 3
    assert all(isinstance(g, float) for g in d["per_axis_gaps"])


# --------------------------------------------------------------------------- #
# E — puckered monolayer + AB graphite regression
# --------------------------------------------------------------------------- #

def test_puckered_monolayer_intrinsic_2d():
    at = puckered_monolayer()
    d = morphology_with_diagnostics(at)
    # geometry sanity: two sub-planes, thickness = 0.72 x bond
    zs = np.sort(at.positions[:, 2])
    assert zs.max() - zs.min() == pytest.approx(0.72 * 2.2, abs=1e-9)
    # in-plane periodic bonding at the spectrum level (2D at the graph lambda)
    assert d["eff_d_star"] == 2
    assert max(d["d_by_lambda"].values()) == 2  # no 2->3 interlayer signature
    # big vacuum along c (orthogonal cell -> axis wins)
    assert d["big_vacuum"] is True
    assert d["vacuum_axis"] == 2
    # the two sub-planes resolve as 2 atomic layers
    assert d["n_layers"] == 2
    # span = sub-plane separation << 1.6 x median bond -> monolayer
    assert d["span"] == pytest.approx(0.72 * 2.2, abs=1e-6)
    assert d["mono_score"] >= morph._SCORE_GATE
    assert d["class_label"] == "intrinsic_2d_like"
    # and morphology() agrees
    assert morphology(at).class_label == "intrinsic_2d_like"


def test_ab_bilayer_graphite_stays_layered_bulk():
    at = ab_bilayer_graphite()
    d = morphology_with_diagnostics(at)
    # interlayer 3.35 A is never bonded (C-C cutoff <= 3.04 A at lambda=2.0)
    assert max(d["d_by_lambda"].values()) == 2
    # span = 3.35 A > 1.6 x 1.42 A median bond -> NOT a monolayer
    assert d["span"] == pytest.approx(3.35, abs=1e-6)
    assert d["mono_score"] < morph._SCORE_GATE
    # c-axis gap 3.35 - 1.52 = 1.83 A: no big vacuum, no slab gate
    assert d["big_vacuum"] is False
    assert d["slab_score"] < morph._SCORE_GATE
    assert d["class_label"] == "layered_bulk"
    assert morphology(at).class_label == "layered_bulk"
