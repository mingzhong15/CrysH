"""Tests for ckt.controls — constructed ground-truth structures.

Run from KT root:
    cd <KT> && $CKT_PY -m pytest controls/tests -q
"""

import re

import numpy as np
import pytest

from crysh import controls as C


@pytest.fixture(scope="module")
def records():
    return C.build_all()


def _ids(records):
    return [sid for sid, _, _ in records]


# ---------------------------------------------------------------------------
# build_all(): shape, id scheme, idempotency, physics sanity
# ---------------------------------------------------------------------------
def test_build_all_count_in_range(records):
    assert 40 <= len(records) <= 60


def test_ids_unique_and_wellformed(records):
    ids = _ids(records)
    assert len(set(ids)) == len(ids)
    pat = re.compile(r"^control_(" + "|".join(C.FAMILIES) + r")_\d{2}$")
    for sid in ids:
        assert pat.match(sid), f"bad structure_id: {sid}"


def test_all_families_present_with_min_count(records):
    counts = {}
    for sid, atoms, e in records:
        counts[e["family"]] = counts.get(e["family"], 0) + 1
        assert sid.startswith(f"control_{e['family']}_")
    for fam in C.FAMILIES:
        assert fam in counts, f"family {fam} missing"
        assert counts[fam] >= 3, f"family {fam} has only {counts[fam]} variants"


def test_build_all_idempotent():
    r1 = C.build_all()
    r2 = C.build_all()
    assert [x[0] for x in r1] == [x[0] for x in r2]
    for (sid1, a1, e1), (sid2, a2, e2) in zip(r1, r2):
        assert sid1 == sid2
        assert np.allclose(a1.positions, a2.positions, atol=1e-10)
        assert np.allclose(a1.cell[:], a2.cell[:], atol=1e-10)
        assert a1.numbers.tolist() == a2.numbers.tolist()
        assert e1 == e2


def test_physics_sanity(records):
    from ase.neighborlist import neighbor_list

    for sid, atoms, e in records:
        # no overlapping atoms: no periodic pair closer than 0.5 A
        d = neighbor_list("d", atoms, 0.5)
        assert len(d) == 0, f"{sid}: overlapping atoms"
        # sane cell
        assert atoms.get_volume() > 0
        # rank of the spectrum can never exceed 3
        assert all(0 <= d <= 3 for d in e["expected_short_lambda_dim"].values())
        assert e["expected_d_star"] in (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# labels.yaml: completeness + frozen-class consistency
# ---------------------------------------------------------------------------
def test_labels_complete_and_valid(records):
    labels = C.load_labels()
    structs = labels["structures"]
    assert set(structs.keys()) == set(_ids(records))
    for sid, atoms, e in records:
        lab = structs[sid]
        for key in ("family", "expected_d_star", "expected_morphology_class",
                    "expected_short_lambda_dim", "notes"):
            assert key in lab, f"{sid}: missing required key {key}"
        assert lab["family"] == e["family"]
        assert lab["expected_d_star"] in (0, 1, 2, 3)
        assert lab["expected_morphology_class"] in C.MORPHOLOGY_CLASSES
        assert set(lab["expected_short_lambda_dim"]) == {
            f"{lam:.2f}" for lam in C.LAMBDAS}
        assert all(0 <= d <= 3 for d in lab["expected_short_lambda_dim"].values())
        assert isinstance(lab["notes"], str) and lab["notes"]


def test_labels_definitions_present():
    labels = C.load_labels()
    meta = labels["meta"]
    for key in ("version", "families", "definitions"):
        assert key in meta
    assert meta["families"] == C.FAMILIES
    assert meta["n_structures_hint"] == len(labels["structures"])


def test_labels_match_recomputed_references():
    # the checked-in labels.yaml must equal a fresh computation
    labels = C.load_labels()
    for sid, atoms, e in C.build_all():
        lab = labels["structures"][sid]
        assert lab == e, f"{sid}: labels.yaml out of sync with build_all()"
        assert lab["expected_short_lambda_dim"] == C._ref_dim_spectrum(atoms)
        assert lab["expected_validity"] == C._validity_reference(atoms)
        assert abs(lab["expected_vacuum_gap"]
                   - round(C._vacuum_gap_estimate(atoms), 3)) < 1e-9


def test_vasp_assets_match_labels():
    import ase.io

    labels = C.load_labels()
    files = sorted(p.name for p in C.STRUCTURES_DIR.glob("*.vasp"))
    assert files == sorted(f"{sid}.vasp" for sid in labels["structures"])
    for sid, atoms, e in C.build_all():
        path = C.STRUCTURES_DIR / f"{sid}.vasp"
        at = ase.io.read(path)
        assert len(at) == len(atoms)
        assert np.allclose(at.positions, atoms.positions, atol=1e-5)
        assert np.allclose(at.cell[:], atoms.cell[:], atol=1e-5)


# ---------------------------------------------------------------------------
# analytic spot-checks (hand-derived ground truth of key spectra)
# ---------------------------------------------------------------------------
def _spec(records, sid):
    for s, a, e in records:
        if s == sid:
            return e["expected_short_lambda_dim"]
    raise KeyError(sid)


def test_known_spectra(records):
    # graphene: C-C 1.42 A bonds at lambda>=1.0 -> 2D
    assert list(_spec(records, "control_intrinsic_2d_like_01").values()) == \
        [0, 2, 2, 2, 2, 2]
    # diamond Si: Cordero under-bonds until lambda=1.1, then 3D
    assert list(_spec(records, "control_dense_bulk_01").values()) == \
        [0, 0, 3, 3, 3, 3]
    # fcc Al: metallic, bonds only at lambda>=1.2
    assert list(_spec(records, "control_dense_bulk_02").values()) == \
        [0, 0, 0, 3, 3, 3]
    # isolated H2O molecule in a box: never connects -> 0D everywhere
    assert list(_spec(records, "control_isolated_molecule_02").values()) == \
        [0, 0, 0, 0, 0, 0]
    # periodically wrapped C chain: 1D at every lambda
    assert list(_spec(records, "control_isolated_chain_04").values()) == \
        [1, 1, 1, 1, 1, 1]
    # finite Se chain: 0D at every lambda (finite cluster per contract)
    assert list(_spec(records, "control_isolated_chain_01").values()) == \
        [0, 0, 0, 0, 0, 0]
    # Se chain solid: 1D, interchain bonds at lambda=1.5 -> 3D
    assert list(_spec(records, "control_chain_solid_01").values()) == \
        [0, 1, 1, 1, 1, 3]


def test_known_cn_and_flags(records):
    def lab(sid):
        for s, a, e in records:
            if s == sid:
                return e
        raise KeyError(sid)

    # high-coordination family is CN=12
    for sid in ("control_high_coordination_01", "control_high_coordination_02",
                "control_high_coordination_03"):
        assert lab(sid)["expected_mean_cn"] == 12.0
    # diamond Si: uniform CN=4, no L0 flags
    si = lab("control_dense_bulk_01")
    assert si["expected_cn_hist"] == {"4": 8}
    assert si["expected_validity"]["overlap_flag"] is False
    assert si["expected_validity"]["pathological_cell_flag"] is False
    # CO2: short double bond trips the frozen q_c threshold (documented)
    co2 = lab("control_isolated_molecule_03")
    assert co2["expected_validity"]["overlap_flag"] is True
    assert co2["expected_validity"]["q_min"] < C.Q_C
    # boxed species: extreme volume expected
    h2o = lab("control_isolated_molecule_02")
    assert h2o["expected_validity"]["extreme_volume_flag"] is True
    # slab: CN gradient 1 (surface) vs 4 (interior)
    slab = lab("control_explicit_vacuum_slab_01")
    assert slab["expected_cn_hist"] == {"1": 8, "4": 24}
    assert slab["expected_validity"]["overlap_flag"] is False


def test_morphology_labels_frozen_classes(records):
    # every expected morphology class must be in the frozen §3.5 list
    for sid, atoms, e in records:
        assert e["expected_morphology_class"] in C.MORPHOLOGY_CLASSES


def test_integer_rank_greedy_matches_real_rank():
    # regression: naive greedy can overestimate; fixpoint version must not
    cases = [
        [[2, 0, 0], [3, 0, 0]],           # rank 1
        [[2, 0, 0], [0, 2, 0], [1, 1, 0], [0, 0, 2]],  # rank 3
        [[1, 0, 0], [0, 1, 0]],           # rank 2
        [[6, 4, 2], [3, 2, 1]],           # rank 1
        [],                                # rank 0
    ]
    expected = [1, 3, 2, 1, 0]
    for vecs, want in zip(cases, expected):
        assert C._integer_rank(np.array(vecs, dtype=int)) == want
        if vecs:
            assert int(np.linalg.matrix_rank(np.array(vecs, dtype=float))) == want
