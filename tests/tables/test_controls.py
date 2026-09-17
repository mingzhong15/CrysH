"""controls 的测试：ground-truth 结构与冻结 labels.yaml 的一致性。

本档（tests/tables）**没有 pymatgen**：CN 参考方法一律显式选无依赖的 "shell"，
与 CN 无关的字段逐字段核对 labels.yaml。完整复现 CrystalNN 口径 labels.yaml 的
测试在 tests/research/test_controls_labels.py（需 crysh[research]）。
"""

import re

import numpy as np
import pytest

from crysh import controls as C


@pytest.fixture(scope="module")
def records():
    # 本档没有 pymatgen，CN 参考方法必须**显式**选无依赖的壳层法。默认的
    # "crystalnn"（= 冻结 labels.yaml 的口径）在这里会直接 ImportError —— 这是
    # 有意的：build_all() 不再按环境隐式换方法（那样有/无 pymatgen 会给出不同 CN）。
    return C.build_all(cn_method=C.CN_METHOD_SHELL)


def _ids(records):
    return [sid for sid, _, _ in records]


#: 壳层法与 CrystalNN 参考在 CN（mean/min/max/hist）上**不同**的 8 个结构：
#: 冻结 labels.yaml 是 CrystalNN 口径，这里核对出来并写死，方法敏感性有据可查。
_METHOD_SENSITIVE_CN = {
    "control_dense_bulk_04", "control_dense_bulk_05", "control_dense_bulk_06",
    "control_molecular_crystal_03", "control_molecular_crystal_04",
    "control_isolated_molecule_02", "control_isolated_molecule_04",
    "control_low_coordination_01",
}


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
    r1 = C.build_all(cn_method=C.CN_METHOD_SHELL)
    r2 = C.build_all(cn_method=C.CN_METHOD_SHELL)
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


_CN_KEYS = ("expected_mean_cn", "expected_cn_min", "expected_cn_max",
            "expected_cn_hist", "cn_method")


def _note_head(notes: str) -> str:
    """"CN note: ..." 之前的构造说明 —— notes 里唯一不随 CN 方法变的部分。"""
    return notes.split("CN note:")[0].strip()


def test_labels_match_recomputed_references(records):
    """冻结 labels.yaml == 重算结果，**除 CN 相关字段外**。

    CN 参考值是唯一随 CN 参考方法变的字段（`cn_method` 与可能随之出现的
    "CN note: ..." 附注）；本档（无 pymatgen）显式走了壳层法，而 labels.yaml 是
    CrystalNN 口径，两者在 8/44 个结构上 CN 不同 —— 所以这里只逐字段比对与 CN
    无关的量，CN 字段的诚实性与差异结构单独校验（见下）。完整复现 CrystalNN 口径
    labels.yaml 的测试在 tests/research/test_controls_labels.py。
    """
    labels = C.load_labels()["structures"]
    assert set(labels) == set(_ids(records))
    for sid, atoms, e in records:
        lab = labels[sid]
        for key in lab:
            if key in _CN_KEYS or key == "notes":
                continue
            assert lab[key] == e[key], \
                f"{sid}: labels.yaml[{key}] != build_all(shell)[{key}]"
        assert (e["notes"].startswith(_note_head(lab["notes"]))
                or lab["notes"].startswith(_note_head(e["notes"]))), \
            f"{sid}: notes 的非 CN 部分不一致\n  labels.yaml: {lab['notes']}\n  shell: {e['notes']}"
        assert lab["expected_short_lambda_dim"] == C._ref_dim_spectrum(atoms)
        assert lab["expected_validity"] == C._validity_reference(atoms)
        assert abs(lab["expected_vacuum_gap"]
                   - round(C._vacuum_gap_estimate(atoms), 3)) < 1e-9


def test_cn_method_recorded_honestly(records):
    """cn_method 必须如实记录实际用的方法；方法敏感的 8 个结构要写死可查。"""
    for sid, _, e in records:
        assert e["cn_method"] in ("shell_1.25", "shell_1.25_unverified"), \
            f"{sid}: cn_method={e['cn_method']!r} 与显式选定的壳层法不符"
    frozen = C.load_labels()["structures"]
    divergent = {sid for sid, _, e in records
                 if any(e[k] != frozen[sid][k] for k in _CN_KEYS[:4])}
    assert divergent == _METHOD_SENSITIVE_CN, (
        "壳层法与 CrystalNN 参考的 CN 差异结构变了："
        f"{sorted(divergent ^ _METHOD_SENSITIVE_CN)}")


def test_crystalnn_method_without_pymatgen_fails_loudly(monkeypatch):
    """缺 pymatgen 时请求 CrystalNN 必须报错，绝不悄悄退回壳层法（旧行为的坑）。"""
    monkeypatch.setattr(C, "_pymatgen_available", lambda: False)
    with pytest.raises(ImportError, match=r"crysh\[research\]"):
        C.build_all(cn_method=C.CN_METHOD_CRYSTALNN)


def test_unknown_cn_method_rejected():
    with pytest.raises(ValueError, match="未知 CN 参考方法") as excinfo:
        C.build_all(cn_method="magic")
    assert "cn_method" in str(excinfo.value)  # 报错要说明值是从哪来的


def test_cn_method_resolution_order(monkeypatch):
    monkeypatch.delenv(C.CN_METHOD_ENV, raising=False)
    # 默认 = 冻结 labels.yaml 的口径
    assert C.resolve_cn_method() == C.DEFAULT_CN_METHOD == C.CN_METHOD_CRYSTALNN
    # 环境变量可覆盖默认（大小写/空白不敏感）
    monkeypatch.setenv(C.CN_METHOD_ENV, " Shell ")
    assert C.resolve_cn_method() == C.CN_METHOD_SHELL
    # 显式参数优先于环境变量
    assert C.resolve_cn_method(C.CN_METHOD_CRYSTALNN) == C.CN_METHOD_CRYSTALNN


def test_vasp_assets_match_labels():
    import ase.io

    labels = C.load_labels()
    files = sorted(p.name for p in C.STRUCTURES_DIR.glob("*.vasp"))
    assert files == sorted(f"{sid}.vasp" for sid in labels["structures"])
    for sid, atoms, e in C.build_all(cn_method=C.CN_METHOD_SHELL):
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

    # 这些结构上壳层法与冻结的 CrystalNN 参考给出同样的 CN（其余见
    # _METHOD_SENSITIVE_CN），所以下面同时对齐重算值与 labels.yaml。
    frozen = C.load_labels()["structures"]

    # high-coordination family is CN=12
    for sid in ("control_high_coordination_01", "control_high_coordination_02",
                "control_high_coordination_03"):
        assert lab(sid)["expected_mean_cn"] == 12.0
        assert lab(sid)["expected_mean_cn"] == frozen[sid]["expected_mean_cn"]
    # diamond Si: uniform CN=4, no L0 flags
    si = lab("control_dense_bulk_01")
    assert si["expected_cn_hist"] == {"4": 8}
    assert si["expected_cn_hist"] == frozen["control_dense_bulk_01"]["expected_cn_hist"]
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
