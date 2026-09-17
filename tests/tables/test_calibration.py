"""pytest for phase0-calibration (src/ckt/calibration.py).

Fixtures are small inline structures (NaCl rocksalt, diamond) so the tests
run in seconds; CrystalNN is used only on these tiny systems.

Run:  cd <KT> && $CKT_PY -m pytest phase0-calibration/tests -q
"""

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from ase.io import write

from crysh.bond import build_bond_graph
from crysh.research.calibration import (
    FALLBACK_COV_LAM,
    R0_TIE_GUARD,
    _cov_sum,
    build_cutoff_table,
    check_cn_consistency,
    cn_from_graph,
    coverage_metrics,
    extract_cnn_bonds,
    extract_structure_cnn_bonds,
    load_table,
    save_cutoff_table,
    table_statistics,
)


def make_nacl(a=5.64):
    return bulk("NaCl", "rocksalt", a=a)


def make_diamond(a=3.567):
    return bulk("C", "diamond", a=a)


def bonds_frame(rows):
    return pd.DataFrame(
        rows, columns=["structure_id", "i", "j", "zi", "zj", "d"]
    )


# ---------------------------------------------------------------------------
# A. extraction
# ---------------------------------------------------------------------------

def test_extract_structure_cnn_bonds_nacl():
    atoms = make_nacl()  # 2-atom primitive rocksalt cell
    bonds = extract_structure_cnn_bonds(atoms)
    # 6 undirected Na-Cl bonds: one Cl neighbour in 6 periodic images. Each
    # bond is seen from BOTH endpoints by CrystalNN; the canonical (a<=b, S)
    # dedup must collapse the two views into a single stored bond.
    assert len(bonds) == 6
    for t in bonds:
        assert len(t) == 5
        i, j, zi, zj, d = t
        assert i < j
        assert (zi, zj) == (11, 17)
        assert abs(d - 2.82) < 0.02  # a/2
    # CN from the undirected bond list == 6 everywhere (rocksalt octahedra)
    i = np.array([t[0] for t in bonds])
    j = np.array([t[1] for t in bonds])
    cn = np.bincount(np.concatenate([i, j]), minlength=len(atoms))
    assert list(cn) == [6, 6]


def test_extract_structure_cnn_bonds_diamond():
    atoms = make_diamond()  # 2-atom primitive diamond cell
    bonds = extract_structure_cnn_bonds(atoms)
    i = np.array([t[0] for t in bonds])
    j = np.array([t[1] for t in bonds])
    d = np.array([t[4] for t in bonds])
    assert len(bonds) == 4  # 4 images of the single C-C bond
    assert np.allclose(d, 1.544, atol=0.02)
    cn = np.bincount(np.concatenate([i, j]), minlength=len(atoms))
    assert list(cn) == [4, 4]


def test_extract_cnn_bonds_end_to_end_and_consistency(tmp_path):
    """Folder -> parquet end-to-end + oracle cnn_cn consistency spot check."""
    from crysh.research.oracle import run_crystalnn_cn  # public API only

    folder = tmp_path / "structs"
    folder.mkdir()
    write(folder / "nacl.vasp", make_nacl(), format="vasp")
    write(folder / "diamond.vasp", make_diamond(), format="vasp")

    # small oracle parquet in the oracle_2k layout
    oracle_rows = []
    for sid, atoms in [("nacl", make_nacl()), ("diamond", make_diamond())]:
        cn = run_crystalnn_cn(atoms)
        oracle_rows.append({"structure_id": sid, "larsen_dim": 3,
                            "cnn_cn_mean": float(cn.mean()), "cnn_cn": list(cn)})
    oracle_pq = tmp_path / "oracle.parquet"
    pd.DataFrame(oracle_rows).to_parquet(oracle_pq)

    out = tmp_path / "bonds.parquet"
    summary = extract_cnn_bonds(folder, out, n_proc=2, oracle_parquet=oracle_pq,
                                n_check=10)
    assert summary["n_structures"] == 2
    assert summary["n_failed"] == 0

    df = pd.read_parquet(out)
    assert list(df.columns) == ["structure_id", "i", "j", "zi", "zj", "d"]
    assert set(df["structure_id"]) == {"nacl", "diamond"}
    assert df["i"].dtype == np.int64 and df["d"].dtype == np.float64
    # CN consistency check passed for both structures
    assert summary["cn_consistency_check"]["all_match"] is True
    assert summary["cn_consistency_check"]["n_checked"] == 2

    # undirected bond count invariant: sum(CN)/2 per structure (image-distinct
    # bonds of the same (i,j) pair are legitimate separate bonds)
    for sid in ("nacl", "diamond"):
        sub = df[df["structure_id"] == sid]
        cn = np.bincount(np.concatenate([sub["i"], sub["j"]]), minlength=2)
        assert len(sub) == cn.sum() / 2

    # spot-check function works standalone too
    rep = check_cn_consistency(out, oracle_pq, n_structures=1)
    assert rep["all_match"] is True and rep["n_checked"] == 1


# ---------------------------------------------------------------------------
# B/C. table build + serialisation + fallback rule
# ---------------------------------------------------------------------------

def test_build_cutoff_table_calibrated_and_fallback(tmp_path):
    rows = []
    rows += [("s1", 0, 1, 6, 6, 1.50)] * 40   # well-sampled pair
    rows += [("s2", 0, 1, 79, 79, 3.00)] * 5    # rare pair -> fallback
    pq = tmp_path / "bonds.parquet"
    bonds_frame(rows).to_parquet(pq)

    table = build_cutoff_table(pq, min_samples=30, quantile=0.95)
    assert table[(6, 6)] == pytest.approx(1.50 * (1.0 + R0_TIE_GUARD))
    assert table[(79, 79)] == pytest.approx(FALLBACK_COV_LAM * _cov_sum(79, 79))
    assert (79, 79) in table and (6, 6) in table

    stats = table_statistics(pq, min_samples=30)
    s6 = stats[stats["zi"] == 6].iloc[0]
    s79 = stats[stats["zi"] == 79].iloc[0]
    assert s6["source"] == "calibrated" and s6["n"] == 40
    assert s79["source"] == "fallback" and s79["n"] == 5
    assert s6["ratio_q95"] == pytest.approx(1.50 / _cov_sum(6, 6))
    # q50/q95/d_max columns present
    for col in ("d_q50", "d_q95", "d_max", "cov_sum", "r0", "pair"):
        assert col in stats.columns

    cov = coverage_metrics(pq, min_samples=30)
    assert cov["bond_coverage"] == pytest.approx(40 / 45)
    assert cov["fallback_pair_fraction"] == 0.5
    assert cov["n_structures_involved_fallback"] == 1


def test_nacl_table_gives_cn6(tmp_path):
    """NaCl fixture: extracted bond lengths -> table -> bond.py CN == 6."""
    atoms = make_nacl()
    bonds = extract_structure_cnn_bonds(atoms)
    pq = tmp_path / "nacl_bonds.parquet"
    bonds_frame([("nacl", *t) for t in bonds]).to_parquet(pq)

    table = build_cutoff_table(pq, min_samples=1, quantile=0.95)
    assert (11, 17) in table
    assert table[(11, 17)] == pytest.approx(2.82 * (1.0 + R0_TIE_GUARD), rel=1e-4)

    g = build_bond_graph(atoms, cutoff_table=table, lam=1.0)
    assert g.cutoff_mode == "table"
    cn = cn_from_graph(g, len(atoms))
    assert list(cn) == [6] * len(atoms)
    # missing pairs (Na-Na / Cl-Cl) fall back to the covalent sum and stay unbonded
    assert {(11, 11), (17, 17)} <= set(g.pair_r0)


def test_diamond_table_gives_cn4(tmp_path):
    atoms = make_diamond()
    bonds = extract_structure_cnn_bonds(atoms)
    pq = tmp_path / "c_bonds.parquet"
    bonds_frame([("diamond", *t) for t in bonds]).to_parquet(pq)
    table = build_cutoff_table(pq, min_samples=1, quantile=0.95)
    g = build_bond_graph(atoms, cutoff_table=table, lam=1.0)
    cn = cn_from_graph(g, len(atoms))
    assert list(cn) == [4] * len(atoms)


def test_save_load_table_roundtrip(tmp_path):
    table = {(8, 1): 1.2, (3, 3): 2.0}  # unsorted keys on purpose
    p = tmp_path / "cal.json"
    save_cutoff_table(table, p, meta={"min_samples": 30})
    loaded = load_table(p)
    assert loaded == {(1, 8): 1.2, (3, 3): 2.0}
    # bare string-keyed json also loads
    p2 = tmp_path / "bare.json"
    p2.write_text('{"11,17": 2.82, "6,6": 1.54}', encoding="utf-8")
    assert load_table(p2) == {(11, 17): 2.82, (6, 6): 1.54}


def test_build_cutoff_table_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_cutoff_table(tmp_path / "missing.parquet")
    pq = tmp_path / "bad.parquet"
    pd.DataFrame({"structure_id": ["s"], "d": [1.0]}).to_parquet(pq)
    with pytest.raises(ValueError):
        build_cutoff_table(pq)
    with pytest.raises((FileNotFoundError, ValueError)):
        extract_cnn_bonds(tmp_path / "nope", tmp_path / "out.parquet")


# ---------------------------------------------------------------------------
# cn_from_graph unit semantics
# ---------------------------------------------------------------------------

def test_cn_from_graph_empty_and_bidirectional(tmp_path):
    atoms = make_nacl()
    pq = tmp_path / "one_bond.parquet"
    bonds_frame([("s", 0, 1, 11, 17, 2.82)]).to_parquet(pq)
    table = build_cutoff_table(pq, min_samples=1)
    g = build_bond_graph(atoms, cutoff_table=table, lam=1.0)
    n = len(atoms)
    assert cn_from_graph(None, n).tolist() == [0] * n
    # bidirectional graph: unique (i, j, S) triples give the undirected degree
    keys = np.column_stack([g.i, g.j, g.S])
    assert len(keys) == len(np.unique(keys, axis=0))
