"""冻结 controls 资产 ``labels.yaml`` 的**复现**测试（research 档）。

为什么这条一致性测试必须放在 research：labels.yaml 里的 CN 参考值是 pymatgen
CrystalNN 口径（42 条 ``CrystalNN`` + 2 条逐结构回退 ``shell_1.25_fallback``）。
没有 pymatgen 时 ``build_all(cn_method="crystalnn")`` 会直接 ImportError，而不是
悄悄换成 1.25x 壳层法（旧行为：壳层法在 8/44 个结构上给出不同 CN）。所以
"labels.yaml == 重算结果"只能在装了 pymatgen 的档里断言；tables 档只比对与 CN
参考方法无关的字段（tests/tables/test_controls.py）。

Run:  pytest tests/research/test_controls_labels.py -q        （需 crysh[research]）
"""

import pytest

from crysh import controls as C

#: 冻结 labels.yaml 记录的方法分布。重新生成资产时若不等于这个分布，就是口径漂移
#: （pymatgen 版本变化、或不小心用壳层法生成了资产），必须显式评审。
_EXPECTED_METHOD_COUNTS = {"CrystalNN": 42, "shell_1.25_fallback": 2}


@pytest.fixture(scope="module")
def crystalnn_records():
    # build_all(crystalnn) 约 17s（CrystalNN 对真空盒子较慢），整模块只算一次。
    return C.build_all(cn_method=C.CN_METHOD_CRYSTALNN)


def test_crystalnn_reference_is_really_available():
    """research 档必须真的装了 pymatgen —— 否则下面的复现测试没有意义。"""
    assert C._pymatgen_available()
    assert C.resolve_cn_method(C.CN_METHOD_CRYSTALNN) == C.CN_METHOD_CRYSTALNN


def test_labels_match_recomputed_crystalnn_references(crystalnn_records):
    labels = C.load_labels()["structures"]
    assert set(labels) == {sid for sid, _, _ in crystalnn_records}
    for sid, atoms, e in crystalnn_records:
        assert labels[sid] == e, \
            f"{sid}: labels.yaml 与 build_all(cn_method='crystalnn') 不一致"
        assert labels[sid]["expected_short_lambda_dim"] == C._ref_dim_spectrum(atoms)
        assert labels[sid]["expected_validity"] == C._validity_reference(atoms)
        assert abs(labels[sid]["expected_vacuum_gap"]
                   - round(C._vacuum_gap_estimate(atoms), 3)) < 1e-9


def test_frozen_labels_record_the_crystalnn_method(crystalnn_records):
    """断言**实际用的是哪种方法**：不是"没报错就算过"。"""
    frozen = C.load_labels()["structures"]
    counts: dict[str, int] = {}
    for sid, _, e in crystalnn_records:
        assert e["cn_method"] == frozen[sid]["cn_method"], sid
        counts[e["cn_method"]] = counts.get(e["cn_method"], 0) + 1
    assert counts == _EXPECTED_METHOD_COUNTS, (
        f"冻结 labels.yaml 的 CN 方法分布变了：{counts} != {_EXPECTED_METHOD_COUNTS}")


def test_default_cn_method_is_the_frozen_recipe(monkeypatch):
    """默认（不传参、无环境变量）就是冻结资产的口径，不随环境静默改变。"""
    monkeypatch.delenv(C.CN_METHOD_ENV, raising=False)
    assert C.resolve_cn_method() == C.DEFAULT_CN_METHOD == C.CN_METHOD_CRYSTALNN
