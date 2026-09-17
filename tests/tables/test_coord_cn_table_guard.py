"""Level 3 percentile 表最小样本守卫单测（v1.3-L3 增补：min_sites 门槛 /
cn_table_coverage_report / CN_MIN_SITES / save-load meta 往返）。

合成站点长表：一个稀有元素（La, Z=57, n=5）+ 一个常见元素（O, Z=8, n=200），
不依赖本地数据。默认 min_sites=CN_MIN_SITES=100 下稀有元素被排除（查表走
NaN 保守路径）；min_sites<=0 全包含；老调用签名兼容。
"""
import json

import numpy as np
import pandas as pd

from crysh.coord import (
    CN_MIN_SITES,
    apply_cn_table,
    calibrate_cn_table,
    cn_table_coverage_report,
    load_cn_table,
    save_cn_table,
)


def _synthetic_sites() -> pd.DataFrame:
    """合成站点长表：La(Z=57) 5 站点 + O(Z=8) 200 站点。"""
    rare = pd.DataFrame({"Z": [57] * 5, "cn": [2, 3, 4, 5, 6]})
    common = pd.DataFrame({"Z": [8] * 200, "cn": [2, 3, 4, 6] * 50})
    return pd.concat([rare, common], ignore_index=True)


def test_cn_min_sites_constant():
    """模块常量：pilot 建议值 100（待 20k/Phase-0 站点规模重评估）。"""
    assert CN_MIN_SITES == 100


def test_default_excludes_rare_element():
    """默认 min_sites=100：稀有元素（n=5）不写入表，查表 → NaN 保守路径；
    常见元素（n=200）正常入表。"""
    df = _synthetic_sites()
    table = calibrate_cn_table(df)
    assert 57 not in table                       # La 被排除
    assert set(table) == {8}
    assert table[8]["n"] == 200
    assert table[8]["cdf"] == {2: 0.25, 3: 0.5, 4: 0.75, 6: 1.0}
    # 被排除元素的站点 percentile = NaN（不计为异常、不扩大异常率）
    p = apply_cn_table(df["Z"].to_numpy(), df["cn"].to_numpy(), table)
    assert np.all(np.isnan(p[df["Z"] == 57]))
    assert np.all(~np.isnan(p[df["Z"] == 8]))
    np.testing.assert_allclose(p[df["Z"] == 8][:4], [0.25, 0.5, 0.75, 1.0])


def test_min_sites_zero_includes_all():
    """min_sites=0 → 不设门槛（全包含），CDF 数值与门槛前的语义一致。"""
    df = _synthetic_sites()
    table = calibrate_cn_table(df, min_sites=0)
    assert set(table) == {57, 8}
    assert table[57]["n"] == 5
    assert table[57]["cdf"] == {2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0}
    assert table[8]["n"] == 200


def test_min_sites_explicit_threshold_boundary():
    """显式门槛的边界语义：n == min_sites 保留（< 才排除）。"""
    df = _synthetic_sites()
    assert set(calibrate_cn_table(df, min_sites=5)) == {57, 8}    # La n=5 保留
    assert set(calibrate_cn_table(df, min_sites=6)) == {8}        # La n=5 排除
    assert set(calibrate_cn_table(df, min_sites=200)) == {8}      # O n=200 保留
    assert set(calibrate_cn_table(df, min_sites=201)) == set()    # O 也排除


def test_old_call_signature_equivalent():
    """老调用签名兼容：不传 min_sites ≡ 传 100 ≡ 传 CN_MIN_SITES。"""
    df = _synthetic_sites()
    t_old = calibrate_cn_table(df)
    assert t_old == calibrate_cn_table(df, min_sites=100)
    assert t_old == calibrate_cn_table(df, min_sites=CN_MIN_SITES)
    # 关键字与位置参数均可（签名: calibrate_cn_table(records_df, min_sites=...)）
    assert calibrate_cn_table(df, 0) == calibrate_cn_table(df, min_sites=0)


def test_coverage_report_fields():
    """覆盖报告：字段结构 / included 判定 / 计数（默认 min_sites=100）。"""
    df = _synthetic_sites()
    rep = cn_table_coverage_report(df)
    assert set(rep) == {"min_sites", "elements", "n_included", "n_excluded",
                        "total_sites"}
    assert rep["min_sites"] == 100
    assert rep["elements"][57] == {"sites": 5, "included": False}
    assert rep["elements"][8] == {"sites": 200, "included": True}
    assert all(isinstance(z, int) for z in rep["elements"])   # int 原子序数键
    assert rep["n_included"] == 1 and rep["n_excluded"] == 1
    assert rep["total_sites"] == 205
    # min_sites=0 → 全包含
    rep0 = cn_table_coverage_report(df, min_sites=0)
    assert rep0["min_sites"] == 0
    assert rep0["elements"][57]["included"] is True
    assert rep0["elements"][8]["included"] is True
    assert rep0["n_included"] == 2 and rep0["n_excluded"] == 0
    # 与 calibrate_cn_table 的写入门槛逐元素一致
    assert set(calibrate_cn_table(df)) == {
        z for z, e in rep["elements"].items() if e["included"]}


def test_coverage_report_zero_site_element():
    """含站点数 0 的元素（Z>0 但 cn<0 的行）：sites=0、included=False、
    不计入 total_sites，也不入表。"""
    df = pd.concat([_synthetic_sites(),
                    pd.DataFrame({"Z": [14], "cn": [-1]})],  # Si，无效站点
                   ignore_index=True)
    rep = cn_table_coverage_report(df)
    assert rep["elements"][14] == {"sites": 0, "included": False}
    assert rep["total_sites"] == 205            # 无效站点不计
    assert rep["n_excluded"] == 2               # La(5) + Si(0)
    assert 14 not in calibrate_cn_table(df, min_sites=0)  # 0 站点永不入表


def test_coverage_report_structure_level_input():
    """结构级降级输入（elements/n_atoms/cn_mean）同样支持覆盖报告。"""
    df = pd.DataFrame({
        "elements": [["Hf", "O"], ["O"]],
        "n_atoms": [[2, 4], [3]],
        "cn_mean": [6.2, 4.0],
    })
    rep = cn_table_coverage_report(df, min_sites=3)
    assert rep["elements"][72] == {"sites": 2, "included": False}  # Hf 2 站点
    assert rep["elements"][8] == {"sites": 7, "included": True}    # O 4+3 站点
    assert rep["total_sites"] == 9
    assert rep["n_included"] == 1 and rep["n_excluded"] == 1
    assert set(calibrate_cn_table(df, min_sites=3)) == {8}


def test_save_load_meta_roundtrip(tmp_path):
    """save/load meta 往返：meta 自动补 min_sites 缺省 CN_MIN_SITES、显式
    覆盖生效、无 meta 旧表兼容、默认返回值仍是纯表。"""
    df = _synthetic_sites()
    table = calibrate_cn_table(df)
    out = save_cn_table(table, tmp_path / "cn_table.json", meta={"source": "test"})
    loaded, meta = load_cn_table(out, return_meta=True)
    assert set(loaded) == {8}                   # int 键恢复
    assert loaded[8]["n"] == 200
    assert meta["min_sites"] == CN_MIN_SITES == 100   # 缺省自动补
    assert meta["source"] == "test" and meta["version"] == "v1"
    # 显式 meta 覆盖（如表实际用 min_sites=0 校准）
    out2 = save_cn_table(table, tmp_path / "cn_table0.json",
                         meta={"min_sites": 0})
    loaded2, meta2 = load_cn_table(out2, return_meta=True)
    assert meta2["min_sites"] == 0 and set(loaded2) == {8}
    # 无 meta 的旧裸表格式：不破坏，meta 为空 dict
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"3": {"cdf": {"6": 1.0}, "n": 3}}),
                    encoding="utf-8")
    loaded_bare, meta_bare = load_cn_table(bare, return_meta=True)
    assert set(loaded_bare) == {3} and loaded_bare[3]["n"] == 3
    assert meta_bare == {}
    # 默认返回值仍是纯表（老调用不破坏）
    assert set(load_cn_table(out)) == {8}
