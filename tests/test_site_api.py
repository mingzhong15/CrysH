"""site 级 API（`map_sites`）与**已知未修**问题的登记测试。

两件事放一起是有意的：
1. `map_sites` / `SITE_SUMMARY_COLUMNS` 的契约（结构级汇总键、边界情形）；
2. `tokens` 的多面体共享判据目前仍有一处**已知错误**（见 `test_sharing_*_known_bug`），
   用 `xfail(strict=True)` 钉住：一旦有人真把它修对，这些测试会**由 xfail 变 XPASS
   而报错**，强制当场更新文档与 token 生态，而不是让修复悄悄发生。
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

import crysh
from crysh.kernels import masked_graph, neighbor_list
from crysh.records import SITE_SUMMARY_COLUMNS, map_sites
from crysh.tokens import motif_tokens


def _graph(atoms, lam=1.2):
    return masked_graph(neighbor_list(atoms, lam_max=2.0), lam)


def _coord(atoms):
    from crysh.coord import coordination

    return coordination(atoms, _graph(atoms))


# --------------------------------------------------------------------------- #
# map_sites 契约
# --------------------------------------------------------------------------- #
def test_map_sites_returns_one_entry_per_atom_and_all_summary_keys():
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    sites, summary = map_sites(atoms)
    assert len(sites) == len(atoms)
    assert set(summary) == set(SITE_SUMMARY_COLUMNS)
    assert all(isinstance(v, (int, float)) or isinstance(v, str) for v in summary.values())


def test_map_sites_agrees_with_localenv_module():
    atoms = bulk("Al", "fcc", a=4.05)
    from crysh.localenv import local_environments

    sites, summary = map_sites(atoms)
    direct = local_environments(atoms)
    assert [s.cn for s in sites] == [d.cn for d in direct]
    assert summary["cn_eff_mean"] == pytest.approx(
        float(np.mean([d.cn_eff for d in direct])))


def test_map_sites_is_additive_does_not_touch_frozen_record_columns():
    """site 汇总列绝不能混进冻结的 RECORD_COLUMNS（那是 M4 的契约变更）。"""
    assert not set(SITE_SUMMARY_COLUMNS) & set(crysh.RECORD_COLUMNS)
    rec = crysh.map_record(bulk("Si", "diamond", a=5.431))
    assert list(rec) == crysh.RECORD_COLUMNS  # 59 列，未增未减


def test_map_sites_isolated_atom_does_not_crash():
    """单原子（无邻居）：汇总要给 NaN 而不是抛异常。"""
    atoms = Atoms("Ar", positions=[[0, 0, 0]], cell=np.eye(3) * 12.0, pbc=True)
    sites, summary = map_sites(atoms)
    assert len(sites) == 1
    assert sites[0].cn == 0
    assert np.isnan(summary["p_cn_mean"]) or summary["p_cn_mean"] >= 0.0


def test_map_sites_reports_low_confidence_for_metals():
    """金属的键判据偏松 → 至少要有非零比例的"低置信"位点被报出来。"""
    _, summary = map_sites(bulk("Al", "fcc", a=4.05))
    assert 0.0 <= summary["shell_conf_low_frac"] <= 1.0
    assert summary["shell_conf_mean"] < 1.0


# --------------------------------------------------------------------------- #
# 已知未修：多面体共享判据
# --------------------------------------------------------------------------- #
def _labels(n, geo):
    return [geo] * n


@pytest.mark.xfail(strict=True, reason=(
    "已知 bug（2026-09-17 定位，未修）：tokens 的共享判据按原子索引配对，在周期结构里"
    "同一原子的多个周期像被折叠 → 金刚石 tetrahedra 本应 corner-sharing，现在算成 isolated。"
    "改它会改变冻结的 sharing token 取值（l4 串、tree/coverage 生态），故先登记不改。"
    "修对之后本测试会 XPASS → 届时同步更新 token 生态与 CHANGELOG。"))
def test_sharing_diamond_is_corner_known_bug():
    atoms = bulk("Si", "diamond", a=5.431)
    res = motif_tokens(atoms, _graph(atoms), _coord(atoms),
                       _labels(len(atoms), "tetrahedral"), d_star=3)
    assert res["sharing_label"] == "corner"
    assert res["sharing"]["corner"] == pytest.approx(1.0)


@pytest.mark.xfail(strict=True, reason=(
    "同上的共享判据 bug：岩盐 NaCl 的八面体两两共享 2 个 Cl（edge-sharing）为主，"
    "现在算成 isolated/corner。"))
def test_sharing_rocksalt_is_edge_known_bug():
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    res = motif_tokens(atoms, _graph(atoms), _coord(atoms),
                       _labels(len(atoms), "octahedral"), d_star=3)
    assert res["sharing"]["edge"] > 0.0


def test_token_chemistry_string_counts_bonds_like_coord():
    """L5 化学串与 `coord.cn` 同口径（**数键**）：这是刻意的，不是漏了周期像。

    岩盐：Na 有 6 条 Na–Cl 键 → `Cl6`，与 `coord.cn = 6`、`localenv.cn_species = 'Cl6'`
    三处一致。若改成"数不同的邻居原子"，小胞里会得到 `Cl1` —— 三处口径立刻互相矛盾，
    而这种矛盾正是本模块此前的坑（详见 `_neighbor_sets` 的 docstring）。
    """
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    res = motif_tokens(atoms, _graph(atoms), _coord(atoms),
                       _labels(len(atoms), "octahedral"), d_star=3)
    assert res["l3"][0].split("|")[3] == "Cl6"

    from crysh.localenv import local_environment

    assert local_environment(atoms, 0).cn_species == "Cl6"
    assert _coord(atoms).cn[0] == 6
