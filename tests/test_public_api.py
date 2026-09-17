"""公共 API 契约测试：`import crysh` 的手感与边界。

这些用例钉住的是"库对外长什么样"，不是物理量本身：
- 顶层 API 必须能一行调通（README 的 quickstart 就是断言）；
- 核心不得拉起可选重依赖；
- 核心不得依赖 `crysh.research`（研究侧可以依赖核心，反之不行）。
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

from ase.build import bulk

import crysh


def test_quickstart_from_readme() -> None:
    rec = crysh.map_record(bulk("Si", "diamond", a=5.43))
    assert rec["d_star"] == 3
    assert rec["morphology_class"] == "dense_bulk"
    assert rec["mean_cn"] == 4.0
    assert rec["sharing_top_label"] == "isolated"


def test_record_has_exactly_the_contract_columns() -> None:
    rec = crysh.map_record(bulk("Al", "fcc", a=4.05))
    assert list(rec) == crysh.RECORD_COLUMNS
    assert set(crysh.LAMBDA_COLS.values()) <= set(rec)


def test_metallic_case_matches_known_physics() -> None:
    """fcc Al：CN=12、cuboctahedral、3D——这条是金属分支的钉死值。"""
    rec = crysh.map_record(bulk("Al", "fcc", a=4.05))
    assert rec["mean_cn"] == 12.0
    assert rec["geom_top_label"] == "cuboctahedral"
    assert rec["d_star"] == 3


def test_mapper_config_scale_and_lambda_grid() -> None:
    cfg = crysh.MapperConfig(scale="20k")
    assert cfg.scale == "20k"
    assert cfg.with_scale("mgfull").scale == "mgfull"
    assert tuple(cfg.lambda_cols()) == crysh.LAMBDAS
    assert cfg.lambda_cols()[1.2] == "d_120"


def test_core_never_imports_research() -> None:
    """核心模块不得依赖 crysh.research（研究侧单向依赖核心）。"""
    code = textwrap.dedent(
        """
        import sys
        import crysh
        from ase.build import bulk
        crysh.map_record(bulk("Si", "diamond", a=5.43))
        leaked = [m for m in sys.modules if m.startswith("crysh.research")]
        print(",".join(leaked))
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"核心路径拉起了研究侧模块: {out.stdout.strip()}"


def test_lazy_submodule_access() -> None:
    assert crysh.validity.__name__ == "crysh.validity"
    assert crysh.morphology.MORPHOLOGY_CLASSES
    try:
        crysh.not_a_module  # noqa: B018
    except AttributeError as exc:
        assert "not_a_module" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("未知属性应当抛 AttributeError")
