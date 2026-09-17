"""骨架守卫：核心包必须"轻"且不依赖任何可选重依赖。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import crysh

_HEAVY = ("pandas", "pyarrow", "pymatgen", "matplotlib")
#: 仓库根（tests/ 的上一级）与 src/（crysh 包的上一级）
_ROOT = Path(__file__).resolve().parents[1]
_SRC = Path(crysh.__file__).resolve().parents[1]


def test_version_is_pep440_like() -> None:
    parts = crysh.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.split("dev")[0].isdigit() for p in parts[:2])


def test_core_import_does_not_pull_optional_deps() -> None:
    """在**干净子进程**里 import crysh：核心包不得拉入可选重依赖。

    用子进程而不是查 `sys.modules`：整套测试跑起来时别的测试模块会 import
    pandas/matplotlib，用全局状态判断会假失败（实测踩过）。
    """
    code = (
        "import sys, crysh;"
        f"heavy=[m for m in {_HEAVY!r} if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "", f"import crysh 拉入了可选重依赖: {out}"


def test_public_api_is_declared() -> None:
    assert "__version__" in crysh.__all__


def test_core_tier_collects_with_optional_deps_blocked() -> None:
    """核心档（tests/ 根目录）在**屏蔽可选重依赖**的子进程里必须能收集。

    与上面的 import-crysh 守卫互补：这里把 pandas/pyarrow/pymatgen/matplotlib 的
    import 直接打掉，再跑 `pytest tests --ignore=tests/{tables,research}
    --collect-only`。根目录任何测试模块（或它 import 的 crysh 子模块）只要在模块级
    拉了重依赖，收集就会红 —— 这正是"目录即边界"要防的回归。
    """
    blocker = f"""
import sys
HEAVY = {_HEAVY!r}


class _BlockHeavy:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in HEAVY:
            raise ImportError("blocked heavy dependency: " + name)
        return None


sys.meta_path.insert(0, _BlockHeavy())
for _name in HEAVY:                      # 自检：屏蔽必须真的生效
    try:
        __import__(_name)
    except ImportError:
        pass
    else:
        raise SystemExit("heavy-dep blocker ineffective for " + _name)

import pytest

raise SystemExit(pytest.main([
    "tests", "-q", "--collect-only", "--no-header",
    "--ignore=tests/tables", "--ignore=tests/research"]))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_SRC)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    proc = subprocess.run([sys.executable, "-c", blocker], cwd=_ROOT,
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, (
        "核心档在屏蔽 pandas/pyarrow/pymatgen/matplotlib 后收集失败：\n"
        + proc.stdout[-4000:] + proc.stderr[-4000:])


def test_tables_ship_with_package() -> None:
    """冻结查表必须随包分发（否则 validity 在干净环境里直接 FileNotFoundError）。"""
    from crysh.validity import _RCORE_DEFAULT_PATH

    assert _RCORE_DEFAULT_PATH.is_file(), _RCORE_DEFAULT_PATH


def test_controls_assets_ship_with_package() -> None:
    """44 个 ground-truth 结构与 labels 必须随包分发。"""
    from crysh.controls import DATA_DIR, LABELS_PATH, STRUCTURES_DIR

    assert DATA_DIR.is_dir()
    assert LABELS_PATH.is_file()
    assert len(list(STRUCTURES_DIR.glob("**/*.vasp"))) >= 40
