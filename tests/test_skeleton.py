"""骨架守卫：核心包必须"轻"且不依赖任何可选重依赖。"""

from __future__ import annotations

import subprocess
import sys

import crysh

_HEAVY = ("pandas", "pyarrow", "pymatgen", "matplotlib")


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
