"""Skeleton tests — replaced by real unit tests as layers land in v0.1.0."""

from __future__ import annotations

import crysh


def test_version_is_pep440_like() -> None:
    parts = crysh.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.split("dev")[0].isdigit() for p in parts[:2])


def test_core_import_does_not_pull_optional_deps() -> None:
    """核心包必须能被只有 numpy+ase 的环境 import。"""
    import sys

    for heavy in ("pandas", "pyarrow", "pymatgen", "matplotlib"):
        assert heavy not in sys.modules, f"{heavy} 不应在 import crysh 时被加载"


def test_public_api_is_declared() -> None:
    assert "__version__" in crysh.__all__
