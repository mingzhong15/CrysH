"""兼容层：record schema 已并入 :mod:`crysh.config`（R2 收敛，见该模块 docstring）。

保留本文件是因为研究工程（`code/levels`、`code/pipeline`）仍按 `crysh.config` 引用它；
R4 的 shim 会把它一起 re-export。新代码请直接用 ``crysh.config.RECORD_COLUMNS``。
"""

from __future__ import annotations

from .config import LAMBDA_COLS, RECORD_COLUMNS

__all__ = ["RECORD_COLUMNS", "LAMBDA_COLS"]
