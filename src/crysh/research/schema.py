"""兼容层：record schema 已并入 :mod:`crysh.config`（R2/R3 收敛）。

保留本模块是因为研究工程的 `code/levels`、`code/pipeline` 仍按 `ckt.schema` /
`crysh.research.schema` 引用它（R4 的 shim 会一起 re-export）。
新代码请直接用 ``crysh.config.RECORD_COLUMNS`` / ``crysh.config.LAMBDA_COLS``。
"""

from __future__ import annotations

from crysh.config import LAMBDA_COLS, RECORD_COLUMNS

__all__ = ["RECORD_COLUMNS", "LAMBDA_COLS"]
