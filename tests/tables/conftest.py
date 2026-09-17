"""表格类测试：需要可选 extra `crysh[tables]`（pandas / pyarrow）。

为什么单独一个目录：核心安装（numpy + ase）下这些用例**无法收集**（模块级 import pandas），
放在 tests/ 根目录会让"核心 CI"在干净环境里直接报收集错误。目录即边界：
    pytest tests/core     # 只装 numpy+ase 即可
    pytest tests/tables   # 需要 pandas
    pytest tests/research # 需要 pymatgen / matplotlib
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_HERE, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
