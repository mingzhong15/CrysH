"""库测试的共享配置。

两点约定（都来自既有测试的写法，不要改）：

1. **兄弟测试模块互相 import**（如 `test_geometry_metallic` 复用 `test_geometry`
   的 fixture）。pytest 默认的 prepend import-mode 会把测试目录放进 `sys.path`，
   所以这里显式再插一次，保证 `pytest tests/` 与 `pytest tests/test_x.py` 都能跑。
2. 测试**不依赖任何研究数据**：需要的结构一律在测试里用 ase 现场构造。
   需要真实数据的用例（子集 POSCAR、集群 parquet）留在研究工程侧，不进本仓。
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
