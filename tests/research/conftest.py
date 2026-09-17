"""研究侧测试：需要 `crysh[research]`（pymatgen / matplotlib）。"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
