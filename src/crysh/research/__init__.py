"""研究侧工具（**可选**，不属于 mapper 的核心路径）。

这些模块服务的是"用 CrysH 做一项研究"的工作流，而不是"把一个结构映射成知识"这件事：

| 模块 | 用途 | 依赖 |
|---|---|---|
| `figdata`  | 把各级 parquet 聚合成前端要的 JSON | pandas |
| `fingerprint` | 53 列结构指纹 + PCA 嵌入（去冗余/可视化） | numpy, pandas |
| `io`       | 子集切分、registry 构建、POSCAR 遍历 | pandas |
| `report`   | atlas 出图（六图 + 过滤漏斗） | matplotlib |
| `oracle`   | pymatgen（CrystalNN / Larsen / ChemEnv）对照基准 | pymatgen |
| `calibration` | Phase-0 pair cutoff / P(CN\\|Z) 校准表构建与验证 | pandas, pymatgen |
| `paths`    | 研究工作区的目录约定解析（`CKT_*` 环境变量） | — |

核心包（`crysh` 顶层）**不 import 本子包**；反之本子包可以 import 核心。
装依赖：``pip install "crysh[research]"``。
"""

from __future__ import annotations

__all__ = ["calibration", "figdata", "fingerprint", "io", "oracle", "paths", "report"]
