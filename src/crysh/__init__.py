"""CrysH — Crystal Hierarchy: a multi-layer crystal knowledge mapper.

把一个晶体结构映射成**分层描述**，并输出可索引的平表：

    L0 validity          合法性：q_min、归一化体积 ν、晶胞条件数、冻芯重叠
    L1 dimensionality    周期维数：d(λ) 谱、d*、persistence、逐分量多标签
    L2 morphology        形态：真空隙、表面性、层数、多孔候选
    L3 coordination      配位：CN、P(CN|Z) 分位、小胞混叠守卫
    L4 geometry          配位几何：14 维特征 + q4/q6 路由出的几何标签
    L5 tokens            局域建筑块 token（`Ti|6|oct|O6`）与 corner/edge/face 共享

快速开始
--------
>>> from ase.build import bulk
>>> import crysh
>>> rec = crysh.map_record(bulk("Si", "diamond", a=5.43))
>>> rec["d_star"], rec["morphology_class"], rec["mean_cn"]
(3, 'dense_bulk', 4.0)

设计约定
--------
- **核心轻依赖**：只 numpy + ase。pandas/pyarrow（写表）与 pymatgen/matplotlib
  （研究工具）在 `crysh[tables]` / `crysh[research]` 里。
- **库不认识工作区**：输出路径一律经 :class:`crysh.config.MapperConfig` 显式传入，
  不推测目录布局。全库只读一个环境变量 ``CRYSH_CN_METHOD``：它只选
  ``crysh.controls`` 的 CN 参考方法（等价于 ``build_all(cn_method=...)``，参数优先），
  与路径 / 工作区布局无关。
- **列名与 token 是数据契约**：见 :mod:`crysh.config`（列）与 :mod:`crysh.tokens`（token）。

模块地图
--------
核心：`config`（配置/契约）、`bond`（键图）、`dimensionality`、`validity`、
`morphology`、`coord`、`geometry`、`tokens`、`metrics`、`records`（记录组装）、
`controls`（44 个 ground-truth 结构）、`dev`（合成结构，仅供冒烟）。
研究侧（可选）：`crysh.research.*`。
"""

from __future__ import annotations

from crysh.config import (
    D_STAR_LAMBDA,
    LAMBDA_COLS,
    LAMBDAS,
    LAMBDAS_STAR,
    RECORD_COLUMNS,
    MapperConfig,
)
from crysh.metrics import accumulation, effective_diversity, novel_gain, richness
from crysh.motifnet import MotifEdge, MotifNet, MotifNode, motif_network
from crysh.records import SITE_SUMMARY_COLUMNS, map_record, map_sites, map_structure, run_batch
from crysh.tokens import SHARING_CLASSES, TOKEN_LEVELS, motif_tokens

__version__ = "0.1.0.dev0"

__all__ = [
    "__version__",
    # 配置与数据契约
    "MapperConfig",
    "RECORD_COLUMNS",
    "LAMBDAS",
    "LAMBDAS_STAR",
    "LAMBDA_COLS",
    "D_STAR_LAMBDA",
    # 主流程
    "map_record",
    "map_structure",
    "map_sites",
    "run_batch",
    # L4 motif 超节点图
    "motif_network",
    "MotifNet",
    "MotifNode",
    "MotifEdge",
    "SITE_SUMMARY_COLUMNS",
    # token 与词表统计
    "motif_tokens",
    "TOKEN_LEVELS",
    "SHARING_CLASSES",
    "richness",
    "effective_diversity",
    "novel_gain",
    "accumulation",
]

_LAZY_SUBMODULES = frozenset({
    "validity", "dimensionality", "morphology", "coord", "geometry", "bond",
    "tokens", "records", "controls", "config", "metrics", "dev", "research",
    "localenv", "motifnet", "kernels",
})


def __getattr__(name: str):
    """子模块惰性入口（`crysh.validity`、`crysh.research` …）。

    惰性而非顶层 import：`import crysh` 只加载主流程真正需要的模块，
    不把全部计算模块（更不把可选重依赖）拖进来；按层取用时手感不变。
    """
    import importlib

    if name in _LAZY_SUBMODULES:
        mod = importlib.import_module(f"crysh.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'crysh' has no attribute {name!r}")
