"""CKT 路径解析 —— 全库唯一的"KT 根目录"真相源。

**为什么需要这个模块**（2026-09-17 目录重组时抽出）：

全库有 145 处 `KT_ROOT` 用法。原先每个模块各自写
`KT_ROOT = Path(__file__).resolve().parents[2]`（`<KT>/src/ckt/x.py` → `<KT>`），
这个约定把"库文件在磁盘上的深度"和"工作目录的位置"焊死在一起：

    <KT>/src/ckt/validity.py   → parents[2] = <KT>          ✔
    code/ckt/validity.py       → parents[2] = <仓库根>       ✘ 全错

只要 `src/ckt` 搬到别处（例如重组后的 `code/ckt/`），所有按位置推导的模块
会**静默**指向错误目录——不是报错，是读不到文件然后走 fallback/NaN。
所以解析逻辑必须集中一处，且必须允许外部显式指定。

**解析优先级**（与 `figdata.py` 既有约定一致，并向后兼容）：
    1. 显式参数（调用方传入的 kt_root）
    2. 环境变量 ``CKT_ROOT`` / ``CKT_WORK``
    3. 从本文件向上找**仓库标记文件**（``contracts.md``，或同时存在
       ``plan.md`` + ``progress.md`` 的目录）
    4. 兜底：``Path(__file__).resolve().parents[2]``（旧的深度假设）

所以搬迁有两种做法都安全：设 ``CKT_ROOT=<新工作目录>``，或把 ``contracts.md``
一起带到新工作目录（标记文件法自动命中）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

__all__ = ["resolve_kt_root", "kt_root", "KT_ROOT", "LEVELS_DIR",
           "WORKING_DIR", "FIGDATA_DIR", "figdata_dir", "SHARED_DIR"]

#: 仓库/工作目录标记文件：只要它在那儿，就能认出"这是 KT 工作目录"
MARKERS = ("contracts.md",)
#: 备选标记（两个同时存在才算）
MARKER_PAIR = ("plan.md", "progress.md")

#: 一次解析后缓存，避免每次调用都走文件系统
_CACHE: dict = {}


def _search_up(start: Path) -> Optional[Path]:
    """从 start 向上逐级查找 KT 工作目录标记。"""
    for base in (start, *start.parents):
        try:
            if any((base / m).is_file() for m in MARKERS):
                return base
            if all((base / m).is_file() for m in MARKER_PAIR):
                return base
        except OSError:  # 权限/断裂符号链接：跳过继续找
            continue
    return None


def resolve_kt_root(kt_root: Optional[os.PathLike | str] = None) -> Path:
    """解析 KT 工作目录（Artifact 根）。见模块 docstring 的优先级说明。

    Parameters
    ----------
    kt_root:
        调用方显式指定的根目录；给定时直接返回（并展开 ``~``）。

    Returns
    -------
    Path
        KT 工作目录的绝对路径。**不保证存在**——调用方需要自行处理缺失
        （历史上多数调用点是"读不到就回退/置 NaN"，保持该行为）。
    """
    if kt_root is not None:
        return Path(kt_root).expanduser()

    env = os.environ.get("CKT_ROOT") or os.environ.get("CKT_WORK")
    if env:
        return Path(env).expanduser()

    key = str(Path(__file__).resolve())
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    here = Path(__file__).resolve()
    found = _search_up(here.parent)
    if found is None:
        # 兜底：保持旧的深度假设（<KT>/src/ckt/paths.py → <KT>），
        # 至少不比搬迁前的行为更差。
        found = here.parents[2] if len(here.parents) > 2 else here.parent
    _CACHE[key] = found
    return found


def _resolve_levels_dir() -> Path:
    """`code/levels/`（各级产物数据所在；重组前是 `<KT>/<level>/`）。

    2026-09-17 重组把各级目录整体搬到 `code/levels/`，而工作目录只剩产物
    （docs/figdata/plan/progress 等）。库里有 11 处按"工作目录布局"拼的路径
    （controls/data、data-subset/data、levelN/data 的 parquet 与阈值表），
    全部改为经本函数解析，这样两种布局都能跑。

    优先级：``CKT_LEVELS`` 环境变量 > 新布局 ``<repo>/code/levels`` > 旧布局 ``<KT>``。
    """
    env = os.environ.get("CKT_LEVELS")
    if env:
        return Path(env).expanduser()
    new = KT_ROOT.parent.parent / "code" / "levels"          # <repo>/code/levels
    if new.is_dir():
        return new
    return KT_ROOT                                          # 旧布局兜底（集群侧）


def _resolve_working_dir() -> Path:
    """`01.working/` —— 跨主题共享产物的根（data/ figdata/ _shared/ 等）。

    2026-09-17 重组（E 组）把主题产物分到 `01.working/<topic>/`，而**跨库共享**的
    中间产物（figdata 聚合、子集 manifest、共享 QA）留在 `01.working/` 这一层，
    不再塞进工作目录（`<KT>`，重组后只剩文档与入口脚本）。

    优先级：``CKT_WORKING`` 环境变量 > 工作目录的上一级 > 工作目录本身（旧布局兜底）。
    """
    env = os.environ.get("CKT_WORKING")
    if env:
        return Path(env).expanduser()
    parent = KT_ROOT.parent
    if (parent / "figdata").is_dir() or (parent / "data").is_dir():
        return parent
    return KT_ROOT


def figdata_dir() -> Path:
    """figdata 聚合层根目录（``CKT_FIGDATA`` > ``<WORKING_DIR>/figdata`` > ``<KT>/figdata``）。"""
    env = os.environ.get("CKT_FIGDATA")
    if env:
        return Path(env).expanduser()
    cand = _resolve_working_dir() / "figdata"
    return cand if cand.is_dir() else KT_ROOT / "figdata"


#: 各级数据根目录（见 :func:`_resolve_levels_dir`）
LEVELS_DIR = _resolve_levels_dir()


def kt_root() -> Path:
    """``resolve_kt_root()`` 的函数式别名（延迟解析，便于测试改环境变量）。"""
    return resolve_kt_root()


#: 兼容旧代码的模块级常量。**注意**：在 import 时求值一次；
#: 若运行中会改 ``CKT_ROOT``，请改用 :func:`kt_root`。
KT_ROOT = resolve_kt_root()

# 下面两个常量依赖 KT_ROOT/RESOLVED 结果，**必须**在 KT_ROOT 之后求值
# （2026-09-17 踩坑：放在文件前部会 NameError）。
#: 跨主题共享产物根（重组前 == KT_ROOT）
WORKING_DIR = _resolve_working_dir()
#: figdata 聚合层根（重组前 == KT_ROOT/figdata）
FIGDATA_DIR = figdata_dir()
#: 跨主题共享资料（docs/ 与一次性脚本；重组前 == KT_ROOT/docs 与 KT_ROOT/scripts）
SHARED_DIR = WORKING_DIR / "_shared"
