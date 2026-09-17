"""Level 5 — diversity/coverage 指标（contracts.md §3.8，签名冻结）。

  * richness(counts, n_min=1)          R(D) = |{m : n_m(D) > n_min}|（plan.md §6）
  * effective_diversity(counts)        D_eff = e^H，H = −Σ p ln p（空集 → 0）
  * novel_gain(tokens_D, tokens_ref)   |D \\ ref|
  * accumulation(token_lists, grid)    前缀 richness 曲线（token_lists 按结构顺序）

纯标准库实现，可移植（200M streaming 时 accumulation 可换增量归并，接口不变）。
"""

from __future__ import annotations

import math

__all__ = ["richness", "effective_diversity", "novel_gain", "accumulation"]


def richness(counts: dict, n_min: int = 1) -> int:
    """motif richness：count > n_min 的 token 数（契约默认 n_min=1）。

    plan.md §6：R(D) = |{m : n_m(D) > n_min}|——严格按此不等式（> 而非 ≥）。
    """
    return sum(1 for c in counts.values() if c > n_min)


def effective_diversity(counts: dict) -> float:
    """Shannon 有效多样性 D_eff = e^H，H = −Σ p_m ln p_m。

    单一 token → 1.0；两类等比 → 2.0；空 counts → 0.0。
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return math.exp(h)


def novel_gain(tokens_D, tokens_ref) -> int:
    """新 motif 增益 R_new = |M_D \\ M_ref|（contracts.md §3.8）。"""
    return len(set(tokens_D) - set(tokens_ref))


def accumulation(token_lists: list[list[str]], grid: list[int]) -> list[int]:
    """前缀 richness 曲线：grid 每个档位 N 返回前 N 个结构的 token richness。

    token_lists 按结构顺序（第 k 项 = 第 k 个结构的 token 列表）；grid 值超出
    结构数时钳制到 len(token_lists)（保守，已文档化）。返回与 grid 等长的 list[int]，
    richness 用契约默认 n_min=1（count > 1 才算覆盖）。
    """
    order = sorted(range(len(grid)), key=lambda k: grid[k])
    counter: dict[str, int] = {}
    acc = [0] * len(grid)
    pos = 0
    n_struct = len(token_lists)
    for k in order:
        target = min(int(grid[k]), n_struct)
        while pos < target:
            for t in token_lists[pos]:
                counter[t] = counter.get(t, 0) + 1
            pos += 1
        acc[k] = richness(counter)
    return acc
