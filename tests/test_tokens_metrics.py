"""metrics 数值断言（contracts.md §3.8）。"""

from __future__ import annotations

import math
import random

import pytest

from crysh.metrics import accumulation, effective_diversity, novel_gain, richness


def test_richness_strict_greater_than():
    counts = {"a": 5, "b": 2, "c": 1}
    assert richness(counts) == 2          # 契约默认 n_min=1：count > 1
    assert richness(counts, n_min=0) == 3  # count > 0 = 全部 distinct
    assert richness(counts, n_min=4) == 1  # 只有 a
    assert richness({}) == 0
    assert richness({}, n_min=0) == 0


def test_effective_diversity_values():
    assert effective_diversity({"a": 10}) == 1.0                     # 单一 token
    assert effective_diversity({"a": 5, "b": 5}) == pytest.approx(2.0, abs=1e-12)  # e^{ln2}
    assert effective_diversity({}) == 0.0
    # 3:1 → e^H，H = -(0.75 ln0.75 + 0.25 ln0.25)
    h = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert effective_diversity({"a": 3, "b": 1}) == pytest.approx(math.exp(h), abs=1e-12)
    # p=0 项不影响
    assert effective_diversity({"a": 3, "b": 1, "c": 0}) == pytest.approx(math.exp(h), abs=1e-12)


def test_novel_gain():
    D = {"a", "b", "c"}
    ref = {"b", "c", "d"}
    assert novel_gain(D, ref) == 1          # {"a"}
    assert novel_gain(D, D) == 0
    assert novel_gain(set(), ref) == 0
    assert novel_gain(D, set()) == 3


def test_accumulation_basic():
    token_lists = [["t1"], ["t1"], ["t2"], ["t2"], ["t2"]]
    assert accumulation(token_lists, [2, 5]) == [1, 2]
    # grid 超界 → 钳制到结构数
    assert accumulation(token_lists, [200]) == [2]
    assert accumulation(token_lists, [0]) == [0]


def test_accumulation_monotonic_and_matches_richness():
    rng = random.Random(42)
    vocab = [f"t{k}" for k in range(8)]
    token_lists = [
        [rng.choice(vocab) for _ in range(rng.randint(1, 6))] for _ in range(60)
    ]
    grid = [5, 10, 20, 40, 60, 100]
    acc = accumulation(token_lists, grid)
    assert all(b >= a for a, b in zip(acc, acc[1:]))          # 单调不减
    # 末档（钳制到 60）= 全量 richness
    full = {}
    for lst in token_lists:
        for t in lst:
            full[t] = full.get(t, 0) + 1
    assert acc[-1] == richness(full)


def test_accumulation_permutation_invariance_of_final_value():
    # 结构顺序无关（同 multiset 同 richness）
    rng = random.Random(7)
    vocab = [f"t{k}" for k in range(6)]
    lists_a = [[rng.choice(vocab) for _ in range(4)] for _ in range(30)]
    lists_b = list(reversed(lists_a))
    a = accumulation(lists_a, [30])
    b = accumulation(lists_b, [30])
    assert a == b
