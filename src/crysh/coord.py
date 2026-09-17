"""Level 3 — Coordination（配位数统计 + 元素条件化低/高配位异常检测）。

契约：<KT>/contracts.md §3.6（签名冻结）。所有者：level3-coordination。

核心语义（与契约一致；疑义按最保守解释并记录于 level3-coordination/progress.md）：

- **CN 定义**：每个原子的**不同 (j, S) 近邻数（去重）**——j 为原子索引、S 为周期平移
  向量。对每条边 (i, j, S)：原子 i 的近邻 += (j, S)；当 i != j 时原子 j 的近邻
  += (i, -S)。i == j 的周期自像边只计 (j, S) 一次（不虚构反向端点，避免自像键
  反向重复计数）。同一 (j, S) 只计一次。
- **percentile = F_Z(cn) = P(CN <= cn | Z)**（经验 CDF，含同值；契约冻结）。
- **low**: p < 0.05；**high**: p > 0.95。
- 无表 / 表中缺 Z / 表中缺该 cn 值 → 该站点 percentile = NaN；
  low/high_cn_fraction 的分母恒为结构总原子数 N，NaN 站点**不计数为异常**
  （保守：未知站点不扩大异常率）。无表时 percentile 全 NaN 且两 fraction = 0。
- graph 参数可为契约 §3.2 BondGraph 或任何具有 `.i` / `.j` / `.S` / `.n_atoms`
  属性的对象（duck typing——level1 未合入时可用 mock / 回退图开发）。
  `.S` 须为 (E, 3) 整数平移向量。

v1.3-L3 增补守卫（纯增补字段 / 关键字参数，冻结签名不动；主会话 L3 评审批准）：

- **小胞镜像混叠守卫**（`CoordResult.cn_cell_warning` / `cn_cell_margin`，带默认值）：
  margin = 0.5·min(|a|,|b|,|c|) / max_pair_cutoff，其中 max_pair_cutoff =
  graph.lam × max(graph.pair_r0.values())；margin < 1（最大 pair cutoff 超过
  最短晶格矢量之半）→ cn_cell_warning=True。**warning ≠ 错误**：CN 的 (j,S)
  计数在混叠区仍严格正确——每个 (j,S) 对应唯一物理位置（周期镜像原子）；
  warning 只标记两种风险：(a) "化学邻居数"的物理解读需谨慎（同一原子索引的
  多个周期镜像可能同时成为近邻，小胞自相互作用进入键图）；(b) 极小胞下
  ASE 镜像枚举可能不全。margin ≥ 1 时对任意 (i, j) 至多一个 S 使 (j,S) 成边
  （无混叠；键判据为严格 d < cutoff，margin==1 亦安全）。保守路径：graph 缺
  .lam / .pair_r0、pair_r0 为空或值非法 → margin=+inf、不 warning（不误报）；
  病态胞（任一格矢长度 ≤ 0 或非有限，优先级最高）→ warning=True、margin=0
  （交 L0 pathological_cell_flag）。
- **percentile 表最小样本守卫**（`calibrate_cn_table(..., min_sites=CN_MIN_SITES)`）：
  有效站点数 n < min_sites 的元素不写入表 → 其 percentile 走 NaN 保守路径
  （不计为异常、不扩大异常率）。CN_MIN_SITES=100 为 pilot 建议值（2k/20k 稀有
  元素的经验 CDF 0.05/0.95 分位在极小样本下脆弱），待 20k/Phase-0 站点规模
  重评估；min_sites <= 0 表示不设门槛。老调用 `calibrate_cn_table(df)` 签名
  兼容（等价于显式传 min_sites=100）。

扩展辅助（非冻结签名，供批处理与测试使用）：
- `apply_cn_table(Z, cn, cn_table)` —— 站点级 percentile 查询（向量化）。
- `cn_histogram(cn)` —— CN 直方图（索引 0..16，CN>16 截断入末 bin）。
- `cn_table_coverage_report(records_df, min_sites=CN_MIN_SITES)` —— 元素级
  站点覆盖报告（配合 min_sites 门槛：{Z: sites, included}，含 0 站点元素）。
- `save_cn_table` / `load_cn_table` —— {Z: {"cdf": {cn: p}, "n": n}} 的 JSON 读写
  （JSON 键转字符串，load 恢复 int 键；save 的 meta 自动补 min_sites 缺省
  CN_MIN_SITES；load 兼容无 meta 的旧表，return_meta=True 可一并取回 meta）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import atomic_numbers

LOW_PERCENTILE = 0.05
HIGH_PERCENTILE = 0.95
CN_HIST_BINS = 17  # cn_hist 索引 0..16
# v1.3-L3 增补：percentile 表最小站点数门槛（pilot 建议值，待 20k/Phase-0
# 站点规模重评估；2k/20k 稀有元素的经验 CDF 0.05/0.95 分位在极小样本下脆弱）。
CN_MIN_SITES = 100


@dataclass
class CoordResult:
    """契约 §3.6 冻结字段 + v1.3-L3 纯增补守卫字段（带默认值，老构造不破坏）。"""
    cn: np.ndarray              # (N,) int   每原子配位数
    cn_percentile: np.ndarray   # (N,) float F_Z(cn)；无表/缺 Z/缺 cn 时为 NaN
    mean_cn: float
    low_cn_fraction: float      # p < 0.05 的站点占比（分母 = N；NaN 不计为异常）
    high_cn_fraction: float     # p > 0.95 的站点占比（分母 = N；NaN 不计为异常）
    cn_min: int
    cn_max: int
    # ---- v1.3-L3 纯增补：小胞镜像混叠守卫（warning ≠ 错误，见模块 docstring）----
    cn_cell_warning: bool = False         # True = 镜像混叠风险（物理解读需谨慎）
    cn_cell_margin: float = float("inf")  # 0.5·min(|a|,|b|,|c|)/max_pair_cutoff；<1 → warning


def _count_cn(i, j, S, n_atoms) -> np.ndarray:
    """每原子不同 (j, S) 近邻数（去重），见模块 docstring 的 CN 定义。"""
    i = np.asarray(i, dtype=np.int64).ravel()
    j = np.asarray(j, dtype=np.int64).ravel()
    S = np.asarray(S).reshape(-1, 3).astype(np.int64)
    if not (i.size == j.size == S.shape[0]):
        raise ValueError(
            f"graph 边数组长度不一致: i={i.size}, j={j.size}, S={S.shape[0]}")
    if i.size == 0:
        return np.zeros(n_atoms, dtype=np.int64)
    same = i == j
    fwd = np.column_stack([i, j, S])                       # i 的近邻 (j, S)
    rev = np.column_stack([j, i, -S])[~same]               # j 的近邻 (i, -S)，i!=j 时
    keys = np.vstack([fwd, rev])
    keys = np.unique(keys, axis=0)                         # (j, S) 去重
    if keys[:, 0].min() < 0 or keys[:, 0].max() >= n_atoms:
        raise ValueError("graph 边端点索引越界（超出 [0, n_atoms)）")
    return np.bincount(keys[:, 0], minlength=n_atoms).astype(np.int64)


def _lookup(mapping: dict | None, key: int) -> Any:
    """dict 查询：先 int 键后 str(int) 键（兼容 JSON 读入未转换的键）。"""
    if not mapping:
        return None
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key))


def apply_cn_table(Z, cn, cn_table: dict | None) -> np.ndarray:
    """站点级 percentile 查询：p_i = F_{Z_i}(cn_i)；无表/缺 Z/缺 cn 值 → NaN。"""
    Z = np.asarray(Z, dtype=np.int64).ravel()
    cn = np.asarray(cn, dtype=np.int64).ravel()
    if Z.shape != cn.shape:
        raise ValueError(f"Z 与 cn 长度不一致: {Z.shape} vs {cn.shape}")
    p = np.full(Z.shape, np.nan, dtype=np.float64)
    if not cn_table:
        return p
    for Zi in np.unique(Z):
        entry = _lookup(cn_table, int(Zi))
        if entry is None:
            continue
        cdf = entry.get("cdf")
        if not cdf:
            continue
        mask = Z == Zi
        for c in np.unique(cn[mask]):
            v = _lookup(cdf, int(c))
            if v is not None:
                p[mask & (cn == c)] = float(v)
    return p


def _cell_aliasing_guard(atoms, graph) -> tuple[bool, float]:
    """小胞镜像混叠守卫（v1.3-L3 增补，非冻结辅助；完整语义见模块 docstring）。

    返回 (cn_cell_warning, cn_cell_margin)：
    - margin = 0.5·min(|a|,|b|,|c|) / max_pair_cutoff，
      max_pair_cutoff = graph.lam × max(graph.pair_r0.values())；
    - margin < 1 → warning=True（最大 pair cutoff 超过最短晶格矢量之半，
      同一原子索引的多个周期镜像可能同时落在 cutoff 内）；
    - margin ≥ 1 → warning=False（对任意 (i,j) 至多一个 S 使 (j,S) 成边；
      键判据为严格 d < cutoff，margin==1 亦安全）；
    - 病态胞（任一格矢长度 ≤ 0 或非有限）→ (True, 0.0)——优先级最高、与
      graph 属性无关（交 L0 pathological_cell_flag）；
    - 保守路径（不误报）：graph 缺 .lam / .pair_r0、pair_r0 为空或 lam/r0
      值非法（非有限正数）→ (False, +inf)。
    """
    cell = np.asarray(atoms.cell, dtype=np.float64).reshape(3, 3)
    lengths = np.linalg.norm(cell, axis=1)
    if not bool(np.all(np.isfinite(lengths))) or bool(np.any(lengths <= 0.0)):
        return True, 0.0
    lam = getattr(graph, "lam", None)
    pair_r0 = getattr(graph, "pair_r0", None)
    if lam is None or not pair_r0:
        return False, float("inf")
    try:
        max_cut = float(lam) * float(max(pair_r0.values()))
    except (TypeError, ValueError, AttributeError):
        return False, float("inf")
    if not np.isfinite(max_cut) or max_cut <= 0.0:
        return False, float("inf")
    margin = 0.5 * float(np.min(lengths)) / max_cut
    return bool(margin < 1.0), float(margin)


def coordination(atoms, graph, cn_table: dict | None = None) -> CoordResult:
    """契约 §3.6：由键图计算每原子 CN 与元素条件化 percentile 及结构级汇总。

    graph：契约 §3.2 BondGraph（或带 .i/.j/.S/.n_atoms 的等价对象）；
    cn_table：{Z: {"cdf": {cn: percentile}, "n": count}}，无表 → percentile 全 NaN、
    low/high_cn_fraction = 0。

    v1.3-L3 增补：同时计算小胞镜像混叠守卫并写入 `cn_cell_warning` /
    `cn_cell_margin`（warning ≠ 错误——CN 的 (j,S) 计数在混叠区仍严格正确，
    语义与保守路径见模块 docstring 与 `_cell_aliasing_guard`）。
    """
    n = len(atoms)
    if n == 0:
        raise ValueError("atoms 为空，无法计算配位")
    g_n = int(getattr(graph, "n_atoms", -1))
    if g_n != n:
        raise ValueError(f"graph.n_atoms={g_n} 与 len(atoms)={n} 不一致")
    cn = _count_cn(graph.i, graph.j, graph.S, n)
    cell_warning, cell_margin = _cell_aliasing_guard(atoms, graph)
    Z = atoms.get_atomic_numbers()
    p = apply_cn_table(Z, cn, cn_table)
    low = float(np.count_nonzero(p < LOW_PERCENTILE) / n)
    high = float(np.count_nonzero(p > HIGH_PERCENTILE) / n)
    return CoordResult(
        cn=cn,
        cn_percentile=p,
        mean_cn=float(np.mean(cn)),
        low_cn_fraction=low,
        high_cn_fraction=high,
        cn_min=int(np.min(cn)),
        cn_max=int(np.max(cn)),
        cn_cell_warning=cell_warning,
        cn_cell_margin=cell_margin,
    )


def _element_z(sym) -> int | None:
    """元素符号 / 原子序数 → int 原子序数（无效返回 None）。"""
    try:
        if isinstance(sym, (int, np.integer)):
            z = int(sym)
            return z if 0 < z < 119 else None
        return atomic_numbers.get(str(sym).capitalize())
    except Exception:
        return None


def _parse_site_records(records_df) -> tuple[np.ndarray, np.ndarray, set]:
    """解析 records_df 为 (Z, cn, seen_Z)（calibrate / 覆盖报告共用，非冻结辅助）。

    - Z / cn：**有效站点**数组（Z > 0 且 cn >= 0，行序保留）；
    - seen_Z：输入中出现过的元素 int 原子序数集合——含所有有有效站点的元素，
      也含**站点数 0 的元素**（长表：Z>0 但 cn<0 的行；结构级表：cn_mean
      四舍五入为负的元素）。Z<=0 的行无法归入元素，不计入。
    输入格式与列校验同 calibrate_cn_table（见其 docstring），不合法时 ValueError。
    """
    cols = set(records_df.columns)
    seen: set[int] = set()
    if {"Z", "cn"} <= cols:
        Z = np.asarray(records_df["Z"], dtype=np.int64).ravel()
        cn = np.asarray(records_df["cn"], dtype=np.int64).ravel()
        if Z.size != cn.size:
            raise ValueError(f"Z 与 cn 长度不一致: {Z.size} vs {cn.size}")
        seen = {int(z) for z in Z.tolist() if z > 0}
    elif {"elements", "n_atoms", "cn_mean"} <= cols:
        z_list: list[int] = []
        c_list: list[int] = []
        for elements, n_atoms, mean in zip(
                records_df["elements"], records_df["n_atoms"],
                records_df["cn_mean"]):
            if elements is None or n_atoms is None:
                continue
            try:
                mean_f = float(mean)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(mean_f):
                continue
            c = int(round(mean_f))
            for sym, cnt in zip(elements, n_atoms):
                Zv = _element_z(sym)
                if Zv is None:
                    continue
                try:
                    n_cnt = int(cnt)
                except (TypeError, ValueError):
                    continue
                if n_cnt <= 0:
                    continue
                seen.add(Zv)
                z_list.extend([Zv] * n_cnt)
                c_list.extend([c] * n_cnt)
        Z = np.asarray(z_list, dtype=np.int64)
        cn = np.asarray(c_list, dtype=np.int64)
    else:
        raise ValueError(
            "records_df 需为站点长表（列 Z, cn）或结构级表"
            "（列 elements, n_atoms, cn_mean）；实际列: " + ", ".join(sorted(cols)))
    ok = (Z > 0) & (cn >= 0)
    return Z[ok], cn[ok], seen


def calibrate_cn_table(records_df, min_sites: int = CN_MIN_SITES) -> dict:
    """由站点级或结构级记录校准 P(CN|Z)，返回 {Z: {"cdf": {cn: p}, "n": n}}。

    两种输入（列名由本模块定义并在此文档化）：
    1. **站点长表**（推荐）：列 ``Z``（int 原子序数）与 ``cn``（int 配位数），一行一站；
    2. **结构级表**（降级近似）：列 ``elements``（元素符号 list）、``n_atoms``
       （与 elements 对齐的每元素原子数 list）与 ``cn_mean``（float，结构平均 CN）——
       结构内每个站点都赋 ``cn = round(cn_mean)``，仅用于无站点级数据时的引导校准。

    percentile = F_Z(cn) = P(CN <= cn | Z)（经验 CDF，含同值；契约冻结）。
    Z 与 cn 均为 int 键；无效行（Z<=0 或 cn<0）剔除。

    min_sites : int，默认 ``CN_MIN_SITES``（=100）
        **最小样本守卫（v1.3-L3 纯增补关键字参数）**：有效站点数 n < min_sites 的
        元素**不写入表**——查表时该元素走 NaN 保守路径（apply_cn_table 缺 Z →
        NaN，站点不计为异常、不扩大异常率）。100 为 pilot 建议值（2k/20k 稀有
        元素的经验 CDF 0.05/0.95 分位在极小样本下脆弱），待 20k/Phase-0 站点
        规模重评估；``min_sites <= 0`` 表示不设门槛（全包含）。老调用
        ``calibrate_cn_table(df)`` 签名兼容（等价于显式传 ``min_sites=100``）。
    """
    Z, cn, _seen = _parse_site_records(records_df)
    threshold = int(min_sites) if min_sites is not None else 0
    table: dict[int, dict[str, Any]] = {}
    for Zi in np.unique(Z):
        c = cn[Z == Zi]
        n = int(len(c))
        if threshold > 0 and n < threshold:
            continue
        vals, counts = np.unique(c, return_counts=True)
        cum = np.cumsum(counts) / n
        table[int(Zi)] = {
            "cdf": {int(v): float(f) for v, f in zip(vals, cum)},
            "n": n,
        }
    return table


def cn_table_coverage_report(records_df, min_sites: int = CN_MIN_SITES) -> dict:
    """元素级站点覆盖报告（v1.3-L3 增补，非冻结辅助；配合 min_sites 门槛）。

    输入格式同 calibrate_cn_table（站点长表 / 结构级表）。返回::

        {
            "min_sites": int,                      # 生效门槛（<=0 = 不设门槛）
            "elements": {Z: {"sites": int,         # 该元素有效站点数
                             "included": bool}},   # 是否会写入校准表
            "n_included": int, "n_excluded": int,
            "total_sites": int,                    # 全部元素有效站点数之和
        }

    - Z 为 int 原子序数键；**含站点数 0 的元素**（出现于输入但无有效站点，
      见 ``_parse_site_records`` 的 seen_Z 语义），其 included=False；
    - ``included == (min_sites <= 0) or (sites >= min_sites)``，与
      calibrate_cn_table 的写入门槛逐元素一致。
    """
    Z, _cn, seen = _parse_site_records(records_df)
    sites: dict[int, int] = {}
    for z in Z.tolist():
        z = int(z)
        sites[z] = sites.get(z, 0) + 1
    threshold = int(min_sites) if min_sites is not None else 0
    elements: dict[int, dict[str, Any]] = {}
    n_in = n_ex = 0
    for z in sorted(set(sites) | set(seen)):
        n = int(sites.get(z, 0))
        included = threshold <= 0 or n >= threshold
        elements[z] = {"sites": n, "included": bool(included)}
        if included:
            n_in += 1
        else:
            n_ex += 1
    return {
        "min_sites": threshold,
        "elements": elements,
        "n_included": int(n_in),
        "n_excluded": int(n_ex),
        "total_sites": int(Z.size),
    }


def cn_histogram(cn, n_bins: int = CN_HIST_BINS) -> list[int]:
    """CN 直方图 list[int]，索引 0..n_bins-1（默认 0..16）；CN>n_bins-1 截断入末 bin。"""
    cn = np.asarray(cn, dtype=np.int64).ravel()
    clipped = np.clip(cn, 0, n_bins - 1)
    return np.bincount(clipped, minlength=n_bins).tolist()


def save_cn_table(cn_table: dict, json_path, meta: dict | None = None) -> Path:
    """写 {Z: {"cdf": {cn: p}, "n": n}} 为 JSON（键转字符串）。

    文件结构: ``{"meta": {...}, "table": {"3": {"n": .., "cdf": {"0": .., ...}}, ...}}``

    v1.3-L3：meta 自动补 ``min_sites`` 缺省 ``CN_MIN_SITES``（=100）。若表实际
    以其他门槛校准（如 min_sites=0），调用方须在 meta 显式传入以记录真值。
    """
    meta_out = dict(meta or {})
    meta_out.setdefault("version", "v1")
    meta_out.setdefault("low_percentile", LOW_PERCENTILE)
    meta_out.setdefault("high_percentile", HIGH_PERCENTILE)
    meta_out.setdefault("min_sites", CN_MIN_SITES)
    ser_table = {}
    for Z in sorted(cn_table):
        entry = cn_table[Z]
        ser_table[str(Z)] = {
            "n": int(entry.get("n", -1)),
            "cdf": {str(c): float(p) for c, p in sorted(entry["cdf"].items())},
        }
    payload = {"meta": meta_out, "table": ser_table}
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return json_path


def load_cn_table(json_path, return_meta: bool = False):
    """读取 save_cn_table 写出的 JSON，恢复 int 键的 {Z: {"cdf": ..., "n": ...}}。

    兼容无 meta 的裸表格式（{"3": {"cdf": {...}}}）。
    v1.3-L3：``return_meta=True`` → 返回 ``(table, meta)``，旧表无 meta 时
    meta 为空 dict；默认返回值仍为纯 table（老调用不破坏）。
    """
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if "table" in raw:
        table_raw, meta = raw["table"], dict(raw.get("meta") or {})
    else:
        table_raw, meta = raw, {}
    table: dict[int, dict[str, Any]] = {}
    for z_str, entry in table_raw.items():
        cdf = {int(c_str): float(p) for c_str, p in entry["cdf"].items()}
        table[int(z_str)] = {"cdf": cdf, "n": int(entry.get("n", -1))}
    if return_meta:
        return table, meta
    return table
