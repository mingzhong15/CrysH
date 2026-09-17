"""L3 — 局域建筑块（motif 实例）的数值本体：m = (Z, CN, G, C, D)。

把"一个位点的局域环境"从散落特征提升为**一等公民的逐位点记录**：

===========================  ====================================================
轴                          字段
===========================  ====================================================
C  配位数（coordination）    `cn`, `cn_eff`, `cn_species`, `cn_by_lambda`, `p_cn`
C  壳层证据                   `shell_gap`, `cn_shell`, `cn1`, `cn2`, `shell_conf`
G  几何                      交给 :mod:`crysh.geometry`（本模块只给径向/角度统计）
C  邻居化学                   `neighbor_symbols`/`neighbor_counts`、`h_neigh`
D  畸变                       `r_tilde_mean/std`、`bond_cv`、`d_i`
===========================  ====================================================

设计取舍（相对 `plan-mapper-v2.md` §1.1 的清单）
------------------------------------------------
- **不落盘 r̃ 原谱**（原方案 #10 也砍了）：44M 位点存全谱不划算；本模块只给汇总量
  （`r_tilde_mean/std`、`bond_cv`），需要时用 `kernels.normalized_neighbor_table` 重算。
- **角度直方图不落盘**：`geometry` 的 q4/q6 已代表角度轴（原方案 #12 同结论）。
- **`cn_by_lambda` 存 list**：λ 网格可变，故用 list + 配置里的网格，不用 `d_090`
  那种定名列（那是 L1 冻结 schema 的做法，L3 不重复）。
- **`d_i` 是 interim 组合量**：`bond_cv`（径向离散）× 角度无序度的加权，**不是 CSM**。
  版本化在 meta 里（`d_i_version`），将来换 ChemEnv 的连续对称性度量时同名替换。
- **CN 双通道**：`cn` = 契约口径（`coord.coordination` 的有向邻居计数，冻结语义）；
  `cn_shell` = 归一化距离最大跳变处的壳层边界（几何证据）。两者**故意都保留**：
  不一致时 `shell_conf` 会低，正是"这个位点的 CN 是否可信"的信号。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import log

import numpy as np
from ase import Atoms

from crysh.bond import BondGraph
from crysh.config import MapperConfig
from crysh.kernels import NeighborList, masked_graph, neighbor_list, normalized_neighbor_table

__all__ = ["SiteEnvironment", "local_environments", "local_environment", "site_table"]

#: `d_i` 的版本标记（换核时改这里 + meta，不改列名）
D_I_VERSION = "interim-1"

#: 壳层分析的尺度（都无量纲，见 `_shell_split` / `_shell_confidence`）
_GAP_SCALE = 0.20         # 相对跳变 Δ/d_min 到 0.20 就算"壳层清晰"
_SHELL_SCAN_N = 26        # 壳层搜索只看最近 N 条邻居（约 2–3 个配位壳）
_PERSIST_FLOOR = 0.35     # CN 在 λ 网格上的持久度下限

#: 角度无序度：用现有几何特征里的 q4/q6 与理想点距离的近似（不做新计算）
_D_I_RADIAL_W = 0.5
_D_I_ANGULAR_W = 0.5


@dataclass
class SiteEnvironment:
    """一个位点的局域环境（m 实例）。字段名即 site 表列名。"""

    structure_id: str
    site_idx: int
    symbol: str
    Z: int

    # --- C：配位数 ---
    cn: int                       # 契约口径（coord.coordination）
    cn_eff: float                 # Σ_j exp(−α(r̃−1))，见 `cn_eff`
    cn_species: str               # 规范串 "O4F2"（按元素符号字母序）
    cn_by_lambda: list[int]       # 与 cfg.lambdas 同序
    p_cn: float                   # #{λ : CN(λ) == cn} / len(lambdas)

    # --- C：壳层证据 ---
    shell_gap: float              # max_n Δ_n（归一化距离排序的相邻跳变）
    cn_shell: int                 # argmax 处（跳变后的邻居数）
    cn1: int | float              # 第一壳 CN（gap 不显著时为 NaN）
    cn2: int | float              # 第二壳 CN（同上）
    shell_conf: float             # 0..1，CN 是否可信

    # --- C：邻居化学 ---
    neighbor_symbols: list[str]
    neighbor_counts: dict[str, int]
    h_neigh: float                # 邻居元素比例香农熵

    # --- D：畸变（interim 组合量，非 CSM）---
    r_tilde_mean: float
    r_tilde_std: float
    bond_cv: float                # σ_d / d̄
    d_i: float

    # --- G：几何标签（由调用方或几何层填；本模块算不出标签）---
    geometry: str = ""

    def to_row(self) -> dict:
        """摊平成一行（list/dict 列保持原样，供 parquet 的 list/map 类型）。"""
        return asdict(self)


def _cn_eff(d: np.ndarray, alpha: float) -> float:
    """有效配位数 ``Σ_j exp(−α(d_j − d_min))``，`α` 单位 1/Å。

    以**该位点最近邻距离**为参照（不是共价半径和）：这样理想配位壳里每个邻居权重
    都是 1（fcc Al 的 12 个等距邻居 → cn_eff = 12，与硬 CN 一致），而更远的壳按
    距离指数衰减。用 r0(共价和) 当参照会被系统性拉偏——实测 fcc Al 的
    d/r0 = 1.18，12 个邻居只加出 4.8，把"12 配位"这件事说没了。

    默认 α=2.0：近邻外 0.3 Å 处权重 0.55、0.5 Å 处 0.37、1.0 Å 处 0.14。
    """
    if d.size == 0:
        return 0.0
    return float(np.sum(np.exp(-float(alpha) * (d - float(np.min(d))))))


def _species_string(symbols: list[str]) -> str:
    """规范配位串："O4F2"（元素符号字母序 + 计数；空壳为 ""）。"""
    counts = Counter(symbols)
    return "".join(f"{el}{counts[el]}" for el in sorted(counts))


def _shannon_entropy(symbols: list[str]) -> float:
    """邻居元素比例的香农熵（nats）；单元素或空壳为 0。"""
    if not symbols:
        return 0.0
    counts = Counter(symbols)
    total = float(len(symbols))
    return float(-sum((c / total) * log(c / total) for c in counts.values()))


def _shell_split(d_sorted: np.ndarray) -> tuple[float, int, int]:
    """找**第一配位壳**的边界：在最近邻附近取径向分布里最大的那条缝。

    只看前 `_SHELL_SCAN_N = 26` 条邻居（约等于 2–3 个配位壳）——再往后是远壳，
    密度高、缝已经不携带"第一壳在哪结束"的信息。

    为什么不用 `cn` 当边界：bcc 金属的 pair 参考键长（共价半径和）恰好落在第一壳
    （8 个，2.74 Å）与第二壳（6 个，3.16 Å）之间，λ*=1.2 时**两壳都进键集**
    （cn = 14，契约判据 `d < λ·r0` 使然），但几何上的第一壳仍是 8。用键集边界当
    参照反而会把 `cn` 的口径问题传染给壳层分析（实测：bcc W 被带成 14）。

    被否掉的写法（都在实测里翻过车，记下免得再走回头路）：
    - 整张表取最大缝（不限前 N 条）：bcc/高配位结构的第二壳→第三壳缝更大，
      边界会落到 14 甚至 18；
    - `d ≤ 1.5·d_min` 截窗口：窗口正好截止在缝之前（diamond Si 的缝在 2.35→3.84，
      窗口只到 3.53），永远看不到缝。
    前 26 条 + 最大缝在 44 个 ground-truth 结构上全部落在教科书壳层上（见
    `tests/test_localenv.py` 的钉死表）。

    Returns
    -------
    (gap, cn1, cn2)
        `gap` = 该缝 / d_min（无量纲，供置信度用）；`cn1` = 缝前邻居数（第一壳 CN）；
        `cn2` = 缝后邻居数。邻居 < 2 时返回 (0, n, 0)。
    """
    n = int(d_sorted.size)
    if n < 2:
        return 0.0, n, 0
    d_min = float(d_sorted[0])
    if d_min <= 0:
        return 0.0, n, 0
    window = d_sorted[:min(_SHELL_SCAN_N, n)]
    diffs = np.diff(window)
    if diffs.size == 0:
        return 0.0, n, 0
    idx = int(np.argmax(diffs))
    cn1 = idx + 1
    return float(diffs[idx] / d_min), cn1, n - cn1


def _shell_confidence(gap: float, p_cn: float, cn_agrees: bool) -> float:
    """壳层置信度 ∈ [0,1]：几何清晰度 × λ 持久度的几何平均，再按"键判据是否
    复现同一壳层"折半。

    三个证据合起来回答一个问题：**这个位点的 CN 能不能当整数用？**
    - `gap`（径向分布里第一壳边界有多清楚）；
    - `p_cn`（在 λ 网格上有多少比例的 λ 给出同一个 CN）；
    - `cn_agrees`（键判据算出的 CN 是否就等于第一壳 CN）。

    只在 `p_cn` 高（如 1.0，理想金刚石）但几何判据与键判据不一致时才折半——
    这种"稳定但口径不同"的情形值得降级提醒（例如小胞混叠、键判据把两壳合并）。
    """
    s_gap = min(1.0, max(0.0, gap / _GAP_SCALE)) if _GAP_SCALE > 0 else 0.0
    s_persist = min(1.0, max(0.0, (p_cn - _PERSIST_FLOOR) / (1.0 - _PERSIST_FLOOR)))
    conf = float(np.sqrt(s_gap * s_persist))
    if not cn_agrees and p_cn >= 0.99:
        conf *= 0.5
    return conf


def local_environments(atoms: Atoms, cfg: MapperConfig | None = None, *,
                       graph: BondGraph | None = None,
                       coord=None,
                       geometry_labels: list[str] | None = None,
                       nl: NeighborList | None = None) -> list[SiteEnvironment]:
    """逐位点算 m 实例。

    Parameters
    ----------
    atoms:
        结构。
    cfg:
        :class:`crysh.config.MapperConfig`（用 `lambdas` 网格、`cn_eff_alpha`）。
    graph, coord:
        可选的既有结果（records 里已经算过，避免重复劳动）。`coord` 需有 `.cn`。
    geometry_labels:
        逐位点几何标签（L4 的产物）；给了就填进 `geometry` 轴。
    nl:
        可选的共享邻居表（避免重复建表；不给则内部建一次）。

    Returns
    -------
    list[SiteEnvironment]
        长度 = 原子数；顺序即 `site_idx`。
    """
    cfg = cfg or MapperConfig()
    if not isinstance(atoms, Atoms):
        raise TypeError(f"atoms must be an ase.Atoms, got {type(atoms).__name__}")
    n_atoms = len(atoms)
    sid = str(atoms.info.get("structure_id", ""))
    symbols = [str(s) for s in atoms.get_chemical_symbols()]
    zs = [int(z) for z in atoms.numbers]

    if nl is None:
        nl = neighbor_list(atoms, cfg.cutoff_table, lam_max=cfg.kernel_lam_max)
    if graph is None:
        graph = masked_graph(nl, cfg.d_star_lambda)
    if coord is None:
        from crysh.coord import coordination

        coord = coordination(atoms, graph, cfg.cn_table)
    cn_contract = np.asarray(coord.cn, dtype=np.int64).ravel()
    if cn_contract.size != n_atoms:
        raise ValueError(f"coord.cn 长度 {cn_contract.size} != n_atoms {n_atoms}")
    if geometry_labels is not None and len(geometry_labels) != n_atoms:
        raise ValueError(
            f"geometry_labels 长度 {len(geometry_labels)} != n_atoms {n_atoms}")

    # 契约口径 CN 在各 λ 上的值（一次建表，8 次纯数组筛选）
    cn_by_lambda = np.empty((n_atoms, len(cfg.lambdas)), dtype=np.int64)
    for k, lam in enumerate(cfg.lambdas):
        g = masked_graph(nl, lam)
        cn_by_lambda[:, k] = np.bincount(np.asarray(g.i, dtype=np.int64),
                                         minlength=n_atoms)

    # 邻居表（按距离排序，覆盖到 lam_max）→ 逐位点切段；nbr 与 d/r̃ 同序
    site, d_sorted, r_tilde, nbr, mask = normalized_neighbor_table(nl, cfg.d_star_lambda)
    starts = np.searchsorted(site, np.arange(n_atoms), side="left")
    ends = np.searchsorted(site, np.arange(n_atoms), side="right")

    out: list[SiteEnvironment] = []
    for a in range(n_atoms):
        sl = slice(starts[a], ends[a])
        d_all = d_sorted[sl]                 # 建表上界内的全部邻居（按距离升序）
        mask_all = mask[sl]                  # 其中属于当前 λ 键集的那些
        dd = d_all[mask_all]                 # 键集内的距离 → 径向统计与 cn_eff 只看这些
        rt = r_tilde[sl][mask_all]           # 对应的无量纲距离
        nbr_syms = [symbols[int(b)] for b in nbr[sl][mask_all]]

        cn = int(cn_contract[a])
        cn_lam = cn_by_lambda[a]
        p_cn = float(np.count_nonzero(cn_lam == cn) / len(cfg.lambdas))

        # 壳层切分：纯几何找第一壳边界（与 λ 无关，见 _shell_split 的 bcc 说明）
        gap, cn1, cn2 = _shell_split(d_all)
        cn_shell = int(cn1)
        # 键判据与几何第一壳是否一致（不一致 → 置信度降级，见 _shell_confidence）
        cn_agrees = cn == cn_shell
        if gap < _GAP_SCALE:  # 缝不够显著 → 壳层解析不可信，对外保守给 NaN
            cn1 = cn2 = float("nan")

        if dd.size:
            r_mean = float(np.mean(rt))
            r_std = float(np.std(rt))
            d_mean = float(np.mean(dd))
            bond_cv = float(np.std(dd) / d_mean) if d_mean > 0 else float("nan")
        else:
            r_mean = r_std = bond_cv = float("nan")

        # d_i：径向离散 + 邻居化学无序的 interim 组合（非 CSM，见模块 docstring）
        h_n = _shannon_entropy(nbr_syms)
        h_max = log(max(2, len(set(nbr_syms)))) if len(set(nbr_syms)) > 1 else 1.0
        angular_disorder = min(1.0, h_n / h_max) if h_max > 0 else 0.0
        radial_disorder = min(1.0, abs(r_std) / abs(r_mean)) if r_mean else 0.0
        d_i = float(_D_I_RADIAL_W * radial_disorder + _D_I_ANGULAR_W * angular_disorder)

        out.append(SiteEnvironment(
            structure_id=sid, site_idx=a, symbol=symbols[a], Z=zs[a],
            cn=cn,
            cn_eff=_cn_eff(dd, cfg.cn_eff_alpha),
            cn_species=_species_string(nbr_syms),
            cn_by_lambda=[int(x) for x in cn_lam],
            p_cn=p_cn,
            shell_gap=gap, cn_shell=int(cn_shell), cn1=cn1, cn2=cn2,
            shell_conf=_shell_confidence(gap, p_cn, cn_agrees=cn_agrees),
            neighbor_symbols=sorted(set(nbr_syms)),
            neighbor_counts=dict(Counter(nbr_syms)),
            h_neigh=h_n,
            r_tilde_mean=r_mean, r_tilde_std=r_std, bond_cv=bond_cv, d_i=d_i,
            geometry=str(geometry_labels[a]) if geometry_labels is not None else "",
        ))
    return out


def local_environment(atoms: Atoms, site_idx: int, cfg: MapperConfig | None = None
                      ) -> SiteEnvironment:
    """单个位点（便捷入口；批量请用 `local_environments` 以免重复建表）。"""
    envs = local_environments(atoms, cfg)
    n = len(envs)
    if not -n <= site_idx < n:
        raise IndexError(f"site_idx {site_idx} 超出 [0, {n})")
    return envs[site_idx]


def site_table(atoms_or_envs, cfg: MapperConfig | None = None):
    """site 表（pandas DataFrame）。

    接受""结构""或已算好的 `SiteEnvironment` 列表；pandas 只在真正要表时才 import
    （核心路径保持 numpy+ase）。
    """
    import pandas as pd  # optional extra: tables

    if isinstance(atoms_or_envs, Atoms):
        envs = local_environments(atoms_or_envs, cfg)
    else:
        envs = list(atoms_or_envs)
    return pd.DataFrame([e.to_row() for e in envs])
