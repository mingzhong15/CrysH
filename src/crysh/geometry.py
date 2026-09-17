"""Level 4 — 局域配位几何识别（FastGeom v0，规则版 + v1.3-L4 前置金属扩展）。

所有权：level4-geometry 子任务（contracts.md §2 / §3.7，签名与特征顺序冻结）。
本模块是 v0 规则分类器（CN 路由 + 角直方图 + q4/q6 到理想参考的距离）；
ChemEnv 蒸馏为 v1，不在本模块。v1.3-L4 前置扩展（集成者预先授权，主会话
L4 评审结论）：GEOMETRY_LABELS 新增三个金属参考标签 cuboctahedral/hcp_like/
bcc_like（纯规则扩展，不动 ChemEnv 蒸馏 v1 计划）——动机：2k 上 77.6% 结构
top label 为 other，根因是 CN=12 金属环境（fcc/hcp）与键图 CN=14 的 bcc
此前无标签全落 other。

依赖说明（contracts.md §6 + v1.1 修订 + 任务书）：
- geometry_features(atoms, graph) 消费 BondGraph（contracts.md §3.2 字段
  i/j/S/d/n_atoms）。**v1.1：BondGraph 为双向邻接**（(i,j,S) 与 (j,i,-S)
  都保留，仅去重完全相同的三元组），因此本模块的每站点计数
  `m = (g.i == k)`（等价 np.bincount(g.i)）直接给出正确 CN；**不做事后
  对称化**（会双计）。历史教训：v1.0 单向 (i<j) 存储下，同一逻辑系统性
  漏计高索引原子（曾致 64.6% 站点 CN=0），v1.1 已由 level1 修复。
- graph 为 None 时内部经 build_graph() 自算：`try import crysh.bond` 成功则用
  level1 真实现；失败（尚未合入）则回退 ase.neighborlist
  (natural_cutoffs×λ, bothways=True)——ASE 对 per-atom 球半径的语义是
  "球重叠即成邻"（d_ij < c_i + c_j），与 BondGraph 的 d_ij < λ·r0_ij 定义
  严格等价（r0_ij = r_cov_i + r_cov_j），且双向保留与 v1.1 语义一致。
  **不阻塞、不死等。**
- 键图标定：v1.1 interim λ* = 1.20（crysh.dimensionality.D_STAR_LAMBDA，pipeline
  L2–L5 公共键图）。build_graph 缺省 lam 取 λ*；phase-0 pair-cutoff 校准表
  落地后取代此常数。
- 测试用自写 mock graph（level4-geometry/tests/test_geometry.py 内联
  SimpleNamespace，按 §3.2 字段构造，双向）。

特征（frozen 顺序，contracts.md §3.7）：
    [cn, d_mean/r0, d_std/d_mean, d_min/d_mean,
     cosθ 直方图 8 bins 等宽 [-1,1]（键角 ij-ik，归一化占比，和=1）, q4, q6]
  - r0 = mean_j (r_cov(Z_i) + r_cov(Z_j))，即中心原子与**已成键邻居**的
    covalent 半径和均值（任务书选项 A 的第一种，文档化选择）；f1=d_mean/r0
    是相对键伸长（≈ λ 效应，理想同元素多面体 ≈ d/(2 r_cov)）。
  - 键角 ij-ik：中心 i 处两邻居方向 u_j·u_k 的夹角；8 bins 等宽 [-1,1]，
    归一化占比；CN<2 时全 0。CN>12 时按图中边序确定性取前 12 条边配对角
    （C(12,2)=66 对），保持 O(CN²) 有界（文档化近似，v0）。
  - q4/q6：Steinhardt 秩参量。**实数球谐**（numpy 手写，关联 Legendre 递推，
    含 Condon-Shortley 相位——相位只改变基符号，不改变 |q_l|），邻居等权：
        q_l = sqrt(4π/(2l+1) · Σ_m ⟨Y_lm⟩²)，⟨Y_lm⟩ = (1/CN) Σ_j Y_lm(u_j)
    实数基与复数基差一个酉变换，故 q_l 与文献（复数）定义数值一致；
    球谐加法定理给出解析校验（q_l² = (1/N²)Σ_ij P_l(cosγ_ij)），
    见 level4-geometry/scripts/calibrate_ideals.py 与 tests。
  - CN=0：f1=f2=f3=0、直方图 0、q4=q6=0；CN=1：f2=0、f3=1、q4=q6=1
    （单矢量 ⟨Y_lm⟩=Y_lm(u)，Σ_m Y²=(2l+1)/4π）。

分类（classify_geometry，v0 规则，全部阈值由理想 fixture 实测校准，
calibrate_ideals.py 输出 data/ideal_q_calibration.tsv，tests 再断言）：
  CN 路由（contracts.md §3.7 + 任务书 B + v1.3-L4 前置金属扩展）：
    2 → linear | other（bent）；3 → trigonal_planar | other；
    4 → tetrahedral | square_planar；5 → trigonal_bipyramidal | square_pyramidal；
    6 → octahedral | trigonal_prismatic；8 → cubic；
    12 → icosahedral | cuboctahedral | hcp_like（三候选，dists/argsort/
         d1-d2 margin 结构不变，d2=次近参考）；14 → bcc_like（单候选）；
    0/1/7/9/10/11/13/15/≥16 → other（conf=1.0，确定性路由，无决策边界）。
  判定量：d_lab = ‖(q4,q6) − REF[lab]‖（2 维欧氏距离），d1=最小，d2=次小。
  - d1 ≤ T_MATCH：label=最近参考；margin = d2 − d1（单候选路由 margin =
    T_MATCH − d1）；conf = margin/(margin + LAMBDA)。
  - d1 > T_MATCH：label="other"；conf = (d1 − T_MATCH)/(d1 − T_MATCH + LAMBDA_OTHER)。
  - conf < 0.7 → label="ambiguous"（contracts.md §3.7）。
  - confidence 语义（文档化）：到最近决策边界的距离的饱和映射到 [0,1]；
    边界附近（margin≲0.023 或 d1−T_MATCH≲0.047）→ ambiguous，宁可不分类。
  - CN=2 特例（任务书 "linear vs bent（cosθ 峰）"）：cosθ 必须落在直方图
    第 0 bin（[−1,−0.75)，即 θ>138.6° 的"峰值在 −1"粗判），再用精确的
    (q4,q6) 距离细化边界与 confidence（CN=2 时 q_l=sqrt((1+P_l(cosθ))/2)
    是单键角的双射函数，两判据单调等价，粗判只作 gate）。

理想参考值 REF_GEOMETRY（解析推导 + 实测校准，见 calibrate_ideals.py；
金属三项为 v1.3-L4 前置扩展，理想壳 fixture 用**本模块 _q_l 实测**后写入，
与文献值 sanity 断言 ±0.005，见 calibrate_metallic_ideals.py / tests）：
    linear(CN2)=(1,1)；trigonal_planar=(0.3750,0.7408)；tetrahedral=(0.5092,0.6285)；
    square_planar=(0.8292,0.5863)；trigonal_bipyramidal=(0.6250,0.4556)；
    square_pyramidal=(0.7746,0.4000)；octahedral=(0.7638,0.3536)；
    trigonal_prismatic=(0.4286,0.1270)；cubic=(0.5092,0.6285)（与四面体
    (q4,q6) 简并，靠 CN 路由区分）；icosahedral=(0.0,0.6633)；
    cuboctahedral=(0.1909,0.5745)（fcc 12 近邻壳，文献 0.190945/0.574524）；
    hcp_like=(0.0972,0.4848)（hcp 12 近邻壳，理想 c/a=√(8/3)，文献
    0.097221/0.484761）；bcc_like=(0.0364,0.5107)（bcc 8+6=14 近邻壳，
    文献 0.036370/0.510688）。
  阈值：T_MATCH=0.12（候选对最近距离 tbp/spy=0.1596 → 理想 conf=0.941>0.9；
  单候选路由理想 conf=0.12/0.13=0.923>0.9；CN=12 三候选最近对 fcc–hcp 参考
  距离 0.1298 > T_MATCH → 理想 conf=0.1298/0.1398=0.928>0.9，icosahedral
  的 d2=0.2033 → conf=0.953）；LAMBDA=0.01；LAMBDA_OTHER=0.02。

已知局限（v0 + v1.3-L4 前置金属扩展，记录于 progress.md）：CN=7（pentagonal
bipyramidal）与 CN=9/10/11/13/15+ 仍无标签 → other（留 ChemEnv 蒸馏 v1）；
tetrahedral 与 cubic 的 (q4,q6) 解析简并靠 CN 路由区分（4↔8，不变）；
bcc_like 的 **CN=14 语义来自键图的 8+6**——bcc 次近邻（6 个 a 轴向）落在
校准 cutoff（λ*×共价和）之内，故键图 CN=14，而非教科书 CN=8；若 cutoff
收紧到只含 8 近邻，站点会走 CN=8 路由判 cubic（顶点=立方体角落）。
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# 契约常量（contracts.md §3.7，frozen；v1.3-L4 前置金属扩展：+cuboctahedral/
# hcp_like/bcc_like，插在 icosahedral 与 other 之间，顺序固定）
# ---------------------------------------------------------------------------
GEOMETRY_LABELS = ["linear", "trigonal_planar", "tetrahedral", "square_planar",
                   "trigonal_bipyramidal", "square_pyramidal", "octahedral",
                   "trigonal_prismatic", "cubic", "icosahedral", "cuboctahedral",
                   "hcp_like", "bcc_like", "other", "ambiguous"]
N_FEATURES = 14

# 特征列位（文档用，frozen 顺序）
F_CN, F_DMEAN_R0, F_DSTD_MEAN, F_DMIN_MEAN = 0, 1, 2, 3
F_HIST_START, F_HIST_END = 4, 12          # 8 bins 等宽 [-1,1]，归一化占比
F_Q4, F_Q6 = 12, 13
HIST_BINS = 8
HIST_RANGE = (-1.0, 1.0)
N_HIST_NEIGH = 12                          # CN>12 时取前 12 条边配对角

# ---------------------------------------------------------------------------
# 分类器校准（理想 fixture 实测，见 calibrate_ideals.py / calibrate_metallic_
# ideals.py / tests）。金属三项为理想壳（fcc 12 顶点 cuboctahedron / hcp 理想
# c/a=√(8/3) 12 近邻壳 / bcc 8+6=14 近邻壳）经**本模块 _q_l 实测**的值（4 位
# 小数），与文献 sanity 断言 ±0.005（calibrate_metallic_ideals.py 可复现）。
# ---------------------------------------------------------------------------
REF_GEOMETRY = {
    "linear": (1.0000, 1.0000),            # CN=2（仅作参考；CN=2 走 cosθ 特例）
    "trigonal_planar": (0.3750, 0.7408),   # CN=3
    "tetrahedral": (0.5092, 0.6285),       # CN=4
    "square_planar": (0.8292, 0.5863),     # CN=4
    "trigonal_bipyramidal": (0.6250, 0.4556),   # CN=5
    "square_pyramidal": (0.7746, 0.4000),       # CN=5
    "octahedral": (0.7638, 0.3536),        # CN=6
    "trigonal_prismatic": (0.4286, 0.1270),     # CN=6
    "cubic": (0.5092, 0.6285),             # CN=8（与四面体 q 简并，CN 路由区分）
    "icosahedral": (0.0000, 0.6633),       # CN=12
    "cuboctahedral": (0.1909, 0.5745),     # CN=12 fcc（实测 0.190941/0.574524）
    "hcp_like": (0.0972, 0.4848),          # CN=12 hcp（实测 0.097222/0.484762）
    "bcc_like": (0.0364, 0.5107),          # CN=14 = 8+6（实测 0.036370/0.510688）
}
T_MATCH = 0.12        # 距理想参考 (q4,q6) 的"匹配"上界
LAMBDA = 0.01         # 两候选路由 confidence 边界宽度
LAMBDA_OTHER = 0.02   # other 路由 confidence 边界宽度
CONF_AMBIG = 0.7      # conf<0.7 → ambiguous（契约 §3.7）

# v1.3-L4：CN=12 三候选（icosahedral/cuboctahedral/hcp_like，fcc–hcp 参考距离
# 0.1298 > T_MATCH，理想 conf=0.928）；CN=14 单候选 bcc_like（margin=T_MATCH−d1，
# 理想 conf=0.923）。其余 CN 不变：0/1/7/9/10/11/13/15/≥16 → other。
_ROUTES = {
    2: ("linear",),
    3: ("trigonal_planar",),
    4: ("tetrahedral", "square_planar"),
    5: ("trigonal_bipyramidal", "square_pyramidal"),
    6: ("octahedral", "trigonal_prismatic"),
    8: ("cubic",),
    12: ("icosahedral", "cuboctahedral", "hcp_like"),
    14: ("bcc_like",),
}

# ---------------------------------------------------------------------------
# BondGraph 依赖（level1 未合入时回退，不死等）
# ---------------------------------------------------------------------------
# 键图与 λ* 口径来自同包的 canonical 实现（R2：不再有"缺 level1 就降级"的分支）
from crysh.bond import build_bond_graph as _bond_build_bond_graph
from crysh.config import D_STAR_LAMBDA as _D_STAR_LAMBDA


def build_graph(atoms, lam: float | None = None):
    """构造 BondGraph（优先 level1 真实现，否则 ase.neighborlist 回退）。

    lam=None 时用 v1.1 interim λ*（D_STAR_LAMBDA=1.20，pipeline L2–L5 公共
    键图标定）。回退语义与 contracts.md §3.2（v1.1 双向邻接）等价：
    per-atom 球半径 c = lam·r_cov(Z)，ASE "球重叠即成邻" ⇒ 边条件
    d_ij < lam·(r_cov_i + r_cov_j)，bothways=True 双向保留，与
    build_bond_graph(atoms, lam=lam) 一致。返回 SimpleNamespace
    （i/j/S/d/n_atoms/lam/cutoff_mode/pair_r0，字段与 §3.2 BondGraph 相同）。
    """
    if lam is None:
        lam = float(_D_STAR_LAMBDA)
    if _bond_build_bond_graph is not None:  # pragma: no cover - 集成后生效
        return _bond_build_bond_graph(atoms, lam=lam)
    return _ase_fallback_graph(atoms, lam=lam)


def _ase_fallback_graph(atoms, lam: float):
    """ase.neighborlist 回退图（双向邻接，等价 level1 build_bond_graph）。

    ASE 3.29 对 per-atom 球半径本就返回双向对；为确保跨版本可移植性，
    缺反向边 (j,i,-S) 时显式补齐（基于 (i,j,S) 三元组集合判重，绝不双计）。
    """
    from ase.data import covalent_radii
    from ase.neighborlist import natural_cutoffs, neighbor_list

    nums = atoms.get_atomic_numbers()
    r_cov = np.asarray(covalent_radii)[nums]
    cutoffs = natural_cutoffs(atoms, mult=lam)
    i, j, d, S = neighbor_list("ijdS", atoms, cutoffs, self_interaction=False)
    # 防御性过滤（与 per-atom 球语义一致；natural_cutoffs 语义已保证，重复无害）
    keep = d < lam * (r_cov[i] + r_cov[j])
    i, j, d, S = (np.asarray(x)[keep] for x in (i, j, d, S))
    # 双向邻接补齐（v1.1 契约）：缺 (j,i,-S) 才补，避免双计
    tri = np.stack([i, j, S[:, 0], S[:, 1], S[:, 2]], axis=1).astype(np.int64)
    dmap = {tuple(t): float(dd) for t, dd in zip(tri.tolist(), np.asarray(d, float))}
    rev = np.stack([tri[:, 1], tri[:, 0], -tri[:, 2], -tri[:, 3], -tri[:, 4]], axis=1)
    missing = [t for t in rev.tolist() if tuple(t) not in dmap]
    if missing:
        mi = np.array([t[0] for t in missing], dtype=np.int32)
        mj = np.array([t[1] for t in missing], dtype=np.int32)
        mS = np.array([t[2:] for t in missing], dtype=np.int32)
        # 反向边与正向边同距：由反向三元组 (j,i,-S) 找回正向 (i,j,S) 的 d
        md = np.array([dmap[(t[1], t[0], -t[2], -t[3], -t[4])] for t in missing],
                      dtype=np.float64)
        i = np.concatenate([i, mi])
        j = np.concatenate([j, mj])
        d = np.concatenate([d, md])
        S = np.concatenate([S, mS], axis=0)
    pair_r0 = {}
    if i.size:
        pairs = np.sort(np.stack([nums[i], nums[j]], axis=1), axis=1)
        for zi, zj in np.unique(pairs, axis=0):
            pair_r0[(int(zi), int(zj))] = float(covalent_radii[zi] + covalent_radii[zj])
    return SimpleNamespace(i=i.astype(np.int32), j=j.astype(np.int32),
                           S=S.astype(np.int32), d=d.astype(np.float64),
                           n_atoms=len(atoms), lam=float(lam),
                           cutoff_mode="covalent", pair_r0=pair_r0)


# ---------------------------------------------------------------------------
# 实数球谐 / Steinhardt q（numpy 手写）
# ---------------------------------------------------------------------------
def _assoc_legendre(lmax: int, x: np.ndarray) -> dict:
    """P_l^m(x)，l=0..lmax，m=0..l，含 Condon-Shortley 相位。x: (K,)"""
    x = np.asarray(x, dtype=float)
    P = {0: [np.ones_like(x)]}
    for l in range(1, lmax + 1):
        prev = P[l - 1]
        cur = [None] * (l + 1)
        cur[l] = -(2 * l - 1) * np.sqrt(np.clip(1.0 - x * x, 0.0, None)) * prev[l - 1]
        cur[l - 1] = x * (2 * l - 1) * prev[l - 1]
        for m in range(l - 1):
            cur[m] = ((2 * l - 1) * x * prev[m] - (l + m - 1) * P[l - 2][m]) / (l - m)
        P[l] = cur
    return P


def _real_sh_block(l: int, u: np.ndarray) -> np.ndarray:
    """degree=l 的实数球谐 (K, 2l+1)，列序 m=-l..l。u: (K,3) 单位向量。"""
    x = u[:, 2]
    phi = np.arctan2(u[:, 1], u[:, 0])
    P = _assoc_legendre(l, x)
    out = np.zeros((u.shape[0], 2 * l + 1))
    for m in range(1, l + 1):
        n = math.sqrt((2 * l + 1) / (4 * math.pi)
                      * math.factorial(l - m) / math.factorial(l + m))
        plm = P[l][m]
        out[:, l + m] = math.sqrt(2.0) * n * plm * np.cos(m * phi)
        out[:, l - m] = math.sqrt(2.0) * n * plm * np.sin(m * phi)
    out[:, l] = math.sqrt((2 * l + 1) / (4 * math.pi)) * P[l][0]
    return out


def _q_l(l: int, u: np.ndarray) -> float:
    """q_l = sqrt(4π/(2l+1) Σ_m ⟨Y_lm⟩²)，邻居等权。u: (K,3) 单位向量。"""
    if u.shape[0] == 0:
        return 0.0
    a = _real_sh_block(l, u).mean(axis=0)
    return float(np.sqrt(4 * math.pi / (2 * l + 1) * float(np.dot(a, a))))


# ---------------------------------------------------------------------------
# 契约接口（contracts.md §3.7）
# ---------------------------------------------------------------------------
def geometry_features(atoms, graph=None) -> np.ndarray:
    """每原子 14 维特征（顺序见模块 docstring / contracts.md §3.7）。

    graph：BondGraph（§3.2 字段 i/j/S/d/n_atoms；v1.1 双向邻接）或 None
    （内部 build_graph() 回退，缺省 λ*=1.20，文档化选择）。返回 (N, 14)
    float64。每站点 CN = 以该原子为 tail 的边数（双向图上即无向近邻数，
    等价 np.bincount(g.i)，不做对称化以避免双计）。
    """
    if graph is None:
        graph = build_graph(atoms)
    from ase.data import covalent_radii

    n_atoms = len(atoms)
    pos = atoms.get_positions()
    cell = np.asarray(atoms.get_cell())
    nums = atoms.get_atomic_numbers()
    i = np.asarray(graph.i)
    j = np.asarray(graph.j)
    S = np.asarray(graph.S, dtype=float)

    F = np.zeros((n_atoms, N_FEATURES), dtype=np.float64)
    for k in range(n_atoms):
        m = i == k
        if not m.any():
            continue  # CN=0：整行 0（f1/f2/f3/q 均 0，文档化）
        jj = j[m]
        v = pos[jj] + S[m] @ cell - pos[k]
        d = np.linalg.norm(v, axis=1)
        cn = len(jj)
        F[k, F_CN] = cn
        r0 = float(np.mean(covalent_radii[nums[k]] + covalent_radii[nums[jj]]))
        dmean = float(d.mean())
        F[k, F_DMEAN_R0] = dmean / r0
        F[k, F_DSTD_MEAN] = float(d.std()) / dmean
        F[k, F_DMIN_MEAN] = float(d.min()) / dmean
        u = v / d[:, None]
        if cn >= 2:
            us = u[:N_HIST_NEIGH]                 # CN>12 取前 12 边（文档化近似）
            dots = us @ us.T
            tri = np.triu_indices(us.shape[0], k=1)
            cosv = np.clip(dots[tri], -1.0, 1.0)
            hist, _ = np.histogram(cosv, bins=HIST_BINS, range=HIST_RANGE)
            F[k, F_HIST_START:F_HIST_END] = hist / hist.sum()
        F[k, F_Q4] = _q_l(4, u)
        F[k, F_Q6] = _q_l(6, u)
    return F


def _conf(margin: float, lam: float) -> float:
    """到最近决策边界的距离 margin 的饱和映射 → [0,1]（文档化）。"""
    if margin <= 0.0:
        return 0.0
    return margin / (margin + lam)


def classify_geometry(feats: np.ndarray) -> list:
    """规则分类。feats: (N,14) 或 (14,) → list[(label, confidence)]，label ∈
    GEOMETRY_LABELS。CN 路由 + (q4,q6) 距理想参考距离 + 边界距离→confidence；
    conf<0.7 → "ambiguous"（契约 §3.7）。"""
    feats = np.asarray(feats, dtype=float)
    single = feats.ndim == 1
    if single:
        feats = feats[None, :]
    out: list = []
    for row in feats:
        cn = int(round(row[F_CN]))
        q4, q6 = float(row[F_Q4]), float(row[F_Q6])
        if cn in (0, 1):
            out.append(("other", 1.0))          # 契约：CN 0/1 → other（确定性）
            continue
        labels = _ROUTES.get(cn)
        if labels is None:
            out.append(("other", 1.0))          # CN 7/9/10/11/13/15/≥16：无类目 → other
            continue
        if cn == 2:
            # cosθ 峰 gate：单键角必须落在 bin0（cosθ ∈ [-1,-0.75)）才有 linear 资格
            bin0 = row[F_HIST_START] > 0.0
            d = math.hypot(q4 - REF_GEOMETRY["linear"][0],
                           q6 - REF_GEOMETRY["linear"][1])
            if bin0 and d <= T_MATCH:
                out.append(("linear", _conf(T_MATCH - d, LAMBDA)))
            else:
                out.append(("other", _conf(d - T_MATCH, LAMBDA_OTHER)))
            continue
        dists = [math.hypot(q4 - REF_GEOMETRY[lab][0], q6 - REF_GEOMETRY[lab][1])
                 for lab in labels]
        order = np.argsort(dists)
        d1, d2 = dists[order[0]], (dists[order[1]] if len(labels) > 1 else T_MATCH)
        if d1 <= T_MATCH:
            margin = d2 - d1 if len(labels) > 1 else T_MATCH - d1
            conf = _conf(margin, LAMBDA)
            if conf < CONF_AMBIG:
                out.append(("ambiguous", conf))
            else:
                out.append((labels[order[0]], conf))
        else:
            conf = _conf(d1 - T_MATCH, LAMBDA_OTHER)
            if conf < CONF_AMBIG:
                out.append(("ambiguous", conf))
            else:
                out.append(("other", conf))
    return out
