"""L4 — motif 超节点图（多面体网络）。

这是 mapper 里"架构"这一层的本体：**节点 = 多面体（motif 实例），边 = 共享配体**。
有了它才能回答"1D 的 TiO₆ 链长在 3D 框架里"这类问题——单看每个位点的 CN/几何是
答不出来的。

术语的适用域（**本模块最重要的设计决定**）
--------------------------------------------
`corner / edge / face` 这套分类来自硅酸盐／硼酸盐化学，它的前提是：**配体只桥接
两个多面体**（Si–O–Si ≈ 145°，每个 O 连接 2 个 SiO₄ 四面体）。在这个前提下，
"共享几个配体"与"共享几个顶点"是一回事：1 → corner，2 → edge，≥3 → face。

但密堆金属与岩盐型离子晶体不满足这个前提，而且**不是近似不满足，是根本不同**：

- 岩盐：相邻两个 Na 八面体共享 4 个 Cl，而这 4 个 Cl 在垂直于 Na–Na 的平面上张成
  一个边长 2.82 Å 的**四边形**——既不是一条边，也不是一个三角面；
- fcc 金属：每个原子被 12 个"多面体"共用，"共享顶点数"没有区分度。

所以本模块**不假装**这些体系能做 corner/edge/face 分类，而是：

1. 显式判定配体的角色（`ligand_kind`）：`bridge`（只接 2 个中心，分类有效）
   / `shared`（接 ≥3 个中心，分类无意义，如实标出）+ 共享配体数 + 共享界面的
   紧致度（最大两两距离 / 键长）。分类**不适用**时给 `"n/a"`，而不是硬塞一个标签。
2. 无论分类是否适用，**连接图本身总是有效的**：谁和谁相连、连成几维、分成几个
   连通分量、每个元素家族各自几维——这些是矩阵/图的性质，不依赖术语。

超胞口径（**必须显式**）
------------------------
"共享几个配体"在单个小胞里是病态的：岩盐的原胞只有 2 个原子，配体只连得上 1 个
中心。所以本模块要求调用方给一个显式的超胞矩阵 `cell_matrix`，默认 `diag(2,2,2)`
——把局部拓扑展开到足够大，使"某个配体接几个中心"这个局部量稳定下来。
选多大的胞会改变**计数**（不改变连通性），所以超胞规模记在返回值的 `meta` 里，
可追溯、可复现。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from ase import Atoms
from ase.build import make_supercell

from crysh.bond import BondGraph
from crysh.config import MapperConfig
from crysh.kernels import NeighborList, masked_graph, neighbor_list

__all__ = [
    "MotifNode", "MotifEdge", "MotifNet",
    "motif_network", "DEFAULT_CELL_MATRIX",
    "LIGAND_BRIDGE", "LIGAND_SHARED", "SHARING_NA",
]

#: 自适应展开：原子数少于这个阈值时，默认按 2×2×2 展开（否则原胞）
_AUTO_EXPAND_BELOW = 8

#: 自适应展开用的矩阵。用 3×3×3 而不是 2×2×2：**偶数倍胞会把子晶格人为切断**
#: （金刚石 2×2×2 得到两个互不相连的子晶格、分量数 2；3×3×3 才是 1）——这是有限
#: 尺寸假象，不是拓扑。
DEFAULT_CELL_MATRIX = np.diag([3, 3, 3])

#: 配体角色
LIGAND_BRIDGE = "bridge"   # 只接 2 个中心 → corner/edge/face 分类有效
LIGAND_SHARED = "shared"   # 接 ≥3 个中心 → 分类不适用（密堆/岩盐）
LIGAND_TERMINAL = "terminal"  # 只接 1 个中心 → 不构成共享

#: 分类不适用时的占位（**不**硬塞一个标签）
SHARING_NA = "n/a"

#: 共享 1/2/≥3 个配体 → corner/edge/face（仅对 `bridge` 配体有效）
_SHARING_BY_COUNT = {1: "corner", 2: "edge"}


@dataclass
class MotifNode:
    """一个多面体（motif 实例）。`site` 是超胞内的原子下标。"""

    node_id: int
    site: int
    symbol: str
    cn: int
    geometry: str
    family: str          # 元素 + 几何（"Ti|oct" 这类家族键）

    def to_row(self) -> dict:
        return {"node_id": self.node_id, "site": self.site, "symbol": self.symbol,
                "cn": self.cn, "geometry": self.geometry, "family": self.family}


@dataclass
class MotifEdge:
    """两个多面体之间的共享关系（一条边）。"""

    a: int               # MotifNode.node_id
    b: int
    n_shared: int        # 共享配体**原子**数
    sharing: str         # "corner" | "edge" | "face" | "n/a"
    ligand_kind: str     # LIGAND_BRIDGE / LIGAND_SHARED
    interface_span: float  # 共享配体两两最大距离 / 平均键长（紧致度；单配体为 0）
    heterogeneous: bool  # 两端 family 是否不同

    def to_row(self) -> dict:
        return {"a": self.a, "b": self.b, "n_shared": self.n_shared,
                "sharing": self.sharing, "ligand_kind": self.ligand_kind,
                "interface_span": self.interface_span,
                "heterogeneous": self.heterogeneous}


@dataclass
class MotifNet:
    """一个结构的 motif 超节点图 + 拓扑摘要。"""

    nodes: list[MotifNode]
    edges: list[MotifEdge]
    #: 整体维度（节点图的整数秩）；0 = 孤立多面体。
    #: 与 :func:`crysh.dimensionality.dimensionality_spectrum` 的 `d_star` 在
    #: 金刚石/岩盐/fcc/bcc/ZnS 上逐一相等（有测试钉住）。
    dim: int
    #: 连通分量数。**注意**：在有限超胞上算得，跨超胞边界的键会被截断，故对"小胞+
    #: 偶数倍超胞"的体系可能偏大（金刚石 3×3×3 得 2，真实周期网络是 1）。要精确值
    #: 请用 :func:`crysh.dimensionality.dimensionality_spectrum`（它在原胞上做整数秩，
    #: 没有边界截断）——本字段是 motifnet 的局部视角，不宣称与它等价。
    n_components: int
    #: 每个元素家族各自诱导子图的维度（"1D TiO₆ 链在 3D 框架里"就是靠它）
    family_dims: dict[str, int]
    #: 每家族的节点数
    family_counts: dict[str, int]
    #: 按 sharing 类型统计的边占比（"n/a" 单列，不与 corner/edge/face 混）
    sharing_fractions: dict[str, float]
    #: 可追溯参数（超胞、λ、判定阈值）
    meta: dict = field(default_factory=dict)

    def edges_top(self, k: int = 5) -> list[tuple[str, int]]:
        """按家族对统计的边类型 top-k：`("A|B|sharing", count)`。"""
        c: Counter = Counter()
        by_id = {n.node_id: n for n in self.nodes}
        for e in self.edges:
            fa, fb = by_id[e.a].family, by_id[e.b].family
            key = f"{fa}--{e.sharing}--{fb}" if fa <= fb else f"{fb}--{e.sharing}--{fa}"
            c[key] += 1
        return c.most_common(k)


def _integer_rank(vectors: list[np.ndarray]) -> int:
    """整数向量组的秩（**精确整数行阶梯化**）。

    为什么不用浮点 SVD：这里的向量是晶格平移的整数坐标（可能到 1e6 量级，见调用处
    的定点化），浮点秩会给出无意义的数字（实测：金刚石的三维网络被算成 7）。
    算法：逐列高斯消元，用整数 gcd 约简主元，统计非零行数。
    """
    if not vectors:
        return 0
    rows = [np.asarray(v, dtype=np.int64).copy() for v in vectors]
    n_rows, n_cols = len(rows), int(rows[0].size)
    rank = 0
    for col in range(n_cols):
        piv = None
        for r in range(rank, n_rows):
            if rows[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for r in range(n_rows):
            if r == rank or rows[r][col] == 0:
                continue
            a, b = int(rows[rank][col]), int(rows[r][col])
            # 消去：rows[r] = rows[r]*a/g - rows[rank]*b/g（保持整数）
            g = int(np.gcd(a, b)) or 1
            rows[r] = (rows[r] * (a // g) - rows[rank] * (b // g))
        rank += 1
        if rank == n_rows:
            break
    return rank


def _lig_pos(lig: tuple[int, int, int, int], pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """配体**实例**的笛卡尔位置。

    实例 key = (原子索引, Sx, Sy, Sz)，S 是该配体相对其中心的位置平移（单位：胞）。
    故位置 = ``pos[原子] + S·cell``。两个实例的空间距离必须用它算 —— 直接比较平移标签
    在不同原子之间恒为 0（实测踩过）。
    """
    a, sx, sy, sz = lig
    return pos[a] + np.asarray([sx, sy, sz], dtype=float) @ cell


def _components(n_nodes: int, adj: list[list[int]]) -> tuple[int, list[int]]:
    """无向图连通分量 → (分量数, 每节点所属分量 id)。"""
    comp = [-1] * n_nodes
    n_comp = 0
    for s in range(n_nodes):
        if comp[s] != -1:
            continue
        stack = [s]
        comp[s] = n_comp
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = n_comp
                    stack.append(v)
        n_comp += 1
    return n_comp, comp


def motif_network(atoms: Atoms, cfg: MapperConfig | None = None, *,
                  cell_matrix: np.ndarray | None = None,
                  bridging_elements: list[str] | None = None,
                  graph: BondGraph | None = None,
                  coord=None,
                  geometry_labels: list[str] | None = None,
                  nl: NeighborList | None = None) -> MotifNet:
    """构造 motif 超节点图。

    Parameters
    ----------
    atoms:
        结构（会在内部按 `cell_matrix` 展开成超胞来做局部判定；**输入不被修改**）。
    cfg:
        :class:`crysh.config.MapperConfig`。
    cell_matrix:
        超胞矩阵（默认 `diag(2,2,2)`）。见模块 docstring 的"超胞口径"。
    bridging_elements:
        **桥连配体**的元素符号表（如硅酸盐的 `["O"]`）。只有这些元素的原子才被当作
        "顶点"参与 corner/edge/face 分类——因为这套术语的前提就是"配体只桥接 2 个
        多面体"。``None``（默认）表示不指定：此时**只算连接图与维数，不做分类**
        （全部标 `"n/a"`），这比硬塞一个标签诚实。给了表则再做一层判定：
        该配体实际接的中心数必须恰为 2，否则仍标 `"n/a"` 并记 `ligand_kind="shared"`。
    graph, coord, geometry_labels:
        可选：已有的键图／配位结果／几何标签（长度须与**输入** atoms 一致；
        内部会按超胞展开）。
    nl:
        可选的共享邻居表（同样按输入 atoms）。

    Returns
    -------
    MotifNet
    """
    cfg = cfg or MapperConfig()
    auto_expanded = False
    if cell_matrix is None:
        # 自适应：小胞（<8 原子）里配体往往只连得上 1 个中心，局部拓扑不成立，
        # 必须展开；大胞不展开，避免把孤立体系复制成多个互不相连的节点。
        if len(atoms) < _AUTO_EXPAND_BELOW:
            cell_matrix = DEFAULT_CELL_MATRIX
            auto_expanded = True
        else:
            cell_matrix = np.eye(3, dtype=int)
    cell_matrix = np.asarray(cell_matrix, dtype=int)
    if cell_matrix.shape != (3, 3):
        raise ValueError(f"cell_matrix 必须是 3x3，得到 {cell_matrix.shape}")

    n_in = len(atoms)
    if geometry_labels is not None and len(geometry_labels) != n_in:
        raise ValueError(
            f"geometry_labels 长度 {len(geometry_labels)} != 原子数 {n_in}")

    # ---- 展开超胞（输入只读；标签按 ASE 的重复顺序拼出来）----
    sup = make_supercell(atoms, cell_matrix, wrap=True)
    n = len(sup)
    reps = n // n_in if n_in else 1
    sup_syms = [str(s) for s in sup.get_chemical_symbols()]

    if nl is None:
        nl = neighbor_list(sup, cfg.cutoff_table, lam_max=cfg.kernel_lam_max)
    if graph is None:
        graph = masked_graph(nl, cfg.d_star_lambda)
    if coord is None:
        from crysh.coord import coordination

        coord = coordination(sup, graph, cfg.cn_table)
    cn = np.asarray(coord.cn, dtype=np.int64).ravel()

    geo = None
    if geometry_labels is not None:
        geo = [str(geometry_labels[i % n_in]) for i in range(n)] if reps > 1 \
            else [str(g) for g in geometry_labels]

    # ---- 节点：多面体中心（CN≥4，或 CN=3 且平面三角形——与 tokens 同一门槛）----
    from crysh.tokens import _CENTER_CN_MIN, _PLANAR_CENTER_GEO

    is_center = [
        int(cn[i]) >= _CENTER_CN_MIN
        or (int(cn[i]) == 3 and geo is not None and geo[i] in _PLANAR_CENTER_GEO)
        for i in range(n)
    ]
    nodes: list[MotifNode] = []
    node_of: dict[int, int] = {}
    for i in range(n):
        if not is_center[i]:
            continue
        g_label = geo[i] if geo is not None else ""
        node_of[i] = len(nodes)
        nodes.append(MotifNode(
            node_id=len(nodes), site=i, symbol=sup_syms[i], cn=int(cn[i]),
            geometry=g_label,
            family=f"{sup_syms[i]}|{g_label}" if g_label else sup_syms[i],
        ))

    # ---- 配体**实例**台账：每个中心 → {(配体原子, 平移) : 到中心的距离} ----
    # "实例"而不是"原子"是关键：岩盐原胞只有 2 个原子，一个 Cl 会以多个周期像同时
    # 配位（同一个原子的 6 个像），按原子数会把 6 个邻居压成 1 个。
    bridge_set = set(bridging_elements) if bridging_elements else None  # 可选：限定哪些元素当配体
    i_arr = np.asarray(graph.i, dtype=np.int64)
    j_arr = np.asarray(graph.j, dtype=np.int64)
    S_arr = np.asarray(graph.S, dtype=np.int64).reshape(-1, 3)
    d_arr = np.asarray(graph.d, dtype=np.float64)
    lig_of: dict[int, dict[tuple[int, int, int, int], float]] = defaultdict(dict)
    for a_, b_, s_, dd_ in zip(i_arr.tolist(), j_arr.tolist(), S_arr.tolist(), d_arr.tolist()):
        if b_ in node_of and (bridge_set is None or sup_syms[a_] in bridge_set):
            lig_of[a_][(int(b_), int(s_[0]), int(s_[1]), int(s_[2]))] = float(dd_)

    # ---- 边：两中心共享的**配体实例**数 ----
    pair_lig: dict[tuple[int, int], list[tuple[int, int, int, int]]] = defaultdict(list)
    ligand_kind_of: dict[tuple[int, int, int, int], str] = {}
    # 每个配体实例服务了多少个中心（用于判定"真桥"）
    centres_of_lig: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    for ci_, ligs in lig_of.items():
        for key in ligs:
            centres_of_lig[key].add(ci_)
    for key, cs in centres_of_lig.items():
        n_c = len(cs)
        ligand_kind_of[key] = (LIGAND_TERMINAL if n_c <= 1
                               else LIGAND_BRIDGE if n_c == 2 else LIGAND_SHARED)
    # 该配体实例接了几个中心（用于判定"真桥"）
    n_centres_of_lig = {key: len(cs) for key, cs in centres_of_lig.items()}

    pos = np.asarray(sup.positions, dtype=np.float64)
    cell = np.asarray(sup.cell, dtype=np.float64)
    inv_cell = np.linalg.inv(cell)

    node_ids = sorted(node_of)
    for x in range(len(node_ids)):
        ci = node_ids[x]
        for y in range(x + 1, len(node_ids)):
            cj = node_ids[y]
            shared = set(lig_of[ci]) & set(lig_of[cj])
            if not shared:
                continue
            pair_lig[(ci, cj)] = sorted(shared)

    edges: list[MotifEdge] = []
    for (ca, cb), ligs in sorted(pair_lig.items()):
        k = len(ligs)
        # 真桥 = 每条共享配体都恰好只接 2 个中心（框架化学的前提）。
        # 任一共享配体接了 ≥3 个中心（岩盐的 Cl 接 6 个、fcc 的 Al 接 12 个）→ 判 n/a：
        # 那种情形下"共享几个配体"不再对应"共享几个顶点"，术语不适用（见模块 docstring）。
        is_true_bridge = all(n_centres_of_lig.get(L, 0) == 2 for L in ligs)
        kind = LIGAND_BRIDGE if is_true_bridge else LIGAND_SHARED
        sharing = _SHARING_BY_COUNT.get(k, "face") if is_true_bridge else SHARING_NA
        # 界面跨度：共享配体在空间上张多开（最大两两距离 / 中心到配体的平均距离）
        dmean = float(np.mean([lig_of[ca][L] for L in ligs])) or 1.0
        if k > 1:
            pts = [_lig_pos(L, pos, cell) for L in ligs]
            dmax = max(float(np.linalg.norm(pts[i] - pts[j]))
                       for i, j in combinations(range(len(pts)), 2))
            span = dmax / dmean
        else:
            span = 0.0
        na, nb = node_of[ca], node_of[cb]
        edges.append(MotifEdge(
            a=na, b=nb, n_shared=k, sharing=sharing, ligand_kind=kind,
            interface_span=round(float(span), 4),
            heterogeneous=nodes[na].family != nodes[nb].family,
        ))

    # ---- 拓扑摘要 ----
    adj: list[list[int]] = [[] for _ in nodes]
    edge_vecs: list[np.ndarray] = []
    for e in edges:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)
        # 桥向量 = 两中心相对位移（整数平移，用于维数）
        sa, sb = nodes[e.a].site, nodes[e.b].site
        v = pos[sb] - pos[sa]
        f = v @ inv_cell
        edge_vecs.append(np.round(f * 1e5).astype(np.int64))
    n_comp, comp_of = _components(len(nodes), adj)

    # 维度：用"桥的平移向量"的整数秩（与 dimension 的秩机器同义）
    # 每个连通分量内，把所有桥向量（相对分量内首节点）取整数秩
    comp_nodes: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(nodes)):
        comp_nodes[comp_of[idx]].append(idx)
    comp_dims: dict[int, int] = {}
    for cid, members in comp_nodes.items():
        rel: list[np.ndarray] = []
        root = members[0]
        # BFS 累计平移
        acc: dict[int, np.ndarray] = {root: np.zeros(3, dtype=np.int64)}
        stack = [root]
        while stack:
            u = stack.pop()
            for e in edges:
                v = None
                if e.a == u:
                    v = e.b
                elif e.b == u:
                    v = e.a
                if v is None or v in acc:
                    continue
                d = pos[nodes[v].site] - pos[nodes[u].site]
                fvec = np.round((d @ inv_cell) * 1e5).astype(np.int64)
                acc[v] = acc[u] + fvec
                stack.append(v)
        rel = [acc[m] for m in members]
        comp_dims[cid] = _integer_rank(rel)
    dim = max(comp_dims.values()) if comp_dims else 0

    fam_nodes: dict[str, list[int]] = defaultdict(list)
    for nd in nodes:
        fam_nodes[nd.family].append(nd.node_id)
    family_dims: dict[str, int] = {}
    for fam, members in fam_nodes.items():
        member_set = set(members)
        sub_adj: list[list[int]] = [[] for _ in nodes]
        for e in edges:
            if e.a in member_set and e.b in member_set:
                sub_adj[e.a].append(e.b)
                sub_adj[e.b].append(e.a)
        n_c_sub, comp_sub = _components(len(nodes), sub_adj)
        cd: dict[int, int] = {}
        for cid in set(comp_sub[m] for m in members):
            group = [m for m in members if comp_sub[m] == cid]
            if not group:
                continue
            root = group[0]
            acc2: dict[int, np.ndarray] = {root: np.zeros(3, dtype=np.int64)}
            stack = [root]
            while stack:
                u = stack.pop()
                for v2 in sub_adj[u]:
                    if v2 in acc2:
                        continue
                    d = pos[nodes[v2].site] - pos[nodes[u].site]
                    acc2[v2] = acc2[u] + np.round((d @ inv_cell) * 1e5).astype(np.int64)
                    stack.append(v2)
            cd[cid] = _integer_rank([acc2[m] for m in group])
        family_dims[fam] = max(cd.values()) if cd else 0

    kinds_count = Counter(e.sharing for e in edges)
    tot = sum(kinds_count.values())
    sharing_fractions = ({k: round(v / tot, 4) for k, v in kinds_count.items()}
                         if tot else {})

    return MotifNet(
        nodes=nodes, edges=edges, dim=int(dim), n_components=int(n_comp),
        family_dims=family_dims,
        family_counts={f: len(m) for f, m in fam_nodes.items()},
        sharing_fractions=sharing_fractions,
        meta={
            "cell_matrix": cell_matrix.tolist(),
            "lam": float(cfg.d_star_lambda),
            "kernel_lam_max": float(cfg.kernel_lam_max),
            "n_atoms_supercell": int(n),
            "n_atoms_input": int(n_in),
            "auto_expanded": bool(auto_expanded),
            "ligand_kinds": dict(Counter(ligand_kind_of.values())),
            "bridging_elements": sorted(bridge_set) if bridge_set else None,
        },
    )
