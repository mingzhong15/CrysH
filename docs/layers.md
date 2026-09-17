# 层次定义与编号对照

CrysH 对外用 **L0–L5** 指代六个层次；代码里**不用编号做模块名**，而是用语义名。
本表是唯一对照处（历史原因：研究工程早期按"每个 L 一个子任务目录"组织，后来合并过两次）。

| 层 | 语义 | 模块 | 输出位置 |
|---|---|---|---|
| L0 | Validity：晶胞/堆积是否合法 | `crysh.validity` | records 的 `q_min`, `volume_norm`, `cell_kappa`, `aspect_ratio`, `n_core_overlap_pairs`, 6 个 flag |
| L1 | Periodic dimensionality：键图 d(λ) 谱 | `crysh.bond` + `crysh.dimensionality` | `d_090…d_200`, `d_star`, `dim_persistence`, `n_components`, `dim_components` |
| L2 | Morphology / vacuum / surface | `crysh.morphology` | `vacuum_gap`, `vacuum_fraction`, `surface_score`, `morphology_class`, `n_layers`, `span`, `f_max`, 三个 score |
| L3 | Coordination | `crysh.coord` | `mean_cn`, `low/high_cn_fraction`, `cn_min/max`, `cn_cell_warning/margin` |
| L4 | Coordination geometry | `crysh.geometry` | `geom_top_label`, `geom_ambiguous_fraction`, `geom_label_counts` |
| L5 | Tokens / sharing（局域建筑块与连接方式） | `crysh.tokens` | `n_distinct_l3`, `h_neigh_mean`, `f_corner/f_edge/f_face`, `sharing_top_label` + 侧文件 token 长表 |

## 与研究方案（mapper v2）编号的差异

规划文档 `plan-mapper-v2.md` 用的是**新的层次切分**，其中 L3/L4 与上面的 L3/L4/L5
不是一回事：

| 新方案 | 语义 | 本库现状 |
|---|---|---|
| L0 Validity | 同 | `validity`（已完成） |
| L1 Dimensionality | 同 | `bond` + `dimensionality`（已完成） |
| L2 Morphology | 同 | `morphology`（已完成） |
| L3 Local building blocks | motif 实例 `m = (Z, CN, G, C, D)` 四正交轴 | **拆在 `coord` + `geometry` + `tokens`**；一等公民的 `m_i` 表在 v0.2 落地 |
| L4 Motif-network architecture | motif 超节点图（corner/edge/face × 同质/异质边 × per-family 维数） | 现只有结构级 sharing 分数（`tokens.sharing`）；超节点图在 v0.2 |
| L5 Electronic environment | 电子环境（Hamiltonian 侧） | 不在本库；见研究工程的 `level6-electronic` |

也就是说：**数据契约（列名、token 串）是冻结且可兼容的**，新方案是在其上追加
`m_i` 表与 motif 图，不是推倒重来。

## 冻结契约

- 列名与列序：`crysh.config.RECORD_COLUMNS`（59 列）
- λ 网格：`crysh.config.LAMBDAS`（8 点 `0.90…2.00`）、λ* = `D_STAR_LAMBDA`（1.20）
- token 格式：`crysh.tokens`（`Z|CN|geom|chem` / `…|sharing|d_star`）
- 合并表的物理路径**不属契约**：由调用方经 `MapperConfig` 传入
