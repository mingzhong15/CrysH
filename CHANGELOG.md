# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-17

首个版本：把研究工程里的 `ckt` 库抽取、重构为可独立发布的 `crysh`。
**数据契约不变**（record 列名与列序、λ 网格、token 格式全部冻结），
变的是"库长什么样、怎么用"。

### Added

- `crysh.config`：`MapperConfig`（λ 网格、λ*、校准表/阈值路径、输出根全部显式传参）、
  `RECORD_COLUMNS`（59 列）、`LAMBDAS`（8 点网格，**全库唯一真源**）、`LAMBDAS_STAR`。
- 公共 API：`map_record`（单结构 → 一行 record）、`map_structure`（另返回聚合侧数据）、
  `run_batch`（manifest → parquet + 侧文件，多进程）；`crysh.<layer>` 惰性子模块入口。
- `crysh.dev`：合成结构生成（`synthetic_atoms` / `write_synthetic_subset`），
  供无真实语料的环境冒烟整条链——只造结构，指标仍走真计算。
- `crysh.controls` 随包分发 `controls_data/`（44 个带标签的 ground-truth 结构）。
- `docs/layers.md`：L0–L5 定义、模块对照，以及与研究方案新编号（L3/L4 重切）的差异说明。
- `crysh.research` 子包：研究侧工具（`figdata` / `fingerprint` / `io` / `report` /
  `oracle` / `calibration` / `paths`），可选依赖 `crysh[research]`。
- CI（GitHub Actions）：3.11/3.12 双档 + ruff + pytest；核心测试标记
  `-m "not research"` 可脱离 pymatgen 运行。

### Changed

- 模块改名（保留 git 历史）：`dimension` → `dimensionality`、`tokenize` → `tokens`、
  `pipeline` → `records`；`schema` 并入 `config`（兼容层留在 `crysh.research.schema`）。
- **λ 网格与 `D_STAR_LAMBDA` 收敛到 `config`**：此前散落在 4 个模块里各写一遍。
- `pipeline.records` 从 1005 行降到 300 行：删除 mock 层（约 400 行）、逐级 parquet 回读、
  7 个 `_HAVE_*` 探测与 8 个 `try/except ImportError`、全部"异常 → 回落 mock"分支。
  现在一条路径：真实现，失败即抛。
- 核心依赖收敛为 **numpy + ase**；pandas/pyarrow 仅在写表时需要，pymatgen/matplotlib
  只在 `crysh[research]` 里。
- 库内**零环境变量**：`crysh.paths`（工作区布局推断）移入 `crysh.research`，
  核心不再认识任何工作区目录。
- `controls` 的 ground-truth 资产随包分发，不再按工作区布局定位。

### Fixed

- `tokens._resolve_d_star` 的导入在模块改名后一直失败，导致**每个结构的 L5 token
  都静默写成 `3D`**；同时发现原兜底值物理上也错（`pbc=False` 的孤立团簇被写成 3D），
  测试期望随之更正为 `0D`。
- `paths.py` 的 `LEVELS_DIR` 在 `KT_ROOT` 之前求值导致 `NameError`（导入顺序）。
- `validity` 的冻芯查表路径指向已搬迁的目录。
- 库测试改为可独立运行：把回归脚本里的旧实现快照 vendor 成
  `tests/_legacy_tokens.py`，测试不再依赖研究工程的脚本。

### Test suite

- 库内 254 个用例（248 迁移 + 公共 API/README 契约守卫），`248 passed / 1 skipped`
  在重构前后逐项一致；另有研究侧用例标记 `research`。

## [Unreleased] — 计划 v0.2.0

### Added

- `crysh.kernels`：**一份邻居表处处复用**。`neighbor_list(atoms, cfg)` 按
  `λ_max · r0(pair)` 建一次表，`masked_graph(nl, λ)` 纯数组筛出任意 λ 的子图，
  `normalized_neighbor_table(nl, λ)` 给出按距离排序的逐位点邻居（含"是否在当前键集内"
  的掩码——壳层边界正在最后一条键的外侧，没有掩码就看不见）。
  等价性是**准入验收**：与 `bond.build_bond_graph` 在 8 个 λ × 49 个结构上边集合与
  距离逐点相等（`tests/test_kernels.py`，100 例）。
- `crysh.localenv`：L3 的 site 级 m 实例 —— `cn` / `cn_eff` / `cn_species` /
  `cn_by_lambda` / `p_cn` / `shell_gap` / `cn_shell` / `cn1` / `cn2` / `shell_conf` /
  邻居化学（`neighbor_counts`、`h_neigh`）/ 径向统计（`r_tilde_mean/std`、`bond_cv`）/
  interim 畸变 `d_i`（版本化 `D_I_VERSION`）。`site_table()` 出 pandas 表。
- `crysh.map_sites(atoms, cfg) -> (sites, summary)`：site 级结果 + 结构级汇总
  （`SITE_SUMMARY_COLUMNS`）。**additive**：不并入冻结的 59 列 record（有测试钉住）。
- `MapperConfig` 增 `kernel_lam_max`（建表上界，默认 2.0）与 `cn_eff_alpha`
  （有效配位数衰减系数，单位 1/Å，默认 2.0）。

### Changed

- **controls 的 CN 参考方法改为显式选定**：`build_all(cn_method=...)`，可用
  `$CRYSH_CN_METHOD` 覆盖，默认 `crystalnn`（= 冻结 `labels.yaml` 的口径）。此前是隐式
  探测：有 pymatgen 用 CrystalNN、没有就悄悄退回 1.25x 壳层法，而 44 个控制结构里有 8 个
  的 CN 因此随环境而变。现在缺 pymatgen 时请求 `crystalnn` 直接 `ImportError`（指向
  `crysh[research]`），无依赖的 `shell` 必须显式请求，实际用的方法如实写进每条 label 的
  `cn_method`。（这是 0.1.0"零环境变量"的唯一例外：该变量只选参考方法，不涉任何路径。）
- 测试分三层（目录即边界），CI 拆成 core / tables / research 三个 job，各自**只装**自己
  那一档 extra；core job 断言 pandas/pyarrow/pymatgen/matplotlib 确实不在，并在屏蔽这些
  依赖的子进程里重新收集 core 档（`tests/test_skeleton.py`），杜绝"本机绿、干净环境红"。
- `research` extra 改为 `crysh[tables,controls]` + pymatgen/matplotlib；`dev` 带 PyYAML。

### Fixed

- `tests/tables/test_controls.py::test_labels_match_recomputed_references` 在无 pymatgen 的
  干净环境里失败（重算得 `shell_1.25_fallback` != 冻结的 `CrystalNN`）。一致性测试按口径
  拆开：与 CN 无关的字段留在 tables 档（显式 `shell` 重算），完整复现 CrystalNN 口径
  `labels.yaml` 的测试移到 `tests/research/test_controls_labels.py`，并断言方法分布
  （42 × `CrystalNN` + 2 × `shell_1.25_fallback`）。

- `crysh.motifnet` —— **L4 motif 超节点图**。节点 = 多面体（CN≥4，或 CN=3 平面三角形），
  边 = 共享配体实例；输出整体维数、**per-family 维数**（"1D TiO₆ 链长在 3D 框架里"就是
  靠它）、连通分量、边类型分布与 top 家族对。维数与 L1 已冻结的 `d_star` 在
  金刚石/岩盐/fcc/bcc/ZnS 上**逐一相等**（独立实现，测试钉住）。
  设计取舍见 `crysh/motifnet.py` 的 docstring，要点两条：

  1. **配体实例台账**：边按"配体**实例**（原子 + 周期像）"计数，而不是按原子数——
     岩盐原胞只有 2 个原子，按原子数会把 6 个 Cl 邻居压成 1 个。
  2. **术语适用域**：`corner/edge/face` 来自硅酸盐/硼酸盐（配体只桥接 2 个多面体）。
     密堆金属与岩盐不满足前提——岩盐相邻八面体共享的 4 个 Cl 在空间上张成**四边形**
     （既不是边也不是面）；fcc 的每个原子被 12 个"多面体"共用。这类情形给 `"n/a"`，
     并同时给出共享配体数与界面跨度（`interface_span`），**不硬塞标签**。

### Known issues

- **多面体共享判据（`tokens` 的 `sharing` / `f_corner|edge|face` / `sharing_label`）
  在周期结构里不可靠**。2026-09-17 复核结论与两次失败的修复尝试：

  - 现象：金刚石（教科书 corner-sharing）、岩盐（edge 为主）、fcc（face 为主）在
    当前实现下都可能被判成 `isolated`；
  - 根因：共享计数建在"每原子一个邻居集合"（按索引去重）之上，而**周期像信息在这层
    已经丢失**——小胞里同一个配体原子的多个像分不开，大胞里同一个配体又可能因索引
    相同被折叠。两个需求（化学串要"数键"、共享要"数原子+分像"）共用一个视图。
  - 试过并否掉的做法：按 `(索引, 平移)` 计数（金刚石算出共享 4 → 误判 face）、
    按最小镜像归并配体（mock 图无 cell → `LinAlgError`）、按中心对求邻居交集
    （8 原子胞仍给 face）。四次尝试都没能在不重写语义的前提下自洽。
  - 结论：**不修**（会改冻结 token 生态），改为登记：`tests/test_site_api.py` 用
    `xfail(strict=True)` 钉住"教科书期望"，修对之日会 XPASS 报错，强制同步更新
    token 生态、`f_corner/edge/face`、UI 树与 coverage-atlas。
  - 修法方向（留给下一版）：把共享判据整体搬到"配体实例 = (原子, 平移)"的显式表示上，
    与 :mod:`crysh.kernels` 的共享邻居表对齐，而不是复用 `tokens` 的邻居集合。
- `motifnet.n_components` 是**有限超胞上的局部视角**：跨超胞边界的键被截断，故对
  小胞体系可能偏大（金刚石 3×3×3 得 2，真实周期网络是 1）。精确值用
  `dimensionality.dimensionality_spectrum`（原胞整数秩，无截断）。已在 docstring 与
  测试里如实标注，不假装两者等价。
- 共享分类只在"真桥"（配体恰接 2 个中心）上给出 `corner/edge/face`；框架材料里
  仍有部分边落在 `n/a`（配体在超胞里接了 >2 个中心）。要把框架材料的分类型做全，
  正确做法是**限制在局部配位多面体**上判定，而不是靠超胞展开——留给下一版。
- `l4@2` token（把 motif 图信息写回 token 串）**未做**——见 `plan-crysh.md` §4 的 M2。

### 两套邻居计数口径（刻意并存，勿互相"修正"）

| 口径 | 含义 | 用在哪 |
|---|---|---|
| **数键** | 保留周期像，一条键算一个邻居 | `tokens` 的 l3 化学串与 `h_neigh`；`localenv.cn_species` / `cn_eff` / 壳层统计 |
| **数原子** | 按原子索引去重 | 多面体共享判据（当前实现所依赖，也是它在周期结构里失准的原因） |

同一结构上两者可以不同（金刚石小胞：4 条键、1 个邻居原子）。改动任一侧前先读
`tests/test_localenv.py::test_two_neighbour_counting_conventions_are_both_pinned`。
