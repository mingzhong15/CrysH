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

## [Unreleased]

### Planned for v0.2.0

- `localenv`：motif 实例 `m = (Z, CN, geometry, neighbour chemistry, distortion)`
  作为一等公民的 site 级表（含 `cn_species`、`cn_eff`、`cn_by_lambda`、`p_cn`、
  `shell_gap/cn1/cn2`、连续畸变 `d_i`）。
- `motifnet`：motif 超节点图（corner/edge/face × 同质/异质边、per-family 维数、
  edge 词表、`l4@2` token）。
- 单一共享邻居表：让 L1 谱、L3 壳层与 L4 motif 图消费同一份 NL（现在是各建各的）。
