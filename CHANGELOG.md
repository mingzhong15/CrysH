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

### Changed

- **controls 的 CN 参考方法改为显式选定**：`build_all(cn_method=...)`，可用
  `$CRYSH_CN_METHOD` 覆盖，默认 `crystalnn`（= 冻结 `labels.yaml` 的口径）。此前是隐式
  探测：有 pymatgen 用 CrystalNN、没有就悄悄退回 1.25x 壳层法，而 44 个控制结构里有 8 个
  的 CN 因此随环境而变。现在缺 pymatgen 时请求 `crystalnn` 直接 `ImportError`（指向
  `crysh[research]`），无依赖的 `shell` 必须显式请求，实际用的方法如实写进每条 label 的
  `cn_method`；`write_assets`/CLI 也接受 `--cn-method`。（这是 0.1.0"零环境变量"的唯一
  例外：该变量只选参考方法，不牵涉任何路径/工作区。）
- 测试分三层（目录即边界），CI 拆成 core / tables / research 三个 job，各自**只装**自己
  那一档 extra：core 只装 `.[dev]`（并断言 pandas/pyarrow/pymatgen/matplotlib 确实不在），
  tables 装 `.[dev,tables]`，research 装 `.[dev,research]`。
- `research` extra 改为 `crysh[tables,controls]` + pymatgen/matplotlib（重算资产需要 ruamel）；
  `dev` 带 PyYAML，使 core/tables 档也能读 `controls_data/labels.yaml`。

### Fixed

- `tests/tables/test_controls.py::test_labels_match_recomputed_references` 在无 pymatgen 的
  干净环境里失败（重算得 `shell_1.25_fallback` != 冻结的 `CrystalNN`）。一致性测试按口径
  拆开：与 CN 无关的字段留在 tables 档（显式 `shell` 重算），完整复现 CrystalNN 口径
  `labels.yaml` 的测试移到 `tests/research/test_controls_labels.py`，并断言方法分布
  （42 × `CrystalNN` + 2 × `shell_1.25_fallback`）。
- 核心档新增守卫：`tests/test_skeleton.py` 在屏蔽 pandas/pyarrow/pymatgen/matplotlib 的
  子进程里重新收集 core 档，根目录测试再也无法悄悄拉重依赖。

### Planned for v0.2.0

- `localenv`：motif 实例 `m = (Z, CN, geometry, neighbour chemistry, distortion)`
  作为一等公民的 site 级表（含 `cn_species`、`cn_eff`、`cn_by_lambda`、`p_cn`、
  `shell_gap/cn1/cn2`、连续畸变 `d_i`）。
- `motifnet`：motif 超节点图（corner/edge/face × 同质/异质边、per-family 维数、
  edge 词表、`l4@2` token）。
- 单一共享邻居表：让 L1 谱、L3 壳层与 L4 motif 图消费同一份 NL（现在是各建各的）。
