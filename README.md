# CrysH — Crystal Hierarchy

**A multi-layer crystal knowledge mapper.** CrysH turns a crystal structure into a layered
description and emits it as flat tables you can index, query and aggregate over millions of
structures.

```
L0 validity          is the cell and packing physically sane? (q_min, ν, cell condition, core overlap)
L1 dimensionality    d(λ) bond-graph spectrum → d*, persistence, per-component labels
L2 morphology        vacuum gap, surface-likeness, layer count, porous candidates
L3 coordination      CN, P(CN|Z) percentiles, small-cell aliasing guard
L4 geometry          14-dim local geometry features → routed geometry labels
L5 tokens            local building blocks (`Ti|6|oct|O6`) + corner/edge/face sharing
```

## Status

**v0.1.0** — the library is a clean extraction and refactor of the research code that produced
it: same frozen data contracts (record columns, token format), one code path per layer, no
filesystem assumptions. The motif-level layers described in the roadmap
(`m = (Z, CN, geometry, chemistry, distortion)` instances and the motif super-node graph) land
in v0.2.

## Install

```bash
pip install crysh              # core: numpy + ase only
pip install "crysh[tables]"    # + pandas/pyarrow for writing parquet tables
pip install "crysh[research]"  # + pymatgen/matplotlib research tooling
pip install "crysh[controls]"  # + ruamel.yaml to (re)write controls/data/labels.yaml
```

## Quickstart

```python
from ase.build import bulk
import crysh

rec = crysh.map_record(bulk("Si", "diamond", a=5.43))
rec["d_star"], rec["morphology_class"], rec["mean_cn"]
# (3, 'dense_bulk', 4.0)

# 一行 record 的全部列（59 列，列序即 parquet 列序）
list(rec) == crysh.RECORD_COLUMNS

# 逐位点的局域建筑块（L3 的 m 实例）+ 结构级汇总
sites, summary = crysh.map_sites(bulk("NaCl", "rocksalt", a=5.64))
sites[0].cn_species, sites[0].cn1, sites[0].cn_eff
# ('Cl6', 6, 6.0)
summary["cn_species_top"], summary["shell_conf_mean"]
# ('Cl6', 0.325)

# 多面体网络（L4）：谁和谁相连、连成几维、每个家族各自几维
net = crysh.motif_network(bulk("NaCl", "rocksalt", a=5.64))
net.dim, net.n_components, net.family_dims
# (3, 2, {'Na': 3, 'Cl': 3})
```

Batch a manifest of structures (multiprocess, writes records + side tables):

```python
from crysh import MapperConfig, run_batch
from pathlib import Path

cfg = MapperConfig(
    scale="20k",
    l0_thresholds_path=Path("calibration/l0_thresholds_v1.json"),  # optional
    cn_table_path=Path("calibration/cn_table_20k.json"),           # optional
)
stats = run_batch("manifest.jsonl", "records_20k.parquet", cfg=cfg, n_proc=32)
# {'n_total': 20000, 'n_ok': 19997, 'n_failed': 3}
```

Every path is passed in explicitly: the library never guesses a directory layout. The one
environment variable it looks at is `CRYSH_CN_METHOD`, which only picks the CN reference method
of `crysh.controls` (same knob as `build_all(cn_method=...)`, and the argument wins).

## What one structure maps to

`map_record()` returns one row whose keys are exactly `crysh.RECORD_COLUMNS`:

| Group | Example columns |
|---|---|
| identity | `structure_id`, `formula`, `elements`, `natom` |
| L0 | `q_min`, `volume_norm`, `cell_kappa`, `n_core_overlap_pairs`, 6 flags |
| L1 | `d_090 … d_200` (8-point λ grid), `d_star`, `dim_persistence`, `dim_components` |
| L2 | `vacuum_gap`, `vacuum_fraction`, `surface_score`, `morphology_class`, `n_layers` |
| L3 | `mean_cn`, `low_cn_fraction`, `cn_min/max`, `cn_cell_warning` |
| L4 | `geom_top_label`, `geom_ambiguous_fraction`, `geom_label_counts` |
| L5 | `n_distinct_l3`, `h_neigh_mean`, `f_corner/f_edge/f_face`, `sharing_top_label` |

Tokens are hierarchical strings, e.g. `Ti|6|oct|O6|corner|3D`
(element | CN | geometry | neighbour chemistry | sharing | dimensionality).
See `crysh.tokens` for the frozen format.

## Design rules

1. **One code path per layer.** No mock fallbacks, no silent `except: pass`. If a layer cannot
   be computed, it raises — a record never quietly contains fabricated values.
2. **Small core.** `numpy` + `ase` only. Table I/O and research tooling are extras.
3. **No filesystem assumptions.** A `MapperConfig` carries every path and knob.
4. **Data contracts are frozen.** Column names, the λ grid and token strings are versioned in
   `crysh.config` / `crysh.tokens`; changing them is a breaking change, not a refactor.
5. **One λ grid.** `LAMBDAS` is defined once and imported everywhere — previously it was
   written out in four modules, which is how "the same structure has two different
   dimensionalities" happens.

## Repository layout

```
src/crysh/                 core (~4.4k lines incl. docstrings)
├── config.py              MapperConfig + frozen columns + the single λ grid
├── kernels.py             ONE neighbour list: masked graphs + sorted per-site neighbours
├── bond.py                periodic bond graph (calibrated pair table or covalent × λ)
├── dimensionality.py      d(λ) spectrum, integer-rank topology
├── validity.py            L0
├── morphology.py          L2
├── coord.py               L3 (coordination numbers)
├── geometry.py            L4 (features + q4/q6 routing)
├── localenv.py            L3 building blocks: m = (Z, CN, geometry, chemistry, distortion)
├── motifnet.py            L4: motif super-node graph (connectivity, per-family dimensionality)
├── tokens.py              L5 tokens and sharing
├── metrics.py             vocabulary statistics (richness, effective diversity, accumulation)
├── records.py             map_record / map_structure / map_sites / run_batch
├── controls.py            + controls_data/ — 44 labelled ground-truth structures
├── dev.py                 synthetic structures (smoke tests, not physics)
└── research/              optional research tooling
    ├── figdata.py  fingerprint.py  io.py  report.py
    └── oracle.py   pymatgen benchmarks    calibration.py  Phase-0 cutoff/CN tables
tests/                     runs standalone, no research data needed
├── test_*.py              core tier (numpy + ase only)
├── tables/                + pandas / pyarrow
└── research/              + pymatgen / matplotlib
```

## Known issues

- The polyhedral **sharing criterion is wrong for periodic structures**: it pairs centres
  by atom index, so periodic images of the same ligand collapse. Diamond tetrahedra
  (physically corner-sharing) come out as `isolated`. Pinned by `xfail(strict=True)` tests
  in `tests/test_site_api.py`; fixing it changes frozen sharing tokens, so it ships with the
  next index/coverage rebuild.
- `motifnet`'s `n_components` is a *local* view (a finite supercell truncates bonds that
  cross its boundary); use `dimensionality.dimensionality_spectrum` for the periodic truth.
- The `corner/edge/face` classification only applies where a ligand bridges exactly two
  polyhedra (silicate/borate chemistry). Dense metals and rock-salt-type ionic solids are
  reported as `"n/a"` with the shared-ligand count and interface span instead — in rock salt
  the four shared Cl of adjacent octahedra span a quadrilateral, which is neither edge nor face.
  Bringing silicates fully into scope needs a *local coordination polyhedron* construction
  rather than supercell expansion; that is the next step.

## Verification

Ground truth is not a spreadsheet — it is code: 44 labelled structures under
`crysh.controls` (diamond Si → 3D/CN 4, fcc Al → CN 12/cuboctahedral, W bcc → CN 8+6,
MoS₂ → prism/edge/2D, …) plus pymatgen oracle benchmarks in `crysh.research.oracle`.
The test suite pins ideal-structure values exactly and cross-checks the legacy
implementation for degenerate cases.

## Tests: three tiers = three install footprints

| tier | directory | needs | command |
|---|---|---|---|
| core | `tests/*.py` | `numpy` + `ase` | `pytest tests -q --ignore=tests/tables --ignore=tests/research` |
| tables | `tests/tables/` | `crysh[tables]` | `pytest tests -q --ignore=tests/research` |
| research | `tests/research/` | `crysh[research]` | `pytest tests -q` |

A tier that is not installed is never collected — and never silently skipped: the core tier
imports no pandas/pymatgen/matplotlib at module level, guarded by `tests/test_skeleton.py`
(which re-collects the core tier in a subprocess with those imports blocked). The controls CN
reference follows the same rule: `build_all(cn_method="crystalnn")` — the recipe behind the
frozen `controls_data/labels.yaml` — needs pymatgen and **raises** without it, instead of
quietly falling back; the dependency-free `"shell"` method has to be requested explicitly, and
`cn_method` in every label records what was actually used.

## License

MIT — see [LICENSE](LICENSE).
