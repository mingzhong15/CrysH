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

Every path is passed in explicitly: the library reads **no environment variables** and never
guesses a directory layout.

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
├── bond.py                periodic bond graph (calibrated pair table or covalent × λ)
├── dimensionality.py      d(λ) spectrum, integer-rank topology
├── validity.py            L0
├── morphology.py          L2
├── coord.py               L3
├── geometry.py            L4 (features + q4/q6 routing)
├── tokens.py              L5 tokens and sharing
├── metrics.py             vocabulary statistics (richness, effective diversity, accumulation)
├── records.py             map_record / map_structure / run_batch
├── controls.py            + controls_data/ — 44 labelled ground-truth structures
├── dev.py                 synthetic structures (smoke tests, not physics)
└── research/              optional research tooling
    ├── figdata.py  fingerprint.py  io.py  report.py
    └── oracle.py   pymatgen benchmarks    calibration.py  Phase-0 cutoff/CN tables
tests/                     runs standalone, no research data needed
```

## Verification

Ground truth is not a spreadsheet — it is code: 44 labelled structures under
`crysh.controls` (diamond Si → 3D/CN 4, fcc Al → CN 12/cuboctahedral, W bcc → CN 8+6,
MoS₂ → prism/edge/2D, …) plus pymatgen oracle benchmarks in `crysh.research.oracle`.
The test suite pins ideal-structure values exactly and cross-checks the legacy
implementation for degenerate cases.

## License

MIT — see [LICENSE](LICENSE).
