# CrysH — Crystal Hierarchy

**A multi-layer crystal knowledge mapper.** CrysH turns a crystal structure into a layered
description — from *is this structure valid* up to *what local building blocks is it made of,
and how are they connected* — and emits those layers as flat tables you can index, query and
aggregate over millions of structures.

```
L0 Validity            is the cell/packing physically sane?
L1 Dimensionality      d(λ) spectrum → d*, persistence, per-component labels
L2 Morphology          vacuum gap, surface-likeness, layers, porous candidates
L3 Local building blocks   m = (Z, CN, geometry, neighbour chemistry, distortion)
L4 Motif-network architecture   motif super-node graph (corner/edge/face sharing)
```

## Status

**v0.1.0 is under construction.** This repository currently holds the project skeleton; the
library code is being extracted and refactored from the research workspace that produced it.
The public API below is the target for v0.1.0.

## Design goals

1. **Layered, not monolithic.** Each layer is a pure function over an ASE `Atoms` plus an
   explicit config; layers compose into a record table.
2. **Small core.** The core library depends on `numpy` + `ase` only. Tables (`parquet`),
   pymatgen-based oracle benchmarks and plotting live in optional extras.
3. **No filesystem assumptions.** Output paths are passed in; the library never guesses a
   workspace layout and reads no environment variables.
4. **One neighbour list, reused.** Bond graph, dimensionality spectrum, coordination shells
   and the motif network all consume a single shared neighbour list.

<!--
## Quickstart (target API for v0.1.0)

```python
from ase.io import read
import crysh

atoms = read("POSCAR")
rec = crysh.map_structure(atoms)               # structure-level record (one row)
sites = crysh.local_environment(atoms)         # L3: one row per site (m_i)
net = crysh.motif_network(atoms, sites)        # L4: motif super-node graph
```
-->

## Repository layout (v0.1.0 target)

```
src/crysh/
├── config.py           MapperConfig — λ grid, kernels, thresholds, output roots
├── kernels.py          shared neighbour list, bond graph, integer-rank topology, shells
├── validity.py         L0
├── dimensionality.py   L1
├── morphology.py       L2
├── localenv.py         L3  — m_i
├── motifnet.py         L4  — motif super-node graph
├── tokens.py           token vocabulary, richness / effective diversity / accumulation
├── records.py          per-structure record assembly
├── controls.py         labelled ground-truth structures used for verification
└── tables/             frozen lookup tables (with checksums)
tests/                  unit tests, runnable without any research data
```

## License

MIT — see [LICENSE](LICENSE).
