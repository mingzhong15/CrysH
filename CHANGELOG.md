# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project skeleton: packaging (`src/` layout), CI, license, citation metadata.

### Planned for v0.1.0
- `crysh.config.MapperConfig` — explicit λ grid, kernel parameters and thresholds.
- `crysh.kernels` — one shared neighbour list, one bond graph builder, one integer-rank
  topology machine, coordination-shell splitting.
- `crysh.validity` — L0: `q_min`, normalised volume ν, cell condition, core-overlap flags.
- `crysh.dimensionality` — L1: d(λ) spectrum, `d_star`, persistence, per-component labels.
- `crysh.morphology` — L2: vacuum gap, surface score, layer count, porous candidates.
- `crysh.localenv` — L3: motif instance `m = (Z, CN, geometry, chemistry, distortion)`.
- `crysh.motifnet` — L4: motif super-node graph with corner/edge/face sharing and motif dims.
- `crysh.tokens`, `crysh.records`, `crysh.controls`.
- Frozen lookup tables shipped with checksums.
