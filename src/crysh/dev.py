"""开发/演示用合成结构（**不属于 mapper 的物理路径**，故与核心分层隔离）。

用途：在没有真实语料的环境里冒烟整条链（CI、容器、新机器），以及给文档出示例图。
这里生成的是**结构**（准格子堆积 + 扰动），随后仍然走 `crysh.records` 的**真计算链**——
不是"假指标"。所以本模块只负责造结构，不负责造 record。

刻意**不导出**到 `crysh` 顶层命名空间：核心 API 里不该出现"假数据"入口。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

__all__ = ["synthetic_atoms", "write_synthetic_subset", "make_synthetic_records"]

#: 造结构时轮换的元素池（金属 / 阴离子 / 轻元素），让 CN、几何与化学环境有分布
_ELEM_POOLS = {
    "metal": ["Li", "Na", "Mg", "Al", "Ca", "Ti", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Zr", "Mo"],
    "anion": ["O", "S", "F", "Cl", "Br", "N", "P"],
    "light": ["H", "B", "C", "Si"],
}


def synthetic_atoms(i: int, rng: np.random.Generator | None = None) -> Atoms:
    """造第 `i` 个合成结构：准格子堆积 + 扰动，避免随机重叠把 L0 flag 打满。

    15% 的结构用带剪切的对角晶胞，用来覆盖 L0 的 `cell_kappa`/`aspect_ratio`
    分支（否则那两条分支在冒烟里永远走不到）。
    """
    rng = rng or np.random.default_rng(12345 + i)
    n_uni = int(rng.choice([2, 2, 3, 3, 4]))
    if n_uni == 2:
        elems = [str(rng.choice(_ELEM_POOLS["metal"])), str(rng.choice(_ELEM_POOLS["anion"]))]
    elif n_uni == 3:
        elems = [str(rng.choice(_ELEM_POOLS["metal"])), str(rng.choice(_ELEM_POOLS["anion"])),
                 str(rng.choice(_ELEM_POOLS["light"]))]
    else:
        elems = [str(e) for e in rng.choice(_ELEM_POOLS["metal"], size=2, replace=False)] + [
            str(rng.choice(_ELEM_POOLS["anion"])), str(rng.choice(_ELEM_POOLS["light"]))]

    counts = rng.integers(1, 5, size=len(elems)).astype(int)
    if counts.sum() < 2:
        counts[0] += 1
    symbols: list[str] = []
    for e, c in zip(elems, counts, strict=False):
        symbols += [e] * int(c)

    n = len(symbols)
    g = max(1, int(np.ceil(n ** (1 / 3))))
    idx = rng.choice(g ** 3, size=n, replace=False)
    coords = np.stack(np.unravel_index(idx, (g, g, g)), axis=1).astype(float)
    pos = (coords + rng.uniform(0.15, 0.85, size=(n, 3))) / g
    cell = np.diag(rng.uniform(3.5, 9.0, size=3))
    if rng.random() < 0.15:
        cell = cell + rng.uniform(-1.2, 1.2, size=(3, 3)) * np.tril(np.ones((3, 3)))

    atoms = Atoms(symbols=symbols, scaled_positions=pos, cell=cell, pbc=True)
    atoms.info["structure_id"] = f"synthetic{i:06d}"
    atoms.info["source_line"] = int(i)
    atoms.info["elements"] = elems
    atoms.info["formula"] = atoms.get_chemical_formula(mode="metal")
    return atoms


def write_synthetic_subset(out_dir: Path | str, n: int = 200, seed: int = 12345,
                           manifest: Path | str | None = None) -> tuple[Path, Path]:
    """写 `n` 个合成结构为 POSCAR，并生成 `run_batch` 可直接消费的 manifest。

    Returns
    -------
    (manifest_path, subset_dir)
        直接喂给 :func:`crysh.records.run_batch` 即可跑通整条链。
    """
    out_dir = Path(out_dir)
    subset = out_dir / f"subset_synthetic{n}"
    subset.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest) if manifest is not None else out_dir / f"synthetic{n}.jsonl"
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(int(n)):
        atoms = synthetic_atoms(i, rng)
        sid = str(atoms.info["structure_id"])
        path = subset / f"{sid}.vasp"
        ase_write(path, atoms, format="vasp", direct=True)
        rows.append({
            "path": str(path),
            "structure_id": sid,
            "source_line": atoms.info["source_line"],
            "elements": atoms.info["elements"],
            "formula": atoms.info["formula"],
        })
    manifest_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return manifest_path, subset


def make_synthetic_records(out_parquet: Path | str, n: int = 2000, seed: int = 42,
                           cfg=None) -> Path:
    """造 `n` 个合成结构并跑**真计算链**，产出一份 records parquet（+ 侧文件）。

    用途：出图/报告的本地冒烟（真实语料只在集群）。注意语义——**结构是合成的，
    指标不是**：这条路径与真实结构走的是同一个 `crysh.records.map_structure`。

    历史名：研究工程里叫 `make_mock_records`（当时的实现会造假指标）；现在只造假结构。
    """
    import numpy as np

    from crysh.config import MapperConfig
    from crysh.records import _write_outputs, map_structure

    out_parquet = Path(out_parquet)
    cfg = cfg or MapperConfig()
    rng = np.random.default_rng(seed)
    records, sides = [], []
    for i in range(int(n)):
        rec, side = map_structure(synthetic_atoms(i, rng), cfg)
        records.append(rec)
        sides.append(side)
    _write_outputs(records, sides, out_parquet)
    return out_parquet
