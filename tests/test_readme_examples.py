"""README 里的可执行示例必须真的能跑（防止文档漂移）。

只抽取 README 中标注为 python 的代码块里**不依赖外部文件**的那部分来跑：
manifest 那个例子需要真实语料，用 fixtures/dev 的合成结构代替。
"""

from __future__ import annotations

import re
from pathlib import Path

from ase.build import bulk

import crysh

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_quickstart_block_runs() -> None:
    text = README.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    quickstart = next(b for b in blocks if "map_record" in b)
    ns: dict = {}
    exec(compile(quickstart, "README.md", "exec"), ns)  # noqa: S102 - 我们自己的文档
    assert ns["rec"]["d_star"] == 3
    assert list(ns["rec"]) == crysh.RECORD_COLUMNS


def test_readme_batch_example_shape() -> None:
    """README 的批量示例：合成结构 + 显式配置，跑通并断言返回结构。"""
    import tempfile
    from pathlib import Path as P

    from crysh import MapperConfig, run_batch
    from crysh.dev import write_synthetic_subset

    with tempfile.TemporaryDirectory() as td:
        manifest, _subset = write_synthetic_subset(td, n=6)
        stats = run_batch(manifest, P(td) / "records.parquet", cfg=MapperConfig(scale="2k"),
                          n_proc=2)
    assert set(stats) == {"n_total", "n_ok", "n_failed"}
    assert stats["n_ok"] + stats["n_failed"] == stats["n_total"] == 6


def test_readme_claims_match_reality() -> None:
    """README 里写死的数字/列数必须与代码一致。"""
    assert len(crysh.RECORD_COLUMNS) == 59
    assert "59 列" in README.read_text(encoding="utf-8")
    rec = crysh.map_record(bulk("Si", "diamond", a=5.43))
    assert (rec["d_star"], rec["morphology_class"], rec["mean_cn"]) == (3, "dense_bulk", 4.0)
