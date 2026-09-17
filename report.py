"""CKT 报告/图（postproc-atlas 所有）：atlas 六图 + 过滤漏斗（fig7）。

- atlas_figs(records_parquet, out_dir)：fig1_dim_pie / fig2_morphology_bar /
  fig3_z_cn_heatmap / fig4_z_geom_heatmap / fig5_top_motifs / fig6_scatter。
- filter_waterfall(records_parquet, out_dir)：fig7_waterfall（L0→L4 漏斗 +
  L5 稀有 token 侧栏条形）。

数据本地化：`records_parquet` 两种模式——
  1. figdata 模式：传 `figdata/atlas` 目录（含 records_lite.parquet + side.json，
     由 ckt.figdata.export_atlas 在数据侧/集群产出）→ 本地零大 parquet 依赖；
  2. 全量 parquet 模式（集群/数据侧）：直接读 records_2k.parquet + pipeline
     侧文件（<stem>_{cn_counts,geom_counts,l3_tokens}.parquet）。
侧聚合缺失时退回基于 elements 列 + 典型 CN 表的确定性合成（图中标注
"[fallback: no side file]"），保证任意 §4 schema parquet 都能出图。

图均为 PNG、300 dpi、轴标签 + 标题（contracts.md §0）。
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 保险（env.sh 已设 MPLBACKEND=Agg）
import matplotlib.pyplot as plt

from .paths import KT_ROOT  # noqa: F401  (KT 工作目录，见 paths.py)
_DPI = 300

_DIM_LABELS = {0: "0D", 1: "1D", 2: "2D", 3: "3D"}
_DIM_COLORS = {0: "tab:red", 1: "tab:orange", 2: "tab:green", 3: "tab:blue"}

# fallback 用：常见元素典型 CN（与 pipeline mock 层同源，独立维护避免跨模块私有依赖）
_FALLBACK_CN = {
    "H": 1, "Li": 4, "B": 3, "C": 4, "N": 3, "O": 2, "F": 1, "Na": 4, "Mg": 6,
    "Al": 6, "Si": 4, "P": 3, "S": 2, "Cl": 1, "K": 6, "Ca": 8, "Ti": 6, "V": 6,
    "Cr": 6, "Mn": 6, "Fe": 6, "Co": 6, "Ni": 6, "Cu": 6, "Zn": 4, "Ga": 4, "Ge": 4,
    "As": 3, "Se": 2, "Br": 1, "Rb": 6, "Sr": 8, "Y": 8, "Zr": 8, "Nb": 6, "Mo": 6,
    "Ru": 6, "Rh": 6, "Pd": 6, "Ag": 6, "Cd": 6, "In": 4, "Sn": 6, "Sb": 3, "Te": 6,
    "I": 1, "Cs": 8, "Ba": 8, "Hf": 6, "Ta": 6, "W": 6, "Re": 6, "Os": 6, "Ir": 6,
    "Pt": 6, "Au": 6, "Hg": 6, "Tl": 6, "Pb": 6, "Bi": 6,
}


def _side(records_parquet: Path, suffix: str) -> Path:
    p = Path(records_parquet)
    return p.parent / f"{p.stem}{suffix}"


class _SideData:
    """fig3/4/5/7 侧聚合统一入口：pipeline 侧 parquet 或 figdata/atlas/side.json。"""

    def __init__(self, cn: pd.DataFrame | None = None,
                 geom: pd.DataFrame | None = None,
                 tokens: pd.Series | None = None,
                 rare: pd.Series | None = None,
                 n_rare_structs: int = 0):
        self.cn = cn          # DataFrame[element, cn, count] | None
        self.geom = geom      # DataFrame[element, geometry, count] | None
        self.tokens = tokens  # Series[token -> count] | None
        self.rare = rare      # Series[token -> count(≤2)] | None
        self.n_rare_structs = n_rare_structs

    @classmethod
    def from_parquet(cls, records_parquet: Path) -> "_SideData":
        side = _side(records_parquet, "_cn_counts.parquet")
        cn = pd.read_parquet(side) if side.exists() else None
        side = _side(records_parquet, "_geom_counts.parquet")
        geom = pd.read_parquet(side) if side.exists() else None
        tokens = rare = None
        n_rare = 0
        side = _side(records_parquet, "_l3_tokens.parquet")
        if side.exists():
            tok = pd.read_parquet(side)
            tokens = tok["token"].value_counts()
            r = tokens[tokens <= 2]
            if len(r):
                rare = r.sort_values(ascending=False)
                n_rare = int(tok.loc[tok["token"].isin(set(r.index)),
                                     "structure_id"].nunique())
        return cls(cn, geom, tokens, rare, n_rare)

    @classmethod
    def from_json(cls, side_json: Path) -> "_SideData":
        d = json.loads(Path(side_json).read_text(encoding="utf-8"))
        cn = pd.DataFrame(d["cn_counts"], columns=["element", "cn", "count"]) \
            if d["cn_counts"] else None
        geom = pd.DataFrame(d["geom_counts"],
                            columns=["element", "geometry", "count"]) \
            if d["geom_counts"] else None
        tokens = pd.Series({t: c for t, c in d["l3_token_counts"]}) \
            if d["l3_token_counts"] else None
        rare = pd.Series({t: c for t, c in d["rare_tokens"]}) \
            if d["rare_tokens"] else None
        return cls(cn, geom, tokens, rare, int(d["n_rare_structures"]))


def _atlas_inputs(fig_source: Path, lite_path: Path | None = None,
                  side_json: Path | None = None) -> tuple[pd.DataFrame, _SideData]:
    """fig_source：figdata/atlas 目录 或 records parquet 路径 → (df, side)。

    lite_path/side_json 显式指定时（20k 等带后缀规模），直接读该
    records_lite_{scale}.parquet + side_{scale}.json，忽略 fig_source。"""
    if lite_path is not None:
        df = pd.read_parquet(Path(lite_path))
        sj = Path(side_json) if side_json is not None else Path(lite_path).parent / "side.json"
        side = _SideData.from_json(sj) if sj.is_file() else _SideData()
        return df, side
    fig_source = Path(fig_source)
    if fig_source.is_dir():
        lite = fig_source / "records_lite.parquet"
        if not lite.is_file():
            raise FileNotFoundError(
                f"{lite} 不存在 —— 先运行 `python -m ckt.figdata export-atlas`")
        df = pd.read_parquet(lite)
        side_json = fig_source / "side.json"
        side = _SideData.from_json(side_json) if side_json.is_file() else _SideData()
        return df, side
    return pd.read_parquet(fig_source), _SideData.from_parquet(fig_source)


def _waterfall_counts(df: pd.DataFrame, l0_eliminate: bool = False) -> dict[str, int]:
    """§5 漏斗（L0→L4）各阶段存活数；L5 稀有 token 不入漏斗。

    规则（contracts.md §5 + v1.3 L0 语义）：
    - **L0（v1.3）**：Alexandria 上只计算统计、不消除（l0_eliminate=False 默认）——
      "+L0 record" 阶段透传全部 N；flag 计数记入 L0_flags 供图注。
      l0_eliminate=True（MatterGen 判异模式）时只消除 **core_overlap**（OpenMX
      冻芯重叠硬 floor）；其余 L0 flag 仍只记录。
    - +L1: ambiguous_dim_flag == False（dim_persistence >= 0.5）
    - +L2: morphology_class != "ambiguous"
    - +L3: low_cn_fraction <= 0.3 且 high_cn_fraction <= 0.3
    - +L4: geom_ambiguous_fraction <= 0.5
    """
    n0 = int(len(df))
    flag_cols = ["core_overlap_flag", "overlap_flag", "overpacked_flag",
                 "sparse_flag", "pathological_cell_flag"]
    flag_counts: dict[str, int] = {}
    for c in flag_cols:
        if c in df.columns:
            flag_counts[c] = int(df[c].fillna(False).astype(bool).sum())
        else:
            flag_counts[c] = 0
    if l0_eliminate:
        d1 = df[~df["core_overlap_flag"].fillna(False).astype(bool)]
    else:
        d1 = df  # Alexandria：L0 只计算统计，不消除（用户指令 2026-09-09）
    d2 = d1[~d1["ambiguous_dim_flag"].fillna(False).astype(bool)]
    d3 = d2[d2["morphology_class"].fillna("ambiguous") != "ambiguous"]
    d4 = d3[(d3["low_cn_fraction"].fillna(0.0) <= 0.3)
            & (d3["high_cn_fraction"].fillna(0.0) <= 0.3)]
    d5 = d4[d4["geom_ambiguous_fraction"].fillna(1.0) <= 0.5]
    return {"N": n0, "L0_pass": int(len(d1)), "L1_pass": int(len(d2)),
            "L2_pass": int(len(d3)), "L3_pass": int(len(d4)), "L4_pass": int(len(d5)),
            "L0_flags": flag_counts, "l0_eliminate": bool(l0_eliminate)}


# ---------------------------------------------------------------------------
# atlas 六图
# ---------------------------------------------------------------------------

def _fig1_dim_pie(df: pd.DataFrame, out_dir: Path) -> None:
    vc = df["d_star"].dropna().astype(int).value_counts()
    order = [d for d in (3, 2, 1, 0) if d in vc.index]
    vals = [int(vc[d]) for d in order]
    labels = [f"{_DIM_LABELS[d]} (n={int(vc[d])})" for d in order]
    colors = [_DIM_COLORS[d] for d in order]
    amb = float(df["ambiguous_dim_flag"].fillna(False).astype(bool).mean()) if len(df) else 0.0
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    wedges, _txts, _auts = ax.pie(
        vals, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90,
        counterclock=False, textprops={"fontsize": 9})
    ax.set_title(f"Fig.1 Dimensionality distribution (d_star), N={len(df)}\n"
                 f"ambiguous_dim_flag fraction: {amb:.2%}")
    fig.savefig(out_dir / "fig1_dim_pie.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _fig2_morphology_bar(df: pd.DataFrame, out_dir: Path) -> None:
    vc = df["morphology_class"].dropna().value_counts()
    if len(vc) == 0:
        return
    labels = list(vc.index)[::-1]
    vals = list(vc.values)[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(labels, vals, color="tab:cyan", edgecolor="k", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v} ({v / max(1, len(df)):.1%})", va="center", fontsize=8)
    ax.set_xlabel("number of structures")
    ax.set_ylabel("morphology class")
    ax.set_title(f"Fig.2 Morphology class distribution, N={len(df)}")
    fig.savefig(out_dir / "fig2_morphology_bar.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _fig3_z_cn_heatmap(df: pd.DataFrame, side: _SideData, out_dir: Path) -> None:
    fallback = False
    if side.cn is not None:
        piv = side.cn.pivot_table(index="element", columns="cn", values="count",
                                  aggfunc="sum", fill_value=0)
    else:  # fallback：elements 列 + 典型 CN 表，确定性合成
        fallback = True
        rng = np.random.default_rng(20240907)
        counts: Counter = Counter()
        for elems in df["elements"].dropna():
            for e in elems:
                base = _FALLBACK_CN.get(e, 6)
                c = int(np.clip(base + rng.integers(-1, 2), 0, 12))
                counts[(e, c)] += 1
        piv = pd.DataFrame([(e, c, n) for (e, c), n in counts.items()],
                           columns=["element", "cn", "count"]).pivot_table(
            index="element", columns="cn", values="count", aggfunc="sum", fill_value=0)
    if len(piv) == 0:
        return
    rows = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index].head(20)
    cols = [c for c in range(0, 13) if c in piv.columns] or list(piv.columns)
    X = rows.reindex(columns=cols, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(np.log1p(X.values), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([str(c) for c in cols])
    ax.set_yticks(range(len(X.index)))
    ax.set_yticklabels(list(X.index), fontsize=8)
    ax.set_xlabel("coordination number (CN)")
    ax.set_ylabel("central element")
    note = " [fallback: no side file]" if fallback else ""
    ax.set_title(f"Fig.3 Element × CN heatmap (log1p counts), top-20 elements{note}")
    fig.colorbar(im, ax=ax, label="log1p(count)")
    fig.savefig(out_dir / "fig3_z_cn_heatmap.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _fig4_z_geom_heatmap(df: pd.DataFrame, side: _SideData, out_dir: Path) -> None:
    fallback = False
    if side.geom is not None:
        piv = side.geom.pivot_table(index="element", columns="geometry",
                                    values="count", aggfunc="sum", fill_value=0)
    else:  # fallback：elements 列 + CN 路由规则，确定性合成
        fallback = True
        rng = np.random.default_rng(20240907)
        counts: Counter = Counter()
        for elems in df["elements"].dropna():
            for e in elems:
                base = _FALLBACK_CN.get(e, 6)
                c = int(np.clip(base + rng.integers(-1, 2), 0, 12))
                if c <= 1:
                    g = "other"
                elif c == 2:
                    g = "linear"
                elif c == 3:
                    g = "trigonal_planar"
                elif c == 4:
                    g = "tetrahedral"
                elif c == 5:
                    g = "trigonal_bipyramidal"
                elif c == 6:
                    g = "octahedral"
                else:
                    g = "other"
                counts[(e, g)] += 1
        piv = pd.DataFrame([(e, g, n) for (e, g), n in counts.items()],
                           columns=["element", "geometry", "count"]).pivot_table(
            index="element", columns="geometry", values="count", aggfunc="sum", fill_value=0)
    if len(piv) == 0:
        return
    rows = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index].head(20)
    cols = list(piv.columns[np.argsort(-piv.sum(axis=0).values)][:12])
    X = rows.reindex(columns=cols, fill_value=0)
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    im = ax.imshow(np.log1p(X.values), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(X.index)))
    ax.set_yticklabels(list(X.index), fontsize=8)
    ax.set_xlabel("coordination geometry")
    ax.set_ylabel("central element")
    note = " [fallback: no side file]" if fallback else ""
    ax.set_title(f"Fig.4 Element × geometry heatmap (log1p counts), top-12 geometries{note}")
    fig.colorbar(im, ax=ax, label="log1p(count)")
    fig.savefig(out_dir / "fig4_z_geom_heatmap.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _fig5_top_motifs(df: pd.DataFrame, side: _SideData, out_dir: Path) -> None:
    fallback = False
    if side.tokens is not None:
        vc = side.tokens.head(20)
    else:  # fallback：元素 × 共享模式合成 top motifs（仅示意）
        fallback = True
        rng = np.random.default_rng(20240907)
        counts: Counter = Counter()
        for _, row in df.iterrows():
            elems = list(row.get("elements") or [])
            if not elems:
                continue
            e = elems[0]
            base = _FALLBACK_CN.get(e, 6)
            c = int(np.clip(base + rng.integers(-1, 2), 1, 12))
            g = "octahedral" if c == 6 else "tetrahedral"
            hist = "".join(f"{x}{rng.integers(1, 4)}" for x in sorted(set(elems)))
            counts[f"{e}|{c}|{g}|{hist}"] += 1
        vc = pd.Series(dict(sorted(counts.items(), key=lambda kv: -kv[1])[:20]))
    if len(vc) == 0:
        return
    labels = list(vc.index)[::-1]
    vals = list(vc.values)[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    ax.barh(labels, vals, color="tab:olive", edgecolor="k", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(v, i, f"  {v}", va="center", fontsize=8)
    ax.set_xlabel("occurrence count (structures)")
    ax.set_ylabel("L3 motif token (Z|CN|geometry|neighbor-hist)")
    note = " [fallback: no side file]" if fallback else ""
    ax.set_title(f"Fig.5 Top-20 L3 motifs, N={len(df)}{note}")
    fig.savefig(out_dir / "fig5_top_motifs.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _finite_mask(df: pd.DataFrame, *cols: str) -> pd.Series:
    """非有限值过滤掩码：L0 的 q_min 对无近邻结构为 +inf（真模块输出），
    任何入图数值列（volume_norm / mean_cn / q_min …）一律先滤掉非有限值。"""
    mask = pd.Series(True, index=df.index)
    for c in cols:
        mask &= df[c].map(np.isfinite)
    return mask


def _fig6_scatter(df: pd.DataFrame, out_dir: Path) -> None:
    # q_min=+inf（无近邻结构）等在入图前过滤；symlog 兜底非正值
    sub = df[_finite_mask(df, "volume_norm", "mean_cn", "q_min")].copy()
    n_dropped = len(df) - len(sub)
    if len(sub) == 0:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for d in (0, 1, 2, 3):
        m = sub["d_star"] == d
        if m.any():
            ax.scatter(sub.loc[m, "volume_norm"], sub.loc[m, "mean_cn"], s=14, alpha=0.45,
                       c=_DIM_COLORS[d], label=_DIM_LABELS[d], edgecolors="none",
                       rasterized=True)
    if (sub["volume_norm"] > 0).all():
        ax.set_xscale("log")
    else:
        ax.set_xscale("symlog")
    ax.set_xlabel("volume_norm  ν = V / Σ(4π/3 r_cov³)")
    ax.set_ylabel("mean_cn")
    ax.legend(title="d_star", loc="best", fontsize=8)
    drop_note = f" | dropped non-finite: {n_dropped}" if n_dropped else ""
    ax.set_title(f"Fig.6 mean_cn vs volume_norm colored by d_star, N={len(sub)}{drop_note}")
    fig.savefig(out_dir / "fig6_scatter.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def atlas_figs(records_parquet: Path, out_dir: Path,
               lite_path: Path | None = None, side_json: Path | None = None) -> None:
    """atlas 六图（fig1–fig6）。records_parquet 可为 figdata/atlas 目录
    （figdata 模式）或 records parquet 路径（全量模式）；lite_path/side_json
    供带规模后缀的 figdata（如 records_lite_20k.parquet）显式指定。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, side = _atlas_inputs(Path(records_parquet), lite_path, side_json)
    _fig1_dim_pie(df, out_dir)
    _fig2_morphology_bar(df, out_dir)
    _fig3_z_cn_heatmap(df, side, out_dir)
    _fig4_z_geom_heatmap(df, side, out_dir)
    _fig5_top_motifs(df, side, out_dir)
    _fig6_scatter(df, out_dir)


# ---------------------------------------------------------------------------
# 过滤漏斗（fig7）
# ---------------------------------------------------------------------------

def filter_waterfall(records_parquet: Path, out_dir: Path,
                      lite_path: Path | None = None,
                      side_json: Path | None = None) -> dict[str, int]:
    """fig7_waterfall：L0→L4 漏斗 + L5 稀有 token（子集内计数 ≤2）侧栏条形。

    records_parquet 可为 figdata/atlas 目录（figdata 模式）或 records parquet
    路径（全量模式）。返回值：各阶段存活数 dict（{"N","L0_pass",...,"L4_pass"}），
    便于测试与集成；调用方可忽略。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, side = _atlas_inputs(Path(records_parquet), lite_path, side_json)
    counts = _waterfall_counts(df)
    n0 = counts["N"]

    # L5 稀有 token（子集内计数 ≤ 2 的 l3 token；不入漏斗）
    rare_series = side.rare
    n_rare_structs = side.n_rare_structs

    fig = plt.figure(figsize=(11.2, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1.0], wspace=0.28)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_r = fig.add_subplot(gs[0, 1])

    stage_names = ["N (all)", "+L0 record", "+L1 pass", "+L2 pass", "+L3 pass", "+L4 pass"]
    vals = [counts["N"], counts["L0_pass"], counts["L1_pass"], counts["L2_pass"],
            counts["L3_pass"], counts["L4_pass"]]
    colors = ["tab:gray"] + ["tab:green"] + ["tab:blue"] * 4
    bars = ax_l.bar(stage_names, vals, color=colors, edgecolor="k", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax_l.text(b.get_x() + b.get_width() / 2, v + n0 * 0.01,
                  f"{v}\n({v / max(1, n0):.1%})", ha="center", va="bottom", fontsize=8)
    for i in range(2, len(vals)):
        drop = vals[i - 1] - vals[i]
        ax_l.text(i, (vals[i - 1] + vals[i]) / 2, f"−{drop}", ha="center",
                  va="center", fontsize=8, color="tab:red")
    ax_l.set_ylim(0, n0 * 1.18)
    ax_l.set_ylabel("surviving structures")
    ax_l.set_xticks(range(len(stage_names)))
    ax_l.set_xticklabels(stage_names, rotation=20, ha="right", fontsize=8)
    # v1.3：L0 只记录——flag 计数以图注呈现，不做消除
    fc = counts.get("L0_flags", {})
    l0_note = "L0 (record-only): " + ", ".join(
        f"{k.replace('_flag', '')}={v}" for k, v in fc.items())
    l0_note += ("\n(elimination disabled on Alexandria; MatterGen 判异模式仅 core_overlap 拦截)"
                if not counts.get("l0_eliminate") else
                "\n(MatterGen 判异模式：仅 core_overlap 被消除)")
    ax_l.set_title(f"Fig.7 Filter waterfall (L0→L4), N={n0}\n{l0_note}")

    if rare_series is not None and len(rare_series):
        top = rare_series.head(15)
        rlabels = list(top.index)[::-1]
        rvals = list(top.values)[::-1]
        ax_r.barh(rlabels, rvals, color="tab:orange", edgecolor="k", linewidth=0.5)
        for i, v in enumerate(rvals):
            ax_r.text(v, i, f" {v}", va="center", fontsize=8)
        ax_r.set_xlim(0, 3)
        ax_r.set_xlabel("structures containing token")
        ax_r.set_title(f"L5 rare tokens (count ≤ 2), side bar\n"
                       f"{n_rare_structs} structures ({n_rare_structs / max(1, n0):.2%}) "
                       f"carry ≥1 rare L3 token")
    else:
        ax_r.text(0.5, 0.5, "rare-token side data unavailable\n"
                            "(run `python -m ckt.figdata export-atlas`\n"
                            " or pipeline.run_batch to produce side data)",
                  ha="center", va="center", fontsize=9, transform=ax_r.transAxes)
        ax_r.set_axis_off()
        ax_r.set_title("L5 rare tokens (count ≤ 2), side bar")

    fig.savefig(out_dir / "fig7_waterfall.png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return counts
