"""最终 per-structure record schema（contracts.md §4 的代码化，集成者维护）。"""

RECORD_COLUMNS = [
    # 元信息
    "structure_id", "source_line", "formula", "elements", "natom",
    # Level 0 — validity（v1.3：core_overlap 硬 floor + ν 拆分）
    "q_min", "n_overlap_pairs", "volume_norm", "cell_kappa", "aspect_ratio",
    "n_core_overlap_pairs", "core_overlap_flag",
    "overlap_flag", "overpacked_flag", "sparse_flag",
    "extreme_volume_flag", "pathological_cell_flag",
    # Level 1 — dimensionality（v1.3-L1：网格扩至 8 点 + 逐分量多标签）
    "d_090", "d_100", "d_110", "d_120", "d_135", "d_150", "d_175", "d_200",
    "d_star", "dim_persistence", "n_components", "dim_components", "ambiguous_dim_flag",
    # Level 2 — morphology（v1.3-L2：附加诊断列，供 fingerprint/embed_qa）
    "vacuum_gap", "vacuum_fraction", "surface_score", "porous_candidate",
    "morphology_class", "n_layers", "span", "f_max", "cn_std",
    "vacuum_score", "slab_score", "mono_score",
    # Level 3 — coordination（v1.3-L3：小胞镜像混叠守卫）
    "mean_cn", "low_cn_fraction", "high_cn_fraction", "cn_min", "cn_max",
    "cn_cell_warning", "cn_cell_margin",
    # Level 4 — geometry
    "geom_top_label", "geom_ambiguous_fraction", "geom_label_counts",
    # Level 5 — tokens
    "n_distinct_l3", "h_neigh_mean", "f_corner", "f_edge", "f_face",
    "sharing_top_label",
    # 汇总
    "quality_flags",
]

LAMBDA_COLS = {
    0.90: "d_090", 1.00: "d_100", 1.10: "d_110",
    1.20: "d_120", 1.35: "d_135", 1.50: "d_150",
    1.75: "d_175", 2.00: "d_200",
}
