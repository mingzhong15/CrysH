"""Continuous per-structure descriptor fingerprint + PCA embedding (L0–L5).

Motivation (main-session review, 2026-09): the levels should emit *sufficiently
many continuous parameters and parameter distributions* as primary data (for
PCA/UMAP structure-similarity analysis); thresholds are only for the final
labeling step, and labels and the reduced space must cross-validate each other.
This module assembles the frozen column set below from the per-structure record
dict (pipeline §4 schema + the level diagnostics), so that every structure
becomes one point in a continuous descriptor space regardless of its labels.

FINGERPRINT_COLUMNS (order FROZEN — 53 columns as of v1.3-L4):

    L0  (5):  q_min, n_overlap_pairs, volume_norm, cell_kappa, aspect_ratio
    L1 (10):  d_090, d_100, d_110, d_120, d_135, d_150, d_175, d_200,
              dim_persistence, n_components
    L2 (10):  vacuum_gap, vacuum_fraction, surface_score, n_layers, span,
              f_max, cn_std, vacuum_score, slab_score, mono_score
              (the last three are the v3 soft-gate scores of crysh.morphology;
               records produced before v3 simply leave them NaN)
    L3  (5):  mean_cn, low_cn_fraction, high_cn_fraction, cn_min, cn_max
    L4 (16):  geom_ambiguous_fraction, geom_cnt_<label> for each of the 15
              GEOMETRY_LABELS (in contracts §3.7 order: linear,
              trigonal_planar, tetrahedral, square_planar,
              trigonal_bipyramidal, square_pyramidal, octahedral,
              trigonal_prismatic, cubic, icosahedral, cuboctahedral,
              hcp_like, bcc_like, other, ambiguous) —
              the bag-of-labels generalization of a per-atom one-hot, kept as
              raw counts so the column stays continuous
    L5  (5):  h_neigh_mean, f_corner, f_edge, f_face, n_distinct_l3
    meta (2): natom, n_elements

Rules:
* missing keys → NaN (never raise): the fingerprint degrades gracefully on
  records from older runs;
* `geom_label_counts` is accepted both as a dict[str, int] and as a JSON
  string of one (parquet round-trip / ui-explorer dataset.jsonl); when the
  key is absent entirely the count columns are NaN (unknown), while a
  present-but-omitted label counts as 0;
* `elements` is accepted as a list or a JSON string of one.

PCA (`pca_embedding`): plain-numpy SVD after mean-centering + z-score
standardization (zero-variance columns dropped, sklearn-free).  Sign convention
is fixed (largest-|component| of each loading positive) so embeddings are
reproducible.  The embedding is ``X_std @ W``.

Owner: postproc-atlas (module creation authorized by the integrator for the
L2-morphology v3 task).  Consumers: postproc-atlas/scripts/embed_qa.py.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

try:  # contracts §3.7 frozen order — single source of truth
    from crysh.geometry import GEOMETRY_LABELS  # type: ignore

    _GEOMETRY_LABELS: list[str] = list(GEOMETRY_LABELS)
except ImportError:  # pragma: no cover - level4 always merged in practice
    _GEOMETRY_LABELS = [
        "linear", "trigonal_planar", "tetrahedral", "square_planar",
        "trigonal_bipyramidal", "square_pyramidal", "octahedral",
        "trigonal_prismatic", "cubic", "icosahedral",
        "cuboctahedral", "hcp_like", "bcc_like",  # v1.3-L4 metallic labels
        "other", "ambiguous",
    ]

_GEOM_COUNT_COLUMNS = [f"geom_cnt_{lab}" for lab in _GEOMETRY_LABELS]

FINGERPRINT_COLUMNS: list[str] = [
    # L0 — validity
    "q_min", "n_overlap_pairs", "volume_norm", "cell_kappa", "aspect_ratio",
    # L1 — dimensionality spectrum (v1.3-L1 8-point grid)
    "d_090", "d_100", "d_110", "d_120", "d_135", "d_150", "d_175", "d_200",
    "dim_persistence", "n_components",
    # L2 — morphology (+ v3 soft-gate scores)
    "vacuum_gap", "vacuum_fraction", "surface_score", "n_layers", "span",
    "f_max", "cn_std", "vacuum_score", "slab_score", "mono_score",
    # L3 — coordination
    "mean_cn", "low_cn_fraction", "high_cn_fraction", "cn_min", "cn_max",
    # L4 — geometry (bag-of-labels over the frozen 15, v1.3-L4)
    "geom_ambiguous_fraction", *_GEOM_COUNT_COLUMNS,
    # L5 — tokenizer
    "h_neigh_mean", "f_corner", "f_edge", "f_face", "n_distinct_l3",
    # meta
    "natom", "n_elements",
]


def _as_float(value: Any) -> float:
    """Best-effort float conversion; NaN for anything non-numeric."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def _parse_counts(value: Any) -> dict[str, float]:
    """geom_label_counts as dict or JSON string -> {label: count}."""
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return {}
    if isinstance(value, dict):
        out: dict[str, float] = {}
        for k, v in value.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out
    return {}


def _n_elements(value: Any) -> float:
    """len(set(elements)); elements as list or JSON string; NaN if absent."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return float("nan")
    try:
        return float(len(set(str(e) for e in value)))
    except TypeError:
        return float("nan")


def structure_fingerprint(record: dict) -> np.ndarray:
    """Assemble the FINGERPRINT_COLUMNS vector for one record dict.

    Missing keys → NaN; `geom_label_counts` accepted as dict or JSON string;
    `elements` accepted as list or JSON string.
    """
    counts_missing = "geom_label_counts" not in record or record.get("geom_label_counts") is None
    counts = {} if counts_missing else _parse_counts(record.get("geom_label_counts"))
    out = np.empty(len(FINGERPRINT_COLUMNS), dtype=np.float64)
    for n, col in enumerate(FINGERPRINT_COLUMNS):
        if col.startswith("geom_cnt_"):
            # absent geom_label_counts -> NaN (unknown); present-but-missing
            # label -> 0 (that geometry simply has no atoms)
            out[n] = float("nan") if counts_missing else counts.get(
                col[len("geom_cnt_"):], 0.0)
        elif col == "n_elements":
            out[n] = _n_elements(record.get("elements"))
        else:
            out[n] = _as_float(record.get(col))
    return out


def build_fingerprint_matrix(records: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Batch `structure_fingerprint` over a records DataFrame.

    Returns (X, columns): X is (N, len(FINGERPRINT_COLUMNS)) float64 with NaN
    for missing values; `columns` lists the columns actually usable — columns
    that are NaN in EVERY row are dropped (a warning is printed, e.g. the v3
    soft-gate scores on records produced before crysh.morphology v3).
    """
    n = len(records)
    X = np.empty((n, len(FINGERPRINT_COLUMNS)), dtype=np.float64)
    for r, record in enumerate(records.to_dict("records")):
        X[r, :] = structure_fingerprint(record)
    all_nan = np.all(np.isnan(X), axis=0)
    dropped = [c for c, bad in zip(FINGERPRINT_COLUMNS, all_nan) if bad]
    if dropped:
        print(f"[fingerprint] WARNING: dropping all-NaN columns (missing in "
              f"every record): {dropped}")
    keep = ~all_nan
    return X[:, keep], [c for c, k in zip(FINGERPRINT_COLUMNS, keep) if k]


def pca_embedding(X, n_components: int = 2) -> dict:
    """PCA via numpy SVD: mean-center + z-score, drop zero-variance columns.

    Parameters
    ----------
    X : (N, p) array-like, finite (impute NaN before calling — the caller,
        e.g. embed_qa.py, owns the imputation policy).
    n_components : number of components to keep (clamped to the feasible max).

    Returns
    -------
    dict with
      X_std    — (N, p') standardized matrix (zero-variance columns removed)
      evr      — (k,) explained-variance ratio of the kept components
      loadings — (k, p') principal axes (rows), sign-fixed
      W        — (p', k) projection matrix; embedding = X_std @ W
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values; impute before PCA")
    n, p = X.shape
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    keep = sd > 0
    if not keep.any():
        raise ValueError("all columns have zero variance; PCA undefined")
    dropped = int(np.count_nonzero(~keep))
    if dropped:
        print(f"[fingerprint] PCA: dropping {dropped} zero-variance column(s)")
    X_std = (X[:, keep] - mu[keep]) / sd[keep]
    p_kept = int(keep.sum())
    k = int(max(1, min(n_components, p_kept, n)))
    # full SVD on the standardized matrix; X_std = U S Vt
    _, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    evr_all = S ** 2 / np.sum(S ** 2) if np.sum(S ** 2) > 0 else np.zeros_like(S)
    # sign convention: largest-|component| of each loading positive
    for row in range(Vt.shape[0]):
        j = int(np.argmax(np.abs(Vt[row])))
        if Vt[row, j] < 0:
            Vt[row] = -Vt[row]
    loadings = Vt[:k]
    W = Vt[:k].T
    return {
        "X_std": X_std,
        "evr": evr_all[:k],
        "loadings": loadings,
        "W": W,
    }
