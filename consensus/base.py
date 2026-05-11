from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform


def compute_run_weights(
    label_matrix: np.ndarray,
    n_objects: int,
    noise_label: int = -1,
) -> np.ndarray:
    n_runs = label_matrix.shape[0]
    weights = np.zeros(n_runs, dtype=np.float64)

    for t, labels in enumerate(label_matrix):
        valid = labels[labels != noise_label]
        if valid.size == 0:
            weights[t] = 0.0
            continue

        coverage = valid.size / n_objects
        unique_clusters = np.unique(valid)
        n_clusters = unique_clusters.size

        if n_clusters <= 1:
            weights[t] = coverage * 0.1
            continue

        counts = np.array(
            [np.sum(valid == c) for c in unique_clusters], dtype=np.float64
        )
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-15))
        max_entropy = np.log2(n_clusters)
        entropy_score = entropy / max_entropy if max_entropy > 0 else 1.0

        weights[t] = coverage * entropy_score

    total = weights.sum()
    if total > 0:
        weights *= n_runs / total
    else:
        weights = np.ones(n_runs, dtype=np.float64)

    return weights


def build_coassociation(
    label_matrix: np.ndarray,
    n_objects: int,
    noise_label: int = -1,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_runs = label_matrix.shape[0]
    if weights is None:
        weights = np.ones(n_runs, dtype=np.float64)

    C = np.zeros((n_objects, n_objects), dtype=np.float64)
    M = np.zeros((n_objects, n_objects), dtype=np.float64)

    for t, labels in enumerate(label_matrix):
        w = float(weights[t])
        valid_idx = np.where(labels != noise_label)[0]

        if valid_idx.size < 2:
            continue

        M[np.ix_(valid_idx, valid_idx)] += w

        for c in np.unique(labels[valid_idx]):
            members = valid_idx[labels[valid_idx] == c]
            if members.size >= 2:
                C[np.ix_(members, members)] += w

    with np.errstate(invalid="ignore", divide="ignore"):
        C_norm = np.where(M > 0, C / M, 0.0)

    np.fill_diagonal(C_norm, 1.0)
    return C_norm, M


def align_labels(ref: np.ndarray, new: np.ndarray, noise_label: int = -1) -> np.ndarray:
    ref_ids = np.unique(ref[ref != noise_label])
    new_ids = np.unique(new[new != noise_label])

    if ref_ids.size == 0 or new_ids.size == 0:
        return new.copy()

    conf = np.zeros((ref_ids.size, new_ids.size), dtype=np.int64)
    for r, rid in enumerate(ref_ids):
        for n, nid in enumerate(new_ids):
            conf[r, n] = np.sum((ref == rid) & (new == nid))

    row_ind, col_ind = linear_sum_assignment(-conf)

    mapping: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if c < new_ids.size:
            mapping[int(new_ids[c])] = int(ref_ids[r])

    aligned = new.copy()
    for old, new_val in mapping.items():
        aligned[new == old] = new_val

    return aligned


def pac_score(
    C_norm: np.ndarray,
    lower: float = 0.1,
    upper: float = 0.9,
) -> float:
    n = C_norm.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = C_norm[mask]
    return float(np.mean((vals > lower) & (vals < upper)))


def _pac_score_for_k(
    C_norm: np.ndarray,
    labels_k: np.ndarray,
    lower: float = 0.1,
    upper: float = 0.9,
) -> float:
    within_vals: List[float] = []
    between_vals: List[float] = []

    unique_labels = np.unique(labels_k)
    for c in unique_labels:
        mem = np.where(labels_k == c)[0]
        if mem.size < 2:
            continue
        sub = C_norm[np.ix_(mem, mem)]
        mask_off = ~np.eye(mem.size, dtype=bool)
        within_vals.extend(sub[mask_off].tolist())

    for i, ci in enumerate(unique_labels):
        for cj in unique_labels[i + 1:]:
            mi = np.where(labels_k == ci)[0]
            mj = np.where(labels_k == cj)[0]
            between_vals.extend(C_norm[np.ix_(mi, mj)].ravel().tolist())

    if not within_vals and not between_vals:
        return 1.0

    all_vals = np.array(within_vals + between_vals)
    return float(np.mean((all_vals > lower) & (all_vals < upper)))


def select_k_adaptive(
    C_norm: np.ndarray,
    k_range: Tuple[int, int],
    lower: float = 0.1,
    upper: float = 0.9,
    run_k_counts: Optional[List[int]] = None,
    flat_pac_threshold: float = 0.02,
) -> Tuple[int, Dict[int, float], str]:
    n = C_norm.shape[0]
    k_min = max(2, k_range[0])
    k_max = min(k_range[1], n - 1)
    if k_max < k_min:
        return k_min, {k_min: 1.0}, "fallback"

    ks = list(range(k_min, k_max + 1))
    cohesion: Dict[int, float] = {}
    pac_scores: Dict[int, float] = {}

    for k in ks:
        labels_k = coassoc_to_labels(C_norm, k)

        pac_scores[k] = _pac_score_for_k(C_norm, labels_k, lower, upper)

        within_vals: List[float] = []
        for c in np.unique(labels_k):
            mem = np.where(labels_k == c)[0]
            if mem.size < 2:
                continue
            sub = C_norm[np.ix_(mem, mem)]
            mask_off = ~np.eye(mem.size, dtype=bool)
            within_vals.append(float(sub[mask_off].mean()))
        cohesion[k] = float(np.mean(within_vals)) if within_vals else 0.0

    pac_vals = np.array([pac_scores[k] for k in ks])
    pac_range = float(pac_vals.max() - pac_vals.min())

    if pac_range >= flat_pac_threshold:
        optimal_k = ks[int(np.argmin(pac_vals))]
        return optimal_k, pac_scores, "pac"

    coh_vals = np.array([cohesion[k] for k in ks])
    coh_range = float(coh_vals.max() - coh_vals.min())

    if coh_range >= 0.01:
        gains = np.diff(coh_vals)
        rel_gains = gains / (gains.max() + 1e-12)
        mask = rel_gains < 0.3
        elbow_idx = int(np.argmax(mask)) if mask.any() else len(ks) - 1
        optimal_k = ks[elbow_idx]
        return optimal_k, pac_scores, "delta_cohesion"

    if run_k_counts is not None and len(run_k_counts) > 0:
        arr = np.array(run_k_counts, dtype=int)
        arr = arr[(arr >= k_min) & (arr <= k_max)]
        if arr.size > 0:
            vals, freqs = np.unique(arr, return_counts=True)
            optimal_k = int(vals[np.argmax(freqs)])
            return optimal_k, pac_scores, "mode_k"

    optimal_k = ks[int(np.argmax(coh_vals))]
    return optimal_k, pac_scores, "fallback"


def coassoc_to_labels(C_norm: np.ndarray, n_clusters: int) -> np.ndarray:
    n = C_norm.shape[0]
    if n_clusters >= n:
        return np.arange(n)

    dist = np.clip(1.0 - C_norm, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    raw = fcluster(Z, t=n_clusters, criterion="maxclust")
    return (raw - 1).astype(int)


class AbstractConsensus(ABC):

    name: str = "abstract"

    @abstractmethod
    def fit(self, X: np.ndarray) -> "AbstractConsensus":
        """Fit consensus on X of shape (n_samples, n_features)."""

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return consensus labels of shape (n_samples,)."""

    def get_stability_scores(self) -> Optional[np.ndarray]:
        return getattr(self, "stability_scores_", None)


__all__ = [
    "AbstractConsensus",
    "build_coassociation",
    "compute_run_weights",
    "align_labels",
    "pac_score",
    "select_k_adaptive",
    "coassoc_to_labels",
]