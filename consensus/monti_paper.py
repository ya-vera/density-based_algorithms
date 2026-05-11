from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .base import AbstractConsensus


def _resample(
    x: np.ndarray, frac: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    size = max(2, int(n * frac))
    size = min(size, n)
    idx = rng.choice(n, size=size, replace=False)
    return idx, x[idx, :]


def relabel_subsample_unique_noise(
    sub_labels: np.ndarray, noise_label: int = -1
) -> np.ndarray:
    sub_labels = np.asarray(sub_labels, dtype=int).copy()
    noise_mask = sub_labels == noise_label
    if not noise_mask.any():
        return sub_labels
    if noise_mask.all():
        sub_labels[:] = np.arange(sub_labels.shape[0], dtype=int)
        return sub_labels
    next_label = int(sub_labels[~noise_mask].max()) + 1
    for pos in np.where(noise_mask)[0]:
        sub_labels[pos] = next_label
        next_label += 1
    return sub_labels


def compute_connectivity_matrix(labels: np.ndarray) -> np.ndarray:
    out_of_bag_idx = np.where(labels == -1)[0]
    connectivity_matrix = np.equal.outer(labels, labels).astype(np.float64)
    if out_of_bag_idx.size > 1:
        rows, cols = zip(*list(combinations(out_of_bag_idx, 2)))
        connectivity_matrix[rows, cols] = 0.0
        connectivity_matrix[cols, rows] = 0.0
    connectivity_matrix[out_of_bag_idx, out_of_bag_idx] = 0.0
    return connectivity_matrix


def compute_identity_matrix(n: int, resampled_indices: np.ndarray) -> np.ndarray:
    identity_matrix = np.zeros((n, n), dtype=np.float64)
    if resampled_indices.size < 2:
        identity_matrix[np.ix_(resampled_indices, resampled_indices)] = 1.0
        return identity_matrix
    rows, cols = zip(*list(combinations(resampled_indices, 2)))
    identity_matrix[rows, cols] = 1.0
    identity_matrix[cols, rows] = 1.0
    identity_matrix[resampled_indices, resampled_indices] = 1.0
    return identity_matrix


def compute_consensus_matrix(
    connectivity_matrices: List[np.ndarray],
    identity_matrices: List[np.ndarray],
) -> np.ndarray:
    s = np.sum(identity_matrices, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(s > 0, np.sum(connectivity_matrices, axis=0) / s, 0.0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(out, 1.0)
    return out


def _area_under_cdf_values(vals: np.ndarray) -> float:
    if vals.size == 0:
        return 0.0
    arr = np.clip(vals.astype(np.float64), 0.0, 1.0)
    hist, bins = np.histogram(arr, bins=50, range=(0.0, 1.0), density=True)
    bin_widths = np.diff(bins)
    ecdf = np.cumsum(hist * bin_widths) 
    return float(np.sum(ecdf * bin_widths)) 


def _area_under_cdf_for_partition(C: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    vals_list: List[float] = []
    for c in np.unique(labels):
        if c < 0:
            continue
        mem = np.where(labels == c)[0]
        if mem.size < 2:
            continue
        sub = C[np.ix_(mem, mem)]
        mask = ~np.eye(mem.size, dtype=bool)
        vals_list.extend(sub[mask].ravel().tolist())
    if not vals_list:
        return 0.0
    return _area_under_cdf_values(np.array(vals_list, dtype=np.float64))


def _cut_consensus(C: np.ndarray, k: int) -> np.ndarray:
    n = C.shape[0]
    if k >= n:
        return np.arange(n, dtype=int)
    dist = np.clip(1.0 - C, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    return (fcluster(Z, t=k, criterion="maxclust") - 1).astype(int)


def _best_k_knee(k_range: List[int], auc_values: List[float]) -> int:
    if len(k_range) == 1:
        return k_range[0]
    xs = np.array(k_range, dtype=float)
    ys = np.array(auc_values, dtype=float)
    if ys.max() == ys.min():
        return k_range[0]
    xs_n = (xs - xs.min()) / (xs.max() - xs.min())
    ys_n = (ys - ys.min()) / (ys.max() - ys.min())
    x0, y0 = xs_n[0], ys_n[0]
    x1, y1 = xs_n[-1], ys_n[-1]
    dx, dy = x1 - x0, y1 - y0
    line_len = np.hypot(dx, dy)
    if line_len < 1e-12:
        return k_range[0]
    dists = np.abs(dy * xs_n - dx * ys_n + x1 * y0 - y1 * x0) / line_len
    return k_range[int(np.argmax(dists))]


class MontiPaperConsensus(AbstractConsensus):
    name = "monti2"

    def __init__(
        self,
        base_algorithm: Callable[[np.ndarray], np.ndarray],
        n_resamples: int = 100,
        p_sample: float = 0.8,
        k_range: Tuple[int, int] = (2, 8),
        n_clusters: Optional[int] = None,
        random_state: Optional[int] = None,
        noise_label: int = -1,
    ) -> None:
        if not callable(base_algorithm):
            raise TypeError("base_algorithm must be a callable Z -> labels")
        self.base_algorithm = base_algorithm
        self.n_resamples = int(n_resamples)
        self.p_sample = float(p_sample)
        self.k_range = k_range
        self.n_clusters = n_clusters
        self.noise_label = noise_label
        self.rng = np.random.default_rng(random_state)

        self.labels_: Optional[np.ndarray] = None
        self.coassoc_matrix_: Optional[np.ndarray] = None
        self.optimal_k_: Optional[int] = None
        self.k_selection_method_: str = ""
        self.auc_curve_: Dict[int, float] = {}

    def fit(self, X: np.ndarray) -> "MontiPaperConsensus":
        n = X.shape[0]
        conn_list: List[np.ndarray] = []
        ident_list: List[np.ndarray] = []

        for _ in range(self.n_resamples):
            idx, X_sub = _resample(X, self.p_sample, self.rng)
            try:
                sub_labels = np.asarray(self.base_algorithm(X_sub), dtype=int)
            except Exception:
                continue

            sub_labels = relabel_subsample_unique_noise(sub_labels, self.noise_label)

            full_labels = np.full(n, -1, dtype=int)
            full_labels[idx] = sub_labels

            conn_list.append(compute_connectivity_matrix(full_labels))
            ident_list.append(compute_identity_matrix(n, idx))

        if not conn_list:
            self.labels_ = np.zeros(n, dtype=int)
            self.coassoc_matrix_ = np.eye(n)
            self.optimal_k_ = 1
            self.k_selection_method_ = "fallback_empty"
            return self

        C_alg = compute_consensus_matrix(conn_list, ident_list)
        np.fill_diagonal(C_alg, 1.0)
        self.coassoc_matrix_ = C_alg

        if self.n_clusters is not None:
            self.optimal_k_ = int(self.n_clusters)
            self.k_selection_method_ = "manual"
        else:
            k_min = max(2, int(self.k_range[0]))
            k_max = min(int(self.k_range[1]), n - 1)
            ks = list(range(k_min, k_max + 1))
            aucs = []
            for k in ks:
                lbl_k = _cut_consensus(C_alg, k)
                aucs.append(_area_under_cdf_for_partition(C_alg, lbl_k))
            self.auc_curve_ = dict(zip(ks, aucs))
            self.optimal_k_ = _best_k_knee(ks, aucs)
            self.k_selection_method_ = "knee_cdf_partition"

        self.labels_ = _cut_consensus(C_alg, int(self.optimal_k_))
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


__all__ = [
    "MontiPaperConsensus",
    "compute_connectivity_matrix",
    "compute_identity_matrix",
    "compute_consensus_matrix",
    "relabel_subsample_unique_noise",
]
