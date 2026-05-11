from __future__ import annotations

import numpy as np
from typing import Optional


def effective_min_samples(requested: Optional[int], n_points: int, default: int = 5) -> int:
    if n_points < 2:
        return 2
    req = int(requested) if requested is not None else default
    cap = max(2, n_points // 2)
    return int(max(2, min(req, cap)))


def effective_min_cluster_size(requested: int, n_points: int) -> int:
    return effective_min_samples(requested, n_points)


def auto_eps_from_knn(X: np.ndarray, k: int) -> float:
    from sklearn.neighbors import NearestNeighbors

    n = X.shape[0]
    k_eff = max(1, min(k, n - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
    dists, _ = nbrs.kneighbors(X)
    knn_dists = np.sort(dists[:, k_eff])[::-1]

    d_min, d_max = knn_dists[-1], knn_dists[0]
    if d_max - d_min < 1e-12:
        return float(d_max)

    n_pts = len(knn_dists)
    x = np.linspace(0.0, 1.0, n_pts)
    y = (knn_dists - d_min) / (d_max - d_min)

    dist_from_diagonal = np.abs(x + y - 1.0)
    knee_idx = int(np.argmax(dist_from_diagonal))
    return float(knn_dists[knee_idx])


def auto_eps_from_distances(
    X: np.ndarray,
    metric: str,
    percentile: float = 12.0,
) -> float:
    from scipy.spatial.distance import pdist

    d = pdist(X, metric=metric)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.5
    return float(np.clip(np.percentile(d, percentile), 1e-9, 1e6))
