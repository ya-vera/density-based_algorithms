from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .base import AbstractDensityAlgorithm, AlgorithmRegistry


@AlgorithmRegistry.register("dac")
class DACWrapper(AbstractDensityAlgorithm):

    def __init__(
        self,
        metric: str = "euclidean",
        alpha_pct: float = 85.0,
        beta_pct: float = 75.0,
        n_clusters: Optional[int] = None,
        max_n_clusters: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            metric=metric,
            n_clusters=n_clusters,
            max_n_clusters=max_n_clusters,
            **kwargs,
        )
        self.metric = metric
        self.alpha_pct = float(alpha_pct)
        self.beta_pct = float(beta_pct)
        self.n_clusters = n_clusters
        self.max_n_clusters = max_n_clusters

        self.labels_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None
        self.rho_: Optional[np.ndarray] = None
        self.delta_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None
        self.d_: Optional[float] = None

    def _data_bound_distance(self, dist: np.ndarray) -> float:
        n = dist.shape[0]
        nn_dist = np.array([
            np.min(dist[i, np.arange(n) != i]) for i in range(n)
        ])
        return float(np.mean(nn_dist))

    def _compute_rho_dac(self, dist: np.ndarray, d: float, theta: float = 2.3) -> np.ndarray:
        n = dist.shape[0]
        rho = np.zeros(n)
        for i in range(n):
            neighbours = np.arange(n) != i
            rho[i] = np.sum(np.exp(-dist[i, neighbours] / (d * theta + 1e-12)))
        return rho

    def _neighbourhood_search(
        self,
        dist: np.ndarray,
        centers: np.ndarray,
        rho: np.ndarray,
        d: float,
    ) -> np.ndarray:
        n = dist.shape[0]
        labels = np.full(n, -1, dtype=int)

        center_order = np.argsort(-rho[centers])
        ordered_centers = centers[center_order]

        for label_id, c in enumerate(ordered_centers):
            if labels[int(c)] != -1:
                continue
            labels[int(c)] = label_id
            queue = [int(c)]
            while queue:
                cur = queue.pop(0)
                for j in range(n):
                    if labels[j] == -1 and dist[cur, j] <= d:
                        labels[j] = label_id
                        queue.append(j)

        unassigned = np.where(labels == -1)[0]
        if unassigned.size:
            centers_arr = np.asarray(ordered_centers, dtype=int)
            for i in unassigned:
                c = int(centers_arr[int(np.argmin(dist[i, centers_arr]))])
                labels[i] = labels[c]

        return labels

    def fit(self, X: np.ndarray) -> "DACWrapper":
        if X.ndim != 2:
            raise ValueError(f"Ожидалась матрица 2D, получено {X.shape}")

        X_scaled = MinMaxScaler().fit_transform(X)
        dist = self._pairwise(X_scaled, self.metric)

        d = self._data_bound_distance(dist)
        self.d_ = d

        rho = self._compute_rho_dac(dist, d)
        delta, nearest_higher, _ = self._compute_delta(dist, rho)
        gamma = rho * delta

        self.rho_ = rho
        self.delta_ = delta
        self.gamma_ = gamma

        n = X_scaled.shape[0]
        alpha = float(np.percentile(delta, self.alpha_pct))
        beta = float(np.percentile(rho, self.beta_pct))
        centers = np.where((delta > alpha) & (rho > beta))[0]

        if centers.size == 0:
            sorted_idx = np.argsort(-gamma, kind="stable")
            centers = sorted_idx[:2]
        cap = getattr(self, 'max_n_clusters', None) or max(2, int(np.sqrt(n)))
        if centers.size > cap:
            centers = centers[np.argsort(-gamma[centers])[:cap]]
        self.centers_ = centers

        self.labels_ = self._neighbourhood_search(dist, centers, rho, d)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_ if self.labels_ is not None else np.array([])

    def get_params(self) -> dict:
        return {
            "metric": self.metric,
            "alpha_pct": self.alpha_pct,
            "beta_pct": self.beta_pct,
            "n_clusters": self.n_clusters,
            "max_n_clusters": self.max_n_clusters,
            "d_used": self.d_,
            **self.params,
        }

    def __repr__(self) -> str:
        return (
            f"DACWrapper(metric='{self.metric}', "
            f"alpha_pct={self.alpha_pct}, beta_pct={self.beta_pct})"
        )


__all__ = ["DACWrapper"]
