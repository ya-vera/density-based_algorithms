from typing import Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .base import AbstractDensityAlgorithm, AlgorithmRegistry


@AlgorithmRegistry.register("rd_dac")
class RDDACWrapper(AbstractDensityAlgorithm):
    def __init__(
        self,
        k: int = 7,
        metric: str = "euclidean",
        use_relative: bool = True,
        n_clusters: Optional[int] = None,
        max_n_clusters: Optional[int] = None,
        min_k: int = 3,
        **kwargs,
    ):
        super().__init__(
            k=k, metric=metric, use_relative=use_relative,
            n_clusters=n_clusters, max_n_clusters=max_n_clusters,
            min_k=min_k, **kwargs
        )
        self.k = int(k)
        self.metric = metric
        self.use_relative = bool(use_relative)
        self.n_clusters = n_clusters
        self.max_n_clusters = max_n_clusters
        self.min_k = int(min_k)

        self.labels_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None
        self.rho_: Optional[np.ndarray] = None
        self.delta_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None
        self.nearest_higher_: Optional[np.ndarray] = None
        self.k_used_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "RDDACWrapper":
        if X.ndim != 2:
            raise ValueError(f"Ожидалась матрица 2D, получено {X.shape}")

        n = X.shape[0]
        if n < 3:
            raise ValueError(f"Недостаточно точек: {n}")

        X_scaled = MinMaxScaler().fit_transform(X)

        k_eff = max(self.min_k, min(self.k, n - 1))
        self.k_used_ = k_eff

        dist = self._pairwise(X_scaled, metric=self.metric)
        knn_idx, knn_dist = self._knn_from_dist(dist, k_eff)

        eps = 1e-12
        sums = knn_dist.sum(axis=1)
        rho_knn = self.k_used_ / (sums + eps)

        if self.use_relative:
            mean_neigh = rho_knn[knn_idx].mean(axis=1)
            rho = rho_knn / (mean_neigh + eps)
        else:
            rho = rho_knn

        delta, nearest_higher, order = self._compute_delta(dist, rho)
        gamma = rho * delta

        self.rho_ = rho
        self.delta_ = delta
        self.gamma_ = gamma
        self.nearest_higher_ = nearest_higher

        sorted_idx = np.argsort(-gamma, kind="stable")
        sorted_gamma = gamma[sorted_idx]
        n_clusters = self._select_n_clusters(sorted_gamma, n)
        self.centers_ = sorted_idx[:n_clusters]

        labels = np.full(n, -1, dtype=int)
        for kk, c in enumerate(self.centers_):
            labels[int(c)] = kk

        is_center = np.zeros(n, dtype=bool)
        is_center[self.centers_] = True

        for i in order:
            i = int(i)
            if is_center[i] or labels[i] != -1:
                continue
            j = int(nearest_higher[i])
            if j >= 0 and labels[j] != -1:
                labels[i] = labels[j]

        unassigned = np.where(labels == -1)[0]
        if unassigned.size:
            centers_arr = np.asarray(self.centers_, dtype=int)
            for i in unassigned:
                c = int(centers_arr[int(np.argmin(dist[i, centers_arr]))])
                labels[i] = labels[c]

        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_ if self.labels_ is not None else np.array([])

    def get_params(self) -> dict:
        return {
            "k": self.k,
            "k_used": self.k_used_,
            "metric": self.metric,
            "use_relative": self.use_relative,
            "n_clusters": self.n_clusters,
            "max_n_clusters": self.max_n_clusters,
            "min_k": self.min_k,
            **self.params,
        }

    def __repr__(self) -> str:
        return f"RDDACWrapper(k={self.k}, metric='{self.metric}', use_relative={self.use_relative})"


__all__ = ["RDDACWrapper"]