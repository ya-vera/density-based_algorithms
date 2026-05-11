from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .base import AbstractDensityAlgorithm, AlgorithmRegistry


@AlgorithmRegistry.register("dpc")
class DPCWrapper(AbstractDensityAlgorithm):

    def __init__(
        self,
        percent: float = 2.0,
        dc: Optional[float] = None,
        use_gaussian: bool = True,
        metric: str = "euclidean",
        n_clusters: Optional[int] = None,
        max_n_clusters: Optional[int] = None,
        center_selection: str = "gamma_elbow",
        rho_pct: float = 85.0,
        delta_pct: float = 90.0,
        compute_halo: bool = False,
        **kwargs,
    ):
        super().__init__(
            percent=percent,
            dc=dc,
            use_gaussian=use_gaussian,
            metric=metric,
            n_clusters=n_clusters,
            max_n_clusters=max_n_clusters,
            center_selection=center_selection,
            rho_pct=rho_pct,
            delta_pct=delta_pct,
            compute_halo=compute_halo,
            **kwargs,
        )
        self.percent = float(percent)
        self.dc = dc
        self.use_gaussian = bool(use_gaussian)
        self.metric = metric
        self.n_clusters = n_clusters
        self.max_n_clusters = max_n_clusters
        self.center_selection = center_selection
        self.rho_pct = float(rho_pct)
        self.delta_pct = float(delta_pct)
        self.compute_halo = bool(compute_halo)

        self.labels_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None
        self.rho_: Optional[np.ndarray] = None
        self.delta_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None
        self.nearest_higher_: Optional[np.ndarray] = None
        self.dc_used_: Optional[float] = None

    def _select_centers_threshold(
        self, rho: np.ndarray, delta: np.ndarray
    ) -> np.ndarray:
        rho_thresh = float(np.percentile(rho, self.rho_pct))
        delta_thresh = float(np.percentile(delta, self.delta_pct))
        centers = np.where((rho > rho_thresh) & (delta > delta_thresh))[0]
        if centers.size == 0:
            centers = np.argsort(-(rho * delta))[:2]
        return centers

    def _compute_halo(
        self, dist: np.ndarray, labels: np.ndarray, rho: np.ndarray, dc: float
    ) -> np.ndarray:
        n = len(labels)
        n_clust = labels.max() + 1
        border_rho = np.full(n_clust, -np.inf)

        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != labels[j] and dist[i, j] < dc:
                    bval = max(rho[i], rho[j])
                    ci, cj = labels[i], labels[j]
                    if bval > border_rho[ci]:
                        border_rho[ci] = bval
                    if bval > border_rho[cj]:
                        border_rho[cj] = bval

        halo_labels = labels.copy()
        for i in range(n):
            c = labels[i]
            if c >= 0 and border_rho[c] > -np.inf and rho[i] < border_rho[c]:
                halo_labels[i] = -1
        return halo_labels

    def fit(self, X: np.ndarray) -> "DPCWrapper":
        if X.ndim != 2:
            raise ValueError(f"Ожидалась матрица 2D, получено {X.shape}")

        X_scaled = MinMaxScaler().fit_transform(X)

        dist = self._pairwise(X_scaled, self.metric)
        dc = self._choose_dc(dist, self.percent, self.dc)
        self.dc_used_ = dc

        rho = self._compute_rho(dist, dc, self.use_gaussian)
        delta, nearest_higher, order = self._compute_delta(dist, rho)
        gamma = rho * delta

        self.rho_ = rho
        self.delta_ = delta
        self.gamma_ = gamma
        self.nearest_higher_ = nearest_higher

        n = X_scaled.shape[0]

        if self.center_selection == "threshold":
            centers = self._select_centers_threshold(rho, delta)
        else:
            sorted_idx = np.argsort(-gamma, kind="stable")
            sorted_gamma = gamma[sorted_idx]
            n_clusters = self._select_n_clusters(sorted_gamma, n)
            centers = sorted_idx[:n_clusters]

        self.centers_ = centers

        is_center = np.zeros(n, dtype=bool)
        is_center[centers] = True

        labels = np.full(n, -1, dtype=int)
        for k, c in enumerate(centers):
            labels[int(c)] = k

        for i in order:
            i = int(i)
            if is_center[i] or labels[i] != -1:
                continue
            j = int(nearest_higher[i])
            if j >= 0 and labels[j] != -1:
                labels[i] = labels[j]

        unassigned = np.where(labels == -1)[0]
        if unassigned.size:
            centers_arr = np.asarray(centers, dtype=int)
            for i in unassigned:
                c = int(centers_arr[int(np.argmin(dist[i, centers_arr]))])
                labels[i] = labels[c]

        if self.compute_halo:
            labels = self._compute_halo(dist, labels, rho, dc)

        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_ if self.labels_ is not None else np.array([])

    def get_params(self) -> dict:
        return {
            "percent": self.percent,
            "dc": self.dc,
            "dc_used": self.dc_used_,
            "use_gaussian": self.use_gaussian,
            "metric": self.metric,
            "n_clusters": self.n_clusters,
            "max_n_clusters": self.max_n_clusters,
            "center_selection": self.center_selection,
            "rho_pct": self.rho_pct,
            "delta_pct": self.delta_pct,
            "compute_halo": self.compute_halo,
            **self.params,
        }

    def __repr__(self) -> str:
        return (
            f"DPCWrapper(percent={self.percent}, dc={self.dc}, "
            f"use_gaussian={self.use_gaussian}, metric='{self.metric}', "
            f"center_selection='{self.center_selection}')"
        )


__all__ = ["DPCWrapper"]