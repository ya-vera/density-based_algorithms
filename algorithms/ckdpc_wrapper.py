from collections import deque
from heapq import heappush, heappop
from typing import List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from .base import AbstractDensityAlgorithm, AlgorithmRegistry


@AlgorithmRegistry.register("ckdpc")
class CKDPCWrapper(AbstractDensityAlgorithm):

    def __init__(
        self,
        alpha: float = 0.5,
        percent: float = 2.0,
        dc: Optional[float] = None,
        k_neighbors: int = 7,
        metric: str = "euclidean",
        n_clusters: Optional[int] = None,
        max_n_clusters: Optional[int] = None,
        density_threshold: float = 0.4,
        keep_noise: bool = True,
        **kwargs,
    ):
        super().__init__(
            alpha=alpha,
            percent=percent,
            dc=dc,
            k_neighbors=k_neighbors,
            metric=metric,
            n_clusters=n_clusters,
            max_n_clusters=max_n_clusters,
            density_threshold=density_threshold,
            keep_noise=keep_noise,
            **kwargs,
        )
        self.alpha = float(alpha)
        self.percent = float(percent)
        self.dc = dc
        self.k_neighbors = int(k_neighbors)
        self.metric = metric
        self.n_clusters = n_clusters
        self.max_n_clusters = max_n_clusters
        self.density_threshold = float(density_threshold)
        self.keep_noise = bool(keep_noise)

        self.labels_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None
        self.rho_: Optional[np.ndarray] = None
        self.delta_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None
        self.nearest_higher_: Optional[np.ndarray] = None
        self.sim_: Optional[np.ndarray] = None
        self.dc_used_: Optional[float] = None

    def _compute_sim(self, X: np.ndarray) -> np.ndarray:
        n, m = X.shape
        Xi = X[:, np.newaxis, :]
        Xj = X[np.newaxis, :, :]
        abs_diff = np.abs(Xi - Xj)
        mean_val = np.abs(Xi + Xj) / 2.0
        tol = self.alpha * mean_val
        agree = (abs_diff <= tol).astype(np.float32)
        return agree.mean(axis=2)

    def _compute_rho_adaptive(self, dist: np.ndarray, sim: np.ndarray, dc: float) -> np.ndarray:
        bandwidth = dc * (sim + 1.0)
        bandwidth = np.where(bandwidth < 1e-12, 1e-12, bandwidth)
        with np.errstate(over="ignore"):
            contrib = np.exp(-((dist / bandwidth) ** 2))
        rho = contrib.sum(axis=1) - 1.0
        return rho

    def _detect_noise(self, dist: np.ndarray, k: int) -> np.ndarray:
        n = dist.shape[0]
        k_eff = min(k, n - 1)
        d = dist.copy()
        np.fill_diagonal(d, np.inf)
        part = np.argpartition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
        rows = np.arange(n)[:, None]
        knn_dists = d[rows, part]
        max_knn = knn_dists.max(axis=1)
        threshold = float(max_knn.mean() + 1.5 * max_knn.std())
        return max_knn > threshold

    def _knn_indices(self, dist: np.ndarray, k: int) -> np.ndarray:
        return self._knn_from_dist(dist, k)[0]

    @staticmethod
    def _build_reverse_knn(knn: np.ndarray, n: int) -> List[List[int]]:
        rev_knn: List[List[int]] = [[] for _ in range(n)]
        for q in range(n):
            for m in knn[q]:
                rev_knn[m].append(q)
        return rev_knn

    def _two_stage_assign(self, labels: np.ndarray, rho: np.ndarray, sim: np.ndarray,
                          knn: np.ndarray, is_noise: np.ndarray, centers: np.ndarray,
                          rho_centers: np.ndarray) -> np.ndarray:
        n = labels.shape[0]

        for k_idx, center in enumerate(centers):
            center = int(center)
            rho_c = float(rho_centers[k_idx])
            queue = deque([center])
            while queue:
                p = queue.popleft()
                for r in knn[p]:
                    r = int(r)
                    if (labels[r] == -1 and not is_noise[r] and
                        float(rho[r]) > self.density_threshold * rho_c):
                        labels[r] = labels[center]
                        queue.append(r)

        rows = np.arange(n)[:, None]
        denom = sim[rows, knn].sum(axis=1) + 1.0

        rev_knn = self._build_reverse_knn(knn, n)

        best_score = np.full(n, -1.0)
        best_label = np.full(n, -1, dtype=int)

        def _refresh(i: int):
            bs, bl = -1.0, -1
            for j in knn[i]:
                j = int(j)
                if labels[j] != -1:
                    s = sim[i, j] / denom[i]
                    if s > bs:
                        bs, bl = s, labels[j]
            best_score[i] = bs
            best_label[i] = bl

        heap: List[Tuple[float, int]] = []
        for i in range(n):
            if labels[i] == -1:
                _refresh(i)
                if best_label[i] != -1:
                    heappush(heap, (-best_score[i], i))

        while heap:
            neg_s, m = heappop(heap)
            m = int(m)
            if labels[m] != -1:
                continue
            if abs(-neg_s - best_score[m]) > 1e-12:
                continue

            labels[m] = best_label[m]

            for q in rev_knn[m]:
                q = int(q)
                if labels[q] != -1:
                    continue
                s = float(sim[q, m]) / float(denom[q])
                if s > best_score[q]:
                    best_score[q] = s
                    best_label[q] = labels[m]
                    heappush(heap, (-s, q))

        return labels

    def fit(self, X: np.ndarray) -> "CKDPCWrapper":
        if X.ndim != 2:
            raise ValueError(f"Ожидалась матрица 2D, получено {X.shape}")
        n = X.shape[0]
        if n < 3:
            raise ValueError(f"Недостаточно точек: {n}")

        X_scaled = MinMaxScaler().fit_transform(X)

        dist = self._pairwise(X_scaled, self.metric)
        sim = self._compute_sim(X_scaled)
        self.sim_ = sim

        dc = self._choose_dc(dist, self.percent, self.dc)
        self.dc_used_ = dc

        rho = self._compute_rho_adaptive(dist, sim, dc)
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

        k_eff = max(3, min(self.k_neighbors, n - 1))
        is_noise = self._detect_noise(dist, k_eff)

        knn = self._knn_indices(dist, k_eff)

        labels = np.full(n, -1, dtype=int)
        for k_idx, c in enumerate(self.centers_):
            labels[int(c)] = k_idx

        rho_centers = rho[self.centers_]
        labels = self._two_stage_assign(
            labels, rho, sim, knn, is_noise, self.centers_, rho_centers
        )

        unassigned = np.where(labels == -1)[0]
        if unassigned.size:
            centers_arr = np.asarray(self.centers_, dtype=int)
            for i in unassigned:
                c = int(centers_arr[int(np.argmin(dist[i, centers_arr]))])
                labels[i] = labels[c]

        if self.keep_noise:
            labels[is_noise] = -1

        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_ if self.labels_ is not None else np.array([])

    def get_params(self) -> dict:
        return {
            "alpha": self.alpha,
            "percent": self.percent,
            "dc": self.dc,
            "dc_used": self.dc_used_,
            "k_neighbors": self.k_neighbors,
            "metric": self.metric,
            "n_clusters": self.n_clusters,
            "max_n_clusters": self.max_n_clusters,
            "density_threshold": self.density_threshold,
            "keep_noise": self.keep_noise,
            **self.params,
        }

    def __repr__(self) -> str:
        return f"CKDPCWrapper(alpha={self.alpha}, percent={self.percent}, k_neighbors={self.k_neighbors})"


__all__ = ["CKDPCWrapper"]