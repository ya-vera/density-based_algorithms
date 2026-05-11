from typing import Optional

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .base import AbstractDensityAlgorithm, AlgorithmRegistry
from .density_params import effective_min_cluster_size, effective_min_samples

import hdbscan

@AlgorithmRegistry.register("hdbscan")
class HDBSCANWrapper(AbstractDensityAlgorithm):

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,  
        cluster_selection_method: str = "eom",
        metric: str = "euclidean",  
        **kwargs
    ):
        super().__init__(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            metric=metric,
            **kwargs
        )
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self.model: hdbscan.HDBSCAN | None = None

        if hdbscan is None:
            raise ImportError("Для использования HDBSCANWrapper требуется пакет hdbscan.")

    def fit(self, X: np.ndarray) -> 'HDBSCANWrapper':
        if X.ndim != 2:
            raise ValueError(f"Ожидалась матрица 2D, получено {X.shape}")

        X_scaled = MinMaxScaler().fit_transform(X)

        n = X_scaled.shape[0]
        mcs = effective_min_cluster_size(self.min_cluster_size, n)
        base_ms = self.min_samples if self.min_samples is not None else max(2, mcs // 2)
        min_s = effective_min_samples(base_ms, n, default=max(2, mcs // 2))

        self.model = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=min_s,
            cluster_selection_method=self.cluster_selection_method,
            metric=self.metric,
            core_dist_n_jobs=-1,
            **{k: v for k, v in self.params.items()
               if k not in ["min_cluster_size", "min_samples", "cluster_selection_method", "metric"]}
        )
        self.model.fit(X_scaled)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.model.labels_ if self.model is not None else np.array([])

    def get_params(self) -> dict:
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "cluster_selection_method": self.cluster_selection_method,
            "metric": self.metric,
            **self.params
        }

    def __repr__(self):
        return (f"HDBSCANWrapper(min_cluster_size={self.min_cluster_size}, "
                f"min_samples={self.min_samples}, "
                f"cluster_selection_method='{self.cluster_selection_method}')")


__all__ = ["HDBSCANWrapper"]