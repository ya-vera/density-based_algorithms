import numpy as np
from sklearn.cluster import OPTICS
from sklearn.preprocessing import MinMaxScaler
from .base import AbstractDensityAlgorithm, AlgorithmRegistry
from .density_params import effective_min_samples


@AlgorithmRegistry.register("optics")
class OPTICSWrapper(AbstractDensityAlgorithm):

    def __init__(
        self,
        min_samples: int = 5,
        xi: float = 0.01,
        min_cluster_size: float = 0.01,
        cluster_method: str = "xi",
        metric: str = "euclidean",
        **kwargs
    ):
        super().__init__(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
            cluster_method=cluster_method,
            metric=metric,
            **kwargs
        )
        self.min_samples = min_samples
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.cluster_method = cluster_method
        self.metric = metric
        self.model: OPTICS | None = None

    def fit(self, X: np.ndarray) -> 'OPTICSWrapper':
        if X.ndim != 2:
            raise ValueError(f"Ожидалась матрица 2D, получено {X.shape}")

        X_scaled = MinMaxScaler().fit_transform(X)

        n = X_scaled.shape[0]
        ms = effective_min_samples(self.min_samples, n, default=5)

        self.model = OPTICS(
            min_samples=ms,
            xi=self.xi,
            min_cluster_size=self.min_cluster_size,
            cluster_method=self.cluster_method,
            metric=self.metric,
            n_jobs=1,
            **{k: v for k, v in self.params.items()
               if k not in ["min_samples", "xi", "min_cluster_size", "cluster_method", "metric"]}
        )
        self.model.fit(X_scaled)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.model.labels_ if self.model is not None else np.array([])

    def get_params(self) -> dict:
        return {
            "min_samples": self.min_samples,
            "xi": self.xi,
            "min_cluster_size": self.min_cluster_size,
            "cluster_method": self.cluster_method,
            "metric": self.metric,
            **self.params
        }

    def __repr__(self):
        return (f"OPTICSWrapper(min_samples={self.min_samples}, "
                f"xi={self.xi}, min_cluster_size={self.min_cluster_size}, "
                f"cluster_method='{self.cluster_method}')")


__all__ = ["OPTICSWrapper"]