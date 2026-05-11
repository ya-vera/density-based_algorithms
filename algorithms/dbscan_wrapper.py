from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from .base import AbstractDensityAlgorithm, AlgorithmRegistry
from .density_params import auto_eps_from_knn, effective_min_samples


def _search_valid_eps(
    X_scaled: np.ndarray,
    min_samples: int,
    eps_knee: float,
    metric: str,
) -> float:
    n = X_scaled.shape[0]
    cap = max(2, int(np.sqrt(n)))
    fracs = [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    best_k2 = None
    for frac in fracs:
        eps = eps_knee * frac
        if eps < 1e-9:
            continue
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=1).fit_predict(X_scaled)
        mask = labels != -1
        k = len(set(labels[mask].tolist())) if mask.any() else 0
        noise = 1.0 - mask.mean()
        if 2 <= k <= cap and noise <= 0.5:
            return eps
        if k >= 2 and best_k2 is None:
            best_k2 = eps

    return best_k2 if best_k2 is not None else eps_knee


@AlgorithmRegistry.register("dbscan")
class DBSCANWrapper(AbstractDensityAlgorithm):

    def __init__(
        self,
        eps: Optional[float] = None,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
        leaf_size: int = 30,
        **kwargs
    ):
        super().__init__(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            leaf_size=leaf_size,
            **kwargs
        )
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.model: DBSCAN | None = None
        self.eps_used_: Optional[float] = None

    def fit(self, X: np.ndarray) -> "DBSCANWrapper":
        if X.ndim != 2:
            raise ValueError(f"Ожидалась матрица 2D, получено {X.shape}")

        X_scaled = MinMaxScaler().fit_transform(X)
        n = X_scaled.shape[0]
        ms = effective_min_samples(self.min_samples, n)
        if self.eps is None:
            eps_knee = auto_eps_from_knn(X_scaled, ms)
            self.eps_used_ = _search_valid_eps(X_scaled, ms, eps_knee, self.metric)
        else:
            self.eps_used_ = float(self.eps)

        self.model = DBSCAN(
            eps=self.eps_used_,
            min_samples=ms,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=1,
            **{
                k: v
                for k, v in self.params.items()
                if k
                not in ["eps", "min_samples", "metric", "algorithm", "leaf_size"]
            },
        )
        self.model.fit(X_scaled)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.model.labels_ if self.model is not None else np.array([])

    def get_params(self) -> dict:
        return {
            "eps": self.eps,
            "eps_used": self.eps_used_,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "algorithm": self.algorithm,
            "leaf_size": self.leaf_size,
            **self.params,
        }

    def __repr__(self):
        return (
            f"DBSCANWrapper(eps={self.eps}, eps_used={self.eps_used_}, "
            f"min_samples={self.min_samples}, metric='{self.metric}')"
        )

__all__ = ["DBSCANWrapper"]