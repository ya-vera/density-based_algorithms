from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.spatial.distance import pdist, squareform


class AbstractDensityAlgorithm(ABC):

    name: str = "unnamed_algorithm"

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'AbstractDensityAlgorithm':
        pass

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def get_params(self) -> Dict[str, Any]:
        return self.params

    @staticmethod
    def _pairwise(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
        dist = squareform(pdist(X, metric=metric))
        return np.clip(dist, 0.0, None)

    @staticmethod
    def _choose_dc(dist: np.ndarray, percent: float,
                   dc: Optional[float] = None) -> float:
        if dc is not None:
            return float(dc)
        iu = np.triu_indices(dist.shape[0], k=1)
        vals = dist[iu]
        return float(np.percentile(vals, percent)) if vals.size > 0 else 0.0

    @staticmethod
    def _compute_rho(dist: np.ndarray, dc: float,
                     use_gaussian: bool = True) -> np.ndarray:
        if use_gaussian:
            with np.errstate(over="ignore"):
                return np.sum(
                    np.exp(-((dist / max(dc, 1e-12)) ** 2)), axis=1
                ) - 1.0
        return (dist < dc).sum(axis=1).astype(float) - 1.0

    @staticmethod
    def _knn_from_dist(dist: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        n = dist.shape[0]
        d = dist.copy()
        np.fill_diagonal(d, np.inf)
        part = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        part_dist = d[rows, part]
        order = np.argsort(part_dist, axis=1, kind="stable")
        return part[rows, order], part_dist[rows, order]

    @staticmethod
    def _compute_delta(dist: np.ndarray,
                       rho: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = dist.shape[0]
        order = np.argsort(-rho, kind="stable")

        rank = np.empty(n, dtype=np.intp)
        rank[order] = np.arange(n)
        is_higher = rank[np.newaxis, :] < rank[:, np.newaxis]
        dist_masked = np.where(is_higher, dist, np.inf)

        nearest_higher = np.argmin(dist_masked, axis=1)
        delta = dist_masked[np.arange(n), nearest_higher]

        top = order[0]
        delta[top] = float(np.max(dist[top])) if dist.size else 0.0
        nearest_higher[top] = -1

        inf_mask = np.isinf(delta)
        inf_mask[top] = False
        if inf_mask.any():
            delta[inf_mask] = float(np.max(dist[top]))

        return delta, nearest_higher, order

    def _select_n_clusters(self, sorted_gamma: np.ndarray, n: int) -> int:
        if hasattr(self, 'n_clusters') and self.n_clusters is not None:
            return max(1, min(int(self.n_clusters), n))
        if sorted_gamma.size < 3:
            return max(1, sorted_gamma.size)

        cap = getattr(self, 'max_n_clusters', None) or max(2, int(np.sqrt(n)))
        search_n = min(cap + 1, sorted_gamma.size)
        sg = sorted_gamma[:search_n]
        ratios = sg[:-1] / (sg[1:] + 1e-12)

        sig = np.where(ratios >= 1.5)[0]
        if sig.size == 0:
            return 2

        k = int(sig[-1]) + 2
        return max(2, min(k, cap, n - 1))


class AlgorithmRegistry:
    _algorithms: Dict[str, type] = {}

    @classmethod
    def register(cls, name: Optional[str] = None):
        def decorator(algorithm_class: type):
            alg_name = name or algorithm_class.__name__.replace("Wrapper", "").lower()
            algorithm_class.name = alg_name
            cls._algorithms[alg_name] = algorithm_class
            return algorithm_class
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        name = name.lower()
        if name not in cls._algorithms:
            raise ValueError(f"Алгоритм '{name}' не найден. Доступны: {list(cls._algorithms.keys())}")
        return cls._algorithms[name]

    @classmethod
    def list_algorithms(cls) -> List[str]:
        return list(cls._algorithms.keys())


__all__ = ["AbstractDensityAlgorithm", "AlgorithmRegistry"]