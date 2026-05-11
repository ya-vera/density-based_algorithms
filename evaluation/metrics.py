from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


class ClusteringMetrics:

    def __init__(
        self,
        labels: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        noise_label: int = -1,
    ) -> None:
        self.labels = np.asarray(labels, dtype=int)
        self.y_true = np.asarray(y_true, dtype=int) if y_true is not None else None
        if self.y_true is not None and self.y_true.shape[0] != self.labels.shape[0]:
            self.y_true = None
        self.X = X
        self.noise_label = noise_label

        self._non_noise_mask = self.labels != noise_label

    # External metrics

    def external(self) -> Dict[str, float]:
        if self.y_true is None:
            return {}
        if np.all(self.y_true == self.noise_label):
            return {}

        mask = self._non_noise_mask & (self.y_true != self.noise_label)
        if mask.sum() < 2:
            return {}

        pred = self.labels[mask]
        true = self.y_true[mask]

        if len(np.unique(pred)) < 2:
            purity = self._purity(true, pred)
            return {
                "ari":       0.0,
                "nmi":       0.0,
                "v_measure": 0.0,
                "fmi":       0.0,
                "purity":    float(purity),
            }

        purity = self._purity(true, pred)
        return {
            "ari":       float(adjusted_rand_score(true, pred)),
            "nmi":       float(normalized_mutual_info_score(true, pred)),
            "v_measure": float(v_measure_score(true, pred)),
            "fmi":       float(fowlkes_mallows_score(true, pred)),
            "purity":    float(purity),
        }

    # Internal metrics

    def internal(self) -> Dict[str, float]:
        if self.X is None:
            return {}

        mask = self._non_noise_mask
        if mask.sum() < 2:
            return {}

        X_sub = self.X[mask]
        labels_sub = self.labels[mask]
        n_clusters = len(np.unique(labels_sub))

        if n_clusters < 2:
            return {}

        result: Dict[str, float] = {}
        try:
            result["silhouette"] = float(silhouette_score(X_sub, labels_sub))
        except Exception:
            result["silhouette"] = float("nan")
        try:
            result["davies_bouldin"] = float(davies_bouldin_score(X_sub, labels_sub))
        except Exception:
            result["davies_bouldin"] = float("nan")
        try:
            result["calinski_harabasz"] = float(calinski_harabasz_score(X_sub, labels_sub))
        except Exception:
            result["calinski_harabasz"] = float("nan")

        return result

    # Density/structure-specific metrics
   
    def structure(self) -> Dict[str, float]:
        n = len(self.labels)
        noise_mask = ~self._non_noise_mask
        noise_frac = float(noise_mask.sum() / n)

        non_noise = self.labels[self._non_noise_mask]
        unique_clusters = np.unique(non_noise) if non_noise.size > 0 else np.array([])
        n_clusters = int(unique_clusters.size)

        if n_clusters == 0:
            return {
                "noise_fraction": noise_frac,
                "n_clusters": 0,
                "largest_cluster_frac": 0.0,
                "smallest_cluster_size": 0,
                "mean_cluster_size": 0.0,
                "cluster_size_entropy": 0.0,
            }

        sizes = np.array([int(np.sum(self.labels == c)) for c in unique_clusters])
        probs = sizes / sizes.sum()
        entropy = float(-np.sum(probs * np.log2(probs + 1e-15)))

        return {
            "noise_fraction":       noise_frac,
            "n_clusters":           n_clusters,
            "largest_cluster_frac": float(sizes.max() / n),
            "smallest_cluster_size": int(sizes.min()),
            "mean_cluster_size":    float(sizes.mean()),
            "cluster_size_entropy": entropy,
        }

    # Combined report

    def all_metrics(self) -> Dict[str, Dict[str, float]]:
        return {
            "external": self.external(),
            "internal": self.internal(),
            "structure": self.structure(),
        }

    def summary_score(self) -> float:
        ext = self.external()
        if "ari" in ext:
            return max(0.0, float(ext["ari"]))
        inn = self.internal()
        if "silhouette" in inn and not np.isnan(inn["silhouette"]):
            return (float(inn["silhouette"]) + 1.0) / 2.0
        return 0.0

    @staticmethod
    def _purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        total = len(y_true)
        if total == 0:
            return 0.0
        correct = 0
        for c in np.unique(y_pred):
            mask = y_pred == c
            if mask.any():
                correct += int(np.bincount(y_true[mask]).max())
        return correct / total
