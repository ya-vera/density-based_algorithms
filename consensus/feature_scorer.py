from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics import adjusted_rand_score


class RunQualityScorer:

    def __init__(
        self,
        label_matrix: np.ndarray,
        n_objects: int,
        noise_label: int = -1,
    ) -> None:
        self.label_matrix = np.asarray(label_matrix, dtype=int)
        self.n_objects = int(n_objects)
        self.noise_label = noise_label

    def pairwise_ari(self) -> float:
        n_runs = self.label_matrix.shape[0]
        if n_runs < 2:
            return 1.0
        aris: List[float] = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                a, b = self.label_matrix[i], self.label_matrix[j]
                mask = (a != self.noise_label) & (b != self.noise_label)
                if mask.sum() < 2:
                    continue
                aris.append(float(adjusted_rand_score(a[mask], b[mask])))
        return float(np.mean(aris)) if aris else 0.0


FeatureStabilityScorer = RunQualityScorer

__all__ = ["RunQualityScorer", "FeatureStabilityScorer"]
