from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import (
    AbstractConsensus,
    align_labels,
    build_coassociation,
    coassoc_to_labels,
    compute_run_weights,
    select_k_adaptive,
)


class CoAssocEnsemble(AbstractConsensus):

    name = "coassoc_ensemble"

    def __init__(
        self,
        algorithms: List[Callable[[np.ndarray], np.ndarray]],
        k_range: Tuple[int, int] = (2, 8),
        n_clusters: Optional[int] = None,
        pac_lower: float = 0.1,
        pac_upper: float = 0.9,
        flat_pac_threshold: float = 0.02,
        noise_label: int = -1,
    ) -> None:
        self.algorithms = algorithms
        self.k_range = k_range
        self.n_clusters = n_clusters
        self.pac_lower = pac_lower
        self.pac_upper = pac_upper
        self.flat_pac_threshold = flat_pac_threshold
        self.noise_label = noise_label

        self.labels_: Optional[np.ndarray] = None
        self.coassoc_matrix_: Optional[np.ndarray] = None
        self.pac_scores_: Optional[Dict[int, float]] = None
        self.optimal_k_: Optional[int] = None
        self.k_selection_method_: Optional[str] = None
        self.label_matrix_: Optional[np.ndarray] = None
        self.run_weights_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "CoAssocEnsemble":
        n_objects = X.shape[0]
        label_rows: List[np.ndarray] = []

        for alg in self.algorithms:
            try:
                labels = alg(X)
                label_rows.append(np.asarray(labels, dtype=int))
            except Exception:
                pass

        if not label_rows:
            self.labels_ = np.zeros(n_objects, dtype=int)
            self.coassoc_matrix_ = np.eye(n_objects)
            return self

        self.label_matrix_ = np.vstack(label_rows)

        self.run_weights_ = compute_run_weights(
            self.label_matrix_, n_objects, noise_label=self.noise_label
        )
        C_norm, _ = build_coassociation(
            self.label_matrix_, n_objects,
            noise_label=self.noise_label,
            weights=self.run_weights_,
        )
        self.coassoc_matrix_ = C_norm

        if self.n_clusters is not None:
            self.optimal_k_ = int(self.n_clusters)
            self.pac_scores_ = {}
            self.k_selection_method_ = "manual"
        else:
            run_k_counts = [
                int(np.unique(row[row != self.noise_label]).size)
                for row in self.label_matrix_
            ]
            self.optimal_k_, self.pac_scores_, self.k_selection_method_ = (
                select_k_adaptive(
                    C_norm,
                    k_range=self.k_range,
                    lower=self.pac_lower,
                    upper=self.pac_upper,
                    run_k_counts=run_k_counts,
                    flat_pac_threshold=self.flat_pac_threshold,
                )
            )

        self.labels_ = coassoc_to_labels(C_norm, self.optimal_k_)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


class VotingEnsemble(AbstractConsensus):

    name = "voting_ensemble"

    def __init__(
        self,
        algorithms: List[Callable[[np.ndarray], np.ndarray]],
        n_clusters: Optional[int] = None,
        noise_label: int = -1,
    ) -> None:
        self.algorithms = algorithms
        self.n_clusters = n_clusters
        self.noise_label = noise_label

        self.labels_: Optional[np.ndarray] = None
        self.label_matrix_: Optional[np.ndarray] = None
        self.run_weights_: Optional[np.ndarray] = None
        self.agreement_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "VotingEnsemble":
        n_objects = X.shape[0]
        label_rows: List[np.ndarray] = []

        for alg in self.algorithms:
            try:
                labels = np.asarray(alg(X), dtype=int)
                label_rows.append(labels)
            except Exception:
                pass

        if not label_rows:
            self.labels_ = np.zeros(n_objects, dtype=int)
            return self

        self.label_matrix_ = np.vstack(label_rows)

        self.run_weights_ = compute_run_weights(
            self.label_matrix_, n_objects, noise_label=self.noise_label
        )

        sort_order = np.argsort(self.run_weights_)[::-1]
        sorted_matrix = self.label_matrix_[sort_order]
        sorted_weights = self.run_weights_[sort_order]

        ref = sorted_matrix[0]
        aligned: List[np.ndarray] = [ref.copy()]
        for row in sorted_matrix[1:]:
            aligned.append(align_labels(ref, row, self.noise_label))
        aligned_matrix = np.vstack(aligned)

        final_labels = np.full(n_objects, self.noise_label, dtype=int)
        agreement = np.zeros(n_objects)

        all_label_ids = np.unique(aligned_matrix[aligned_matrix != self.noise_label])

        for i in range(n_objects):
            votes = aligned_matrix[:, i]
            valid_mask = votes != self.noise_label
            if not valid_mask.any():
                continue
            weighted_counts: Dict[int, float] = {}
            for c in all_label_ids:
                w_sum = float(sorted_weights[valid_mask & (votes == c)].sum())
                if w_sum > 0:
                    weighted_counts[int(c)] = w_sum
            if not weighted_counts:
                continue
            best = max(weighted_counts, key=weighted_counts.__getitem__)
            final_labels[i] = best
            valid_weight = float(sorted_weights[valid_mask].sum())
            agreement[i] = weighted_counts[best] / valid_weight if valid_weight > 0 else 0.0

        self.labels_ = final_labels
        self.agreement_ = agreement
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


__all__ = ["CoAssocEnsemble", "VotingEnsemble"]
