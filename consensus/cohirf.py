from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import adjusted_rand_score

from .base import AbstractConsensus


@dataclass
class CoHiRFNode:
    level: int
    n_in: int
    n_out: int
    cluster_map: Dict[int, List[int]] = field(default_factory=dict)


def _canonicalize(labels: np.ndarray) -> np.ndarray:
    out = np.empty_like(labels)
    mapping: Dict[int, int] = {}
    nxt = 0
    for i, v in enumerate(labels.tolist()):
        if v not in mapping:
            mapping[v] = nxt
            nxt += 1
        out[i] = mapping[v]
    return out


def _noise_split_before_canonicalize(
    raw: np.ndarray, noise_label: int = -1
) -> np.ndarray:
    """
    Density noise (-1) must not collapse into one cluster after _canonicalize.
    Assign each noise point a distinct id before remapping to 0..K-1.
    """
    raw = np.asarray(raw, dtype=int).copy()
    m = raw == noise_label
    if not m.any():
        return raw
    if m.all():
        raw[:] = np.arange(raw.shape[0], dtype=int)
        return raw
    base = int(raw[~m].max()) + 1
    k = 0
    for pos in np.where(m)[0]:
        raw[pos] = base + k
        k += 1
    return raw


def _get_clusters_strict(
    X_active: np.ndarray,
    base_algorithms: List[Callable],
    n_repetitions: int,
    q: int,
    p_orig: int,
    rng: np.random.Generator,
    noise_label: int = -1,
) -> Optional[np.ndarray]:
    n_active = X_active.shape[0]
    n_algs = len(base_algorithms)
    codes: List[np.ndarray] = []

    for r in range(n_repetitions):
        feat_idx = rng.choice(p_orig, size=q, replace=False)
        X_view = X_active[:, feat_idx]
        alg = base_algorithms[r % n_algs]
        try:
            raw = np.asarray(alg(X_view), dtype=int)
            raw = _noise_split_before_canonicalize(raw, noise_label)
            codes.append(_canonicalize(raw))
        except Exception:
            continue

    if not codes:
        return None

    code_matrix = np.column_stack(codes)

    groups: Dict[tuple, List[int]] = defaultdict(list)
    for i in range(n_active):
        key = tuple(code_matrix[i].tolist())
        groups[key].append(i)

    label_vec = np.empty(n_active, dtype=int)
    for cluster_id, members in enumerate(groups.values()):
        for m in members:
            label_vec[m] = cluster_id

    return label_vec


def _get_clusters_relaxed(
    X_active: np.ndarray,
    base_algorithms: List[Callable],
    n_repetitions: int,
    q: int,
    p_orig: int,
    rng: np.random.Generator,
    loo_threshold: float = 0.8,
    noise_label: int = -1,
) -> Optional[np.ndarray]:
    n_algs = len(base_algorithms)
    raw_partitions: List[np.ndarray] = []

    for r in range(n_repetitions):
        feat_idx = rng.choice(p_orig, size=q, replace=False)
        X_view = X_active[:, feat_idx]
        alg = base_algorithms[r % n_algs]
        try:
            raw = np.asarray(alg(X_view), dtype=int)
            raw = _noise_split_before_canonicalize(raw, noise_label)
            raw_partitions.append(_canonicalize(raw))
        except Exception:
            continue

    if not raw_partitions:
        return None

    if len(raw_partitions) == 1:
        label_vec = np.empty(X_active.shape[0], dtype=int)
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, v in enumerate(raw_partitions[0].tolist()):
            groups[v].append(i)
        for cid, members in enumerate(groups.values()):
            for m in members:
                label_vec[m] = cid
        return label_vec

    def _strict_consensus(partitions: List[np.ndarray]) -> np.ndarray:
        n = partitions[0].shape[0]
        code_matrix = np.column_stack(partitions)
        g: Dict[tuple, List[int]] = defaultdict(list)
        for i in range(n):
            g[tuple(code_matrix[i].tolist())].append(i)
        lv = np.empty(n, dtype=int)
        for cid, members in enumerate(g.values()):
            for m in members:
                lv[m] = cid
        return lv

    active = list(range(len(raw_partitions)))
    while len(active) > 1:
        current_parts = [raw_partitions[r] for r in active]
        consensus_full = _strict_consensus(current_parts)

        loo_scores = []
        for idx, r in enumerate(active):
            loo_parts = [raw_partitions[rr] for rr in active if rr != r]
            consensus_loo = _strict_consensus(loo_parts)
            try:
                score = adjusted_rand_score(consensus_full, consensus_loo)
            except Exception:
                score = 1.0
            loo_scores.append((score, idx))

        min_score, min_idx = min(loo_scores, key=lambda x: x[0])
        if min_score < loo_threshold:
            active.pop(min_idx)
        else:
            break

    final_parts = [raw_partitions[r] for r in active]
    return _strict_consensus(final_parts)


def _choose_medoids(
    X_orig: np.ndarray,
    active_indices: np.ndarray,
    label_vec: np.ndarray,
) -> Tuple[np.ndarray, List[List[int]]]:
    n_clusters = int(label_vec.max()) + 1
    new_active: List[int] = []
    groups_orig: List[List[int]] = []

    for k in range(n_clusters):
        members_local = np.where(label_vec == k)[0]
        members_global = active_indices[members_local]

        if len(members_local) == 1:
            medoid_local = 0
        else:
            X_group = X_orig[members_global]
            norms = np.linalg.norm(X_group, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            X_norm = X_group / norms
            gram = np.abs(X_norm @ X_norm.T)
            medoid_local = int(np.argmax(gram.sum(axis=1)))

        new_active.append(int(members_global[medoid_local]))
        groups_orig.append(members_global.tolist())

    return np.array(new_active, dtype=int), groups_orig


def _update_parents(
    parent: np.ndarray,
    label_vec: np.ndarray,
    active_indices: np.ndarray,
    new_active: np.ndarray,
) -> np.ndarray:
    n_clusters = int(label_vec.max()) + 1
    for k in range(n_clusters):
        members_local = np.where(label_vec == k)[0]
        members_global = active_indices[members_local]
        medoid_global = new_active[k]
        for m in members_global:
            if m != medoid_global:
                parent[m] = medoid_global
    return parent


def _get_final_labels(parent: np.ndarray) -> np.ndarray:
    n = len(parent)
    root = np.arange(n, dtype=int)
    for i in range(n):
        j = i
        while parent[j] != j:
            j = parent[j]
        root[i] = j

    unique_roots = np.unique(root)
    label_map = {r: idx for idx, r in enumerate(unique_roots)}
    return np.array([label_map[root[i]] for i in range(n)], dtype=int)


def _best_level_labels(
    hierarchy: List[CoHiRFNode],
    n_orig: int,
    min_k: int = 2,
) -> Tuple[np.ndarray, int]:
    for node in reversed(hierarchy):
        if node.n_out >= min_k:
            lbl = np.zeros(n_orig, dtype=int)
            for cid, members in node.cluster_map.items():
                for m in members:
                    lbl[m] = cid
            return lbl, node.n_out
    lbl = np.zeros(n_orig, dtype=int)
    return lbl, 1


class CoHiRFConsensus(AbstractConsensus):

    name = "cohirf"

    def __init__(
        self,
        base_algorithm: Union[
            Callable[[np.ndarray], np.ndarray],
            List[Callable[[np.ndarray], np.ndarray]],
            None,
        ] = None,
        n_repetitions: int = 5,
        n_q_features: Optional[int] = None,
        relaxed: bool = False,
        loo_threshold: float = 0.8,
        max_levels: int = 50,
        random_state: Optional[int] = None,
        noise_label: int = -1,
    ) -> None:
        if base_algorithm is None:
            self.base_algorithms: List[Callable] = []
        elif callable(base_algorithm):
            self.base_algorithms = [base_algorithm]
        else:
            self.base_algorithms = list(base_algorithm)

        self.n_repetitions = max(2, int(n_repetitions))
        self.n_q_features = n_q_features
        self.relaxed = bool(relaxed)
        self.loo_threshold = float(loo_threshold)
        self.max_levels = int(max_levels)
        self.noise_label = noise_label
        self.rng = np.random.default_rng(random_state)

        self.labels_: Optional[np.ndarray] = None
        self.hierarchy_: List[CoHiRFNode] = []
        self.pac_scores_: Dict[int, float] = {}
        self.optimal_k_: Optional[int] = None
        self.k_selection_method_: str = ""
        self.coassoc_matrix_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "CoHiRFConsensus":
        if not self.base_algorithms:
            raise ValueError(
                "base_algorithm must be provided when calling fit(X)."
            )

        n_orig, p_orig = X.shape

        q = self.n_q_features
        if q is None:
            q = max(2, min(p_orig, max(2, p_orig // 3)))
        q = int(q)
        if p_orig <= 1:
            q = 1
        elif p_orig == 2:
            q = 1
        else:
            q = max(2, min(q, p_orig - 1))

        parent = np.arange(n_orig, dtype=int)
        active_indices = np.arange(n_orig, dtype=int)
        self.hierarchy_ = []

        n_prev = 0
        n_active = n_orig

        for lvl in range(self.max_levels):
            if n_active == n_prev or n_active <= 1:
                break
            n_prev = n_active

            X_active = X[active_indices]

            if self.relaxed:
                label_vec = _get_clusters_relaxed(
                    X_active=X_active,
                    base_algorithms=self.base_algorithms,
                    n_repetitions=self.n_repetitions,
                    q=q,
                    p_orig=p_orig,
                    rng=self.rng,
                    loo_threshold=self.loo_threshold,
                    noise_label=self.noise_label,
                )
            else:
                label_vec = _get_clusters_strict(
                    X_active=X_active,
                    base_algorithms=self.base_algorithms,
                    n_repetitions=self.n_repetitions,
                    q=q,
                    p_orig=p_orig,
                    rng=self.rng,
                    noise_label=self.noise_label,
                )

            if label_vec is None:
                break

            n_new = int(label_vec.max()) + 1

            if n_new >= n_active:
                break

            new_active, groups_orig = _choose_medoids(X, active_indices, label_vec)
            parent = _update_parents(parent, label_vec, active_indices, new_active)

            self.hierarchy_.append(CoHiRFNode(
                level=lvl,
                n_in=n_active,
                n_out=n_new,
                cluster_map={k: groups_orig[k] for k in range(n_new)},
            ))

            active_indices = new_active
            n_active = len(active_indices)

        final_labels = _get_final_labels(parent)
        final_k = int(final_labels.max()) + 1

        if final_k >= 2:
            self.labels_ = final_labels
            self.optimal_k_ = final_k
            self.k_selection_method_ = "cfh_convergence"
        elif self.hierarchy_:
            self.labels_, self.optimal_k_ = _best_level_labels(
                self.hierarchy_, n_orig, min_k=2
            )
            self.k_selection_method_ = "best_level_fallback"
        else:
            self.labels_ = np.zeros(n_orig, dtype=int)
            self.optimal_k_ = 1
            self.k_selection_method_ = "fallback_single"

        lbl = self.labels_
        coassoc = np.zeros((n_orig, n_orig), dtype=np.float64)
        for c in np.unique(lbl):
            members = np.where(lbl == c)[0]
            coassoc[np.ix_(members, members)] = 1.0
        np.fill_diagonal(coassoc, 1.0)
        self.coassoc_matrix_ = coassoc

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def get_labels_at_level(self, level: int) -> Optional[np.ndarray]:
        if level < 0 or level >= len(self.hierarchy_):
            return None
        node = self.hierarchy_[level]
        n = max(
            (max(idxs) for idxs in node.cluster_map.values() if idxs),
            default=-1,
        ) + 1
        if n == 0:
            return None
        lbl = np.full(n, -1, dtype=int)
        for gid, objs in node.cluster_map.items():
            for i in objs:
                if 0 <= i < n:
                    lbl[i] = gid
        return lbl


__all__ = ["CoHiRFConsensus", "CoHiRFNode"]