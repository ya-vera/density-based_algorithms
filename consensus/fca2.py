from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .base import AbstractConsensus


def _build_context(
    label_matrix: np.ndarray,
    noise_label: int = -1,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    n_runs, n_objects = label_matrix.shape
    attrs: List[Tuple[int, int]] = []
    for t in range(n_runs):
        row = label_matrix[t]
        for c in sorted(set(row[row != noise_label].tolist())):
            attrs.append((t, int(c)))

    n_attrs = len(attrs)
    context = np.zeros((n_objects, n_attrs), dtype=bool)
    for col, (t, c) in enumerate(attrs):
        context[:, col] = label_matrix[t] == c

    return context, attrs


def _intent_run_support(intent_mask: np.ndarray, attrs: List[Tuple[int, int]]) -> int:
    runs: Set[int] = set()
    for col, (t, _) in enumerate(attrs):
        if intent_mask[col]:
            runs.add(t)
    return len(runs)


def _derive_intent(context: np.ndarray, extent: FrozenSet[int]) -> np.ndarray:
    if not extent:
        return np.ones(context.shape[1], dtype=bool)
    idx = list(extent)
    return np.all(context[idx, :], axis=0)


def _derive_extent(context: np.ndarray, intent_mask: np.ndarray) -> FrozenSet[int]:
    if not intent_mask.any():
        return frozenset(range(context.shape[0]))
    return frozenset(int(i) for i in np.where(np.all(context[:, intent_mask], axis=1))[0])


def _cbo_process(
    context: np.ndarray,
    attrs: List[Tuple[int, int]],
    min_support: int,
    min_extent_size: int,
    extent: FrozenSet[int],
    intent_mask: np.ndarray,
    start_attr: int,
    concepts: List[Dict],
    seen_extents: Set[FrozenSet[int]],
    n_objects: int,
) -> None:
    support = _intent_run_support(intent_mask, attrs)
    if support >= min_support and len(extent) >= min_extent_size:
        if extent not in seen_extents:
            seen_extents.add(extent)
            concepts.append({
                "extent": extent,
                "intent_mask": intent_mask.copy(),
                "support": support,
                "size": len(extent),
            })

    n_attrs = context.shape[1]
    for j in range(start_attr, n_attrs):
        if intent_mask[j]:
            continue

        new_intent = intent_mask.copy()
        new_intent[j] = True
        new_extent = _derive_extent(context, new_intent)
        closed_intent = _derive_intent(context, new_extent)

        skip = False
        for jj in range(j):
            if closed_intent[jj] and not intent_mask[jj]:
                skip = True
                break
        if skip:
            continue

        if new_extent not in seen_extents:
            _cbo_process(
                context, attrs, min_support, min_extent_size,
                new_extent, closed_intent, j + 1,
                concepts, seen_extents, n_objects,
            )


class FCAConsensus(AbstractConsensus):

    name = "fca"

    def __init__(
        self,
        base_algorithms: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
        min_support_frac: float = 0.5,
        min_extent_size: int = 1,
        greedy_order: str = "size_desc",
        use_caspailleur: bool = False,
        noise_label: int = -1,
    ) -> None:
        self.base_algorithms = base_algorithms or []
        self.min_support_frac = float(min_support_frac)
        self.min_extent_size = int(min_extent_size)
        self.greedy_order = greedy_order
        self.use_caspailleur = use_caspailleur
        self.noise_label = noise_label

        self.labels_: Optional[np.ndarray] = None
        self.antichain_: Optional[List[Dict]] = None
        self.context_: Optional[np.ndarray] = None
        self.attrs_: Optional[List[Tuple[int, int]]] = None
        self.label_matrix_: Optional[np.ndarray] = None
        self.n_runs_: int = 0

    def fit_from_labels(self, label_matrix: np.ndarray) -> "FCAConsensus":
        self.label_matrix_ = label_matrix
        self.n_runs_ = label_matrix.shape[0]
        n_objects = label_matrix.shape[1]
        min_support = max(1, math.ceil(self.min_support_frac * self.n_runs_))

        context, attrs = _build_context(label_matrix, noise_label=self.noise_label)
        self.context_ = context
        self.attrs_ = attrs

        if self.use_caspailleur:
            candidates = self._mine_concepts_caspailleur(
                context, attrs, min_support, n_objects
            )
        else:
            candidates = self._mine_concepts_cbo(
                context, attrs, min_support, n_objects
            )

        if self.greedy_order == "support_desc":
            candidates.sort(key=lambda c: (-c["support"], -c["size"]))
        elif self.greedy_order == "natural":
            pass
        else:
            candidates.sort(key=lambda c: (-c["size"], -c["support"]))

        antichain, covered_mask = self._extract_antichain(candidates, n_objects)
        self.antichain_ = antichain

        labels = self._assign_labels(antichain, covered_mask, n_objects, context)
        self.labels_ = labels
        return self

    def fit(self, X: np.ndarray) -> "FCAConsensus":
        if not self.base_algorithms:
            raise ValueError(
                "base_algorithms must be provided when calling fit(X). "
                "Alternatively, call fit_from_labels(label_matrix) directly."
            )
        rows = []
        for alg in self.base_algorithms:
            try:
                rows.append(np.asarray(alg(X), dtype=int))
            except Exception as e:
                warnings.warn(f"FCAConsensus: base algorithm failed: {e}")
        if not rows:
            self.labels_ = np.zeros(X.shape[0], dtype=int)
            return self
        return self.fit_from_labels(np.vstack(rows))

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def _mine_concepts_cbo(
        self,
        context: np.ndarray,
        attrs: List[Tuple[int, int]],
        min_support: int,
        n_objects: int,
    ) -> List[Dict]:
        concepts: List[Dict] = []
        seen_extents: Set[FrozenSet[int]] = set()

        full_extent = frozenset(range(n_objects))
        full_intent = _derive_intent(context, full_extent)

        _cbo_process(
            context=context,
            attrs=attrs,
            min_support=min_support,
            min_extent_size=self.min_extent_size,
            extent=full_extent,
            intent_mask=full_intent,
            start_attr=0,
            concepts=concepts,
            seen_extents=seen_extents,
            n_objects=n_objects,
        )

        return concepts

    def _mine_concepts_caspailleur(
        self,
        context: np.ndarray,
        attrs: List[Tuple[int, int]],
        min_support: int,
        n_objects: int,
    ) -> List[Dict]:
        try:
            import pandas as pd
            import caspailleur as csp

            col_names = [f"r{t}_c{c}" for (t, c) in attrs]
            df = pd.DataFrame(context, columns=col_names)
            df.index = [f"o{i}" for i in range(n_objects)]

            concepts_df = csp.mine_concepts(
                df,
                min_support=min_support / n_objects,
                to_compute=["extent", "intent"],
            )

            candidates: List[Dict] = []
            for _, row in concepts_df.iterrows():
                obj_names = row["extent"]
                extent_set = frozenset(int(fn[1:]) for fn in obj_names)

                intent_names: set = row["intent"]
                intent_mask = np.array(
                    [name in intent_names for name in col_names], dtype=bool
                )

                support = _intent_run_support(intent_mask, attrs)
                size = len(extent_set)

                if support >= min_support and size >= self.min_extent_size:
                    candidates.append(
                        {
                            "extent": extent_set,
                            "intent_mask": intent_mask,
                            "support": support,
                            "size": size,
                        }
                    )
            return candidates

        except Exception as e:
            warnings.warn(
                f"caspailleur failed ({e}); falling back to CbO approach."
            )
            return self._mine_concepts_cbo(context, attrs, min_support, n_objects)

    def _extract_antichain(
        self,
        candidates: List[Dict],
        n_objects: int,
    ) -> Tuple[List[Dict], np.ndarray]:
        covered_mask = np.zeros(n_objects, dtype=bool)
        antichain: List[Dict] = []

        for concept in candidates:
            extent = concept["extent"]
            uncovered = frozenset(obj for obj in extent if not covered_mask[obj])
            if not uncovered:
                continue
            if len(uncovered) < self.min_extent_size:
                continue

            antichain.append(
                {
                    "extent": uncovered,
                    "support": concept["support"],
                    "intent_mask": concept["intent_mask"],
                    "size": len(uncovered),
                }
            )
            for f in uncovered:
                covered_mask[f] = True

        return antichain, covered_mask

    def _assign_labels(
        self,
        antichain: List[Dict],
        covered_mask: np.ndarray,
        n_objects: int,
        context: np.ndarray,
    ) -> np.ndarray:
        labels = np.full(n_objects, -1, dtype=int)

        for cluster_id, concept in enumerate(antichain):
            for f in concept["extent"]:
                labels[f] = cluster_id

        uncovered = np.where(labels == -1)[0]
        if uncovered.size > 0:
            if not antichain:
                labels[:] = 0
                return labels

            concept_intent = np.vstack(
                [c["intent_mask"].astype(float) for c in antichain]
            )

            for i in uncovered:
                row = context[i].astype(float)
                overlaps = concept_intent @ row
                labels[i] = int(np.argmax(overlaps))

        return labels

    def describe_antichain(self) -> str:
        if self.antichain_ is None:
            return "FCAConsensus not yet fitted."
        lines = [f"FCA-Consensus antichain ({len(self.antichain_)} clusters):"]
        for k, c in enumerate(self.antichain_):
            objs = sorted(c["extent"])
            lines.append(
                f"  Cluster {k}: {len(objs)} objects, "
                f"support={c['support']}/{self.n_runs_} runs  "
                f"[objects: {objs[:8]}{'...' if len(objs) > 8 else ''}]"
            )
        return "\n".join(lines)


__all__ = ["FCAConsensus"]