from __future__ import annotations

import math
import warnings
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .base import AbstractConsensus


def _ensure_skmine_sklearn_validate_shim() -> None:
    from sklearn import base as sklearn_base
    from sklearn.utils import validation as skl_val

    if hasattr(sklearn_base.BaseEstimator, "_validate_data"):
        return

    _KW_RENAME = {"force_all_finite": "ensure_all_finite"}

    def _validate_data(
        self,
        X,
        y=None,
        reset=True,
        validate_separately=False,
        skip_check_array=False,
        **check_params,
    ):
        for old, new in _KW_RENAME.items():
            if old in check_params and new not in check_params:
                check_params[new] = check_params.pop(old)
            elif old in check_params:
                check_params.pop(old)
        y_kw = "no_validation" if y is None else y
        return skl_val.validate_data(
            self,
            X=X,
            y=y_kw,
            reset=reset,
            validate_separately=validate_separately,
            skip_check_array=skip_check_array,
            **check_params,
        )

    sklearn_base.BaseEstimator._validate_data = _validate_data


def _build_context_df(
    label_matrix: np.ndarray,
    noise_label: int = -1,
):
    import pandas as pd

    n_runs, n_objects = label_matrix.shape
    col_names: List[str] = []
    col_arrays: List[np.ndarray] = []

    for t in range(n_runs):
        row = label_matrix[t]
        for c in sorted(set(row[row != noise_label].tolist())):
            col_names.append(f"r{t}_c{c}")
            col_arrays.append(row == c)

    if not col_arrays:
        return pd.DataFrame(
            np.zeros((n_objects, 1), dtype=bool),
            index=[f"o{i}" for i in range(n_objects)],
            columns=["r0_c0"],
        )

    data = np.column_stack(col_arrays)
    return pd.DataFrame(
        data,
        index=[f"o{i}" for i in range(n_objects)],
        columns=col_names,
    )


def _attr_run_support(intent_names: frozenset, n_runs: int) -> int:
    runs = set()
    for name in intent_names:
        t_str = name.split("_")[0][1:]
        runs.add(t_str)
    return len(runs)


class FCAConsensus(AbstractConsensus):
    name = "fca"

    def __init__(
        self,
        base_algorithms: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
        min_support_frac: float = 0.5,
        min_extent_size: int = 1,
        greedy_order: str = "size_desc",
        noise_label: int = -1,
        min_delta_stability: Optional[float] = None,
        n_stable_concepts: Optional[int] = None,
    ) -> None:
        self.base_algorithms = base_algorithms or []
        self.min_support_frac = float(min_support_frac)
        self.min_extent_size = int(min_extent_size)
        self.greedy_order = greedy_order
        self.noise_label = noise_label
        self.min_delta_stability = min_delta_stability
        self.n_stable_concepts = n_stable_concepts

        self.labels_: Optional[np.ndarray] = None
        self.antichain_: Optional[List[Dict]] = None
        self.context_df_ = None
        self.concepts_df_ = None
        self.label_matrix_: Optional[np.ndarray] = None
        self.n_runs_: int = 0

    def fit_from_labels(self, label_matrix: np.ndarray) -> "FCAConsensus":
        from caspailleur.api import mine_concepts

        _ensure_skmine_sklearn_validate_shim()

        self.label_matrix_ = label_matrix
        self.n_runs_ = label_matrix.shape[0]
        n_objects = label_matrix.shape[1]

        min_support_abs = max(1, math.ceil(self.min_support_frac * self.n_runs_))

        df = _build_context_df(label_matrix, noise_label=self.noise_label)
        self.context_df_ = df

        to_compute = ["extent", "intent", "support"]
        mine_kwargs: Dict = dict(
            min_support=min_support_abs / n_objects,
            to_compute=to_compute,
        )
        if self.min_delta_stability is not None:
            mine_kwargs["min_delta_stability"] = self.min_delta_stability
            if "delta_stability" not in to_compute:
                mine_kwargs["to_compute"] = to_compute + ["delta_stability"]
        if self.n_stable_concepts is not None:
            mine_kwargs["n_stable_concepts"] = self.n_stable_concepts

        try:
            concepts_df = mine_concepts(df, **mine_kwargs)
        except Exception as exc:
            warnings.warn(
                f"FCAConsensus: mine_concepts (caspailleur) failed ({exc}). "
                "Falling back to single-cluster assignment."
            )
            self.labels_ = np.zeros(n_objects, dtype=int)
            self.antichain_ = []
            return self

        self.concepts_df_ = concepts_df

        candidates: List[Dict] = []
        for _, row in concepts_df.iterrows():
            extent_names: frozenset = row["extent"]
            intent_names: frozenset = row["intent"]

            extent_ints = frozenset(int(name[1:]) for name in extent_names)

            run_support = _attr_run_support(intent_names, self.n_runs_)
            size = len(extent_ints)

            if run_support < min_support_abs:
                continue
            if size < self.min_extent_size:
                continue

            candidates.append(
                {
                    "extent": extent_ints,
                    "intent_names": intent_names,
                    "run_support": run_support,
                    "size": size,
                    "lib_support": int(row["support"]) if "support" in row else size,
                }
            )

        if self.greedy_order == "support_desc":
            candidates.sort(key=lambda c: (-c["run_support"], -c["size"]))
        elif self.greedy_order == "natural":
            pass
        else:
            candidates.sort(key=lambda c: (-c["size"], -c["run_support"]))

        antichain, covered_mask = self._extract_antichain(candidates, n_objects)
        self.antichain_ = antichain

        self.labels_ = self._assign_labels(antichain, covered_mask, n_objects)
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
                    "run_support": concept["run_support"],
                    "intent_names": concept["intent_names"],
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
    ) -> np.ndarray:
        labels = np.full(n_objects, -1, dtype=int)

        for cluster_id, concept in enumerate(antichain):
            for obj in concept["extent"]:
                labels[obj] = cluster_id

        uncovered = np.where(labels == -1)[0]
        if uncovered.size > 0:
            if not antichain:
                labels[:] = 0
                return labels

            if self.context_df_ is not None:
                context_arr = self.context_df_.values.astype(float)
                col_names = list(self.context_df_.columns)
                col_index = {name: j for j, name in enumerate(col_names)}

                concept_vecs = np.zeros((len(antichain), len(col_names)), dtype=float)
                for cid, concept in enumerate(antichain):
                    for attr_name in concept["intent_names"]:
                        if attr_name in col_index:
                            concept_vecs[cid, col_index[attr_name]] = 1.0

                for i in uncovered:
                    obj_vec = context_arr[i]
                    overlaps = concept_vecs @ obj_vec
                    labels[i] = int(np.argmax(overlaps))
            else:
                labels[uncovered] = 0

        return labels

    def describe_antichain(self) -> str:
        if self.antichain_ is None:
            return "FCAConsensus not yet fitted."
        lines = [f"FCA-Consensus antichain ({len(self.antichain_)} clusters):"]
        for k, c in enumerate(self.antichain_):
            objs = sorted(c["extent"])
            lines.append(
                f"  Cluster {k}: {len(objs)} objects, "
                f"run_support={c['run_support']}/{self.n_runs_} runs  "
                f"[objects: {objs[:8]}{'...' if len(objs) > 8 else ''}]"
            )
        return "\n".join(lines)

    def describe_concepts(self) -> str:
        if self.concepts_df_ is None:
            return "FCAConsensus not yet fitted (no caspailleur output)."
        n = len(self.concepts_df_)
        cols = list(self.concepts_df_.columns)
        return (
            f"caspailleur mined {n} concepts. "
            f"Available columns: {cols}.\n"
            f"{self.concepts_df_[['extent', 'intent']].head(5).to_string()}"
        )


__all__ = ["FCAConsensus"]
