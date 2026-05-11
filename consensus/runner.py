from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .base import build_coassociation, compute_run_weights
from .cohirf import CoHiRFConsensus
from .ensemble import CoAssocEnsemble, VotingEnsemble
from .fca import FCAConsensus
from .feature_scorer import RunQualityScorer
from .monti_helpers import auto_best_params, builtin_fit_predict_callable
from .monti_paper import MontiPaperConsensus, relabel_subsample_unique_noise


@dataclass
class ConsensusResult:

    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    coassoc_matrix: Optional[np.ndarray] = None
    monti2_coassoc_matrix: Optional[np.ndarray] = None
    label_matrix: Optional[np.ndarray] = None
    pac_scores: Dict[str, Dict[int, float]] = field(default_factory=dict)
    global_ari: float = 0.0
    k_found: Dict[str, int] = field(default_factory=dict)
    k_selection_methods: Dict[str, str] = field(default_factory=dict)
    runtime_sec: Dict[str, float] = field(default_factory=dict)
    run_weights: Optional[np.ndarray] = None

    def evaluate(self, y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for method, labels in self.labels.items():
            mask = labels >= 0
            if mask.sum() < 2:
                continue
            metrics[method] = {
                "ari": float(adjusted_rand_score(y_true[mask], labels[mask])),
                "nmi": float(
                    normalized_mutual_info_score(y_true[mask], labels[mask])
                ),
            }
        return metrics


SUPPORTED_METHODS = ("monti2", "coassoc", "voting", "cohirf", "cohirf_relaxed", "fca")


class ConsensusRunner:

    def __init__(
        self,
        algorithm_names: Sequence[str] = (
            "dbscan", "hdbscan", "dpc", "rd_dac", "ckdpc",
        ),
        algorithm_params: Optional[Dict] = None,
        consensus_methods: Sequence[str] = (
            "monti2", "coassoc", "voting", "cohirf", "fca",
        ),
        n_bootstrap: int = 30,
        p_sample: float = 0.8,
        k_range: Tuple[int, int] = (2, 8),
        n_clusters: Optional[int] = None,
        cohirf_n_repetitions: int = 5,
        cohirf_q_features: Optional[int] = None,
        cohirf_base_callable: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        cohirf_algorithm_name: Optional[str] = None,
        fca_min_support_frac: float = 0.5,
        fca_min_extent_size: int = 1,
        fca_greedy_order: str = "size_desc",
        fca_bootstrap_replicates: int = 0,
        noise_label: int = -1,
        random_state: Optional[int] = 42,
        verbose: bool = False,
        monti2_base_callable: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        monti2_algorithm_name: Optional[str] = None,
        monti2_algorithm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.algorithm_names = list(algorithm_names)
        self.algorithm_params = algorithm_params or {}
        self.monti2_base_callable = monti2_base_callable
        self.monti2_algorithm_name = monti2_algorithm_name
        self.monti2_algorithm_params = dict(monti2_algorithm_params or {})
        self.consensus_methods = [m for m in consensus_methods if m in SUPPORTED_METHODS]
        self.n_bootstrap = int(n_bootstrap)
        self.p_sample = float(p_sample)
        self.k_range = k_range
        self.n_clusters = n_clusters
        self.cohirf_n_repetitions = int(cohirf_n_repetitions)
        self.cohirf_q_features = cohirf_q_features
        self.cohirf_base_callable = cohirf_base_callable
        self.cohirf_algorithm_name = cohirf_algorithm_name
        self.fca_min_support_frac = float(fca_min_support_frac)
        self.fca_min_extent_size = int(fca_min_extent_size)
        self.fca_greedy_order = fca_greedy_order
        self.fca_bootstrap_replicates = int(fca_bootstrap_replicates)
        self.noise_label = noise_label
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose

    def fit(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> ConsensusResult:
        from algorithms.base import AlgorithmRegistry

        n_objects = X.shape[0]
        result = ConsensusResult()

        all_labels: List[np.ndarray] = []
        alg_callables: List = []
        fca_bootstrap_seed_cls_mp: Optional[Tuple[Any, Dict[str, Any]]] = None

        for name in self.algorithm_names:
            try:
                cls = AlgorithmRegistry.get(name)
            except ValueError:
                continue

            param_list = self.algorithm_params.get(name, [{}])
            if isinstance(param_list, dict):
                param_list = [param_list]

            for params in param_list:
                try:
                    pdict = dict(params) if params else {}
                    if pdict:
                        mp: Dict[str, Any] = pdict
                    else:
                        mp = dict(auto_best_params(name, X, y_true))
                    alg = cls(**mp)
                    labels = np.asarray(alg.fit_predict(X), dtype=int)
                    all_labels.append(labels)

                    def _fp(Z: np.ndarray, _cls=cls, _mp=dict(mp)) -> np.ndarray:
                        return np.asarray(_cls(**_mp).fit_predict(Z), dtype=int)

                    alg_callables.append(_fp)

                    if fca_bootstrap_seed_cls_mp is None:
                        fca_bootstrap_seed_cls_mp = (cls, dict(mp))

                    if self.verbose:
                        n_c = len(np.unique(labels[labels >= 0]))
                        print(f"  {name}({mp}): {n_c} clusters")
                except Exception as e:
                    if self.verbose:
                        print(f"  [error] {name}({params}): {e}")

        fb = self.fca_bootstrap_replicates
        if (
            fb > 0
            and "fca" in self.consensus_methods
            and fca_bootstrap_seed_cls_mp is not None
            and n_objects >= 3
        ):
            b_cls, b_mp = fca_bootstrap_seed_cls_mp
            for _ in range(fb):
                size = max(2, int(n_objects * self.p_sample))
                idx = self.rng.choice(
                    n_objects, size=min(size, n_objects), replace=False
                )
                try:
                    sub_l = np.asarray(b_cls(**b_mp).fit_predict(X[idx]), dtype=int)
                except Exception:
                    continue
                sub_l = relabel_subsample_unique_noise(sub_l, self.noise_label)
                full = np.full(n_objects, self.noise_label, dtype=int)
                full[idx] = sub_l
                all_labels.append(full)

        if "monti2" in self.consensus_methods:
            t0 = time.perf_counter()
            monti_fn: Callable[[np.ndarray], np.ndarray]
            if self.monti2_base_callable is not None:
                monti_fn = self.monti2_base_callable
            else:
                mname = self.monti2_algorithm_name or (
                    self.algorithm_names[0] if self.algorithm_names else "hdbscan"
                )
                try:
                    mcls = AlgorithmRegistry.get(mname)
                except ValueError:
                    mcls = AlgorithmRegistry.get("hdbscan")
                    mname = "hdbscan"
                mparams: Optional[Dict[str, Any]] = None
                if self.monti2_algorithm_params:
                    mparams = dict(self.monti2_algorithm_params)
                if mparams is None:
                    plist = self.algorithm_params.get(mname, [{}])
                    if isinstance(plist, dict):
                        plist = [plist]
                    if plist and plist[0]:
                        mparams = dict(plist[0])
                if mparams is None:
                    mparams = dict(auto_best_params(mname, X, y_true))

                def _monti_closure(Z: np.ndarray, _c=mcls, _p=mparams) -> np.ndarray:
                    return np.asarray(_c(**_p).fit_predict(Z), dtype=int)

                monti_fn = _monti_closure
            monti2 = MontiPaperConsensus(
                base_algorithm=monti_fn,
                n_resamples=self.n_bootstrap,
                p_sample=self.p_sample,
                k_range=self.k_range,
                n_clusters=self.n_clusters,
                random_state=int(self.rng.integers(0, 2**31)),
                noise_label=self.noise_label,
            )
            monti2.fit(X)
            result.labels["monti2"] = monti2.labels_
            result.monti2_coassoc_matrix = monti2.coassoc_matrix_
            result.k_selection_methods["monti2"] = monti2.k_selection_method_ or ""
            result.k_found["monti2"] = int(monti2.optimal_k_) if monti2.optimal_k_ else 0
            result.runtime_sec["monti2"] = time.perf_counter() - t0

        if all_labels:
            result.label_matrix = np.vstack(all_labels)
            run_weights = compute_run_weights(
                result.label_matrix, n_objects, noise_label=self.noise_label
            )
            result.run_weights = run_weights
            C_global, _ = build_coassociation(
                result.label_matrix, n_objects,
                noise_label=self.noise_label,
                weights=run_weights,
            )
            result.coassoc_matrix = C_global
            scorer = RunQualityScorer(
                result.label_matrix, n_objects, noise_label=self.noise_label
            )
            result.global_ari = scorer.pairwise_ari()
        else:
            result.label_matrix = None
            result.run_weights = None
            result.global_ari = 0.0

        if "coassoc" in self.consensus_methods and alg_callables:
            t0 = time.perf_counter()
            coassoc = CoAssocEnsemble(
                algorithms=alg_callables,
                k_range=self.k_range,
                n_clusters=self.n_clusters,
                noise_label=self.noise_label,
            )
            coassoc.fit(X)
            result.labels["coassoc"] = coassoc.labels_
            result.pac_scores["coassoc"] = coassoc.pac_scores_ or {}
            result.k_selection_methods["coassoc"] = coassoc.k_selection_method_ or ""
            result.k_found["coassoc"] = int(coassoc.optimal_k_) if coassoc.optimal_k_ else 0
            result.runtime_sec["coassoc"] = time.perf_counter() - t0

        if "voting" in self.consensus_methods and alg_callables:
            t0 = time.perf_counter()
            voting = VotingEnsemble(
                algorithms=alg_callables,
                n_clusters=self.n_clusters,
                noise_label=self.noise_label,
            )
            voting.fit(X)
            result.labels["voting"] = voting.labels_
            k_v = len(np.unique(voting.labels_[voting.labels_ >= 0]))
            result.k_found["voting"] = int(k_v)
            result.runtime_sec["voting"] = time.perf_counter() - t0

        if "cohirf" in self.consensus_methods or "cohirf_relaxed" in self.consensus_methods:
            cohirf_fn: Callable[[np.ndarray], np.ndarray]
            if self.cohirf_base_callable is not None:
                cohirf_fn = self.cohirf_base_callable
            else:
                cname = self.cohirf_algorithm_name or (
                    self.algorithm_names[0] if self.algorithm_names else "hdbscan"
                )
                try:
                    cohirf_fn = builtin_fit_predict_callable(cname, X, y_true)
                except Exception:
                    cohirf_fn = builtin_fit_predict_callable("hdbscan", X, y_true)
            rs = int(self.rng.integers(0, 2**31))
            if "cohirf" in self.consensus_methods:
                t0 = time.perf_counter()
                cohirf = CoHiRFConsensus(
                    base_algorithm=cohirf_fn,
                    n_repetitions=self.cohirf_n_repetitions,
                    n_q_features=self.cohirf_q_features,
                    relaxed=False,
                    random_state=rs,
                    noise_label=self.noise_label,
                )
                cohirf.fit(X)
                result.labels["cohirf"] = cohirf.labels_
                result.pac_scores["cohirf"] = cohirf.pac_scores_ or {}
                result.k_selection_methods["cohirf"] = cohirf.k_selection_method_ or ""
                result.k_found["cohirf"] = int(cohirf.optimal_k_) if cohirf.optimal_k_ else 0
                result.runtime_sec["cohirf"] = time.perf_counter() - t0
            if "cohirf_relaxed" in self.consensus_methods:
                t1 = time.perf_counter()
                cohirf_r = CoHiRFConsensus(
                    base_algorithm=cohirf_fn,
                    n_repetitions=self.cohirf_n_repetitions,
                    n_q_features=self.cohirf_q_features,
                    relaxed=True,
                    random_state=rs + 1,
                    noise_label=self.noise_label,
                )
                cohirf_r.fit(X)
                result.labels["cohirf_relaxed"] = cohirf_r.labels_
                result.pac_scores["cohirf_relaxed"] = cohirf_r.pac_scores_ or {}
                result.k_selection_methods["cohirf_relaxed"] = (
                    cohirf_r.k_selection_method_ or ""
                )
                result.k_found["cohirf_relaxed"] = (
                    int(cohirf_r.optimal_k_) if cohirf_r.optimal_k_ else 0
                )
                result.runtime_sec["cohirf_relaxed"] = time.perf_counter() - t1

        if "fca" in self.consensus_methods and result.label_matrix is not None:
            t0 = time.perf_counter()
            fca = FCAConsensus(
                min_support_frac=self.fca_min_support_frac,
                min_extent_size=self.fca_min_extent_size,
                greedy_order=self.fca_greedy_order,
                noise_label=self.noise_label,
            )
            fca.fit_from_labels(result.label_matrix)
            result.labels["fca"] = fca.labels_
            k_f = len(np.unique(fca.labels_[fca.labels_ >= 0]))
            result.k_found["fca"] = int(k_f)
            result.runtime_sec["fca"] = time.perf_counter() - t0

        return result

    def fit_evaluate(
        self, X: np.ndarray, y_true: np.ndarray
    ) -> Tuple[ConsensusResult, Dict]:
        result = self.fit(X, y_true=y_true)
        metrics = result.evaluate(y_true)
        return result, metrics


__all__ = ["ConsensusRunner", "ConsensusResult"]
