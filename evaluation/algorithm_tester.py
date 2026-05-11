from __future__ import annotations

import importlib.util
import inspect
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .metrics import ClusteringMetrics


@dataclass
class AlgorithmReport:

    algorithm_name: str = ""
    dataset_results: List[Dict[str, Any]] = field(default_factory=list)
    consensus_contribution: Dict[str, float] = field(default_factory=dict)
    run_weight_mean: float = 0.0
    issues: List[str] = field(default_factory=list)

    def mean_metric(self, group: str, metric: str) -> float:
        vals = [
            r[group][metric]
            for r in self.dataset_results
            if group in r and metric in r[group]
            and not np.isnan(r[group][metric])
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"Algorithm: {self.algorithm_name}",
            f"{'='*60}",
            f"Datasets tested: {len(self.dataset_results)}",
        ]

        # External metrics
        ari = self.mean_metric("external", "ari")
        nmi = self.mean_metric("external", "nmi")
        purity = self.mean_metric("external", "purity")
        if not np.isnan(ari):
            lines += [
                "",
                "--- Clustering Quality (external, avg across datasets) ---",
                f"  ARI:     {ari:.3f}  (1.0 = perfect)",
                f"  NMI:     {nmi:.3f}",
                f"  Purity:  {purity:.3f}",
            ]

        # Internal metrics
        sil = self.mean_metric("internal", "silhouette")
        db  = self.mean_metric("internal", "davies_bouldin")
        if not np.isnan(sil):
            lines += [
                "",
                "--- Internal Quality (avg across datasets) ---",
                f"  Silhouette:      {sil:.3f}  (→1 = best)",
                f"  Davies-Bouldin:  {db:.3f}  (→0 = best)",
            ]

        # Structure
        noise = self.mean_metric("structure", "noise_fraction")
        n_c   = self.mean_metric("structure", "n_clusters")
        if not np.isnan(noise):
            lines += [
                "",
                "--- Algorithm Behaviour ---",
                f"  Avg noise fraction:  {noise:.1%}",
                f"  Avg clusters found:  {n_c:.1f}",
                f"  Run quality weight:  {self.run_weight_mean:.3f}  (1.0 = average)",
            ]

        # Consensus
        if self.consensus_contribution:
            lines += ["", "--- Consensus Contribution ---"]
            for k, v in self.consensus_contribution.items():
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.3f}")
                else:
                    lines.append(f"  {k}: {v}")

        # Issues
        if self.issues:
            lines += ["", "--- Warnings / Issues ---"]
            for issue in self.issues:
                lines.append(f"  ⚠ {issue}")

        lines.append("=" * 60)
        return "\n".join(lines)


def load_algorithm_from_file(
    filepath: str | Path,
    class_name: Optional[str] = None,
    init_params: Optional[Dict] = None,
) -> Any:

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Algorithm file not found: {filepath}")

    spec = importlib.util.spec_from_file_location("user_algorithm", filepath)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from {filepath}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ValueError(f"Error executing {filepath}: {e}") from e

    # Find suitable class
    candidates = [
        (name, obj)
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if obj.__module__ == module.__name__
        and hasattr(obj, "fit_predict")
    ]

    if not candidates:
        raise ValueError(
            f"No class with 'fit_predict' method found in {filepath}. "
            "Make sure your class implements fit_predict(X) -> labels."
        )

    if class_name is not None:
        match = [c for c in candidates if c[0] == class_name]
        if not match:
            raise ValueError(
                f"Class '{class_name}' not found in {filepath}. "
                f"Available: {[c[0] for c in candidates]}"
            )
        cls = match[0][1]
    else:
        cls = candidates[0][1]

    init_params = init_params or {}
    try:
        return cls(**init_params)
    except Exception as e:
        raise ValueError(f"Cannot instantiate {cls.__name__}({init_params}): {e}") from e


def validate_algorithm(algorithm: Any) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    if not hasattr(algorithm, "fit_predict"):
        issues.append("Missing 'fit_predict' method.")
        return False, issues

    sig = inspect.signature(algorithm.fit_predict)
    params = list(sig.parameters.keys())
    # Should accept at least 'X' (besides 'self' which is stripped)
    if not params:
        issues.append("fit_predict() must accept at least one argument X.")

    # Quick smoke test with tiny data
    try:
        X_tiny = np.random.default_rng(0).standard_normal((5, 3))
        out = algorithm.fit_predict(X_tiny)
        out = np.asarray(out)
        if out.shape != (5,):
            issues.append(
                f"fit_predict should return shape (n_samples,), got {out.shape}."
            )
        if not np.issubdtype(out.dtype, np.integer):
            issues.append(
                f"fit_predict should return integer labels, got dtype={out.dtype}."
            )
    except Exception as e:
        issues.append(
            f"Запуск на тестовых данных завершился ошибкой — скорее всего, "
            f"неверные значения параметров. Причина: {e}"
        )

    return len(issues) == 0, issues


class AlgorithmTester:

    def __init__(
        self,
        algorithm: Any,
        algorithm_name: str = "user_algorithm",
        noise_label: int = -1,
        consensus_methods: List[str] = ("coassoc", "voting"),
        k_range: Tuple[int, int] = (2, 8),
        random_state: int = 42,
        verbose: bool = True,
    ) -> None:
        if callable(algorithm) and not hasattr(algorithm, "fit_predict"):
            class _Wrapper:
                def fit_predict(self_, X: np.ndarray) -> np.ndarray:
                    return algorithm(X)
            self.algorithm = _Wrapper()
        else:
            self.algorithm = algorithm

        self.algorithm_name = algorithm_name
        self.noise_label = noise_label
        self.consensus_methods = list(consensus_methods)
        self.k_range = k_range
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose

    def test(
        self,
        datasets: List[Dict],
        include_consensus: bool = True,
    ) -> AlgorithmReport:
        report = AlgorithmReport(algorithm_name=self.algorithm_name)

        valid, val_issues = validate_algorithm(self.algorithm)
        report.issues.extend(val_issues)
        if not valid:
            return report

        all_weights: List[float] = []

        for ds in datasets:
            X = np.asarray(ds["X"])
            y_true = ds.get("y_true")
            name = ds.get("name", "unnamed")

            if self.verbose:
                print(f"  Testing on dataset '{name}' {X.shape}…")

            ds_result: Dict[str, Any] = {"dataset": name}

            # Run algorithm
            t0 = time.perf_counter()
            try:
                labels = np.asarray(self.algorithm.fit_predict(X), dtype=int)
            except Exception as e:
                report.issues.append(f"[{name}] fit_predict failed: {e}")
                traceback.print_exc()
                continue

            if labels.shape[0] != X.shape[0]:
                report.issues.append(
                    f"[{name}] fit_predict returned {labels.shape[0]} labels "
                    f"but X has {X.shape[0]} rows (n_samples). "
                    f"Expected one label per object (row of X)."
                )
                continue

            ds_result["runtime_sec"] = time.perf_counter() - t0

            # Compute metrics
            m = ClusteringMetrics(
                labels=labels,
                y_true=y_true,
                X=X,
                noise_label=self.noise_label,
            )
            ds_result.update(m.all_metrics())
            ds_result["labels"] = labels

            # Compute run quality weight
            from consensus.base import compute_run_weights
            lm = labels.reshape(1, -1)
            w = compute_run_weights(lm, X.shape[0], noise_label=self.noise_label)
            ds_result["run_weight"] = float(w[0])
            all_weights.append(float(w[0]))

            if self.verbose:
                ext = ds_result.get("external", {})
                struct = ds_result.get("structure", {})
                print(
                    f"    ARI={ext.get('ari', 'n/a'):.3f}  "
                    f"k={struct.get('n_clusters', '?')}  "
                    f"noise={struct.get('noise_fraction', 0):.0%}  "
                    f"weight={ds_result['run_weight']:.3f}  "
                    f"({ds_result['runtime_sec']*1000:.0f}ms)"
                    if "ari" in ext else
                    f"    k={struct.get('n_clusters', '?')}  "
                    f"noise={struct.get('noise_fraction', 0):.0%}  "
                    f"weight={ds_result['run_weight']:.3f}  "
                    f"({ds_result['runtime_sec']*1000:.0f}ms)"
                )

            report.dataset_results.append(ds_result)

        report.run_weight_mean = float(np.mean(all_weights)) if all_weights else 0.0

        # Consensus contribution 
        if include_consensus and report.dataset_results:
            try:
                report.consensus_contribution = self._test_consensus_contribution(
                    datasets
                )
            except Exception as e:
                report.issues.append(f"Consensus contribution test failed: {e}")

        return report


    def _test_consensus_contribution(
        self, datasets: List[Dict]
    ) -> Dict[str, float]:
        from consensus.runner import ConsensusRunner
        import algorithms.registry  # ensure base algorithms are registered

        results_with: List[float] = []
        results_without: List[float] = []

        for ds in datasets:
            X = np.asarray(ds["X"])
            y_true = ds.get("y_true")
            if y_true is None:
                continue  # skip datasets without ground truth for this test

            # Build callable from user algorithm
            user_fn = self.algorithm.fit_predict

            # Consensus WITH user algorithm (inject into coassoc ensemble)
            from consensus.ensemble import CoAssocEnsemble
            from algorithms.base import AlgorithmRegistry

            # Get 2-3 reference algorithms
            ref_names = ["hdbscan", "dpc", "dac"]
            ref_callables = []
            for name in ref_names:
                try:
                    cls = AlgorithmRegistry.get(name)
                    ref_callables.append(cls().fit_predict)
                except Exception:
                    pass

            if not ref_callables:
                continue

            # Without user algorithm
            ens_without = CoAssocEnsemble(
                algorithms=ref_callables,
                k_range=self.k_range,
                noise_label=self.noise_label,
            )
            try:
                ens_without.fit(X)
                labels_without = ens_without.labels_
                ari_without = adjusted_rand_score(y_true, labels_without)
                results_without.append(ari_without)
            except Exception:
                continue

            # With user algorithm added
            ens_with = CoAssocEnsemble(
                algorithms=ref_callables + [user_fn],
                k_range=self.k_range,
                noise_label=self.noise_label,
            )
            try:
                ens_with.fit(X)
                labels_with = ens_with.labels_
                ari_with = adjusted_rand_score(y_true, labels_with)
                results_with.append(ari_with)
            except Exception:
                continue

        if not results_with:
            return {}

        mean_with = float(np.mean(results_with))
        mean_without = float(np.mean(results_without))
        delta = mean_with - mean_without

        return {
            "consensus_ari_with":    mean_with,
            "consensus_ari_without": mean_without,
            "consensus_ari_delta":   delta,
            "interpretation": (
                "IMPROVES consensus" if delta > 0.02
                else "NEUTRAL" if abs(delta) <= 0.02
                else "HURTS consensus"
            ),
        }


try:
    from sklearn.metrics import adjusted_rand_score
except ImportError:
    pass
