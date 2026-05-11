from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class BenchmarkResult:
    algorithm_names: List[str] = field(default_factory=list)
    dataset_names:   List[str] = field(default_factory=list)
    # ari[alg_idx][dataset_idx]
    ari_matrix:    Optional[np.ndarray] = None
    nmi_matrix:    Optional[np.ndarray] = None
    noise_matrix:  Optional[np.ndarray] = None
    weight_matrix: Optional[np.ndarray] = None
    runtime_matrix: Optional[np.ndarray] = None

    def summary_table(self) -> str:
        if self.ari_matrix is None:
            return "No results."
        lines = ["ARI matrix (rows=algorithms, cols=datasets)", ""]
        col_w = max(12, max((len(d) for d in self.dataset_names), default=0) + 2)
        alg_w = max(14, max((len(a) for a in self.algorithm_names), default=0) + 2)
        header = f"{'Algorithm':<{alg_w}}" + "".join(
            f"{d:>{col_w}}" for d in self.dataset_names
        ) + f"{'Mean':>{col_w}}"
        lines.append(header)
        lines.append("-" * len(header))
        for i, alg in enumerate(self.algorithm_names):
            row = self.ari_matrix[i]
            mean = float(np.nanmean(row))
            line = f"{alg:<{alg_w}}" + "".join(
                f"{v:>{col_w}.3f}" for v in row
            ) + f"{mean:>{col_w}.3f}"
            lines.append(line)
        return "\n".join(lines)


class BenchmarkSuite:

    def __init__(self, n_samples: int = 50, random_state: int = 42) -> None:
        self.n_samples = n_samples
        self.rng = np.random.default_rng(random_state)

    def gaussian_clusters(
        self,
        n_clusters: int = 3,
        n_per_cluster: int = 15,
        noise: float = 0.15,
        name: str = "gaussian_3c",
    ) -> Dict:
        S = self.n_samples
        latent = self.rng.standard_normal((n_clusters, S))
        groups = [
            latent[c] + self.rng.standard_normal((n_per_cluster, S)) * noise
            for c in range(n_clusters)
        ]
        X = np.vstack(groups)
        y_true = np.repeat(np.arange(n_clusters), n_per_cluster)
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": f"{n_clusters} Gaussian feature groups, noise={noise}",
        }

    def varying_density(
        self,
        name: str = "varying_density",
    ) -> Dict:
        S = self.n_samples
        latent = [self.rng.standard_normal(S) for _ in range(3)]
        g1 = latent[0] + self.rng.standard_normal((10, S)) * 0.05   # tight
        g2 = latent[1] + self.rng.standard_normal((10, S)) * 0.40   # medium
        g3 = latent[2] + self.rng.standard_normal((10, S)) * 1.20   # loose
        X = np.vstack([g1, g2, g3])
        y_true = np.repeat([0, 1, 2], 10)
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": "3 groups with very different within-group variances",
        }

    def high_noise(
        self,
        n_clusters: int = 3,
        n_per_cluster: int = 10,
        n_noise_features: int = 15,
        name: str = "high_noise",
    ) -> Dict:
        S = self.n_samples
        latent = self.rng.standard_normal((n_clusters, S))
        groups = [
            latent[c] + self.rng.standard_normal((n_per_cluster, S)) * 0.2
            for c in range(n_clusters)
        ]
        noise_feats = self.rng.standard_normal((n_noise_features, S))
        X = np.vstack(groups + [noise_feats])
        # Ground truth: noise features labeled as a separate "noise" group (label=n_clusters)
        y_true = np.concatenate([
            np.repeat(np.arange(n_clusters), n_per_cluster),
            np.full(n_noise_features, n_clusters),
        ])
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": f"{n_clusters} groups + {n_noise_features} noise features",
        }

    def overlapping_clusters(
        self,
        name: str = "overlapping",
    ) -> Dict:
        S = self.n_samples
        g1 = self.rng.standard_normal((15, S)) * 1.0
        g2 = g1 + self.rng.standard_normal((15, S)) * 0.8 + 0.5  # offset + noise
        X = np.vstack([g1, g2])
        y_true = np.repeat([0, 1], 15)
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": "2 overlapping feature groups (hard case)",
        }

    def many_clusters(
        self,
        n_clusters: int = 7,
        n_per_cluster: int = 8,
        name: str = "many_clusters",
    ) -> Dict:
        S = self.n_samples
        latent = self.rng.standard_normal((n_clusters, S))
        groups = [
            latent[c] + self.rng.standard_normal((n_per_cluster, S)) * 0.1
            for c in range(n_clusters)
        ]
        X = np.vstack(groups)
        y_true = np.repeat(np.arange(n_clusters), n_per_cluster)
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": f"{n_clusters} tight feature clusters",
        }

    def time_series_style(
        self,
        n_clusters: int = 3,
        n_per_cluster: int = 12,
        name: str = "time_series",
    ) -> Dict:
        S = self.n_samples
        t = np.linspace(0, 4 * np.pi, S)
        slopes = self.rng.uniform(0.1, 0.8, n_clusters)
        freqs  = self.rng.uniform(0.3, 1.5, n_clusters)
        groups = []
        for c in range(n_clusters):
            base = slopes[c] * np.linspace(0, 1, S) + np.sin(t * freqs[c])
            feats = base + self.rng.standard_normal((n_per_cluster, S)) * 0.15
            groups.append(feats)
        X = np.vstack(groups)
        y_true = np.repeat(np.arange(n_clusters), n_per_cluster)
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": f"{n_clusters} time-series groups (trend + seasonal patterns)",
        }

    def blobs_style(
        self,
        n_clusters: int = 4,
        n_per_cluster: int = 12,
        cluster_std: float = 0.5,
        name: str = "blobs",
    ) -> Dict:
        S = self.n_samples
        centers = self.rng.standard_normal((n_clusters, S))
        groups = [
            centers[c] + self.rng.standard_normal((n_per_cluster, S)) * cluster_std
            for c in range(n_clusters)
        ]
        X = np.vstack(groups)
        y_true = np.repeat(np.arange(n_clusters), n_per_cluster)
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": f"{n_clusters} blobs, cluster_std={cluster_std:.2f}",
        }

    def low_dim_projection(
        self,
        n_clusters: int = 3,
        n_per_cluster: int = 15,
        n_latent: int = 3,
        name: str = "low_dim_proj",
    ) -> Dict:
        S = self.n_samples
        latent_signals = self.rng.standard_normal((n_clusters, n_latent))
        proj = self.rng.standard_normal((n_latent, S))
        groups = []
        for c in range(n_clusters):
            base  = latent_signals[c] @ proj          # (S,)
            feats = base + self.rng.standard_normal((n_per_cluster, S)) * 0.3
            groups.append(feats)
        X = np.vstack(groups)
        y_true = np.repeat(np.arange(n_clusters), n_per_cluster)
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": f"{n_clusters} groups via {n_latent}D latent projection",
        }

    def regression_style(
        self,
        n_clusters: int = 3,
        n_per_cluster: int = 12,
        noise_level: float = 0.3,
        name: str = "regression",
    ) -> Dict:
        S = self.n_samples
        x_base = np.linspace(-2, 2, S)
        slopes = self.rng.uniform(-2, 2, n_clusters)
        biases = self.rng.uniform(-1, 1, n_clusters)
        groups = []
        for c in range(n_clusters):
            base  = slopes[c] * x_base + biases[c]
            feats = base + self.rng.standard_normal((n_per_cluster, S)) * noise_level
            groups.append(feats)
        X = np.vstack(groups)
        y_true = np.repeat(np.arange(n_clusters), n_per_cluster)
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": f"{n_clusters} groups with linear regression-style patterns",
        }

    def mixed_distributions(
        self,
        n_per_cluster: int = 12,
        name: str = "mixed_dist",
    ) -> Dict:
        S = self.n_samples
        # Normal
        g0_base = self.rng.standard_normal(S)
        g0 = g0_base + self.rng.standard_normal((n_per_cluster, S)) * 0.2
        # Uniform (centered)
        g1_base = self.rng.uniform(-1, 1, S)
        g1 = g1_base + self.rng.standard_normal((n_per_cluster, S)) * 0.15
        # Exponential (zero-meaned)
        g2_base = self.rng.exponential(1.0, S); g2_base -= g2_base.mean()
        g2 = g2_base + self.rng.standard_normal((n_per_cluster, S)) * 0.25
        X = np.vstack([g0, g1, g2])
        y_true = np.repeat([0, 1, 2], n_per_cluster)
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": "3 clusters from Normal / Uniform / Exponential distributions",
        }

    def custom(
        self,
        n_clusters: int = 3,
        n_per_cluster: int = 12,
        noise_level: float = 0.2,
        n_noise_features: int = 0,
        ds_type: str = "gaussian",
        cluster_std: float = 0.3,
        name: str = "custom",
    ) -> Dict:
        type_map = {
            "gaussian":       lambda: self.gaussian_clusters(n_clusters, n_per_cluster, noise_level, name=name),
            "blobs":          lambda: self.blobs_style(n_clusters, n_per_cluster, cluster_std, name=name),
            "time_series":    lambda: self.time_series_style(n_clusters, n_per_cluster, name=name),
            "low_dim":        lambda: self.low_dim_projection(n_clusters, n_per_cluster, name=name),
            "regression":     lambda: self.regression_style(n_clusters, n_per_cluster, noise_level, name=name),
            "mixed":          lambda: self.mixed_distributions(n_per_cluster, name=name),
            "varying_density":lambda: self.varying_density(name=name),
            "high_noise":     lambda: self.high_noise(n_clusters, n_per_cluster, n_noise_features or 10, name=name),
            "overlapping":    lambda: self.overlapping_clusters(name=name),
            "many_clusters":  lambda: self.many_clusters(n_clusters, n_per_cluster, name=name),
        }
        ds = type_map.get(ds_type, type_map["gaussian"])()
        if n_noise_features > 0 and ds_type not in ("high_noise",):
            noise_X = self.rng.standard_normal((n_noise_features, self.n_samples))
            ds["X"] = np.vstack([ds["X"], noise_X])
            ds["y_true"] = np.concatenate([
                ds["y_true"], np.full(n_noise_features, n_clusters),
            ])
            ds["description"] += f" + {n_noise_features} noise objects"
        return ds

    def high_dimensional(
        self,
        n_clusters: int = 3,
        n_per_cluster: int = 20,
        n_samples: int = 200,
        name: str = "high_dim",
    ) -> Dict:
        rng = self.rng
        latent = rng.standard_normal((n_clusters, n_samples))
        groups = [
            latent[c] + rng.standard_normal((n_per_cluster, n_samples)) * 0.2
            for c in range(n_clusters)
        ]
        X = np.vstack(groups)
        y_true = np.repeat(np.arange(n_clusters), n_per_cluster)
        return {
            "X": X, "y_true": y_true, "name": name,
            "description": f"{n_clusters} groups, {n_samples} samples (high-dim)",
        }

    def standard_suite(self) -> List[Dict]:
        return [
            self.gaussian_clusters(n_clusters=3, n_per_cluster=15, noise=0.15),
            self.varying_density(),
            self.high_noise(),
            self.many_clusters(n_clusters=5, n_per_cluster=8),
        ]

    def all_datasets(self) -> List[Dict]:
        return [
            self.gaussian_clusters(n_clusters=3, n_per_cluster=15, noise=0.15),
            self.gaussian_clusters(n_clusters=5, n_per_cluster=10, noise=0.25,
                                   name="gaussian_5c"),
            self.varying_density(),
            self.high_noise(),
            self.overlapping_clusters(),
            self.many_clusters(n_clusters=5, n_per_cluster=8),
            self.time_series_style(),
            self.blobs_style(),
            self.low_dim_projection(),
            self.regression_style(),
        ]

    @classmethod
    def tester_full_suite(
        cls,
        n_samples: int = 50,
        random_state: int = 42,
        user_datasets: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict]:
        suite = cls(n_samples=n_samples, random_state=random_state)
        seen: set[str] = set()
        out: List[Dict] = []

        for ds in suite.all_datasets():
            nm = str(ds.get("name", ""))
            if nm in seen:
                continue
            seen.add(nm)
            out.append(ds)

        md = suite.mixed_distributions()
        nm_md = str(md.get("name", "mixed_dist"))
        if nm_md not in seen:
            seen.add(nm_md)
            out.append(md)

        try:
            from data_generator.registry import build_all
        except ImportError:
            build_all = None  # type: ignore[misc, assignment]
        if build_all is not None:
            rng = np.random.default_rng(random_state)
            n_prof = int(np.clip(n_samples, 16, 256))
            n_habr = int(np.clip(n_samples, 20, 300))
            for d in build_all(
                n_profile_samples=n_prof,
                n_habr_features=n_habr,
                rng=rng,
            ):
                nm = str(d.name)
                if nm in seen:
                    continue
                seen.add(nm)
                out.append(d.to_dict())

        if user_datasets:
            for name, d in user_datasets.items():
                nm = str(name)
                if nm in seen:
                    nm = f"user::{nm}"
                if nm in seen:
                    continue
                seen.add(nm)
                X = np.asarray(d["X"])
                if X.ndim != 2:
                    continue
                y_true = d.get("y_true")
                if y_true is None:
                    y_true = np.full(X.shape[0], -1, dtype=np.int64)
                else:
                    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
                    if y_true.shape[0] != X.shape[0]:
                        continue
                out.append({
                    "name": str(name),
                    "X": X,
                    "y_true": y_true,
                    "description": d.get("description", ""),
                    "source": d.get("source", "user"),
                })
        return out

    # Human-readable registry for webapp display
    DATASET_TYPE_LABELS: Dict = {
        "gaussian":        "Gaussian clusters (easy baseline)",
        "blobs":           "Blobs (adjustable spread, like sklearn)",
        "time_series":     "Time-series patterns (trend + seasonal)",
        "low_dim":         "Low-dim projection (latent structure)",
        "regression":      "Regression-style (linear feature patterns)",
        "mixed":           "Mixed distributions (Normal / Uniform / Exp)",
        "varying_density": "Varying density (different within-group variances)",
        "high_noise":      "High noise (real groups + random noise objects)",
        "overlapping":     "Overlapping clusters (hard case)",
        "many_clusters":   "Many small clusters (count detection)",
    }

    @classmethod
    def list_types(cls) -> List[str]:
        return list(cls.DATASET_TYPE_LABELS.keys())

    def compare_algorithms(
        self,
        algorithm_dict: Dict[str, object],
        datasets: Optional[List[Dict]] = None,
        noise_label: int = -1,
    ) -> BenchmarkResult:
        from .metrics import ClusteringMetrics
        from consensus.base import compute_run_weights

        if datasets is None:
            datasets = self.standard_suite()

        alg_names = list(algorithm_dict.keys())
        ds_names = [ds["name"] for ds in datasets]
        n_alg = len(alg_names)
        n_ds = len(ds_names)

        ari_m    = np.full((n_alg, n_ds), np.nan)
        nmi_m    = np.full((n_alg, n_ds), np.nan)
        noise_m  = np.full((n_alg, n_ds), np.nan)
        weight_m = np.full((n_alg, n_ds), np.nan)
        time_m   = np.full((n_alg, n_ds), np.nan)

        for j, ds in enumerate(datasets):
            X = np.asarray(ds["X"])
            y_true = ds.get("y_true")
            for i, (name, alg) in enumerate(algorithm_dict.items()):
                try:
                    import time as _time
                    t0 = _time.perf_counter()
                    labels = np.asarray(alg.fit_predict(X), dtype=int)
                    time_m[i, j] = _time.perf_counter() - t0

                    m = ClusteringMetrics(labels, y_true, X, noise_label)
                    ext = m.external()
                    struct = m.structure()
                    ari_m[i, j]    = ext.get("ari", np.nan)
                    nmi_m[i, j]    = ext.get("nmi", np.nan)
                    noise_m[i, j]  = struct.get("noise_fraction", np.nan)
                    lm = labels.reshape(1, -1)
                    weight_m[i, j] = float(
                        compute_run_weights(lm, X.shape[0], noise_label)[0]
                    )
                except Exception:
                    pass

        return BenchmarkResult(
            algorithm_names=alg_names,
            dataset_names=ds_names,
            ari_matrix=ari_m,
            nmi_matrix=nmi_m,
            noise_matrix=noise_m,
            weight_matrix=weight_m,
            runtime_matrix=time_m,
        )
