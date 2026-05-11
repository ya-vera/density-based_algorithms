from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.datasets import make_blobs

from .schema import ClusteringDataset


def _stack_groups(groups: list[np.ndarray], labels: list[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.vstack(groups)
    y = np.concatenate(labels)
    return X, y.astype(np.int64)


def make_numpy_linear_features(
    n_per_group: int = 50,
    n_groups: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> ClusteringDataset:
    rng = np.random.default_rng(0) if rng is None else rng
    slopes  = [0.5, 0.5, 0.5]
    offsets = [0.0, 10.0, 20.0]
    groups, labs = [], []
    for g in range(n_groups):
        x = rng.uniform(0, 10, n_per_group)
        y = slopes[g] * x + offsets[g] + rng.normal(0, 0.6, n_per_group)
        groups.append(np.column_stack([x, y]))
        labs.append(np.full(n_per_group, g))
    X, y_true = _stack_groups(groups, labs)
    return ClusteringDataset(
        name="habr_numpy_linear",
        X=X,
        y_true=y_true,
        description=f"Линейный тренд (NumPy): {n_per_group * n_groups} объектов, 2 признака, {n_groups} кластера",
        category="habr",
        source="https://habr.com/ru/companies/netologyru/articles/841952/",
    )


def make_numpy_timeseries_features(
    n_per_group: int = 50,
    n_groups: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> ClusteringDataset:
    rng = np.random.default_rng(1) if rng is None else rng
    t_vals = np.arange(0, 40, dtype=float)
    configs = [
        (0.10, 1.0, 0.0),
        (0.20, 2.0, 14.0),
        (0.35, 3.0, 30.0),
    ]
    groups, labs = [], []
    for g, (trend_w, amp, offset) in enumerate(configs):
        idx = rng.choice(len(t_vals), n_per_group, replace=True)
        t_i = t_vals[idx]
        y_i = trend_w * t_i + amp * np.sin(0.3 * t_i) + offset + rng.normal(0, 0.5, n_per_group)
        groups.append(np.column_stack([t_i, y_i]))
        labs.append(np.full(n_per_group, g))
    X, y_true = _stack_groups(groups, labs)
    return ClusteringDataset(
        name="habr_numpy_timeseries",
        X=X,
        y_true=y_true,
        description=f"Тренд + сезонность (NumPy): {n_per_group * n_groups} объектов, 2 признака, {n_groups} кластера",
        category="habr",
        source="https://habr.com/ru/companies/netologyru/articles/841952/",
    )


def make_sklearn_blobs_features(
    n_objects: int = 300,
    n_features: int = 2,
    n_clusters: int = 3,
    cluster_std: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> ClusteringDataset:
    rng = np.random.default_rng(2) if rng is None else rng
    seed = int(rng.integers(0, 2**31 - 1))
    X, y_true = make_blobs(
        n_samples=n_objects,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=seed,
    )
    return ClusteringDataset(
        name="habr_sklearn_blobs",
        X=X.astype(np.float64),
        y_true=y_true.astype(np.int64),
        description=f"make_blobs: {n_objects} объектов, {n_features} признака, {n_clusters} кластера",
        category="habr",
        source="sklearn.datasets.make_blobs",
    )



def make_sklearn_regression_style_features(
    n_per_group: int = 50,
    n_groups: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> ClusteringDataset:
    rng = np.random.default_rng(4) if rng is None else rng
    configs = [
        (0.5, 0.0,  0.5),
        (0.5, 10.0, 0.5),
        (0.5, 20.0, 0.5),
    ]
    groups, labs = [], []
    for g, (slope, bias, noise) in enumerate(configs):
        x = rng.uniform(0, 10, n_per_group)
        y = slope * x + bias + rng.normal(0, noise, n_per_group)
        groups.append(np.column_stack([x, y]))
        labs.append(np.full(n_per_group, g))
    X, y_true = _stack_groups(groups, labs)
    return ClusteringDataset(
        name="habr_sklearn_regression_style",
        X=X,
        y_true=y_true,
        description=f"make_regression-style: {n_per_group * n_groups} объектов, 2 признака, {n_groups} кластера",
        category="habr",
        source="sklearn.datasets.make_regression",
    )


def make_scipy_mixed_distribution_features(
    n_per: int = 50,
    rng: Optional[np.random.Generator] = None,
) -> ClusteringDataset:
    rng = np.random.default_rng(5) if rng is None else rng
    g0 = rng.normal(0.0, 0.8, size=(n_per, 2))
    g1 = rng.uniform(-1.0, 1.0, size=(n_per, 2)) + np.array([8.0, 0.0])
    g2_raw = rng.exponential(0.5, size=(n_per, 2))
    g2 = g2_raw + np.array([4.0, 8.0])
    X, y_true = _stack_groups(
        [g0, g1, g2],
        [np.zeros(n_per, dtype=np.int64),
         np.ones(n_per, dtype=np.int64),
         np.full(n_per, 2, dtype=np.int64)],
    )
    return ClusteringDataset(
        name="habr_scipy_mixed",
        X=X,
        y_true=y_true,
        description=f"Normal / Uniform / Exponential (SciPy): {3 * n_per} объектов, 2 признака, 3 кластера",
        category="habr",
        source="scipy.stats",
    )


def all_habr_datasets(
    rng: Optional[np.random.Generator] = None,
) -> list[ClusteringDataset]:
    rng = np.random.default_rng(42) if rng is None else rng
    return [
        make_numpy_linear_features(rng=rng),
        make_numpy_timeseries_features(rng=rng),
        make_sklearn_blobs_features(rng=rng),
        make_sklearn_regression_style_features(rng=rng),
        make_scipy_mixed_distribution_features(rng=rng),
    ]
