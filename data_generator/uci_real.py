from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import LabelEncoder

from .schema import ClusteringDataset, default_openml_cache


def _as_sample_arrays(raw_X, raw_y) -> tuple[np.ndarray, np.ndarray]:
    X = raw_X.values if hasattr(raw_X, "values") else np.asarray(raw_X, dtype=float)
    y = raw_y.values.ravel() if hasattr(raw_y, "values") else np.asarray(raw_y).ravel()
    if y.dtype == object or str(y.dtype).startswith("str") or y.dtype.name == "category":
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = np.asarray(y, dtype=np.int64)
    return X.astype(np.float64), y.astype(np.int64)


def _tabular_to_ds(
    name: str,
    X_samples: np.ndarray,
    y_sample: np.ndarray,
    description: str,
    source: str,
) -> ClusteringDataset:
    return ClusteringDataset(
        name=name,
        X=X_samples.copy(),
        y_true=y_sample,
        description=description,
        category="uci_real",
        source=source,
    )


def load_iris_fc() -> ClusteringDataset:
    d = load_iris()
    Xs, ys = _as_sample_arrays(pd.DataFrame(d.data), pd.Series(d.target))
    return _tabular_to_ds(
        "uci_iris",
        Xs,
        ys,
        "Iris Fisher 1936, 4 признака, 150 объектов, 3 класса",
        "https://archive.ics.uci.edu/dataset/53/iris",
    )


def load_wine_fc() -> ClusteringDataset:
    d = load_wine()
    Xs, ys = _as_sample_arrays(pd.DataFrame(d.data), pd.Series(d.target))
    return _tabular_to_ds(
        "uci_wine",
        Xs,
        ys,
        "Wine chemical analysis, 13 признаков, 178 объектов, 3 класса",
        "https://archive.ics.uci.edu/dataset/109/wine",
    )


def _load_ecoli_raw() -> tuple[np.ndarray, np.ndarray]:
    import io
    import urllib.request

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    rows = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) == 9:
            rows.append(parts)
    df = pd.DataFrame(rows, columns=["seq", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "cls"])
    X = df[["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]].astype(float).values
    y = LabelEncoder().fit_transform(df["cls"].values)
    return X, y.astype(np.int64)


def load_ecoli_fc(
    data_home: Optional[str] = None,
    min_class_size: int = 10,
) -> ClusteringDataset:
    try:
        from sklearn.datasets import fetch_openml
        home = data_home or default_openml_cache()
        raw = fetch_openml(name="ecoli", version=1, parser="auto", data_home=home)
        Xs, ys = _as_sample_arrays(raw.data, raw.target)
    except Exception:
        Xs, ys = _load_ecoli_raw()

    counts = np.bincount(ys)
    keep_labels = np.where(counts >= min_class_size)[0]
    mask = np.isin(ys, keep_labels)
    Xs, ys = Xs[mask], ys[mask]

    label_map = {old: new for new, old in enumerate(keep_labels)}
    ys = np.array([label_map[v] for v in ys], dtype=np.int64)

    k = len(keep_labels)
    return _tabular_to_ds(
        "uci_ecoli",
        Xs,
        ys,
        f"E.coli protein localisation, 7 признаков, {len(ys)} объектов, {k} классов",
        "https://archive.ics.uci.edu/dataset/39/ecoli",
    )


def load_seeds_fc(
    data_home: Optional[str] = None,
) -> ClusteringDataset:
    from sklearn.datasets import fetch_openml

    home = data_home or default_openml_cache()
    raw = fetch_openml(name="seeds", version=1, parser="auto", data_home=home)
    Xs, ys = _as_sample_arrays(raw.data, raw.target)
    return _tabular_to_ds(
        "uci_seeds",
        Xs,
        ys,
        "Seeds wheat kernels, 7 признаков, 210 объектов, 3 сорта",
        "https://archive.ics.uci.edu/dataset/236/seeds",
    )


def load_segment_fc(
    data_home: Optional[str] = None,
) -> ClusteringDataset:
    from sklearn.datasets import fetch_openml

    home = data_home or default_openml_cache()
    raw = fetch_openml(name="segment", version=1, parser="auto", data_home=home)
    Xs, ys = _as_sample_arrays(raw.data, raw.target)
    return _tabular_to_ds(
        "uci_statlog_segment",
        Xs,
        ys,
        "Statlog image segmentation, 19 признаков, 2310 объектов, 7 классов",
        "https://archive.ics.uci.edu/ml/datasets/Statlog+(Image+Segmentation)",
    )


def all_uci_datasets(data_home: Optional[str] = None) -> list[ClusteringDataset]:
    return [
        load_iris_fc(),
        load_wine_fc(),
        load_ecoli_fc(data_home=data_home),
        load_seeds_fc(data_home=data_home),
        load_segment_fc(data_home=data_home),
    ]
