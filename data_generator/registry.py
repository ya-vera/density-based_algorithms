from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .classic_shapes import all_classic_embedded, build_shape_dataset
from .generator_pdf import build_generator_pdf_mixture
from .habr_synthetic import (
    all_habr_datasets,
    make_numpy_linear_features,
    make_numpy_timeseries_features,
    make_scipy_mixed_distribution_features,
    make_sklearn_blobs_features,
    make_sklearn_regression_style_features,
)
from .schema import ClusteringDataset, default_openml_cache
from .uci_real import (
    all_uci_datasets,
    load_iris_fc,
    load_segment_fc,
    load_seeds_fc,
    load_ecoli_fc,
    load_wine_fc,
)


_SHAPE_ALIASES: Dict[str, str] = {
    "spiral": "shape_spiral",
    "flame": "shape_flame",
    "jain": "shape_jain",
    "aggregation": "shape_aggregation",
    "r15": "shape_r15",
    "d31": "shape_d31",
}


def normalize_dataset_key(key: str) -> str:
    return _SHAPE_ALIASES.get(key, key)


DATA_GENERATOR_BUILTIN_KEYS: List[str] = [
    "generator_pdf_mixture",
    "shape_spiral",
    "shape_flame",
    "shape_jain",
    "shape_aggregation",
    "shape_r15",
    "shape_d31",
    "habr_numpy_linear",
    "habr_numpy_timeseries",
    "habr_sklearn_blobs",
    "habr_sklearn_regression_style",
    "habr_scipy_mixed",
    "uci_iris",
    "uci_wine",
    "uci_ecoli",
    "uci_seeds",
    "uci_statlog_segment",
    "spiral",
    "flame",
    "jain",
    "aggregation",
    "r15",
    "d31",
]


def load_data_generator_dataset(
    key: str,
    n_samples: int = 50,
    rng_seed: int = 42,
    openml_data_home: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    key = normalize_dataset_key(key)
    rng = np.random.default_rng(rng_seed)
    n_prof = int(np.clip(n_samples, 16, 256))
    n_feat = int(np.clip(n_samples, 20, 300))
    oml = openml_data_home or default_openml_cache()

    if key.startswith("shape_"):
        kind = key[6:]
        try:
            return build_shape_dataset(kind, n_profile_samples=n_prof, rng=rng).to_dict()
        except ValueError:
            return None

    if key == "generator_pdf_mixture":
        n_obj = int(np.clip(n_samples * 15, 300, 4000))
        return build_generator_pdf_mixture(N=n_obj, V=15, k=7, rng=rng).to_dict()

    habr: Dict[str, Callable[[], ClusteringDataset]] = {
        "habr_numpy_linear": lambda: make_numpy_linear_features(rng=rng),
        "habr_numpy_timeseries": lambda: make_numpy_timeseries_features(rng=rng),
        "habr_sklearn_blobs": lambda: make_sklearn_blobs_features(rng=rng),
        "habr_sklearn_regression_style": lambda: make_sklearn_regression_style_features(rng=rng),
        "habr_scipy_mixed": lambda: make_scipy_mixed_distribution_features(rng=rng),
    }
    if key in habr:
        return habr[key]().to_dict()

    uci: Dict[str, Callable[[], ClusteringDataset]] = {
        "uci_iris": load_iris_fc,
        "uci_wine": load_wine_fc,
        "uci_ecoli": lambda: load_ecoli_fc(data_home=oml),
        "uci_seeds": lambda: load_seeds_fc(data_home=oml),
        "uci_statlog_segment": lambda: load_segment_fc(data_home=oml),
    }
    if key in uci:
        return uci[key]().to_dict()

    return None


def build_all(
    n_profile_samples: int = 64,
    n_habr_features: int = 60,
    openml_data_home: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[ClusteringDataset]:
    rng = np.random.default_rng(42) if rng is None else rng
    out: List[ClusteringDataset] = []
    out.append(build_generator_pdf_mixture(rng=rng))
    out.extend(all_habr_datasets(rng=rng))
    out.extend(all_classic_embedded(n_profile_samples=n_profile_samples, rng=rng))
    out.extend(all_uci_datasets(data_home=openml_data_home))
    return out


def registry_functions() -> Dict[str, Callable[..., ClusteringDataset]]:
    reg: Dict[str, Callable[..., ClusteringDataset]] = {
        "generator_pdf_mixture": build_generator_pdf_mixture,
        "shape_spiral": lambda **kw: build_shape_dataset("spiral", **kw),
        "shape_flame": lambda **kw: build_shape_dataset("flame", **kw),
        "shape_jain": lambda **kw: build_shape_dataset("jain", **kw),
        "shape_aggregation": lambda **kw: build_shape_dataset("aggregation", **kw),
        "shape_r15": lambda **kw: build_shape_dataset("r15", **kw),
        "shape_d31": lambda **kw: build_shape_dataset("d31", **kw),
        "uci_iris": load_iris_fc,
        "uci_wine": load_wine_fc,
        "uci_ecoli": load_ecoli_fc,
        "uci_seeds": load_seeds_fc,
        "uci_statlog_segment": load_segment_fc,
    }
    return reg
