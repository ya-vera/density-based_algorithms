from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class ClusteringDataset:
    name: str
    X: np.ndarray
    y_true: np.ndarray
    description: str = ""
    category: str = ""
    source: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features)")
        if self.y_true.shape != (self.X.shape[0],):
            raise ValueError("y_true must have shape (n_samples,)")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "X": self.X,
            "y_true": self.y_true,
            "description": self.description,
            "category": self.category,
            "source": self.source,
        }
        d.update(self.meta)
        return d


FeatureClusteringDataset = ClusteringDataset


def default_openml_cache() -> str:
    from pathlib import Path

    return str(Path(__file__).resolve().parents[1] / ".openml_cache")


def embed_points_radial_landmarks(
    xy: np.ndarray,
    n_features: int,
    rng: np.random.Generator,
    sigma: float = 0.12,
    noise: float = 0.02,
) -> np.ndarray:
    if n_features < 4:
        raise ValueError("n_features must be >= 4")
    side = int(np.ceil(np.sqrt(n_features)))
    g = np.linspace(0.05, 0.95, side)
    gx, gy = np.meshgrid(g, g)
    landmarks = np.column_stack([gx.ravel(), gy.ravel()])
    if landmarks.shape[0] > n_features:
        idx = rng.choice(landmarks.shape[0], size=n_features, replace=False)
        idx.sort()
        L = landmarks[idx]
    else:
        L = landmarks[:n_features]
    n = xy.shape[0]
    out = np.empty((n, L.shape[0]))
    for i in range(n):
        d2 = np.sum((L - xy[i]) ** 2, axis=1)
        out[i] = np.exp(-d2 / (2 * sigma * sigma))
    out += rng.normal(0, noise, size=out.shape)
    return out
