from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Tuple

import numpy as np

from .schema import ClusteringDataset, embed_points_radial_landmarks

_SIPU_DIR = Path(__file__).resolve().parent.parent / "data" / "sipu"
_SIPU_NAMES = ["flame", "jain", "spiral", "aggregation", "r15", "d31"]


def load_sipu_shapes(sipu_dir: str | Path | None = None) -> list[ClusteringDataset]:
    base = Path(sipu_dir) if sipu_dir else _SIPU_DIR
    datasets = []
    for name in _SIPU_NAMES:
        data_path   = base / f"{name}.data.gz"
        labels_path = base / f"{name}.labels0.gz"
        if not data_path.exists() or not labels_path.exists():
            raise FileNotFoundError(
                f"SIPU file not found: {data_path}\n"
                f"Download from https://github.com/gagolews/clustering-data-v1/tree/master/sipu"
            )
        X = np.loadtxt(gzip.open(data_path))
        y = np.loadtxt(gzip.open(labels_path), dtype=int) - 1
        lo, hi = X.min(axis=0), X.max(axis=0)
        X = (X - lo) / np.maximum(hi - lo, 1e-9)
        datasets.append(ClusteringDataset(
            name=f"shape_{name}",
            X=X.astype(np.float64),
            y_true=y,
            description=f"SIPU {name}: {len(y)} pts, {len(np.unique(y))} clusters",
            category="classic_2d",
            source="sipu",
            meta={"shape_kind": name},
        ))
    return datasets


def _normalize_xy(xy: np.ndarray) -> np.ndarray:
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return (xy - lo) / span


def make_spiral(
    n_per_arm: int = 100,
    n_arms: int = 3,
    a: float = 0.12,
    noise: float = 0.02,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0) if rng is None else rng
    pts = []
    lab = []
    theta_max = 2.6 * np.pi
    for arm in range(n_arms):
        theta = np.linspace(0, theta_max, n_per_arm)
        r = a * theta
        off = 2 * np.pi * arm / n_arms
        x = r * np.cos(theta + off) + rng.normal(0, noise, n_per_arm)
        y = r * np.sin(theta + off) + rng.normal(0, noise, n_per_arm)
        pts.append(np.column_stack([x, y]))
        lab.append(np.full(n_per_arm, arm))
    xy = np.vstack(pts)
    y = np.concatenate(lab)
    return xy, y.astype(np.int64)


def make_flame(
    n_upper: int = 120,
    n_lower: int = 120,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1) if rng is None else rng
    t = np.linspace(-1.2, 1.2, n_upper)
    xu = 0.9 * t + rng.normal(0, 0.04, n_upper)
    yu = 1.4 * (1.0 - t**2 * 0.35) + rng.normal(0, 0.05, n_upper)
    upper = np.column_stack([xu, yu])

    s = np.linspace(-1.0, 1.0, n_lower // 2)
    xl1 = 0.35 * s + rng.normal(0, 0.03, len(s))
    yl1 = -0.2 * s**2 - 0.15 + rng.normal(0, 0.04, len(s))
    s2 = np.linspace(-0.4, 0.4, n_lower - len(s))
    xl2 = s2 + rng.normal(0, 0.02, len(s2))
    yl2 = -0.55 + 0.15 * np.sin(s2 * 6) + rng.normal(0, 0.03, len(s2))
    lower = np.column_stack([np.r_[xl1, xl2], np.r_[yl1, yl2]])

    xy = np.vstack([upper, lower])
    y = np.concatenate([np.zeros(n_upper), np.ones(n_lower)]).astype(np.int64)
    return xy, y


def make_jain(
    n_per_cluster: int = 180,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2) if rng is None else rng
    t1 = np.linspace(0, np.pi, n_per_cluster)
    r1 = 1.0
    x1 = r1 * np.cos(t1) + rng.normal(0, 0.03, n_per_cluster)
    y1 = r1 * np.sin(t1) - 0.25 + rng.normal(0, 0.03, n_per_cluster)
    t2 = np.linspace(0, np.pi, n_per_cluster)
    x2 = r1 * np.cos(t2 + np.pi) + 0.35 + rng.normal(0, 0.03, n_per_cluster)
    y2 = r1 * np.sin(t2 + np.pi) + 0.25 + rng.normal(0, 0.03, n_per_cluster)
    c0 = np.column_stack([x1, y1])
    c1 = np.column_stack([x2, y2])
    xy = np.vstack([c0, c1])
    y = np.concatenate(
        [np.zeros(n_per_cluster), np.ones(n_per_cluster)]
    ).astype(np.int64)
    return xy, y


def make_aggregation(
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3) if rng is None else rng
    centers = np.array(
        [
            [2.0, 5.5],
            [6.0, 8.0],
            [5.5, 3.5],
            [9.0, 2.5],
            [3.0, 2.5],
            [7.0, 6.0],
            [9.0, 8.5],
        ]
    )
    counts = [60, 80, 100, 90, 110, 130, 218]
    stds = [0.35, 0.45, 0.5, 0.4, 0.55, 0.45, 0.65]
    pts = []
    labs = []
    for k, (mu, n, sd) in enumerate(zip(centers, counts, stds)):
        p = rng.normal(mu, sd, size=(n, 2))
        pts.append(p)
        labs.append(np.full(n, k))
    xy = np.vstack(pts)
    y = np.concatenate(labs).astype(np.int64)
    return xy, y


def make_r15(
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4) if rng is None else rng
    K = 15
    R = 3.2
    counts = [40] * K
    noise = 0.18
    pts = []
    labs = []
    for k in range(K):
        ang = 2 * np.pi * k / K + 0.12 * rng.standard_normal()
        cx = R * np.cos(ang) + 5.0 + 0.25 * rng.standard_normal()
        cy = R * np.sin(ang) + 5.0 + 0.25 * rng.standard_normal()
        n = counts[k]
        p = rng.normal([cx, cy], noise, size=(n, 2))
        pts.append(p)
        labs.append(np.full(n, k))
    xy = np.vstack(pts)
    y = np.concatenate(labs).astype(np.int64)
    return xy, y


def make_d31(
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5) if rng is None else rng
    xs = np.linspace(1.0, 9.0, 7)
    ys = np.linspace(1.0, 6.5, 5)
    gx, gy = np.meshgrid(xs, ys)
    flat = np.column_stack([gx.ravel(), gy.ravel()])
    centers = flat[:31]
    centers += rng.normal(0, 0.08, centers.shape)
    n_per = 100
    noise = 0.22
    pts = []
    labs = []
    for k in range(31):
        p = rng.normal(centers[k], noise, size=(n_per, 2))
        pts.append(p)
        labs.append(np.full(n_per, k))
    xy = np.vstack(pts)
    y = np.concatenate(labs).astype(np.int64)
    return xy, y


def build_shape_dataset(
    kind: str,
    n_profile_samples: int = 64,
    rng: np.random.Generator | None = None,
) -> ClusteringDataset:
    rng = np.random.default_rng(42) if rng is None else rng
    kind = kind.lower()
    if kind == "spiral":
        xy, yp = make_spiral(rng=rng)
        desc = "Три спирали Архимеда + шум"
    elif kind == "flame":
        xy, yp = make_flame(rng=rng)
        desc = "Два пламени (верх/низ), параметрическая модель"
    elif kind == "jain":
        xy, yp = make_jain(rng=rng)
        desc = "Две дуги + выбросы (Jain-style)"
    elif kind == "aggregation":
        xy, yp = make_aggregation(rng=rng)
        desc = "Семь гауссовых групп (Aggregation-style)"
    elif kind == "r15":
        xy, yp = make_r15(rng=rng)
        desc = "15 групп на кольце (R15-style)"
    elif kind == "d31":
        xy, yp = make_d31(rng=rng)
        desc = "31 группа на сетке (D31-style)"
    else:
        raise ValueError(f"Unknown shape kind: {kind}")

    xy_n = _normalize_xy(xy)
    X = embed_points_radial_landmarks(xy_n, n_profile_samples, rng)

    return ClusteringDataset(
        name=f"shape_{kind}",
        X=X,
        y_true=yp,
        description=desc,
        category="classic_2d_embedded",
        source="algorithmic (compare with Clustering-Datasets references)",
        meta={
            "X_point": xy,
            "y_point": yp,
            "shape_kind": kind,
            "embedding": "radial_landmarks",
        },
    )


def all_classic_embedded(
    n_profile_samples: int = 64,
    rng: np.random.Generator | None = None,
) -> list[ClusteringDataset]:
    rng = np.random.default_rng(42) if rng is None else rng
    names = ["spiral", "flame", "jain", "aggregation", "r15", "d31"]
    return [build_shape_dataset(n, n_profile_samples=n_profile_samples, rng=rng) for n in names]


def build_shape_dataset_2d(
    kind: str,
    rng: np.random.Generator | None = None,
) -> ClusteringDataset:
    rng = np.random.default_rng(42) if rng is None else rng
    kind = kind.lower()
    makers = {
        "spiral":      (make_spiral,      "Три спирали Архимеда"),
        "flame":       (make_flame,        "Два кластера (Flame)"),
        "jain":        (make_jain,         "Две дуги (Jain)"),
        "aggregation": (make_aggregation,  "7 гауссовых групп (Aggregation)"),
        "r15":         (make_r15,          "15 групп на кольце (R15)"),
        "d31":         (make_d31,          "31 группа на сетке (D31)"),
    }
    if kind not in makers:
        raise ValueError(f"Unknown shape kind: {kind}")
    fn, desc = makers[kind]
    xy, yp = fn(rng=rng)
    xy = _normalize_xy(xy)
    return ClusteringDataset(
        name=f"shape_{kind}",
        X=xy,
        y_true=yp,
        description=desc,
        category="classic_2d",
        source="algorithmic",
        meta={"shape_kind": kind},
    )


def all_classic_2d(
    rng: np.random.Generator | None = None,
) -> list[ClusteringDataset]:
    rng = np.random.default_rng(42) if rng is None else rng
    return [build_shape_dataset_2d(k, rng=rng) for k in
            ["flame", "jain", "spiral", "aggregation", "r15", "d31"]]
