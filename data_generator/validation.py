from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def load_reference_xy_labels(
    path: str | Path,
    delimiter: str = ",",
    has_header: bool = False,
    label_col: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    path = Path(path)
    data = np.loadtxt(path, delimiter=delimiter, skiprows=1 if has_header else 0)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xy = data[:, :2].astype(np.float64)
    labels = None
    if label_col is not None and data.shape[1] > label_col:
        labels = data[:, label_col].astype(np.int64)
    elif data.shape[1] >= 3 and label_col is None:
        labels = data[:, 2].astype(np.int64)
    return xy, labels


def _mean_nn_distance(A: np.ndarray, B: np.ndarray) -> float:
    if cKDTree is None:
        raise ImportError("scipy required for validation.compare_point_clouds")
    tree = cKDTree(B)
    d, _ = tree.query(A, k=1)
    return float(np.mean(d))


def compare_point_clouds(
    gen_xy: np.ndarray,
    ref_xy: np.ndarray,
    gen_labels: Optional[np.ndarray] = None,
    ref_labels: Optional[np.ndarray] = None,
    normalize: bool = True,
) -> Dict[str, Any]:
    g = np.asarray(gen_xy, dtype=np.float64).copy()
    r = np.asarray(ref_xy, dtype=np.float64).copy()
    if normalize:
        lo, hi = r.min(axis=0), r.max(axis=0)
        span = np.maximum(hi - lo, 1e-9)
        r = (r - lo) / span
        g = (g - lo) / span

    d_gr = _mean_nn_distance(g, r)
    d_rg = _mean_nn_distance(r, g)
    out: Dict[str, Any] = {
        "mean_min_dist_gen_to_ref": d_gr,
        "mean_min_dist_ref_to_gen": d_rg,
        "symmetric_chamfer_mean": 0.5 * (d_gr + d_rg),
        "n_gen": g.shape[0],
        "n_ref": r.shape[0],
    }

    if gen_labels is not None and ref_labels is not None and g.shape[0] == r.shape[0]:
        try:
            from sklearn.metrics import adjusted_rand_score
        except ImportError:
            adjusted_rand_score = None
        if adjusted_rand_score is not None and cKDTree is not None:
            tree = cKDTree(r)
            _, idx = tree.query(g, k=1)
            matched_ref_labels = ref_labels[idx]
            out["ari_labels_nn_match"] = float(
                adjusted_rand_score(gen_labels, matched_ref_labels)
            )
    return out


def validate_shape_against_file(
    gen_xy: np.ndarray,
    ref_path: str | Path,
    gen_labels: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    ref_xy, ref_labels = load_reference_xy_labels(ref_path, **kwargs)
    return compare_point_clouds(gen_xy, ref_xy, gen_labels, ref_labels)
