from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

import numpy as np


_ALG_KEY_ATTRS = {
    "dbscan": ["eps_used_", "min_samples"],
    "hdbscan": ["min_cluster_size", "min_samples"],
    "dpc": ["dc_used_", "percent"],
    "rd_dac": ["k_used_"],
    "ckdpc": ["dc_used_", "alpha", "k_neighbors"],
}


def extract_inst_params(inst, alg_name: str) -> dict:
    import inspect as insp

    keys = _ALG_KEY_ATTRS.get(alg_name, [])
    if keys:
        out = {}
        for attr in keys:
            val = getattr(inst, attr, None)
            if val is not None:
                dk = attr.rstrip("_").replace("eps_used", "eps").replace("dc_used", "dc")
                out[dk] = round(val, 4) if isinstance(val, float) else val
        return out
    try:
        sig = insp.signature(inst.__class__.__init__)
        out = {}
        for k, p in sig.parameters.items():
            if k == "self":
                continue
            if p.kind in (insp.Parameter.VAR_POSITIONAL, insp.Parameter.VAR_KEYWORD):
                continue
            if hasattr(inst, k) and getattr(inst, k) is not None:
                out[k] = getattr(inst, k)
            elif p.default is not insp.Parameter.empty and p.default is not None:
                out[k] = p.default
        return out
    except Exception:
        return {}


def adaptive_params(alg_name: str, X: np.ndarray) -> dict:
    from scipy.spatial.distance import pdist as spd_pdist
    from sklearn.preprocessing import MinMaxScaler as MMS

    from algorithms.density_params import auto_eps_from_knn

    n, d = X.shape
    X_sc = MMS().fit_transform(X)
    if alg_name == "dbscan":
        k = min(max(3, d + 1), 15)
        try:
            eps = float(auto_eps_from_knn(X_sc, k=k))
        except Exception:
            eps = float(np.percentile(spd_pdist(X_sc), 2))
        return {"eps": round(eps, 4), "min_samples": max(2, int(np.log(max(n, 2))))}
    if alg_name == "hdbscan":
        return {"min_cluster_size": max(5, n // 50), "min_samples": max(3, int(np.log(max(n, 2))))}
    if alg_name == "dpc":
        return {"dc": round(float(np.percentile(spd_pdist(MMS().fit_transform(X)), 2)), 4)}
    if alg_name == "rd_dac":
        k_rec = max(3, min(15, int(np.log(max(n, 2)) * 1.5)))
        return {"k": k_rec, "min_k": max(2, k_rec // 3)}
    if alg_name == "ckdpc":
        return {"percent": 2.0, "alpha": 0.5, "k_neighbors": 7}
    return {}


def score_labeling(lbl: np.ndarray, X: np.ndarray, y_true: Optional[np.ndarray]) -> float:
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import MinMaxScaler as MMS

    mask = lbl != -1
    k = len(set(lbl[mask].tolist())) if mask.any() else 0
    if k < 2:
        return float("-inf")
    try:
        if y_true is not None:
            return float(adjusted_rand_score(y_true, lbl))
        return float(silhouette_score(MMS().fit_transform(X[mask]), lbl[mask]))
    except Exception:
        return float("-inf")


def auto_best_params(reg_name: str, X: np.ndarray, y_true: Optional[np.ndarray]) -> dict:
    from algorithms.base import AlgorithmRegistry

    cls = AlgorithmRegistry.get(reg_name)
    p_rec = adaptive_params(reg_name, X)
    try:
        lbl_rec = np.asarray(cls(**p_rec).fit_predict(X), dtype=int)
        s_rec = score_labeling(lbl_rec, X, y_true)
    except Exception:
        s_rec = float("-inf")
    try:
        inst = cls()
        lbl_auto = np.asarray(inst.fit_predict(X), dtype=int)
        p_auto = extract_inst_params(inst, reg_name)
        s_auto = score_labeling(lbl_auto, X, y_true)
    except Exception:
        p_auto, s_auto = {}, float("-inf")
    return p_rec if s_rec >= s_auto else p_auto


def builtin_fit_predict_callable(
    reg_name: str, X: np.ndarray, y_true: Optional[np.ndarray]
) -> Callable[[np.ndarray], np.ndarray]:
    from algorithms.base import AlgorithmRegistry

    reg_cls = AlgorithmRegistry.get(reg_name)
    mp = auto_best_params(reg_name, X, y_true)
    return lambda Z, _c=reg_cls, _p=mp: np.asarray(_c(**_p).fit_predict(Z), dtype=int)


def user_class_fit_predict_callable(
    cls: Type[Any],
    X: np.ndarray,
    y_true: Optional[np.ndarray],
    explicit_params: Optional[Dict] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if explicit_params:
        p = dict(explicit_params)
        return lambda Z, _c=cls, _p=p: np.asarray(_c(**_p).fit_predict(Z), dtype=int)
    return lambda Z, _c=cls: np.asarray(_c().fit_predict(Z), dtype=int)


__all__ = [
    "adaptive_params",
    "auto_best_params",
    "builtin_fit_predict_callable",
    "extract_inst_params",
    "score_labeling",
    "user_class_fit_predict_callable",
]
