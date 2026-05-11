from __future__ import annotations

from typing import Optional

import numpy as np

from .generator import generdat_fast
from .schema import ClusteringDataset


def build_generator_pdf_mixture(
    N: int = 1000,
    V: int = 15,
    k: int = 7,
    alpha: float = 0.25,
    nmin: int = 20,
    rng: Optional[np.random.Generator] = None,
    use_gpu: bool = False,
    name: str = "generator_pdf_mixture",
) -> ClusteringDataset:
    rng = np.random.default_rng(42) if rng is None else rng
    Nk, R, rf, X_nv, cen = generdat_fast(
        N=N, V=V, k=k, alpha=alpha, nmin=nmin, rng=rng, use_gpu=use_gpu
    )
    X = X_nv.astype(np.float64, copy=False)
    y_true = (rf - 1).astype(np.int64)
    return ClusteringDataset(
        name=name,
        X=X,
        y_true=y_true,
        description=f"Гауссова смесь (Generator.pdf), N={N}, V={V}, k={k}",
        category="generator_pdf",
        source="data_generator/generator.py",
        meta={
            "Nk": Nk,
            "R": R,
            "cen": cen,
            "alpha": alpha,
            "true_k": k,
        },
    )
