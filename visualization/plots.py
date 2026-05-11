from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        import warnings
        warnings.filterwarnings(
            "ignore",
            message="Glyph.*missing from font",
            category=UserWarning,
        )
        return plt, matplotlib
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. Install it with: pip install matplotlib"
        )


def _clean_label(name: str) -> str:
    return name.replace("[U] ", "[U] ")


def plot_coassociation(
    C_norm: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Co-association Matrix",
    figsize: tuple = (7, 6),
):

    plt, mpl = _require_matplotlib()

    if labels is not None:
        order = np.argsort(labels)
        C_plot = C_norm[np.ix_(order, order)]
    else:
        C_plot = C_norm

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(C_plot, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Co-association")
    ax.set_title(title)
    ax.set_xlabel("Object index (sorted by cluster)")
    ax.set_ylabel("Object index (sorted by cluster)")

    if labels is not None:
        # Draw cluster boundaries
        sorted_labels = labels[np.argsort(labels)]
        unique, counts = np.unique(sorted_labels[sorted_labels >= 0], return_counts=True)
        boundaries = np.cumsum(counts)[:-1] - 0.5
        for b in boundaries:
            ax.axhline(b, color="red", linewidth=0.8, alpha=0.8)
            ax.axvline(b, color="red", linewidth=0.8, alpha=0.8)

    fig.tight_layout()
    return fig


def plot_feature_scores(
    scores: np.ndarray,
    labels: Optional[np.ndarray] = None,
    selected_indices: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Feature Stability Scores",
    figsize: tuple = (10, 4),
):
    plt, mpl = _require_matplotlib()

    n = len(scores)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        cmap = mpl.colormaps.get_cmap("tab10")
        unique_labels = np.unique(labels[labels >= 0])
        colors = [
            cmap(int(labels[i]) % 10) if labels[i] >= 0 else (0.7, 0.7, 0.7, 1.0)
            for i in range(n)
        ]
        bars = ax.bar(x, scores, color=colors)
    else:
        bars = ax.bar(x, scores, color="steelblue")

    if selected_indices is not None:
        for idx in selected_indices:
            bars[idx].set_edgecolor("red")
            bars[idx].set_linewidth(2)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="threshold=0.5")
    ax.set_ylabel("Stability score")
    ax.set_xlabel("Feature index")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    if feature_names is not None and len(feature_names) <= 40:
        ax.set_xticks(x)
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)

    fig.tight_layout()
    return fig


def plot_algorithm_comparison(
    ari_dict: Dict[str, float],
    noise_dict: Optional[Dict[str, float]] = None,
    weight_dict: Optional[Dict[str, float]] = None,
    title: str = "Algorithm Comparison",
    figsize: tuple = (9, 4),
):

    plt, _ = _require_matplotlib()

    names = list(ari_dict.keys())
    clean_names = [_clean_label(n) for n in names]   # strip emoji
    x = np.arange(len(names))

    n_groups = 1 + (noise_dict is not None) + (weight_dict is not None)
    width = 0.8 / n_groups
    offset = -(n_groups - 1) / 2 * width

    fig, ax = plt.subplots(figsize=figsize)

    ari_vals = [ari_dict[n] for n in names]
    ax.bar(x + offset, ari_vals, width, label="ARI", color="steelblue")
    offset += width

    if noise_dict is not None:
        noise_vals = [noise_dict.get(n, 0.0) for n in names]
        ax.bar(x + offset, noise_vals, width, label="Noise fraction", color="salmon")
        offset += width

    if weight_dict is not None:
        w_vals = [weight_dict.get(n, 1.0) / 2.0 for n in names]
        ax.bar(x + offset, w_vals, width, label="Run weight / 2", color="mediumseagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(clean_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_benchmark_heatmap(
    ari_matrix: np.ndarray,
    algorithm_names: List[str],
    dataset_names: List[str],
    title: str = "ARI Benchmark Heatmap",
    figsize: tuple = (8, 5),
):

    plt, _ = _require_matplotlib()

    clean_alg   = [_clean_label(n) for n in algorithm_names]  # strip emoji
    clean_ds    = [_clean_label(n) for n in dataset_names]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(ari_matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="ARI")

    ax.set_xticks(np.arange(len(clean_ds)))
    ax.set_yticks(np.arange(len(clean_alg)))
    ax.set_xticklabels(clean_ds, rotation=30, ha="right")
    ax.set_yticklabels(clean_alg)
    ax.set_title(title)

    # Annotate cells
    for i in range(len(algorithm_names)):
        for j in range(len(dataset_names)):
            val = ari_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.4 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_color)

    fig.tight_layout()
    return fig


def plot_cluster_projection(
    X: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    title: str = "Object Cluster Projection",
    figsize: tuple = (6, 5),
):
    plt, mpl = _require_matplotlib()
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA

    X_sc = MinMaxScaler().fit_transform(np.asarray(X, dtype=float))

    if X_sc.shape[1] == 2:
        proj = X_sc
        xlabel, ylabel = "X", "Y"
        x0, x1, y0, y1 = -0.05, 1.05, -0.05, 1.05
        use_01_ticks = True
    else:
        proj = PCA(n_components=2).fit_transform(X_sc)
        xlabel, ylabel = "PC1", "PC2"
        pad = 0.05
        x0 = proj[:, 0].min() - pad * (proj[:, 0].max() - proj[:, 0].min() + 1e-9)
        x1 = proj[:, 0].max() + pad * (proj[:, 0].max() - proj[:, 0].min() + 1e-9)
        y0 = proj[:, 1].min() - pad * (proj[:, 1].max() - proj[:, 1].min() + 1e-9)
        y1 = proj[:, 1].max() + pad * (proj[:, 1].max() - proj[:, 1].min() + 1e-9)
        use_01_ticks = False

    fig, ax = plt.subplots(figsize=figsize)
    cmap = mpl.colormaps.get_cmap("tab10")

    unique_labels = np.unique(labels)
    for c in unique_labels:
        mask = labels == c
        color = (0.55, 0.55, 0.55) if c == -1 else cmap(int(c) % 10)
        label_str = "шум" if c == -1 else f"кластер {c}"
        marker = "x" if c == -1 else "o"
        if c == -1:
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=[color], label=label_str,
                s=35, alpha=0.6, marker="x", linewidths=1.0,
            )
        else:
            ax.scatter(
                proj[mask, 0], proj[mask, 1],
                c=[color], label=label_str,
                s=55, alpha=0.75, marker="o",
                edgecolors="k", linewidths=0.3,
            )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    if use_01_ticks:
        ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    ax.legend(markerscale=1.2, fontsize=8, loc="best")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def plot_pac_curve(
    pac_scores: Dict[int, float],
    optimal_k: Optional[int] = None,
    k_method: str = "",
    title: str = "PAC / k-selection Curve",
    figsize: tuple = (6, 4),
):

    plt, _ = _require_matplotlib()

    ks = sorted(pac_scores.keys())
    vals = [pac_scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ks, vals, "o-", color="steelblue", linewidth=2, markersize=6)

    if optimal_k is not None and optimal_k in pac_scores:
        ax.axvline(optimal_k, color="red", linestyle="--", linewidth=1.2,
                   label=f"k={optimal_k} ({k_method})")
        ax.scatter([optimal_k], [pac_scores[optimal_k]],
                   color="red", s=100, zorder=5)
        ax.legend()

    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(ks)
    fig.tight_layout()
    return fig
