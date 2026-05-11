from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.registry import get_algorithm, list_algorithms
from data_generator.generator_pdf import build_generator_pdf_mixture
from data_generator.registry import load_data_generator_dataset
from evaluation.metrics import ClusteringMetrics
from visualization.plots import plot_cluster_projection


MANIFEST_FIELDS = [
    "timestamp_utc",
    "dataset_id",
    "algorithm_id",
    "n_features",
    "n_samples",
    "runtime_sec",
    "noise_fraction",
    "n_clusters",
    "ari",
    "nmi",
    "silhouette",
    "rel_metrics_json",
    "rel_figure_pca",
    "rel_metrics_file",
    "error",
]


def _load_dataset(dataset_id: str, rng: np.random.Generator, args: argparse.Namespace) -> Dict[str, Any]:
    if dataset_id == "generator_pdf":
        ds = build_generator_pdf_mixture(
            N=args.N,
            V=args.V,
            k=args.k,
            alpha=args.alpha,
            nmin=args.nmin,
            rng=rng,
            name="generator_pdf_mixture",
        )
        return ds.to_dict()
    reg_key = {"generator_pdf": "generator_pdf_mixture"}.get(dataset_id, dataset_id)
    dg = load_data_generator_dataset(reg_key, n_samples=args.n_profile, rng_seed=args.seed)
    if dg is not None:
        return dg
    raise ValueError(
        f"Unknown dataset_id: {dataset_id!r}. "
        f"Built-in generator: generator_pdf. Other ids: keys from data_generator.registry.DATA_GENERATOR_BUILTIN_KEYS."
    )


def _append_manifest(root: Path, row: Dict[str, Any]) -> None:
    path = root / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def _json_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif isinstance(v, dict):
            out[k] = _json_safe(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def _csv_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return ""
    return str(v)


def run_batch(root: Path, dataset_id: str, algorithms: List[str], args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    batch_ts = datetime.now(timezone.utc).isoformat()
    ds_dict = _load_dataset(dataset_id, rng, args)
    X = np.asarray(ds_dict["X"], dtype=float)
    y_true = ds_dict.get("y_true")
    if y_true is not None:
        y_true = np.asarray(y_true)

    base = root / "datasets" / dataset_id
    base.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset_id": dataset_id,
        "timestamp_utc": batch_ts,
        "seed": args.seed,
        "X_shape": list(X.shape),
        "source": ds_dict.get("source", ""),
        "description": ds_dict.get("description", ""),
        "params": {
            "N": args.N,
            "V": args.V,
            "k": args.k,
            "alpha": args.alpha,
            "nmin": args.nmin,
            "n_profile": args.n_profile,
        },
    }
    (base / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for alg_name in algorithms:
        alg_dir = base / alg_name
        alg_dir.mkdir(parents=True, exist_ok=True)
        err: Optional[str] = None
        labels: Optional[np.ndarray] = None
        runtime = float("nan")
        metrics_flat: Dict[str, Any] = {}

        t0 = time.perf_counter()
        try:
            alg = get_algorithm(alg_name)()
            labels = np.asarray(alg.fit_predict(X), dtype=int)
        except Exception as e:
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        else:
            runtime = time.perf_counter() - t0
            cm = ClusteringMetrics(labels=labels, y_true=y_true, X=X, noise_label=-1)
            metrics_flat = {
                "external": cm.external(),
                "internal": cm.internal(),
                "structure": cm.structure(),
            }

        fig_rel = f"datasets/{dataset_id}/{alg_name}/fig_pca.png"
        json_rel = f"datasets/{dataset_id}/{alg_name}/metrics.json"

        if err is None and labels is not None:
            title = f"{alg_name} | {dataset_id} | PCA (features)"
            try:
                fig = plot_cluster_projection(
                    X, labels, method="pca", title=title, figsize=(6.5, 5.2)
                )
                fig.savefig(alg_dir / "fig_pca.png", dpi=150, bbox_inches="tight")
                import matplotlib.pyplot as plt

                plt.close(fig)
            except Exception as e:
                err = (err or "") + f"\nplot: {type(e).__name__}: {e}"

        payload = {
            "dataset_id": dataset_id,
            "algorithm_id": alg_name,
            "runtime_sec": runtime,
            "metrics": metrics_flat,
            "error": err,
        }
        (alg_dir / "metrics.json").write_text(
            json.dumps(_json_safe(payload), indent=2), encoding="utf-8"
        )

        struct = metrics_flat.get("structure", {}) if metrics_flat else {}
        ext = metrics_flat.get("external", {}) if metrics_flat else {}
        inn = metrics_flat.get("internal", {}) if metrics_flat else {}

        row = {
            "timestamp_utc": batch_ts,
            "dataset_id": dataset_id,
            "algorithm_id": alg_name,
            "n_features": X.shape[0],
            "n_samples": X.shape[1],
            "runtime_sec": f"{runtime:.6f}" if err is None and np.isfinite(runtime) else "",
            "noise_fraction": _csv_val(struct.get("noise_fraction", "")),
            "n_clusters": _csv_val(struct.get("n_clusters", "")),
            "ari": _csv_val(ext.get("ari", "")),
            "nmi": _csv_val(ext.get("nmi", "")),
            "silhouette": _csv_val(inn.get("silhouette", "")),
            "rel_metrics_json": json_rel,
            "rel_figure_pca": fig_rel if err is None and labels is not None else "",
            "rel_metrics_file": json_rel,
            "error": (err or "").replace("\n", " | ")[:2000],
        }
        _append_manifest(root, row)
        print(f"[ok] {dataset_id} / {alg_name}  runtime={row['runtime_sec']}  k={row['n_clusters']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Paper-style batch: one dataset × all algorithms.")
    p.add_argument("--root", type=Path, default=ROOT / "experiment_runs", help="Output root")
    p.add_argument("--dataset", type=str, default="generator_pdf", help="Dataset id")
    p.add_argument(
        "--algorithms",
        type=str,
        default="",
        help="Comma-separated algorithm names; empty = all registered",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--N", type=int, default=1000)
    p.add_argument("--V", type=int, default=15)
    p.add_argument("--k", type=int, default=7)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--nmin", type=int, default=20)
    p.add_argument(
        "--n-profile",
        type=int,
        default=64,
        help="For registry datasets (not generator_pdf): embedding / column count passed to load_data_generator_dataset",
    )
    args = p.parse_args()

    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    if args.algorithms.strip():
        algorithms = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]
    else:
        algorithms = list_algorithms()

    print(f"Root: {root}")
    print(f"Dataset: {args.dataset}")
    print(f"Algorithms ({len(algorithms)}): {', '.join(algorithms)}")
    run_batch(root, args.dataset, algorithms, args)


if __name__ == "__main__":
    main()
