import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def _sample_cluster_sizes_fast(N: int, k: int, nmin: int, rng: np.random.Generator) -> np.ndarray:
    if k * nmin > N:
        raise ValueError(f"k * nmin = {k * nmin} > N = {N}")
    remainder = N - k * nmin
    extras = rng.multinomial(remainder, np.full(k, 1.0 / k))
    return extras + nmin


def generdat_fast(
    N: int,
    V: int,
    k: int,
    alpha: float,
    nmin: int,
    rng: Optional[np.random.Generator] = None,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng() if rng is None else rng

    Nk = _sample_cluster_sizes_fast(N, k, nmin, rng)
    counts = Nk.astype(int)

    starts = np.cumsum(np.r_[0, counts[:-1]])
    ends = np.cumsum(counts)
    R = [np.arange(s + 1, e + 1) for s, e in zip(starts, ends)]
    rf = np.repeat(np.arange(1, k + 1), counts)

    if use_gpu:
        try:
            import cupy as cp
            cp.random.seed(int(rng.integers(0, 2**31 - 1)))
            cen_gpu = (alpha - 1) + 2 * (1 - alpha) * cp.random.random((k, V))
            blocks = []
            for i, nk in enumerate(counts):
                sig = 0.05 + 0.05 * cp.random.random(V)
                Xk = cp.random.standard_normal((nk, V))
                Xk = Xk * sig + cen_gpu[i]
                blocks.append(Xk)
            X = cp.asnumpy(cp.vstack(blocks))
            cen = cp.asnumpy(cen_gpu)
        except ImportError:
            raise ImportError("Для use_gpu=True требуется cupy (pip install cupy-cuda12x)")
    else:
        cen = (alpha - 1) + 2 * (1 - alpha) * rng.random((k, V))
        blocks = []
        for i, nk in enumerate(counts):
            sig = 0.05 + 0.05 * rng.random(V)
            Xk = rng.standard_normal((nk, V))
            Xk = Xk * sig + cen[i]
            blocks.append(Xk)
        X = np.vstack(blocks)

    return Nk, R, rf, X, cen


def _generate_one_combo_fast(args):
    (combo_dir, N, V, true_k, alpha, nmin, n_repeats,
     save_data, generate_plots, seed, use_gpu) = args

    combo_dir = Path(combo_dir)
    combo_dir.mkdir(parents=True, exist_ok=True)

    min_size = max_size = None

    for rep in range(1, n_repeats + 1):
        rng = np.random.default_rng(seed + rep)
        Nk, R, rf, X, cen = generdat_fast(
            N=N, V=V, k=true_k, alpha=alpha, nmin=nmin, rng=rng, use_gpu=use_gpu
        )

        if X.shape != (N, V):
            raise RuntimeError(f"Неверная размерность X: {X.shape}, ожидалось ({N}, {V})")

        min_size = int(np.min(Nk))
        max_size = int(np.max(Nk))

        if save_data:
            np.save(combo_dir / f"rep{rep:03d}_X.npy", X)
            np.save(combo_dir / f"rep{rep:03d}_rf.npy", rf)
            np.save(combo_dir / f"rep{rep:03d}_cen.npy", cen)

        if generate_plots:
            X2d = PCA(n_components=2, random_state=42).fit_transform(X)
            plt.figure(figsize=(8, 6))
            for label in np.unique(rf):
                idx = rf == label
                plt.scatter(X2d[idx, 0], X2d[idx, 1], s=18, alpha=0.65, edgecolors="none")
            plt.title(f"V={V}, K*={true_k}, α={alpha:.2f}, rep={rep:03d}")
            plt.tight_layout()
            plt.savefig(combo_dir / f"rep{rep:03d}_fig2_style.png", dpi=120, bbox_inches='tight')
            plt.close()

    return {
        "combo": f"V={V:2d}_K={true_k:2d}_alpha={alpha:.2f}",
        "min_size": min_size,
        "max_size": max_size,
        "generated": n_repeats,
    }


def run_full_experiment_fast(
    base_dir: str = "synthetic_datasets",
    N: int = 1000,
    nmin: int = 20,
    n_repeats: int = 30,
    V_list: Tuple[int, ...] = (15, 50),
    K_list: Tuple[int, ...] = (7, 15, 21),
    alpha_list: Tuple[float, ...] = (0.25, 0.50, 0.75),
    save_data: bool = True,
    generate_plots: bool = True,
    n_jobs: int = -1,
    seed: int = 42,
    use_gpu: bool = False,
    print_summary: bool = True,
) -> str:
    root_dir = Path(base_dir) / "experiment"
    root_dir.mkdir(parents=True, exist_ok=True)

    combos = [(V, k, a) for V in V_list for k in K_list for a in alpha_list]
    total = len(combos)

    if print_summary:
        print(f"Всего комбинаций: {total}")
        print(f"Датасетов на комбинацию: {n_repeats}")
        print(f"Всего будет сгенерировано: {total * n_repeats} наборов")
        print(f"Папка результата: {root_dir}\n")

    if n_jobs == -1:
        cpu_cnt = os.cpu_count() or 4
        n_jobs = max(1, cpu_cnt - 1)

    if use_gpu and n_jobs != 1:
        if print_summary:
            print("use_gpu=True → устанавливаем n_jobs=1 для стабильности")
        n_jobs = 1

    tasks = []
    for idx, (V, true_k, alpha) in enumerate(combos):
        combo_str = f"V={V:2d}_K={true_k:2d}_alpha={alpha:.2f}"
        combo_dir = root_dir / combo_str
        combo_seed = seed + idx * 100000
        tasks.append((
            combo_dir, N, V, true_k, alpha, nmin, n_repeats,
            save_data, generate_plots, combo_seed, use_gpu
        ))

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_generate_one_combo_fast)(args) for args in tasks
    )

    summary_path = root_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"N={N}, nmin={nmin}, repeats={n_repeats}, n_jobs={n_jobs}, use_gpu={use_gpu}\n")
        for r in sorted(results, key=lambda x: x["combo"]):
            f.write(
                f"{r['combo']:25s} | размеры: {r['min_size']:3d}-{r['max_size']:4d} | "
                f"сгенерировано: {r['generated']}\n"
            )

    if print_summary:
        print(f"\nГенерация успешно завершена!")
        print(f"Результаты сохранены в: {root_dir}")
        print(f"Лог комбинаций: {summary_path}")

    return str(root_dir)


if __name__ == "__main__":
    run_full_experiment_fast(
        base_dir="synthetic_datasets",
        N=1000,
        nmin=20,
        n_repeats=30,
        generate_plots=True,
        n_jobs=-1,
        use_gpu=False,
    )