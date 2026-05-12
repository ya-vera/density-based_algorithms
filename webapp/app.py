from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

warnings.filterwarnings("ignore", message="Glyph.*missing from font", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generator.registry import DATA_GENERATOR_BUILTIN_KEYS, load_data_generator_dataset
from webapp.agent import (
    AgentContext,
    ChatMessage,
    LLMError,
    ask_agent,
    build_backend,
)

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Consensus Clustering System",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Global CSS — minimalist dark sidebar + clean cards
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c1c2e 0%, #16213e 100%);
    border-right: 1px solid #2e2e4e;
}
[data-testid="stSidebar"] * {
    color: #d4d4e8 !important;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.875rem;
    font-weight: 500;
    letter-spacing: 0.01em;
}
[data-testid="stSidebar"] hr {
    border-color: #3a3a5c !important;
}
[data-testid="stSidebar"] code {
    background: #2a2a45 !important;
    color: #a8d8ff !important;
    font-size: 0.82rem;
    border-radius: 4px;
    padding: 0.1em 0.35em;
}

/* ── Main content ────────────────────────── */
.block-container { padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1200px; }

/* ── Metric cards ────────────────────────── */
[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.6rem 1rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700;
    color: #1a1a2e;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

/* ── Buttons ─────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #4361ee;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.45rem 1.4rem;
    transition: background 0.18s, box-shadow 0.18s;
    box-shadow: 0 1px 3px rgba(67,97,238,0.25);
}
.stButton > button[kind="primary"]:hover {
    background: #3451d1;
    box-shadow: 0 3px 8px rgba(67,97,238,0.35);
}
.stButton > button[kind="secondary"] {
    border-radius: 6px;
    font-weight: 500;
}

/* ── Tabs ────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.875rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    margin-right: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    border-bottom: 2px solid #e2e8f0;
    gap: 6px;
}

/* ── Section headers ─────────────────────── */
h1 { font-size: 1.8rem !important; font-weight: 800; color: #1c1c2e; letter-spacing: -0.02em; }
h2 { font-size: 1.3rem !important; font-weight: 700; color: #1e293b; }
h3 { font-size: 1.05rem !important; font-weight: 600; color: #374151; }

/* ── Divider ─────────────────────────────── */
hr { border: none; border-top: 1px solid #e2e8f0; margin: 1rem 0; }

/* ── DataFrames ──────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }

/* ── Expanders ───────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    font-size: 0.9rem;
}

/* ── Info / warning / success tweaks ───── */
[data-testid="stInfoBox"]    { border-radius: 6px; border-left: 4px solid #3b82f6; }
[data-testid="stWarningBox"] { border-radius: 6px; border-left: 4px solid #f59e0b; }
[data-testid="stSuccessBox"] { border-radius: 6px; border-left: 4px solid #10b981; }

/* ── Caption ─────────────────────────────── */
[data-testid="stCaptionContainer"] p {
    font-size: 0.8rem;
    color: #64748b;
}

/* ── Code blocks ─────────────────────────── */
.stCodeBlock { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state — persists across page navigations
# ──────────────────────────────────────────────────────────────────────────────
if "user_algorithms" not in st.session_state:
    st.session_state.user_algorithms: Dict = {}

if "user_consensus" not in st.session_state:
    st.session_state.user_consensus: Dict = {}

if "user_datasets" not in st.session_state:
   st.session_state.user_datasets: Dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# Algorithm helpers
# ──────────────────────────────────────────────────────────────────────────────

BUILTIN_ALGORITHMS = ["dbscan", "hdbscan", "dpc", "rd_dac", "ckdpc"]


@st.cache_resource(show_spinner=False)
def _load_registry():
    import algorithms.registry
    from algorithms.base import AlgorithmRegistry
    return AlgorithmRegistry


def _builtin_names() -> List[str]:
    try:
        reg = _load_registry()
        all_names = reg.list_algorithms()
        return [n for n in all_names if n not in ("optics", "dac")] if all_names else BUILTIN_ALGORITHMS
    except Exception:
        return BUILTIN_ALGORITHMS


def _user_names() -> List[str]:
    return [f"[U] {n}" for n in st.session_state.user_algorithms]


def _all_algorithm_names() -> List[str]:
    return _builtin_names() + _user_names()


def _resolve_algorithm(name: str):
    if name.startswith("[U] "):
        real = name[len("[U] "):]
        entry = st.session_state.user_algorithms.get(real)
        if entry is None:
            raise KeyError(f"User algorithm '{real}' not found in session.")
        return entry["instance"]
    reg = _load_registry()
    return reg.get(name)()


from consensus.monti_helpers import (
    builtin_fit_predict_callable as _monti_builtin_fp,
    user_class_fit_predict_callable as _user_class_fp,
)


def _build_monti2_fit_predict(choice: str, X: np.ndarray, y_true):
    if choice.startswith("[U] "):
        real = choice[len("[U] "):]
        entry = st.session_state.user_algorithms[real]
        ucls = entry["instance"].__class__
        params = dict(entry.get("params") or {})
        return _user_class_fp(ucls, X, y_true, params if params else None)
    return _monti_builtin_fp(choice, X, y_true)


def _save_user_algorithm(display_name: str, instance, file_name: str, params: dict,
                          source_code: bytes = b""):
    st.session_state.user_algorithms[display_name] = {
        "instance": instance, "file_name": file_name, "params": params,
        "source_code": source_code,
    }


def _remove_user_algorithm(display_name: str):
    st.session_state.user_algorithms.pop(display_name, None)


def _save_user_consensus(name: str, instance, file_name: str, params: dict,
                          source_code: bytes = b""):
    st.session_state.user_consensus[name] = {
        "instance": instance, "file_name": file_name, "params": params,
        "source_code": source_code,
    }


def _remove_user_consensus(name: str):
    st.session_state.user_consensus.pop(name, None)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

BUILTIN_DATASET_NAMES = DATA_GENERATOR_BUILTIN_KEYS


def _save_dataset(name: str, X: np.ndarray, y_true, description: str, source: str = "generated"):
    updated = dict(st.session_state.user_datasets)
    updated[name] = {
        "X": X, "y_true": y_true,
        "description": description,
        "source": source,
        "n_samples": X.shape[0],
        "n_features": X.shape[1] if X.ndim == 2 else 0,
    }
    st.session_state.user_datasets = updated


def _delete_dataset(name: str):
    updated = dict(st.session_state.user_datasets)
    updated.pop(name, None)
    st.session_state.user_datasets = updated


def _load_builtin_dataset(key: str, n_samples: int = 50) -> Optional[Dict]:
    """Load one of the built-in research datasets on demand."""
    dg = load_data_generator_dataset(key, n_samples=n_samples)
    return dg


def _all_dataset_names() -> List[str]:
    """Built-in names + user-saved names."""
    return BUILTIN_DATASET_NAMES + list(st.session_state.user_datasets.keys())


def _resolve_dataset(name: str, n_samples: int = 50) -> Optional[Dict]:
    """Return dataset dict for any name (built-in or user-saved)."""
    if name in st.session_state.user_datasets:
        return st.session_state.user_datasets[name]
    return _load_builtin_dataset(name, n_samples)


# ──────────────────────────────────────────────────────────────────────────────
# Shared UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def plot_safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def close_fig(fig):
    if fig is not None:
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass


def _dataset_picker(key_suffix: str = "", n_samples: int = 50):
    """
    Reusable dataset selector widget.
    Returns (X, y_true, ds_name, ds_description).
    """
    all_ds = _all_dataset_names()
    saved_ds = list(st.session_state.user_datasets.keys())

    tabs = st.tabs(["Built-in / Saved", "⬆ Upload file"])

    with tabs[0]:
        col_a, col_b = st.columns([3, 1])
        with col_a:
            choice = st.selectbox(
                "Choose dataset",
                all_ds,
                key=f"ds_pick_{key_suffix}",
                help="Built-in benchmarks or datasets you generated/uploaded",
            )
        with col_b:
            n_samp_local = st.number_input(
                "n_samples (built-in)", 20, 300, n_samples, step=10,
                key=f"ns_{key_suffix}",
                help="Для датасетов data_generator (shape_*, habr_*): "
                "число столбцов / размер вложения; для UCI не меняет скачанные таблицы.",
            )
        ds = _resolve_dataset(choice, n_samp_local)
        if ds is None:
            st.warning("Dataset not found."); return None, None, None, None
        X, y_true = np.asarray(ds["X"]), ds.get("y_true")
        st.caption(f"**{choice}** — {ds.get('description','')}")
        return X, y_true, choice, ds.get("description", "")

    with tabs[1]:
        up_x = st.file_uploader("Upload X (CSV or NPY)", type=["csv", "npy"],
                                 key=f"up_x_{key_suffix}")
        up_y = st.file_uploader("Upload y_true (optional)", type=["csv", "npy"],
                                 key=f"up_y_{key_suffix}")
        if up_x is None:
            return None, None, None, None
        X = (np.load(io.BytesIO(up_x.getbuffer())) if up_x.name.endswith(".npy")
             else np.loadtxt(io.BytesIO(up_x.getbuffer()), delimiter=","))
        y_true = None
        if up_y is not None:
            y_true = (np.load(io.BytesIO(up_y.getbuffer())) if up_y.name.endswith(".npy")
                      else np.loadtxt(io.BytesIO(up_y.getbuffer()), delimiter=",").astype(int))
        return X, y_true, up_x.name, f"Uploaded file: {up_x.name}"


def _run_benchmark(alg_name_list, n_samples=50):
    from evaluation.benchmark import BenchmarkSuite
    suite = BenchmarkSuite(n_samples=n_samples, random_state=42)
    datasets = suite.standard_suite()
    alg_dict, errors = {}, []
    for name in alg_name_list:
        try:
            alg_dict[name] = _resolve_algorithm(name)
        except Exception as e:
            errors.append(f"{name}: {e}")
    bm = suite.compare_algorithms(alg_dict, datasets)
    return bm, errors


# ──────────────────────────────────────────────────────────────────────────────
# AI agent helper
# ──────────────────────────────────────────────────────────────────────────────

def _build_agent_backend():
    return build_backend(
        "ollama",
        model=st.session_state.get("_agent_ollama_model", "qwen2.5:7b"),
        host=st.session_state.get("_agent_ollama_host", "http://localhost:11434"),
    )


@st.fragment
def _agent_chat_fragment(context_key: str, chat_key: str) -> None:
    ctx_data = st.session_state.get(context_key)
    if not ctx_data:
        st.caption(
            "Сначала запусти сравнение или консенсус — агенту нужен "
            "результат, чтобы дать предметный ответ."
        )
        return

    ctx = AgentContext(
        dataset_name=ctx_data.get("dataset_name"),
        n_samples=ctx_data.get("n_samples"),
        n_features=ctx_data.get("n_features"),
        k_true=ctx_data.get("k_true"),
        section=ctx_data.get("section"),
        algorithm_results=ctx_data.get("algorithm_results", []) or [],
        consensus_results=ctx_data.get("consensus_results", []) or [],
        extra_notes=ctx_data.get("extra_notes", []) or [],
    )

    backend = _build_agent_backend()
    if not backend.is_available():
        st.warning(
            "Ollama не отвечает на "
            f"`{st.session_state.get('_agent_ollama_host', 'http://localhost:11434')}`. "
            "Запусти `ollama serve` и убедись, что модель скачана "
            "(`ollama pull qwen2.5:7b`)."
        )
        return

    history = st.session_state.get(chat_key, [])
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Спроси что-нибудь о результатах…", key=f"{chat_key}_input")
    if user_q:
        history = list(history)
        history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        chat_hist = [ChatMessage(role=m["role"], content=m["content"]) for m in history[:-1]]
        is_first = not st.session_state.get("_agent_first_call_done", False)
        spinner_text = (
            "Агент думает… (первый запрос может занять 1–2 минуты, "
            "пока модель загружается в память)"
            if is_first
            else "Агент думает…"
        )
        try:
            with st.spinner(spinner_text):
                answer = ask_agent(backend, ctx, user_q, chat_history=chat_hist)
            st.session_state["_agent_first_call_done"] = True
        except LLMError as e:
            answer = f"Ошибка LLM: {e}"
        except Exception as e:
            msg = str(e)
            if "timed out" in msg.lower() or "timeout" in msg.lower():
                answer = (
                    "Запрос к Ollama не уложился в таймаут. Это бывает на CPU "
                    "с крупными моделями. Попробуй модель поменьше — "
                    "впиши `qwen2.5:3b` или `llama3.2` в боковой панели."
                )
            else:
                answer = f"Неожиданная ошибка: {e}"
        history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state[chat_key] = history

    if history and st.button("Очистить историю", key=f"{chat_key}_clear"):
        st.session_state[chat_key] = []
        st.rerun(scope="fragment")


def _render_agent_chat(context_key: str, chat_key: str, title: str = "Спросить агента") -> None:
    with st.expander(title, expanded=False):
        _agent_chat_fragment(context_key, chat_key)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────────

PAGES = [
    "Главная",
    "Датасеты",
    "Мой алгоритм",
    "Мой консенсус",
    "Сравнение алгоритмов",
    "Консенсус-анализ",
]

with st.sidebar:
    st.markdown("## Консенсус-кластеризация")
    st.caption("Дипломная работа - Вера Слипченко, 2026")
    st.markdown("---")
    page = st.radio("Навигация", PAGES, label_visibility="collapsed")
    st.markdown("---")

    with st.expander("Встроенные алгоритмы", expanded=False):
        for name in BUILTIN_ALGORITHMS:
            st.markdown(f"  `{name}`")

    if st.session_state.user_algorithms:
        st.markdown("**Загруженные алгоритмы:**")
        for uname in list(st.session_state.user_algorithms):
            c1, c2 = st.columns([4, 1])
            c1.markdown(f"  `{uname}`")
            if c2.button("x", key=f"del_{uname}", help=f"Удалить {uname}"):
                _remove_user_algorithm(uname)
                st.rerun()
    else:
        st.caption("Нет загруженных алгоритмов.")

    if st.session_state.user_consensus:
        st.markdown("**Загруженные методы консенсуса:**")
        for ucname in list(st.session_state.user_consensus):
            cc1, cc2 = st.columns([4, 1])
            cc1.markdown(f"  `{ucname}`")
            if cc2.button("x", key=f"del_uc_{ucname}", help=f"Удалить {ucname}"):
                _remove_user_consensus(ucname)
                st.rerun()

    st.markdown("---")
    if st.session_state.user_datasets:
        st.markdown("**Сохранённые датасеты:**")
        for dsname in list(st.session_state.user_datasets):
            d1, d2 = st.columns([4, 1])
            d1.markdown(f"  `{dsname}`")
            if d2.button("x", key=f"del_ds_{dsname}", help=f"Удалить {dsname}"):
                _delete_dataset(dsname)
                st.rerun()
    else:
        st.caption("Нет сохранённых датасетов.")

    st.markdown("---")
    with st.expander("AI-агент (Ollama)", expanded=False):
        st.caption(
            "Локальный чат-агент через Ollama. Перед использованием запусти "
            "`ollama serve` и скачай модель через `ollama pull <name>`. "
            "**Для качественных ответов на русском рекомендуется "
            "`qwen2.5:7b`** или `llama3.1:8b` — модель `llama3.2` (3B) "
            "часто зацикливается и мешает языки."
        )
        st.session_state["_agent_ollama_model"] = st.text_input(
            "Модель Ollama",
            value=st.session_state.get("_agent_ollama_model", "qwen2.5:7b"),
            help="Скачивается через `ollama pull <name>`. Примеры: qwen2.5:7b, llama3.1:8b, llama3.2.",
        )
        st.session_state["_agent_ollama_host"] = st.text_input(
            "Хост Ollama",
            value=st.session_state.get("_agent_ollama_host", "http://localhost:11434"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — Home
# ══════════════════════════════════════════════════════════════════════════════

if page == PAGES[0]:
    st.title("Система тестирования алгоритмов кластеризации")
    st.markdown("---")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
**Что делает система:**

| Вопрос | Где |
|---|---|
| Насколько хорош мой алгоритм? | ARI, AMI, NMI, FMI, SC, CHI, DBI |
| Сколько шума он находит? | Noise fraction |
| Сколько кластеров он обнаружил? | k_found |
""")

    with col_r:
        st.markdown("""
**Быстрый старт:**

1. **Датасеты** — выбери UCI / SIPU, сгенерируй или загрузи свой
2. **Мой алгоритм** — загрузи свой базовый алгоритм и задай параметры
3. **Мой консенсус** — загрузи свой метод консенсуса
4. **Сравнение алгоритмов** — запусти все алгоритмы на любом датасете
5. **Консенсус-анализ** — запусти консенсус-кластеризацию

**Встроенные алгоритмы:**

`DBSCAN` · `HDBSCAN` · `DPC` · `RD-DAC` · `CKDPC`

**Методы консенсуса:**

`Monti` · `CoAssoc` · `Voting` · `CoHiRF` · `FCA`

**Датасеты:**

UCI (Iris, Wine, Seeds, Ecoli, Segmentation) ·
SIPU (Flame, Jain, Spiral, Aggregation, R15, D31) ·
Гауссовские кластеры · Habr синтетические
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Datasets
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[1]:
    st.title("Датасеты")
    st.markdown("Используй готовые исследовательские датасеты, генерируй синтетические или загружай свои.")
    st.markdown("---")

    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA as _PCA

    for _sk, _sv in [("_up_file_ctr", 0), ("_up_conflict", None)]:
        if _sk not in st.session_state:
            st.session_state[_sk] = _sv

    def _draw_preview_plot(X_p, y_p, title: str):
        if X_p.shape[1] == 2:
            X_2d = X_p
            already_01 = (X_p.min() >= 0.0 and X_p.max() <= 1.0)
        else:
            X_sc = MinMaxScaler().fit_transform(X_p)
            X_2d = MinMaxScaler().fit_transform(_PCA(n_components=2, random_state=0).fit_transform(X_sc))
            already_01 = True
        cmap_p = plt.get_cmap("tab10")
        fig_p, ax_p = plt.subplots(figsize=(5, 4), constrained_layout=True)
        lbl_arr = np.asarray(y_p if y_p is not None else np.zeros(len(X_2d), dtype=int), dtype=int)
        for i, u in enumerate(sorted(set(lbl_arr.tolist()))):
            mask = lbl_arr == u
            ax_p.scatter(X_2d[mask, 0], X_2d[mask, 1],
                         c=[cmap_p(i % 10)], s=14, linewidths=0, alpha=0.8)
        ax_p.set_xlabel("x"); ax_p.set_ylabel("y")
        ax_p.grid(True, linewidth=0.3, color="#cccccc", linestyle="--")
        if already_01:
            ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ax_p.set_xticks(ticks); ax_p.set_yticks(ticks)
            ax_p.set_xlim(-0.02, 1.02); ax_p.set_ylim(-0.02, 1.02)
        ax_p.set_title(title, fontsize=10, fontweight="bold")
        ax_p.tick_params(labelsize=7)
        for sp in ax_p.spines.values():
            sp.set_linewidth(0.4)
        st.pyplot(fig_p); plt.close(fig_p)

    def _preview_only(ds_dict: dict, title: str = ""):
        X_p = np.asarray(ds_dict["X"])
        y_p = ds_dict.get("y_true")
        k_p = int(len(np.unique(y_p[y_p >= 0]))) if y_p is not None else "?"
        st.markdown(f"**{ds_dict.get('description','—')}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Объектов", X_p.shape[0])
        c2.metric("Признаков", X_p.shape[1])
        c3.metric("k (ground truth)", k_p)
        _draw_preview_plot(X_p, y_p, title or ds_dict.get("name", ""))
        st.caption("Эти датасеты всегда доступны в разделах **Сравнение алгоритмов** и **Мой алгоритм** — сохранять не нужно.")

    def _preview_and_save(ds_dict: dict, save_key: str, title: str = "",
                          session_clear_key: str = None, counter_keys: list = None):
        X_p = np.asarray(ds_dict["X"])
        y_p = ds_dict.get("y_true")
        k_p = int(len(np.unique(y_p[y_p >= 0]))) if y_p is not None else "?"

        st.markdown(f"**{ds_dict.get('description','—')}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Объектов", X_p.shape[0])
        c2.metric("Признаков", X_p.shape[1])
        c3.metric("k (ground truth)", k_p)

        _draw_preview_plot(X_p, y_p, title if title else ds_dict.get("name", save_key))

        _ds_name_val = ds_dict.get("name", save_key)
        sname = st.text_input("Имя для сохранения", value=_ds_name_val,
                               key=f"sname_{save_key}_{_ds_name_val}")

        _conflict_key = f"_pnp_conflict_{save_key}"
        _msg_key      = f"_pnp_msg_{save_key}"

        if _msg_key in st.session_state:
            st.success(st.session_state.pop(_msg_key))

        if st.session_state.get(_conflict_key) == sname:
            st.warning(f"Датасет **`{sname}`** уже в системе.")
            oc1, oc2, oc3 = st.columns([2, 2, 4])
            if oc1.button("Перезаписать", key=f"ow_{save_key}", type="primary"):
                _save_dataset(sname, X_p, y_p, ds_dict.get("description", ""), source="builtin")
                st.session_state.pop(_conflict_key, None)
                st.session_state[_msg_key] = f"**{sname}** перезаписан"
                if session_clear_key:
                    st.session_state.pop(session_clear_key, None)
                for _ck in (counter_keys or []):
                    if _ck in st.session_state:
                        st.session_state[_ck] += 1
                st.rerun()
            if oc2.button("Отмена", key=f"cancel_{save_key}"):
                st.session_state.pop(_conflict_key, None)
                st.rerun()
        else:
            s1, s2, s3 = st.columns(3)
            with s1:
                if st.button("Сохранить в систему", key=f"save_{save_key}", type="primary"):
                    if sname in st.session_state.user_datasets:
                        st.session_state[_conflict_key] = sname
                        st.rerun()
                    else:
                        _save_dataset(sname, X_p, y_p, ds_dict.get("description", ""), source="builtin")
                        st.session_state[_msg_key] = f"**{sname}** сохранён"
                        if session_clear_key:
                            st.session_state.pop(session_clear_key, None)
                        for _ck in (counter_keys or []):
                            if _ck in st.session_state:
                                st.session_state[_ck] += 1
                        st.rerun()
            with s2:
                buf_x = io.BytesIO(); np.save(buf_x, X_p); buf_x.seek(0)
                st.download_button("X (.npy)", buf_x, file_name=f"{sname}_X.npy",
                                    key=f"dlx_{save_key}")
            with s3:
                if y_p is not None:
                    buf_y = io.BytesIO(); np.save(buf_y, y_p); buf_y.seek(0)
                    st.download_button("y_true (.npy)", buf_y, file_name=f"{sname}_y.npy",
                                        key=f"dly_{save_key}")

    _ds_section = st.radio(
        "Раздел датасетов",
        ["Готовые датасеты", "Генерация", "Загрузить файл", "Сохранённые"],
        horizontal=True,
        key="ds_section_radio",
        label_visibility="collapsed",
    )
    st.markdown("")

    if _ds_section == "Готовые датасеты":
        st.subheader("UCI — реальные датасеты")
        st.caption("Данные загружаются с UCI Machine Learning Repository. Параметры фиксированы.")

        _UCI_META = {
            "uci_iris":           ("Iris", "150 × 4, k=3"),
            "uci_wine":           ("Wine", "178 × 13, k=3"),
            "uci_seeds":          ("Seeds", "210 × 7, k=3"),
            "uci_ecoli":          ("Ecoli", "327 × 7, k=5"),
            "uci_statlog_segment":("Statlog Segmentation", "2310 × 19, k=7"),
        }
        uci_choice = st.selectbox(
            "Выбери датасет",
            list(_UCI_META.keys()),
            format_func=lambda k: f"{_UCI_META[k][0]}  ({_UCI_META[k][1]})",
            key="uci_pick",
        )
        if st.button("Загрузить и предпросмотреть", key="uci_load", type="primary"):
            with st.spinner("Загрузка…"):
                try:
                    ds_d = load_data_generator_dataset(uci_choice)
                    st.session_state["_builtin_preview"] = (ds_d, uci_choice)
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        if st.session_state.get("_builtin_preview", (None, None))[1] == uci_choice:
            ds_d, _ = st.session_state["_builtin_preview"]
            if ds_d:
                _preview_only(ds_d, title=_UCI_META[uci_choice][0])

        st.markdown("---")
        st.subheader("SIPU — классические 2D формы")
        st.caption("Данные из репозитория SIPU (data/sipu/). X нормирован в [0,1]. Параметры фиксированы.")

        _SIPU_META = {
            "flame":       ("Flame",       "240 × 2, k=2"),
            "jain":        ("Jain",         "373 × 2, k=2"),
            "spiral":      ("Spiral",       "312 × 2, k=3"),
            "aggregation": ("Aggregation",  "788 × 2, k=7"),
            "r15":         ("R15",          "600 × 2, k=15"),
            "d31":         ("D31",          "3100 × 2, k=31"),
        }
        sipu_choice = st.selectbox(
            "Выбери форму",
            list(_SIPU_META.keys()),
            format_func=lambda k: f"{_SIPU_META[k][0]}  ({_SIPU_META[k][1]})",
            key="sipu_pick",
        )
        if st.button("Загрузить и предпросмотреть", key="sipu_load", type="primary"):
            with st.spinner("Загрузка…"):
                try:
                    from data_generator.classic_shapes import load_sipu_shapes
                    sipu_all = {ds.meta["shape_kind"]: ds for ds in load_sipu_shapes()}
                    ds_obj = sipu_all[sipu_choice]
                    st.session_state["_sipu_preview"] = (ds_obj.to_dict(), sipu_choice)
                except Exception as e:
                    st.error(f"Ошибка: {e}")

        if st.session_state.get("_sipu_preview", (None, None))[1] == sipu_choice:
            ds_d, _ = st.session_state["_sipu_preview"]
            if ds_d:
                _preview_only(ds_d, title=_SIPU_META[sipu_choice][0])

    elif _ds_section == "Генерация":
        gen_type = st.radio(
            "Тип генерации",
            ["Гауссовские кластеры (generator.py)", "Habr синтетические"],
            horizontal=True, key="gen_type_radio",
        )

        if gen_type.startswith("Гауссовские"):
            st.subheader("Гауссовские кластеры")
            st.caption("Генерируется один датасет с заданными параметрами на основе `generator.py`.")

            gc1, gc2, gc3, gc4, gc5 = st.columns(5)
            g_N     = gc1.number_input("N (объектов)", 100, 5000, 1000, step=100, key="gN")
            g_V     = gc2.number_input("V (признаков)", 2, 100, 15, step=1, key="gV")
            g_K     = gc3.number_input("K (кластеров)", 2, 30, 7, step=1, key="gK")
            g_alpha = gc4.slider("alpha (перекрытие)", 0.05, 0.95, 0.25, step=0.05, key="gAlpha")
            g_seed  = gc5.number_input("Seed", 0, 99999, 42, key="gSeed")
            g_nmin  = st.number_input("нмин. объектов в кластере", 5, 200, 20, step=5,
                                       key="gNmin")
            _g_auto = f"gauss_V{g_V}_K{g_K}_a{g_alpha:.2f}"
            g_name  = st.text_input("Имя датасета", value=_g_auto,
                                     key=f"gname_{g_V}_{g_K}_{g_alpha}")

            if st.button("▶ Сгенерировать", type="primary", key="gen_gauss"):
                from data_generator.generator import generdat_fast
                rng_g = np.random.default_rng(int(g_seed))
                try:
                    _, _, rf, X_g, _ = generdat_fast(
                        N=int(g_N), V=int(g_V), k=int(g_K),
                        alpha=float(g_alpha), nmin=int(g_nmin), rng=rng_g,
                    )
                    y_g = rf - 1
                    ds_g = {
                        "name": g_name,
                        "X": X_g, "y_true": y_g,
                        "description": (f"Гауссовские кластеры: N={g_N}, V={g_V}, "
                                        f"K={g_K}, α={g_alpha:.2f}"),
                    }
                    st.session_state["_gauss_preview"] = ds_g
                except Exception as e:
                    st.error(f"Ошибка: {e}")

            if "_gauss_preview" in st.session_state:
                _preview_and_save(st.session_state["_gauss_preview"], "gauss_gen",
                                  title=st.session_state["_gauss_preview"].get("name", "Гауссовские кластеры"),
                                  session_clear_key="_gauss_preview")

        else:
            st.subheader("Habr синтетические датасеты")
            st.caption("Датасеты на основе примеров из статьи на Habr (2D, объектная кластеризация).")

            _HABR_OPTS = {
                "habr_numpy_linear":           "Линейный тренд (NumPy linear)",
                "habr_numpy_timeseries":       "Временной ряд (NumPy timeseries)",
                "habr_sklearn_blobs":          "Гауссовские сгустки (sklearn blobs)",
                "habr_sklearn_regression_style": "Регрессионный стиль",
                "habr_scipy_mixed":            "Смешанные распределения (SciPy)",
            }
            habr_choice = st.selectbox(
                "Тип датасета",
                list(_HABR_OPTS.keys()),
                format_func=lambda k: _HABR_OPTS[k],
                key="habr_pick",
            )
            habr_n = st.number_input("n_per_group (объектов на кластер)", 10, 500, 50, step=10,
                                      key="habr_n")
            habr_seed = st.number_input("Seed", 0, 99999, 42, key="habr_seed")
            _HABR_SHORT = {
                "habr_numpy_linear":             "habr_linear",
                "habr_numpy_timeseries":         "habr_timeseries",
                "habr_sklearn_blobs":            "habr_blobs",
                "habr_sklearn_regression_style": "habr_regression",
                "habr_scipy_mixed":              "habr_mixed",
            }
            _h_auto = _HABR_SHORT.get(habr_choice, habr_choice)
            habr_name = st.text_input("Имя датасета", value=_h_auto,
                                      key=f"habr_name_{habr_choice}")

            if st.button("▶ Сгенерировать", type="primary", key="gen_habr"):
                from data_generator.habr_synthetic import (
                    make_numpy_linear_features, make_numpy_timeseries_features,
                    make_sklearn_blobs_features, make_sklearn_regression_style_features,
                    make_scipy_mixed_distribution_features,
                )
                rng_h = np.random.default_rng(int(habr_seed))
                _habr_fn = {
                    "habr_numpy_linear":           lambda: make_numpy_linear_features(n_per_group=int(habr_n), rng=rng_h),
                    "habr_numpy_timeseries":       lambda: make_numpy_timeseries_features(n_per_group=int(habr_n), rng=rng_h),
                    "habr_sklearn_blobs":          lambda: make_sklearn_blobs_features(n_objects=int(habr_n) * 3, rng=rng_h),
                    "habr_sklearn_regression_style": lambda: make_sklearn_regression_style_features(n_per_group=int(habr_n), rng=rng_h),
                    "habr_scipy_mixed":            lambda: make_scipy_mixed_distribution_features(n_per=int(habr_n), rng=rng_h),
                }
                try:
                    ds_obj = _habr_fn[habr_choice]()
                    ds_h = ds_obj.to_dict()
                    ds_h["name"] = habr_name
                    st.session_state["_habr_preview"] = ds_h
                except Exception as e:
                    st.error(f"Ошибка: {e}")

            if "_habr_preview" in st.session_state:
                _preview_and_save(st.session_state["_habr_preview"], "habr_gen",
                                  title=st.session_state["_habr_preview"].get("name", habr_choice),
                                  session_clear_key="_habr_preview")

    elif _ds_section == "Загрузить файл":
        import gzip as _gzip_up

        def _load_array(file, is_labels: bool = False):
            buf = io.BytesIO(file.getbuffer())
            name = file.name.lower()
            if name.endswith(".npy"):
                arr = np.load(buf)
            elif name.endswith(".gz"):
                arr = np.loadtxt(_gzip_up.open(buf))
            else:
                raw = buf.read()
                try:
                    arr = np.loadtxt(io.BytesIO(raw), delimiter=",")
                except ValueError:
                    arr = np.loadtxt(io.BytesIO(raw), delimiter=",", skiprows=1)
            return arr.astype(np.int64 if is_labels else np.float64)

        def _validate_X(arr: np.ndarray) -> tuple[list[str], list[str]]:
            errors, warnings = [], []
            if arr.ndim == 1:
                errors.append(
                    "Файл X содержит одномерный массив — нужна матрица (n_samples × n_features). "
                    "Возможно, ты перепутал(а) файл X и файл меток y."
                )
            elif arr.ndim > 2:
                errors.append(f"Неверная размерность массива: {arr.ndim}D (нужно 2D).")
            elif arr.shape[0] < 2:
                errors.append(f"Слишком мало объектов: {arr.shape[0]} (нужно ≥ 2).")
            elif arr.shape[1] == 0:
                errors.append("Нет признаков: число столбцов = 0.")
            if arr.ndim == 2:
                nan_cnt = int(np.isnan(arr).sum())
                inf_cnt = int(np.isinf(arr).sum())
                if nan_cnt:
                    warnings.append(f"Найдено **{nan_cnt}** значений NaN — они будут заменены средним по столбцу.")
                if inf_cnt:
                    warnings.append(f"Найдено **{inf_cnt}** значений Inf — они будут заменены средним по столбцу.")
            return errors, warnings

        def _fix_X(arr: np.ndarray) -> np.ndarray:
            if arr.ndim != 2:
                return arr
            for col in range(arr.shape[1]):
                bad = ~np.isfinite(arr[:, col])
                if bad.any():
                    col_mean = np.nanmean(arr[:, col])
                    arr[bad, col] = col_mean if np.isfinite(col_mean) else 0.0
            return arr

        up_tab_file, up_tab_uci = st.tabs(["Загрузить файл", "UCI по ссылке"])

        with up_tab_file:
            st.caption(
                "Поддерживаемые форматы: **CSV** (числа, через запятую), **NPY** (numpy), "
                "**GZ** (gzip-сжатый текстовый файл — SIPU `.data.gz`). "
                "Заголовок CSV определяется автоматически."
            )

            _up_msg_key = "_up_save_msg"
            if _up_msg_key in st.session_state:
                st.success(st.session_state.pop(_up_msg_key))

            _ctr = st.session_state["_up_file_ctr"]
            up_name = st.text_input("Название датасета", value="my_dataset",
                                    key=f"up_name_{_ctr}")
            up_x    = st.file_uploader(
                "X — матрица признаков (CSV, NPY или GZ)",
                type=["csv", "npy", "gz"], key=f"upload_X_{_ctr}",
            )
            up_y    = st.file_uploader(
                "y_true — метки кластеров (опционально, CSV / NPY / GZ)",
                type=["csv", "npy", "gz"], key=f"upload_y_{_ctr}",
            )
            up_sipu = st.checkbox(
                "SIPU-формат: метки 1-indexed (будут сдвинуты на −1), X нормировать в [0,1] для корректной визуализации",
                value=False, key=f"up_sipu_{_ctr}",
                help="На работу алгоритмов нормировка не влияет — они нормируют данные внутри себя. "
                     "Нужно только чтобы график показывал оси 0–1 как в статьях.",
            )
            up_desc = st.text_input("Описание", value="Загруженный датасет",
                                    key=f"up_desc_{_ctr}")

            if up_x is not None:
                try:
                    X_up = _load_array(up_x, is_labels=False)
                    errs, warns = _validate_X(X_up)
                    for e_msg in errs:
                        st.error(f"{e_msg}")
                    for w_msg in warns:
                        st.warning(w_msg)

                    if not errs:
                        X_up = _fix_X(X_up)

                        y_up = None
                        _y_ok = True
                        if up_y is not None:
                            y_raw_up = _load_array(up_y, is_labels=True)
                            if y_raw_up.ndim != 1:
                                st.error("Файл меток должен быть одномерным вектором (n_samples,).")
                                _y_ok = False
                            elif len(y_raw_up) != X_up.shape[0]:
                                st.error(f"Число меток ({len(y_raw_up)}) не совпадает с числом объектов X ({X_up.shape[0]}).")
                                _y_ok = False
                            else:
                                y_up = (y_raw_up - 1).astype(np.int64) if up_sipu else y_raw_up.astype(np.int64)

                        if _y_ok:
                            if up_sipu:
                                from sklearn.preprocessing import MinMaxScaler as _MMS_up2
                                X_up = _MMS_up2().fit_transform(X_up)

                            info_str = f"**{X_up.shape[0]}** объектов × **{X_up.shape[1]}** признаков"
                            if y_up is not None:
                                info_str += f", **{len(np.unique(y_up))}** кластер(а/ов)"
                            st.success(f"Загружено: {info_str}")

                            _in_conflict = (st.session_state.get("_up_conflict") == up_name)
                            if _in_conflict:
                                st.warning(f"Датасет **`{up_name}`** уже в системе.")
                                _oc1, _oc2, _oc3 = st.columns([2, 2, 4])
                                if _oc1.button("Перезаписать", key=f"up_ow_{_ctr}", type="primary"):
                                    _save_dataset(up_name, X_up, y_up, up_desc, source="uploaded")
                                    st.session_state["_up_conflict"] = None
                                    st.session_state["_up_file_ctr"] += 1
                                    st.session_state[_up_msg_key] = f"**{up_name}** перезаписан"
                                    st.rerun()
                                if _oc2.button("Отмена", key=f"up_cancel_{_ctr}"):
                                    st.session_state["_up_conflict"] = None
                                    st.rerun()
                            else:
                                if st.button("Сохранить в систему",
                                             key=f"save_uploaded_ds_{_ctr}", type="primary"):
                                    if up_name in st.session_state.user_datasets:
                                        st.session_state["_up_conflict"] = up_name
                                        st.rerun()
                                    else:
                                        _save_dataset(up_name, X_up, y_up, up_desc, source="uploaded")
                                        st.session_state["_up_file_ctr"] += 1
                                        st.session_state[_up_msg_key] = f"**{up_name}** сохранён"
                                        st.rerun()
                except Exception as e:
                    st.error(f"Ошибка загрузки: {e}")

        with up_tab_uci:
            st.caption(
                "Загрузка через UCI ML Repository API (ucimlrepo). "
                "Если датасет не поддерживается API — автоматически используется OpenML."
            )

            if "_uci_url_ctr" not in st.session_state:
                st.session_state["_uci_url_ctr"] = 0
            _uci_ctr = st.session_state["_uci_url_ctr"]

            uci_url = st.text_input(
                "Ссылка на UCI датасет",
                value="", placeholder="https://archive.ics.uci.edu/dataset/236/seeds",
                key=f"uci_url_{_uci_ctr}",
            )

            import re as _re_uci
            _m_uci   = _re_uci.search(r'/dataset/\d+/([^/?&#]+)', uci_url)
            _uci_slug = _m_uci.group(1).replace("-", "_") if _m_uci else ""
            _uci_auto_name = f"uci_{_uci_slug}" if _uci_slug else "uci_dataset"
            if _uci_slug:
                st.caption(f"Датасет будет сохранён как **`{_uci_auto_name}`**")

            if st.button("Загрузить с UCI", type="primary", key=f"load_uci_url_{_uci_ctr}",
                         disabled=not uci_url.strip()):
                import re as _re
                from sklearn.preprocessing import LabelEncoder as _LE
                m = _re.search(r'/dataset/(\d+)/([^/?&#]+)', uci_url)
                if not m:
                    st.error("Не удалось извлечь ID из ссылки. Формат: `https://archive.uci.edu/dataset/{id}/{name}`")
                else:
                    ds_id   = int(m.group(1))
                    ds_slug = m.group(2)

                    def _parse_uci_frame(X_raw, y_raw):
                        num_cols = X_raw.select_dtypes(include="number").columns.tolist()
                        dropped  = [c for c in X_raw.columns if c not in num_cols]
                        if dropped:
                            st.warning(
                                f"Удалены нечисловые столбцы ({len(dropped)}): "
                                f"`{'`, `'.join(dropped[:5])}{'…' if len(dropped) > 5 else ''}`"
                            )
                        X_arr = X_raw[num_cols].values.astype(np.float64)
                        X_arr = _fix_X(X_arr)
                        all_nan = np.isnan(X_arr).all(axis=1)
                        n_dropped = int(all_nan.sum())
                        keep_idx  = np.where(~all_nan)[0]
                        if n_dropped:
                            st.warning(f"Удалено **{n_dropped}** строк, где все значения NaN.")
                        X_num = X_arr[keep_idx]
                        if X_num.shape[0] == 0:
                            raise ValueError(
                                "После предобработки не осталось ни одного объекта. "
                                "Все строки содержали пропуски. Попробуй загрузить файл вручную."
                            )
                        y_out = None
                        if y_raw is not None and len(y_raw.columns) > 0:
                            y_col = y_raw.iloc[:, 0].values[keep_idx]
                            try:
                                y_out = np.asarray(y_col, dtype=np.int64)
                            except (ValueError, TypeError):
                                y_out = _LE().fit_transform(y_col.astype(str)).astype(np.int64)
                        return X_num, y_out

                    X_uci_num, y_uci, meta_name = None, None, None

                    with st.spinner(f"Загрузка датасета ID={ds_id} с UCI…"):
                        try:
                            from ucimlrepo import fetch_ucirepo
                            uci_ds    = fetch_ucirepo(id=ds_id)
                            X_uci_num, y_uci = _parse_uci_frame(uci_ds.data.features, uci_ds.data.targets)
                            meta_name = uci_ds.metadata.get("name", f"UCI_{ds_id}")
                        except Exception as _uci_err:
                            _msg = str(_uci_err)
                            if "not available for import" in _msg or "exists in the repository" in _msg:
                                st.info(
                                    f"Датасет ID={ds_id} есть на UCI, но **не поддерживается Python API**. "
                                    f"Пробую загрузить через **OpenML**…"
                                )
                                try:
                                    from sklearn.datasets import fetch_openml
                                    import pandas as _pd
                                    oml = fetch_openml(
                                        name=ds_slug.replace("-", " ").replace("_", " "),
                                        as_frame=True, parser="auto",
                                    )
                                    X_oml = oml.data if isinstance(oml.data, _pd.DataFrame) else _pd.DataFrame(oml.data)
                                    y_oml = _pd.DataFrame({"target": oml.target}) if oml.target is not None else None
                                    X_uci_num, y_uci = _parse_uci_frame(X_oml, y_oml)
                                    meta_name = oml.details.get("name", ds_slug)
                                    st.success(f"Загружено с OpenML: **{meta_name}**")
                                except Exception as _oml_err:
                                    st.error(
                                        f"Не удалось загрузить ни с UCI API, ни с OpenML.\n\n"
                                        f"**UCI:** {_uci_err}\n\n**OpenML:** {_oml_err}\n\n"
                                        f"Попробуй загрузить файл вручную через вкладку **Загрузить файл**."
                                    )
                            else:
                                st.error(f"Ошибка загрузки UCI: {_uci_err}")

                    if X_uci_num is not None:
                        if meta_name is None:
                            meta_name = ds_slug
                        desc = f"{meta_name}: {X_uci_num.shape[0]} объектов × {X_uci_num.shape[1]} признаков"
                        st.session_state["_uci_preview"] = {
                            "name": _uci_auto_name, "X": X_uci_num, "y_true": y_uci,
                            "description": desc, "source": uci_url,
                        }

            if "_uci_preview" in st.session_state:
                prev = st.session_state["_uci_preview"]
                _preview_and_save(prev, "uci_url_ds", title=prev.get("name", "UCI датасет"),
                                  session_clear_key="_uci_preview",
                                  counter_keys=["_uci_url_ctr"])

    else:
        saved = st.session_state.user_datasets
        if not saved:
            st.info("Пока нет сохранённых датасетов. Используй разделы выше чтобы загрузить или сгенерировать.")
        else:
            st.markdown(f"**{len(saved)} датасет(а/ов) в системе:**")
            for ds_name_s, meta in list(saved.items()):
                with st.expander(
                    f"{ds_name_s}  —  {meta.get('n_samples','?')} объектов × {meta.get('n_features','?')} признаков"
                ):
                    st.markdown(f"**Описание:** {meta.get('description','—')}")
                    st.markdown(f"**Источник:** {meta.get('source','—')}")

                    dc1, dc2, dc3 = st.columns(3)
                    with dc1:
                        buf = io.BytesIO()
                        np.save(buf, meta["X"]); buf.seek(0)
                        st.download_button("X (.npy)", buf,
                                           file_name=f"{ds_name_s}_X.npy",
                                           key=f"dl_x_{ds_name_s}")
                    with dc2:
                        if meta.get("y_true") is not None:
                            buf2 = io.BytesIO()
                            np.save(buf2, meta["y_true"]); buf2.seek(0)
                            st.download_button("y (.npy)", buf2,
                                               file_name=f"{ds_name_s}_y.npy",
                                               key=f"dl_y_{ds_name_s}")
                    with dc3:
                        csv_buf = io.StringIO()
                        np.savetxt(csv_buf, meta["X"], delimiter=",")
                        st.download_button("X (.csv)", csv_buf.getvalue().encode(),
                                           file_name=f"{ds_name_s}_X.csv",
                                           mime="text/csv", key=f"dl_csv_{ds_name_s}")

                    if st.button(f"Удалить «{ds_name_s}»", key=f"rm_ds_{ds_name_s}"):
                        _delete_dataset(ds_name_s)
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Мой алгоритм
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[2]:
    import inspect as _inspect
    from evaluation.algorithm_tester import load_algorithm_from_file, validate_algorithm
    from evaluation.algorithm_tester import load_algorithm_from_file as _lafp

    st.title("Мой алгоритм")
    st.markdown("Загрузи свой алгоритм в формате `.py`, задай параметры и сохрани в систему — он появится во всех разделах рядом со встроенными.")
    st.markdown("""
**Интерфейс алгоритма**

Класс принимает параметры в `__init__` (значения задаёшь в JSON на шаге загрузки). Обязателен метод кластеризации по матрице объектов:

```python
import numpy as np

class MyAlgorithm:
    def __init__(self, eps: float = 0.1, **kwargs):
        ...

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        # X: (n_samples, n_features) — строки = объекты
        # вернуть: метки (n_samples,), dtype int; шум / выбросы = -1
        ...
```
""")
    st.markdown("---")

    if "_alg_file_ctr" not in st.session_state:
        st.session_state["_alg_file_ctr"] = 0

    if "_alg_save_msg" in st.session_state:
        st.success(st.session_state.pop("_alg_save_msg"))

    # ── STEP 1: Upload ────────────────────────────────────────────────────────
    st.subheader("Шаг 1 — Загрузи файл")
    _alg_ctr = st.session_state["_alg_file_ctr"]
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Файл .py с классом алгоритма",
            type=["py"],
            help="Класс должен иметь метод fit_predict(X) → labels",
            key=f"upload_alg_{_alg_ctr}",
        )
    with col2:
        alg_display_name = st.text_input("Название алгоритма", value="MyAlgorithm")
        params_input = st.text_area(
            "Параметры (JSON)",
            value="{}",
            height=80,
            key=f"params_json_{_alg_ctr}",
            help="Параметры передаются в __init__ класса. Пример: {\"eps\": 0.15, \"min_samples\": 5}. "
                 "Если класс принимает **kwargs — лишние ключи игнорируются без ошибки. "
                 "Если оставить {}, используются значения по умолчанию класса.",
        )

    # ── Upload processing: Steps 2 & 3 appear directly after upload ───────────
    if uploaded is not None:
        _src_bytes = bytes(uploaded.getbuffer())

        try:
            init_params = json.loads(params_input) if params_input.strip() else {}
        except json.JSONDecodeError:
            st.error("Неверный JSON в параметрах.")
            init_params = None

        if init_params is not None:
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="wb") as f:
                f.write(_src_bytes)
                tmp_path = f.name

            _load_ok = True
            try:
                loaded_alg = load_algorithm_from_file(tmp_path, init_params=init_params)
                valid, val_issues = validate_algorithm(loaded_alg)
            except Exception as e:
                st.error(f"Не удалось загрузить алгоритм: {e}")
                _load_ok = False

            if _load_ok:
                if val_issues:
                    for issue in val_issues:
                        st.warning(f"{issue}")
                    if not valid:
                        _load_ok = False

            if _load_ok:
                # Inspect constructor: check unknown params and collect effective defaults
                _init_sig = _inspect.signature(loaded_alg.__class__.__init__)
                _has_var_kw = any(
                    p.kind == _inspect.Parameter.VAR_KEYWORD
                    for p in _init_sig.parameters.values()
                )
                _valid_kw = set(_init_sig.parameters.keys()) - {"self"}
                if not _has_var_kw and init_params:
                    _unknown_kw = set(init_params.keys()) - _valid_kw
                    if _unknown_kw:
                        st.warning(
                            f"Параметры не найдены в `__init__`: `{', '.join(_unknown_kw)}` — "
                            "они будут проигнорированы."
                        )

                # Build effective params: JSON values + class defaults for the rest
                _eff_params = {}
                for _pn, _pp in _init_sig.parameters.items():
                    if _pn == "self":
                        continue
                    if _pp.kind in (_inspect.Parameter.VAR_POSITIONAL, _inspect.Parameter.VAR_KEYWORD):
                        continue
                    if _pn in init_params:
                        _eff_params[_pn] = (init_params[_pn], "из JSON")
                    elif _pp.default is not _inspect.Parameter.empty:
                        _eff_params[_pn] = (_pp.default, "по умолчанию")

                st.success(f"Класс **`{loaded_alg.__class__.__name__}`** загружен.")
                st.markdown("---")

                st.subheader("Шаг 2 — Параметры")
                if _eff_params:
                    for _pk, (_pv, _src) in _eff_params.items():
                        st.markdown(f"- `{_pk}` = **{_pv}** _{_src}_")
                else:
                    st.markdown("_Параметры не определены в конструкторе._")

                st.markdown("---")
                st.subheader("Шаг 3 — Сохрани в систему")
                st.caption("После сохранения алгоритм появится в **Сравнении алгоритмов**.")

                _alg_conflict_key = "_alg_conflict"
                # Params to save: only the JSON-specified ones (class handles defaults itself)
                _params_to_save = init_params

                if st.session_state.get(_alg_conflict_key) == alg_display_name:
                    st.warning(f"Алгоритм **`{alg_display_name}`** уже в системе.")
                    _ac1, _ac2, _ac3 = st.columns([2, 2, 4])
                    if _ac1.button("Обновить", key="alg_overwrite_btn", type="primary"):
                        _save_user_algorithm(alg_display_name, loaded_alg, uploaded.name,
                                             _params_to_save, source_code=_src_bytes)
                        st.session_state[_alg_conflict_key] = None
                        st.session_state["_alg_file_ctr"] += 1
                        st.session_state["_alg_save_msg"] = f"**{alg_display_name}** обновлён"
                        st.rerun()
                    if _ac2.button("Отмена", key="alg_cancel_btn"):
                        st.session_state[_alg_conflict_key] = None
                        st.rerun()
                else:
                    if st.button(f"Сохранить **{alg_display_name}**", type="primary"):
                        if alg_display_name in st.session_state.user_algorithms:
                            st.session_state[_alg_conflict_key] = alg_display_name
                            st.rerun()
                        else:
                            _save_user_algorithm(alg_display_name, loaded_alg, uploaded.name,
                                                 _params_to_save, source_code=_src_bytes)
                            st.session_state["_alg_file_ctr"] += 1
                            st.session_state["_alg_save_msg"] = (
                                f"**{alg_display_name}** сохранён! "
                                "Перейди в **Сравнение алгоритмов**."
                            )
                            st.rerun()
    else:
        st.info("Загрузи `.py` файл чтобы начать. Пример — `simple_test2.py` в корне проекта.")

    # ── Saved algorithms editor (always at bottom) ─────────────────────────────
    if st.session_state.user_algorithms:
        st.markdown("---")
        st.subheader("Сохранённые алгоритмы")
        for _sa_name, _sa_data in list(st.session_state.user_algorithms.items()):
            with st.expander(f"**{_sa_name}** — `{_sa_data.get('file_name', '')}`", expanded=False):
                _sa_src = _sa_data.get("source_code", b"")
                if _sa_src:
                    st.code(_sa_src.decode("utf-8", errors="replace"), language="python")
                _sa_cur_params = _sa_data.get("params", {})
                _sa_new_json = st.text_area(
                    "Параметры (JSON)",
                    value=json.dumps(_sa_cur_params, ensure_ascii=False),
                    height=70,
                    key=f"edit_params_{_sa_name}_{_alg_ctr}",
                )
                _ep1, _ep2, _ep3 = st.columns([2, 2, 4])
                if _ep1.button("Применить параметры", key=f"apply_p_{_sa_name}"):
                    try:
                        _sa_new_params = json.loads(_sa_new_json) if _sa_new_json.strip() else {}
                    except json.JSONDecodeError:
                        st.error("Неверный JSON.")
                    else:
                        if _sa_src:
                            try:
                                with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="wb") as _tf:
                                    _tf.write(_sa_src)
                                    _tf_path = _tf.name
                                _new_inst = _lafp(_tf_path, init_params=_sa_new_params)
                                _save_user_algorithm(
                                    _sa_name, _new_inst,
                                    _sa_data.get("file_name", ""),
                                    _sa_new_params,
                                    source_code=_sa_src,
                                )
                                st.session_state["_alg_save_msg"] = f"Параметры **{_sa_name}** обновлены."
                                st.rerun()
                            except Exception as _ep_err:
                                st.error(f"Ошибка: {_ep_err}")
                        else:
                            st.warning("Исходный код не сохранён — перезагрузи файл для смены параметров.")
                if _ep2.button("Удалить", key=f"del_alg_{_sa_name}", type="secondary"):
                    _remove_user_algorithm(_sa_name)
                    st.session_state["_alg_save_msg"] = f"**{_sa_name}** удалён из системы."
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Мой консенсус
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[3]:
    import inspect as _inspect_con
    from evaluation.algorithm_tester import load_algorithm_from_file as _lafp_con, validate_algorithm as _val_con

    st.title("Мой консенсус")
    st.markdown(
        "Загрузи свой метод консенсуса в формате `.py`, задай параметры и сохрани — "
        "он появится в разделе **Консенсус-анализ** рядом со встроенными методами."
    )
    st.markdown(r"""
**Два варианта интерфейса**

Система сама выбирает режим по наличию метода **`fit_from_consensus`**. Отдельного «базового класса» от проекта не требуется — достаточно обычного класса с `__init__` и одним из вариантов ниже.

---

**1) Только по данным `X` (без матриц базовых алгоритмов)**

Подходит для проверки загрузки или своего кластеризатора «с нуля». Достаточно **`fit_predict`** — как у встроенного вызова sklearn:

```python
import numpy as np

class MyConsensus:
    def __init__(self, **kwargs):
        ...

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        # X: (n_samples, n_features)
        # вернуть метки (n_samples,), int; шум = -1
        ...
```

В **Консенсус-анализ** при запуске будет вызван только `fit_predict(X)` (результаты выбранных базовых алгоритмов в этот путь не передаются).

---

**2) Консенсус по прогонам выбранных в интерфейсе базовых алгоритмов**

Если в классе есть вызываемый **`fit_from_consensus`**, система после пайплайна консенсуса передаёт туда матрицы, собранные из **встроенных** и **[U] пользовательских** базовых алгоритмов, отмеченных в мультиселекте «Базовые алгоритмы».

Обязательно: после `fit_from_consensus` задать атрибут **`labels_`** — `np.ndarray` формы `(n_samples,)`, `dtype` целочисленный.

```python
import numpy as np

class MyConsensus:
    def __init__(self, **kwargs):
        self.labels_: np.ndarray | None = None

    def fit_from_consensus(
        self,
        X: np.ndarray,
        label_matrix: np.ndarray | None = None,
        coassoc_matrix: np.ndarray | None = None,
        run_weights: np.ndarray | None = None,
        noise_label: int = -1,
        **kwargs,
    ) -> "MyConsensus":
        # X: (n_samples, n_features) — те же объекты, что в анализе
        # label_matrix: (n_runs, n_samples) — строка = один прогон одного базового
        #   алгоритма, столбец j = метка j-го объекта в этом прогоне
        # coassoc_matrix: (n_samples, n_samples) — нормированная матрица
        #   со-присутствия (если пайплайн её посчитал); можно не использовать
        # run_weights: (n_runs,) — веса прогонов (как у voting / coassoc)
        # noise_label: метка шума; такие значения в label_matrix обычно игнорируют
        n = X.shape[0]
        ...
        self.labels_ = ...  # форма (n,), int
        return self
```

Если одновременно есть **`fit_from_consensus`** и **`fit_predict`**, в консенсус-анализе используется **`fit_from_consensus`** (с матрицами); `fit_predict` остаётся запасным для других сценариев.

""")
    st.markdown("---")

    if "_con_file_ctr" not in st.session_state:
        st.session_state["_con_file_ctr"] = 0

    if "_con_save_msg" in st.session_state:
        st.success(st.session_state.pop("_con_save_msg"))

    st.subheader("Шаг 1 — Загрузи файл")
    _con_ctr = st.session_state["_con_file_ctr"]
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_con = st.file_uploader(
            "Файл .py с классом консенсуса",
            type=["py"],
            help="Либо fit_predict(X), либо fit_from_consensus(...)+labels_ — см. описание интерфейса выше.",
            key=f"upload_con_{_con_ctr}",
        )
    with col2:
        con_display_name = st.text_input("Название метода", value="MyConsensus",
                                         key=f"con_name_{_con_ctr}")
        con_params_input = st.text_area(
            "Параметры (JSON)",
            value="{}",
            height=80,
            key=f"con_params_{_con_ctr}",
            help="Параметры передаются в __init__ класса. Если оставить {} — используются значения по умолчанию.",
        )

    if uploaded_con is not None:
        _con_src = bytes(uploaded_con.getbuffer())

        try:
            con_init_params = json.loads(con_params_input) if con_params_input.strip() else {}
        except json.JSONDecodeError:
            st.error("Неверный JSON в параметрах.")
            con_init_params = None

        if con_init_params is not None:
            import tempfile as _tmp_con
            with _tmp_con.NamedTemporaryFile(suffix=".py", delete=False, mode="wb") as _f_con:
                _f_con.write(_con_src)
                _tmp_con_path = _f_con.name

            _con_load_ok = True
            try:
                loaded_con = _lafp_con(_tmp_con_path, init_params=con_init_params)
                _con_valid, _con_issues = _val_con(loaded_con)
            except Exception as _e_con:
                st.error(f"Не удалось загрузить метод: {_e_con}")
                _con_load_ok = False

            if _con_load_ok:
                for _issue in _con_issues:
                    st.warning(f"{_issue}")
                if not _con_valid:
                    _con_load_ok = False

            if _con_load_ok:
                _con_sig = _inspect_con.signature(loaded_con.__class__.__init__)
                _con_eff = {}
                for _pn, _pp in _con_sig.parameters.items():
                    if _pn == "self":
                        continue
                    if _pp.kind in (_inspect_con.Parameter.VAR_POSITIONAL,
                                    _inspect_con.Parameter.VAR_KEYWORD):
                        continue
                    if _pn in con_init_params:
                        _con_eff[_pn] = (con_init_params[_pn], "из JSON")
                    elif _pp.default is not _inspect_con.Parameter.empty:
                        _con_eff[_pn] = (_pp.default, "по умолчанию")

                st.success(f"Класс **`{loaded_con.__class__.__name__}`** загружен.")
                st.markdown("---")

                st.subheader("Шаг 2 — Параметры")
                if _con_eff:
                    for _pk, (_pv, _src) in _con_eff.items():
                        st.markdown(f"- `{_pk}` = **{_pv}** _{_src}_")
                else:
                    st.markdown("_Параметры не определены в конструкторе._")

                st.markdown("---")
                st.subheader("Шаг 3 — Сохрани в систему")
                st.caption("После сохранения метод появится в **Консенсус-анализ**.")

                _con_conflict_key = "_con_conflict"
                if st.session_state.get(_con_conflict_key) == con_display_name:
                    st.warning(f"Метод **`{con_display_name}`** уже в системе.")
                    _cc1, _cc2, _cc3 = st.columns([2, 2, 4])
                    if _cc1.button("Обновить", key="con_overwrite_btn", type="primary"):
                        _save_user_consensus(con_display_name, loaded_con, uploaded_con.name,
                                             con_init_params, source_code=_con_src)
                        st.session_state[_con_conflict_key] = None
                        st.session_state["_con_file_ctr"] += 1
                        st.session_state["_con_save_msg"] = f"**{con_display_name}** обновлён"
                        st.rerun()
                    if _cc2.button("Отмена", key="con_cancel_btn"):
                        st.session_state[_con_conflict_key] = None
                        st.rerun()
                else:
                    if st.button(f"Сохранить **{con_display_name}**", type="primary",
                                 key="con_save_btn"):
                        if con_display_name in st.session_state.user_consensus:
                            st.session_state[_con_conflict_key] = con_display_name
                            st.rerun()
                        else:
                            _save_user_consensus(con_display_name, loaded_con, uploaded_con.name,
                                                 con_init_params, source_code=_con_src)
                            st.session_state["_con_file_ctr"] += 1
                            st.session_state["_con_save_msg"] = (
                                f"**{con_display_name}** сохранён! "
                                "Перейди в **Консенсус-анализ**."
                            )
                            st.rerun()
    else:
        st.info("Загрузи `.py` файл чтобы начать.")

    if st.session_state.user_consensus:
        st.markdown("---")
        st.subheader("Сохранённые методы консенсуса")
        for _sc_name, _sc_data in list(st.session_state.user_consensus.items()):
            with st.expander(f"**{_sc_name}** — `{_sc_data.get('file_name', '')}`",
                             expanded=False):
                _sc_src = _sc_data.get("source_code", b"")
                if _sc_src:
                    st.code(_sc_src.decode("utf-8", errors="replace"), language="python")
                _sc_cur_params = _sc_data.get("params", {})
                _sc_new_json = st.text_area(
                    "Параметры (JSON)",
                    value=json.dumps(_sc_cur_params, ensure_ascii=False),
                    height=70,
                    key=f"edit_con_params_{_sc_name}_{_con_ctr}",
                )
                _cp1, _cp2, _cp3 = st.columns([2, 2, 4])
                if _cp1.button("Применить параметры", key=f"apply_cp_{_sc_name}"):
                    try:
                        _sc_new_params = json.loads(_sc_new_json) if _sc_new_json.strip() else {}
                    except json.JSONDecodeError:
                        st.error("Неверный JSON.")
                    else:
                        if _sc_src:
                            try:
                                import tempfile as _tmp2
                                with _tmp2.NamedTemporaryFile(suffix=".py", delete=False,
                                                              mode="wb") as _tf2:
                                    _tf2.write(_sc_src)
                                    _tf2_path = _tf2.name
                                _new_con_inst = _lafp_con(_tf2_path, init_params=_sc_new_params)
                                _save_user_consensus(
                                    _sc_name, _new_con_inst,
                                    _sc_data.get("file_name", ""),
                                    _sc_new_params, source_code=_sc_src,
                                )
                                st.session_state["_con_save_msg"] = (
                                    f"Параметры **{_sc_name}** обновлены."
                                )
                                st.rerun()
                            except Exception as _cp_err:
                                st.error(f"Ошибка: {_cp_err}")
                        else:
                            st.warning("Исходный код не сохранён — перезагрузи файл.")
                if _cp2.button("Удалить", key=f"del_con_{_sc_name}", type="secondary"):
                    _remove_user_consensus(_sc_name)
                    st.session_state["_con_save_msg"] = f"**{_sc_name}** удалён из системы."
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Сравнение алгоритмов
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[4]:
    import itertools as _itertools
    import matplotlib.pyplot as _plt
    from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                                  normalized_mutual_info_score, fowlkes_mallows_score,
                                  silhouette_score, davies_bouldin_score,
                                  calinski_harabasz_score)
    from sklearn.preprocessing import MinMaxScaler as _MMS
    from sklearn.decomposition import PCA as _PCA2
    from scipy.spatial.distance import pdist as _pdist

    st.title("Сравнение алгоритмов")
    st.markdown("---")

    col_alg, col_mode = st.columns([3, 2])
    with col_alg:
        _all_for_cmp = _builtin_names() + list(st.session_state.user_algorithms.keys())
        selected_algs = st.multiselect(
            "Алгоритмы для сравнения",
            options=_all_for_cmp, default=_all_for_cmp, key="cmp_alg_select",
        )
    with col_mode:
        param_mode = st.radio(
            "Режим параметров",
            ["Авто", "Перебор по сетке"],
            key="cmp_param_mode",
        )

    if not selected_algs:
        st.warning("Выбери хотя бы один алгоритм.")
        st.stop()

    _MODE_CAPTIONS = {
        "Авто": (
            "Запускаются два варианта: внутренняя авто-логика алгоритма и параметры, "
            "вычисленные из данных (k-distance elbow, перцентили расстояний). "
            "Выбирается результат с лучшим **ARI** (если есть метки) или **Silhouette**."
        ),
        "Перебор по сетке": (
            "Сетка строится вокруг рекомендуемых значений (≤40 комбинаций на алгоритм). "
            "Победитель — по **ARI** (если есть метки) или **Silhouette**. "
            "Для пользовательских алгоритмов — сохранённые параметры; если JSON пустой — "
            "вызов ``cls()`` (дефолты из кода), без режима «Авто» как у встроенных."
        ),
    }
    st.caption(_MODE_CAPTIONS[param_mode])
    st.markdown("---")

    _CMP_DS_GROUPS = {
        "UCI реальные": {
            "uci_iris": "Iris", "uci_wine": "Wine", "uci_seeds": "Seeds",
            "uci_ecoli": "Ecoli", "uci_statlog_segment": "Segmentation",
        },
        "Habr синтетические": {
            "habr_numpy_linear": "Linear", "habr_numpy_timeseries": "Timeseries",
            "habr_sklearn_blobs": "Blobs", "habr_sklearn_regression_style": "Regression",
            "habr_scipy_mixed": "Mixed",
        },
        "SIPU формы": {
            "shape_flame": "Flame", "shape_jain": "Jain", "shape_spiral": "Spiral",
            "shape_aggregation": "Aggregation", "shape_r15": "R15", "shape_d31": "D31",
        },
    }

    @st.cache_data(show_spinner=False)
    def _sipu_cache_cmp():
        from data_generator.classic_shapes import load_sipu_shapes
        return {ds.meta["shape_kind"]: (ds.X, ds.y_true) for ds in load_sipu_shapes()}

    def _load_cmp_ds(key: str):
        if key.startswith("shape_"):
            kind = key[len("shape_"):]
            cache = _sipu_cache_cmp()
            if kind in cache:
                return np.asarray(cache[kind][0]), np.asarray(cache[kind][1])
        from data_generator.registry import load_data_generator_dataset
        dg = load_data_generator_dataset(key)
        if dg is not None:
            return np.asarray(dg["X"]), dg.get("y_true")
        return None, None

    def _get_k(lbl: np.ndarray) -> int:
        mask = lbl != -1
        return len(set(lbl[mask].tolist())) if mask.any() else 0

    _ALG_KEY_PARAMS = {
        "dbscan":  ["eps_used_", "min_samples"],
        "hdbscan": ["min_cluster_size", "min_samples"],
        "dpc":     ["dc_used_", "percent"],
        "rd_dac":  ["k_used_"],
        "ckdpc":   ["dc_used_", "alpha", "k_neighbors"],
    }

    def _extract_inst_params(inst, alg_name: str = "") -> dict:
        key_attrs = _ALG_KEY_PARAMS.get(alg_name, [])
        if key_attrs:
            out = {}
            for attr in key_attrs:
                val = getattr(inst, attr, None)
                if val is not None:
                    display_key = attr.rstrip("_").replace("eps_used", "eps").replace("dc_used", "dc")
                    out[display_key] = round(val, 4) if isinstance(val, float) else val
            return out
        import inspect as _insp
        try:
            sig = _insp.signature(inst.__class__.__init__)
            out = {}
            for k, p in sig.parameters.items():
                if k == "self":
                    continue
                if p.kind in (_insp.Parameter.VAR_POSITIONAL, _insp.Parameter.VAR_KEYWORD):
                    continue
                if hasattr(inst, k) and getattr(inst, k) is not None:
                    out[k] = getattr(inst, k)
                elif p.default is not _insp.Parameter.empty and p.default is not None:
                    out[k] = p.default
            return out
        except Exception:
            return {}

    def _adaptive_params_for(alg_name: str, X: np.ndarray) -> dict:
        n, d = X.shape
        X_sc = _MMS().fit_transform(X)
        if alg_name == "dbscan":
            from algorithms.density_params import auto_eps_from_knn
            k = min(max(3, d + 1), 15)
            try:
                eps = float(auto_eps_from_knn(X_sc, k=k))
            except Exception:
                eps = float(np.percentile(_pdist(X_sc), 2))
            return {"eps": round(eps, 4), "min_samples": max(2, int(np.log(n)))}
        if alg_name == "hdbscan":
            return {"min_cluster_size": max(5, n // 50), "min_samples": max(3, int(np.log(n)))}
        if alg_name == "dpc":
            return {"dc": round(float(np.percentile(_pdist(_MMS().fit_transform(X)), 2)), 4)}
        if alg_name == "rd_dac":
            k_rec = max(3, min(15, int(np.log(n) * 1.5)))
            return {"k": k_rec, "min_k": max(2, k_rec // 3)}
        if alg_name == "ckdpc":
            k_nbr = max(5, min(15, int(np.log(n) * 1.5)))
            return {"percent": 2.0, "alpha": 0.5, "k_neighbors": k_nbr}
        return {}

    def _score_lbl(lbl: np.ndarray, X: np.ndarray, y_true) -> float:
        mask = lbl != -1
        k = len(set(lbl[mask].tolist())) if mask.any() else 0
        if k < 2:
            return -np.inf
        try:
            if y_true is not None:
                return adjusted_rand_score(y_true, lbl)
            return silhouette_score(_MMS().fit_transform(X[mask]), lbl[mask])
        except Exception:
            return -np.inf

    def _auto_best_params(alg_name: str, X: np.ndarray, y_true) -> dict:
        cls = _load_registry().get(alg_name)
        p_rec = _adaptive_params_for(alg_name, X)
        try:
            lbl_rec = np.asarray(cls(**p_rec).fit_predict(X), dtype=int)
            s_rec = _score_lbl(lbl_rec, X, y_true)
        except Exception:
            s_rec = -np.inf
        try:
            _ai = _resolve_algorithm(alg_name)
            lbl_auto = np.asarray(_ai.fit_predict(X), dtype=int)
            p_auto = _extract_inst_params(_ai, alg_name)
            s_auto = _score_lbl(lbl_auto, X, y_true)
        except Exception:
            p_auto, s_auto = {}, -np.inf
        return p_rec if s_rec >= s_auto else p_auto

    def _param_grid_for(alg_name: str, X: np.ndarray, center_params: dict = None) -> dict:
        rec = center_params if center_params else _adaptive_params_for(alg_name, X)

        def _geom_around(center: float, lo_f: float, hi_f: float, n: int,
                         min_v: float = 1e-4) -> list:
            center = max(min_v, float(center))
            lo = max(min_v, center * lo_f)
            hi = max(lo * 1.5, center * hi_f)
            pts = np.geomspace(lo, hi, n)
            return np.unique(np.round(np.append(pts, center), 4)).tolist()

        def _int_around(center: int, n_below: int, n_above: int, min_v: int = 2) -> list:
            lo = max(min_v, int(center) - n_below)
            hi = int(center) + n_above
            return list(range(lo, hi + 1))

        if alg_name == "dbscan":
            eps_rec = float(rec.get("eps", 0.1))
            ms_rec  = int(rec.get("min_samples", 5))
            return {
                "eps":         _geom_around(eps_rec, 0.2, 5.0, 12),
                "min_samples": _int_around(ms_rec, 3, 4),
            }
        if alg_name == "hdbscan":
            mcs_rec = int(rec.get("min_cluster_size", 10))
            ms_rec  = int(rec.get("min_samples", 5))
            return {
                "min_cluster_size": _int_around(mcs_rec, 4, 6),
                "min_samples":      _int_around(ms_rec, 2, 3),
            }
        if alg_name == "dpc":
            dc_rec = float(rec.get("dc", 0.05))
            return {"dc": _geom_around(dc_rec, 0.15, 8.0, 14)}
        if alg_name == "rd_dac":
            k_rec   = int(rec.get("k", max(3, min(20, 7))))
            mink_rec = int(rec.get("min_k", max(2, k_rec // 3)))
            return {
                "k":     _int_around(k_rec, 4, 12, min_v=3),
                "min_k": _int_around(mink_rec, 1, 3, min_v=2),
            }
        if alg_name == "ckdpc":
            pct_rec = float(rec.get("percent", 2.0))
            alp_rec = float(rec.get("alpha", 0.5))
            kn_rec  = int(rec.get("k_neighbors", 7))
            return {
                "percent":    _geom_around(pct_rec, 0.3, 5.0, 6, min_v=0.5),
                "alpha":      _geom_around(alp_rec, 0.1, 20.0, 7, min_v=0.05),
                "k_neighbors": _int_around(kn_rec, 3, 3, min_v=3),
            }
        return {}

    def _grid_search_for(alg_name: str, X: np.ndarray, y_true, param_grid: dict, max_combos: int = 80):
        reg  = _load_registry()
        cls  = reg.get(alg_name)
        keys = list(param_grid.keys())
        raw_lists = [param_grid[k] for k in keys]

        total = 1
        for lst in raw_lists:
            total *= len(lst)

        if total > max_combos:
            ratio = (max_combos / total) ** (1.0 / max(1, len(keys)))
            trimmed = []
            for lst in raw_lists:
                n_keep = max(2, min(len(lst), round(len(lst) * ratio)))
                idx = np.round(np.linspace(0, len(lst) - 1, n_keep)).astype(int)
                trimmed.append([lst[i] for i in idx])
            combos = list(_itertools.product(*trimmed))
            if len(combos) > max_combos:
                step = max(1, len(combos) // max_combos)
                combos = combos[::step][:max_combos]
        else:
            combos = list(_itertools.product(*raw_lists))
        best_score, best_params, best_labels = -np.inf, {}, None
        for combo in combos:
            p = dict(zip(keys, combo))
            try:
                lbl  = np.asarray(cls(**p).fit_predict(X), dtype=int)
                mask = lbl != -1
                k    = len(set(lbl[mask].tolist())) if mask.any() else 0
                if k < 2:
                    continue
                if y_true is not None:
                    score = adjusted_rand_score(y_true, lbl)
                else:
                    score = silhouette_score(_MMS().fit_transform(X[mask]), lbl[mask])
                if score > best_score:
                    best_score, best_params, best_labels = score, p, lbl
            except Exception:
                pass
        if best_labels is None:
            # No combo gave k≥2 — try recommended params as fallback
            try:
                _fb_p = _adaptive_params_for(alg_name, X)
                _fb_lbl = np.asarray(cls(**_fb_p).fit_predict(X), dtype=int)
                best_params, best_labels = _fb_p, _fb_lbl
            except Exception:
                try:
                    best_labels = np.asarray(cls().fit_predict(X), dtype=int)
                except Exception:
                    best_labels = np.zeros(len(X), dtype=int)
        return best_params, best_labels, best_score if best_score > -np.inf else None

    def _compute_metrics(X_raw, y_true, labels):
        labels = np.asarray(labels, dtype=int)
        mask   = labels != -1
        k      = len(set(labels[mask].tolist())) if mask.any() else 0
        noise  = float((~mask).sum()) / len(labels)
        result = {"k": k, "шум": f"{noise:.1%}",
                  "ARI": "—", "AMI": "—", "NMI": "—", "FMI": "—",
                  "SC": "—", "CHI": "—", "DBI": "—"}
        if y_true is not None and k >= 2:
            y = np.asarray(y_true, dtype=int)
            result["ARI"] = f"{adjusted_rand_score(y, labels):.4f}"
            result["AMI"] = f"{adjusted_mutual_info_score(y, labels):.4f}"
            result["NMI"] = f"{normalized_mutual_info_score(y, labels):.4f}"
            result["FMI"] = f"{fowlkes_mallows_score(y, labels):.4f}"
        if mask.sum() >= 2 and k >= 2:
            try:
                X_sc = _MMS().fit_transform(X_raw[mask])
                l_sc = labels[mask]
                result["SC"]  = f"{silhouette_score(X_sc, l_sc):.4f}"
                result["CHI"] = f"{calinski_harabasz_score(X_sc, l_sc):.1f}"
                result["DBI"] = f"{davies_bouldin_score(X_sc, l_sc):.4f}"
            except Exception:
                pass
        return result

    def _cluster_ax(ax, X_2d, labels, title, already_01):
        cmap = _plt.get_cmap("tab10")
        lbl  = np.asarray(labels, dtype=int)
        for i, u in enumerate(sorted(set(lbl.tolist()))):
            mask   = lbl == u
            color  = "#888888" if u == -1 else cmap(i % 10)
            marker = "x"      if u == -1 else "o"
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], marker=marker,
                       s=12, linewidths=0.5 if u == -1 else 0,
                       zorder=3 if u == -1 else 2)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xlabel("x", fontsize=6); ax.set_ylabel("y", fontsize=6)
        ax.tick_params(labelsize=5)
        if already_01:
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
        ax.grid(True, linewidth=0.3, color="#cccccc", linestyle="--")
        for sp in ax.spines.values():
            sp.set_linewidth(0.3)

    def _make_2d(X_raw):
        if X_raw.shape[1] == 2:
            already_01 = (float(X_raw.min()) >= -0.01 and float(X_raw.max()) <= 1.01)
            return X_raw, already_01
        X_sc = _MMS().fit_transform(X_raw)
        X_2d = _MMS().fit_transform(_PCA2(n_components=2, random_state=0).fit_transform(X_sc))
        return X_2d, True

    def _fmt_params(p) -> str:
        if not p:
            return "—"
        if isinstance(p, dict):
            return ", ".join(f"{k} = {v}" for k, v in p.items())
        return str(p)

    def _show_results(X_cmp, y_cmp, X_2d_cmp, ok01, active_label, metric_rows, figs_list):
        if metric_rows:
            import pandas as _pd
            st.dataframe(_pd.DataFrame(metric_rows).set_index("Алгоритм"), width="stretch")
        if figs_list:
            st.markdown("#### Визуализация")
            _gt_labels = y_cmp if y_cmp is not None else np.zeros(len(X_2d_cmp), dtype=int)
            _, _mid, _ = st.columns([1, 2, 1])
            with _mid:
                _fig_gt, _ax_gt = _plt.subplots(figsize=(6, 5))
                _cluster_ax(_ax_gt, X_2d_cmp, _gt_labels, active_label, ok01)
                _fig_gt.tight_layout()
                st.pyplot(_fig_gt)
                _plt.close(_fig_gt)
            st.markdown("")
            n_alg = len(figs_list)
            for _rs in range(0, n_alg, 3):
                _row_items = figs_list[_rs:_rs + 3]
                _row_cols = st.columns(3)
                for _ci, (_an, _lbl) in enumerate(_row_items):
                    with _row_cols[_ci]:
                        _fig_a, _ax_a = _plt.subplots(figsize=(4.5, 3.8))
                        _cluster_ax(_ax_a, X_2d_cmp, _lbl, _an, ok01)
                        _fig_a.tight_layout()
                        st.pyplot(_fig_a)
                        _plt.close(_fig_a)

    for group_name, ds_map in _CMP_DS_GROUPS.items():
        st.markdown(f"**{group_name}**")
        btn_cols = st.columns(len(ds_map))
        for col, (ds_key, ds_label) in zip(btn_cols, ds_map.items()):
            if col.button(ds_label, key=f"cmp_{ds_key}"):
                st.session_state["_cmp_open"]    = ds_key
                st.session_state["_cmp_is_user"] = False
                st.session_state.pop("_cmp_gs_cache", None)
                st.session_state.pop("_cmp_auto_cache", None)
                st.session_state["_agent_chat_cmp"] = []

    user_ds_saved = st.session_state.user_datasets
    if user_ds_saved:
        st.markdown("**Мои сохранённые датасеты**")
        _u_keys   = list(user_ds_saved.keys())
        _u_n_row  = 5
        for _u_row_start in range(0, len(_u_keys), _u_n_row):
            _u_row_keys = _u_keys[_u_row_start:_u_row_start + _u_n_row]
            _u_cols     = st.columns(len(_u_row_keys))
            for col, uds_key in zip(_u_cols, _u_row_keys):
                if col.button(uds_key, key=f"cmp_u_{uds_key}"):
                    st.session_state["_cmp_open"]    = uds_key
                    st.session_state["_cmp_is_user"] = True
                    st.session_state.pop("_cmp_gs_cache", None)
                    st.session_state.pop("_cmp_auto_cache", None)
                    st.session_state["_agent_chat_cmp"] = []

    active_key = st.session_state.get("_cmp_open")
    is_user_ds = st.session_state.get("_cmp_is_user", False)

    if active_key:
        active_label = active_key
        for grp, dm in _CMP_DS_GROUPS.items():
            if active_key in dm:
                active_label = dm[active_key]
                break

        st.markdown("---")
        st.subheader(f"{active_label}")

        if is_user_ds and active_key in st.session_state.user_datasets:
            meta_cmp = st.session_state.user_datasets[active_key]
            X_cmp = np.asarray(meta_cmp["X"])
            y_cmp = np.asarray(meta_cmp["y_true"]) if meta_cmp.get("y_true") is not None else None
        else:
            with st.spinner(f"Загрузка {active_label}…"):
                X_cmp, y_cmp = _load_cmp_ds(active_key)

        if X_cmp is None:
            st.error(f"Датасет {active_key!r} не найден.")
        elif X_cmp.ndim != 2 or X_cmp.shape[0] < 2:
            st.error(
                f"Датасет **{active_label}** имеет неверную форму `{X_cmp.shape}`. "
                "Нужна матрица (n_samples × n_features) с ≥ 2 объектами. "
                "Удали его из системы и загрузи заново."
            )
        else:
            st.caption(f"{X_cmp.shape[0]} объектов × {X_cmp.shape[1]} признаков")
            X_2d_cmp, ok01 = _make_2d(X_cmp)


            if param_mode == "Перебор по сетке":
                gs_cache = st.session_state.get("_cmp_gs_cache", {})
                cache_key = (active_key, tuple(sorted(selected_algs)))
                if cache_key not in gs_cache:
                    builtin_gs = [a for a in selected_algs if a not in st.session_state.user_algorithms]
                    total_combos = sum(
                        min(40, len(list(_itertools.product(
                            *(_param_grid_for(a, X_cmp).values() or [[]])
                        )))) for a in builtin_gs
                    )
                    st.info(
                        f"Будет проверено до **{total_combos}** комбинаций параметров "
                        f"по **{len(builtin_gs)}** встроенным алгоритмам. "
                        "Нажми кнопку чтобы запустить."
                    )
                    if not st.button("▶ Запустить перебор параметров", type="primary", key="run_gs_btn"):
                        st.stop()
                    metric_rows_gs, figs_list_gs = [], []
                    for aname in selected_algs:
                        try:
                            if aname in st.session_state.user_algorithms:
                                inst = st.session_state.user_algorithms[aname]["instance"]
                                with st.spinner(f"  {aname} (сохранённые параметры)…"):
                                    lbl = np.asarray(inst.fit_predict(X_cmp), dtype=int)
                                _sp = st.session_state.user_algorithms[aname].get("params") or {}
                                p_str = _fmt_params(_sp) if _sp else _fmt_params(_extract_inst_params(inst))
                            else:
                                _center = _auto_best_params(aname, X_cmp, y_cmp)
                                pg = _param_grid_for(aname, X_cmp, center_params=_center)
                                n_c = len(list(_itertools.product(*pg.values()))) if pg else 0
                                with st.spinner(f"  {aname}: {min(40, n_c)} комбинаций…"):
                                    best_p, lbl, best_sc = _grid_search_for(aname, X_cmp, y_cmp, pg)
                                p_str = _fmt_params(best_p)
                            mets = _compute_metrics(X_cmp, y_cmp, lbl)
                            row  = {"Алгоритм": aname}
                            row.update(mets)
                            row["параметры"] = p_str
                            metric_rows_gs.append(row)
                            figs_list_gs.append((aname, lbl))
                        except Exception as e:
                            metric_rows_gs.append({"Алгоритм": aname, "k": "ERR", "шум": str(e)[:60]})
                    gs_cache[cache_key] = (metric_rows_gs, figs_list_gs)
                    st.session_state["_cmp_gs_cache"] = gs_cache
                else:
                    metric_rows_gs, figs_list_gs = gs_cache[cache_key]
                    st.success("Показаны кэшированные результаты перебора. Нажми датасет снова для сброса.")
                    if st.button("Перезапустить перебор", key="rerun_gs_btn"):
                        st.session_state.pop("_cmp_gs_cache", None)
                        st.rerun()
                _show_results(X_cmp, y_cmp, X_2d_cmp, ok01, active_label, metric_rows_gs, figs_list_gs)
                st.session_state["_last_cmp_context"] = {
                    "dataset_name": active_label,
                    "n_samples": int(X_cmp.shape[0]),
                    "n_features": int(X_cmp.shape[1]),
                    "k_true": int(len(np.unique(y_cmp))) if y_cmp is not None else None,
                    "section": "Сравнение алгоритмов (перебор по сетке)",
                    "algorithm_results": metric_rows_gs,
                }

            else:
                auto_cache = st.session_state.get("_cmp_auto_cache", {})
                auto_key = (active_key, tuple(sorted(selected_algs)))
                if auto_key not in auto_cache:
                    metric_rows, figs_list = [], []
                    for aname in selected_algs:
                        try:
                            if aname in st.session_state.user_algorithms:
                                inst = st.session_state.user_algorithms[aname]["instance"]
                                with st.spinner(f"  {aname}…"):
                                    lbl = np.asarray(inst.fit_predict(X_cmp), dtype=int)
                                _sp = st.session_state.user_algorithms[aname].get("params") or {}
                                p_str = _fmt_params(_sp) if _sp else _fmt_params(_extract_inst_params(inst))
                            else:
                                cls = _load_registry().get(aname)
                                with st.spinner(f"  {aname}…"):
                                    p_best = _auto_best_params(aname, X_cmp, y_cmp)
                                    lbl = np.asarray(cls(**p_best).fit_predict(X_cmp), dtype=int)
                                p_str = _fmt_params(p_best)
                                _cur_score = _score_lbl(lbl, X_cmp, y_cmp)
                                _need_fallback = _get_k(lbl) < 2 or _cur_score < 0
                                if _need_fallback:
                                    with st.spinner(f"  {aname}: результат неудовлетворительный → перебор…"):
                                        pg = _param_grid_for(aname, X_cmp, center_params=p_best)
                                        p_fb, lbl_fb, sc_fb = _grid_search_for(
                                            aname, X_cmp, y_cmp, pg, max_combos=30)
                                    _fb_score = _score_lbl(lbl_fb, X_cmp, y_cmp)
                                    if _get_k(lbl_fb) >= 2 and _fb_score > _cur_score:
                                        lbl, p_str = lbl_fb, _fmt_params(p_fb)
                            mets = _compute_metrics(X_cmp, y_cmp, lbl)
                            row  = {"Алгоритм": aname}
                            row.update(mets)
                            row["параметры"] = p_str
                            metric_rows.append(row)
                            figs_list.append((aname, lbl))
                        except Exception as e:
                            metric_rows.append({"Алгоритм": aname, "k": "ERR", "шум": str(e)[:60]})
                    auto_cache[auto_key] = (metric_rows, figs_list)
                    st.session_state["_cmp_auto_cache"] = auto_cache
                    st.session_state["_agent_chat_cmp"] = []
                else:
                    metric_rows, figs_list = auto_cache[auto_key]
                _show_results(X_cmp, y_cmp, X_2d_cmp, ok01, active_label, metric_rows, figs_list)
                st.session_state["_last_cmp_context"] = {
                    "dataset_name": active_label,
                    "n_samples": int(X_cmp.shape[0]),
                    "n_features": int(X_cmp.shape[1]),
                    "k_true": int(len(np.unique(y_cmp))) if y_cmp is not None else None,
                    "section": "Сравнение алгоритмов (авто-параметры)",
                    "algorithm_results": metric_rows,
                }

            st.markdown("---")
            _render_agent_chat(
                context_key="_last_cmp_context",
                chat_key="_agent_chat_cmp",
                title="Спросить AI-агента о результатах",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Консенсус-анализ
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[5]:
    st.title("Консенсус-анализ")
    st.markdown("Запусти полный консенсус-пайплайн с любыми алгоритмами и данными.")
    st.markdown("---")

    st.subheader("1. Выбор данных")

    _CON_DS_GROUPS = {
        "UCI реальные": {
            "uci_iris": "Iris", "uci_wine": "Wine", "uci_seeds": "Seeds",
            "uci_ecoli": "Ecoli", "uci_statlog_segment": "Segmentation",
        },
        "Habr синтетические": {
            "habr_numpy_linear": "Linear", "habr_numpy_timeseries": "Timeseries",
            "habr_sklearn_blobs": "Blobs", "habr_sklearn_regression_style": "Regression",
            "habr_scipy_mixed": "Mixed",
        },
        "SIPU формы": {
            "shape_flame": "Flame", "shape_jain": "Jain", "shape_spiral": "Spiral",
            "shape_aggregation": "Aggregation", "shape_r15": "R15", "shape_d31": "D31",
        },
    }
    if st.session_state.user_datasets:
        _CON_DS_GROUPS["Сохранённые"] = {
            k: k for k in st.session_state.user_datasets
        }

    @st.cache_data(show_spinner=False)
    def _sipu_cache_con():
        from data_generator.classic_shapes import load_sipu_shapes
        return {ds.meta["shape_kind"]: (ds.X, ds.y_true) for ds in load_sipu_shapes()}

    def _load_con_ds(key: str):
        if key.startswith("shape_"):
            kind = key[len("shape_"):]
            cache = _sipu_cache_con()
            if kind in cache:
                return np.asarray(cache[kind][0]), np.asarray(cache[kind][1])
        if key in st.session_state.user_datasets:
            meta = st.session_state.user_datasets[key]
            return np.asarray(meta["X"]), meta.get("y_true")
        from data_generator.registry import load_data_generator_dataset
        dg = load_data_generator_dataset(key)
        if dg is not None:
            return np.asarray(dg["X"]), dg.get("y_true")
        return None, None

    _con_group_names = list(_CON_DS_GROUPS.keys())
    _con_group_sel = st.radio(
        "Категория датасетов", _con_group_names, horizontal=True, key="con_ds_group"
    )
    _con_ds_map = _CON_DS_GROUPS[_con_group_sel]
    _con_ds_labels = list(_con_ds_map.values())
    _con_ds_keys   = list(_con_ds_map.keys())

    _con_ds_cols = st.columns(min(len(_con_ds_labels), 6))
    if "con_active_ds" not in st.session_state:
        st.session_state["con_active_ds"] = _con_ds_keys[0]

    for _ci, (_ck, _cl) in enumerate(zip(_con_ds_keys, _con_ds_labels)):
        _is_active = st.session_state["con_active_ds"] == _ck
        if _con_ds_cols[_ci % len(_con_ds_cols)].button(
            _cl, key=f"con_ds_btn_{_ck}",
            type="primary" if _is_active else "secondary",
        ):
            st.session_state["con_active_ds"] = _ck
            st.rerun()

    _con_active_key = st.session_state["con_active_ds"]
    if _con_active_key not in _con_ds_keys:
        st.session_state["con_active_ds"] = _con_ds_keys[0]
        _con_active_key = _con_ds_keys[0]

    X, y_true = _load_con_ds(_con_active_key)
    ds_name = _con_ds_map.get(_con_active_key, _con_active_key)

    if X is None:
        st.warning("Датасет не найден или не загружен.")
        st.stop()

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2:
        st.error("Датасет должен быть матрицей (n_samples × n_features) с ≥ 2 объектами.")
        st.stop()

    st.info(f"**{ds_name}** · **{X.shape[0]}** объектов × **{X.shape[1]}** признаков"
            + (f" · {len(np.unique(y_true))} классов" if y_true is not None else ""))

    st.markdown("---")
    st.subheader("2. Настройки алгоритмов и консенсуса")

    all_names = _all_algorithm_names()
    default_con = ["hdbscan", "dpc", "ckdpc"] + _user_names()

    methods = st.multiselect(
        "Встроенные методы консенсуса",
        ["monti2", "coassoc", "voting", "cohirf", "fca"],
        default=["monti2", "coassoc", "voting"],
        help=(
            "**monti2** — Monti et al. (2003), реализация по burtonrj/consensusclustering "
            "(connectivity / identity / consensus), один выбранный плотностный алгоритм, "
            "параметры как в режиме «Авто» в сравнении. "
            "**coassoc** — Fred & Jain (2005). **voting** — взвешенное голосование. "
            "**cohirf** — CoHiRF (один плотностной базовый алгоритм с авто-параметрами; "
            "случайные проекции признаков и иерархия — внутри метода). "
            "**fca** — формальные концепты."
        ),
    )

    monti2_choice = None
    if "monti2" in methods:
        _monti_opts = list(all_names)
        _def_i = 0
        for _i, _nm in enumerate(_monti_opts):
            if _nm == "hdbscan" or _nm in default_con:
                _def_i = _i
                break
        monti2_choice = st.selectbox(
            "Плотностный алгоритм для Monti2 (ровно один)",
            _monti_opts,
            index=min(_def_i, len(_monti_opts) - 1),
            key="con_monti2_single_alg",
            format_func=lambda n: n[len("[U] "):] if n.startswith("[U] ") else n,
            help="Параметры подбираются так же, как в «Сравнение алгоритмов» → режим «Авто» (рекомендуемые vs внутренний авто).",
        )
        st.caption(
            "Для **monti2** используется только этот алгоритм. "
            "Ниже — отдельный список базовых прогонов для coassoc / voting / fca."
        )

    cohirf_choice = None
    if "cohirf" in methods:
        _cohirf_opts = list(all_names)
        _cd_i = 0
        for _i, _nm in enumerate(_cohirf_opts):
            if _nm == "hdbscan" or _nm in default_con:
                _cd_i = _i
                break
        if "monti2" in methods and monti2_choice is not None:
            _same_cohirf = st.checkbox(
                "CoHiRF: тот же плотностный базовый алгоритм, что и для Monti2",
                value=True,
                key="con_cohirf_same_as_monti",
            )
            if _same_cohirf:
                cohirf_choice = monti2_choice
            else:
                cohirf_choice = st.selectbox(
                    "Плотностный алгоритм для CoHiRF (ровно один, авто-параметры)",
                    _cohirf_opts,
                    index=min(_cd_i, len(_cohirf_opts) - 1),
                    key="con_cohirf_single_alg_diff",
                    format_func=lambda n: n[len("[U] "):] if n.startswith("[U] ") else n,
                    help="Как в «Сравнение алгоритмов» → «Авто». CoHiRF сам строит случайные проекции и иерархию.",
                )
        else:
            cohirf_choice = st.selectbox(
                "Плотностный алгоритм для CoHiRF (ровно один, авто-параметры)",
                _cohirf_opts,
                index=min(_cd_i, len(_cohirf_opts) - 1),
                key="con_cohirf_single_alg",
                format_func=lambda n: n[len("[U] "):] if n.startswith("[U] ") else n,
                help="Как в «Сравнение алгоритмов» → «Авто». CoHiRF сам строит случайные проекции и иерархию.",
            )

    _other_methods = [m for m in methods if m != "monti2"]
    _multibase_methods = frozenset({"coassoc", "voting", "fca"})
    _ms_base_label = "Базовые алгоритмы"
    if "monti2" in methods and "cohirf" in methods:
        _ms_base_label = "Базовые алгоритмы (для coassoc / voting / fca; Monti2 и CoHiRF — выше)"
    elif "monti2" in methods:
        _ms_base_label = "Базовые алгоритмы (для coassoc / voting / fca; Monti2 — выше)"
    elif "cohirf" in methods:
        _ms_base_label = "Базовые алгоритмы (для coassoc / voting / fca; CoHiRF — выше)"

    alg_sel = st.multiselect(
        _ms_base_label,
        all_names,
        default=[n for n in default_con if n in all_names],
        format_func=lambda n: n[len("[U] "):] if n.startswith("[U] ") else n,
        help=(
            None
            if not ("monti2" in methods or "cohirf" in methods)
            else "Не влияет на Monti2 и CoHiRF (у них свой одиночный плотностной базовый алгоритм)."
        ),
    )

    _user_con_names = list(st.session_state.user_consensus.keys())
    if _user_con_names:
        user_con_sel = st.multiselect(
            "Мои методы консенсуса",
            _user_con_names,
            default=_user_con_names,
            help=(
                "См. раздел Мой консенсус: с fit_from_consensus — матрицы базовых прогонов; "
                "без него — только fit_predict(X)."
            ),
        )
    else:
        user_con_sel = []
        st.caption("Нет загруженных пользовательских методов консенсуса. Загрузи в разделе **Мой консенсус**.")

    run_btn = st.button("▶ Запустить консенсус", type="primary")

    _needs_multibase_algorithms = bool(set(_other_methods) & _multibase_methods)
    _can_run = (methods or user_con_sel) and (
        not _needs_multibase_algorithms or bool(alg_sel)
    ) and (("monti2" not in methods) or (monti2_choice is not None))
    _can_run = _can_run and (("cohirf" not in methods) or (cohirf_choice is not None))

    if run_btn and not _can_run:
        st.warning(
            "Выбери базовые алгоритмы для coassoc / voting / fca (если они включены), "
            "настрой Monti2 / CoHiRF при необходимости, или сними лишние методы."
        )

    if run_btn and _can_run and (methods or user_con_sel):
        from consensus.runner import ConsensusRunner

        builtin_sel = [n for n in alg_sel if not n.startswith("[U] ")]
        user_sel = [n for n in alg_sel if n.startswith("[U] ")]

        _n_obj = X.shape[0]
        _k_max_auto = max(6, min(15, int(round(_n_obj ** 0.4))))
        _monti_call = None
        if "monti2" in methods and monti2_choice is not None:
            _monti_call = _build_monti2_fit_predict(monti2_choice, X, y_true)

        _cohirf_call = None
        if "cohirf" in methods and cohirf_choice is not None:
            _cohirf_call = _build_monti2_fit_predict(cohirf_choice, X, y_true)

        runner = ConsensusRunner(
            algorithm_names=builtin_sel,
            consensus_methods=methods if methods else [],
            k_range=(2, _k_max_auto),
            verbose=False,
            monti2_base_callable=_monti_call,
            cohirf_base_callable=_cohirf_call,
        )

        with st.spinner("Консенсус-пайплайн выполняется…"):
            result = runner.fit(X, y_true=y_true)

        # Inject user base-algorithm labels: saved JSON params, or cls() defaults (no auto_best)
        if user_sel:
            from consensus.base import build_coassociation, compute_run_weights
            extra_labels = []
            for uname in user_sel:
                try:
                    _ufp = _build_monti2_fit_predict(uname, X, y_true)
                    extra_labels.append(np.asarray(_ufp(X), dtype=int))
                except Exception as e:
                    st.warning(f"Алгоритм '{uname}': {e}")
            if extra_labels:
                if result.label_matrix is not None:
                    combined = np.vstack([result.label_matrix] + extra_labels)
                else:
                    combined = np.vstack(extra_labels)
                w = compute_run_weights(combined, X.shape[0])
                C, _ = build_coassociation(combined, X.shape[0], weights=w)
                result.label_matrix, result.run_weights, result.coassoc_matrix = combined, w, C

        # Run user consensus methods
        if user_con_sel:
            import time as _time_con
            for _ucn in user_con_sel:
                _uc_inst = st.session_state.user_consensus[_ucn]["instance"]
                _t0_uc = _time_con.perf_counter()
                try:
                    _fc = getattr(_uc_inst, "fit_from_consensus", None)
                    if callable(_fc):
                        _fc(
                            X,
                            label_matrix=result.label_matrix,
                            coassoc_matrix=result.coassoc_matrix,
                            run_weights=result.run_weights,
                            noise_label=runner.noise_label,
                        )
                        _raw_lbl = getattr(_uc_inst, "labels_", None)
                        if _raw_lbl is None:
                            st.warning(
                                f"Метод '{_ucn}': после fit_from_consensus ожидается атрибут labels_."
                            )
                            continue
                        _uc_labels = np.asarray(_raw_lbl, dtype=int)
                        if _uc_labels.shape[0] != X.shape[0]:
                            st.warning(
                                f"Метод '{_ucn}': labels_ длины {_uc_labels.shape[0]}, ожидалось {X.shape[0]}."
                            )
                            continue
                    else:
                        _uc_labels = np.asarray(_uc_inst.fit_predict(X), dtype=int)
                    _uc_key = _ucn
                    result.labels[_uc_key] = _uc_labels
                    result.k_found[_uc_key] = int(
                        len(np.unique(_uc_labels[_uc_labels >= 0]))
                    )
                    result.runtime_sec[_uc_key] = _time_con.perf_counter() - _t0_uc
                    if hasattr(_uc_inst, "k_selection_method_"):
                        result.k_selection_methods[_uc_key] = (
                            _uc_inst.k_selection_method_ or ""
                        )
                except Exception as _uc_e:
                    st.warning(f"Метод '{_ucn}' упал: {_uc_e}")

        from evaluation.metrics import ClusteringMetrics
        from sklearn.preprocessing import MinMaxScaler as _MMS
        _X_sc = _MMS().fit_transform(X)

        _rows = []
        for _mname, _mlabels in result.labels.items():
            _cm = ClusteringMetrics(_mlabels, y_true=y_true, X=_X_sc)
            _ext = _cm.external()
            _inn = _cm.internal()
            _row = {"Метод": _mname, "k": result.k_found.get(_mname, "?")}
            if _ext:
                _row["ARI"]  = round(_ext.get("ari",  float("nan")), 3)
                _row["NMI"]  = round(_ext.get("nmi",  float("nan")), 3)
                try:
                    from sklearn.metrics import adjusted_mutual_info_score as _ami_fn
                    _row["AMI"] = round(float(_ami_fn(y_true, _mlabels)), 3)
                except Exception:
                    _row["AMI"] = float("nan")
                _row["FMI"]  = round(_ext.get("fmi",  float("nan")), 3)
            if _inn:
                _row["SC"]   = round(_inn.get("silhouette",        float("nan")), 3)
                _row["CHI"]  = round(_inn.get("calinski_harabasz", float("nan")), 1)
                _row["DBI"]  = round(_inn.get("davies_bouldin",    float("nan")), 3)
            _rows.append(_row)

        st.session_state["_consensus_state"] = {
            "X": X,
            "y_true": y_true,
            "ds_name": ds_name,
            "labels": dict(result.labels),
            "k_found": dict(result.k_found),
            "k_selection_methods": dict(result.k_selection_methods),
            "coassoc_matrix": result.coassoc_matrix,
            "monti2_coassoc_matrix": getattr(result, "monti2_coassoc_matrix", None),
            "run_weights": result.run_weights,
            "rows": _rows,
            "builtin_sel": list(builtin_sel),
            "user_sel": list(user_sel),
            "methods": list(methods or []),
        }
        st.session_state["_last_consensus_context"] = {
            "dataset_name": ds_name,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "k_true": int(len(np.unique(y_true))) if y_true is not None else None,
            "section": "Консенсус-анализ",
            "consensus_results": _rows,
            "base_algorithms": list(builtin_sel) + list(user_sel),
        }
        st.session_state["_agent_chat_consensus"] = []

    _consensus_state = st.session_state.get("_consensus_state")
    if _consensus_state:
        _cX = _consensus_state["X"]
        _cy = _consensus_state["y_true"]
        _c_ds_name = _consensus_state["ds_name"]
        _c_labels = _consensus_state["labels"]
        _c_k_found = _consensus_state["k_found"]
        _c_ksm = _consensus_state["k_selection_methods"]
        _c_coassoc = _consensus_state["coassoc_matrix"]
        _c_monti2_coassoc = _consensus_state["monti2_coassoc_matrix"]
        _c_weights = _consensus_state["run_weights"]
        _c_rows = _consensus_state["rows"]
        _c_builtin_sel = _consensus_state["builtin_sel"]
        _c_user_sel = _consensus_state["user_sel"]
        _c_methods = _consensus_state["methods"]

        st.markdown("---")
        st.subheader("Результаты")

        if _c_rows:
            import pandas as _pd
            _df = _pd.DataFrame(_c_rows).set_index("Метод")
            st.dataframe(_df, width="stretch")

        if _c_k_found:
            _krow = "  |  ".join(
                f"**{m}**: k={k}" for m, k in _c_k_found.items()
            )
            st.caption(f"Найдено кластеров — {_krow}")

        if _c_weights is not None:
            with st.expander("Качество базовых прогонов (веса применяются ко всем методам консенсуса)"):
                _run_names = list(_c_builtin_sel) + [n[len("[U] "):] for n in _c_user_sel]
                _n_show = min(len(_run_names), len(_c_weights))
                _max_w = max(float(_c_weights.max()), 1e-9)
                for _ri in range(_n_show):
                    _bar = int(_c_weights[_ri] / _max_w * 25)
                    st.text(f"{_run_names[_ri]:20s}: {'█'*_bar} {_c_weights[_ri]:.3f}")
                st.caption(
                    "Вес = покрытие × нормированная энтропия кластеров. "
                    "Одинаков для всех методов консенсуса, т.к. вычисляется один раз из прогонов базовых алгоритмов."
                )
        elif "monti2" in (_c_methods or []) and not _c_builtin_sel and not _c_user_sel:
            st.caption(
                "Monti2: веса ensemble не считались (нет базового multiselect). "
                "На вкладке слева — матрица co-association **только Monti2** "
                "(``monti2_coassoc_matrix``; глобальная ``coassoc_matrix`` не смешивается)."
            )

        from visualization.plots import plot_coassociation, plot_cluster_projection

        tab_names = ["Co-association"] + [f"{m} (k={_c_k_found.get(m,'?')})" for m in _c_labels]
        tabs = st.tabs(tab_names)

        with tabs[0]:
            _first_lbl = next(iter(_c_labels.values()), None)
            _co_mat = _c_coassoc if _c_coassoc is not None else _c_monti2_coassoc
            _co_title = (
                "Co-association Matrix"
                if _c_coassoc is not None
                else "Co-association Matrix (Monti2)"
            )
            fig = plot_safe(plot_coassociation, _co_mat, _first_lbl,
                            title=_co_title, figsize=(5, 4))
            if fig: st.pyplot(fig); close_fig(fig)

        for _ti, (_mname, _mlabels) in enumerate(_c_labels.items(), start=1):
            with tabs[_ti]:
                _k_info = _c_k_found.get(_mname, "?")
                _ksm    = _c_ksm.get(_mname, "")
                _t_info = f"{_mname} · k={_k_info}"
                if _ksm:
                    _t_info += f" (метод: {_ksm})"
                fig = plot_safe(
                    plot_cluster_projection, _cX, _mlabels,
                    title=f"{_c_ds_name} — {_t_info}",
                )
                if fig: st.pyplot(fig); close_fig(fig)

        st.markdown("---")
        if st.button("Сбросить результаты консенсуса", key="con_reset_results"):
            for _ck in ("_consensus_state", "_last_consensus_context",
                        "_agent_chat_consensus"):
                st.session_state.pop(_ck, None)
            st.rerun()
        _render_agent_chat(
            context_key="_last_consensus_context",
            chat_key="_agent_chat_consensus",
            title="Спросить AI-агента о консенсусе",
        )


