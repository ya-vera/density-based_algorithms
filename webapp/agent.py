from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


KNOWLEDGE_BASE = """
БАЗОВЫЕ АЛГОРИТМЫ:
- DBSCAN: ищет плотные ε-окрестности. Параметры: eps (радиус), min_samples (минимум точек в окрестности). Точки в редких областях помечаются как шум (-1).
- HDBSCAN: иерархическая версия DBSCAN, сам подбирает плотность. Параметры: min_cluster_size, min_samples.
- DPC (Density Peaks): находит точки с высокой локальной плотностью ρ и большим расстоянием δ до более плотного соседа. Параметр: percent (для cutoff dc).
- RD-DAC: иерархическое разделение по KNN-графу. Параметр: k (число соседей).
- CKDPC: KNN-плотность + DPC. Параметры: percent, alpha, k_neighbors.

МЕТОДЫ КОНСЕНСУСА:
- CoAssoc: строит co-association матрицу C[i,j] = доля прогонов, где i и j оказались в одном кластере. Кластеризует её через average-linkage.
- Voting: мажоритарное голосование по меткам базовых алгоритмов.
- Monti (Monti2): bootstrap-субсемплирование, оптимальное k выбирается через PAC-score на CDF-кривых co-association матрицы.
- CoHiRF: random-forest подход на основе медоидов.
- FCA: формальный концептный анализ co-association матрицы через библиотеку caspailleur.

CO-ASSOCIATION MATRIX: симметричная матрица n×n, где элемент [i,j] = доля прогонов базовых алгоритмов, в которых объекты i и j оказались в одном кластере. Строится из label_matrix всех M прогонов: C[i,j] = (1/M) · Σ 1[π_m(i)=π_m(j)].

МЕТРИКИ:
- ARI (Adjusted Rand Index): согласованность с y_true, [-1, 1], 1 = идеально.
- AMI/NMI: взаимная информация с y_true.
- FMI: Фолкса-Мэллоуса, сходство пар.
- SC (Silhouette): внутренняя метрика разделимости, [-1, 1].
- CHI (Calinski-Harabasz): дисперсия между / внутри кластеров.
- DBI (Davies-Bouldin): чем меньше — тем лучше.

РЕЖИМЫ ПОДБОРА ПАРАМЕТРОВ В СИСТЕМЕ:
- «Авто»: одна эвристика на основе размера данных (быстро, может промахиваться).
- «Перебор по сетке» (grid search): перебор до 80 комбинаций, выбор по ARI/Silhouette (медленнее, точнее).
- «Ручной»: пользователь задаёт значения сам.
"""

SYSTEM_PROMPT = (
    "Ты — русскоязычный ассистент по плотностной кластеризации, встроенный "
    "в систему тестирования алгоритмов. Тебе передан КОНТЕКСТ с текущим "
    "состоянием (датасет, метрики, параметры) и ВОПРОС пользователя.\n\n"
    "ЯЗЫК:\n"
    "- Отвечай ТОЛЬКО на русском языке. Никогда не используй английские, "
    "немецкие, китайские и другие иностранные слова в обычной речи. "
    "Пиши «возможно», а не «möglich»; «рекомендую», а не «recommend».\n"
    "- Названия алгоритмов (DBSCAN, HDBSCAN, DPC, RD-DAC, CKDPC, Monti, "
    "CoAssoc, Voting, CoHiRF, FCA), параметров (eps, min_samples, k, dc) "
    "и метрик (ARI, AMI, NMI, FMI, SC, CHI, DBI) — оставляй как есть.\n"
    "- Никаких иероглифов и транслитерации.\n\n"
    "КАК ОТВЕЧАТЬ — определи тип вопроса:\n"
    "1) Если вопрос НЕ про кластеризацию/данные/метрики (например, «где "
    "купить билеты», «расскажи анекдот», «погода») — кратко ответь: «Я "
    "помогаю только с анализом результатов кластеризации в этой системе. "
    "Задай вопрос про алгоритмы, метрики или данные.» — и ничего больше.\n"
    "2) Если вопрос концептуальный («как работает X», «что такое Y», «как "
    "построена матрица») — объясни теорию по существу, используя «БАЗА "
    "ЗНАНИЙ» ниже. Не давай рекомендаций про кнопки.\n"
    "3) Если вопрос диагностический («почему плохой результат», «какой "
    "метод лучше», «что улучшить») — опирайся на цифры из контекста, "
    "сначала вывод, потом конкретное действие.\n\n"
    "ОБЩИЕ ПРАВИЛА:\n"
    "- Опирайся ТОЛЬКО на цифры из контекста, не выдумывай ARI/k/шум.\n"
    "- Не повторяй один и тот же шаблон «вывод + рекомендация Авто» "
    "везде. Если уже использован Авто-режим — предложи «Перебор по сетке».\n"
    "- Если данных в контексте мало или они отсутствуют — честно скажи.\n"
    "- Кратко: 2–5 предложений. Без воды.\n\n"
    "БАЗА ЗНАНИЙ (используй для концептуальных ответов):\n"
    + KNOWLEDGE_BASE
)


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class AgentContext:
    dataset_name: Optional[str] = None
    n_samples: Optional[int] = None
    n_features: Optional[int] = None
    k_true: Optional[int] = None
    section: Optional[str] = None
    algorithm_results: List[Dict[str, Any]] = field(default_factory=list)
    consensus_results: List[Dict[str, Any]] = field(default_factory=list)
    extra_notes: List[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        lines: List[str] = ["### КОНТЕКСТ СИСТЕМЫ ###"]
        if self.section:
            lines.append(f"Раздел интерфейса: {self.section}")
        if self.dataset_name:
            ds_line = f"Датасет: {self.dataset_name}"
            if self.n_samples is not None:
                ds_line += f", n={self.n_samples}"
            if self.n_features is not None:
                ds_line += f", d={self.n_features}"
            if self.k_true is not None:
                ds_line += f", k_true={self.k_true}"
            lines.append(ds_line)
        else:
            lines.append("Датасет не выбран.")

        if self.algorithm_results:
            lines.append("\nРезультаты базовых алгоритмов:")
            for row in self.algorithm_results:
                lines.append(self._format_alg_row(row))
        else:
            lines.append("\nБазовые алгоритмы пока не запущены.")

        if self.consensus_results:
            lines.append("\nРезультаты методов консенсуса:")
            for row in self.consensus_results:
                lines.append(self._format_consensus_row(row))

        for note in self.extra_notes:
            lines.append(f"\nЗаметка: {note}")
        return "\n".join(lines)

    @staticmethod
    def _format_alg_row(row: Dict[str, Any]) -> str:
        name = row.get("Алгоритм") or row.get("algorithm") or "?"
        k = row.get("k", "?")
        noise = row.get("шум", row.get("noise", "?"))
        ari = row.get("ARI", "—")
        ami = row.get("AMI", "—")
        sc = row.get("SC", "—")
        params = row.get("параметры", row.get("params", ""))
        return (
            f"- {name}: k={k}, шум={noise}, ARI={ari}, AMI={ami}, SC={sc}"
            f"{', параметры: ' + str(params) if params else ''}"
        )

    @staticmethod
    def _format_consensus_row(row: Dict[str, Any]) -> str:
        name = row.get("method") or row.get("Метод") or "?"
        k = row.get("k_found", row.get("k", "?"))
        ari = row.get("ARI", "—")
        ami = row.get("AMI", "—")
        nmi = row.get("NMI", "—")
        sec = row.get("sec", row.get("time", "—"))
        return f"- {name}: k_found={k}, ARI={ari}, AMI={ami}, NMI={nmi}, time={sec}"


class LLMError(RuntimeError):
    pass


class LLMBackend:
    name = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def chat(self, system: str, history: List[ChatMessage]) -> str:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434",
                 timeout: int = 300):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def chat(self, system: str, history: List[ChatMessage]) -> str:
        messages = [{"role": "system", "content": system}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192,
                "top_p": 0.9,
                "repeat_penalty": 1.15,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as e:
            raise LLMError(f"Не удалось связаться с Ollama: {e}") from e
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as e:
            raise LLMError(f"Ollama вернул некорректный ответ: {body[:200]}") from e
        msg = parsed.get("message", {})
        text = msg.get("content", "").strip()
        if not text:
            raise LLMError(f"Ollama не вернул текст. Полный ответ: {body[:200]}")
        return text


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.openai.com/v1",
                 timeout: int = 120):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, history: List[ChatMessage]) -> str:
        if not self.api_key:
            raise LLMError("OpenAI API key не задан.")
        messages = [{"role": "system", "content": system}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        payload = {"model": self.model, "messages": messages, "temperature": 0.2}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            raise LLMError(f"OpenAI HTTP {e.code}: {err_body[:300]}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"OpenAI недоступен: {e}") from e
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as e:
            raise LLMError(f"OpenAI вернул некорректный JSON: {body[:200]}") from e
        choices = parsed.get("choices") or []
        if not choices:
            raise LLMError(f"OpenAI не вернул choices. Полный ответ: {body[:300]}")
        return choices[0].get("message", {}).get("content", "").strip()


class HeuristicBackend(LLMBackend):
    name = "heuristic"

    def is_available(self) -> bool:
        return True

    def chat(self, system: str, history: List[ChatMessage]) -> str:
        ctx_text = next((m.content for m in history if m.role == "user"), "")
        rules: List[str] = []
        low = ctx_text.lower()
        if "шум=" in ctx_text:
            try:
                noise_vals = []
                for line in ctx_text.splitlines():
                    if "шум=" in line:
                        frag = line.split("шум=", 1)[1].split(",", 1)[0]
                        pct = float(frag.replace("%", "").strip().replace("—", "0"))
                        noise_vals.append(pct)
                if noise_vals and max(noise_vals) > 50:
                    rules.append(
                        "Высокий уровень шума (>50%). Для DBSCAN/HDBSCAN это значит, "
                        "что плотностный порог слишком строгий. Попробуй увеличить "
                        "eps (DBSCAN) или уменьшить min_cluster_size (HDBSCAN). "
                        "Самый простой путь — нажать «Авто» в режиме параметров."
                    )
            except Exception:
                pass
        if "k=1" in ctx_text or " k=1," in ctx_text:
            rules.append(
                "Алгоритм нашёл всего 1 кластер. Скорее всего, параметр чувствительности "
                "слишком мягкий: попробуй уменьшить eps (DBSCAN), уменьшить dc (DPC), "
                "или включить «Перебор по сетке» для автоматического подбора."
            )
        if "ari=—" in low or "ari=-" in low:
            rules.append(
                "Метрика ARI не посчиталась — обычно это означает k_found<2 (один "
                "кластер или всё шум). См. пункт выше."
            )
        if not rules:
            rules.append(
                "Не задано подключение к LLM (Ollama/OpenAI). "
                "Включи один из бэкендов в боковой панели, "
                "чтобы получать развёрнутые ответы. Сейчас доступны только базовые "
                "эвристики, и в твоих результатах явных проблем не обнаружено."
            )
        return "\n\n".join(rules)


def build_backend(kind: str, **kwargs: Any) -> LLMBackend:
    if kind == "ollama":
        return OllamaBackend(
            model=kwargs.get("model", "qwen2.5:7b"),
            host=kwargs.get("host", "http://localhost:11434"),
        )
    if kind == "openai":
        return OpenAIBackend(
            model=kwargs.get("model", "gpt-4o-mini"),
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url", "https://api.openai.com/v1"),
        )
    return HeuristicBackend()


def ask_agent(backend: LLMBackend, context: AgentContext,
              user_question: str,
              chat_history: Optional[List[ChatMessage]] = None,
              max_history_turns: int = 2) -> str:
    full_history: List[ChatMessage] = list(chat_history or [])
    trimmed: List[ChatMessage] = full_history[-(2 * max_history_turns):]

    user_message_content = (
        context.to_prompt()
        + "\n\n### ВОПРОС ПОЛЬЗОВАТЕЛЯ ###\n"
        + user_question.strip()
        + "\n\nОтвечай только на русском языке. Если вопрос не про "
        "кластеризацию/данные/метрики — скажи об этом одной фразой."
    )
    messages: List[ChatMessage] = trimmed + [
        ChatMessage(role="user", content=user_message_content)
    ]
    return backend.chat(SYSTEM_PROMPT, messages)
