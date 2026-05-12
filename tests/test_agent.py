import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webapp.agent import (
    AgentContext,
    ChatMessage,
    HeuristicBackend,
    LLMError,
    OllamaBackend,
    OpenAIBackend,
    SYSTEM_PROMPT,
    ask_agent,
    build_backend,
)


def _mock_urlopen_response(payload: dict):
    body = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    resp.read = MagicMock(return_value=body)
    resp.status = 200
    return resp


class TestAgentContext:
    def test_to_prompt_includes_dataset_metadata(self):
        ctx = AgentContext(
            dataset_name="iris", n_samples=150, n_features=4, k_true=3,
            section="Сравнение",
        )
        text = ctx.to_prompt()
        assert "iris" in text
        assert "n=150" in text
        assert "d=4" in text
        assert "k_true=3" in text
        assert "Сравнение" in text

    def test_to_prompt_empty_context(self):
        ctx = AgentContext()
        text = ctx.to_prompt()
        assert "Датасет не выбран" in text
        assert "не запущены" in text

    def test_to_prompt_with_algorithm_results(self):
        ctx = AgentContext(
            dataset_name="iris", n_samples=150, n_features=4, k_true=3,
            algorithm_results=[
                {"Алгоритм": "DBSCAN", "k": 0, "шум": "90%",
                 "ARI": "—", "AMI": "—", "SC": "—",
                 "параметры": "eps=0.05"},
                {"Алгоритм": "HDBSCAN", "k": 3, "шум": "5%",
                 "ARI": "0.82", "AMI": "0.81", "SC": "0.5",
                 "параметры": "mcs=10"},
            ],
        )
        text = ctx.to_prompt()
        assert "DBSCAN" in text
        assert "HDBSCAN" in text
        assert "ARI=0.82" in text
        assert "eps=0.05" in text

    def test_to_prompt_with_consensus_results(self):
        ctx = AgentContext(
            dataset_name="wine",
            consensus_results=[
                {"method": "Monti2", "k_found": 3, "ARI": 0.75, "AMI": 0.7,
                 "NMI": 0.71, "sec": 1.2},
                {"method": "CoAssoc", "k_found": 3, "ARI": 0.8, "AMI": 0.75,
                 "NMI": 0.76, "sec": 0.8},
            ],
        )
        text = ctx.to_prompt()
        assert "Monti2" in text
        assert "CoAssoc" in text
        assert "ARI=0.75" in text


class TestHeuristicBackend:
    def test_is_always_available(self):
        backend = HeuristicBackend()
        assert backend.is_available() is True

    def test_detects_high_noise(self):
        backend = HeuristicBackend()
        msg = ChatMessage(
            role="user",
            content="- DBSCAN: k=0, шум=92%, ARI=—",
        )
        answer = backend.chat(SYSTEM_PROMPT, [msg])
        assert "шум" in answer.lower()
        assert "eps" in answer.lower() or "min_cluster_size" in answer.lower()

    def test_detects_single_cluster(self):
        backend = HeuristicBackend()
        msg = ChatMessage(
            role="user",
            content="- HDBSCAN: k=1, шум=5%, ARI=0.0",
        )
        answer = backend.chat(SYSTEM_PROMPT, [msg])
        assert "1 кластер" in answer.lower() or "один кластер" in answer.lower()

    def test_no_problems_returns_default_hint(self):
        backend = HeuristicBackend()
        msg = ChatMessage(
            role="user",
            content="- DBSCAN: k=3, шум=10%, ARI=0.85",
        )
        answer = backend.chat(SYSTEM_PROMPT, [msg])
        assert "LLM" in answer or "проблем" in answer.lower()


class TestOllamaBackend:
    def test_is_available_when_server_responds(self):
        backend = OllamaBackend(model="test", host="http://localhost:11434")
        fake_resp = MagicMock()
        fake_resp.__enter__ = MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = MagicMock(return_value=False)
        fake_resp.status = 200
        with patch("urllib.request.urlopen", return_value=fake_resp):
            assert backend.is_available() is True

    def test_is_unavailable_on_connection_error(self):
        import urllib.error
        backend = OllamaBackend(model="test", host="http://localhost:11434")
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.URLError("nope")):
            assert backend.is_available() is False

    def test_chat_returns_assistant_content(self):
        backend = OllamaBackend(model="test", host="http://localhost:11434")
        resp = _mock_urlopen_response(
            {"message": {"role": "assistant", "content": "Ответ агента."}}
        )
        with patch("urllib.request.urlopen", return_value=resp):
            text = backend.chat("system", [ChatMessage("user", "вопрос")])
        assert text == "Ответ агента."

    def test_chat_raises_on_empty_response(self):
        backend = OllamaBackend(model="test", host="http://localhost:11434")
        resp = _mock_urlopen_response({"message": {"content": ""}})
        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(LLMError):
                backend.chat("system", [ChatMessage("user", "вопрос")])

    def test_chat_raises_on_invalid_json(self):
        backend = OllamaBackend(model="test", host="http://localhost:11434")
        resp = MagicMock()
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        resp.read = MagicMock(return_value=b"not json")
        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(LLMError):
                backend.chat("system", [ChatMessage("user", "вопрос")])


class TestOpenAIBackend:
    def test_is_available_requires_api_key(self):
        assert OpenAIBackend(api_key="").is_available() is False
        assert OpenAIBackend(api_key="sk-test").is_available() is True

    def test_chat_returns_choices_content(self):
        backend = OpenAIBackend(api_key="sk-test")
        resp = _mock_urlopen_response(
            {"choices": [{"message": {"role": "assistant", "content": "Привет."}}]}
        )
        with patch("urllib.request.urlopen", return_value=resp):
            text = backend.chat("system", [ChatMessage("user", "вопрос")])
        assert text == "Привет."

    def test_chat_raises_without_key(self):
        backend = OpenAIBackend(api_key="")
        with pytest.raises(LLMError):
            backend.chat("system", [ChatMessage("user", "вопрос")])

    def test_chat_raises_on_empty_choices(self):
        backend = OpenAIBackend(api_key="sk-test")
        resp = _mock_urlopen_response({"choices": []})
        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(LLMError):
                backend.chat("system", [ChatMessage("user", "вопрос")])


class TestBuildBackend:
    def test_ollama_kind(self):
        b = build_backend("ollama", model="custom", host="http://x:1")
        assert isinstance(b, OllamaBackend)
        assert b.model == "custom"

    def test_openai_kind(self):
        b = build_backend("openai", api_key="sk-test")
        assert isinstance(b, OpenAIBackend)

    def test_heuristic_fallback(self):
        b = build_backend("unknown")
        assert isinstance(b, HeuristicBackend)


class TestAskAgent:
    def test_passes_context_in_user_message(self):
        ctx = AgentContext(dataset_name="iris", n_samples=150, n_features=4, k_true=3)
        captured = {}

        class StubBackend(HeuristicBackend):
            def chat(self, system, history):
                captured["system"] = system
                captured["history"] = history
                return "ответ"

        ask_agent(StubBackend(), ctx, "какой метод лучше?")
        assert "iris" in captured["history"][-1].content
        assert "какой метод лучше?" in captured["history"][-1].content
        assert "русск" in captured["system"].lower()

    def test_trims_old_history(self):
        ctx = AgentContext(dataset_name="iris")
        captured = {}

        class StubBackend(HeuristicBackend):
            def chat(self, system, history):
                captured["count"] = len(history)
                return "ответ"

        old_history = [
            ChatMessage("user", "q1"), ChatMessage("assistant", "a1"),
            ChatMessage("user", "q2"), ChatMessage("assistant", "a2"),
            ChatMessage("user", "q3"), ChatMessage("assistant", "a3"),
            ChatMessage("user", "q4"), ChatMessage("assistant", "a4"),
        ]
        ask_agent(StubBackend(), ctx, "новый вопрос",
                  chat_history=old_history, max_history_turns=2)
        assert captured["count"] == 5
