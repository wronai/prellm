"""Tests for preLLM OpenAI-compatible API server."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from prellm.server import app, create_app, _parse_model_pair, SMALL_MODEL, LARGE_MODEL


def _mock_litellm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_completion_side_effect():
    """Returns a side_effect that handles small LLM (classify+compose) then large LLM."""
    call_count = 0

    async def mock(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return _mock_litellm_response('{"intent": "general", "confidence": 0.8, "domain": "general"}')
        return _mock_litellm_response("Server response content")

    return mock


class TestParseModelPair:
    def test_default(self):
        small, large = _parse_model_pair("prellm:default")
        assert small == SMALL_MODEL
        assert large == LARGE_MODEL

    def test_arrow_unicode(self):
        small, large = _parse_model_pair("prellm:qwen→claude")
        assert small == "qwen"
        assert large == "claude"

    def test_arrow_ascii(self):
        small, large = _parse_model_pair("prellm:phi3->gpt-4o")
        assert small == "phi3"
        assert large == "gpt-4o"

    def test_single_model(self):
        small, large = _parse_model_pair("prellm:gpt-4o-mini")
        assert "kimi" in large or "gpt-4" in large or "claude" in large

    def test_no_prefix(self):
        small, large = _parse_model_pair("gpt-4o-mini")
        assert small == SMALL_MODEL
        assert large == LARGE_MODEL  # no colon → returns server defaults


class TestHealthEndpoint:
    def test_health(self):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.4.3"

    def test_models(self):
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1


class TestChatCompletions:
    def test_basic_completion(self):
        client = TestClient(app)
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Server response content")
        )):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:default",
                "messages": [{"role": "user", "content": "Hello world"}],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Server response content"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_with_prellm_extras(self):
        client = TestClient(app)
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Server response content")
        )):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:qwen→claude",
                "messages": [{"role": "user", "content": "Deploy app"}],
                "prellm": {
                    "user_context": "gdansk embedded python",
                    "strategy": "classify",
                },
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["prellm_meta"] is not None
        assert data["prellm_meta"]["strategy"] == "classify"

    def test_with_yaml_response_format(self):
        client = TestClient(app)
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Server response content")
        )):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:default",
                "messages": [{"role": "user", "content": "Generate config"}],
                "prellm": {"response_format": "yaml"},
            })

        assert resp.status_code == 200

    def test_prellm_meta_in_response(self):
        client = TestClient(app)
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Server response content")
        )):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:default",
                "messages": [{"role": "user", "content": "Test query"}],
            })

        data = resp.json()
        meta = data["prellm_meta"]
        assert "small_model" in meta
        assert "large_model" in meta
        assert "strategy" in meta
        assert "retries" in meta

    def test_passthrough_strategy(self):
        client = TestClient(app)
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Direct response")
        )):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:default",
                "messages": [{"role": "user", "content": "Simple question"}],
                "prellm": {"strategy": "passthrough"},
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Direct response"
        assert data["prellm_meta"]["strategy"] == "passthrough"

    def test_empty_messages_error(self):
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={
            "model": "prellm:default",
            "messages": [],
        })
        assert resp.status_code == 400

    def test_no_user_message_error(self):
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={
            "model": "prellm:default",
            "messages": [{"role": "system", "content": "You are helpful"}],
        })
        assert resp.status_code == 400

    def test_usage_info(self):
        client = TestClient(app)
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Server response content")
        )):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:default",
                "messages": [{"role": "user", "content": "Hello world"}],
            })

        data = resp.json()
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] > 0
        assert data["usage"]["total_tokens"] > 0

    def test_with_domain_rules(self):
        call_count = 0

        async def mock(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_litellm_response('{"intent": "deploy", "confidence": 0.95, "domain": "devops"}')
            return _mock_litellm_response("Deploy with safety")

        client = TestClient(app)
        with patch("litellm.acompletion", side_effect=mock):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:default",
                "messages": [{"role": "user", "content": "Deploy to production"}],
                "prellm": {
                    "domain_rules": [{
                        "name": "prod_deploy",
                        "keywords": ["deploy", "production"],
                        "intent": "deploy",
                        "required_fields": ["environment", "version"],
                        "severity": "critical",
                    }],
                },
            })

        data = resp.json()
        assert data["prellm_meta"]["matched_rule"] == "prod_deploy"
        assert "environment" in data["prellm_meta"]["missing_fields"]

    def test_max_tokens_and_temperature(self):
        client = TestClient(app)
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Server response content")
        )):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:default",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 4096,
                "temperature": 0.9,
            })

        assert resp.status_code == 200


class TestStreamingEndpoint:
    def test_streaming_response(self):
        client = TestClient(app)
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Server response content")
        )):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:default",
                "messages": [{"role": "user", "content": "Stream test"}],
                "stream": True,
            })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 2

        last_data = data_lines[-1].replace("data: ", "")
        assert last_data == "[DONE]"

    def test_streaming_with_stages(self):
        client = TestClient(app)
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Server response content")
        )):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:default",
                "messages": [{"role": "user", "content": "Stage test"}],
                "stream": True,
                "prellm": {"show_stages": True},
            })

        assert resp.status_code == 200
        text = resp.text
        assert "preprocessing" in text
        assert "execution" in text


class TestBatchEndpoint:
    def test_batch_processing(self):
        call_count = 0

        async def mock(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 in (1, 2):
                return _mock_litellm_response('{"intent": "test", "confidence": 0.5, "domain": "general"}')
            return _mock_litellm_response(f"Response {call_count // 3}")

        client = TestClient(app)
        with patch("litellm.acompletion", side_effect=mock):
            resp = client.post("/v1/batch", json=[
                {"query": "task 1", "context": "python"},
                {"query": "task 2", "context": "devops"},
            ])

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "batch"
        assert len(data["results"]) == 2
        assert data["results"][0]["query"] == "task 1"
        assert data["results"][1]["query"] == "task 2"

    def test_batch_empty_error(self):
        client = TestClient(app)
        resp = client.post("/v1/batch", json=[])
        assert resp.status_code == 400


class TestCreateApp:
    def test_factory_function(self):
        result = create_app(
            small_model="test-small",
            large_model="test-large",
            strategy="structure",
        )
        assert result is app
