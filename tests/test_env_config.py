"""Tests for LiteLLM-compatible .env configuration and auth middleware."""

from __future__ import annotations

import os
import tempfile
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prellm.env_config import (
    EnvConfig,
    get_env_config,
    load_dotenv_if_available,
    check_providers,
    PROVIDER_KEY_MAP,
)


class TestEnvConfigDefaults:
    def test_default_config(self):
        cfg = EnvConfig()
        assert cfg.small_model == "ollama/qwen2.5:3b"
        assert cfg.large_model == "gpt-4o-mini"
        assert cfg.strategy == "classify"
        assert cfg.master_key is None
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8080
        assert cfg.fallbacks == []
        assert cfg.monthly_budget is None

    def test_provider_key_map_has_all_providers(self):
        assert "openai" in PROVIDER_KEY_MAP
        assert "anthropic" in PROVIDER_KEY_MAP
        assert "groq" in PROVIDER_KEY_MAP
        assert "mistral" in PROVIDER_KEY_MAP
        assert "azure" in PROVIDER_KEY_MAP
        assert "ollama" in PROVIDER_KEY_MAP


class TestLoadDotenv:
    def test_load_env_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "PRELLM_SMALL_DEFAULT=test-small\n"
            "PRELLM_LARGE_DEFAULT=test-large\n"
            "LITELLM_MASTER_KEY=sk-test-123\n"
            "# This is a comment\n"
            "\n"
            "PRELLM_STRATEGY=structure\n"
        )

        # Clear any existing env vars
        for key in ["PRELLM_SMALL_DEFAULT", "PRELLM_LARGE_DEFAULT", "LITELLM_MASTER_KEY", "PRELLM_STRATEGY"]:
            os.environ.pop(key, None)

        load_dotenv_if_available(str(env_file))

        assert os.environ.get("PRELLM_SMALL_DEFAULT") == "test-small"
        assert os.environ.get("PRELLM_LARGE_DEFAULT") == "test-large"
        assert os.environ.get("LITELLM_MASTER_KEY") == "sk-test-123"
        assert os.environ.get("PRELLM_STRATEGY") == "structure"

        # Cleanup
        for key in ["PRELLM_SMALL_DEFAULT", "PRELLM_LARGE_DEFAULT", "LITELLM_MASTER_KEY", "PRELLM_STRATEGY"]:
            os.environ.pop(key, None)

    def test_load_env_does_not_override_existing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_PRELLM_VAR=from_file\n")

        os.environ["TEST_PRELLM_VAR"] = "from_env"
        load_dotenv_if_available(str(env_file))

        assert os.environ["TEST_PRELLM_VAR"] == "from_env"
        os.environ.pop("TEST_PRELLM_VAR", None)

    def test_load_env_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            'TEST_PRELLM_QUOTED="quoted_value"\n'
            "TEST_PRELLM_SINGLE='single_quoted'\n"
        )

        for key in ["TEST_PRELLM_QUOTED", "TEST_PRELLM_SINGLE"]:
            os.environ.pop(key, None)

        load_dotenv_if_available(str(env_file))

        assert os.environ.get("TEST_PRELLM_QUOTED") == "quoted_value"
        assert os.environ.get("TEST_PRELLM_SINGLE") == "single_quoted"

        for key in ["TEST_PRELLM_QUOTED", "TEST_PRELLM_SINGLE"]:
            os.environ.pop(key, None)

    def test_load_nonexistent_file(self):
        load_dotenv_if_available("/nonexistent/.env")
        # Should not raise


class TestGetEnvConfig:
    def test_reads_from_env(self):
        env_vars = {
            "PRELLM_SMALL_DEFAULT": "phi3:mini",
            "PRELLM_LARGE_DEFAULT": "anthropic/claude-sonnet-4-20250514",
            "PRELLM_STRATEGY": "enrich",
            "LITELLM_MASTER_KEY": "sk-master-key",
            "PRELLM_HOST": "127.0.0.1",
            "PRELLM_PORT": "9090",
            "PRELLM_FALLBACKS": "gpt-4o-mini,llama3",
            "PRELLM_MONTHLY_BUDGET": "50.0",
            "PRELLM_LOG_LEVEL": "debug",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            cfg = get_env_config()

        assert cfg.small_model == "phi3:mini"
        assert cfg.large_model == "anthropic/claude-sonnet-4-20250514"
        assert cfg.strategy == "enrich"
        assert cfg.master_key == "sk-master-key"
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 9090
        assert cfg.fallbacks == ["gpt-4o-mini", "llama3"]
        assert cfg.monthly_budget == 50.0
        assert cfg.log_level == "debug"

    def test_reads_from_dotenv_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "PRELLM_SMALL_DEFAULT=dotenv-small\n"
            "PRELLM_LARGE_DEFAULT=dotenv-large\n"
        )

        for key in ["PRELLM_SMALL_DEFAULT", "PRELLM_LARGE_DEFAULT"]:
            os.environ.pop(key, None)

        cfg = get_env_config(str(env_file))

        assert cfg.small_model == "dotenv-small"
        assert cfg.large_model == "dotenv-large"

        for key in ["PRELLM_SMALL_DEFAULT", "PRELLM_LARGE_DEFAULT"]:
            os.environ.pop(key, None)

    def test_litellm_compat_env_vars(self):
        """LiteLLM env vars (OPENAI_API_KEY etc.) are detected in providers."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test-openai",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            cfg = get_env_config()

        assert cfg.providers["openai"]["has_key"] is True
        assert cfg.providers["anthropic"]["has_key"] is True
        assert cfg.providers["groq"]["has_key"] is False

    def test_small_model_fallback_to_old_env(self):
        """SMALL_MODEL env var (old name) still works."""
        with patch.dict(os.environ, {"SMALL_MODEL": "old-small"}, clear=False):
            os.environ.pop("PRELLM_SMALL_DEFAULT", None)
            cfg = get_env_config()

        assert cfg.small_model == "old-small"
        os.environ.pop("SMALL_MODEL", None)

    def test_empty_master_key_is_none(self):
        with patch.dict(os.environ, {"LITELLM_MASTER_KEY": ""}, clear=False):
            cfg = get_env_config()
        assert cfg.master_key is None

    def test_ollama_base_url(self):
        with patch.dict(os.environ, {"OLLAMA_API_BASE": "http://ollama:11434"}, clear=False):
            cfg = get_env_config()
        assert cfg.providers["ollama"]["base_url"] == "http://ollama:11434"


class TestCheckProviders:
    def test_check_with_keys(self):
        env_vars = {
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            cfg = get_env_config()
            results = check_providers(cfg)

        assert results["openai"]["status"] == "configured"
        assert results["anthropic"]["status"] == "configured"
        assert results["ollama"]["status"] == "configured"

    def test_check_without_keys(self):
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"]:
            os.environ.pop(key, None)

        cfg = get_env_config()
        results = check_providers(cfg)

        assert results["openai"]["status"] == "no_key"
        assert results["anthropic"]["status"] == "no_key"
        assert results["ollama"]["status"] == "configured"


class TestAuthMiddleware:
    """Test Bearer token auth in the API server."""

    def test_no_auth_when_no_master_key(self):
        from prellm.server import app, create_app
        create_app(master_key="")

        from fastapi.testclient import TestClient
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200

        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_resp("OK")
        )):
            resp = client.post("/v1/chat/completions", json={
                "model": "prellm:default",
                "messages": [{"role": "user", "content": "test"}],
                "prellm": {"strategy": "passthrough"},
            })
            assert resp.status_code == 200

    def test_auth_required_when_master_key_set(self):
        from prellm.server import app, create_app
        create_app(master_key="sk-test-secret")

        from fastapi.testclient import TestClient
        client = TestClient(app)

        # Health is always open
        resp = client.get("/health")
        assert resp.status_code == 200

        # No auth → 401
        resp = client.post("/v1/chat/completions", json={
            "model": "prellm:default",
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["error"]["message"]

    def test_auth_with_bearer_token(self):
        from prellm.server import app, create_app
        create_app(master_key="sk-test-secret")

        from fastapi.testclient import TestClient
        client = TestClient(app)

        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_resp("Authed response")
        )):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "prellm:default",
                    "messages": [{"role": "user", "content": "test"}],
                    "prellm": {"strategy": "passthrough"},
                },
                headers={"Authorization": "Bearer sk-test-secret"},
            )
            assert resp.status_code == 200
            assert resp.json()["choices"][0]["message"]["content"] == "Authed response"

    def test_auth_with_x_api_key(self):
        from prellm.server import app, create_app
        create_app(master_key="sk-test-secret")

        from fastapi.testclient import TestClient
        client = TestClient(app)

        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_resp("X-API response")
        )):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "prellm:default",
                    "messages": [{"role": "user", "content": "test"}],
                    "prellm": {"strategy": "passthrough"},
                },
                headers={"x-api-key": "sk-test-secret"},
            )
            assert resp.status_code == 200

    def test_wrong_key_rejected(self):
        from prellm.server import app, create_app
        create_app(master_key="sk-correct")

        from fastapi.testclient import TestClient
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "prellm:default", "messages": [{"role": "user", "content": "test"}]},
            headers={"Authorization": "Bearer sk-wrong"},
        )
        assert resp.status_code == 401

    def teardown_method(self):
        """Reset master key after each test."""
        from prellm.server import create_app
        create_app(master_key="")


def _mock_resp(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp
