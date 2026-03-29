"""Tests for env_config storage helpers and config_set persistence."""

from __future__ import annotations

import os
from pathlib import Path

from prellm.env_config import config_set
from prellm.env_config import storage


def test_write_env_file_plain_backend(tmp_path, monkeypatch):
    """Plain .env writes should preserve comments, update values, and append new keys."""
    monkeypatch.setattr(storage, "_get_env_store_class", lambda: None)

    path = tmp_path / ".env"
    path.write_text(
        "# existing comment\n"
        "FOO=old\n"
        "BAR=keep\n"
    )

    storage.write_env_file(path, {"FOO": "new", "BAZ": "3"}, comments=["generated"])

    assert path.read_text() == (
        "# generated\n"
        "# existing comment\n"
        "FOO=new\n"
        "BAR=keep\n"
        "BAZ=3\n"
    )


def test_write_env_file_env_store_backend(tmp_path, monkeypatch):
    """If getv is available, EnvStore should receive update/save calls."""

    class FakeEnvStore:
        instances: list["FakeEnvStore"] = []

        def __init__(self, path: Path):
            self.path = path
            self.entries: dict[str, str] = {}
            self.saved = False
            FakeEnvStore.instances.append(self)

        def update(self, entries: dict[str, str]) -> None:
            self.entries.update(entries)

        def save(self) -> None:
            self.saved = True

    FakeEnvStore.instances.clear()
    monkeypatch.setattr(storage, "_get_env_store_class", lambda: FakeEnvStore)

    path = tmp_path / ".env"
    storage.write_env_file(path, {"FOO": "bar"}, comments=["ignored"])

    assert len(FakeEnvStore.instances) == 1
    instance = FakeEnvStore.instances[0]
    assert instance.path == path
    assert instance.entries == {"FOO": "bar"}
    assert instance.saved is True
    assert not path.exists()


def test_config_set_writes_project_env_file(tmp_path, monkeypatch):
    """config_set should resolve aliases and persist values to the current project .env."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prellm.env_config.commands._get_env_store_class", lambda: None)
    monkeypatch.setattr("prellm.env_config.storage._get_env_store_class", lambda: None)
    os.environ.pop("PRELLM_SMALL_DEFAULT", None)

    try:
        env_var, path = config_set("small-model", "tiny")

        assert env_var == "PRELLM_SMALL_DEFAULT"
        assert path == Path(".env")
        assert (tmp_path / ".env").read_text() == "PRELLM_SMALL_DEFAULT=tiny\n"
        assert os.environ[env_var] == "tiny"
    finally:
        os.environ.pop("PRELLM_SMALL_DEFAULT", None)
