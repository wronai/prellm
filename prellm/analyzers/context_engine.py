"""ContextEngine — gathers runtime context (env vars, git info, system state) for prompt enrichment."""

from __future__ import annotations

import os
import subprocess
from typing import Any


class ContextEngine:
    """Collects context from environment, git, and system for prompt enrichment.

    Used by both core Prellm (auto-inject context) and ProcessChain (step-level context).
    """

    def __init__(self, context_sources: list[dict[str, Any]] | None = None):
        self.sources = context_sources or []

    def gather(self) -> dict[str, str]:
        """Gather all configured context into a flat dict."""
        ctx: dict[str, str] = {}

        for source in self.sources:
            if "env" in source:
                ctx.update(self._gather_env(source["env"]))
            if "git" in source:
                ctx.update(self._gather_git(source["git"]))
            if "system" in source:
                ctx.update(self._gather_system(source["system"]))

        return ctx

    def enrich_prompt(self, prompt: str, extra: dict[str, str] | None = None) -> str:
        """Substitute {VARIABLE} placeholders in a prompt with gathered context."""
        ctx = self.gather()
        if extra:
            ctx.update(extra)

        enriched = prompt
        for key, value in ctx.items():
            enriched = enriched.replace(f"{{{key}}}", value)

        return enriched

    @staticmethod
    def _gather_env(keys: list[str]) -> dict[str, str]:
        result = {}
        for key in keys:
            val = os.environ.get(key, "")
            if val:
                result[key] = val
        return result

    @staticmethod
    def _gather_git(fields: list[str]) -> dict[str, str]:
        # Try gitpython first (stable, no subprocess), fall back to subprocess
        try:
            return ContextEngine._gather_git_gitpython(fields)
        except Exception:
            return ContextEngine._gather_git_subprocess(fields)

    @staticmethod
    def _gather_git_gitpython(fields: list[str]) -> dict[str, str]:
        """Gather git info using gitpython (pip install prellm[git])."""
        import git as gitmodule
        result = {}
        try:
            repo = gitmodule.Repo(search_parent_directories=True)
        except (gitmodule.InvalidGitRepositoryError, gitmodule.NoSuchPathError):
            return result

        for field in fields:
            try:
                if field == "branch":
                    if not repo.head.is_detached:
                        result[field] = repo.active_branch.name
                elif field == "last_commit":
                    result[field] = repo.head.commit.hexsha
                elif field == "last_commit_msg":
                    result[field] = repo.head.commit.message.strip()
                elif field == "short_sha":
                    result[field] = repo.head.commit.hexsha[:7]
                elif field == "tag":
                    tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime, reverse=True)
                    if tags:
                        result[field] = tags[0].name
                elif field == "remote_url":
                    if repo.remotes:
                        result[field] = repo.remotes.origin.url
            except Exception:
                pass
        return result

    @staticmethod
    def _gather_git_subprocess(fields: list[str]) -> dict[str, str]:
        """Fallback: gather git info using subprocess calls."""
        result = {}
        git_commands = {
            "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            "last_commit": ["git", "log", "-1", "--format=%H"],
            "last_commit_msg": ["git", "log", "-1", "--format=%s"],
            "short_sha": ["git", "rev-parse", "--short", "HEAD"],
            "tag": ["git", "describe", "--tags", "--abbrev=0"],
            "remote_url": ["git", "remote", "get-url", "origin"],
        }
        for field in fields:
            cmd = git_commands.get(field)
            if cmd:
                try:
                    out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if out.returncode == 0:
                        result[field] = out.stdout.strip()
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
        return result

    @staticmethod
    def _gather_system(fields: list[str]) -> dict[str, str]:
        result = {}
        import platform
        system_map = {
            "hostname": platform.node,
            "os": platform.system,
            "arch": platform.machine,
            "python": platform.python_version,
        }
        for field in fields:
            fn = system_map.get(field)
            if fn:
                result[field] = fn()
        return result
