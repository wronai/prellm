"""ContextEngine — gathers runtime context (env vars, git info, system state) for prompt enrichment."""

from __future__ import annotations

import logging
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("prellm.analyzers.context_engine")


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

    def gather_runtime(self) -> "RuntimeContext":
        """Gather full runtime snapshot as a Pydantic RuntimeContext model.

        Collects env (filtered), process, locale, network, git, system info.
        Uses SensitiveDataFilter if available.
        """
        from prellm.models import RuntimeContext
        t0 = time.monotonic()

        # Env — auto-collect with sensitive filtering
        env_safe, blocked_count = self._auto_collect_env()

        # Process info
        process_info = self._gather_process()

        # Locale
        locale_info = self._gather_locale()

        # Network
        network_info = self._gather_network()

        # Git — try common fields
        git_info = self._gather_git(["branch", "short_sha", "last_commit_msg"]) or None

        # System
        system_info = self._gather_system(["hostname", "os", "arch", "python"])

        collected_at = datetime.now(timezone.utc).isoformat()

        ctx = RuntimeContext(
            env_safe=env_safe,
            process=process_info,
            locale=locale_info,
            network=network_info,
            git=git_info if git_info else None,
            system=system_info,
            collected_at=collected_at,
            sensitive_blocked_count=blocked_count,
        )
        # Token estimate (~4 chars/token)
        ctx.token_estimate = len(ctx.model_dump_json()) // 4

        duration_ms = (time.monotonic() - t0) * 1000
        logger.debug(f"RuntimeContext collected in {duration_ms:.1f}ms (blocked={blocked_count})")
        return ctx

    def _auto_collect_env(self) -> tuple[dict[str, str], int]:
        """Collect all env vars, filter through SensitiveDataFilter. Returns (safe_dict, blocked_count)."""
        try:
            from prellm.context.sensitive_filter import SensitiveDataFilter
            sf = SensitiveDataFilter()
            raw = dict(os.environ)
            filtered = sf.filter_dict(raw)
            report = sf.get_filter_report()
            return filtered, len(report.blocked_keys)
        except Exception as e:
            logger.warning(f"Auto env collection failed, falling back to safe keys: {e}")
            # Fallback: only known-safe keys
            safe_keys = {
                "LANG", "TERM", "SHELL", "HOME", "USER", "PWD", "PATH",
                "EDITOR", "HOSTNAME", "TZ", "VIRTUAL_ENV", "PYTHONPATH",
            }
            return {k: v for k, v in os.environ.items() if k in safe_keys}, 0

    @staticmethod
    def _gather_process() -> dict[str, Any]:
        """PID, CWD, user, parent PID, TTY."""
        try:
            tty = os.ttyname(0) if hasattr(os, "ttyname") else ""
        except (OSError, AttributeError):
            tty = ""
        return {
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "user": os.environ.get("USER", os.environ.get("LOGNAME", "")),
            "parent_pid": os.getppid() if hasattr(os, "getppid") else None,
            "tty": tty,
        }

    @staticmethod
    def _gather_locale() -> dict[str, str]:
        """LANG, timezone, encoding — critical for Bielik (Polish locale)."""
        import sys
        tz = ""
        try:
            tz = time.tzname[0] if time.tzname else ""
        except (IndexError, AttributeError):
            pass
        return {
            "lang": os.environ.get("LANG", ""),
            "lc_all": os.environ.get("LC_ALL", ""),
            "timezone": os.environ.get("TZ", tz),
            "encoding": sys.getdefaultencoding(),
        }

    @staticmethod
    def _gather_network() -> dict[str, str]:
        """Hostname, local IP — NO public IP queries."""
        hostname = ""
        local_ip = ""
        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = platform.node()
        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "127.0.0.1"
        return {"hostname": hostname, "local_ip": local_ip}

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
