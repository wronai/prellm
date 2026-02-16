"""ShellContextCollector — gathers full shell environment context for prompt enrichment.

Collects: environment variables, process info, locale, shell info, network context.
Filters sensitive data by default (API keys, tokens, passwords).
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import time
from typing import Any

from prellm.models import (
    LocaleInfo,
    NetworkContext,
    ProcessInfo,
    ShellContext,
    ShellInfo,
)

logger = logging.getLogger("prellm.context.shell_collector")

# Patterns in env var NAMES that indicate sensitive data
_SENSITIVE_KEY_PATTERNS = (
    "API_KEY", "TOKEN", "SECRET", "PASSWORD", "PRIVATE_KEY",
    "CREDENTIAL", "AUTH", "SESSION", "COOKIE",
)

# Safe env var names (always allowed)
_SAFE_KEYS = {
    "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "LC_NUMERIC",
    "TERM", "SHELL", "HOME", "USER", "LOGNAME", "PWD", "OLDPWD",
    "PATH", "EDITOR", "VISUAL", "PAGER", "HOSTNAME", "COLUMNS", "LINES",
    "SHLVL", "DISPLAY", "XDG_SESSION_TYPE", "XDG_CURRENT_DESKTOP",
    "TZ", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "PYTHONPATH",
    "GOPATH", "RUSTUP_HOME", "CARGO_HOME", "NVM_DIR",
    "SSH_TTY", "TERM_PROGRAM", "TERM_PROGRAM_VERSION",
}


class ShellContextCollector:
    """Collects full shell environment context for LLM prompt enrichment."""

    def __init__(self, extra_safe_keys: set[str] | None = None):
        self._safe_keys = _SAFE_KEYS | (extra_safe_keys or set())

    def collect_env_vars(self, safe_only: bool = True) -> dict[str, str]:
        """Collect environment variables, optionally filtering sensitive ones."""
        result: dict[str, str] = {}
        for key, value in os.environ.items():
            if safe_only:
                if not self._is_safe_key(key):
                    continue
            result[key] = value
        return result

    def collect_process_info(self) -> ProcessInfo:
        """Collect current process information."""
        try:
            tty = os.ttyname(0) if hasattr(os, "ttyname") else ""
        except (OSError, AttributeError):
            tty = ""

        return ProcessInfo(
            pid=os.getpid(),
            cwd=os.getcwd(),
            user=os.environ.get("USER", os.environ.get("LOGNAME", "")),
            parent_pid=os.getppid() if hasattr(os, "getppid") else None,
            tty=tty,
        )

    def collect_locale_info(self) -> LocaleInfo:
        """Collect locale and timezone information."""
        import sys

        tz = ""
        try:
            tz = time.tzname[0] if time.tzname else ""
        except (IndexError, AttributeError):
            pass

        return LocaleInfo(
            lang=os.environ.get("LANG", ""),
            lc_all=os.environ.get("LC_ALL", ""),
            timezone=os.environ.get("TZ", tz),
            encoding=sys.getdefaultencoding(),
        )

    def collect_shell_info(self) -> ShellInfo:
        """Collect shell and terminal information."""
        columns = 0
        lines = 0
        try:
            size = os.get_terminal_size()
            columns = size.columns
            lines = size.lines
        except (OSError, ValueError):
            columns = int(os.environ.get("COLUMNS", "0"))
            lines = int(os.environ.get("LINES", "0"))

        return ShellInfo(
            shell=os.environ.get("SHELL", ""),
            term=os.environ.get("TERM", ""),
            columns=columns,
            lines=lines,
        )

    def collect_network_context(self) -> NetworkContext:
        """Collect network context — hostname, local IP. No public IP queries."""
        hostname = ""
        local_ip = ""
        dns_suffix = ""

        try:
            hostname = socket.gethostname()
        except Exception:
            hostname = platform.node()

        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "127.0.0.1"

        try:
            fqdn = socket.getfqdn()
            if "." in fqdn:
                dns_suffix = ".".join(fqdn.split(".")[1:])
        except Exception:
            pass

        return NetworkContext(
            hostname=hostname,
            local_ip=local_ip,
            dns_suffix=dns_suffix,
        )

    def collect_all(self, safe_only: bool = True) -> ShellContext:
        """Collect full shell context snapshot as a Pydantic model."""
        t0 = time.monotonic()

        env_vars = self.collect_env_vars(safe_only=safe_only)
        process = self.collect_process_info()
        locale = self.collect_locale_info()
        shell = self.collect_shell_info()
        network = self.collect_network_context()

        duration_ms = (time.monotonic() - t0) * 1000

        return ShellContext(
            env_vars=env_vars,
            process=process,
            locale=locale,
            shell=shell,
            network=network,
            collection_duration_ms=round(duration_ms, 2),
        )

    def _is_safe_key(self, key: str) -> bool:
        """Check if an env var key is safe to expose."""
        if key in self._safe_keys:
            return True
        key_upper = key.upper()
        for pattern in _SENSITIVE_KEY_PATTERNS:
            if pattern in key_upper:
                return False
        return True
