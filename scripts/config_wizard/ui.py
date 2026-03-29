"""UI utilities for the config wizard."""

from __future__ import annotations

# Colors
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
DIM = "\033[2m"
NC = "\033[0m"


def _print_status(msg: str, color: str, icon: str) -> None:
    """Print a colored status message with icon."""
    print(f"  {color}{icon} {msg}{NC}")


def ok(msg: str) -> None:
    """Print success message."""
    _print_status(msg, GREEN, "✔")


def warn(msg: str) -> None:
    """Print warning message."""
    _print_status(msg, YELLOW, "⚠")


def fail(msg: str) -> None:
    """Print error message."""
    _print_status(msg, RED, "✘")


def info(msg: str) -> None:
    """Print info message."""
    print(f"  {DIM}{msg}{NC}")


def ask(question: str, default: str = "", required: bool = False) -> str:
    """Ask user for input with optional default."""
    if default:
        prompt = f"{question} [{default}]: "
    else:
        prompt = f"{question}: "

    while True:
        response = input(prompt).strip()
        if response:
            return response
        if default:
            return default
        if not required:
            return ""
        print("  (required)")


def ask_yn(question: str, default: bool = False) -> bool:
    """Ask yes/no question."""
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{question} {suffix}: ").strip().lower()
    if not response:
        return default
    return response in ("y", "yes", "tak", "t")


def ask_choice(question: str, options: list, default: int = 0) -> str:
    """Ask user to choose from options.

    Options can be strings or (label, value) tuples.
    """
    print(f"\n{question}")
    labels = []
    values = []
    for opt in options:
        if isinstance(opt, tuple):
            label, value = opt
        else:
            label = opt
            value = opt
        labels.append(label)
        values.append(value)

    for i, label in enumerate(labels, 1):
        marker = " (default)" if i - 1 == default else ""
        print(f"  {i}. {label}{marker}")

    while True:
        response = input("Choice [1-{}]: ".format(len(options))).strip()
        if not response:
            return values[default]
        try:
            idx = int(response) - 1
            if 0 <= idx < len(options):
                return values[idx]
        except ValueError:
            pass
        print("  (invalid choice)")
