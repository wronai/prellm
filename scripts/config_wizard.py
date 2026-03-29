#!/usr/bin/env python3
"""Interactive preLLM configuration wizard with diagnostics.

This module is a thin wrapper around the config_wizard package for backward compatibility.
All implementation has been moved to the config_wizard package.
"""

from __future__ import annotations

# Re-export main entry point
from config_wizard.wizard import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
