"""
Utilities for locating and importing the shared Astra configuration package.

The brain service expects a Python package called ``config`` to be available,
but the actual configuration (with environment-specific secrets) is intentionally
kept outside of the tracked source tree.  This module centralises the logic for
adding the correct directory to ``sys.path`` so that ``import config`` works in
local development, Docker containers, and production deployments where the
configuration is mounted as a volume.
"""

from __future__ import annotations

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _candidate_root_from_env() -> Optional[Path]:
    """
    Resolve the configuration root based on ``ASTRA_CONFIG_ROOT``.

    The variable may point either at the parent directory that contains the
    ``config`` package or directly at the package itself.  ``None`` is returned
    when the variable is unset.
    """
    env_value = os.getenv("ASTRA_CONFIG_ROOT")
    if not env_value:
        return None

    candidate = Path(env_value).expanduser().resolve()
    if not candidate.exists():
        raise RuntimeError(
            f"ASTRA_CONFIG_ROOT={env_value!r} does not exist; "
            "mount the configuration directory or adjust the variable."
        )

    if candidate.is_dir():
        if candidate.name == "config" and (candidate / "__init__.py").exists():
            # Variable points directly at the package; return its parent.
            return candidate.parent
        if (candidate / "config").is_dir():
            return candidate

    raise RuntimeError(
        "ASTRA_CONFIG_ROOT must reference a directory that contains a "
        "'config' package (with an __init__.py). "
        f"Received: {env_value!r}"
    )


@lru_cache(maxsize=1)
def ensure_config_root() -> Path:
    """
    Ensure the configuration package is importable and return the root path.

    Order of precedence:
    1. ``ASTRA_CONFIG_ROOT`` environment variable (preferred for deployments).
    2. Project root inferred from the location of this file (sufficient for
       local development when running from the mono-repo).
    """
    env_root = _candidate_root_from_env()
    if env_root is not None:
        root = env_root
    else:
        root = Path(__file__).resolve().parents[2]

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        importlib.import_module("config")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import the 'config' package. "
            "Ensure the configuration directory is available. "
            f"Searched root: {root}"
        ) from exc

    return root


def get_config_root() -> Path:
    """Public wrapper that mirrors ``ensure_config_root`` semantics."""
    return ensure_config_root()
