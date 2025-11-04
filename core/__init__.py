"""
Core utilities for the brain service.

Importing this package guarantees that the external configuration package is
discoverable, irrespective of the runtime environment.
"""

from .config_loader import ensure_config_root

# Ensure side-effect at import time so any downstream module can ``import config``.
ensure_config_root()

__all__ = ["ensure_config_root"]
