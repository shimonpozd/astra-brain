"""
Compatibility shim that makes ``import brain_service`` work even when the
current working directory *is* ``brain_service`` (e.g., /opt/astra/brain_service).

In that scenario Python adds the directory itself to ``sys.path``, so it would
normally look for ``brain_service/brain_service/__init__.py``. This module lives
at that location and then points ``brain_service``'s module search path back to
the actual project root.
"""

from __future__ import annotations

import os
import sys
from typing import List

_shim_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_shim_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Allow submodules to be resolved from both the shim directory (for future
# Python modules that might live alongside this file) and the real project root.
__path__: List[str] = [_shim_dir, _project_root]
