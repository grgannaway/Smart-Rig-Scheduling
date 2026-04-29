"""Importable shim for ``v1.1_rig_scheduler_genetic_algorithm.py``.

Python module names cannot contain ``.`` so the v1.1 engine, whose filename
is ``v1.1_rig_scheduler_genetic_algorithm.py``, cannot be imported via the
normal ``import`` statement.  This shim loads that file once via importlib
and re-exports its public surface so callers can simply do::

    from v1_1_rig_scheduler_genetic_algorithm import SimConfig, ...

The real source of truth remains ``v1.1_rig_scheduler_genetic_algorithm.py``
in the same directory; edits should be made there.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ENGINE_PATH = Path(__file__).resolve().parent / "v1.1_rig_scheduler_genetic_algorithm.py"

if not _ENGINE_PATH.is_file():
    raise ImportError(
        f"v1.1 engine source not found at {_ENGINE_PATH}. "
        "The shim v1_1_rig_scheduler_genetic_algorithm.py expects it next to itself."
    )

_spec = importlib.util.spec_from_file_location(
    "v1_1_rig_scheduler_genetic_algorithm_impl", str(_ENGINE_PATH)
)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not build module spec for {_ENGINE_PATH}")

_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

# Re-export every public attribute so `from v1_1_... import X` works.
globals().update({
    name: getattr(_module, name)
    for name in dir(_module)
    if not name.startswith("_")
})

# Also re-export commonly imported single-underscore helpers used by the apps.
for _priv in ("_evaluate_ordering", "_add_net_fcf_columns"):
    if hasattr(_module, _priv):
        globals()[_priv] = getattr(_module, _priv)
