"""Route-independent result envelope: the calculation plus its exact input.

Kept separate from :mod:`xray_workbench.server` so that batch and library use
never import the web stack, and so that the HTTP route and the command line
provably produce the same document.
"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from . import __version__
from .physics import calculate

#: XrayDB holds a singleton database connection and the uncertainty calculation
#: is memory-intensive. Both are serialized in this local, single-user tool.
_engine_lock = threading.Lock()

SCHEMA_VERSION = 1


def run_calculation(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate ``payload`` and attach the exact input and its fingerprint.

    The digest is taken over a canonical JSON encoding of the *input*, so two
    results carrying the same ``configuration_sha256`` were produced from
    byte-identical configurations regardless of key order or whitespace.
    """
    with _engine_lock:
        result = calculate(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["configuration"] = payload
    result["schema_version"] = SCHEMA_VERSION
    result["provenance"]["configuration_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    result["provenance"]["workbench_version"] = __version__
    return result
