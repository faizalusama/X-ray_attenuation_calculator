"""The browser build's stand-in for the HTTP routes in :mod:`xray_workbench.server`.

The static site has no server, so ``web/engine_worker.js`` calls :func:`handle` inside
Pyodide. It answers with the same status codes and JSON documents as the local server,
without importing the web stack, and ``tests/test_web.py`` holds the two routes together.
"""

from __future__ import annotations

import json
from typing import Any

from . import MODEL, __version__
from .physics import presets
from .results import run_calculation
from .sources import xray_sources


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON number: {value}")


def handle(route: str, body: str | None = None, runtime: str = "") -> tuple[int, str]:
    """Return ``(status, JSON text)`` for an API route. ``runtime`` is recorded in provenance."""
    if route == "health":
        return 200, json.dumps({"status": "ok", "version": __version__, "model": MODEL})
    if route == "presets":
        return 200, json.dumps({"presets": presets()})
    if route == "xray-sources":
        return 200, json.dumps({"sources": xray_sources()})
    if route != "calculate":
        return 404, json.dumps({"detail": "Not Found"})
    try:
        payload = json.loads(body or "", parse_constant=_invalid_constant)
        if not isinstance(payload, dict):
            raise ValueError("Configuration must be a JSON object.")
        result = run_calculation(payload)
        if runtime:
            result["provenance"]["runtime"] = runtime
        return 200, json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return 422, json.dumps({"detail": str(exc)})
