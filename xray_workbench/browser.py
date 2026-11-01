"""The browser build's stand-in for the HTTP routes in :mod:`xray_workbench.server`.

The static site has no server, so ``web/engine_worker.js`` calls :func:`handle` inside
Pyodide. It answers with the same status codes and JSON documents as the local server,
without importing the web stack, and ``tests/test_web.py`` holds the two routes together.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import parse_qsl, unquote

from . import MODEL, __version__, materials
from .citation import citation
from .comparison import compare
from .physics import presets
from .results import run_calculation
from .scattering import total_scattering
from .sources import xray_sources


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON number: {value}")


def _materials(route: str, query: dict[str, str]) -> tuple[int, str]:
    thickness = query.get("thickness_mm", "1")
    if route == "materials":
        verified = query.get("verified_density", "").lower() in ("1", "true", "yes", "on")
        answer = materials.catalogue(query.get("q", ""), query.get("category"), query.get("tier"), verified)
        return 200, json.dumps(answer, allow_nan=False)
    missing = 422 if route == "materials/landscape" else 404
    try:
        if route == "materials/landscape":
            answer = materials.landscape_report(float(query.get("energy_keV", "60")), float(thickness))
        else:
            answer = materials.report(unquote(route.removeprefix("materials/")), float(thickness))
        return 200, json.dumps(answer, allow_nan=False)
    except ValueError as exc:
        return missing, json.dumps({"detail": str(exc)})


def handle(route: str, body: str | None = None, runtime: str = "") -> tuple[int, str]:
    """Return ``(status, JSON text)`` for an API route. ``runtime`` is recorded in provenance."""
    route, _, query = route.partition("?")
    if route == "materials" or route.startswith("materials/"):
        return _materials(route, dict(parse_qsl(query)))
    if route == "health":
        return 200, json.dumps({"status": "ok", "version": __version__, "model": MODEL})
    if route == "citation":
        return 200, json.dumps(citation())
    if route == "presets":
        return 200, json.dumps({"presets": presets()})
    if route == "xray-sources":
        return 200, json.dumps({"sources": xray_sources()})
    if route not in ("calculate", "total-scattering", "compare"):
        return 404, json.dumps({"detail": "Not Found"})
    try:
        payload = json.loads(body or "", parse_constant=_invalid_constant)
        if not isinstance(payload, dict):
            raise ValueError("Configuration must be a JSON object.")
        if route == "compare":
            return 200, json.dumps(compare(payload), allow_nan=False)
        if route == "total-scattering":
            return 200, json.dumps(total_scattering(payload), allow_nan=False)
        result = run_calculation(payload)
        if runtime:
            result["provenance"]["runtime"] = runtime
        return 200, json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return 422, json.dumps({"detail": str(exc)})
