"""Loopback web application. The scientific engine is also usable independently."""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import MODEL, __version__, materials
from ._types import JsonObject
from .comparison import compare
from .physics import presets
from .results import run_calculation
from .scattering import total_scattering
from .sources import xray_sources


def _static_root() -> Path:
    """Locate the browser assets in an installed wheel or in the source tree.

    The wheel copies ``static/`` next to this module; a checkout keeps it at the
    repository root. Probing for ``index.html`` fails loudly at import time
    rather than serving 404s from a silently empty directory.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here / "static", here.parent / "static"):
        if (candidate / "index.html").is_file():
            return candidate
    raise RuntimeError(
        "Browser assets not found. Expected static/index.html beside the "
        f"xray_workbench package ({here / 'static'}) or at the repository "
        f"root ({here.parent / 'static'})."
    )


STATIC_ROOT = _static_root()
MAX_BODY_BYTES = 2_000_000
#: The quantities in static/dashboard.js, each exportable as plot data.
PLOT_DATA_MODES = ("transmission", "removed", "mass", "linear", "channels", "tau", "hvl", "tvl", "length", "design", "map")
app = FastAPI(title="X-ray Attenuation Calculator", version=__version__,
              description="Local narrow-beam X-ray attenuation calculator for multicomponent materials.")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])
# Plotly alone is 4.2 MB of text; compressed it travels at roughly a third of
# that. Loopback is fast, but the same server also runs on shared machines.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def security_headers(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    path = request.url.path
    if path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    elif path.startswith("/static/"):
        # Assets referenced with ?v=… or a versioned vendor filename never
        # change in place, so the browser may keep them for a year; anything
        # else is revalidated, so an edit is never masked by a stale copy.
        versioned = "v=" in request.url.query or path.startswith("/static/vendor/")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if versioned else "no-cache"
    return response


@app.get("/api/health")
def health() -> JsonObject:
    return {"status": "ok", "version": __version__, "model": MODEL}


@app.get("/api/presets")
def get_presets() -> JsonObject:
    return {"presets": presets()}


@app.get("/api/xray-sources")
def get_xray_sources() -> JsonObject:
    return {"sources": xray_sources()}


@app.get("/api/materials")
def list_materials(q: str = "", category: str | None = None, tier: str | None = None,
                   verified_density: bool = False) -> JsonObject:
    """Search the materials library."""
    return materials.catalogue(q, category, tier, verified_density)


@app.get("/api/materials/landscape")
def materials_landscape(energy_keV: float = 60.0, thickness_mm: float = 1.0) -> JsonObject:
    """Every library entry evaluated at one energy: the data behind the overview plot."""
    try:
        return materials.landscape_report(energy_keV, thickness_mm)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/materials/{identifier}")
def material_detail(identifier: str, thickness_mm: float = 1.0) -> JsonObject:
    """Full provenance for one entry, plus a ready-to-use calculation layer."""
    try:
        return materials.report(identifier, thickness_mm)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


async def _answer(request: Request, respond: Callable[[Any], JsonObject]) -> JsonObject:
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise HTTPException(415, "Send an application/json calculation configuration.")
    chunks = bytearray()
    async for chunk in request.stream():
        chunks.extend(chunk)
        if len(chunks) > MAX_BODY_BYTES:
            raise HTTPException(413, "Configuration exceeds the 2 MB limit.")
    try:
        def invalid_constant(value: str) -> Any:
            raise ValueError(f"Non-finite JSON number: {value}")
        payload = json.loads(chunks, parse_constant=invalid_constant)
        if not isinstance(payload, dict):
            raise ValueError("Configuration must be a JSON object.")
        return await run_in_threadpool(respond, payload)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/calculate")
async def calculate_request(request: Request) -> JsonObject:
    return await _answer(request, run_calculation)


@app.post("/api/compare")
async def compare_request(request: Request) -> JsonObject:
    return await _answer(request, compare)


@app.post("/api/total-scattering")
async def total_scattering_request(request: Request) -> JsonObject:
    """Planning values for a total-scattering measurement of one layer."""
    return await _answer(request, total_scattering)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.post("/api/download")
async def download_artifact(request: Request) -> Response:
    """Return a native attachment, including in browsers that restrict blob URLs.

    Content is generated by the local GUI and immediately returned to the same
    browser. No file is written on the server and no download is retained.
    """
    if request.headers.get("content-type", "").split(";")[0] != "application/x-www-form-urlencoded":
        raise HTTPException(415, "Expected an encoded export form.")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 32_000_000:
            raise HTTPException(413, "Export exceeds the 32 MB limit.")
    try:
        fields = parse_qs(body.decode("utf-8"), keep_blank_values=True, max_num_fields=4)
        filename = fields["filename"][0]
        content = fields["content"][0]
    except (ValueError, UnicodeError, KeyError, IndexError) as exc:
        raise HTTPException(422, "Invalid export form.") from exc
    allowed = {"attenuation-project.json", "attenuation-results.json", "attenuation-sweep.csv", "attenuation-dashboard.svg"}
    allowed.update(f"attenuation-{mode}.svg" for mode in ("transmission", "removed", "mass", "linear", "channels"))
    allowed.update({"attenuation-all-curves.csv", "attenuation-all-curves-settings.json", "attenuation-spectrum.csv",
                    "total-scattering-curves.csv", "total-scattering-summary.json", "attenuation-comparison.csv"})
    allowed.update(f"attenuation-plot-{mode}.csv" for mode in PLOT_DATA_MODES)
    if filename not in allowed:
        raise HTTPException(422, "Unsupported export filename.")
    return Response(content.encode("utf-8"), media_type="application/octet-stream",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"', "Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
