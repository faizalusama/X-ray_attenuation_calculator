# Contributing

This is scientific software: a wrong number that looks plausible is worse than
a crash. The rules below exist to keep that from happening quietly.

## Environment

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Python 3.12 or later. CI covers 3.12, 3.13 and 3.14 on Linux, Windows and macOS.

## Before proposing a change

```powershell
python -m pytest
python -m ruff check .
python -m mypy
python tools/manifest.py verify
```

With Node.js available, also:

```powershell
node --check static/app.js
node tests/test_frontend.cjs
```

If you changed any file deliberately, refresh the integrity record:

```powershell
python tools/manifest.py write
```

## Scientific changes

A change to `xray_workbench/physics.py` is a change to published numbers.

- **State the physics, not just the code.** Say which equation or convention
  changed, and cite the source. `docs/SCIENTIFIC_METHODS.md` must stay true.
- **Never widen a tolerance to make a test pass.** A reference comparison that
  starts failing is evidence, not an obstacle. Investigate the disagreement.
- **Keep units explicit.** Public inputs are keV, mm normal thickness, g/cm³,
  and degrees from the surface normal. Conversions happen once, at a named
  boundary, with a comment saying so.
- **Assert the reason.** `pytest.raises(ValueError)` alone is not a test here:
  the engine raises `ValueError` for every invalid input, so an unqualified
  check can pass because an unrelated guard fired. Always pass `match=`.
- **Do not average databases.** Where two sources disagree, surface the
  disagreement. Silently blending them destroys the information.
- **New capability needs new limits.** Anything added to the model must also be
  added to the stated scope, with what it does *not* cover.

## Style

The engine is written in a deliberate compact style — grouped dict literals and
aligned continuations that read as tabulated data. `ruff format` is
intentionally **not** part of the checks, because running it would expand those
blocks without improving them. `ruff check` enforces what matters; match the
surrounding code by eye.

Comments explain *why*, especially where a line encodes a physical or numerical
decision that is not obvious from the expression.

## Version control

Bump the version in `xray_workbench/__init__.py` only — `pyproject.toml`,
the API, the launcher's health check and the result provenance all read it from
there. Duplicating the literal anywhere else reintroduces the drift that once
left the package declaring `1.0.0` while everything else said `0.2.0`.

If you are working without Git, `python tools/snapshot.py save --label why`
before a large change, and `python tools/snapshot.py diff` to see what moved.

## Scope

`legacy/` is archived for attribution and reproducibility. Do not fix, extend
or calculate with it. Its original script applied each component's mass
fraction twice; that bug is preserved there on purpose as part of the record.
