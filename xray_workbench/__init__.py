"""Reproducible narrow-beam X-ray attenuation for materials and layered systems."""

from __future__ import annotations

#: Single source of truth for the distribution version. ``pyproject.toml``
#: reads this value, so it must never be duplicated as a literal elsewhere.
__version__ = "1.0.0"

#: Identifier for the implemented physical model. The launcher compares this
#: against a running server's health response before reusing that session, so
#: it must change whenever the model itself changes.
MODEL = "narrow-beam Beer-Lambert"

__all__ = ["MODEL", "__version__"]
