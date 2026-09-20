"""Registry of attenuation data sources.

A backend is selected by its stable identifier. Unknown identifiers are an
error listing what is available — never a silent fallback to the default,
because quietly substituting a different database changes the physics.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import (
    COHERENT,
    INCOHERENT,
    KNOWN_CHANNELS,
    PAIR_ELECTRON,
    PAIR_NUCLEAR,
    PHOTOELECTRIC,
    REQUIRED_CHANNELS,
    AttenuationBackend,
    BackendInfo,
)
from .elam import elam_backend

#: The backend used when a configuration does not name one. Changing this
#: changes every result the workbench produces by default.
DEFAULT_BACKEND = "elam"

#: Constructors rather than instances, so an optional backend whose package is
#: not installed only fails when someone actually asks for it.
_BACKENDS: dict[str, Callable[[], AttenuationBackend]] = {
    "elam": elam_backend,
}


def available_backends() -> list[str]:
    """Identifiers that can be requested, default first."""
    return [DEFAULT_BACKEND, *sorted(set(_BACKENDS) - {DEFAULT_BACKEND})]


def get_backend(identifier: str | None = None) -> AttenuationBackend:
    """Return the backend named by *identifier*, or the default for ``None``."""
    name = DEFAULT_BACKEND if identifier is None else identifier
    if not isinstance(name, str) or name not in _BACKENDS:
        raise ValueError(
            f"Unknown attenuation backend {name!r}. Available: {', '.join(available_backends())}.")
    return _BACKENDS[name]()


def register_backend(identifier: str, factory: Callable[[], AttenuationBackend]) -> None:
    """Add a backend. Refuses to replace an existing identifier.

    Silently rebinding a name would let results claim provenance they do not
    have, so shadowing is an error rather than a last-one-wins convenience.
    """
    if identifier in _BACKENDS:
        raise ValueError(f"A backend named {identifier!r} is already registered.")
    _BACKENDS[identifier] = factory


__all__ = [
    "COHERENT",
    "DEFAULT_BACKEND",
    "INCOHERENT",
    "KNOWN_CHANNELS",
    "PAIR_ELECTRON",
    "PAIR_NUCLEAR",
    "PHOTOELECTRIC",
    "REQUIRED_CHANNELS",
    "AttenuationBackend",
    "BackendInfo",
    "available_backends",
    "get_backend",
    "register_backend",
]
