"""Shared type aliases.

Kept in their own module so that the attenuation backends and the physics
engine can both use them without importing each other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Every coefficient array is float64 over an evaluation grid.
Array = NDArray[np.float64]

#: Decoded JSON objects: request payloads, layers and result documents.
JsonObject = dict[str, Any]

__all__ = ["Array", "JsonObject"]
