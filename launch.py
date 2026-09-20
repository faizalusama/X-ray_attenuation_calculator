#!/usr/bin/env python3
"""Checkout entry point: ``python launch.py``.

The implementation lives in :mod:`xray_workbench.cli`, which is also installed
as the ``xray-workbench`` command. This file stays so that START_WINDOWS.cmd,
the documentation and existing scripts keep working unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xray_workbench.cli import main

if __name__ == "__main__":
    main()
