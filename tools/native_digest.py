"""Repeat the 16-case native attenuation digest on a fixed reference environment.

Reports a canonical SHA-256 excluding provenance and warning text.
Use --expect only when comparing the same numerical environment: different NumPy/
maths-library builds may round differently. Browser parity uses its separate metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_parity import configurations

from xray_workbench.physics import calculate


def native_digest() -> str:
    digest = hashlib.sha256()
    for config in configurations().values():
        answer = calculate(config)
        for key in ("provenance", "warnings"):
            answer.pop(key)
        digest.update(json.dumps(answer, sort_keys=True).encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", help="optional digest from the same reference numerical environment")
    args = parser.parse_args()
    value = native_digest()
    print(value)
    if args.expect is not None and value != args.expect:
        print("Native digest differs: investigate before accepting changed reference results.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
