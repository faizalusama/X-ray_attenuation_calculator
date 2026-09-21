"""Resolve a DOI and print the verification block for a literature entry.

    python tools/verify_doi.py 10.1016/S0969-806X(01)00227-4
    python tools/verify_doi.py --library      # re-check every DOI in the library

Queries Crossref first and DataCite second (NIST and Zenodo DOIs live there).
The output is the ``doi_resolved`` object that a literature entry must carry.
It proves the DOI exists and records what it points to; it does **not** prove
the composition was transcribed correctly. That needs a person to compare the
entry with the paper and sign ``values_checked_by`` / ``values_checked_on``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

USER_AGENT = "xray-attenuation-workbench/verify_doi (https://github.com/faizalusama/X-ray_attenuation_calculator)"


def _get(url: str) -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:  # fixed public metadata APIs
        return json.loads(response.read())


def resolve(doi: str) -> dict[str, Any]:
    """Return title, container, year and authors for *doi*, or raise LookupError."""
    try:
        message = _get(f"https://api.crossref.org/works/{quote(doi)}")["message"]
        issued = (message.get("issued") or {}).get("date-parts") or [[None]]
        return {
            "registry": "Crossref",
            "title": (message.get("title") or [""])[0],
            "container": (message.get("container-title") or [""])[0],
            "year": issued[0][0],
            "authors": [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                        for a in message.get("author", [])][:12],
            "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    except HTTPError as exc:
        if exc.code != 404:
            raise
    try:
        attributes = _get(f"https://api.datacite.org/dois/{quote(doi)}")["data"]["attributes"]
    except HTTPError as exc:
        raise LookupError(f"{doi} is not registered with Crossref or DataCite.") from exc
    publisher = attributes.get("publisher")
    return {
        "registry": "DataCite",
        "title": (attributes.get("titles") or [{}])[0].get("title", ""),
        "container": publisher.get("name", "") if isinstance(publisher, dict) else (publisher or ""),
        "year": attributes.get("publicationYear"),
        "authors": [c.get("name", "") for c in attributes.get("creators", [])][:12],
        "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _library_dois() -> list[tuple[str, str]]:
    from xray_workbench.materials import datasets
    found = []
    for dataset_id, dataset in datasets().items():
        if dataset.get("doi"):
            found.append((f"dataset {dataset_id}", dataset["doi"]))
        for entry in dataset["entries"]:
            literature = entry.get("literature") or {}
            if literature.get("doi"):
                found.append((entry["id"], literature["doi"]))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("doi", nargs="?", help="DOI to resolve, without the https://doi.org/ prefix")
    group.add_argument("--library", action="store_true", help="check every DOI in the materials library")
    args = parser.parse_args(argv)
    targets = _library_dois() if args.library else [("argument", args.doi.removeprefix("https://doi.org/"))]
    failures = 0
    for owner, doi in targets:
        try:
            record = resolve(doi)
        except (LookupError, URLError, HTTPError) as exc:
            failures += 1
            print(f"FAIL  {owner}: {doi}: {exc}", file=sys.stderr)
            continue
        if args.library:
            print(f"ok    {owner}: {doi} -> {record['title'][:80]} ({record['year']})")
        else:
            print(json.dumps({"doi_resolved": record}, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
