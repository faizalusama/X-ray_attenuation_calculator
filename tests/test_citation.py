"""One name, one version: the citation files must agree with each other and the code."""

import json
import re
from pathlib import Path

from xray_workbench import __version__

ROOT = Path(__file__).resolve().parents[1]
CFF = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
ZENODO = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))


def cff_value(key):
    return re.search(rf'^{key}: "?(.+?)"?$', CFF, re.MULTILINE).group(1)


def test_every_published_title_is_the_same():
    title = cff_value("title")
    assert ZENODO["title"] == title
    assert (ROOT / "README.md").read_text(encoding="utf-8").startswith(f"# {title}\n")
    assert f"<title>{title}</title>" in (ROOT / "static" / "index.html").read_text(encoding="utf-8")


def test_citation_version_matches_the_package():
    assert cff_value("version") == __version__
    assert f"(Version {__version__})" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_author_identity_is_complete_in_both_files():
    creator = ZENODO["creators"][0]
    assert f"orcid.org/{creator['orcid']}" in CFF
    assert f"affiliation: {creator['affiliation']}" in CFF
    assert all(f"- {keyword}" in CFF for keyword in ZENODO["keywords"])

def test_bibtex_matches_cff_and_declares_unreleased_status():
    import re

    from xray_workbench.citation import citation

    data = citation()
    assert data["title"] == re.search(r'^title: "([^"]+)"', CFF, re.M)[1]
    assert data["doi"] == re.search(r'^doi: (.+)', CFF, re.M)[1]
    assert data["version"] == __version__
    assert data["author"] == re.search(r"family-names: (.+)", CFF)[1] + ", " + re.search(r"given-names: (.+)", CFF)[1]
    assert "Unreleased" in data["note"]
    assert "\n  title = {" in data["bibtex"]
