"""Materials library: provenance rules, data integrity and an independent cross-check."""
import copy
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from xray_workbench import materials
from xray_workbench.materials import LibraryError, _parse_entry
from xray_workbench.physics import calculate
from xray_workbench.server import app

client = TestClient(app)

#: NIST SRD 126 materials annotated with the stoichiometric formula they
#: correspond to. NIST's tabulated elemental mass fractions and the fractions
#: our parser derives from that formula come from different places, so
#: agreement checks the importer, the annotation and the parser together.
NIST_STOICHIOMETRIC = {m.id: m.formula for m in materials.search(tier="reference_data") if m.formula}


def test_the_expected_nist_compounds_are_annotated():
    assert set(NIST_STOICHIOMETRIC.values()) == {
        "GaAs", "CdTe", "CaF2", "LiF", "CsI", "HgI2", "CaSO4", "Li2B4O7", "MgB4O7", "Gd2O2S", "H2O"}


def mass_fractions(material_or_layer):
    layer = material_or_layer if isinstance(material_or_layer, dict) else material_or_layer.to_layer()
    result = calculate({"energy": {"min_keV": 10, "max_keV": 20, "points": 2, "reference_keV": 15},
                        "layers": [layer]})
    return result["layers"][0]["elemental_mass_fractions"]


def test_library_loads_with_unique_ids_across_datasets():
    loaded = materials.load()
    assert len(loaded) == 501
    assert len({m.id for m in loaded}) == len(loaded)
    assert {m.tier for m in loaded} >= {"reference_data", "stoichiometric"}


@pytest.mark.parametrize("material", materials.load(), ids=lambda m: m.id)
def test_every_entry_is_accepted_by_the_engine(material):
    # Every formula must parse and every entry must produce finite physics.
    result = calculate({"energy": {"min_keV": 10, "max_keV": 200, "points": 20, "reference_keV": 60},
                        "layers": [material.to_layer(1.0)]})
    assert np.all(np.isfinite(result["transmission"]))
    assert 0 < result["reference"]["transmission"] <= 1


@pytest.mark.parametrize("identifier,formula", sorted(NIST_STOICHIOMETRIC.items()))
def test_nist_fractions_agree_with_the_formula_parser(identifier, formula):
    nist = mass_fractions(materials.get(identifier))
    derived = mass_fractions({"components": [{"formula": formula, "fraction": 1}],
                              "density_g_cm3": 1.0, "thickness_mm": 1})
    assert set(nist) == set(derived)
    for element, fraction in derived.items():
        # Measured worst case 3.4e-5 (21 Sep 2026): NIST prints six decimals and
        # standard atomic weights have been revised since 1995. 1e-4 absolute
        # keeps a 3x margin yet is far below a wrong formula or a misread digit.
        assert nist[element] == pytest.approx(fraction, abs=1e-4), (identifier, element)


def test_nist_dataset_records_its_provenance():
    dataset = materials.datasets()["nist-srd126-table2"]
    assert dataset["url"].startswith("https://physics.nist.gov/")
    assert dataset["doi"] == "10.18434/T4D01F"
    assert len(dataset["retrieved_sha256"]) == 64
    assert dataset["retrieved_utc"]
    assert any("nominal" in caveat for caveat in dataset["caveats"])


def test_nist_mass_fractions_sum_to_one():
    for material in (m for m in materials.load() if m.dataset_id == "nist-srd126-table2"):
        assert material.basis == "mass"
        assert sum(x for _, x in material.components) == pytest.approx(1.0, abs=1e-5), material.id


def test_pnnl_rev2_import_is_complete_multielement_and_traceable():
    dataset = materials.datasets()["pnnl-15870-rev2"]
    imported = [m for m in materials.load() if m.dataset_id == "pnnl-15870-rev2"]
    assert len(imported) == 359
    assert dataset["doi"] == "10.2172/1782721"
    assert dataset["report_number"] == "PNNL-15870 Rev. 2 / 200-DMAMC-128170"
    assert dataset["retrieved_sha256"] == "72b26dba2c3b5583b86fe5d5fe27a43d2890331d0515ce787f1c18fd7321cee6"
    assert len({m.reference_constants["pnnl_material_number"] for m in imported}) == len(imported)
    assert all(len(m.components) >= 2 for m in imported)
    assert all(m.density_status == "sourced" for m in imported)
    assert all(sum(value for _, value in m.components) == pytest.approx(1, abs=5e-4) for m in imported)


@pytest.mark.parametrize("identifier,category", [
    ("pnnl-r2-008-aluminum-alloy-2024-o", "alloy"),
    ("pnnl-r2-101-concrete-magnetite", "concrete"),
    ("pnnl-r2-158-glass-lead", "glass"),
    ("pnnl-r2-274-polyethylene-borated", "polymer"),
])
def test_pnnl_representative_engineering_materials(identifier, category):
    material = materials.get(identifier)
    assert material.category == category
    assert material.reference_constants["pnnl_material_number"] > 0


def test_pnnl_printed_rounding_is_preserved_and_engine_normalizes_it():
    material = materials.get("pnnl-r2-321-sodium-iodide-with-8-wt-lithium-0-10-wt-thalium-doped")
    assert sum(value for _, value in material.components) == pytest.approx(0.999672)
    assert sum(mass_fractions(material).values()) == pytest.approx(1)


def test_every_stoichiometric_density_is_marked_unverified():
    for material in materials.search(tier="stoichiometric"):
        assert material.density_status == "unverified", material.id
        assert "measured" in material.density_note, material.id


def test_search_by_formula_name_and_filters():
    assert {m.id for m in materials.search("zirconia")} >= {"stoich-zirconia-3y", "stoich-zirconia-8y"}
    assert any(m.id == "stoich-alumina-corundum" for m in materials.search("al2o3"))
    assert all(m.category == "glass" for m in materials.search(category="glass"))
    assert all(m.density_verified for m in materials.search(verified_density_only=True))


# --- Provenance rules: the loader must refuse claims that exceed the evidence.

VALID_LITERATURE = {
    "id": "lit-example", "name": "Example glass", "category": "glass", "tier": "literature",
    "composition": {"basis": "mole", "components": [{"formula": "SiO2", "fraction": 70},
                                                    {"formula": "Na2O", "fraction": 30}]},
    "density": {"value_g_cm3": 2.5, "kind": "measured", "status": "sourced"},
    "literature": {
        "doi": "10.1016/S0969-806X(01)00227-4", "locator": "Table 2, sample A",
        "verification": {"doi_resolved": {"title": "A new atomic database", "registry": "Crossref"},
                         "values_checked_by": "A. Person", "values_checked_on": "2026-09-21"},
    },
}


def test_a_complete_literature_entry_is_accepted():
    material = _parse_entry(copy.deepcopy(VALID_LITERATURE), "test", 0)
    assert material.tier == "literature"
    assert material.density_verified


@pytest.mark.parametrize("path,value,message", [
    (("literature", "doi"), "not-a-doi", "invalid DOI"),
    (("literature", "locator"), "", "locator"),
    (("literature", "verification", "values_checked_by"), "", "a person must have compared"),
    (("literature", "verification", "doi_resolved"), {}, "doi_resolved"),
    (("literature",), None, "need a 'literature' object"),
    (("density", "status"), "unverified", "must say so in its note"),
    (("category",), "ceramic", "unknown category"),
    (("tier",), "journal", "unknown tier"),
    (("id",), "Bad Id", "lowercase-hyphenated"),
])
def test_literature_entries_missing_evidence_are_refused(path, value, message):
    raw = copy.deepcopy(VALID_LITERATURE)
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(LibraryError, match=message):
        _parse_entry(raw, "test", 0)


def test_a_formula_cannot_claim_a_sourced_density():
    raw = {"id": "stoich-x", "name": "X", "category": "oxide_ceramic", "tier": "stoichiometric",
           "composition": {"basis": "mole", "components": [{"formula": "MgO", "fraction": 1}]},
           "density": {"value_g_cm3": 3.58, "kind": "typical", "status": "sourced"}}
    with pytest.raises(LibraryError, match="unverified unless a literature source"):
        _parse_entry(raw, "test", 0)


# --- HTTP API

def test_api_lists_and_filters_materials():
    listing = client.get("/api/materials", params={"q": "carbide"}).json()
    assert listing["count"] >= 5
    assert all("density_status" in m for m in listing["materials"])
    verified = client.get("/api/materials", params={"verified_density": True}).json()
    assert verified["count"] == len([m for m in materials.load() if m.density_verified])


def test_api_detail_carries_provenance_and_a_usable_layer():
    detail = client.get("/api/materials/nist-glass-borosilicate-pyrex", params={"thickness_mm": 2}).json()
    assert detail["source"]["doi"] == "10.18434/T4D01F"
    assert detail["density"]["status"] == "sourced"
    assert detail["layer"]["thickness_mm"] == 2
    response = client.post("/api/calculate", json={"layers": [detail["layer"]]})
    assert response.status_code == 200
    json.dumps(detail, allow_nan=False)


def test_api_unknown_material_is_404():
    assert client.get("/api/materials/not-a-material").status_code == 404


@pytest.mark.parametrize("energy", [15.0, 60.0, 300.0])
def test_landscape_fast_path_matches_the_full_engine(energy):
    by_id = {p["id"]: p for p in materials.landscape(energy, thickness_mm=2.0)}
    assert len(by_id) == len(materials.load())
    for identifier in ("nist-glass-lead", "stoich-zirconia-3y", "stoich-boron-carbide", "nist-water-liquid"):
        full = calculate({"energy": {"min_keV": 10, "max_keV": 20, "points": 2, "reference_keV": energy},
                          "layers": [materials.get(identifier).to_layer(2.0)]})
        reference = full["layers"][0]["reference"]
        point = by_id[identifier]
        for key in ("mu_mass_cm2_g", "mu_linear_cm_inv", "hvl_mm", "transmission"):
            assert point[key] == pytest.approx(reference[key], rel=1e-12), (identifier, key)


def test_api_landscape_and_its_guards():
    data = client.get("/api/materials/landscape", params={"energy_keV": 60}).json()
    assert data["energy_keV"] == 60
    assert len(data["points"]) == len(materials.load())
    assert all(p["hvl_mm"] > 0 and "density_status" in p for p in data["points"])
    assert client.get("/api/materials/landscape", params={"energy_keV": 900}).status_code == 422
    assert client.get("/api/materials/landscape", params={"thickness_mm": 0}).status_code == 422


def test_api_summaries_carry_formulas_for_formula_search():
    listing = client.get("/api/materials").json()["materials"]
    gd2o2s = next(m for m in listing if m["id"] == "nist-gadolinium-oxysulfide")
    # NIST names carry no formula, and NIST lists elements; both are searchable.
    assert set(gd2o2s["formulas"]) == {"O", "S", "Gd", "Gd2O2S"}
    zirconia = next(m for m in listing if m["id"] == "stoich-zirconia-3y")
    assert zirconia["formulas"] == ["ZrO2", "Y2O3"]
