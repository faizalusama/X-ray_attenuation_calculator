"use strict";

const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const test = require("node:test");
const {runInNewContext} = require("node:vm");

const context = {};
function loadScript(name) {
  const filename = join(__dirname, "../static", name);
  runInNewContext(readFileSync(filename, "utf8"), context, {filename});
}

loadScript("guide-family-links.js");
const {parse, hashFor} = context.WorkbenchGuideFamilyLinks;
const families = ["total", "saxs", "waxs", "tomo", "optics", "attenuation"];

function assertFamily(hash, family) {
  // Copy the VM result into this realm for strict object comparison.
  const result = parse(hash);
  assert.ok(result, hash);
  assert.deepEqual({...result}, {family}, hash);
}

test("the bare technique guide fragment opens all families", () => {
  assertFamily("#technique-guide", "");
  assert.equal(hashFor(""), "#technique-guide");
});

test("all six technique family links round-trip through the URL fragment", () => {
  for (const family of families) {
    const expected = "#technique-guide/" + family;
    const hash = hashFor(family);
    assert.equal(hash, expected);
    assertFamily(new URL(hash, "https://example.test/workbench/").hash, family);
  }
});

test("the supported family IDs match the guide catalogs", () => {
  loadScript("guide-scattering.js");
  loadScript("guide-imaging.js");
  const catalogIds = Array.from(context.WorkbenchGuideFamilies, family => family.id);
  assert.deepEqual(catalogIds.sort(), [...families].sort());
});

test("unknown and malformed technique family fragments open all families", () => {
  for (const suffix of [
    "neutron", "constructor", "", "SAXS", "%73axs", "%", "saxs/", "saxs/tomo",
    "saxs?filter=plan", "saxs#details",
  ]) {
    assertFamily("#technique-guide/" + suffix, "");
  }
});

test("unrelated and saved-project fragments are left to their existing handlers", () => {
  for (const hash of [
    "", "#", "#overview", "#technique-guide-other/saxs", "#technique-guide?saxs",
    "technique-guide/saxs", "#p1", "#p1r.eyJ2IjoxfQ", "#p1g.H4sIAAAAAAAA", null, undefined,
  ]) {
    assert.equal(parse(hash), null, String(hash));
  }
});

test("unsupported family values generate the bare guide fragment", () => {
  for (const family of [
    "neutron", "constructor", "SAXS", " saxs", "saxs/tomo", "#p1r.abc", null, undefined, 1, ["saxs"],
  ]) {
    assert.equal(hashFor(family), "#technique-guide", String(family));
  }
});
