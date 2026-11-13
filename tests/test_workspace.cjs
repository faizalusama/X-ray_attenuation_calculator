"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const {MODE_KEY, preferredMode, advancedActive, disclosureVisible, activeSettings, create} = require("../static/workspace.js");
const {initial, cleanProject} = require("../static/app.js");
const clone = value => JSON.parse(JSON.stringify(value));

test("mode defaults safely, validates storage and gives explicit project UI priority", () => {
  assert.equal(preferredMode(), "simple");
  for (const value of [null, "", "expert", {}, true]) assert.equal(preferredMode(null, value), "simple");
  assert.equal(preferredMode(null, "advanced"), "advanced");
  assert.equal(preferredMode({workspaceMode: "simple"}, "advanced"), "simple");
  assert.equal(preferredMode({workspaceMode: "advanced"}, "simple"), "advanced");
  assert.equal(preferredMode({workspaceMode: "invalid"}, "advanced"), "advanced");
});

test("Simple hides inactive controls while Advanced exposes them without mutating configuration", () => {
  const config = clone(initial), before = clone(config);
  for (const key of ["angle", "density", "areal", "grid", "explanation"]) {
    assert.equal(disclosureVisible("simple", key, config, 0), false, key);
    assert.equal(disclosureVisible("advanced", key, config, 0), true, key);
  }
  assert.deepEqual(activeSettings(config), []);
  assert.deepEqual(config, before);
});

test("active angle, density, areal and custom grid survive Simple disclosure, including zero areal density", () => {
  const config = clone(initial);
  Object.assign(config.layers[0], {angle_deg: -23, density_mode: "solid", porosity: .12, areal_density_mg_cm2: 0});
  config.energy.points = 700;
  config.spectrum = {energy_keV: [20, 40], weights: [3, 2], weighting: "energy", label: "Measured"};
  config.uncertainty = {enabled: true, samples: 100, seed: 11};
  const before = clone(config);
  for (const key of ["angle", "density", "areal", "grid"]) assert.equal(disclosureVisible("simple", key, config, 0), true, key);
  const settings = activeSettings(config);
  for (const phrase of ["Spectrum enabled", "Uncertainty enabled", "density model", "areal density", "angle from normal", "porosity", "Custom energy grid"])
    assert.ok(settings.some(setting => setting.label.includes(phrase)), phrase);
  assert.deepEqual(config, before);
  assert.deepEqual(cleanProject({schema_version: 1, configuration: config}), cleanProject({schema_version: 1, configuration: before}));
});

test("active disclosure is per layer, handles ideal density, linear grids and incomplete angle", () => {
  const config = clone(initial);
  config.layers.push({...clone(config.layers[0]), angle_deg: 15, density_mode: "ideal"});
  assert.equal(advancedActive("angle", config, 0), false);
  assert.equal(advancedActive("angle", config, 1), true);
  assert.equal(advancedActive("density", config, 1), true);
  config.layers[0].angle_deg = null;
  assert.equal(advancedActive("angle", config, 0), true);
  config.energy.spacing = "linear";
  assert.equal(advancedActive("grid", config), true);
});

test("invalid or explicitly revealed controls remain recoverable in Simple", () => {
  for (const key of ["angle", "density", "areal", "grid", "explanation"]) {
    assert.equal(disclosureVisible("simple", key, initial, 0, true), true);
    assert.equal(disclosureVisible("simple", key, initial, 0, false, true), true);
  }
});

test("visual groups preserve one flat tablist and the original seven tab IDs and order", () => {
  const html = fs.readFileSync(require.resolve("../static/index.html"), "utf8");
  const start = html.indexOf('<div class="tabs"');
  const end = html.indexOf('<span class="result-status"', start);
  const tabs = html.slice(start, end);
  assert.equal((tabs.match(/role="tablist"/g) || []).length, 1);
  assert.deepEqual([...tabs.matchAll(/id="tab-([^"]+)"/g)].map(m => m[1]), ["overview", "materials", "spectrum", "uncertainty", "scattering", "tools", "methods"]);
  for (const [group, label, count] of [["attenuation", "Attenuation", 4], ["scattering", "Scattering", 2], ["reference", "Reference", 1]]) {
    assert.ok(tabs.includes('id="' + group + '-tab-group" class="tab-group-label">' + label));
    assert.equal((tabs.match(new RegExp('aria-describedby="' + group + '-tab-group"', "g")) || []).length, count);
  }
});

// A small DOM stand-in exercises the real controller's preference, collapse,
// disclosure and recovery paths without adding a browser dependency to Node CI.
function harness(stored = null, throwStorage = false) {
  const elements = new Map(), observers = [];
  let doc;
  function element(dataset = {}) {
    const handlers = {}, attributes = {}, classes = new Set();
    return {
      dataset, hidden: false, children: [], validity: {valid: true},
      classList: {contains: key => classes.has(key), add: key => classes.add(key), remove: key => classes.delete(key)},
      setAttribute(key, value) { attributes[key] = value; },
      getAttribute(key) { return attributes[key]; },
      addEventListener(key, fn) { handlers[key] = fn; },
      click() { handlers.click?.({stopImmediatePropagation() {}}); },
      focus() { doc.activeElement = this; },
      contains(target) { return this === target || this.children.includes(target); },
      querySelectorAll() { return this.children; },
      closest() { return null; },
      append(child) { this.children.push(child); },
      replaceChildren() { this.children = []; },
    };
  }
  const simple = element({workspaceMode: "simple"}), advanced = element({workspaceMode: "advanced"});
  const angle = element({workspaceAdvanced: "angle", layer: "0"}), grid = element({workspaceAdvanced: "grid"});
  const angleInput = element();
  angle.children.push(angleInput);
  for (const selector of [".workspace", "#workspace-inputs", "#close-inputs", "#dashboard-focus", "#workspace-invalid", "#workspace-active-settings", "#workspace-library-slot"])
    elements.set(selector, element());
  const input = element();
  const config = clone(initial), snapshot = clone(config), writes = [];
  const modes = [simple, advanced], disclosures = [angle, grid];
  const selects = {
    "[data-workspace-mode]": modes, "[data-workspace-advanced]": disclosures,
    "#config-form input, #config-form select": [input, angleInput], "#view-uncertainty input": [],
  };
  doc = {
    documentElement: element(), activeElement: null, readyState: "complete",
    querySelector: key => elements.get(key) || null,
    querySelectorAll: key => selects[key] || [],
    createElement: () => element(),
  };
  let changes = 0, tab = "", layer = -1;
  const previous = global.MutationObserver;
  global.MutationObserver = class { constructor(fn) { observers.push(fn); } observe() {} disconnect() {} };
  let controller;
  try {
    controller = create(doc, {
      getConfiguration: () => config,
      setInputs(visible) { const cl = elements.get(".workspace").classList; cl[visible ? "remove" : "add"]("dashboard-wide"); observers.forEach(fn => fn()); },
      activateTab(value) { tab = value; },
      onLayerOpen(value) { layer = value; },
      onChange() { changes++; },
    }, {
      getItem() { if (throwStorage) throw Error("blocked"); return stored; },
      setItem(key, value) { if (throwStorage) throw Error("blocked"); writes.push([key, value]); },
    });
  } finally { global.MutationObserver = previous; }
  return {doc, elements, observers, simple, advanced, angle, grid, input, angleInput, config, snapshot, writes, controller, get changes() { return changes; }, get tab() { return tab; }, get layer() { return layer; }, element};
}

test("controller mode changes preserve every setting and project restore overrides browser preference", () => {
  const h = harness("advanced");
  assert.equal(h.angle.hidden, false);
  h.simple.click();
  assert.equal(h.angle.hidden, true);
  assert.deepEqual(h.writes, [[MODE_KEY, "simple"]]);
  assert.equal(h.changes, 1);
  h.controller.restore({workspaceMode: "advanced"});
  assert.deepEqual(h.controller.preferences(), {workspaceMode: "advanced"});
  assert.equal(h.angle.hidden, false);
  assert.equal(h.writes.length, 1, "project restoration must not overwrite the global preference");
  h.controller.restore({workspaceMode: "simple"});
  assert.equal(h.angle.hidden, true);
  assert.deepEqual(h.config, h.snapshot);
});

test("controller works when browser storage throws and keeps active or invalid controls visible", () => {
  const h = harness(null, true);
  h.advanced.click(); h.simple.click();
  h.config.layers[0].angle_deg = 30;
  h.controller.update();
  assert.equal(h.angle.hidden, false);
  h.config.layers[0].angle_deg = 0;
  h.angleInput.validity.valid = false;
  h.controller.update();
  assert.equal(h.angle.hidden, false);
  assert.equal(h.elements.get("#workspace-invalid").hidden, false);
  h.elements.get("#workspace-invalid").click();
  assert.equal(h.doc.activeElement, h.angleInput);
});

test("collapse and reopen controls expose expanded state and return keyboard focus", () => {
  const h = harness();
  h.elements.get("#close-inputs").click();
  assert.equal(h.elements.get("#workspace-inputs").getAttribute("aria-expanded"), "false");
  assert.equal(h.doc.activeElement, h.elements.get("#workspace-inputs"));
  h.elements.get("#workspace-inputs").click();
  assert.equal(h.elements.get("#workspace-inputs").getAttribute("aria-expanded"), "true");
  assert.equal(h.doc.activeElement, h.elements.get("#close-inputs"));
});

test("recovery opens a collapsed layer, reveals optional ancestors and activates a hidden tab", () => {
  const h = harness();
  const body = h.element(); body.hidden = true;
  const toggle = h.element({index: "2"});
  const card = h.element(); card.querySelector = selector => selector === ".layer-toggle" ? toggle : body;
  const disclosure = h.element({workspaceAdvanced: "angle"}); disclosure.hidden = true;
  h.input.parentElement = disclosure;
  h.input.closest = selector => selector === ".layer-card" ? card : selector === ".configuration" ? h.element() : null;
  h.elements.get("#close-inputs").click();
  h.controller.reveal(h.input);
  assert.equal(body.hidden, false);
  assert.equal(disclosure.hidden, false);
  assert.equal(disclosure.dataset.workspaceRevealed, "true");
  assert.equal(h.layer, 2);
  assert.equal(toggle.getAttribute("aria-expanded"), "true");
  assert.equal(h.elements.get("#workspace-inputs").getAttribute("aria-expanded"), "true");
  h.input.closest = selector => selector === '[role="tabpanel"]' ? {id: "view-uncertainty"} : null;
  h.controller.reveal(h.input);
  assert.equal(h.tab, "uncertainty");
});

test("mode changes retain focus on an optional input until the user leaves it", () => {
  const h = harness("advanced");
  h.angleInput.focus();
  h.controller.restore({workspaceMode: "simple"});
  assert.equal(h.angle.hidden, false);
  h.doc.activeElement = null;
  h.controller.update();
  assert.equal(h.angle.hidden, true);
});

test("library entry installed by a later deferred module moves intact into the global toolbar", () => {
  const h = harness();
  const library = h.element();
  let opened = false;
  library.addEventListener("click", () => { opened = true; });
  h.elements.set("#open-materials-library", library);
  h.observers.forEach(notify => notify());
  assert.ok(h.elements.get("#workspace-library-slot").children.includes(library));
  library.click();
  assert.equal(opened, true, "moving the original button preserves its library handler");
});
