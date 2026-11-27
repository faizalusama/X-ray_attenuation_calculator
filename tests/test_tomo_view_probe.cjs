"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {runInNewContext} = require("node:vm");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");

// Implement only the DOM surface used by the probe so the tests need no browser or UI dependency.
class Element {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.attributes = new Map();
    this.listeners = new Map();
    this.style = {};
  }
  append(...children) { this.children.push(...children); }
  setAttribute(name, value) { this.attributes.set(name, String(value)); }
  getAttribute(name) { return this.attributes.get(name); }
  addEventListener(event, handler) { this.listeners.set(event, handler); }
  dispatch(event) { this.listeners.get(event)(); }
  find(predicate) {
    if (predicate(this)) return this;
    for (const child of this.children) {
      const found = child.find(predicate);
      if (found) return found;
    }
    return undefined;
  }
}

const document = {
  createElement: tag => new Element(tag),
  createElementNS: (_namespace, tag) => new Element(tag),
};
const context = {document};
const filename = join(__dirname, "../static/tomo-view-probe.js");
runInNewContext(readFileSync(filename, "utf8"), context, {filename});
const {create} = context.WorkbenchTomoViewProbe;

const formax = Object.freeze({
  minimum_views: 1609,
  angular_step_deg: 0.11187072715972654,
  rotation_range_deg: 180,
  nyquist_resolution_um: 8,
  pitch_supports_resolution: true,
});
const plan = Object.freeze({
  planned_views: 1600,
  planned_angular_step_deg: 0.1125,
  planned_views_meet_minimum: false,
  planned_view_shortfall: 9,
});

const createProbe = data => create({tool: "tomo_sampling", data: Object.freeze(data)});
const byId = (panel, id) => panel.find(node => node.id === id);

function controls(panel) {
  const range = byId(panel, "tomo-view-index");
  const selector = byId(panel, "tomo-view-schedule");
  const readout = byId(panel, "tomo-view-readout");
  return {range, selector, readout};
}

function selectView(range, view) {
  range.value = String(view);
  range.dispatch("input");
}

function selectSchedule(selector, schedule) {
  selector.value = schedule;
  selector.dispatch("change");
}

test("ForMAX planned views use a one-based native slider and an independently rotating sample", () => {
  const panel = createProbe({...formax, ...plan});
  const {range, selector, readout} = controls(panel);
  assert.equal(panel.className, "technique-probe");
  assert.equal(range.tagName, "input");
  assert.equal(range.type, "range");
  assert.equal(range.min, "1");
  assert.equal(range.step, "1");
  assert.equal(range.max, "1600");
  assert.equal(selector.tagName, "select");
  assert.equal(selector.value, "planned");
  assert.equal(selector.disabled, false);
  for (const control of [range, selector]) {
    const label = panel.find(node => node.tagName === "label" && node.htmlFor === control.id);
    assert.ok(label?.textContent);
  }
  assert.equal(readout.getAttribute("role"), "status");
  assert.equal(readout.textContent, "View 1 of 1600 · 0°");
  assert.equal(byId(panel, "tomo-view-pixel-status").textContent,
    "Pixel sampling supports the target period. The floor is 8 µm.");
  assert.equal(byId(panel, "tomo-view-angular-status").textContent,
    "The planned schedule falls short by 9 views.");

  const svg = panel.find(node => node.tagName === "svg");
  assert.equal(svg.getAttribute("role"), "img");
  assert.match(svg.getAttribute("aria-label"), /fixed X-ray source.*fixed detector.*vertical axis/);
  assert.match(panel.find(node => node.tagName === "figcaption").textContent, /Illustrative/);
  const instrument = svg.innerHTML;
  const sample = svg.find(node => node.getAttribute("class") === "tomo-view-sample");
  const initialPose = sample.innerHTML;
  for (const [view, angle] of [[2, "0.1125"], [801, "90"], [1600, "179.8875"]]) {
    selectView(range, view);
    assert.equal(readout.textContent, `View ${view} of 1600 · ${angle}°`);
    assert.equal(range.getAttribute("aria-valuetext"), `View ${view} of 1600, angle ${angle} degrees`);
    assert.notEqual(sample.innerHTML, initialPose);
    assert.equal(svg.innerHTML, instrument);
  }

  selectSchedule(selector, "minimum");
  assert.equal(range.max, "1609");
  assert.equal(range.value, "1609");
  assert.equal(readout.textContent, "View 1609 of 1609 · 179.8881292728°");
  assert.match(byId(panel, "tomo-view-angular-status").textContent, /planned schedule.*9 views/);
  selectView(range, 1);
  selectSchedule(selector, "planned");
  assert.equal(readout.textContent, "View 1 of 1600 · 0°");
});

test("an angularly sufficient ForMAX plan can fail the independent pixel check", () => {
  const panel = createProbe({...formax, ...plan,
    planned_views: 1609,
    planned_angular_step_deg: formax.angular_step_deg,
    planned_views_meet_minimum: true,
    planned_view_shortfall: 0,
    nyquist_resolution_um: 10,
    pitch_supports_resolution: false,
  });
  assert.match(byId(panel, "tomo-view-angular-status").textContent, /meets the angular minimum \(0 additional views/);
  assert.match(byId(panel, "tomo-view-pixel-status").textContent, /below the pixel sampling floor of 10 µm/);
});

test("the minimum schedule remains available when the plan is omitted", () => {
  const panel = createProbe(formax);
  const {range, selector, readout} = controls(panel);
  assert.equal(selector.disabled, true);
  assert.equal(selector.value, "minimum");
  assert.deepEqual(selector.children.map(option => option.value), ["minimum"]);
  assert.equal(byId(panel, "tomo-view-angular-status"), undefined);
  assert.ok(byId(panel, "tomo-view-pixel-status"));
  assert.equal(readout.textContent, "View 1 of 1609 · 0°");
  selectView(range, 2);
  assert.equal(readout.textContent, "View 2 of 1609 · 0.1118707271597°");
});

test("large schedules keep exact view indices and the final half-open angle", () => {
  const count = Number.MAX_SAFE_INTEGER;
  const panel = createProbe({...formax, minimum_views: count, angular_step_deg: 1.998401444325282e-14});
  const {range, readout} = controls(panel);
  assert.equal(range.disabled, false);
  assert.equal(range.max, String(count));
  selectView(range, count - 1);
  assert.equal(readout.textContent, "View 9007199254740990 of 9007199254740991 · 180° − 3.996802888651e-14°");
  selectView(range, count);
  assert.equal(readout.textContent, "View 9007199254740991 of 9007199254740991 · 180° − 1.998401444325e-14°");
  assert.match(range.getAttribute("aria-valuetext"), /angle 180 degrees minus 1\.998401444325e-14 degrees$/);
});

test("a schedule beyond the browser's exact integer range has a clear fallback", () => {
  const panel = createProbe({...formax, ...plan,
    planned_views: 1e100,
    planned_angular_step_deg: 1.8e-98,
    planned_views_meet_minimum: true,
    planned_view_shortfall: 0,
  });
  const {range, selector, readout} = controls(panel);
  assert.equal(selector.value, "minimum");
  assert.equal(range.max, "1609");
  selectSchedule(selector, "planned");
  assert.equal(range.disabled, true);
  assert.match(readout.textContent, /too many views for a slider with exact integer steps/);
  selectSchedule(selector, "minimum");
  assert.equal(range.disabled, false);
  assert.equal(readout.textContent, "View 1 of 1609 · 0°");
});

test("answers without a valid minimum schedule do not produce a probe", () => {
  assert.equal(create(null), null);
  assert.equal(create({tool: "guinier", data: formax}), null);
  assert.equal(createProbe({...formax, minimum_views: 0}), null);
  assert.equal(createProbe({...formax, angular_step_deg: NaN}), null);
});
