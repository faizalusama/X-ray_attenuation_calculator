"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const path = require("node:path");

class Element extends EventTarget {
  constructor(tag) {
    super();
    this.tag = tag;
    this.children = [];
    this.attributes = new Map();
    this.style = {};
  }

  append(...children) {
    this.children.push(...children);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name);
  }

  focus() {
    document.activeElement = this;
  }
}

globalThis.document = {
  createElement: tag => new Element(tag),
  createElementNS: (_, tag) => new Element(tag),
};
require("../static/guinier-probe.js");
const {create} = globalThis.WorkbenchGuinierProbe;

const provider = spawnSync("python", ["-B", "-c", `
import json
from pathlib import Path

from xray_workbench.guinier import guinier

rows = [line.split() for line in Path("tests/data/sasdue4.dat").read_text(encoding="ascii").splitlines()
        if line.strip() and not line.lstrip().startswith("#")]
answers = {}
for key, width in (("equal", 2), ("weighted", 3)):
    answers[key] = guinier({
        "intensity_data": "\\n".join(",".join(row[:width]) for row in rows),
        "qmin_inv_angstrom": 0.0,
        "qmax_inv_angstrom": 0.083,
    })
print(json.dumps(answers, allow_nan=False))
`], {cwd: path.join(__dirname, ".."), encoding: "utf8"});
assert.ifError(provider.error);
assert.equal(provider.status, 0, provider.stderr);
const answers = JSON.parse(provider.stdout);

function find(root, predicate) {
  if (predicate(root)) return root;
  for (const child of root.children) {
    const match = find(child, predicate);
    if (match) return match;
  }
  return null;
}

function checkReadout(readout, number, expected) {
  const values = readout.textContent.match(
    /^Observation (\d+) of 101 · q = ([\d.e+-]+) Å⁻¹ · Measured ln\(I\) = ([\d.e+-]+) · Fitted ln\(I\) = ([\d.e+-]+) · Residual \(measured − fitted\) = ([\d.e+-]+)$/
  );
  assert.ok(values, readout.textContent);
  const [, row, q, measured, fitted, residual] = values;
  assert.equal(Number(row), number);
  assert.equal(Number(q), expected[0]);
  for (const [index, value] of [measured, fitted, residual].entries()) {
    assert.ok(Math.abs(Number(value) - expected[index + 1]) < 1e-10, value);
  }
  assert.equal(residual[0], expected[3] > 0 ? "+" : "-");
}

test("Guinier probe requires aligned finite data from a Guinier result", () => {
  assert.equal(create(null), null);
  assert.equal(create({tool: "tomo_sampling", data: answers.equal.data}), null);
  assert.equal(create({tool: "guinier", data: {}}), null);
  const invalid = structuredClone(answers.equal);
  invalid.plots[0].traces[0].y.pop();
  assert.equal(create(invalid), null);
  invalid.plots[0].traces[0].y.push(null);
  assert.equal(create(invalid), null);
});

for (const [weighting, expectedMiddle] of [
  ["equal", [0.02960806, -0.05542148787547206, -0.05659497325187279, 0.001173485376400725]],
  ["weighted", [0.02960806, -0.05542148787547206, -0.05536047999880821, -0.00006100787666385443]],
]) {
  test(`Guinier probe shows the ${weighting} SASDUE4 fit at row 51 and both endpoints`, () => {
    const answer = answers[weighting];
    const {ln_intensity_fit: fitted, ln_intensity_residuals: residuals} = answer.data;
    const measured = answer.plots[0].traces[0].y;
    const original = structuredClone(answer);
    const root = create(answer);
    assert.equal(root.className, "technique-probe");
    const slider = find(root, node => node.id === "guinier-observation");
    const label = find(root, node => node.tag === "label");
    const readout = find(root, node => node.getAttribute("role") === "status");
    const trace = find(root, node => node.getAttribute("class") === "guinier-residual-trace");
    const marker = find(trace, node => node.getAttribute("class") === "guinier-residual-selection");
    const points = find(trace, node => node.tag === "polyline").getAttribute("points").split(" ");
    assert.equal(label.getAttribute("for"), slider.id);
    assert.equal(label.textContent, "Guinier observation");
    assert.equal(readout.tag, "output");
    assert.equal(readout.getAttribute("for"), slider.id);
    assert.equal(trace.getAttribute("aria-hidden"), "true");
    assert.equal(points.length, 101);
    assert.deepEqual([slider.type, slider.min, slider.max, slider.step, slider.value], ["range", "1", "101", "1", "1"]);

    slider.focus();
    // Model the range input events the browser emits after changing its value.
    for (const [number, expected] of [
      [1, [0.0103382, measured[0], fitted[0], residuals[0]]],
      [51, expectedMiddle],
      [101, [0.08249249, measured[100], fitted[100], residuals[100]]],
      [1, [0.0103382, measured[0], fitted[0], residuals[0]]],
    ]) {
      if (slider.value !== String(number)) {
        slider.value = String(number);
        slider.dispatchEvent(new Event("input"));
      }
      checkReadout(readout, number, expected);
      assert.equal(slider.getAttribute("aria-valuetext"), number + " of 101");
      assert.equal(marker.getAttribute("cx") + "," + marker.getAttribute("cy"), points[number - 1]);
      assert.equal(Number(marker.getAttribute("cy")) < 40, expected[3] > 0);
      assert.equal(find(root, node => node.id === slider.id), document.activeElement);
    }
    assert.equal(Number(points[0].split(",")[0]), 8);
    assert.equal(Number(points[100].split(",")[0]), 352);
    assert.ok(Math.abs(Number(points[50].split(",")[0]) - 47.53) < 0.01);
    assert.deepEqual(answer, original);
  });
}
