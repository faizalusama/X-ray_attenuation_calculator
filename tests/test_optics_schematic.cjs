"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {opticsScene} = require("../static/optics-schematic.js");
const answer = {plots: [{x: [0, 45, 90], traces: [{y: [2, 10, 15]}]}],
  data: {delta: 1e-6, beta: 2e-8, critical_angle_deg: null, wavelength_angstrom: 1.5}};

test("optics probe pairs angle and depth without interpolation", () => {
  const scene = opticsScene(answer, 1);
  assert.equal(scene.exact.grazing_angle_deg, 45);
  assert.equal(scene.exact.intensity_depth_nm, 10);
  assert.equal(scene.exact.critical_angle_deg, null);
  assert.equal(scene.maxIndex, 2);
  assert.match(scene.summary, /from surface/);
  assert.match(scene.summary, /symbolic/);
});
test("optics clamps selection and renders finite zero and normal angles", () => {
  for (const [input, expected] of [[-5, 0], [999, 2], [NaN, 0], [Infinity, 0]]) {
    const scene = opticsScene(answer, input);
    assert.equal(scene.index, expected);
    assert.doesNotMatch(scene.markup, /NaN|Infinity/);
    assert.equal(scene.exact.intensity_depth_nm, answer.plots[0].traces[0].y[expected]);
  }
  assert.match(opticsScene(answer, 0).markup, /x1="160" y1="180"/);
  const incident = opticsScene(answer, 1).markup.match(/<line x1="([^"]+)" y1="([^"]+)" x2="300" y2="180"/);
  assert.ok(Math.abs((300 - Number(incident[1])) - (180 - Number(incident[2]))) < 1e-12);
});
