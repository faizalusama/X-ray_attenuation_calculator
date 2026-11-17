"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {pdfScene} = require("../static/pdf-schematic.js");
const fixture = (sq, modified) => ({data: {q: [0, 1, 2], sq, r_angstrom: [0, 2, 4],
  g_inv_angstrom2: [0, 0, 0], window: "lorch"}, plots: [{}, {traces: [{y: modified}]}]});

test("PDF probe reads actual input and modified integrand", () => {
  const answer = fixture([1, -2, 3], [0, -2.5, 0]);
  const scene = pdfScene(answer, 1);
  assert.equal(scene.exact.q_inv_angstrom, 1);
  assert.equal(scene.exact.sq, -2);
  assert.equal(scene.exact.modified_integrand_inv_angstrom, -2.5);
  assert.equal(scene.exact.q_max_inv_angstrom, 2);
  assert.equal(scene.exact.r_max_angstrom, 4);
  assert.match(scene.summary, /entire Q interval/);
});
test("flat S(Q) and boundary selections stay finite", () => {
  const answer = fixture([1, 1, 1], [0, 0, 0]);
  for (const [input, expected] of [[-10, 0], [99, 2], [NaN, 0], [Infinity, 0]]) {
    const scene = pdfScene(answer, input);
    assert.equal(scene.index, expected);
    assert.equal(scene.exact.modified_integrand_inv_angstrom, 0);
    assert.doesNotMatch(scene.markup, /NaN|Infinity/);
  }
});
