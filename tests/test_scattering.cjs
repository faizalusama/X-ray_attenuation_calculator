"use strict";
const test = require('node:test');
const assert = require('node:assert/strict');
const {curvesTable} = require('../static/scattering.js');
const {detectorScene, projectionScene} = require('../static/technique-schematics.js');

test('detector probe keeps validated radius, angle and q paired at boundaries', () => {
  const data = {distance_mm: 100, inner_radius_mm: 0, outer_radius_mm: 200,
    radius_mm: [0, 100, 200], two_theta_deg: [0, 45, 63.43494882292201],
    q_inv_angstrom: [0, 4.8089418390747705, 6.606531998388247]};
  const middle = detectorScene(data, 1);
  assert.equal(middle.radius, 100);
  assert.equal(middle.angle, 45);
  assert.equal(middle.q, 4.8089418390747705);
  assert.equal(detectorScene(data, 200).q, data.q_inv_angstrom[2]);
  assert.equal(detectorScene(data, -2).radius, 0);
  assert.doesNotMatch(middle.markup, /NaN|Infinity/);
});

test('count bars use a shared scale and preserve above-flat transmissions', () => {
  const scene = projectionScene({sample_counts: 2050, flat_counts: 1050,
    dark_counts: 50, transmission: 2, line_integral: -0.6931471805599453});
  assert.match(scene.markup, /width="440".*sg-transmitted/);
  assert.match(scene.markup, /width="225.3658536585366".*sg-flat/);
  assert.match(scene.summary, /projection -0.69315/);
});

test('the curves export has one fixed-name column per quantity and pair',()=>{
  const table=curvesTable({curves:{q:[0,1],form_factors:{Si:[14,12],O:[8,6]},mean_f_squared:[100,64],mean_of_f_squared:[108,70],laue:[8,6],
    pairs:[{pair:'O-O',weights:[.28,.25]},{pair:'O-Si',weights:[.5,.5]},{pair:'Si-Si',weights:[.22,.25]}]}});
  assert.deepEqual(table.headers,['q_inv_angstrom','f0_Si','f0_O','mean_f_squared','mean_of_f_squared','laue_diffuse','weight_O-O','weight_O-Si','weight_Si-Si']);
  assert.deepEqual(table.rows[1],[1,12,6,64,70,6,.25,.5,.25]);
});

test('tool CSV preserves precision and quotes curve metadata',()=>{
  const {curvesCSV}=require('../static/scattering-tools.js');
  const csv=curvesCSV({plots:[{title:'A, B',xlabel:'r (Å)',ylabel:'G',x:[0,.1234567890123456],traces:[{name:'G(r)',y:[0,1.2]}]}]});
  assert.ok(csv.includes('"A, B"'));
  assert.ok(csv.includes('0.1234567890123456'));
  assert.equal(csv.split('\r\n').filter(Boolean).length,3);
});

test('batch CSV quotes names and prevents spreadsheet formula execution',()=>{
  const {exportCSV}=require('../static/comparison.js');
  const csv=exportCSV({rows:[{row:2,name:'=1+1',transmission:0.5,error:null}]});
  assert.ok(csv.includes('"' + "'=1+1" + '"'));
  assert.ok(csv.includes('"0.5"'));
});
