"use strict";
const test = require('node:test');
const assert = require('node:assert/strict');
const {curvesTable} = require('../static/scattering.js');

test('the curves export has one fixed-name column per quantity and pair',()=>{
  const table=curvesTable({curves:{q:[0,1],form_factors:{Si:[14,12],O:[8,6]},mean_f_squared:[100,64],mean_of_f_squared:[108,70],laue:[8,6],
    pairs:[{pair:'O-O',weights:[.28,.25]},{pair:'O-Si',weights:[.5,.5]},{pair:'Si-Si',weights:[.22,.25]}]}});
  assert.deepEqual(table.headers,['q_inv_angstrom','f0_Si','f0_O','mean_f_squared','mean_of_f_squared','laue_diffuse','weight_O-O','weight_O-Si','weight_Si-Si']);
  assert.deepEqual(table.rows[1],[1,12,6,64,70,6,.25,.5,.25]);
});
