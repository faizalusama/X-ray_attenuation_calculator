"use strict";
const test = require('node:test');
const assert = require('node:assert/strict');
const {esc, fmt, sub, FAMILIES, CATEGORY_LABEL, familyFor, filterMaterials, libraryOrder, massPercent, withComparison, shortestHvl, MAX_COMPARE} = require('../static/materials.js');
const items = [
  {id:'zirconia', name:'Zirconia', category:'oxide_ceramic', density_status:'sourced', formulas:['ZrO2']},
  {id:'sic', name:'Silicon carbide', category:'non_oxide_ceramic', density_status:'unverified', formulas:['SiC']},
  {id:'alumina', name:'Alumina', category:'oxide_ceramic', density_status:'unverified', formulas:['Al2O3']},
  {id:'air', name:'Air, dry', category:'gas', density_status:'sourced'},
  {id:'mystery', name:'Mystery', category:'not_a_category', density_status:'sourced', formulas:[]},
];
const ids = list => list.map(m => m.id);

test('escaping neutralises every HTML-significant character and missing values',()=>{
  assert.equal(esc(`<a href="x" title='y'>&</a>`),'&lt;a href=&quot;x&quot; title=&#39;y&#39;&gt;&amp;&lt;/a&gt;');
  assert.equal(esc(null),'');assert.equal(esc(undefined),'');assert.equal(esc(0),'0');
});

test('number formatting keeps significant digits and drops only trailing zeros',()=>{
  assert.equal(fmt(2.5),'2.5');assert.equal(fmt(100),'100');assert.equal(fmt(1000),'1000');
  assert.equal(fmt(0.012345),'0.01235');assert.equal(fmt(12345),'1.235e+4');assert.equal(fmt(100000),'1e+5');
  assert.equal(fmt(1.5e-7),'1.5e-7');assert.equal(fmt(0.1234567, 6),'0.123457');assert.equal(fmt(-2.000),'-2');
  for (const missing of [null, undefined, NaN, Infinity, -Infinity]) assert.equal(fmt(missing),'—');
});

test('formulas subscript stoichiometric numbers after escaping',()=>{
  assert.equal(sub('Al2O3'),'Al<sub>2</sub>O<sub>3</sub>');
  assert.equal(sub('Na0.5K0.5NbO3'),'Na<sub>0.5</sub>K<sub>0.5</sub>NbO<sub>3</sub>');
  assert.equal(sub('<b>'),'&lt;b&gt;');
});

test('every labelled category belongs to exactly one family and unknown ones fall back to the last',()=>{
  const members = FAMILIES.flatMap(f => f.members);
  assert.equal(new Set(members).size, members.length);
  for (const category of Object.keys(CATEGORY_LABEL)) assert.ok(members.includes(category), category);
  assert.equal(familyFor('glass').id,'glass');assert.equal(familyFor('gas').id,'reference');
  assert.equal(familyFor('not_a_category'), FAMILIES[FAMILIES.length - 1]);
});

test('search matches every word in any order across id, name and formula, ignoring case',()=>{
  assert.deepEqual(ids(filterMaterials(items)), ids(items));
  assert.deepEqual(ids(filterMaterials(items, {query:'  '})), ids(items));
  assert.deepEqual(ids(filterMaterials(items, {query:'sic'})), ['sic']);
  assert.deepEqual(ids(filterMaterials(items, {query:'CARBIDE silicon'})), ['sic']);
  assert.deepEqual(ids(filterMaterials(items, {query:'al2o3'})), ['alumina']);
  assert.deepEqual(ids(filterMaterials(items, {query:'zirconia carbide'})), []);
});

test('family and sourced-density filters combine with search',()=>{
  assert.deepEqual(ids(filterMaterials(items, {family:'oxide'})), ['zirconia','alumina']);
  assert.deepEqual(ids(filterMaterials(items, {family:'oxide', sourcedOnly:true})), ['zirconia']);
  assert.deepEqual(ids(filterMaterials(items, {sourcedOnly:true, query:'a'})), ['zirconia','air']);
  assert.deepEqual(ids(filterMaterials(items, {family:'special'})), ['mystery']);
});

test('library order follows the family legend, then name, without mutating the input',()=>{
  const before = ids(items);
  assert.deepEqual(ids(libraryOrder(items)), ['alumina','zirconia','sic','air','mystery']);
  assert.deepEqual(ids(items), before);
});

test('mass percentages keep three decimals only for trace elements',()=>{
  assert.equal(massPercent(0.5),'50.00');assert.equal(massPercent(0.001),'0.10');
  assert.equal(massPercent(0.0009),'0.090');assert.equal(massPercent(1),'100.00');
});

test('comparison toggles a material and keeps at most three, dropping the oldest',()=>{
  assert.equal(MAX_COMPARE, 3);
  const start = ['a'];
  assert.deepEqual(withComparison(start,'b'),['a','b']);assert.deepEqual(start,['a']);
  assert.deepEqual(withComparison(['a','b'],'a'),['b']);
  assert.deepEqual(withComparison(['a','b','c'],'d'),['b','c','d']);
  assert.deepEqual(withComparison(['a','b','c'],'b'),['a','c']);
});

test('the shortest half-value layer is reported, or nothing for an empty selection',()=>{
  const points = [{id:'a', hvl_mm:2}, {id:'b', hvl_mm:0.5}, {id:'c', hvl_mm:0.5}, {id:'d', hvl_mm:9}];
  assert.equal(shortestHvl(points).id,'b');assert.deepEqual(ids(points),['a','b','c','d']);
  assert.equal(shortestHvl([]), undefined);
});
