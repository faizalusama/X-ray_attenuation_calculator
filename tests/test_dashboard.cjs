"use strict";
const test = require('node:test');
const assert = require('node:assert/strict');
const {LENGTH_UNITS, toMM, fromMM, valuesFor, traceData, thicknessMap, cleanPreferences, plotHeight} = require('../static/dashboard.js');
const {initial, cleanProject} = require('../static/app.js');
const near = (a,b) => assert.ok(Math.abs(a-b) <= 1e-12 * Math.max(1, Math.abs(b)), `${a} vs ${b}`);
const result = {energy_keV:[10,20,40], optical_depth:[2,1,.5], transmission:[Math.exp(-2),Math.exp(-1),Math.exp(-.5)], removed_fraction:[1-Math.exp(-2),1-Math.exp(-1),1-Math.exp(-.5)], layers:[{name:'Sample', path_length_mm:10, mu_linear_cm_inv:[2,1,.5], mu_mass_cm2_g:[1,.5,.25], photoelectric_cm2_g:[.8,.3,.1], coherent_cm2_g:[.1,.1,.05], incoherent_cm2_g:[.1,.1,.1]}]};
const config = {layers:[{thickness_mm:10}], target_transmission:.1};

test('unit changes preserve physical thickness over nm to metres',()=>{
  for (const unit of Object.keys(LENGTH_UNITS)) for (const mm of [0, 1e-6, .03, 1, 250, 1e9]) near(toMM(fromMM(mm, unit),unit),mm);
  assert.equal(fromMM(1,'µm'),1000);assert.equal(fromMM(1,'cm'),.1);assert.equal(toMM(1000,'nm'),.001);
  assert.equal(toMM(null,'µm'),null);
});
test('plot height grows with available screen space and remains readable',()=>{
  assert.equal(plotHeight(500,700),430);
  assert.equal(plotHeight(900,900),594);
  assert.equal(plotHeight(1800,1400),720);
});
test('HVL, TVL and attenuation lengths use ray-path coefficients and selected units',()=>{
  const hvl=valuesFor(result,config,'hvl',0,'µm')[0].values;
  hvl.forEach((value,i)=>near(Math.exp(-result.layers[0].mu_linear_cm_inv[i]*toMM(value,'µm')/10),.5));
  valuesFor(result,config,'tvl',0,'cm')[0].values.forEach((value,i)=>near(Math.exp(-result.layers[0].mu_linear_cm_inv[i]*value),.1));
  valuesFor(result,config,'length',0,'mm')[0].values.forEach((value,i)=>near(Math.exp(-result.layers[0].mu_linear_cm_inv[i]*value/10),Math.exp(-1)));
});
test('target-thickness sweep reaches the specified transmission at every sampled energy',()=>{
  valuesFor(result,config,'design',0,'µm')[0].values.forEach((value,i)=>near(Math.exp(-result.optical_depth[i]*toMM(value,'µm')/10),.1));
  assert.deepEqual(valuesFor({...result,optical_depth:[0,0,0]},{...config,layers:[{thickness_mm:0}]},'design')[0].values,[null,null,null]);
});
test('thickness map obeys Beer–Lambert at zero, one and scaled thickness',()=>{
  const map=thicknessMap(result,2,3);
  assert.deepEqual(map.x,result.energy_keV);assert.deepEqual(map.y,[0,1,2]);assert.deepEqual(map.z[0],[100,100,100]);
  map.z[1].forEach((value,i)=>near(value,result.transmission[i]*100));map.z[2].forEach((value,i)=>near(value,100*Math.exp(-2*result.optical_depth[i])));
});
test('dual-axis traces keep independent units and omit zeros only on logarithmic axes',()=>{
  const r={...result, transmission:[0,1e-200,1]};
  const t=traceData(r,config,'transmission',0,'mm','y',true)[0];
  assert.equal(t.y[0],null);near(t.y[1],1e-198);assert.equal(t.y[2],100);
  const m=traceData(result,config,'mass',0,'mm','y2',true)[0];
  assert.equal(m.yaxis,'y2');assert.deepEqual(m.y,[1,.5,.25]);assert.match(m.hovertemplate,/cm²\/g/);
  assert.equal(m.line.shape,'linear');assert.equal(m.line.simplify,false);
});
test('transmission intervals retain the 95% bounds in percentage units',()=>{
  const r={...result,uncertainty:{transmission_p025:[0,.2,.4],transmission_p975:[.3,.5,.9]}};
  const traces=traceData(r,config,'transmission',0,'mm','y',true);
  assert.deepEqual(traces[0].y,[null,20,40]);assert.deepEqual(traces[1].y,[30,50,90]);assert.equal(traces[1].fill,'tonexty');
});
test('restored settings validate toggles, layouts, independent scales, visibility and ranges',()=>{
  const prefs=cleanPreferences({dashboard:{gesture:'locked',logX:false,edges:false,linked:true,layout:'grid',font:16,maxScale:8,active:1,panels:[{primary:'transmission',secondary:'mass',logY:false,logY2:true,ranges:{xaxis:[1,2]},visibility:{'y-transmission-stack':'legendonly'}},{primary:'map',secondary:'mass',logY:true,ranges:{yaxis:[2,1],garbage:[0,1]}}]}});
  assert.equal(prefs.gesture,'locked');assert.equal(prefs.logX,false);assert.equal(prefs.panels[0].logY2,true);assert.equal(prefs.panels[0].visibility['y-transmission-stack'],'legendonly');
  assert.equal(prefs.panels[1].secondary,'none');assert.equal(prefs.panels[1].logY,false);assert.deepEqual(prefs.panels[1].ranges,{});
  assert.deepEqual(cleanPreferences({dashboard:{gesture:'execute',font:400,panels:[{primary:'__proto__'}]}}),{});
});
test('imported v2 backend selection is never silently dropped',()=>{
  const configuration={...structuredClone(initial),backend:'xraylib'};
  assert.equal(cleanProject({schema_version:1,configuration}).backend,'xraylib');
  configuration.backend='future_source';assert.equal(cleanProject({schema_version:1,configuration}).backend,'future_source');
  configuration.backend=23;assert.throws(()=>cleanProject({schema_version:1,configuration}),/Backend/);
});


test('every available quantity has an open default plot and existing settings survive expansion',()=>{
  const {allPanels,QUANTITIES,MAX_PLOTS}=require('../static/dashboard.js');
  const panels=allPanels();
  assert.equal(panels.length,11);assert.equal(MAX_PLOTS,11);
  assert.deepEqual(panels.map(p=>p.primary),Object.keys(QUANTITIES));
  assert.equal(new Set(panels.map(p=>p.id)).size,11);
  const existing={...panels[0],secondary:'mass',logY2:true};
  assert.equal(allPanels([existing])[0].secondary,'mass');
  assert.equal(cleanPreferences({dashboard:{layoutVersion:3,panels}}).panels.length,11);
});

test('independent plot scales and typography survive preference restoration',()=>{
  const p=cleanPreferences({dashboard:{panels:[{primary:'mass',logX:false,logY:true,font:18,bold:false,edges:false,grid:false,layers:false},{primary:'transmission',logX:true,font:15,bold:true}]}}).panels;
  assert.equal(p[0].logX,false); assert.equal(p[1].logX,true);
  assert.equal(p[0].font,18); assert.equal(p[1].font,15);
  for(const key of ['bold','edges','grid','layers']) assert.equal(p[0][key],false);
  assert.equal(cleanPreferences({dashboard:{panels:[{primary:'mass',font:999}]}}).panels[0].font,18);
});
test('linked energy ranges convert between linear and log coordinates safely',()=>{
  const {energyRange}=require('../static/dashboard.js');
  assert.deepEqual(energyRange([1,2],true,false),[10,100]);
  assert.deepEqual(energyRange([10,100],false,true),[1,2]);
  assert.deepEqual(energyRange([10,100],false,false),[10,100]);
  assert.equal(energyRange([-10,100],false,true),undefined);
  assert.equal(energyRange(undefined,true,false),undefined);
});
