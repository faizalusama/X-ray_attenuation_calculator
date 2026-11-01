"use strict";
const test = require('node:test');
const assert = require('node:assert/strict');
const {LENGTH_UNITS, toMM, fromMM, valuesFor, traceData, thicknessMap, cleanPreferences, plotHeight} = require('../static/dashboard.js');
const {initial, cleanProject} = require('../static/app.js');
const near = (a,b) => assert.ok(Math.abs(a-b) <= 1e-12 * Math.max(1, Math.abs(b)), `${a} vs ${b}`);
const result = {energy_keV:[10,20,40], optical_depth:[2,1,.5], transmission:[Math.exp(-2),Math.exp(-1),Math.exp(-.5)], removed_fraction:[1-Math.exp(-2),1-Math.exp(-1),1-Math.exp(-.5)], layers:[{name:'Sample', path_length_mm:10, mu_linear_cm_inv:[2,1,.5], mu_mass_cm2_g:[1,.5,.25], photoelectric_cm2_g:[.8,.3,.1], coherent_cm2_g:[.1,.1,.05], incoherent_cm2_g:[.1,.1,.1]}]};
const config = {layers:[{thickness_mm:10}], target_transmission:.1};
const {plotReadout, readableNumber, allPanels} = require('../static/dashboard.js');
const referenceResult = {...result, reference:{energy_keV:30, transmission:.456789123, removed_fraction:.543210877, optical_depth:.783533429, target_transmission:.1, target_thickness_scale:2.938719}, layers:result.layers.map(l=>({...l, reference:{mu_mass_cm2_g:.3917667145, mu_linear_cm_inv:.783533429, transmission:.456789123, hvl_mm:8.84642, tvl_mm:29.38719, attenuation_length_mm:12.76269}}))};

test('every default plot has a text readout with named quantities and explicit units',()=>{
  for(const p of allPanels()){
    const readout=plotReadout(referenceResult,config,p);
    assert.match(readout.text,/Reference 30 keV/);
    assert.ok(readout.rows.length>0,p.primary);
    for(const row of readout.rows){assert.ok(row.name);assert.ok(row.unit);assert.ok(Number.isFinite(row.value));}
  }
});

test('text readouts use direct reference values instead of nearest sweep transmission',()=>{
  const p=allPanels()[0], readout=plotReadout(referenceResult,config,p);
  assert.equal(readout.rows[0].value,100*referenceResult.reference.transmission);
  assert.notEqual(readout.rows[0].value,100*result.transmission[1]);
  const offSweep=plotReadout({...referenceResult,reference:{...referenceResult.reference,energy_keV:80}},config,p);
  assert.match(offSweep.text,/outside the plotted sweep/);
  assert.equal(offSweep.rows[0].value,readout.rows[0].value);
});

test('reference readouts respect hidden series, layer controls and independent axes',()=>{
  const p={...allPanels()[0],secondary:'mass',logY2:true};
  const r={...referenceResult,layers:[...referenceResult.layers,{...referenceResult.layers[0],name:'Second layer'}]};
  assert.equal(plotReadout(r,config,p).rows.length,5);
  assert.equal(plotReadout(r,config,{...p,layers:false}).rows.length,3);
  const hidden=plotReadout(r,config,{...p,layers:false,visibility:{'y-transmission-stack':'legendonly','y2-mass-mass-1':false}});
  assert.equal(hidden.rows.length,1);assert.equal(hidden.rows[0].axis,'Right');assert.equal(hidden.rows[0].unit,'cm²/g');
  assert.match(plotReadout(referenceResult,config,{...allPanels()[0],visibility:{'y-transmission-stack':'legendonly'}}).text,/All series are hidden/);
});

test('reference readouts preserve zero and very small values on log plots',()=>{
  const r={...referenceResult,reference:{...referenceResult.reference,transmission:0}};
  const zero=plotReadout(r,config,{...allPanels()[0],logY:true});
  assert.equal(zero.rows[0].value,0);assert.match(zero.rows[0].note,/Zero is omitted/);
  assert.equal(readableNumber(1e-200),'1.00000e-200');assert.equal(readableNumber(null),'Unavailable');
  assert.equal(readableNumber(Infinity),'Unavailable');assert.equal(readableNumber(0),'0');
});

test('channel readout discloses the exact evaluated sample and direct reference total',()=>{
  const p=allPanels().find(p=>p.primary==='channels');
  const readout=plotReadout(referenceResult,config,p);
  assert.equal(readout.rows[0].energy,20);assert.equal(readout.rows[0].value,.3);
  assert.match(readout.notes[0],/total mass attenuation at the reference is 0.391767/);
  assert.match(readout.notes[0],/sample at 20 keV; no interpolation/);
  assert.match(readout.text,/sampled at 20 keV/);
});

test('path length and target readouts convert the exact API values to display units',()=>{
  const p=allPanels().find(p=>p.primary==='hvl');
  assert.equal(plotReadout(referenceResult,config,p,'µm').rows[0].value,referenceResult.layers[0].reference.hvl_mm*1000);
  const design={...p,primary:'design'};
  assert.equal(plotReadout(referenceResult,config,design,'cm').rows[0].value,referenceResult.reference.target_thickness_scale);
  assert.equal(plotReadout({...referenceResult,reference:{...referenceResult.reference,target_thickness_scale:null}},config,design).rows[0].value,null);
});

test('map readout labels scaling and uses the reference optical depth',()=>{
  const p=allPanels().find(p=>p.primary==='map');
  const rows=plotReadout(referenceResult,config,p,'mm',.5).rows;
  assert.equal(rows.length,2);assert.equal(rows[0].value,100);
  assert.equal(rows[1].value,100*Math.exp(-referenceResult.reference.optical_depth*.5));
  assert.equal(rows[1].name,'0.5× thickness');
  assert.match(plotReadout(referenceResult,config,p,'mm',.5).text,/0.5× thickness/);
  assert.match(plotReadout(referenceResult,config,p).text,/1× thickness/);
});

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

const {plotTable, allCurvesTable, HC_KEV_ANGSTROM} = require('../static/dashboard.js');
const twoLayers = {...result, reference:{target_transmission:.1}, layers:[0,1].map(i=>({name:`L${i}`, thickness_mm:i+1, path_length_mm:i+1, mu_mass_cm2_g:[3,2,1], mu_linear_cm_inv:[6,4,2], photoelectric_cm2_g:[2,1,.5], coherent_cm2_g:[.5,.5,.25], incoherent_cm2_g:[.5,.5,.25]}))};
test('all-curves export names every column with a unit and keeps full precision',()=>{
  const table=allCurvesTable(twoLayers), column=name=>table.headers.indexOf(name);
  assert.equal(table.columns.length,table.headers.length);
  for(const c of table.columns){assert.ok(c.unit,c.name);assert.ok(c.description,c.name);}
  assert.equal(table.rows.length,3);
  for(const row of table.rows) assert.equal(row.length,table.headers.length);
  near(table.rows[0][column('wavelength_angstrom')],HC_KEV_ANGSTROM/10);
  assert.equal(table.rows[1][column('transmission')],result.transmission[1]);
  near(table.rows[0][column('layer_2_optical_depth')],6*2/10);
  near(table.rows[0][column('layer_2_hvl_mm')],10*Math.LN2/6);
  near(table.rows[0][column('target_total_thickness_mm')],3*-Math.log(.1)/2);
  assert.equal(column('transmission_p025'),-1);
});
test('all-curves export adds the uncertainty interval only when it was calculated',()=>{
  const table=allCurvesTable({...twoLayers,uncertainty:{transmission_p025:[.1,.2,.3],transmission_p975:[.2,.3,.4]}});
  assert.equal(table.rows[2][table.headers.indexOf('transmission_p975')],.4);
});
test('plot data follows the plot: primary and secondary series, display units, and the thickness map',()=>{
  const table=plotTable(twoLayers,{layers:[{thickness_mm:1},{thickness_mm:2}],target_transmission:.1},{primary:'transmission',secondary:'optical_depth',layer:'all'},'mm',3);
  assert.equal(table.headers[0],'energy_keV');
  assert.ok(table.headers.length>=3);
  assert.deepEqual(table.rows.map(row=>row[0]),result.energy_keV);
  const map=plotTable(twoLayers,{layers:[{thickness_mm:1},{thickness_mm:2}],target_transmission:.1},{primary:'map',secondary:'none',layer:'all'},'mm',3);
  assert.deepEqual(map.headers.slice(1),result.energy_keV);
  assert.equal(map.rows[0].length,result.energy_keV.length+1);
});
