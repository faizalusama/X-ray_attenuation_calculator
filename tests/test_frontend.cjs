"use strict";
const test = require('node:test');
const assert = require('node:assert/strict');
const {cleanProject, parseSpectrumText, logTicks, linePath, intervalPath, exportedSVG, initial, limits} = require('../static/app.js');
const project = () => ({schema_version:1,configuration:structuredClone(initial)});
const {createLiveScheduler} = require('../static/app.js');

function fakeClock() {
  let next = 0;
  const pending = new Map();
  return {set(callback) {pending.set(++next, callback);return next;}, clear(id) {pending.delete(id);},
    fire() {const callbacks = [...pending.values()];pending.clear();callbacks.forEach(callback=>callback());},
    get size() {return pending.size;}};
}
const settle = () => new Promise(resolve=>setImmediate(resolve));

test('live updates debounce typing and read only the latest configuration',async()=>{
  const clock=fakeClock(), values=[];let input=1;
  const live=createLiveScheduler(async options=>values.push({input,...options}),300,clock);
  live.schedule();input=2;live.schedule();input=3;live.schedule();
  assert.equal(clock.size,1);assert.equal(values.length,0);
  clock.fire();await settle();
  assert.deepEqual(values,[{input:3,automatic:true}]);
});

test('edits during a running calculation coalesce into one latest follow-up',async()=>{
  const clock=fakeClock(), values=[], finish=[];let input=1, active=0, maximum=0;
  const live=createLiveScheduler(async()=>{
    active++;maximum=Math.max(maximum,active);values.push(input);
    await new Promise(resolve=>finish.push(resolve));active--;
  },300,clock);
  live.schedule();clock.fire();
  input=2;live.schedule();clock.fire();input=3;live.schedule();clock.fire();
  assert.deepEqual(values,[1]);finish.shift()();await settle();
  assert.deepEqual(values,[1,3]);assert.equal(maximum,1);
  finish.shift()();await settle();
});

test('manual refresh cancels the debounce without creating duplicate work',async()=>{
  const clock=fakeClock(), calls=[];
  const live=createLiveScheduler(async options=>calls.push(options),300,clock);
  live.schedule();await live.flush();clock.fire();await settle();
  assert.deepEqual(calls,[{automatic:false}]);
});

test('a failed live request does not stop subsequent updates',async()=>{
  const clock=fakeClock(), errors=[];let calls=0;
  const live=createLiveScheduler(async()=>{if(++calls===1)throw new Error('offline');},300,clock,e=>errors.push(e.message));
  live.schedule();clock.fire();await settle();live.schedule();clock.fire();await settle();
  assert.equal(calls,2);assert.deepEqual(errors,['offline']);
});

test('project round trips composition, spectrum, uncertainties and negative angle without coercion',()=>{
  const p=project(); p.configuration.layers[0].angle_deg=-45;
  p.configuration.layers[0].name='A'.repeat(200);
  p.configuration.uncertainty={enabled:true,samples:100,seed:4294967295};
  p.configuration.spectrum={energy_keV:[30,20,20],weights:[5,0,7],weighting:'energy',label:'Measured'};
  assert.deepEqual(cleanProject(p),p.configuration);
});
test('project inputs match bounded API capacities',()=>{
  const mutations=[
    [p=>p.configuration.layers=Array(25).fill(p.configuration.layers[0]), /1–24/],
    [p=>p.configuration.layers[0].components=Array(41).fill({formula:'SiO2',fraction:1}), /1–40/],
    [p=>p.configuration.energy.points=4001, /4000/],
    [p=>p.configuration.energy.points=2.5, /integer/],
    [p=>p.configuration.uncertainty.enabled='false', /true or false/],
    [p=>p.configuration.uncertainty.samples=99, /100/],
    [p=>p.configuration.energy.min_keV=null, /finite number/],
    [p=>p.configuration.layers[0].name='A'.repeat(201), /200/],
    [p=>p.configuration.target_transmission=0, /strictly/],
    [p=>p.configuration.layers[0].angle_deg=89.9, /strictly/],
    [p=>p.configuration.layers[0].porosity=.2, /bulk density/],
    [p=>p.configuration.layers[0].components.forEach(c=>c.fraction=0), /positive/],
  ];
  for(const [mutate,message] of mutations){const p=project();mutate(p);assert.throws(()=>cleanProject(p),message);}
});
test('volume and ideal density require constituent densities',()=>{
  for(const mode of ['volume','ideal']) {
    const p=project();if(mode==='volume')p.configuration.layers[0].basis='volume';else p.configuration.layers[0].density_mode='ideal';
    assert.throws(()=>cleanProject(p),/Constituent density/);
    p.configuration.layers[0].components.forEach(c=>c.density_g_cm3=2);
    assert.equal(cleanProject(p).layers[0].components[0].density_g_cm3,2);
  }
});
test('spectrum parser preserves scientific notation, original bin order, duplicates and CSV quotes',()=>{
  const parsed=parseSpectrumText('\uFEFFenergy_keV,weight\n"30", "1e2"\n10;2\n10\t0','energy',' Sample ');
  assert.deepEqual(parsed,{energy_keV:[30,10,10],weights:[100,2,0],weighting:'energy',label:'Sample'});
  assert.deepEqual(parseSpectrumText('1e1 2e2').energy_keV,[10]);
});
test('spectrum rejects malformed first row, empty fields, invalid range and oversized input',()=>{
  for(const source of ['nonsense,weight\n20,5','20,,5','20,','20,-1','20,0','801,1','NaN,2']) assert.throws(()=>parseSpectrumText(source));
  assert.throws(()=>parseSpectrumText(Array(limits.spectrumBins+1).fill('20,1').join('\n')),/10,000/);
});
test('project spectrum rejects empty/nonpositive and mismatched arrays',()=>{
  for(const spectrum of [{energy_keV:[],weights:[]},{energy_keV:[10],weights:[0]},{energy_keV:[10,20],weights:[1]}]){
    const p=project();p.configuration.spectrum={...spectrum,weighting:'photon'};assert.throws(()=>cleanProject(p));
  }
});
test('logarithmic line never replaces zero with an invented small value or connects through zeros',()=>{
  const line=linePath([1,2,3,4],[1e-200,0,1e-100,1],x=>x,y=>Math.log10(y),true);
  assert.equal(line,'M1.00,-200.00 M3.00,-100.00 L4.00,0.00');
  assert.equal(linePath([1,2],[0,1],x=>x,y=>y,false),'M1.00,0.00 L2.00,1.00');
});
test('log uncertainty bands split before zero quantiles and retain true positive values',()=>{
  const band=intervalPath([1,2,3],[.1,0,.01],[.5,.2,.1],x=>x,y=>Math.log10(y),true);
  assert.match(band,/M1\.00/);assert.match(band,/M3\.00/);assert.doesNotMatch(band,/Infinity|NaN|[ML]2\.00,/);
});
test('logarithmic tick generation covers extremely attenuated positive values without overflow',()=>{
  const ticks=logTicks(1e-300,1,8);assert.ok(ticks.length>1);assert.ok(ticks.length<=8);assert.ok(ticks.every(v=>Number.isFinite(v)&&v>0));
});
test('standalone SVG has visible title, legend, units, sources and exact embedded configuration',()=>{
  const provenance={engine:'XrayDB / Elam',xraydb_version:'4.5.8',workbench_version:'0.1.0',model:'narrow-beam Beer-Lambert',timestamp_utc:'2026-09-20T12:00:00Z',configuration_sha256:'a'.repeat(64)};
  const chart={width:870,height:356,title:'Mass attenuation',subtitle:'Selected material <sample>',series:[{label:'Glass & ceramic',color:'#218a74'}],logX:true,logY:true};
  const output=exportedSVG('<text>Mass attenuation · cm²/g</text>',chart,provenance,initial,'mass',false);
  assert.match(output,/<text[^>]*>Mass attenuation<\/text>/);
  assert.match(output,/>Glass &amp; ceramic<\/text>/);
  assert.match(output,/cm²\/g/);assert.match(output,/>Source: XrayDB/);assert.match(output,/Configuration SHA-256/);
  assert.match(output,/zero values omitted/);assert.match(output,/<metadata>/);assert.match(output,/&quot;reference_keV&quot;:30/);
});
