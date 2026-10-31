"use strict";

// Runs the calculator's Python engine through Pyodide, off the main thread, so the
// page stays responsive. Everything happens in this browser; nothing is uploaded.
let handle = null;
let runtime = "";

function progress(text) { self.postMessage({type: "progress", text}); }

async function boot() {
  const started = performance.now();
  const steps = {};
  const mark = name => { steps[name] = (performance.now() - started) / 1000; };
  const base = new URL("../../engine/", self.location);
  const info = await (await fetch(new URL("build-info.json", base), {cache: "no-cache"})).json();
  progress("Loading the Python runtime (step 1 of 3)…");
  const {loadPyodide} = await import(`${info.pyodide_url}pyodide.mjs`);
  const pyodide = await loadPyodide({indexURL: info.pyodide_url});
  mark("runtime");
  progress("Loading the scientific libraries (step 2 of 3)…");
  await pyodide.loadPackage(info.pyodide_packages, {messageCallback: () => {}});
  await pyodide.loadPackage(new URL(info.xraydb_wheel, base).href, {messageCallback: () => {}});
  mark("libraries");
  progress("Loading the calculation engine (step 3 of 3)…");
  const archive = await (await fetch(new URL(info.engine_zip, base))).arrayBuffer();
  pyodide.unpackArchive(archive, "zip", {extractDir: "/home/pyodide/engine"});
  pyodide.runPython('import sys; sys.path.insert(0, "/home/pyodide/engine")');
  handle = pyodide.pyimport("xray_workbench.browser").handle;
  runtime = `Pyodide ${info.pyodide_version}`;
  mark("engine");
  self.postMessage({type: "ready", seconds: steps.engine, steps, runtime});
}

const ready = boot().catch(exception => {
  self.postMessage({type: "failed", message: String((exception && exception.message) || exception)});
  throw exception;
});

self.onmessage = async ({data}) => {
  await ready;
  const answer = handle(data.route, data.body ?? undefined, runtime);
  const [status, text] = answer.toJs();
  answer.destroy();
  self.postMessage({type: "result", id: data.id, status, text});
};
