"use strict";

// Browser build only. Answers the local server's /api routes from the same Python
// engine, running in a Web Worker, so app.js and dashboard.js need no second code path.
(function () {
  const ROUTES = {"/api/calculate": "calculate", "/api/presets": "presets", "/api/health": "health", "/api/xray-sources": "xray-sources"};
  const LOCAL_INSTALL = "https://github.com/faizalusama/X-ray_attenuation_calculator#start-the-graphical-application";
  const pending = new Map();
  const nativeFetch = window.fetch.bind(window);
  const worker = new Worker(new URL("engine_worker.js", document.currentScript.src), {type: "module"});
  const strip = document.createElement("div");
  let nextId = 1, failure = null, settleReady;

  strip.id = "engine-status";
  strip.setAttribute("role", "status");
  strip.textContent = "Loading the calculator. The first visit takes longer; later visits use your browser's cache.";
  document.addEventListener("DOMContentLoaded", () => document.body.prepend(strip));
  setTimeout(() => {
    if (!strip.className) strip.textContent += " This is taking longer than usual; please keep this page open.";
  }, 30000);

  const ready = new Promise((resolve, reject) => { settleReady = {resolve, reject}; });
  ready.catch(() => {});

  function request(route, body) {
    if (failure) return Promise.reject(failure);
    return new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, {resolve, reject});
      worker.postMessage({id, route, body});
    });
  }

  function fail(message) {
    failure = new Error(`The calculator could not load in this browser: ${message}`);
    strip.className = "failed";
    strip.textContent = "The calculator could not load. Check your connection and refresh, try a current browser, or ";
    const link = document.createElement("a");
    link.href = LOCAL_INSTALL;
    link.textContent = "install the local version";
    strip.append(link, ".");
    settleReady.reject(failure);
    for (const {reject} of pending.values()) reject(failure);
    pending.clear();
  }

  worker.onmessage = ({data}) => {
    if (data.type === "progress") strip.textContent = data.text;
    else if (data.type === "failed") fail(data.message);
    else if (data.type === "ready") {
      strip.className = "ready";
      strip.textContent = "Runs in your browser. Nothing you enter is uploaded.";
      window.AttenuationWeb.loadSeconds = data.seconds;
      window.AttenuationWeb.loadSteps = data.steps;
      window.AttenuationWeb.runtime = data.runtime;
      settleReady.resolve();
    } else if (data.type === "result") {
      const call = pending.get(data.id);
      pending.delete(data.id);
      if (call) call.resolve({status: data.status, text: data.text});
    }
  };
  worker.onerror = event => { if (!failure) fail(event.message || "the engine stopped unexpectedly"); };

  window.fetch = function (input, init) {
    const route = ROUTES[typeof input === "string" ? input : input.url];
    if (!route) return nativeFetch(input, init);
    return request(route, init && init.body).then(({status, text}) =>
      new Response(text, {status, headers: {"Content-Type": "application/json"}}));
  };

  function download(contents, filename, type) {
    const url = URL.createObjectURL(new Blob([contents], {type: type || "application/octet-stream"}));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  window.AttenuationWeb = {ready, request, download};
})();
