"use strict";
(function () {
  const $ = s => document.querySelector(s);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const number = value => value == null ? "undefined" : typeof value === "number" ? (value !== 0 && (Math.abs(value) < 0.001 || Math.abs(value) >= 100000) ? value.toExponential(5) : value.toLocaleString("en-US", {maximumSignificantDigits: 6, useGrouping: false})) : String(value);
  const colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"];
  const standaloneTools = ["tomo_sampling", "guinier", "pdf_window", "projection"];
  let latest = null, revision = 0, timer, schematics = null;
  function invalidate() {
    schematics?.clear($("#st-tool").value);
    revision++; latest = null; $("#st-output").innerHTML = ""; $("#st-output").dataset.tool = ""; $("#st-status").textContent = "Waiting for updated calculation."; $("#st-export").disabled = true; $("#st-export-csv").disabled = true;
  }
  function curvesCSV(answer) {
    const rows = [["plot", "curve", "x", "x_units", "y", "y_units"]];
    for (const p of answer.plots) for (const t of p.traces) p.x.forEach((x,i) => rows.push([p.title,t.name,x,p.xlabel,t.y[i],p.ylabel]));
    return rows.map(row => row.map(value => '"' + String(value).replaceAll('"','""') + '"').join(",")).join("\r\n") + "\r\n";
  }
  function draw(answer) {
    schematics?.update(answer);
    $("#st-output").dataset.tool = answer.tool;
    $("#st-output").innerHTML = '<div class="ts-cards">' + answer.values.map(v => '<article class="ts-card"><div class="ts-card-label">' + esc(v.label) + '</div><div class="ts-card-value">' + esc(number(v.value)) + ' <span>' + esc(v.unit) + '</span></div></article>').join("") + '</div>' +
      '<ul>' + answer.notes.map(note => '<li>' + esc(note) + '</li>').join("") + '</ul>' +
      answer.tables.map(t => '<div class="ts-table-wrap" tabindex="0" role="region" aria-label="' + esc(t.caption) + '"><table class="ts-table"><caption>' + esc(t.caption) + '</caption><thead><tr>' + t.headers.map(h => '<th scope="col">' + esc(h) + '</th>').join("") + '</tr></thead><tbody>' + t.rows.map(row => '<tr>' + row.map((v,i) => (i ? '<td>' : '<th scope="row">') + esc(number(v)) + (i ? '</td>' : '</th>')).join("") + '</tr>').join("") + '</tbody></table></div>').join("") +
      '<div class="ts-plots">' + answer.plots.map((p,i) => '<div id="st-plot-' + i + '" class="ts-plot" role="img" aria-label="' + esc(p.title) + '"></div>').join("") + '</div>';
    if (!window.Plotly) return;
    const theme = getComputedStyle(document.documentElement);
    const panel = theme.getPropertyValue("--wb-panel").trim();
    const grid = theme.getPropertyValue("--wb-border").trim();
    const ink = theme.getPropertyValue("--wb-ink").trim();
    answer.plots.forEach((p,i) => window.Plotly.react("st-plot-" + i,
      p.traces.map((t,j) => ({type: "scatter", mode: "lines", name: t.name, x: p.x, y: t.y, line: {width: 2.5, color: colors[j % colors.length]}})),
      {title: {text: p.title, font: {size: 15}}, margin: {l: 72, r: 18, t: 48, b: 62}, xaxis: {title: {text: p.xlabel}, gridcolor: grid, automargin:true},
        yaxis: {title: {text: p.ylabel}, type: p.log_y ? "log" : "linear", gridcolor: grid, automargin:true}, font: {size: 13, color: ink},
        paper_bgcolor: panel, plot_bgcolor: panel, legend: {orientation: "h"}},
      {responsive: true, displayModeBar: false}));
  }
  async function run() {
    if ($("#view-tools").hidden) return;
    const api = window.AttenuationWorkbench, layers = api.layers();
    const tool = $("#st-tool").value;
    const needsLayer = !standaloneTools.includes(tool) && tool !== "detector";
    $("#st-layer-label").hidden = !needsLayer;
    document.querySelectorAll("[data-st-tool]").forEach(panel => { panel.hidden = panel.dataset.stTool !== tool; });
    invalidate(); const id = revision;
    const energy = api.referenceEnergy();
    if ((needsLayer && !layers.length) || (!standaloneTools.includes(tool) && !energy)) return;
    const select = $("#st-layer"), kept = Math.min(Number(select.value) || 0, layers.length - 1);
    select.innerHTML = layers.map((layer,i) => '<option value="' + i + '">' + esc(layer.name) + '</option>').join(""); select.value = String(kept);
    document.querySelectorAll("[data-layer-choice]").forEach(s => {
      const previous = s.value, value = previous === "" ? Math.min(1, layers.length-1) : Math.min(Number(previous), layers.length-1);
      s.innerHTML = layers.map((layer,i) => '<option value="' + i + '">' + esc(layer.name) + '</option>').join(""); s.value = String(value);
    });
    const payload = {tool};
    if (!standaloneTools.includes(tool)) payload.energy_keV = energy;
    if (needsLayer) Object.assign(payload, {layer: layers[kept], layers});
    document.querySelectorAll('[data-st-tool="' + tool + '"] [data-arg]').forEach(input => {
      payload[input.dataset.arg] = input.type === "checkbox" ? input.checked : (input.type === "number" || input.hasAttribute("data-number")) ? (input.value.trim() === "" ? null : Number(input.value)) : input.value;
    });
    $("#st-status").textContent = "Calculating…";
    try {
      const response = await fetch("/api/scattering-tools", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      const answer = await response.json(); if (id !== revision) return;
      if (!response.ok) throw new Error(answer.detail || "Scattering calculation failed.");
      latest = {...answer, configuration: payload}; draw(answer); $("#st-export").disabled = false; $("#st-export-csv").disabled = !answer.plots.length;
      $("#st-status").textContent = standaloneTools.includes(tool) ? "Calculation complete. Inputs and results are ready to export." :
        "Calculated at " + energy + " keV. Export includes inputs and data provenance.";
    } catch (exception) { if (id === revision) $("#st-status").textContent = exception.message; }
  }
  function schedule() { clearTimeout(timer); timer = setTimeout(run, 250); }
  function install() {
    schematics = window.TechniqueSchematics.create($("#st-schematic"), (field, value) => {
      const input = document.querySelector('[data-st-tool="' + $("#st-tool").value + '"] [data-arg="' + field + '"]');
      input.value = value;
      input.dispatchEvent(new Event("input", {bubbles: true}));
    });
    $("#st-sq-file").addEventListener("change", async event => {
      const file = event.target.files[0]; if (!file) return;
      try {
        if (file.size > 800000) throw new Error("S(Q) CSV must be smaller than 800,000 bytes.");
        $("#st-sq-data").value = await file.text(); invalidate(); schedule();
      } catch (exception) { $("#st-status").textContent = exception.message; }
      finally { event.target.value = ""; }
    });
    $("#st-demo-pdf").addEventListener("click", () => {
      const rows = Array.from({length: 4001}, (_,i) => { const q = i/100; return q + "," + (1 + (q === 0 ? 3 : Math.exp(-q)*Math.sin(3*q)/q)); });
      $("#st-sq-data").value = "q,S\n" + rows.join("\n"); invalidate(); schedule();
    });
    $("#st-tool").addEventListener("change", () => { invalidate(); schedule(); });
    $("#st-controls").addEventListener("input", () => { invalidate(); schedule(); });
    $("#st-run").addEventListener("click", run);
    $("#st-export-csv").addEventListener("click", () => { if (latest) window.AttenuationWorkbench.download(curvesCSV(latest), "scattering-tools.csv", "text/csv;charset=utf-8"); });
    $("#st-export").addEventListener("click", () => { if (latest) window.AttenuationWorkbench.download(JSON.stringify(latest,null,2) + "\n", "scattering-tools.json", "application/json"); });
    window.addEventListener("attenuation:update", schedule);
  }
  if (typeof module !== "undefined") module.exports = {curvesCSV};
  if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", install);
})();
