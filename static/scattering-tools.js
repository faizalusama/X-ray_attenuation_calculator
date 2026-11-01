"use strict";
(function () {
  const $ = s => document.querySelector(s);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const number = value => value == null ? "undefined" : typeof value === "number" ? value.toLocaleString("en-US", {maximumSignificantDigits: 6, useGrouping: false}) : String(value);
  const colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"];
  let latest = null, revision = 0, timer;
  function invalidate() {
    revision++; latest = null; $("#st-output").innerHTML = ""; $("#st-export").disabled = true;
  }
  function draw(answer) {
    $("#st-output").innerHTML = '<div class="ts-cards">' + answer.values.map(v => '<article class="ts-card"><div class="ts-card-label">' + esc(v.label) + '</div><div class="ts-card-value">' + esc(number(v.value)) + ' <span>' + esc(v.unit) + '</span></div></article>').join("") + '</div>' +
      '<ul>' + answer.notes.map(note => '<li>' + esc(note) + '</li>').join("") + '</ul>' +
      answer.tables.map(t => '<div class="ts-table-wrap" tabindex="0" role="region" aria-label="' + esc(t.caption) + '"><table class="ts-table"><caption>' + esc(t.caption) + '</caption><thead><tr>' + t.headers.map(h => '<th scope="col">' + esc(h) + '</th>').join("") + '</tr></thead><tbody>' + t.rows.map(row => '<tr>' + row.map((v,i) => (i ? '<td>' : '<th scope="row">') + esc(number(v)) + (i ? '</td>' : '</th>')).join("") + '</tr>').join("") + '</tbody></table></div>').join("") +
      '<div class="ts-plots">' + answer.plots.map((p,i) => '<div id="st-plot-' + i + '" class="ts-plot" role="img" aria-label="' + esc(p.title) + '"></div>').join("") + '</div>';
    if (!window.Plotly) return;
    const dark = document.documentElement.dataset.theme === "dark";
    answer.plots.forEach((p,i) => window.Plotly.react("st-plot-" + i,
      p.traces.map((t,j) => ({type: "scatter", mode: p.sticks ? "lines" : "lines", name: t.name, x: p.x, y: t.y, line: {width: 2.5, color: colors[j % colors.length]}})),
      {title: {text: p.title, font: {size: 15}}, margin: {l: 72, r: 18, t: 48, b: 62}, xaxis: {title: {text: p.xlabel}},
        yaxis: {title: {text: p.ylabel}, type: p.log_y ? "log" : "linear"}, font: {size: 13, color: dark ? "#dbe5ee" : "#253647"},
        paper_bgcolor: dark ? "#18222c" : "#fff", plot_bgcolor: dark ? "#18222c" : "#fff", legend: {orientation: "h"}},
      {responsive: true, displayModeBar: false}));
  }
  async function run() {
    if ($("#view-tools").hidden) return;
    const api = window.AttenuationWorkbench, layers = api.layers();
    if (!layers.length || !api.referenceEnergy()) return;
    const select = $("#st-layer"), kept = Math.min(Number(select.value) || 0, layers.length - 1);
    select.innerHTML = layers.map((layer,i) => '<option value="' + i + '">' + esc(layer.name) + '</option>').join(""); select.value = String(kept);
    document.querySelectorAll("[data-layer-choice]").forEach(s => {
      const previous = s.value, value = previous === "" ? Math.min(1, layers.length-1) : Math.min(Number(previous), layers.length-1);
      s.innerHTML = layers.map((layer,i) => '<option value="' + i + '">' + esc(layer.name) + '</option>').join(""); s.value = String(value);
    });
    const tool = $("#st-tool").value;
    document.querySelectorAll("[data-st-tool]").forEach(p => { p.hidden = p.dataset.stTool !== tool; });
    invalidate(); const id = revision;
    const payload = {tool, layer: layers[kept], layers, energy_keV: api.referenceEnergy()};
    document.querySelectorAll('[data-st-tool="' + tool + '"] [data-arg]').forEach(input => {
      payload[input.dataset.arg] = input.type === "checkbox" ? input.checked : (input.type === "number" || input.hasAttribute("data-number")) ? Number(input.value) : input.value;
    });
    $("#st-status").textContent = "Calculating…";
    try {
      const response = await fetch("/api/scattering-tools", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      const answer = await response.json(); if (id !== revision) return;
      if (!response.ok) throw new Error(answer.detail || "Scattering calculation failed.");
      latest = {...answer, configuration: payload}; draw(answer); $("#st-export").disabled = false;
      $("#st-status").textContent = "Calculated at " + api.referenceEnergy() + " keV. Export includes inputs and data provenance.";
    } catch (exception) { if (id === revision) $("#st-status").textContent = exception.message; }
  }
  function schedule() { clearTimeout(timer); timer = setTimeout(run, 250); }
  function install() {
    $("#st-tool").addEventListener("change", schedule);
    $("#st-controls").addEventListener("input", () => { invalidate(); schedule(); });
    $("#st-run").addEventListener("click", run);
    $("#st-export").addEventListener("click", () => { if (latest) window.AttenuationWorkbench.download(JSON.stringify(latest,null,2) + "\n", "scattering-tools.json", "application/json"); });
    window.addEventListener("attenuation:update", schedule);
  }
  if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", install);
})();
