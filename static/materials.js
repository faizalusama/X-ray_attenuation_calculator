"use strict";

// Materials library browser: search, provenance, attenuation preview, library
// landscape, and "add to stack". Self-contained; it talks to the application
// only through window.AttenuationWorkbench, so presets and library entries
// enter the calculation through exactly the same path.
(() => {
  if (typeof document === "undefined") return;

  const $ = (selector, scope = document) => scope.querySelector(selector);
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const fmt = (value, digits = 4) => (value == null || !Number.isFinite(value)) ? "—" : Number(value).toPrecision(digits).replace(/(\.\d*?)0+(e|$)/, "$1$2").replace(/\.(e|$)/, "$1");
  const sub = formula => esc(formula).replace(/(\d+(?:\.\d+)?)/g, "<sub>$1</sub>");

  // Families use a colour-blind-conscious base palette plus distinct neutral
  // engineering colours. Marker shape also communicates provenance status.
  const FAMILIES = [
    {id: "oxide", label: "Oxide ceramics", color: "#0072B2", members: ["oxide_ceramic", "nuclear_ceramic"]},
    {id: "nonoxide", label: "Non-oxide ceramics", color: "#D55E00", members: ["non_oxide_ceramic"]},
    {id: "silicate", label: "Silicates, bioceramics, cement", color: "#009E73", members: ["silicate", "bioceramic", "cement_phase"]},
    {id: "functional", label: "Electro- and magnetic ceramics", color: "#CC79A7", members: ["electroceramic", "magnetic_ceramic"]},
    {id: "detector", label: "Scintillators and halides", color: "#E69F00", members: ["scintillator", "halide"]},
    {id: "semiconductor", label: "Semiconductors", color: "#56B4E9", members: ["semiconductor"]},
    {id: "glass", label: "Glasses, borates, sulfates", color: "#7A5195", members: ["glass", "glass_ceramic", "borate", "sulfate"]},
    {id: "concrete", label: "Concrete", color: "#8C6D31", members: ["concrete"]},
    {id: "reference", label: "Reference media (tissue, polymer, fluid)", color: "#8A8F94", members: ["biological_reference", "polymer", "liquid", "gas", "detector_medium"]},
    {id: "engineering", label: "Alloys and engineering materials", color: "#4E5D6C", members: ["alloy", "engineering_reference"]},
    {id: "natural", label: "Geological materials", color: "#A6761D", members: ["geological"]},
    {id: "special", label: "Nuclear and energetic materials", color: "#B2182B", members: ["nuclear_material", "explosive"]},
  ];
  const FAMILY_OF = Object.fromEntries(FAMILIES.flatMap(f => f.members.map(m => [m, f])));
  const CATEGORY_LABEL = {
    oxide_ceramic: "Oxide ceramic", non_oxide_ceramic: "Non-oxide ceramic", nuclear_ceramic: "Nuclear ceramic",
    silicate: "Silicate", bioceramic: "Bioceramic", electroceramic: "Electroceramic", magnetic_ceramic: "Magnetic ceramic",
    scintillator: "Scintillator", halide: "Halide", semiconductor: "Semiconductor", glass: "Glass",
    glass_ceramic: "Glass-ceramic", borate: "Borate", sulfate: "Sulfate", concrete: "Concrete", cement_phase: "Cement phase",
    biological_reference: "Biological reference", polymer: "Polymer", liquid: "Liquid", gas: "Gas", detector_medium: "Detector medium",
    alloy: "Alloy", geological: "Geological material", explosive: "Energetic material", nuclear_material: "Nuclear material",
    engineering_reference: "Engineering reference",
  };
  const TIER_LABEL = {reference_data: "Reference data", stoichiometric: "Stoichiometric", literature: "Literature"};
  const COMPARE_COLORS = ["#D55E00", "#009E73", "#CC79A7"];
  const MAX_COMPARE = 3;

  const ui = {items: [], query: "", family: "", sourcedOnly: false, selected: null, compare: [],
              details: new Map(), view: "material", landscape: new Map(), landscapeEnergy: 60, landscapeAxis: "hvl_mm"};

  function markup() {
    const dialog = document.createElement("dialog");
    dialog.id = "materials-dialog";
    dialog.className = "ml-dialog";
    dialog.setAttribute("aria-labelledby", "ml-title");
    dialog.innerHTML = `
      <header class="ml-header">
        <div><span class="ml-eyebrow">MATERIALS LIBRARY</span><h2 id="ml-title">Choose a material with known provenance</h2>
        <p id="ml-subtitle" class="ml-subtitle">Loading library…</p></div>
        <button type="button" class="ml-close" id="ml-close" aria-label="Close materials library">✕</button>
      </header>
      <div class="ml-body">
        <aside class="ml-browse" aria-label="Find a material">
          <label class="ml-field">Search name or formula<input id="ml-search" type="search" autocomplete="off" placeholder="e.g. zirconia, SiC, glass"></label>
          <label class="ml-field">Family<select id="ml-family"><option value="">All families</option>${FAMILIES.map(f => `<option value="${f.id}">${esc(f.label)}</option>`).join("")}</select></label>
          <label class="ml-check"><input id="ml-sourced" type="checkbox"> Sourced density only</label>
          <p id="ml-count" class="ml-count" role="status" aria-live="polite"></p>
          <ul id="ml-list" class="ml-list" aria-label="Materials"></ul>
        </aside>
        <section class="ml-main">
          <div class="ml-tabs" role="tablist" aria-label="Library views">
            <button type="button" role="tab" id="ml-tab-material" aria-controls="ml-view-material" aria-selected="true">Material</button>
            <button type="button" role="tab" id="ml-tab-landscape" aria-controls="ml-view-landscape" aria-selected="false" tabindex="-1">Library landscape</button>
          </div>
          <div id="ml-view-material" role="tabpanel" aria-labelledby="ml-tab-material" class="ml-view"></div>
          <div id="ml-view-landscape" role="tabpanel" aria-labelledby="ml-tab-landscape" class="ml-view" hidden>
            <div class="ml-landscape-controls">
              <label class="ml-field inline">Energy (keV)<input id="ml-energy" type="number" min="1" max="800" step="any" value="60"></label>
              <label class="ml-field inline">Vertical axis<select id="ml-axis">
                <option value="hvl_mm">Half-value layer (mm)</option>
                <option value="mu_mass_cm2_g">Mass attenuation μ/ρ (cm²/g)</option>
                <option value="mu_linear_cm_inv">Linear attenuation μ (1/cm)</option></select></label>
              <p class="ml-hint">Each point is one library entry at its listed density. <strong>Filled</strong> markers use a sourced density; <strong>hollow</strong> markers use an unverified typical density. Select a point to open it.</p>
            </div>
            <div id="ml-landscape-plot" class="ml-plot tall" role="img" aria-label="Library landscape plot"></div>
            <p id="ml-landscape-summary" class="ml-summary"></p>
          </div>
        </section>
      </div>`;
    document.body.appendChild(dialog);
    return dialog;
  }

  function familyFor(category) { return FAMILY_OF[category] || FAMILIES[FAMILIES.length - 1]; }

  function filtered() {
    const words = ui.query.toLowerCase().split(/\s+/).filter(Boolean);
    return ui.items.filter(m => {
      if (ui.family && familyFor(m.category).id !== ui.family) return false;
      if (ui.sourcedOnly && m.density_status !== "sourced") return false;
      const hay = `${m.id} ${m.name} ${(m.formulas || []).join(" ")}`.toLowerCase();
      return words.every(w => hay.includes(w));
    });
  }

  function renderList() {
    const list = filtered();
    $("#ml-count").textContent = `${list.length} of ${ui.items.length} materials`;
    $("#ml-list").innerHTML = list.map(m => {
      const fam = familyFor(m.category);
      const pressed = m.id === ui.selected;
      return `<li><button type="button" class="ml-item" data-id="${esc(m.id)}" aria-pressed="${pressed}">
        <span class="ml-swatch" style="background:${fam.color}" aria-hidden="true"></span>
        <span class="ml-item-text"><span class="ml-item-name">${esc(m.name)}</span>
        <span class="ml-item-meta">${esc(CATEGORY_LABEL[m.category] || m.category)} · ${fmt(m.density_g_cm3, 4)} g/cm³
        <span class="ml-density ${m.density_status}">${m.density_status === "sourced" ? "sourced" : "unverified"}</span></span></span></button></li>`;
    }).join("") || `<li class="ml-empty">No materials match. Clear the search or choose another family.</li>`;
  }

  async function getJSON(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = `${response.status}`;
      try { detail = (await response.json()).detail || detail; } catch { /* keep status */ }
      throw new Error(detail);
    }
    return response.json();
  }

  async function detail(id) {
    if (!ui.details.has(id)) ui.details.set(id, await getJSON(`/api/materials/${encodeURIComponent(id)}`));
    return ui.details.get(id);
  }

  function compositionBar(fractions) {
    const entries = Object.entries(fractions).sort((a, b) => b[1] - a[1]);
    const shades = ["#1f4e79", "#2e7d99", "#3a9d8f", "#78b17a", "#c2b75a", "#d98b4a", "#b35c63", "#7d5a8c", "#5c6f82", "#9aa5ad"];
    const pct = w => (w * 100).toFixed(w < 0.001 ? 3 : 2);
    const text = entries.map(([el, w]) => `${el} ${pct(w)}%`).join(", ");
    const segments = entries.map(([el, w], i) => `<span class="ml-seg" style="flex-basis:${w * 100}%;background:${shades[i % shades.length]}" title="${esc(el)}: ${pct(w)} wt%">${w > 0.07 ? esc(el) : ""}</span>`).join("");
    return `<div class="ml-bar" role="img" aria-label="Elemental mass fractions: ${esc(text)}">${segments}</div>
      <p class="ml-bar-legend">${entries.map(([el, w], i) => `<span><i style="background:${shades[i % shades.length]}"></i>${esc(el)} ${pct(w)}%</span>`).join("")}</p>`;
  }

  function provenance(d) {
    const s = d.source || {};
    const doi = s.doi ? `<a href="https://doi.org/${esc(s.doi)}" target="_blank" rel="noopener noreferrer">doi:${esc(s.doi)} ↗</a>` : "";
    const url = s.url ? `<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">source page ↗</a>` : "";
    const retrieved = s.retrieved_utc ? `Retrieved ${esc(s.retrieved_utc.slice(0, 10))}.` : "";
    const lit = d.literature ? `<p>Paper: <a href="https://doi.org/${esc(d.literature.doi)}" target="_blank" rel="noopener noreferrer">doi:${esc(d.literature.doi)}</a>, ${esc(d.literature.locator)}. Values checked by ${esc(d.literature.verification?.values_checked_by)} on ${esc(d.literature.verification?.values_checked_on)}.</p>` : "";
    const caveats = (s.caveats || []).map(c => `<li>${esc(c)}</li>`).join("");
    return `<details class="ml-provenance" open><summary>Source and provenance</summary>
      <p><strong>${esc(s.title || s.id)}</strong>${s.authors ? ` — ${esc(s.authors)}` : ""}${s.publisher ? `. ${esc(s.publisher)}` : ""}.</p>
      <p>${[doi, url].filter(Boolean).join(" · ")} ${retrieved}</p>${s.method ? `<p>${esc(s.method)}</p>` : ""}${lit}
      ${caveats ? `<ul>${caveats}</ul>` : ""}</details>`;
  }

  async function renderMaterial() {
    const view = $("#ml-view-material");
    if (!ui.selected) {
      view.innerHTML = `<div class="ml-placeholder"><h3>Select a material</h3><p>Choose an entry on the left to see its composition, the source of every number, and its attenuation. Use <strong>Library landscape</strong> to see every entry at once.</p></div>`;
      return;
    }
    view.setAttribute("aria-busy", "true");
    let d;
    try { d = await detail(ui.selected); } catch (exc) { view.innerHTML = `<p class="ml-error" role="alert">Could not load this material: ${esc(exc.message)}</p>`; return; }
    const fam = familyFor(d.category);
    const unverified = d.density.status !== "sourced";
    const comps = d.composition.components.map(c => `<tr><td>${sub(c.formula)}</td><td>${fmt(c.fraction, 6)}</td></tr>`).join("");
    const basis = {mass: "mass fraction", mole: "mole fraction (formula units)", volume: "volume fraction"}[d.composition.basis];
    const comparing = ui.compare.includes(d.id);
    view.innerHTML = `
      <div class="ml-title-row">
        <div><h3 id="ml-material-name">${esc(d.name)}</h3>
        <p class="ml-badges"><span class="ml-badge" style="--c:${fam.color}">${esc(CATEGORY_LABEL[d.category] || d.category)}</span>
        <span class="ml-badge tier">${esc(TIER_LABEL[d.tier] || d.tier)}</span>
        <span class="ml-badge ${unverified ? "warn" : "ok"}">${unverified ? "⚠ Density unverified" : "✓ Density sourced"}</span></p></div>
        <div class="ml-actions">
          <label class="ml-field inline">Thickness (mm)<input id="ml-thickness" type="number" min="0" step="any" value="1"></label>
          <button type="button" class="ml-primary" id="ml-add">Add to stack</button>
          <button type="button" class="button outline small" id="ml-compare" aria-pressed="${comparing}">${comparing ? "Remove from comparison" : "Compare"}</button>
        </div>
      </div>
      <div class="ml-grid">
        <section class="ml-card"><h4>Composition <span>(${esc(basis)})</span></h4>
          <table class="ml-table"><thead><tr><th scope="col">Component</th><th scope="col">Fraction</th></tr></thead><tbody>${comps}</tbody></table>
          <h4>Elemental mass fractions</h4><div id="ml-elements"><p class="ml-hint">Calculating…</p></div></section>
        <section class="ml-card"><h4>Density</h4>
          <p class="ml-density-value">${fmt(d.density.value_g_cm3, 4)} <span>g/cm³</span> <em>(${esc(d.density.kind)})</em></p>
          <p class="${unverified ? "ml-warning" : "ml-note"}">${esc(d.density.note)}</p>
          ${d.notes ? `<p class="ml-note">${esc(d.notes)}</p>` : ""}
          ${provenance(d)}</section>
      </div>
      <section class="ml-card ml-chart-card">
        <div class="ml-chart-head"><h4>Mass attenuation μ/ρ, 1–800 keV</h4>
          <p class="ml-hint">Solid: total. Dashed: photoelectric, coherent, incoherent. ${ui.compare.length ? "Coloured lines: compared materials (total)." : "Use <strong>Compare</strong> to overlay up to three other materials."}</p></div>
        <div id="ml-curve" class="ml-plot" role="img" aria-label="Mass attenuation coefficient of ${esc(d.name)} from 1 to 800 keV"></div>
        <p id="ml-curve-summary" class="ml-summary"></p>
      </section>`;
    view.removeAttribute("aria-busy");
    $("#ml-add").addEventListener("click", () => addToStack(d));
    $("#ml-compare").addEventListener("click", () => toggleCompare(d.id));
    drawCurve(d);
  }

  async function drawCurve(d) {
    const ids = [d.id, ...ui.compare.filter(id => id !== d.id)];
    const details = await Promise.all(ids.map(detail));
    let result;
    try {
      result = await getJSON("/api/calculate", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({energy: {min_keV: 1, max_keV: 800, points: 320, spacing: "log", reference_keV: 60},
                              layers: details.map(x => x.layer)})});
    } catch (exc) { $("#ml-curve-summary").textContent = `Could not calculate: ${exc.message}`; return; }
    if (ui.selected !== d.id) return; // the user moved on while this was running
    const E = result.energy_keV, main = result.layers[0];
    $("#ml-elements").innerHTML = compositionBar(main.elemental_mass_fractions);
    const positive = values => values.map(v => (v > 0 ? v : null));
    const traces = [
      {x: E, y: positive(main.mu_mass_cm2_g), name: `${d.name} — total`, line: {color: "#203951", width: 2.6}},
      {x: E, y: positive(main.photoelectric_cm2_g), name: "Photoelectric", line: {color: "#0072B2", width: 1.4, dash: "dash"}},
      {x: E, y: positive(main.coherent_cm2_g), name: "Coherent (Rayleigh)", line: {color: "#009E73", width: 1.4, dash: "dot"}},
      {x: E, y: positive(main.incoherent_cm2_g), name: "Incoherent (Compton)", line: {color: "#E69F00", width: 1.4, dash: "dashdot"}},
      ...details.slice(1).map((x, i) => ({x: E, y: positive(result.layers[i + 1].mu_mass_cm2_g), name: x.name,
                                          line: {color: COMPARE_COLORS[i], width: 2.2}})),
    ].map(t => ({type: "scatter", mode: "lines", hovertemplate: "%{x:.4g} keV<br>%{y:.4g} cm²/g<extra>%{fullData.name}</extra>", ...t}));
    const layout = {margin: {l: 70, r: 16, t: 10, b: 52}, font: {family: "Segoe UI, Arial, sans-serif", size: 13, color: "#253647"},
      xaxis: {type: "log", title: {text: "Photon energy (keV)"}, gridcolor: "#e3eaee", zeroline: false},
      yaxis: {type: "log", title: {text: "μ/ρ (cm²/g)"}, gridcolor: "#e3eaee", exponentformat: "power", zeroline: false},
      legend: {orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: {size: 12}}, paper_bgcolor: "#fff", plot_bgcolor: "#fff", hovermode: "x unified"};
    if (window.Plotly) window.Plotly.react("ml-curve", traces, layout, {responsive: true, displaylogo: false, modeBarButtonsToRemove: ["select2d", "lasso2d"]});
    const ref = main.reference;
    $("#ml-curve-summary").textContent = `At 60 keV: μ/ρ = ${fmt(ref.mu_mass_cm2_g)} cm²/g, μ = ${fmt(ref.mu_linear_cm_inv)} cm⁻¹, half-value layer ${fmt(ref.hvl_mm)} mm at ${fmt(d.density.value_g_cm3)} g/cm³` +
      (d.density.status !== "sourced" ? " (unverified density)." : ".") +
      (details.length > 1 ? ` Compared with ${details.slice(1).map((x, i) => `${x.name}: ${fmt(result.layers[i + 1].reference.mu_mass_cm2_g)} cm²/g`).join("; ")}.` : "");
  }

  function toggleCompare(id) {
    if (ui.compare.includes(id)) ui.compare = ui.compare.filter(x => x !== id);
    else {
      if (ui.compare.length >= MAX_COMPARE) ui.compare.shift();
      ui.compare.push(id);
    }
    renderMaterial();
  }

  function addToStack(d) {
    const thickness = Number($("#ml-thickness").value);
    if (!Number.isFinite(thickness) || thickness < 0) { $("#ml-thickness").setCustomValidity("Enter a thickness of 0 mm or more."); $("#ml-thickness").reportValidity(); return; }
    $("#ml-thickness").setCustomValidity("");
    const api = window.AttenuationWorkbench;
    if (!api) return;
    const layer = {...d.layer, thickness_mm: thickness};
    const note = d.density.status === "sourced"
      ? `Added ${d.name} from ${d.source?.id || "the library"}.`
      : `Added ${d.name}. Its density is a typical value without a checked source — replace it with your measured bulk density.`;
    if (api.addLayer(layer, note)) $("#materials-dialog").close();
  }

  async function renderLandscape() {
    const energy = ui.landscapeEnergy;
    const plot = $("#ml-landscape-plot"), summary = $("#ml-landscape-summary");
    if (!(energy >= 1 && energy <= 800)) { summary.textContent = "Enter an energy between 1 and 800 keV."; return; }
    summary.textContent = "Calculating every entry…";
    let data = ui.landscape.get(energy);
    try {
      if (!data) { data = await getJSON(`/api/materials/landscape?energy_keV=${encodeURIComponent(energy)}`); ui.landscape.set(energy, data); }
    } catch (exc) { summary.textContent = `Could not calculate the landscape: ${exc.message}`; return; }
    const axis = ui.landscapeAxis;
    const axisTitle = {hvl_mm: "Half-value layer (mm)", mu_mass_cm2_g: "μ/ρ (cm²/g)", mu_linear_cm_inv: "μ (1/cm)"}[axis];
    // Gases sit three orders of magnitude below every solid; plotting them
    // would squeeze the whole solid library into one corner of the density axis.
    const gases = data.points.filter(p => p.category === "gas");
    const visible = new Set(filtered().filter(m => m.category !== "gas").map(m => m.id));
    const traces = [];
    for (const fam of FAMILIES) {
      for (const sourced of [true, false]) {
        const pts = data.points.filter(p => familyFor(p.category).id === fam.id && (p.density_status === "sourced") === sourced && visible.has(p.id));
        if (!pts.length) continue;
        traces.push({type: "scatter", mode: "markers", name: `${fam.label} — ${sourced ? "sourced" : "unverified"} density`,
          legendgroup: fam.id,
          x: pts.map(p => p.density_g_cm3), y: pts.map(p => p[axis]), customdata: pts.map(p => p.id), text: pts.map(p => p.name),
          // Plotly draws an open symbol's outline in marker.color, so the family
          // colour goes there for both; only the symbol changes with status.
          marker: sourced ? {size: 10, color: fam.color, symbol: "circle", line: {color: "#fff", width: 0.8}}
                          : {size: 10, color: fam.color, symbol: "circle-open", line: {width: 2}},
          hovertemplate: `%{text}<br>ρ = %{x:.4g} g/cm³<br>${esc(axisTitle)} = %{y:.4g}${sourced ? "" : "<br><i>unverified density</i>"}<extra></extra>`});
      }
    }
    const layout = {margin: {l: 74, r: 16, t: 10, b: 56}, font: {family: "Segoe UI, Arial, sans-serif", size: 13, color: "#253647"},
      xaxis: {type: "log", title: {text: "Density (g/cm³)"}, gridcolor: "#e3eaee", zeroline: false},
      yaxis: {type: "log", title: {text: `${axisTitle} at ${energy} keV`}, gridcolor: "#e3eaee", exponentformat: "power", zeroline: false},
      legend: {font: {size: 11}, bgcolor: "rgba(255,255,255,0.85)"}, paper_bgcolor: "#fff", plot_bgcolor: "#fff", hovermode: "closest"};
    if (!window.Plotly) { summary.textContent = "Plotting library unavailable."; return; }
    await window.Plotly.react(plot, traces, layout, {responsive: true, displaylogo: false});
    plot.removeAllListeners?.("plotly_click");
    plot.on("plotly_click", event => { const id = event.points?.[0]?.customdata; if (id) select(id, true); });
    const shown = data.points.filter(p => visible.has(p.id));
    const best = [...shown].sort((a, b) => a.hvl_mm - b.hvl_mm)[0];
    summary.textContent = `${shown.length} materials at ${energy} keV (1 mm, listed densities). ` +
      (gases.length ? `${gases.length} gases are omitted from this plot because their densities are about 1000 times lower; they remain in the list. ` : "") +
      (best ? `Shortest half-value layer: ${best.name}, ${fmt(best.hvl_mm)} mm${best.density_status === "sourced" ? "" : " (unverified density)"}.` : "");
  }

  function setView(view) {
    ui.view = view;
    for (const [name, tab] of [["material", "#ml-tab-material"], ["landscape", "#ml-tab-landscape"]]) {
      const active = name === view;
      $(tab).setAttribute("aria-selected", String(active));
      $(tab).tabIndex = active ? 0 : -1;
      $(`#ml-view-${name}`).hidden = !active;
    }
    if (view === "landscape") renderLandscape();
  }

  function select(id, openMaterialView = false) {
    ui.selected = id;
    renderList();
    if (openMaterialView) setView("material");
    renderMaterial();
  }

  function bind(dialog) {
    $("#ml-close").addEventListener("click", () => dialog.close());
    // Escape is handled explicitly rather than left to the browser's close
    // watcher, which can swallow the key; focus then returns to the opener.
    dialog.addEventListener("keydown", event => {
      if (event.key === "Escape" && dialog.open) { event.preventDefault(); dialog.close(); }
    });
    dialog.addEventListener("close", () => $("#open-materials-library")?.focus());
    dialog.addEventListener("click", event => { if (event.target === dialog) dialog.close(); });
    $("#ml-search").addEventListener("input", event => { ui.query = event.target.value; renderList(); if (ui.view === "landscape") renderLandscape(); });
    $("#ml-family").addEventListener("change", event => { ui.family = event.target.value; renderList(); if (ui.view === "landscape") renderLandscape(); });
    $("#ml-sourced").addEventListener("change", event => { ui.sourcedOnly = event.target.checked; renderList(); if (ui.view === "landscape") renderLandscape(); });
    $("#ml-list").addEventListener("click", event => { const b = event.target.closest(".ml-item"); if (b) select(b.dataset.id); });
    $("#ml-list").addEventListener("keydown", event => {
      if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
      const buttons = [...dialog.querySelectorAll(".ml-item")];
      const i = buttons.indexOf(document.activeElement);
      if (i < 0) return;
      event.preventDefault();
      const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : Math.max(0, Math.min(buttons.length - 1, i + (event.key === "ArrowDown" ? 1 : -1)));
      buttons[next].focus();
    });
    $("#ml-tab-material").addEventListener("click", () => setView("material"));
    $("#ml-tab-landscape").addEventListener("click", () => setView("landscape"));
    dialog.querySelector(".ml-tabs").addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const next = ui.view === "material" ? "landscape" : "material";
      setView(next); $(`#ml-tab-${next}`).focus();
    });
    let energyTimer;
    $("#ml-energy").addEventListener("input", event => {
      clearTimeout(energyTimer);
      energyTimer = setTimeout(() => { ui.landscapeEnergy = Number(event.target.value); renderLandscape(); }, 350);
    });
    $("#ml-axis").addEventListener("change", event => { ui.landscapeAxis = event.target.value; renderLandscape(); });
  }

  async function open() {
    const dialog = $("#materials-dialog") || (() => { const d = markup(); bind(d); return d; })();
    dialog.showModal();
    $("#ml-search").focus();
    if (!ui.items.length) {
      try {
        const data = await getJSON("/api/materials");
        const rank = id => FAMILIES.indexOf(familyFor(id));
        ui.items = data.materials.sort((a, b) => rank(a.category) - rank(b.category) || a.name.localeCompare(b.name));
        const sourced = ui.items.filter(m => m.density_status === "sourced").length;
        $("#ml-subtitle").textContent = `${ui.items.length} entries · ${sourced} with sourced densities · every number carries its source`;
      } catch (exc) { $("#ml-subtitle").textContent = `Could not load the library: ${exc.message}`; }
    }
    renderList();
    renderMaterial();
  }

  function install() {
    const row = $(".add-layer-row");
    if (!row || $("#open-materials-library")) return;
    const button = document.createElement("button");
    button.type = "button";
    button.id = "open-materials-library";
    button.className = "button outline small ml-open";
    button.innerHTML = `<span aria-hidden="true">▦</span> Materials library`;
    button.setAttribute("aria-haspopup", "dialog");
    button.addEventListener("click", open);
    row.after(button);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install);
  else install();
})();
