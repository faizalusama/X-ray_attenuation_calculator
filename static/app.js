"use strict";

(() => {
  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
  const esc = value => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]));
  const clone = object => JSON.parse(JSON.stringify(object));
  const limits = Object.freeze({layers: 24, components: 40, spectrumBins: 10000, points: 4000});
  const colors = ["#218a74", "#7095b7", "#c29c62", "#9478ae", "#b57877", "#729285", "#869bc1", "#aea078"];
  const initial = {
    energy: {min_keV: 5, max_keV: 120, points: 500, spacing: "log", reference_keV: 30},
    layers: [{name: "Borosilicate glass", basis: "mole", components: [{formula: "SiO2", fraction: 80}, {formula: "B2O3", fraction: 15}, {formula: "Na2O", fraction: 5}], density_g_cm3: 2.23, thickness_mm: 1, angle_deg: 0, porosity: 0, density_mode: "bulk", density_uncertainty_pct: 0, thickness_uncertainty_pct: 0}],
    target_transmission: 0.1,
    spectrum: null,
    uncertainty: {enabled: false, samples: 1000, seed: 42}
  };
  const state = {configuration: clone(initial), resultConfiguration: null, presets: [], openLayer: 0, result: null, revision: 0, activeRequest: 0, busy: false, tab: "overview", mode: "transmission", channelLayer: 0, hiddenSeries: new Set(), chart: null, stale: false};
  let toastTimer;
  let live, dashboard;
  state.lengthUnit = "mm";
  state.savedConfiguration = clone(initial);
  const displayUnits = typeof AttenuationDashboard !== "undefined" ? AttenuationDashboard : (typeof require === "function" ? require("./dashboard.js") : null);
  const displayLength = value => displayUnits.fromMM(value, state.lengthUnit);
  const canonicalLength = value => displayUnits.toMM(value, state.lengthUnit);
  const STORAGE_KEY = "attenuation-lab-workspace-v1";
  function remember() {
    if (!dashboard) return;
    try {
      const saved = {schema_version: 1, configuration: state.resultConfiguration || state.savedConfiguration, ui: {...dashboard.preferences(), tab: state.tab, lengthUnit: state.lengthUnit, scopeOpen: $("#model-scope").open, suggestionsOpen: !$("#suggestions-section").hidden}};
      localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
      $("#remember-status").textContent = "Settings and last valid inputs saved on this browser.";
    } catch { $("#remember-status").textContent = "Browser storage unavailable; use Save project to keep your settings."; }
  }
  function syncLengthButtons() {
    $("#length-units").innerHTML = Object.keys(displayUnits.LENGTH_UNITS).map(unit => `<button type="button" data-length-unit="${unit}" aria-pressed="${state.lengthUnit === unit}">${unit}</button>`).join("");
  }
  function renderSuggestions() {
    const result = state.result, config = state.resultConfiguration;
    const suggestions = [];
    if (!result) { $("#smart-suggestions").innerHTML = '<p class="help-text">Define your material to receive contextual suggestions.</p>'; return; }
    if (config.layers.some(l => l.density_mode === "ideal")) suggestions.push(["Use measured density", "Ideal mixing can miss nonadditive volume in glasses and ceramics. A measured bulk density improves this input.", "Review material", "inputs"]);
    else if (config.layers.some(l => l.thickness_mm > 0 && l.thickness_mm < .1) && state.lengthUnit !== "µm") suggestions.push(["Work in micrometres", "Your stack includes a thin layer. Switch the display to µm for easier thickness editing.", "Use µm", "micrometres"]);
    else suggestions.push(["Explore thickness and energy", "See where your current stack transmits or removes the primary beam as all layer thicknesses scale together.", "Open thickness map", "map"]);
    if (config.energy.max_keV > 250) suggestions.push(["Cross-check the high-energy range", "The sweep exceeds 250 keV. Validate relevant energies against an independent database before drawing quantitative conclusions.", "Read model limits", "methods"]);
    else if (result.reference.transmission < .01) suggestions.push(["Resolve small transmission", "Less than 1% of the primary beam remains at the reference energy. Optical depth is easier to inspect when transmission approaches zero.", "Inspect optical depth", "tau"]);
    else suggestions.push(["Design to your target", `Inspect the total stack thickness needed for ${(100 * config.target_transmission).toPrecision(3)}% primary transmission across the energy sweep.`, "Open target thickness", "design"]);
    if (!result.uncertainty) suggestions.push(["Quantify input uncertainty", "Enter measured density and thickness uncertainties to view a reproducible 95% Monte Carlo interval.", "Set uncertainties", "uncertainty"]);
    else if (!result.spectrum) suggestions.push(["Include your beam spectrum", "A measured photon-fluence spectrum lets you inspect spectral transmission and beam hardening.", "Add measured spectrum", "spectrum"]);
    else suggestions.push(["Check the model boundary", "This model predicts primary transmission. Removed photons are not a dose or absorbed-energy estimate.", "Review assumptions", "methods"]);
    $("#smart-suggestions").innerHTML = suggestions.slice(0, 3).map(([title, text, action, target], i) => `<article><span class="suggestion-number">0${i + 1}</span><div><h3>${esc(title)}</h3><p>${esc(text)}</p><button type="button" data-suggestion="${target}">${esc(action)} ↗</button></div></article>`).join("");
  }

  // Coalesce rapid edits and keep at most one scientific calculation in flight.
  // A new edit supersedes a queued one; the revision check below rejects old results.
  function createLiveScheduler(run, delay = 300, timers = {set: (callback, ms) => setTimeout(callback, ms), clear: id => clearTimeout(id)}, onError = exception => console.error(exception)) {
    let timer = null, running = false, due = false, automatic = true;
    async function drain() {
      if (running || !due) return;
      due = false;
      running = true;
      const options = {automatic};
      try { await run(options); } catch (exception) { onError(exception); }
      finally { running = false; if (due) void drain(); }
    }
    return {
      schedule() {
        if (timer !== null) timers.clear(timer);
        due = false;
        automatic = true;
        timer = timers.set(() => {timer = null; due = true; void drain();}, delay);
      },
      flush() {
        if (timer !== null) timers.clear(timer);
        timer = null;
        automatic = false;
        due = true;
        return drain();
      }
    };
  }

  function number(value, significant = 4) {
    if (value == null || !Number.isFinite(Number(value))) return "—";
    const n = Number(value);
    if (n === 0) return "0";
    if (Math.abs(n) >= 1e5 || Math.abs(n) < 0.001) return n.toExponential(significant - 1);
    return Number(n.toPrecision(significant)).toLocaleString("en-US", {maximumFractionDigits: 8});
  }
  function percent(value) { return value == null ? "—" : number(value * 100, 4); }
  function numeric(input) { return input.value === "" ? null : Number(input.value); }
  function toast(message) {
    clearTimeout(toastTimer);
    $("#toast").textContent = message;
    $("#toast").hidden = false;
    toastTimer = setTimeout(() => { $("#toast").hidden = true; }, 3300);
  }
  function error(message) {
    $("#notification").textContent = message;
    $("#notification").hidden = false;
  }
  function setStatus(message, kind = "") {
    $("#result-status").className = `result-status ${kind}`;
    $("#result-status").innerHTML = `<i></i>${esc(message)}`;
  }
  function dirty() {
    state.revision++;
    state.stale = true;
    $("#notification").hidden = true;
    setStatus("Updating automatically…", "pending");
    markPending("Updating automatically…");
    syncSliders();
    live.schedule();
  }
  function markPending(message) {
    $("#live-update-banner").hidden = false;
    $("#live-update-banner").textContent = `${message}${state.result ? " Showing the previous calculation until the new inputs are valid and evaluated." : " Complete the inputs to see the live result."}`;
    ["#export-csv", "#export-json", "#export-svg", "#quick-export-svg"].forEach(id => { $(id).disabled = true; });
    $("#results-workspace").classList.toggle("results-pending", !!state.result);
  }
  function syncSliders() {
    $$('[data-slider-field]').forEach(slider => {
      let value = state.configuration.layers[Number(slider.dataset.layer)]?.[slider.dataset.sliderField];
      if (slider.dataset.sliderField === "thickness_mm") value = displayLength(value);
      if (!Number.isFinite(value)) return;
      if (slider.dataset.sliderField === "thickness_mm") slider.max = Math.max(displayLength(10), Number(slider.max), value > Number(slider.max) ? value * 1.2 : Number(slider.max));
      slider.value = value;
      const box = slider.closest(".slider-box");
      const output = box?.querySelector("[data-slider-output]");
      const maximum = box?.querySelector("[data-slider-maximum]");
      if (output) output.textContent = slider.dataset.sliderField === "thickness_mm" ? `${number(value)} ${state.lengthUnit}` : `${number(value)}°`;
      if (maximum) maximum.textContent = slider.dataset.sliderField === "thickness_mm" ? `${number(Number(slider.max))} ${state.lengthUnit}` : `${number(Number(slider.max))}°`;
    });
    const energy = state.configuration.energy;
    if (energy.min_keV > 0 && energy.max_keV > energy.min_keV) {
      $("#reference-slider").min = energy.min_keV;
      $("#reference-slider").max = energy.max_keV;
    }
    if (Number.isFinite(energy.reference_keV)) $("#reference-slider").value = energy.reference_keV;
  }
  function geometrySlider(fieldName, value, index, label, maximum) {
    const thickness = fieldName === "thickness_mm";
    const minimum = thickness ? 0 : -89.8;
    const unit = thickness ? state.lengthUnit : "°";
    return `<label class="slider-box ${thickness ? "thickness-slider" : "angle-slider"}"><span class="slider-heading"><span>${label}</span><output data-slider-output>${number(value)} ${unit}</output></span><span class="slider-track-shell"><input type="range" data-layer="${index}" data-slider-field="${fieldName}" aria-label="${label} slider" min="${minimum}" max="${maximum}" value="${esc(value)}" step="any"></span><span class="slider-scale"><span>${number(minimum)} ${unit}</span><span data-slider-maximum>${number(maximum)} ${unit}</span></span></label>`;
  }
  function field(label, name, value, index, extra = "", unit = "") {
    return `<label>${label}${unit ? '<div class="input-unit">' : ""}<input type="number" data-layer="${index}" data-field="${name}" value="${esc(value)}" step="any" ${extra}>${unit ? `<span>${unit}</span></div>` : ""}</label>`;
  }
  function renderLayers() {
    const layers = state.configuration.layers;
    $("#layer-count").textContent = `${layers.length} LAYER${layers.length === 1 ? "" : "S"}`;
    $("#layer-list").innerHTML = layers.map((layer, index) => {
      const isOpen = state.openLayer === index;
      const ideal = layer.density_mode === "ideal";
      return `<article class="layer-card ${isOpen ? "open" : ""}">
        <div class="layer-header"><span class="layer-number">${String(index + 1).padStart(2, "0")}</span><button type="button" class="layer-toggle" data-action="toggle" data-index="${index}" aria-expanded="${isOpen}" aria-controls="layer-body-${index}"><span>${esc(layer.name)}</span><span class="chevron">⌄</span></button><div class="layer-tools"><button type="button" class="mini-button" data-action="up" data-index="${index}" title="Move layer up" aria-label="Move ${esc(layer.name)} up" ${index === 0 ? "disabled" : ""}>↑</button><button type="button" class="mini-button" data-action="down" data-index="${index}" title="Move layer down" aria-label="Move ${esc(layer.name)} down" ${index === layers.length - 1 ? "disabled" : ""}>↓</button><button type="button" class="mini-button danger" data-action="remove" data-index="${index}" title="Remove layer" aria-label="Remove ${esc(layer.name)}" ${layers.length === 1 ? "disabled" : ""}>×</button></div></div>
        <div class="layer-body" id="layer-body-${index}" ${isOpen ? "" : "hidden"}>
          <label>Material name<input type="text" value="${esc(layer.name)}" data-layer="${index}" data-field="name" required maxlength="200"></label>
          <div class="composition-heading"><span>Composition</span><select data-layer="${index}" data-field="basis" aria-label="Composition fraction basis"><option value="mass" ${layer.basis === "mass" ? "selected" : ""}>Mass parts</option><option value="mole" ${layer.basis === "mole" ? "selected" : ""}>Mole parts</option><option value="volume" ${layer.basis === "volume" ? "selected" : ""}>Volume parts</option></select></div>
          <div class="composition-grid composition-labels"><span>FORMULA</span><span>PARTS</span><span>ρ · g/cm³</span><span></span></div>
          ${layer.components.map((component, ci) => `<div class="composition-grid component-row"><input type="text" value="${esc(component.formula)}" data-layer="${index}" data-component="${ci}" data-component-field="formula" required aria-label="Component ${ci + 1} chemical formula" placeholder="SiO2"><input type="number" value="${esc(component.fraction)}" data-layer="${index}" data-component="${ci}" data-component-field="fraction" min="0" step="any" required aria-label="Component ${ci + 1} fraction parts"><input type="number" value="${esc(component.density_g_cm3 ?? "")}" data-layer="${index}" data-component="${ci}" data-component-field="density_g_cm3" min="0.000000000001" max="10000" step="any" ${ideal || layer.basis === "volume" ? "required" : ""} aria-label="Component ${ci + 1} density in grams per cubic centimetre" placeholder="${ideal || layer.basis === "volume" ? "req." : "opt."}"><button type="button" class="mini-button danger" data-action="remove-component" data-index="${index}" data-component="${ci}" aria-label="Remove component ${ci + 1}" ${layer.components.length === 1 ? "disabled" : ""}>×</button></div>`).join("")}
          <button type="button" class="add-component" data-action="add-component" data-index="${index}">+ Add component</button>
          <div class="density-help">Relative parts are normalized automatically.${layer.basis === "volume" ? " Constituent densities are required." : " Chemical formula units define mole parts."}</div>
          <div class="field-grid"><label>Density model<select data-layer="${index}" data-field="density_mode"><option value="bulk" ${layer.density_mode === "bulk" ? "selected" : ""}>Measured bulk</option><option value="solid" ${layer.density_mode === "solid" ? "selected" : ""}>Solid + porosity</option><option value="ideal" ${ideal ? "selected" : ""}>Ideal mixture</option></select></label>${field(ideal ? "Density (computed)" : layer.density_mode === "solid" ? "Solid density · g/cm³" : "Bulk density · g/cm³", "density_g_cm3", ideal ? "" : layer.density_g_cm3, index, `${ideal ? 'placeholder="Computed automatically"' : ""} min="0.000000000001" max="10000" ${ideal ? "disabled" : "required"}`)}</div>
          ${layer.density_mode !== "bulk" ? `<div class="field-grid">${field("Porosity · fraction", "porosity", layer.porosity, index, 'min="0" max="0.999999" required')}<p class="density-help">${ideal ? "Additive constituent volume; measured densities are preferred." : "Effective density = solid density × (1 − porosity)."}</p></div>` : '<div class="density-help">Measured bulk density already accounts for pores.</div>'}
          <div class="field-grid layer-bottom-fields">${field("Normal thickness", "thickness_mm", displayLength(layer.thickness_mm), index, `min="0" max="${displayLength(1e9)}" required`, state.lengthUnit)}${field("Angle from normal", "angle_deg", layer.angle_deg, index, 'min="-89.9" max="89.9" required', "°")}</div>
          <div class="geometry-sliders">${geometrySlider("thickness_mm", displayLength(layer.thickness_mm), index, `Explore thickness · ${state.lengthUnit}`, displayLength(Math.max(10, layer.thickness_mm * 1.2)))}${geometrySlider("angle_deg", layer.angle_deg, index, "Explore angle", 89.8)}</div>
        </div></article>`;
    }).join("");
    renderUncertaintyInputs();
  }
  function renderUncertaintyInputs() {
    $("#uncertainty-inputs").innerHTML = state.configuration.layers.map((layer, index) => `<div class="uncertainty-layer"><span>${String(index + 1).padStart(2, "0")} · ${esc(layer.name)}</span>${field("Density σ · %", "density_uncertainty_pct", layer.density_uncertainty_pct ?? 0, index, 'min="0" max="100" required')}${field("Thickness σ · %", "thickness_uncertainty_pct", layer.thickness_uncertainty_pct ?? 0, index, 'min="0" max="100" required')}</div>`).join("");
  }
  function syncControls() {
    $$('[data-energy]').forEach(input => { input.value = state.configuration.energy[input.dataset.energy]; });
    $("#target-transmission").value = state.configuration.target_transmission;
    $("#uncertainty-enabled").checked = state.configuration.uncertainty.enabled;
    $("#uncertainty-samples").value = state.configuration.uncertainty.samples;
    $("#uncertainty-seed").value = state.configuration.uncertainty.seed;
    const spectrum = state.configuration.spectrum;
    $("#spectrum-enabled").checked = !!spectrum;
    $("#spectrum-label").value = spectrum?.label || "Measured spectrum";
    $("#spectrum-weighting").value = spectrum?.weighting || "photon";
    $("#spectrum-data").value = spectrum ? "energy_keV,weight\n" + spectrum.energy_keV.map((energy, index) => `${energy},${spectrum.weights[index]}`).join("\n") : "";
    syncLengthButtons();
    renderLayers();
    syncSliders();
  }

  function parseSpectrumText(source, weighting = "photon", label = "Measured spectrum") {
    const text = source.trim().replace(/^\uFEFF/, "");
    if (!text) throw new Error("Add spectrum energies and photon fluence weights, or disable spectrum input.");
    const rows = text.split(/\r?\n/).map(line => line.trim()).filter(line => line && !line.startsWith("#"));
    const energy_keV = [], weights = [];
    rows.forEach((row, index) => {
      const cells = row.split(/[,;\t]/.test(row) ? /[,;\t]/ : /\s+/).map(cell => cell.trim().replace(/^"(.*)"$/, "$1"));
      if (index === 0 && cells.length === 2 && /^(energy_keV|energy)$/i.test(cells[0]) && /^(weight|weights|photon_fluence)$/i.test(cells[1])) return;
      if (cells.length !== 2 || !cells.every(cell => cell !== "" && Number.isFinite(Number(cell)))) throw new Error(`Spectrum row ${index + 1}: expected exactly two finite numbers (energy_keV, photon fluence).`);
      const [energy, weight] = cells.map(Number);
      if (energy < 1 || energy > 800) throw new Error(`Spectrum row ${index + 1}: energy must be within 1–800 keV.`);
      if (weight < 0) throw new Error(`Spectrum row ${index + 1}: photon fluence cannot be negative.`);
      energy_keV.push(energy); weights.push(weight);
    });
    if (energy_keV.length > limits.spectrumBins) throw new Error(`A spectrum supports up to ${limits.spectrumBins.toLocaleString("en-US")} discrete bins.`);
    if (!energy_keV.length || !weights.some(weight => weight > 0)) throw new Error("The spectrum must contain at least one energy bin and positive total photon fluence.");
    if (!["photon", "energy"].includes(weighting)) throw new Error("Unsupported spectrum weighting.");
    return {energy_keV, weights, weighting, label: label.trim() || "Measured spectrum"};
  }
  function parseSpectrum() {
    return $("#spectrum-enabled").checked ? parseSpectrumText($("#spectrum-data").value, $("#spectrum-weighting").value, $("#spectrum-label").value) : null;
  }
  function payload() {
    const config = clone(state.configuration);
    config.spectrum = parseSpectrum();
    config.uncertainty = {enabled: $("#uncertainty-enabled").checked, samples: numeric($("#uncertainty-samples")), seed: numeric($("#uncertainty-seed"))};
    if (!(config.energy.min_keV < config.energy.max_keV)) throw new Error("The minimum energy must be below the maximum energy.");
    if (!config.layers.length) throw new Error("Add at least one material layer.");
    return cleanProject({schema_version: 1, configuration: config});
  }
  function validateForms(automatic = false) {
    const inputs = [...$$("#config-form input"), ...$$("#config-form select")];
    if ($("#uncertainty-enabled").checked) inputs.push(...$$("#view-uncertainty input"));
    for (const input of inputs) {
      if (input.disabled || input.checkValidity()) continue;
      if (automatic) { setStatus("Waiting for valid input", "pending"); markPending("Input incomplete."); return false; }
      const card = input.closest(".layer-card");
      if (card) {
        card.classList.add("open");
        $(".layer-body", card).hidden = false;
        $(".layer-toggle", card).setAttribute("aria-expanded", "true");
      }
      if (input.closest("#view-uncertainty")) activateTab("uncertainty");
      input.reportValidity();
      input.focus();
      return false;
    }
    return true;
  }
  async function calculate({automatic = false} = {}) {
    if (!validateForms(automatic)) return;
    let config;
    try { config = payload(); } catch (exception) { setStatus("Check input", "pending"); markPending("Input incomplete."); error(exception.message); return; }
    const revision = state.revision;
    const requestId = ++state.activeRequest;
    state.busy = true;
    $("#run-button").disabled = true;
    $("#run-button span").textContent = "Calculating…";
    $("#notification").hidden = true;
    setStatus("Calculating…", "loading");
    try {
      const response = await fetch("/api/calculate", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(config)});
      let result;
      try { result = await response.json(); } catch { throw new Error("The calculation server returned an unreadable response. Verify the local server is running, then try again."); }
      if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : Array.isArray(result.detail) ? result.detail.map(item => item.msg || JSON.stringify(item)).join("; ") : "The calculation could not be completed.");
      if (requestId !== state.activeRequest) return;
      if (revision !== state.revision) return;
      state.result = result;
      state.resultConfiguration = clone(config);
      state.savedConfiguration = clone(config);
      state.configuration = config;
      state.stale = false;
      $("#live-update-banner").hidden = true;
      $("#results-workspace").classList.remove("results-pending");
      setStatus("Live · up to date");
      renderResults();
      remember();
    } catch (exception) {
      if (requestId === state.activeRequest && revision === state.revision) {
        state.stale = true;
        markPending("Unable to update the result.");
        setStatus("Check configuration", "error");
        error(exception.message === "Failed to fetch" ? "Cannot reach the local calculation server. Start the workbench server and refresh this page." : exception.message);
      }
    } finally {
      if (requestId === state.activeRequest) {
        state.busy = false;
        $("#run-button").disabled = false;
        $("#run-button span").textContent = "Refresh now";
      }
    }
  }

  function metric(label, value, unit, note, icon = "") {
    return `<article class="metric"><div class="metric-label">${label}<span class="metric-icon" aria-hidden="true">${icon}</span></div><div class="metric-value">${value}<span class="metric-unit">${unit}</span></div><div class="metric-note">${esc(note)}</div></article>`;
  }
  function renderResults() {
    const result = state.result;
    const ref = result?.reference;
    const first = result?.layers?.[0];
    const energyNote = result ? `At ${number(ref.energy_keV)} keV` : "Updating automatically";
    $("#metrics").innerHTML = metric("Reference transmission", ref ? percent(ref.transmission) : "—", "%", energyNote + (ref ? " · primary beam" : ""), "↗") + metric("Optical depth", ref ? number(ref.optical_depth) : "—", "", energyNote, "τ") + metric("Half-value layer", first ? number(displayLength(first.reference.hvl_mm)) : "—", state.lengthUnit, first ? `${first.name} · path length` : "First material · path length", "½") + metric("Attenuation length", first ? number(displayLength(first.reference.attenuation_length_mm)) : "—", state.lengthUnit, first ? `${first.name} · path length` : "First material · path length", "ℓ");
    ["#export-csv", "#export-json", "#export-svg", "#quick-export-svg"].forEach(id => { $(id).disabled = !result || state.stale; });
    $("#warnings").hidden = !result?.warnings?.length;
    $("#warnings").innerHTML = result?.warnings?.length ? `<strong>Model notes</strong><ul>${result.warnings.map(warning => `<li>${esc(warning)}</li>`).join("")}</ul>` : "";
    $("#plot-context").textContent = result ? `${result.energy_keV.length.toLocaleString()} energies · ${state.resultConfiguration.energy.spacing === "log" ? "Logarithmic" : "Linear"} x-axis · ${result.edges?.length || 0} reported edges` : "Direct evaluation at every energy";
    $("#provenance-short").textContent = result ? `${result.provenance.backend?.name || result.provenance.engine} ${result.provenance.backend?.dataset_version || result.provenance.xraydb_version} · ${result.provenance.model}` : "Scientific model and data sources are documented under Methods.";
    $("#provenance-details").innerHTML = result ? `<div class="provenance-block"><strong>Current calculation provenance</strong><br>Source: ${esc(result.provenance.backend?.name || result.provenance.engine)} ${esc(result.provenance.backend?.dataset_version || result.provenance.xraydb_version)}<br>Dataset: ${esc(result.provenance.backend?.dataset || "Elam atomic data")}<br>Computed: ${esc(result.provenance.timestamp_utc)}${result.provenance.workbench_version ? `<br>Workbench: ${esc(result.provenance.workbench_version)}` : ""}${result.provenance.configuration_sha256 ? `<br>Configuration SHA-256: <span style="overflow-wrap:anywhere">${esc(result.provenance.configuration_sha256)}</span>` : ""}<br>Complete configuration and provenance are included in the results JSON export.</div>` : "";
    renderChart();
    renderStack();
    renderMaterials();
    renderSpectrum();
    renderUncertainty();
    renderSuggestions();
  }
  function renderStack() {
    const layers = state.resultConfiguration?.layers || state.configuration.layers;
    $("#stack-visual").innerHTML = `<div class="stack-diagram"><span class="beam-arrow" title="Incident beam">→</span>${layers.map((layer, index) => `<div class="stack-block" title="${esc(layer.name)} · ${number(displayLength(layer.thickness_mm))} ${state.lengthUnit} normal thickness"><strong>${String(index + 1).padStart(2, "0")}</strong><span>${number(displayLength(layer.thickness_mm))} ${state.lengthUnit}</span></div>`).join("")}<span class="beam-arrow final" title="Transmitted primary beam">→</span></div><div class="stack-items">${layers.map((layer, index) => `<span><i style="background:${colors[index % colors.length]}"></i>${esc(layer.name)}</span>`).join("")}</div><div class="stack-footnote">Schematic · Not to scale · Layer order does not change primary transmission</div>`;
    const ref = state.result?.reference;
    $("#target-insight").innerHTML = ref ? `<div class="target-number">${number(ref.target_thickness_scale)}<span>× thickness</span></div><p class="target-copy">Scale every layer to reach <strong>${percent(ref.target_transmission)}% transmission</strong> at ${number(ref.energy_keV)} keV.</p><div class="target-divider"></div><p class="target-note">${ref.target_thickness_scale == null ? "A thickness scaling solution is unavailable for this configuration." : `Total normal thickness: ${number(displayLength(layers.reduce((sum, layer) => sum + layer.thickness_mm, 0) * ref.target_thickness_scale))} ${state.lengthUnit}. Composition, density, and angle remain fixed.`}</p>` : '<div class="target-number">—<span>× thickness</span></div><p class="target-copy">Calculate to find the stack thickness required for your target transmission.</p><div class="target-divider"></div><p class="target-note">Scales all layer thicknesses proportionally at the reference energy.</p>';
  }
  function detailMetric(label, value, unit = "") {
    return `<div class="detail-metric"><span>${label}</span><strong>${number(value)}<small>${unit}</small></strong></div>`;
  }
  function empty(message) { return `<div class="empty-state">${esc(message)}</div>`; }
  function renderMaterials() {
    if (!state.result) { $("#materials-results").innerHTML = empty(state.stale ? "Your configuration has changed. Calculate to refresh material properties." : "Calculate your system to inspect material properties."); return; }
    $("#materials-results").innerHTML = state.result.layers.map((layer, index) => `<article class="panel material-result"><h3><span class="section-number">${String(index + 1).padStart(2, "0")}</span> ${esc(layer.name)}</h3><div class="material-meta"><span>Effective density <strong>${number(layer.density_g_cm3)} g/cm³</strong></span><span>Normal thickness <strong>${number(displayLength(layer.thickness_mm))} ${state.lengthUnit}</strong></span><span>Ray path <strong>${number(displayLength(layer.path_length_mm))} ${state.lengthUnit}</strong></span></div><div class="eyebrow subtle">REFERENCE · ${number(state.result.reference.energy_keV)} keV</div><div class="detail-metrics">${detailMetric("Mass attenuation", layer.reference.mu_mass_cm2_g, "cm²/g")}${detailMetric("Linear attenuation", layer.reference.mu_linear_cm_inv, "cm⁻¹")}${detailMetric("Layer transmission", layer.reference.transmission * 100, "%")}${detailMetric("Half-value layer", displayLength(layer.reference.hvl_mm), state.lengthUnit)}${detailMetric("Tenth-value layer", displayLength(layer.reference.tvl_mm), state.lengthUnit)}${detailMetric("Attenuation length", displayLength(layer.reference.attenuation_length_mm), state.lengthUnit)}</div><p class="help-text">Attenuation lengths are path lengths in this material, evaluated directly at the reference energy.</p><div class="eyebrow subtle">ELEMENTAL MASS FRACTIONS</div><div class="composition-breakdown">${Object.entries(layer.elemental_mass_fractions).sort((a,b) => b[1] - a[1]).map(([element, fraction]) => `<span class="element-chip"><strong>${esc(element)}</strong>${percent(fraction)}%</span>`).join("")}</div><div class="table-wrap"><table class="component-table"><thead><tr><th>Component formula</th><th>Normalized mass fraction</th><th>Molar mass · g/mol</th></tr></thead><tbody>${layer.components.map(component => `<tr><td>${esc(component.formula)}</td><td>${percent(component.mass_fraction)}%</td><td>${number(component.molar_mass_g_mol, 6)}</td></tr>`).join("")}</tbody></table></div></article>`).join("");
  }
  function renderSpectrum() {
    const spectrum = state.result?.spectrum;
    if (!spectrum) { $("#spectrum-results").innerHTML = empty("Enable a spectrum and calculate to inspect weighted transmission and the change in mean energy."); return; }
    $("#spectrum-results").innerHTML = `<section class="panel result-panel"><div class="eyebrow subtle">SPECTRUM RESULT</div><h3 style="margin-top:8px">${esc(spectrum.label)}</h3><div class="detail-metrics">${detailMetric(`${spectrum.weighting === "energy" ? "Energy" : "Photon"}-weighted transmission`, spectrum.transmission * 100, "%")}${detailMetric("Incident mean energy", spectrum.mean_energy_in_keV, "keV")}${detailMetric("Transmitted mean energy", spectrum.mean_energy_out_keV, "keV")}</div><p class="help-text">${spectrum.energy_keV.length} discrete bins · Means are photon-fluence weighted. Incident and transmitted fluence share a common vertical scale.</p><div class="spectrum-mini-chart" id="spectrum-mini-chart"></div><div class="legend"><span class="legend-item"><span class="legend-swatch" style="--series-color:#b8ccc5"></span>Incident photon fluence</span><span class="legend-item"><span class="legend-swatch" style="--series-color:#218a74"></span>Transmitted photon fluence</span></div></section>`;
    const width = Math.max(280, Math.min(1100, ($("#spectrum-results").clientWidth || 826) - 46)), height = 180, left = 51, right = 22, top = 15, bottom = 35;
    const energies = spectrum.energy_keV;
    const min = Math.min(...energies), max = Math.max(...energies);
    const largest = Math.max(...spectrum.input_weights) || 1;
    const x = energy => left + (max === min ? .5 : (energy - min) / (max - min)) * (width - left - right);
    const y = weight => height - bottom - weight / largest * (height - top - bottom);
    $("#spectrum-mini-chart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Incident and transmitted discrete photon fluence"><line x1="${left}" y1="${height-bottom}" x2="${width-right}" y2="${height-bottom}" stroke="#dbe4e3"/>${energies.map((energy, index) => `<line x1="${x(energy)}" y1="${height-bottom}" x2="${x(energy)}" y2="${y(spectrum.input_weights[index])}" stroke="#c3d4ce" stroke-width="${Math.max(1, Math.min(7, 250 / energies.length))}"/><line x1="${x(energy)}" y1="${height-bottom}" x2="${x(energy)}" y2="${y(spectrum.transmitted_weights[index])}" stroke="#218a74" stroke-width="${Math.max(1, Math.min(4, 150 / energies.length))}"/>`).join("")}<text x="${left}" y="${height-15}" font-size="10" fill="#879e9f">${number(min)} keV</text><text x="${width-right}" y="${height-15}" text-anchor="end" font-size="10" fill="#879e9f">${number(max)} keV</text><text x="${left-8}" y="${top+4}" text-anchor="end" font-size="10" fill="#879e9f">${number(largest)}</text></svg>`;
  }
  function renderUncertainty() {
    const uncertainty = state.result?.uncertainty;
    if (!uncertainty) { $("#uncertainty-results").innerHTML = empty("Enable uncertainty sampling and calculate to see the 95% interval."); return; }
    const ref = uncertainty.reference;
    $("#uncertainty-results").innerHTML = `<section class="panel result-panel"><div class="eyebrow subtle">95% MONTE CARLO INTERVAL</div><h3 style="margin-top:8px">Reference transmission at ${number(state.result.reference.energy_keV)} keV</h3><div class="detail-metrics">${detailMetric("Mean transmission", ref.mean * 100, "%")}${detailMetric("Standard deviation", ref.std * 100, "percentage points")}${detailMetric("Realizations", uncertainty.samples)}</div><div class="interval-bar" role="img" aria-label="95 percent interval ${percent(ref.p025)} to ${percent(ref.p975)} percent on a zero to one hundred percent transmission scale"><span class="interval-range" style="left:${ref.p025*100}%;width:${(ref.p975-ref.p025)*100}%"></span><span class="interval-mean" style="left:${ref.mean*100}%" title="Mean transmission"></span></div><div class="interval-labels"><span>0% transmission</span><span>100% transmission</span></div><div class="interval-labels"><span>2.5th: ${percent(ref.p025)}%</span><span>97.5th: ${percent(ref.p975)}%</span></div><p class="help-text">Interval: <strong>${percent(ref.p025)}–${percent(ref.p975)}%</strong>. Seed ${esc(uncertainty.seed)}. ${esc(uncertainty.assumptions || "Independent lognormal density and thickness uncertainties.")} The shaded band is also available on the transmission plot.</p></section>`;
  }

  function seriesForMode() {
    const result = state.result;
    if (!result) return [];
    switch (state.mode) {
      case "removed": return [{id: "removed", label: "Removed from primary beam", color: "#b48d59", values: result.removed_fraction}];
      case "mass": return result.layers.map((layer, index) => ({id: `mass-${index}`, label: layer.name, color: colors[index % colors.length], values: layer.mu_mass_cm2_g}));
      case "linear": return result.layers.map((layer, index) => ({id: `linear-${index}`, label: layer.name, color: colors[index % colors.length], values: layer.mu_linear_cm_inv}));
      case "channels": {
        const layer = result.layers[state.channelLayer] || result.layers[0];
        return [{id: "photoelectric", label: "Photoelectric", color: "#218a74", values: layer.photoelectric_cm2_g}, {id: "coherent", label: "Coherent", color: "#7198bc", values: layer.coherent_cm2_g}, {id: "incoherent", label: "Incoherent", color: "#b49b68", values: layer.incoherent_cm2_g}];
      }
      default: return [{id: "stack", label: "Whole stack", color: colors[0], values: result.transmission}, ...(result.layers.length > 1 ? result.layers.map((layer, index) => ({id: `layer-${index}`, label: layer.name, color: colors[(index + 1) % colors.length], values: layer.mu_linear_cm_inv.map(value => Math.exp(-value * layer.path_length_mm / 10)), dashed: true})) : [])];
    }
  }
  function niceTicks(min, max, count = 5) {
    if (!(max > min)) return [min];
    const rough = (max-min)/count;
    const power = 10 ** Math.floor(Math.log10(rough));
    const part = rough/power;
    const step = (part <= 1 ? 1 : part <= 2 ? 2 : part <= 2.5 ? 2.5 : part <= 5 ? 5 : 10)*power;
    const ticks = [];
    for (let value = Math.ceil(min / step) * step; value <= max + step * 1e-8 && ticks.length < 30; value += step) ticks.push(Math.abs(value) < step * 1e-8 ? 0 : value);
    return ticks;
  }
  function logTicks(min, max, count = 8) {
    const ticks = [];
    const span = Math.log10(max) - Math.log10(min);
    for (let power = Math.floor(Math.log10(min)); power <= Math.ceil(Math.log10(max)); power++) {
      for (const multiplier of span <= 2 ? [1,2,5] : [1]) {
        const value = multiplier * 10 ** power;
        if (value >= min && value <= max) ticks.push(value);
      }
    }
    if (ticks.length < 2) return niceTicks(min,max,5);
    return ticks.filter((_,index) => index % Math.max(1,Math.ceil(ticks.length/count)) === 0);
  }
  function linePath(energies, values, x, y, logarithmic = false) {
    let connected = false;
    return values.map((value, index) => {
      if (!Number.isFinite(value) || (logarithmic && value <= 0)) { connected = false; return ""; }
      const command = connected ? "L" : "M";
      connected = true;
      return `${command}${x(energies[index]).toFixed(2)},${y(value).toFixed(2)}`;
    }).filter(Boolean).join(" ");
  }
  function intervalPath(energies, lower, upper, x, y, logarithmic = false) {
    const runs = []; let run = [];
    for (let i = 0; i < energies.length; i++) {
      if (!Number.isFinite(lower[i]) || !Number.isFinite(upper[i]) || (logarithmic && (lower[i] <= 0 || upper[i] <= 0))) { if (run.length) runs.push(run); run = []; }
      else run.push(i);
    }
    if (run.length) runs.push(run);
    return runs.map(indices => indices.map((i, j) => `${j ? "L" : "M"}${x(energies[i]).toFixed(2)},${y(upper[i]).toFixed(2)}`).join(" ") + " " + indices.slice().reverse().map(i => `L${x(energies[i]).toFixed(2)},${y(lower[i]).toFixed(2)}`).join(" ") + "Z").join(" ");
  }
  function renderChart() {
    if (dashboard) dashboard.update({result: state.result, configuration: state.resultConfiguration, lengthUnit: state.lengthUnit});
  }
  function activateTab(name) {
    state.tab = name;
    remember();
    $$("[data-tab]").forEach(button => { const active = button.dataset.tab === name; button.classList.toggle("active",active); button.setAttribute("aria-selected",String(active)); button.tabIndex = active ? 0 : -1; });
    $$('[role="tabpanel"]').forEach(panel => { panel.hidden = panel.id !== `view-${name}`; });
    if (name === "overview") renderChart();
    if (name === "spectrum") renderSpectrum();
  }
  function download(contents, filename, type) {
    // Native HTTP attachments also work in embedded browsers that block blob URLs.
    let frame = $("#download-target");
    if (!frame) {frame=document.createElement("iframe");frame.id="download-target";frame.name="download-target";frame.hidden=true;frame.title="File export";document.body.appendChild(frame);}
    const form=document.createElement("form");
    form.method="POST";form.action="/api/download";form.target="download-target";form.acceptCharset="UTF-8";form.hidden=true;
    for (const [name,value] of Object.entries({filename,content:contents})) {
      const input=document.createElement("textarea");input.name=name;input.value=value;form.appendChild(input);
    }
    document.body.appendChild(form);form.submit();setTimeout(()=>form.remove(),1000);
  }
  function csvCell(value) { const text = String(value ?? ""); return /[",\r\n]/.test(text) ? `"${text.replace(/"/g,'""')}"` : text; }
  function exportCSV() {
    if (!state.result || state.stale) return;
    const result = state.result;
    const headers = ["energy_keV","transmission","removed_fraction","optical_depth"];
    const arrays = [result.energy_keV,result.transmission,result.removed_fraction,result.optical_depth];
    result.layers.forEach((layer,index) => {
      ["mu_mass_cm2_g","mu_linear_cm_inv","photoelectric_cm2_g","coherent_cm2_g","incoherent_cm2_g"].forEach(key => {headers.push(`layer_${index+1}_${key}`);arrays.push(layer[key]);});
    });
    if (result.uncertainty) {headers.push("transmission_p025","transmission_p975"); arrays.push(result.uncertainty.transmission_p025,result.uncertainty.transmission_p975);}
    const provenanceKeys = ["engine", "xraydb_version", "workbench_version", "timestamp_utc", "configuration_sha256"];
    headers.push(...provenanceKeys.map(key => `provenance_${key}`));
    const provenance = provenanceKeys.map(key => csvCell(result.provenance[key]));
    for (const key of ["identifier", "dataset_version", "dataset"]) {
      headers.push(`provenance_backend_${key}`);
      provenance.push(csvCell(result.provenance.backend?.[key]));
    }
    const rows = [headers.map(csvCell).join(","), ...result.energy_keV.map((_, index) => [...arrays.map(array => csvCell(array[index])), ...provenance].join(","))];
    download(rows.join("\r\n")+"\r\n","attenuation-sweep.csv","text/csv;charset=utf-8");
    toast("Energy sweep exported with full numeric precision and provenance.");
  }
  function exportedSVG(chartMarkup, chart, provenance, configuration, mode, hasInterval) {
    const lines = (value, limit = Math.max(22, Math.floor((chart.width - 95) / 6))) => String(value).match(new RegExp(`.{1,${limit}}(?:\\s|$)|.{1,${limit}}`, "g"))?.map(line => line.trim()) || [""];
    const titleLines = lines(chart.title, Math.max(18, Math.floor((chart.width - 76) / 10)));
    const subtitle = lines(chart.subtitle), subtitleStart = 36 + titleLines.length * 22;
    const headingHeight = subtitleStart + subtitle.length * 16 + 10;
    const footer = []; let cursor = headingHeight + chart.height + 15;
    const svgText = (value, x, y, size = 11, color = "#465e6b", weight = "normal") => `<text x="${x}" y="${y}" font-size="${size}" fill="${color}" font-weight="${weight}">${esc(value)}</text>`;
    if (!chart.series.length) { footer.push(svgText("No data series selected", 38, cursor)); cursor += 20; }
    chart.series.forEach(series => {
      footer.push(`<line x1="38" y1="${cursor-4}" x2="61" y2="${cursor-4}" stroke="${series.color}" stroke-width="2.5" ${series.dashed ? 'stroke-dasharray="6 4"' : ""}/>`);
      lines(series.label).forEach(label => { footer.push(svgText(label, 70, cursor)); cursor += 16; });
      cursor += 4;
    });
    if (hasInterval) { footer.push(`<rect x="38" y="${cursor-9}" width="23" height="8" fill="#c4e0d3"/>` + svgText("Pointwise 95% Monte Carlo interval", 70, cursor)); cursor += 22; }
    const details = [
      `${chart.logX ? "Logarithmic" : "Linear"} energy axis · ${chart.logY ? "Logarithmic y axis; zero values omitted" : "Linear y axis"}`,
      `Reference energy: ${configuration.energy.reference_keV} keV · Narrow-beam primary attenuation`,
      `Source: ${provenance.engine} · XrayDB ${provenance.xraydb_version}${provenance.workbench_version ? ` · Workbench ${provenance.workbench_version}` : ""}`,
      `Model: ${provenance.model} · Computed: ${provenance.timestamp_utc}`,
      ...(provenance.configuration_sha256 ? [`Configuration SHA-256: ${provenance.configuration_sha256}`] : []),
      "Complete configuration and provenance are embedded in this SVG's metadata."
    ];
    cursor += 9;
    details.flatMap(detail => lines(detail)).forEach(detail => { footer.push(svgText(detail, 38, cursor, 10)); cursor += 15; });
    const metadata = {quantity: mode, configuration, provenance, visible_series: chart.series.map(item => item.label), x_axis: chart.logX ? "logarithmic" : "linear", y_axis: chart.logY ? "logarithmic" : "linear", zero_values_omitted: chart.logY};
    return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${chart.width}" height="${cursor+22}" viewBox="0 0 ${chart.width} ${cursor+22}" role="img" aria-labelledby="export-title" style="font-family:Segoe UI,Arial,sans-serif"><title id="export-title">${esc(chart.title)}</title><metadata>${esc(JSON.stringify(metadata))}</metadata><rect width="100%" height="100%" fill="#fff"/>${titleLines.map((line,index) => svgText(line,38,31+index*22,18,"#203c49","600")).join("")}${subtitle.map((line,index) => svgText(line,38,subtitleStart+index*16)).join("")}<g transform="translate(0 ${headingHeight})">${chartMarkup}</g>${footer.join("")}</svg>`;
  }
  async function exportSVG() {
    if (!state.result || state.stale || !dashboard) return;
    $("#export-svg").disabled = true; $("#quick-export-svg").disabled = true;
    try { await dashboard.exportSVG(); } catch (exception) { error(`Could not export figure: ${exception.message}`); }
    finally { $("#export-svg").disabled = !state.result || state.stale; $("#quick-export-svg").disabled = !state.result || state.stale; }
  }
  function cleanProject(documentValue) {
    const object = (value, label) => { if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object.`); return value; };
    const finite = (value, label, min = -Infinity, max = Infinity, integer = false) => {
      if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${label} must be a finite number.`);
      if (value < min || value > max || (integer && !Number.isInteger(value))) throw new Error(`${label} must be ${integer ? "an integer " : ""}between ${min} and ${max}.`);
      return value;
    };
    const text = (value, label, fallback = "") => {
      if (value == null) return fallback;
      if (typeof value !== "string" || value.length > 200) throw new Error(`${label} must be text with at most 200 characters.`);
      return value.trim() || fallback;
    };
    const optional = (value, fallback) => value === undefined ? fallback : value;
    if (documentValue?.schema_version !== 1) throw new Error("This is not a supported project. Expected schema_version 1 and a configuration object.");
    const c = object(documentValue.configuration, "Project configuration");
    const energy = object(c.energy, "Energy settings");
    const uncertainty = object(optional(c.uncertainty, {}), "Uncertainty settings");
    if (uncertainty.enabled !== undefined && typeof uncertainty.enabled !== "boolean") throw new Error("Uncertainty enabled must be true or false.");
    if (!Array.isArray(c.layers) || !c.layers.length || c.layers.length > limits.layers) throw new Error(`A project must contain 1–${limits.layers} layers.`);
    const config = {energy: {}, layers: [], target_transmission: finite(optional(c.target_transmission, .1), "Target transmission", 0, 1), spectrum: null,
      uncertainty: {enabled: optional(uncertainty.enabled, false), samples: finite(optional(uncertainty.samples, 1000), "Uncertainty samples", 100, 20000, true), seed: finite(optional(uncertainty.seed, 42), "Random seed", 0, 4294967295, true)}};
    // Preserve the v2 backend identity. Unknown identifiers are rejected by the
    // server registry; never silently substitute a different scientific source.
    if (c.backend != null) {
      if (typeof c.backend !== "string" || !/^[a-z0-9_-]{1,80}$/.test(c.backend)) throw new Error("Backend must be a valid source identifier.");
      config.backend = c.backend;
    }
    if (config.target_transmission <= 0 || config.target_transmission >= 1) throw new Error("Target transmission must be strictly between 0 and 1.");
    for (const key of ["min_keV", "max_keV", "reference_keV"]) config.energy[key] = finite(energy[key], `Energy ${key}`, 1, 800);
    config.energy.points = finite(energy.points, "Energy grid points", 2, limits.points, true);
    if (!(config.energy.min_keV < config.energy.max_keV)) throw new Error("The minimum energy must be below the maximum energy.");
    if (!["log", "linear"].includes(energy.spacing)) throw new Error("Energy spacing must be log or linear.");
    config.energy.spacing = energy.spacing;
    config.layers = c.layers.map((entry, index) => {
      const layer = object(entry, `Layer ${index + 1}`), name = text(layer.name, "Material name", `Material ${index + 1}`);
      const basis = optional(layer.basis, "mass"), mode = optional(layer.density_mode, "bulk");
      if (!["mass", "mole", "volume"].includes(basis) || !["bulk", "solid", "ideal"].includes(mode)) throw new Error(`Layer ${index + 1} has an unsupported fraction basis or density model.`);
      if (!Array.isArray(layer.components) || !layer.components.length || layer.components.length > limits.components) throw new Error(`Layer ${index + 1} must have 1–${limits.components} components.`);
      const next = {name, basis, density_mode: mode, components: layer.components.map((component, ci) => {
        object(component, `Layer ${index + 1}, component ${ci + 1}`);
        const formula = text(component.formula, "Chemical formula");
        if (!formula) throw new Error(`Layer ${index + 1}, component ${ci + 1} needs a chemical formula.`);
        const value = {formula, fraction: finite(component.fraction, "Component fraction", 0)};
        if (component.density_g_cm3 != null || basis === "volume" || mode === "ideal") value.density_g_cm3 = finite(component.density_g_cm3, "Constituent density", 1e-12, 1e4);
        return value;
      })};
      if (!next.components.some(component => component.fraction > 0)) throw new Error(`Layer ${index + 1} needs at least one positive component fraction.`);
      next.density_g_cm3 = finite(mode === "ideal" ? optional(layer.density_g_cm3, 1) : layer.density_g_cm3, "Material density", 1e-12, 1e4);
      next.thickness_mm = finite(optional(layer.thickness_mm, 1), "Normal thickness", 0, 1e9);
      next.angle_deg = finite(optional(layer.angle_deg, 0), "Angle from normal", -89.9, 89.9);
      if (Math.abs(next.angle_deg) >= 89.9) throw new Error("Angle must lie strictly between −89.9° and 89.9°.");
      next.porosity = finite(optional(layer.porosity, 0), "Porosity", 0, 1);
      if (next.porosity >= 1) throw new Error("Porosity must be less than 1.");
      if (mode === "bulk" && next.porosity !== 0) throw new Error("Measured bulk density already includes pores; set porosity to 0 or choose solid density.");
      for (const key of ["density_uncertainty_pct", "thickness_uncertainty_pct"]) next[key] = finite(optional(layer[key], 0), key, 0, 100);
      return next;
    });
    if (c.spectrum != null) {
      const spectrum = object(c.spectrum, "Spectrum");
      if (!Array.isArray(spectrum.energy_keV) || !Array.isArray(spectrum.weights) || !spectrum.energy_keV.length || spectrum.energy_keV.length !== spectrum.weights.length || spectrum.energy_keV.length > limits.spectrumBins) throw new Error("Spectrum arrays must have the same length (1–10,000 bins).");
      if (!["photon", "energy"].includes(spectrum.weighting)) throw new Error("Unsupported spectrum weighting.");
      config.spectrum = {energy_keV: spectrum.energy_keV.map(value => finite(value, "Spectrum energy", 1, 800)), weights: spectrum.weights.map(value => finite(value, "Spectrum weight", 0)), weighting: spectrum.weighting, label: text(spectrum.label, "Spectrum label", "Measured spectrum")};
      if (!config.spectrum.weights.some(value => value > 0)) throw new Error("Spectrum needs at least one positive photon fluence.");
    }
    return config;
  }

  // Pure data helpers are also exercised directly by Node regression tests.
  if (typeof module !== "undefined" && module.exports) module.exports = {cleanProject, parseSpectrumText, niceTicks, logTicks, csvCell, linePath, intervalPath, exportedSVG, createLiveScheduler, initial, limits};
  if (typeof document === "undefined") return;

  $("#config-form").noValidate=true;
  live = createLiveScheduler(calculate, 300, undefined, exception => {error(exception.message);setStatus("Unable to update", "error");});
  $("#config-form").addEventListener("submit",event=>{event.preventDefault();live.flush();});
  document.addEventListener("input",event=>{
    const input=event.target;
    if(input.dataset.sliderField){
      const index = Number(input.dataset.layer), fieldName = input.dataset.sliderField;
      state.configuration.layers[index][fieldName] = fieldName === "thickness_mm" ? canonicalLength(numeric(input)) : numeric(input);
      $(`input[data-layer="${index}"][data-field="${fieldName}"]`).value = input.value;
      dirty();
    }else if(input.id === "reference-slider"){
      state.configuration.energy.reference_keV = numeric(input);
      $("#energy-reference").value = input.value;
      dirty();
    }else if(input.dataset.componentField){
      const component=state.configuration.layers[Number(input.dataset.layer)].components[Number(input.dataset.component)];
      if(input.dataset.componentField==="formula") component.formula=input.value;
      else if(input.dataset.componentField==="density_g_cm3"&&input.value==="") delete component.density_g_cm3;
      else component[input.dataset.componentField]=numeric(input);
      dirty();
    }else if(input.dataset.field){
      const layer=state.configuration.layers[Number(input.dataset.layer)];
      layer[input.dataset.field]=input.dataset.field==="thickness_mm"?canonicalLength(numeric(input)):input.type==="number"?numeric(input):input.value;
      if(input.dataset.field==="name") {
        const toggle=$(`[data-action="toggle"][data-index="${input.dataset.layer}"]`);
        if(toggle) toggle.firstElementChild.textContent=input.value;
        renderUncertaintyInputs();
      }
      dirty();
    }else if(input.dataset.energy){state.configuration.energy[input.dataset.energy]=input.type==="number"?numeric(input):input.value;dirty();}
    else if(input.id==="target-transmission"){state.configuration.target_transmission=numeric(input);dirty();}
    else if(["spectrum-label","spectrum-data","spectrum-weighting","spectrum-enabled","uncertainty-enabled","uncertainty-samples","uncertainty-seed"].includes(input.id)){dirty();}
  });
  document.addEventListener("change",event=>{
    const input=event.target;
    if(input.dataset.field==="density_mode") {
      const layer=state.configuration.layers[Number(input.dataset.layer)];
      if(layer.density_mode==="bulk") layer.porosity=0;
      renderLayers();
    } else if(input.dataset.field==="basis") renderLayers();
  });
  $("#layer-list").addEventListener("click",event=>{
    const button=event.target.closest("[data-action]");if(!button)return;
    const index=Number(button.dataset.index),action=button.dataset.action,layers=state.configuration.layers;
    if(action==="toggle"){state.openLayer=state.openLayer===index?-1:index;renderLayers();return;}
    if(action==="remove"&&layers.length>1){layers.splice(index,1);state.openLayer=Math.min(index,layers.length-1);}
    if(action==="up"&&index>0){[layers[index-1],layers[index]]=[layers[index],layers[index-1]];state.openLayer=index-1;}
    if(action==="down"&&index<layers.length-1){[layers[index+1],layers[index]]=[layers[index],layers[index+1]];state.openLayer=index+1;}
    if(action==="add-component"){if(layers[index].components.length>=limits.components){error(`A material supports up to ${limits.components} components.`);return;}layers[index].components.push({formula:"",fraction:0});}
    if(action==="remove-component"&&layers[index].components.length>1)layers[index].components.splice(Number(button.dataset.component),1);
    dirty();renderLayers();
  });
  $("#add-layer").addEventListener("click",()=>{
    if(state.configuration.layers.length>=limits.layers){error(`This workspace supports up to ${limits.layers} layers.`);return;}
    const preset=state.presets.find(item=>item.id===$("#new-preset").value);
    const layer=preset?clone(preset.layer):{name:`Material ${state.configuration.layers.length+1}`,basis:"mass",components:[{formula:"Al2O3",fraction:1}],density_g_cm3:3.95,thickness_mm:.5,angle_deg:0,porosity:0,density_mode:"bulk",density_uncertainty_pct:0,thickness_uncertainty_pct:0};
    layer.density_uncertainty_pct??=0;layer.thickness_uncertainty_pct??=0;
    state.configuration.layers.push(layer);state.openLayer=state.configuration.layers.length-1;dirty();renderLayers();
    if(preset?.description)toast(preset.description);
  });
  // Narrow public hook for optional modules such as the materials library. Layers
  // enter through the same path as presets, so limits, live recalculation and
  // browser memory all apply; nothing else about the application is exposed.
  window.AttenuationWorkbench = Object.freeze({
    addLayer(layer, message = "") {
      if (state.configuration.layers.length >= limits.layers) { error(`This workspace supports up to ${limits.layers} layers.`); return false; }
      const copy = clone(layer);
      copy.density_uncertainty_pct ??= 0; copy.thickness_uncertainty_pct ??= 0;
      state.configuration.layers.push(copy); state.openLayer = state.configuration.layers.length - 1;
      dirty(); renderLayers();
      if (message) toast(message);
      return true;
    },
    layerCount: () => state.configuration.layers.length,
    energy: () => clone(state.configuration.energy),
  });
  $$("[data-tab]").forEach(button=>button.addEventListener("click",()=>activateTab(button.dataset.tab)));
  $(".tabs").addEventListener("keydown",event=>{
    if(!["ArrowLeft","ArrowRight","Home","End"].includes(event.key))return;
    event.preventDefault();const buttons=$$("[data-tab]");const current=buttons.indexOf(document.activeElement);
    const next=event.key==="Home"?0:event.key==="End"?buttons.length-1:(current+(event.key==="ArrowRight"?1:-1)+buttons.length)%buttons.length;
    activateTab(buttons[next].dataset.tab);buttons[next].focus();
  });
  let resizeTimer;
  window.addEventListener("resize",()=>{
    clearTimeout(resizeTimer);
    resizeTimer=setTimeout(()=>{if(state.tab==="overview")renderChart();if(state.tab==="spectrum")renderSpectrum();},100);
  });
  $("#export-csv").addEventListener("click",exportCSV);
  $("#export-svg").addEventListener("click",exportSVG);
  $("#quick-export-svg").addEventListener("click",exportSVG);
  $("#export-json").addEventListener("click",()=>{if(state.result&&!state.stale)download(JSON.stringify({...state.result,schema_version:1,configuration:state.resultConfiguration},null,2)+"\n","attenuation-results.json","application/json");});
  $("#save-project").addEventListener("click",()=>{try{const config=payload();download(JSON.stringify({schema_version:1,configuration:config,ui:{...dashboard.preferences(),tab:state.tab,lengthUnit:state.lengthUnit,scopeOpen:$("#model-scope").open,suggestionsOpen:!$("#suggestions-section").hidden}},null,2)+"\n","attenuation-project.json","application/json");toast("Project saved with configuration and spectrum.");}catch(exception){error(exception.message);}});
  $("#load-project").addEventListener("click",()=>$("#project-file").click());
  const fullscreenButton = $("#fullscreen-workspace");
  if (!document.documentElement.requestFullscreen) fullscreenButton.hidden = true;
  else {
    fullscreenButton.addEventListener("click", async () => {
      try {
        if (document.fullscreenElement) await document.exitFullscreen();
        else await document.documentElement.requestFullscreen();
      } catch (exception) { error(`Full-screen mode is unavailable: ${exception.message}`); }
    });
    document.addEventListener("fullscreenchange", () => {
      const active = !!document.fullscreenElement;
      fullscreenButton.textContent = active ? "Exit full screen" : "Full screen";
      fullscreenButton.setAttribute("aria-pressed", String(active));
    });
  }
  $("#project-file").addEventListener("change",async event=>{
    const file=event.target.files[0];if(!file)return;
    try{if(file.size>10*1024*1024)throw new Error("Project files must be smaller than 10 MB.");const documentValue=JSON.parse(await file.text());const config=cleanProject(documentValue);restoreUI(documentValue.ui);state.configuration=config;state.openLayer=0;syncControls();dirty();toast("Project loaded. Updating the live results.");}catch(exception){error(`Could not open project: ${exception.message}`);}finally{event.target.value="";}
  });
  $("#import-spectrum").addEventListener("click",()=>$("#spectrum-file").click());
  $("#spectrum-file").addEventListener("change",async event=>{
    const file=event.target.files[0];if(!file)return;
    try{if(file.size>5*1024*1024)throw new Error("Spectrum files must be smaller than 5 MB.");$("#spectrum-data").value=await file.text();$("#spectrum-label").value=file.name.replace(/\.[^.]+$/,"");$("#spectrum-enabled").checked=true;dirty();const spectrum=parseSpectrum();toast(`${spectrum.energy_keV.length} spectrum bins imported. Updating the live results.`);}catch(exception){error(exception.message);}finally{event.target.value="";}
  });
  function restoreUI(ui) {
    if (!ui || typeof ui !== "object") return;
    if (Object.hasOwn(displayUnits.LENGTH_UNITS, ui.lengthUnit)) state.lengthUnit = ui.lengthUnit;
    if (["overview", "materials", "spectrum", "uncertainty", "methods"].includes(ui.tab)) state.tab = ui.tab;
    if (typeof ui.scopeOpen === "boolean") $("#model-scope").open = ui.scopeOpen;
    if (typeof ui.suggestionsOpen === "boolean") { $("#suggestions-section").hidden = !ui.suggestionsOpen; $("#ideas-button").setAttribute("aria-expanded", ui.suggestionsOpen); }
    dashboard.restore(ui);
  }
  $("#length-units").addEventListener("click", event => {
    const button = event.target.closest("[data-length-unit]");
    if (!button) return;
    state.lengthUnit = button.dataset.lengthUnit;
    syncLengthButtons(); renderLayers(); renderResults(); remember();
  });
  $("#smart-suggestions").addEventListener("click", event => {
    const action = event.target.closest("[data-suggestion]")?.dataset.suggestion;
    if (!action) return;
    if (action === "micrometres") { state.lengthUnit = "µm"; syncLengthButtons(); renderLayers(); renderResults(); remember(); }
    else if (action === "inputs") { dashboard.showInputs(); document.querySelector(".configuration").scrollIntoView({behavior: "smooth", block: "start"}); }
    else if (["map", "tau", "design"].includes(action)) { activateTab("overview"); dashboard.selectQuantity(action); $("#plot-grid").scrollIntoView({behavior: "smooth", block: "start"}); }
    else { activateTab(action); $("#results-workspace").scrollIntoView({behavior: "smooth", block: "start"}); }
  });
  async function initialize() {
    dashboard = displayUnits.create(document, window.Plotly, {onError: error, onChange: remember,
      onExport: (svg, filename) => { download(svg, filename, "image/svg+xml"); toast("Figure exported with current view, units and calculation provenance."); },
      onReference: value => { if (value >= 1 && value <= 800) { state.configuration.energy.reference_keV = value; $("#energy-reference").value = value; dirty(); } }});
    let restored = false;
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) { const saved = JSON.parse(raw); state.configuration = cleanProject(saved); state.savedConfiguration = clone(state.configuration); restoreUI(saved.ui); restored = true; }
    } catch { /* Invalid or unavailable browser storage never blocks the workbench. */ }
    syncControls(); activateTab(state.tab); renderResults();
    if (restored) toast("Restored saved controls and the last valid system from this browser.");
    try {
      const response=await fetch("/api/presets");if(!response.ok)throw new Error("Preset request failed");const data=await response.json();
      state.presets=Array.isArray(data.presets)?data.presets:[];
      $("#new-preset").innerHTML='<option value="custom">Custom material</option>'+state.presets.map(preset=>`<option value="${esc(preset.id)}">${esc(preset.name)}</option>`).join("");
    } catch { /* The default material is sufficient; calculate reports connection errors. */ }
    live.flush();
  }
  $("#model-scope").addEventListener("toggle", remember);
  function showIdeas(open) {
    $("#suggestions-section").hidden = !open;
    $("#ideas-button").setAttribute("aria-expanded", open);
    remember();
    if (open) $("#suggestions-section").scrollIntoView({behavior: "smooth", block: "center"});
    else $("#ideas-button").focus();
  }
  $("#ideas-button").addEventListener("click", () => showIdeas($("#suggestions-section").hidden));
  $("#close-suggestions").addEventListener("click", () => showIdeas(false));
  initialize();
})();
