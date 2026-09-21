"use strict";

// Pure display transforms keep the scientific API in its documented units.
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.AttenuationDashboard = api;
})(typeof globalThis === "undefined" ? this : globalThis, function () {
  const LENGTH_UNITS = Object.freeze({nm: 1e-6, "µm": 1e-3, mm: 1, cm: 10, m: 1000});
  const QUANTITIES = Object.freeze({
    transmission: {label: "Transmission", title: "Primary-beam transmission", unit: "%", percent: true},
    removed: {label: "Removed fraction", title: "Removal from the primary beam", unit: "%", percent: true},
    mass: {label: "Mass attenuation", title: "Mass attenuation by material", unit: "cm²/g", log: true},
    linear: {label: "Linear attenuation", title: "Linear attenuation by material", unit: "cm⁻¹", log: true},
    channels: {label: "Interactions", title: "Interaction mechanisms", unit: "cm²/g", log: true},
    tau: {label: "Optical depth", title: "Optical depth through the stack", unit: "dimensionless", log: true},
    hvl: {label: "HVL", title: "Half-value path length", length: true, log: true},
    tvl: {label: "TVL", title: "Tenth-value path length", length: true, log: true},
    length: {label: "Attenuation length", title: "Attenuation path length", length: true, log: true},
    design: {label: "Target thickness", title: "Stack thickness required for the target", length: true, log: true},
    map: {label: "Thickness map", title: "Transmission across energy and thickness", unit: "× current thickness"}
  });
  const PALETTE = ["#087f8c", "#6366c9", "#dc8042", "#c34e76", "#319577", "#6686ad", "#9370af", "#987133"];
  const FONT = '"Aptos", "Segoe UI", Arial, sans-serif';
  const MAX_PLOTS = Object.keys(QUANTITIES).length;
  const safe = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  const copy = value => JSON.parse(JSON.stringify(value));
  function panel(primary, id) { return {id, primary, secondary: "none", logY: !!QUANTITIES[primary].log, logY2: false, layer: 0, ranges: {}, visibility: {}}; }
  function allPanels(existing = []) {
    return Object.keys(QUANTITIES).map((mode, i) => ({...copy(existing.find(p => p.primary === mode) || panel(mode, i + 1)), id: i + 1}));
  }
  function fromMM(value, unit = "mm") { return value == null ? null : value / LENGTH_UNITS[unit]; }
  function toMM(value, unit = "mm") { return value == null ? null : value * LENGTH_UNITS[unit]; }
  function axisTitle(mode, lengthUnit) { const q = QUANTITIES[mode]; return `${q.label} · ${q.length ? lengthUnit : q.unit}`; }
  function valuesFor(result, configuration, mode, layerIndex = 0, lengthUnit = "mm") {
    const one = (id, name, values, index = 0, dash = false) => ({id, name, values, color: PALETTE[index % PALETTE.length], dash});
    const material = (key, transform = value => value) => result.layers.map((l, i) => one(`${mode}-${i}`, l.name, l[key].map(transform), i));
    switch (mode) {
      case "transmission": return [one("stack", "Whole stack", result.transmission.map(v => 100 * v)), ...(result.layers.length > 1 ? result.layers.map((l, i) => one(`layer-${i}`, l.name, l.mu_linear_cm_inv.map(v => 100 * Math.exp(-v * l.path_length_mm / 10)), i + 1, true)) : [])];
      case "removed": return [one("removed", "Whole stack", result.removed_fraction.map(v => 100 * v))];
      case "mass": return material("mu_mass_cm2_g");
      case "linear": return material("mu_linear_cm_inv");
      case "tau": return [one("tau", "Whole stack", result.optical_depth), ...result.layers.map((l, i) => one(`tau-${i}`, l.name, l.mu_linear_cm_inv.map(v => v * l.path_length_mm / 10), i + 1, true))];
      case "hvl": return material("mu_linear_cm_inv", v => v > 0 ? fromMM(10 * Math.LN2 / v, lengthUnit) : null);
      case "tvl": return material("mu_linear_cm_inv", v => v > 0 ? fromMM(10 * Math.LN10 / v, lengthUnit) : null);
      case "length": return material("mu_linear_cm_inv", v => v > 0 ? fromMM(10 / v, lengthUnit) : null);
      case "design": {
        const total = configuration.layers.reduce((sum, l) => sum + l.thickness_mm, 0);
        return [one("design", `${configuration.target_transmission * 100}% target`, result.optical_depth.map(t => t > 0 && total > 0 ? fromMM(total * -Math.log(configuration.target_transmission) / t, lengthUnit) : null))];
      }
      case "channels": {
        const l = result.layers[layerIndex] || result.layers[0];
        return [one("photoelectric", `${l.name} · Photoelectric`, l.photoelectric_cm2_g), one("coherent", `${l.name} · Coherent`, l.coherent_cm2_g, 1), one("incoherent", `${l.name} · Incoherent`, l.incoherent_cm2_g, 2)];
      }
      default: return [];
    }
  }
  function thicknessMap(result, maxScale = 3, steps = 41) {
    const scales = Array.from({length: steps}, (_, i) => i * maxScale / (steps - 1));
    return {x: result.energy_keV, y: scales, z: scales.map(s => result.optical_depth.map(t => 100 * Math.exp(-t * s)))};
  }
  function traceData(result, configuration, mode, layerIndex, lengthUnit, axis = "y", log = false, showLayers = true) {
    const unit = QUANTITIES[mode].length ? lengthUnit : QUANTITIES[mode].unit;
    const traces = valuesFor(result, configuration, mode, layerIndex, lengthUnit)
      .filter(series => showLayers || !["transmission", "tau"].includes(mode) || !series.dash)
      .map((series, index) => ({type: "scatter", mode: "lines", x: result.energy_keV,
        y: series.values.map(v => Number.isFinite(v) && (!log || v > 0) ? v : null),
        name: safe(`${series.name}${axis === "y2" ? " · right" : ""}`), uid: `${axis}-${mode}-${series.id}`,
        legendgroup: `${axis}-${mode}-${series.id}`, yaxis: axis,
        line: {color: axis === "y2" ? PALETTE[(index + 1) % PALETTE.length] : series.color, width: series.dash ? 1.7 : 2.6, dash: axis === "y2" ? "dot" : series.dash ? "dash" : "solid", shape: "linear", simplify: false},
        connectgaps: false, hovertemplate: `%{y:.6g} ${safe(unit)}<extra>%{fullData.name}</extra>`}));
    if (mode === "transmission" && result.uncertainty) {
      const group = `${axis}-transmission-stack`, convert = values => values.map(v => log && v <= 0 ? null : v * 100);
      traces.unshift({type: "scatter", x: result.energy_keV, y: convert(result.uncertainty.transmission_p025), yaxis: axis,
        uid: `${axis}-interval-low`, legendgroup: group, mode: "lines", line: {width: 0}, showlegend: false, hoverinfo: "skip", connectgaps: false},
      {type: "scatter", x: result.energy_keV, y: convert(result.uncertainty.transmission_p975), yaxis: axis,
        uid: `${axis}-interval-high`, legendgroup: group, mode: "lines", line: {width: 0}, fill: "tonexty", fillcolor: "rgba(8,127,140,0.15)", name: `95% interval${axis === "y2" ? " · right" : ""}`, hoverinfo: "skip", connectgaps: false});
    }
    return traces;
  }
  function cleanPreferences(value) {
    const result = {}, saved = value?.dashboard;
    if (!saved || typeof saved !== "object") return result;
    for (const key of ["logX", "edges", "layers", "linked", "wheel", "clickReference", "secondaryOpen", "controlsOpen"]) if (typeof saved[key] === "boolean") result[key] = saved[key];
    if (saved.inputLayoutVersion === 1 && typeof saved.wide === "boolean") result.wide = saved.wide;
    if (["zoom", "pan", "locked"].includes(saved.gesture)) result.gesture = saved.gesture;
    if (["stack", "grid"].includes(saved.layout)) result.layout = saved.layout;
    if (Number.isInteger(saved.font) && saved.font >= 11 && saved.font <= 18) result.font = saved.font;
    if (Number.isFinite(saved.maxScale) && saved.maxScale >= .01 && saved.maxScale <= 1000) result.maxScale = saved.maxScale;
    if (Array.isArray(saved.panels) && saved.panels.length >= 1 && saved.panels.length <= MAX_PLOTS) {
      const panels = saved.panels.filter(p => p && Object.hasOwn(QUANTITIES, p.primary)).map((p, i) => ({id: i + 1,
        primary: p.primary, secondary: p.primary !== "map" && p.secondary !== p.primary && p.secondary !== "map" && Object.hasOwn(QUANTITIES, p.secondary) ? p.secondary : "none",
        logY: p.primary !== "map" && (typeof p.logY === "boolean" ? p.logY : !!QUANTITIES[p.primary].log), logY2: typeof p.logY2 === "boolean" ? p.logY2 : false,
        layer: Number.isInteger(p.layer) && p.layer >= 0 && p.layer < 24 ? p.layer : 0,
        visibility: Object.fromEntries(Object.entries(p.visibility || {}).slice(0, 500).filter(([key, v]) => /^[a-z0-9-]{1,100}$/.test(key) && (v === true || v === false || v === "legendonly"))),
        ranges: Object.fromEntries(Object.entries(p.ranges || {}).filter(([key, range]) => ["xaxis", "yaxis", "yaxis2"].includes(key) && Array.isArray(range) && range.length === 2 && range.every(Number.isFinite) && range[1] > range[0]))}));
      if (panels.length) { result.panels = panels; result.active = Number.isInteger(saved.active) && saved.active >= 0 && saved.active < panels.length ? saved.active : 0; }
    }
    return result;
  }
  function create(document, Plotly, callbacks) {
    const $ = selector => document.querySelector(selector);
    const $$ = selector => [...document.querySelectorAll(selector)];
    const settings = {panels: allPanels(), active: 0, layout: "grid", logX: true,
      edges: true, layers: true, linked: true, gesture: "zoom", wheel: false, clickReference: false, font: 13, maxScale: 3, revision: 0, secondaryOpen: false, controlsOpen: false, wide: false, layoutVersion: 3, inputLayoutVersion: 1};
    let result = null, configuration = null, unit = "mm", sequence = MAX_PLOTS, syncing = false, rendering = Promise.resolve(), queued = false;
    const chosen = () => settings.panels[settings.active];
    const button = (label, attr, active, disabled = false) => `<button type="button" ${attr} aria-pressed="${active}" ${disabled ? "disabled" : ""}>${safe(label)}</button>`;
    function controls() {
      const focus = document.activeElement;
      const focusKey = ["select-plot", "quantity", "secondary", "channel-layer"].find(key => focus?.hasAttribute(`data-${key}`));
      const focusValue = focusKey ? focus.getAttribute(`data-${focusKey}`) : null;
      const p = chosen(), isMap = p.primary === "map";
      $("#plot-selectors").innerHTML = settings.panels.map((item, i) => button(`Plot ${i + 1} · ${QUANTITIES[item.primary].label}`, `data-select-plot="${i}"`, i === settings.active)).join("");
      $("#plot-quantities").innerHTML = Object.entries(QUANTITIES).map(([key, q]) => button(q.label, `data-quantity="${key}"`, p.primary === key)).join("");
      $("#secondary-quantities").innerHTML = button("None", 'data-secondary="none"', p.secondary === "none", isMap) + Object.entries(QUANTITIES).filter(([key]) => key !== "map").map(([key, q]) => button(q.label, `data-secondary="${key}"`, p.secondary === key, isMap || key === p.primary)).join("");
      $("#secondary-controls").classList.toggle("unavailable", isMap);
      $("#plot-layer-controls").hidden = ![p.primary, p.secondary].includes("channels");
      $("#plot-layer-buttons").innerHTML = (result?.layers || []).map((l, i) => button(l.name, `data-channel-layer="${i}"`, p.layer === i)).join("");
      $("#map-controls").hidden = !isMap;
      if (document.activeElement !== $("#max-thickness-scale")) $("#max-thickness-scale").value = settings.maxScale;
      $("#add-plot").disabled = settings.panels.length >= MAX_PLOTS;
      $("#remove-plot").disabled = settings.panels.length <= 1;
      $("#log-y").checked = p.logY; $("#log-y").disabled = isMap;
      $("#log-y2").checked = p.logY2; $("#log-y2").disabled = isMap || p.secondary === "none";
      $("#log-x").checked = settings.logX;
      for (const [id, key] of Object.entries({"show-edges":"edges", "show-layer-curves":"layers", "link-axes":"linked", "wheel-zoom":"wheel", "reference-pick":"clickReference"})) $("#" + id).checked = settings[key];
      $("#plot-font-size").value = settings.font;
      $("#secondary-details").open = settings.secondaryOpen;
      $("#plot-settings").open = settings.controlsOpen;
      document.querySelector(".workspace").classList.toggle("dashboard-wide", settings.wide);
      $("#dashboard-focus").setAttribute("aria-pressed", settings.wide);
      $("#dashboard-focus").textContent = settings.wide ? "Show inputs" : "Hide inputs";
      $$("[data-gesture]").forEach(b => b.setAttribute("aria-pressed", b.dataset.gesture === settings.gesture));
      $$("[data-layout]").forEach(b => b.setAttribute("aria-pressed", b.dataset.layout === settings.layout));
      $("#wheel-zoom").disabled = settings.gesture === "locked";
      $("#reference-pick").disabled = settings.gesture === "locked";
      $("#plot-control-title").textContent = `Configure plot ${settings.active + 1}`;
      $("#gesture-help").textContent = settings.gesture === "locked" ? "Gestures locked · hover and legends remain available" : settings.gesture === "pan" ? "Drag to pan · double-click to reset" : "Drag a box to zoom · drag along an axis to scale · double-click to reset";
      $("#export-svg").textContent = `Export plot ${settings.active + 1} SVG ↓`;
      if (focusKey) $(`[data-${focusKey}="${focusValue}"]`)?.focus({preventScroll: true});
    }
    function axis(mode, log, side) {
      const color = side === "right" ? "#6954af" : "#38526a";
      return {title: {text: axisTitle(mode, unit), font: {size: settings.font + 1, color}, standoff: 14},
        type: log ? "log" : "linear", tickfont: {size: settings.font, color}, tickformat: "~g", nticks: 6,
        ticks: "outside", ticklen: 5, tickcolor: "#91a4b5", linecolor: "#b8c8d4", showline: true,
        showgrid: side !== "right", gridcolor: "#e8eef3", zeroline: false, automargin: true,
        exponentformat: "power", showexponent: "all", fixedrange: settings.gesture === "locked",
        rangemode: "tozero", ...(!log && QUANTITIES[mode].percent ? {range: [0, 100], autorange: false} : {autorange: true})};
    }
    function figure(p, exporting = false) {
      const isMap = p.primary === "map", secondary = !isMap && p.secondary !== "none";
      let traces;
      if (isMap) traces = [{type: "heatmap", ...thicknessMap(result, settings.maxScale), zmin: 0, zmax: 100, zsmooth: false,
        colorscale: [[0,"#172947"],[.2,"#215879"],[.45,"#188c99"],[.7,"#6ac8b2"],[1,"#f2edb6"]],
        colorbar: {title: {text: "T · %", side: "top"}, thickness: 12, outlinewidth: 0, ticksuffix: "%", tickfont: {size: settings.font}},
        hovertemplate: "Energy: %{x:.6g} keV<br>Thickness scale: %{y:.4g}×<br>Transmission: %{z:.6g}%<extra></extra>"}];
      else traces = [...traceData(result, configuration, p.primary, p.layer, unit, "y", p.logY, settings.layers),
        ...(secondary ? traceData(result, configuration, p.secondary, p.layer, unit, "y2", p.logY2, settings.layers) : [])];
      const shape = (energy, color, dash, width = 1) => ({type: "line", xref: "x", yref: "paper", x0: energy, x1: energy, y0: 0, y1: 1, layer: isMap ? "above" : "below", line: {color, dash, width}});
      const shapes = [], annotations = [];
      const [min, max] = [result.energy_keV[0], result.energy_keV.at(-1)];
      if (settings.edges) {
        let lastPosition = -1;
        for (const edge of (result.edges || [])) {
          if (edge.energy_keV < min || edge.energy_keV > max) continue;
          shapes.push(shape(edge.energy_keV, "#d5b581", "dot"));
          const position = settings.logX ? Math.log(edge.energy_keV / min) / Math.log(max / min) : (edge.energy_keV - min) / (max - min);
          if (position - lastPosition > .13) {
            annotations.push({xref: "x", yref: "paper", x: settings.logX ? Math.log10(edge.energy_keV) : edge.energy_keV, y: 1.02, text: `${safe(edge.element)} ${safe(edge.shell)}`, showarrow: false, font: {size: settings.font - 2, color: "#8a6d39"}});
            lastPosition = position;
          }
        }
      }
      const ref = result.reference.energy_keV;
      if (ref >= min && ref <= max) {
        shapes.push(shape(ref, "#72849e", "dash", 1.4));
        annotations.push({xref: "x", yref: "paper", x: settings.logX ? Math.log10(ref) : ref, y: 1.10, text: `${ref.toPrecision(4).replace(/\.?0+$/, "")} keV ref`, showarrow: false, bgcolor: "#fff", font: {size: settings.font - 1, color: "#485d78"}});
      }
      const layout = {height: exporting ? 760 : settings.layout === "grid" ? 450 : 540,
        margin: {l: 75, r: secondary || isMap ? 78 : 25, t: 43, b: 80},
        paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff", font: {family: FONT, size: settings.font, color: "#314b61"},
        hovermode: isMap ? "closest" : "x unified", hoverlabel: {bgcolor: "#fff", bordercolor: "#c5d6e2", font: {family: FONT, size: settings.font, color: "#243d56"}},
        dragmode: settings.gesture === "locked" ? false : settings.gesture,
        uirevision: `${p.id}-${p.primary}-${p.secondary}-${unit}-${settings.revision}`,
        legend: {orientation: "h", x: 0, y: -.23, xanchor: "left", yanchor: "top", font: {size: settings.font - 1}, groupclick: "togglegroup"},
        xaxis: {title: {text: "Photon energy · keV", font: {size: settings.font + 1}, standoff: 12}, type: settings.logX ? "log" : "linear",
          range: settings.logX ? [Math.log10(min), Math.log10(max)] : [min, max], autorange: false,
          ticks: "outside", ticklen: 5, tickfont: {size: settings.font}, tickformat: "~g", nticks: settings.layout === "grid" ? 5 : 8, ...(settings.logX ? {dtick: "D2"} : {}),
          linecolor: "#b8c8d4", tickcolor: "#91a4b5", showline: true, gridcolor: "#edf2f6", zeroline: false,
          automargin: true, fixedrange: settings.gesture === "locked", showspikes: true, spikemode: "across", spikesnap: "cursor", spikecolor: "#92a7b8", spikethickness: 1, hoverformat: ".6g"},
        yaxis: axis(p.primary, !isMap && p.logY, "left"), shapes, annotations,
        ...(secondary ? {yaxis2: {...axis(p.secondary, p.logY2, "right"), overlaying: "y", side: "right", tickmode: "auto"}} : {})};
      for (const [name, range] of Object.entries(p.ranges)) if (layout[name]) Object.assign(layout[name], {range, autorange: false});
      if (settings.logX && layout.xaxis.range[1] - layout.xaxis.range[0] < .5) layout.xaxis.dtick = null;
      traces.forEach(trace => { if (trace.uid && Object.hasOwn(p.visibility || {}, trace.uid)) trace.visible = p.visibility[trace.uid]; });
      if (isMap && !p.ranges.yaxis) Object.assign(layout.yaxis, {range: [0, settings.maxScale], autorange: false});
      if (exporting) {
        layout.width = 1400; layout.margin.t = 120; layout.margin.b = 145;
        layout.title = {text: `${safe(QUANTITIES[p.primary].title)}${secondary ? ` / ${safe(QUANTITIES[p.secondary].label)}` : ""}`, x: .05, xanchor: "left", font: {size: 24}};
        layout.annotations.push({xref: "paper", yref: "paper", x: 0, y: -.32, xanchor: "left", yanchor: "top", align: "left", showarrow: false,
          text: `Narrow-beam Beer–Lambert · ${safe(result.provenance.backend?.name || result.provenance.engine)} ${safe(result.provenance.backend?.dataset_version || result.provenance.xraydb_version)} · ${safe(result.provenance.timestamp_utc)}<br>Configuration and figure settings are embedded in SVG metadata.`, font: {size: 12, color: "#607286"}});
      }
      return {data: traces, layout};
    }
    function plotOptions() { return {responsive: true, displayModeBar: false, displaylogo: false, scrollZoom: settings.wheel && settings.gesture !== "locked", doubleClick: settings.gesture === "locked" ? false : "reset", showTips: false}; }
    function bindPlot(graph, original) {
      const current = () => settings.panels.find(p => p.id === original.id);
      graph.on("plotly_restyle", () => { const p = current(); if (!p) return; p.visibility = Object.fromEntries(graph.data.filter(t => t.uid && t.visible != null).map(t => [t.uid, t.visible])); callbacks.onChange?.(); });
      graph.on("plotly_relayout", async changes => {
        if (syncing) return;
        const p = current(); if (!p) return;
        let hasX = false;
        for (const name of ["xaxis", "yaxis", "yaxis2"]) {
          if (changes[`${name}.autorange`]) { delete p.ranges[name]; if (name === "xaxis") hasX = true; }
          const range = changes[`${name}.range`] || (changes[`${name}.range[0]`] !== undefined ? [changes[`${name}.range[0]`], changes[`${name}.range[1]`]] : null);
          if (range?.every(Number.isFinite)) { p.ranges[name] = range; if (name === "xaxis") hasX = true; }
        }
        if (hasX && settings.linked) {
          syncing = true;
          try {
            for (const other of settings.panels) {
              if (other.id === p.id) continue;
              if (p.ranges.xaxis) other.ranges.xaxis = [...p.ranges.xaxis]; else delete other.ranges.xaxis;
              const node = $(`#plot-${other.id}`);
              if (node?.data) await Plotly.relayout(node, p.ranges.xaxis ? {"xaxis.range": p.ranges.xaxis, "xaxis.autorange": false, "xaxis.dtick": settings.logX && p.ranges.xaxis[1] - p.ranges.xaxis[0] >= .5 ? "D2" : null} : {"xaxis.autorange": true});
            }
          } catch (e) { callbacks.onError(e.message); } finally { syncing = false; }
        }
        callbacks.onChange?.();
      });
      graph.on("plotly_click", event => {
        if (settings.clickReference && settings.gesture !== "locked" && event.points?.length) callbacks.onReference(event.points[0].x);
      });
    }
    async function draw() {
      controls();
      if (!result || $("#view-overview").hidden) return;
      if (!Plotly) throw new Error("The local plotting bundle could not be loaded. Refresh the workbench.");
      const grid = $("#plot-grid"); grid.dataset.layout = settings.layout;
      const currentIds = settings.panels.map(p => String(p.id));
      [...grid.children].forEach(card => { if (!currentIds.includes(card.dataset.plotCard)) { Plotly.purge(card.querySelector(".scientific-plot")); card.remove(); } });
      for (const [index, p] of settings.panels.entries()) {
        if (p.layer >= result.layers.length) p.layer = 0;
        let card = $(`[data-plot-card="${p.id}"]`);
        if (!card) {
          card = document.createElement("article"); card.className = "plot-card"; card.dataset.plotCard = p.id;
          card.innerHTML = `<header><button type="button" data-select-card="${p.id}"></button><span class="axis-badge"></span></header><div class="scientific-plot" id="plot-${p.id}" role="img"></div><p class="plot-note"></p>`;
          grid.appendChild(card);
        }
        card.classList.toggle("selected", index === settings.active);
        card.querySelector("header button").textContent = `${String(index + 1).padStart(2, "0")}  ${QUANTITIES[p.primary].title}`;
        card.querySelector("header button").setAttribute("aria-label", `Configure plot ${index + 1}: ${QUANTITIES[p.primary].label}`);
        card.querySelector("header button").setAttribute("aria-pressed", index === settings.active);
        card.querySelector(".axis-badge").textContent = p.secondary !== "none" ? `R · ${QUANTITIES[p.secondary].label}` : "";
        card.querySelector(".plot-note").textContent = p.primary === "map" ? "All layers scale together; composition, density and angle are fixed. Colours show primary transmission. Hover reads sampled cells." : p.primary === "design" ? "Total normal thickness with every layer scaled proportionally. Undefined for a zero-thickness stack." : p.primary === "removed" ? "Removed fraction includes photons absorbed or scattered out of the primary beam. It is not absorbed energy or dose." : ["hvl", "tvl", "length"].includes(p.primary) ? "Path length in each homogeneous material, evaluated at each sampled energy." : "Hover reads evaluated samples · click a legend to hide a series · double-click a legend to isolate";
        const node = card.querySelector(".scientific-plot"), fresh = !node.data, f = figure(p);
        // Re-measure when switching between rows, grid and the input sidebar.
        // Plotly.react otherwise keeps a previous wide plot inside a narrow card.
        f.layout.width = Math.max(240, Math.floor(node.getBoundingClientRect().width));
        node.setAttribute("aria-label", `${QUANTITIES[p.primary].title}${p.secondary !== "none" ? ` with ${QUANTITIES[p.secondary].label} on the right axis` : ""}`);
        await Plotly.react(node, f.data, f.layout, plotOptions());
        if (fresh) bindPlot(node, p);
      }
      $("#plot-context").textContent = `${result.energy_keV.length.toLocaleString()} evaluated energies · ${result.edges?.length || 0} reported edges · Straight segments preserve edge samples${settings.panels.some(p => p.logY || p.logY2) ? " · Log axes omit zero values" : ""}`;
    }
    function render() {
      queued = true;
      callbacks.onChange?.();
      rendering = rendering.then(async () => { if (!queued) return; queued = false; await draw(); }).catch(e => callbacks.onError(`Plot: ${e.message}`));
      return rendering;
    }
    function reset(all = false) { (all ? settings.panels : [chosen()]).forEach(p => p.ranges = {}); settings.revision++; return render(); }
    $(".chart-panel").addEventListener("click", event => {
      const b = event.target.closest("button"); if (!b || b.disabled) return;
      if (b.id === "export-svg" || b.id === "quick-export-svg" || b.dataset.selectCard) return;
      const p = chosen();
      if (b.dataset.selectPlot !== undefined) settings.active = Number(b.dataset.selectPlot);
      if (b.dataset.quantity) { p.primary = b.dataset.quantity; p.logY = !!QUANTITIES[p.primary].log; if (p.secondary === p.primary || p.primary === "map") p.secondary = "none"; p.ranges = {}; settings.revision++; }
      if (b.dataset.secondary) { p.secondary = b.dataset.secondary; p.logY2 = !!QUANTITIES[p.secondary]?.log; delete p.ranges.yaxis2; settings.revision++; }
      if (b.dataset.channelLayer !== undefined) p.layer = Number(b.dataset.channelLayer);
      if (b.dataset.gesture) settings.gesture = b.dataset.gesture;
      if (b.dataset.layout) settings.layout = b.dataset.layout;
      if (b.id === "add-plot" && settings.panels.length < MAX_PLOTS) { const mode = Object.keys(QUANTITIES).find(key => !settings.panels.some(item => item.primary === key)) || "tau"; const next = panel(mode, ++sequence); if (settings.linked && p.ranges.xaxis) next.ranges.xaxis = [...p.ranges.xaxis]; settings.panels.push(next); settings.active = settings.panels.length - 1; }
      if (b.id === "show-all-plots") { settings.panels = allPanels(settings.panels); sequence = MAX_PLOTS; settings.active = 0; }
      if (b.id === "remove-plot" && settings.panels.length > 1) { settings.panels.splice(settings.active, 1); settings.active = Math.min(settings.active, settings.panels.length - 1); }
      if (b.id === "reset-view") { reset(settings.linked); return; }
      if (b.id === "dashboard-focus") settings.wide = !settings.wide;
      render();
    });
    $("#plot-grid").addEventListener("click", event => { const b = event.target.closest("[data-select-card]"); if (b) { settings.active = settings.panels.findIndex(p => p.id === Number(b.dataset.selectCard)); render(); } });
    $("#dashboard-controls").addEventListener("input", event => {
      const el = event.target;
      if (el.id === "plot-font-size") { settings.font = Number(el.value); render(); }
      if (el.id === "max-thickness-scale" && el.value !== "" && el.checkValidity()) { settings.maxScale = Number(el.value); delete chosen().ranges.yaxis; settings.revision++; render(); }
    });
    $("#plot-settings").addEventListener("toggle", event => { settings.controlsOpen = event.target.open; callbacks.onChange?.(); });
    $("#close-inputs").addEventListener("click", () => { settings.wide = true; render().then(() => $("#dashboard-focus").focus()); });
    $("#secondary-details").addEventListener("toggle", event => { settings.secondaryOpen = event.target.open; callbacks.onChange?.(); });
    $(".chart-panel").addEventListener("change", event => {
      const el = event.target, p = chosen();
      const flags = {"show-edges": "edges", "show-layer-curves": "layers", "link-axes": "linked", "wheel-zoom": "wheel", "reference-pick": "clickReference"};
      if (flags[el.id]) settings[flags[el.id]] = el.checked;
      if (el.id === "link-axes" && el.checked) settings.panels.forEach(other => { if (p.ranges.xaxis) other.ranges.xaxis = [...p.ranges.xaxis]; else delete other.ranges.xaxis; });
      if (el.id === "log-x") { settings.logX = el.checked; settings.panels.forEach(item => delete item.ranges.xaxis); settings.revision++; }
      if (el.id === "log-y") { p.logY = el.checked; delete p.ranges.yaxis; settings.revision++; }
      if (el.id === "log-y2") { p.logY2 = el.checked; delete p.ranges.yaxis2; settings.revision++; }
      if (el.id === "plot-font-size") settings.font = Number(el.value);
      if (el.id === "max-thickness-scale") { if (el.value === "" || !el.checkValidity()) { el.reportValidity(); return; } settings.maxScale = Number(el.value); delete p.ranges.yaxis; settings.revision++; }
      render();
    });
    return {
      update(data) {
        const changedDomain = configuration && data.configuration && ["min_keV", "max_keV", "spacing"].some(key => configuration.energy[key] !== data.configuration.energy[key]);
        const changedUnit = unit !== data.lengthUnit;
        result = data.result; configuration = data.configuration; unit = data.lengthUnit || "mm";
        if (changedDomain) { settings.logX = configuration.energy.spacing === "log"; settings.panels.forEach(p => p.ranges = {}); settings.revision++; }
        if (changedUnit) { settings.panels.forEach(p => { delete p.ranges.yaxis; delete p.ranges.yaxis2; }); settings.revision++; }
        return render();
      },
      async exportSVG() {
        await rendering;
        if (!result) return;
        const p = copy(chosen()), snapshot = copy({configuration, provenance: result.provenance, settings, unit}), graph = $(`#plot-${p.id}`);
        const f = figure(p, true);
        // Match the displayed zoom, hidden series and both independent axis scales.
        const visibility = new Map((graph?.data || []).map(t => [t.uid, t.visible]));
        f.data.forEach(t => { if (visibility.has(t.uid)) t.visible = visibility.get(t.uid); });
        for (const name of ["xaxis", "yaxis", "yaxis2"]) if (f.layout[name] && graph?._fullLayout?.[name]) Object.assign(f.layout[name], {range: [...graph._fullLayout[name].range], autorange: false});
        const host = document.createElement("div"); host.className = "export-stage"; document.body.appendChild(host);
        try {
          await Plotly.newPlot(host, f.data, f.layout, {staticPlot: true, displayModeBar: false});
          const url = await Plotly.toImage(host, {format: "svg", width: 1400, height: 760});
          const svg = new DOMParser().parseFromString(decodeURIComponent(url.split(",").slice(1).join(",")), "image/svg+xml");
          const metadata = svg.createElementNS("http://www.w3.org/2000/svg", "metadata");
          metadata.textContent = JSON.stringify({...snapshot, active_figure: p, axes: {x: f.layout.xaxis, left: f.layout.yaxis, right: f.layout.yaxis2}, visible_series: f.data.filter(t => t.visible !== "legendonly" && t.visible !== false).map(t => t.name)});
          svg.documentElement.prepend(metadata);
          callbacks.onExport(new XMLSerializer().serializeToString(svg), "attenuation-dashboard.svg");
        } finally { Plotly.purge(host); host.remove(); }
      },
      restore(value) {
        Object.assign(settings, cleanPreferences(value));
        // Upgrade an older two/four-plot workspace once. Later custom choices
        // remain remembered; Show all plots restores the complete dashboard.
        if (value?.dashboard?.layoutVersion !== 3) { settings.panels = allPanels(settings.panels); settings.active = 0; settings.layout = "grid"; }
        sequence = Math.max(...settings.panels.map(p => p.id));
        if (Object.hasOwn(LENGTH_UNITS, value?.lengthUnit)) unit = value.lengthUnit;
        controls();
      },
      selectQuantity(mode) {
        if (!Object.hasOwn(QUANTITIES, mode)) return;
        const p = chosen(); p.primary = mode; p.secondary = "none"; p.logY = !!QUANTITIES[mode].log; p.ranges = {}; settings.revision++;
        return render();
      },
      showInputs() { settings.wide = false; return render(); },
      preferences() { return {lengthUnit: unit, dashboard: copy(settings)}; }
    };
  }
  return {LENGTH_UNITS, QUANTITIES, MAX_PLOTS, allPanels, fromMM, toMM, valuesFor, thicknessMap, traceData, cleanPreferences, create};
});
