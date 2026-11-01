"use strict";
// Total scattering tab. Everything numeric comes from POST /api/total-scattering;
// this file only draws it. It reaches the application only through
// window.AttenuationWorkbench and the attenuation:update event, like the materials library.
(function () {
  const COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A5195", "#8C6D31"];
  const MAX_PAIR_CURVES = 8;
  const ionsByLayer = new Map();
  let latest = null, requestId = 0, timer = null;

  function number(value, digits = 4) {
    if (value === null || value === undefined) return "n/a";
    return Number(value).toLocaleString("en-US", {maximumSignificantDigits: digits, useGrouping: false});
  }
  function esc(value) {
    return String(value).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  }

  // Every curve in one table with fixed column names, for other plotting software.
  function curvesTable(result) {
    const curves = result.curves;
    const columns = [["q_inv_angstrom", curves.q]];
    for (const [symbol, values] of Object.entries(curves.form_factors)) columns.push([`f0_${symbol}`, values]);
    columns.push(["mean_f_squared", curves.mean_f_squared], ["mean_of_f_squared", curves.mean_of_f_squared], ["laue_diffuse", curves.laue]);
    for (const pair of curves.pairs) columns.push([`weight_${pair.pair}`, pair.weights]);
    if (result.differential) {
      for (const key of ["S_mean", "incoherent_barn_sr_atom", "coherent_barn_sr_atom", "coherent_fraction"]) columns.push([key, result.differential[key]]);
    }
    return {headers: columns.map(column => column[0]), rows: curves.q.map((_, i) => columns.map(column => column[1][i]))};
  }

  function card(label, value, unit, note) {
    return `<article class="ts-card"><div class="ts-card-label">${label}</div><div class="ts-card-value">${value}${unit ? ` <span>${unit}</span>` : ""}</div><div class="ts-card-note">${note}</div></article>`;
  }

  function table(caption, headers, rows) {
    return `<div class="ts-table-wrap" tabindex="0" role="region" aria-label="${esc(caption)}"><table class="ts-table"><caption>${esc(caption)}</caption><thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map((cell, i) => i ? `<td>${cell}</td>` : `<th scope="row">${cell}</th>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function renderCards(r) {
    const a = r.attenuation, s = r.sample, q = r.resolution;
    document.querySelector("#ts-cards").innerHTML = [
      card("Qmax", number(q.q_max), "Å⁻¹", `At 2θ = ${number(r.two_theta_max_deg)}°, λ = ${number(r.wavelength_angstrom)} Å`),
      card("r-grid step", number(q.r_step_nyquist), "Å", "Nyquist step, π/Qmax"),
      card("Peak width from Qmax", number(q.fwhm_step, 3), "Å", `FWHM. ${number(q.fwhm_lorch, 3)} Å with a Lorch window`),
      card("Number density ρ₀", number(s.number_density), "atoms/Å³", `From ${number(s.density_g_cm3)} g/cm³`),
      card("G(r) slope at low r", number(s.low_r_slope), "Å⁻²", "−4πρ₀, below the first peak"),
      card("μD of the capillary", number(a.mu_d, 3), "", `T = ${number(100 * a.transmission, 3)}% across ${number(r.capillary_diameter_mm)} mm; μR = ${number(a.mu_r, 3)}`),
      card("Best flat-plate thickness", number(a.plate_optimum_mm, 3), "mm", "t = 1/μ, for most scatter at normal incidence"),
      card("Compton at 2θmax", number(r.compton.energy_keV), "keV", `Shifted ${number(r.compton.shift_keV, 3)} keV; ${number(100 * (a.shares.incoherent ?? 0), 3)}% of μ/ρ is incoherent`),
    ].join("");
    document.querySelector("#ts-composition").textContent = s.composition;
    document.querySelector("#ts-values").textContent = `λ = ${number(r.wavelength_angstrom, 6)} Å · ρ₀ = ${number(s.number_density)} atoms/Å³ · Qmax = ${number(q.q_max)} Å⁻¹ · ⟨Z⟩ = ${number(s.mean_atomic_number)} · ⟨A⟩ = ${number(s.mean_atomic_mass)} u`;
  }

  function renderTables(r) {
    document.querySelector("#ts-ion-controls").innerHTML = r.elements.map(e =>
      '<label>' + esc(e.symbol) + ' form factor<select data-ion="' + esc(e.symbol) + '">' + e.ion_choices.map(ion =>
        '<option value="' + esc(ion) + '"' + (ion === e.ion ? ' selected' : '') + '>' + esc(ion) + '</option>').join("") + '</select></label>').join("");
    document.querySelector("#ts-elements").innerHTML = table("Elements", ["Element", "Atomic fraction", "Mass fraction", "f₀(0)", "f′", "f″"],
      r.elements.map(e => [esc(e.symbol), number(e.atomic_fraction), number(e.mass_fraction), number(e.f0_at_zero), number(e.f_prime), number(e.f_double_prime)]));
    document.querySelector("#ts-pairs").innerHTML = table("Pair weights", ["Pair", "w at Q = 0", `w at Q = ${number(r.curves.q.at(-1), 3)} Å⁻¹`],
      r.curves.pairs.map(p => [esc(p.pair), number(p.at_zero), number(p.at_q_max)]));
    document.querySelector("#ts-sources").innerHTML = table(`Sources at 2θ = ${number(r.two_theta_max_deg)}° for this sample`, ["Source", "E (keV)", "λ (Å)", "Qmax (Å⁻¹)", `2θ for Q = ${number(r.target_q)} Å⁻¹`, "μD", "T across D"],
      r.sources.map(s => [esc(s.name), number(s.energy_keV), number(s.wavelength_angstrom), number(s.q_max), s.two_theta_for_target_deg === null ? "out of reach" : `${number(s.two_theta_for_target_deg, 3)}°`, number(s.mu_d, 3), `${number(100 * s.transmission, 3)}%`]));
    document.querySelector("#ts-fluorescence").innerHTML = r.fluorescence.length
      ? table("Edges the beam can excite", ["Element", "Edge", "Edge (keV)", "Strongest line", "Line (keV)"], r.fluorescence.map(f => [esc(f.element), esc(f.edge), number(f.edge_keV), esc(f.line), number(f.line_keV)]))
      : "<p>The beam excites no K or L3 line at or above 1 keV in this sample.</p>";
    const notes = document.querySelector("#ts-warnings");
    notes.hidden = !r.warnings.length;
    notes.innerHTML = r.warnings.length ? `<strong>Check this</strong><ul>${r.warnings.map(w => `<li>${esc(w)}</li>`).join("")}</ul>` : "";
  }

  function layout(title, yTitle) {
    const dark = document.documentElement.dataset.theme === "dark";
    const ink = dark ? "#dbe5ee" : "#253647", grid = dark ? "#2c3b49" : "#e3eaee", panel = dark ? "#18222c" : "#fff";
    return {margin: {l: 66, r: 16, t: 34, b: 52}, title: {text: title, font: {size: 15}}, font: {family: "Segoe UI, Arial, sans-serif", size: 13, color: ink},
      xaxis: {title: {text: "Q (Å⁻¹)"}, gridcolor: grid, zeroline: false}, yaxis: {title: {text: yTitle}, gridcolor: grid, zeroline: false},
      legend: {font: {size: 12}}, paper_bgcolor: panel, plot_bgcolor: panel, hovermode: "x unified"};
  }

  function renderPlots(r) {
    if (!window.Plotly) return;
    const curves = r.curves, q = curves.q;
    const line = (name, y, color, dash) => ({type: "scatter", mode: "lines", x: q, y, name, line: {color, width: 2, dash}, hovertemplate: "%{y:.4g}"});
    const symbols = Object.keys(curves.form_factors);
    const dark = document.documentElement.dataset.theme === "dark";
    window.Plotly.react("ts-plot-factors", [
      ...symbols.map((s, i) => line(`f₀ ${s}`, curves.form_factors[s], COLORS[i % COLORS.length])),
    ], layout("Form factors", "f₀(Q) (electrons)"), {responsive: true, displayModeBar: false});
    window.Plotly.react("ts-plot-means", [
      line("⟨f²⟩", curves.mean_of_f_squared, dark ? "#dbe5ee" : "#203951"),
      line("⟨f⟩²", curves.mean_f_squared, COLORS[1], "dash"),
      line("Laue, ⟨f²⟩ − ⟨f⟩²", curves.laue, COLORS[2], "dot"),
    ], {...layout("Composition averages", "electrons²")}, {responsive: true, displayModeBar: false});
    window.Plotly.react("ts-plot-weights", curves.pairs.slice(0, MAX_PAIR_CURVES).map((p, i) => line(p.pair, p.weights, COLORS[i % COLORS.length])),
      layout(curves.pairs.length > MAX_PAIR_CURVES ? `Largest ${MAX_PAIR_CURVES} of ${curves.pairs.length} pair weights` : "Faber-Ziman pair weights", "w(Q)"), {responsive: true, displayModeBar: false});
    const extra = document.querySelector("#ts-plot-compton");
    window.Plotly.react(extra, [line("Coherent", r.differential.coherent_barn_sr_atom, COLORS[0]),
      line("Compton", r.differential.incoherent_barn_sr_atom, COLORS[1])],
      layout("Differential scattering (unpolarized)", "barn sr⁻¹ atom⁻¹"), {responsive: true, displayModeBar: false});
    window.Plotly.react("ts-plot-incoherent", [line("⟨S(Q)⟩", r.differential.S_mean, COLORS[1]),
      {...line("Coherent fraction", r.differential.coherent_fraction, COLORS[0]), yaxis: "y2"}],
      {...layout("Incoherent function and coherent fraction", "⟨S(Q)⟩ (electrons)"),
        yaxis2: {title: {text: "Coherent fraction"}, overlaying: "y", side: "right", range: [0, 1]},
        margin: {l: 66, r: 70, t: 34, b: 52}}, {responsive: true, displayModeBar: false});
    document.querySelector("#ts-cross-check").innerHTML = r.differential.comparison.length ?
      table("Integrated differential scattering versus Elam", ["Channel", "Integral (cm²/g)", "Elam (cm²/g)", "Difference"],
        r.differential.comparison.map(c => [esc(c.channel), number(c.integrated_cm2_g), number(c.elam_cm2_g), number(c.relative_difference * 100, 3) + "%"])) :
      "<p>Full-angle integration is unavailable at this energy: it would exceed the form-factor table range.</p>";
    const cap = r.capillary, capLayout = layout("Capillary geometry and polarization", "Factor");
    capLayout.xaxis.title.text = "2θ (°)";
    const capCurves = Object.entries(cap.polarization).map(([name, y], i) => ({...line(name.replaceAll("_", " "), y, COLORS[i + 1], "dot"), x: cap.two_theta_deg}));
    if (cap.absorption) capCurves.unshift({...line("Absorption factor", cap.absorption, COLORS[0]), x: cap.two_theta_deg});
    window.Plotly.react("ts-plot-capillary", capCurves, capLayout, {responsive: true, displayModeBar: false});
    document.querySelector("#ts-capillary-note").textContent = cap.note;
    const biggest = curves.pairs[0];
    document.querySelector("#ts-plot-summary").textContent = `${symbols.length} element${symbols.length === 1 ? "" : "s"}, ${curves.pairs.length} pair${curves.pairs.length === 1 ? "" : "s"}. The largest pair weight at Q = 0 is ${biggest.pair}, ${number(biggest.at_zero, 3)}, becoming ${number(biggest.at_q_max, 3)} at Q = ${number(q.at(-1), 3)} Å⁻¹.`;
  }

  // Results are hidden, not left stale, whenever the inputs are invalid.
  function showResults(visible) {
    document.querySelectorAll("[data-ts-result]").forEach(element => { element.hidden = !visible; });
  }

  function show(result) {
    latest = result;
    showResults(true);
    renderCards(result);
    renderTables(result);
    renderPlots(result);
    document.querySelector("#ts-export").disabled = false;
    document.querySelector("#ts-status").textContent = `${result.sample.name} at ${number(result.energy_keV)} keV`;
  }

  async function refresh() {
    const panel = document.querySelector("#view-scattering"), api = window.AttenuationWorkbench;
    if (!panel || panel.hidden || !api) return;
    const layers = api.layers(), energy = api.referenceEnergy();
    if (!layers.length || !energy) return;
    const select = document.querySelector("#ts-layer"), kept = Number(select.value) || 0;
    select.innerHTML = layers.map((layer, i) => `<option value="${i}">${esc(layer.name || `Layer ${i + 1}`)}</option>`).join("");
    const index = Math.min(kept, layers.length - 1);
    select.value = String(index);
    const id = ++requestId;
    const ions = ionsByLayer.get(index) || {};
    const body = {layer: layers[index], ions, energy_keV: energy, two_theta_max_deg: Number(document.querySelector("#ts-two-theta").value),
                  capillary_diameter_mm: Number(document.querySelector("#ts-capillary").value), target_q: Number(document.querySelector("#ts-target").value)};
    const status = document.querySelector("#ts-status");
    try {
      const response = await fetch("/api/total-scattering", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
      const answer = await response.json();
      if (id !== requestId) return;
      if (!response.ok) throw new Error(answer.detail || "The request failed.");
      show(answer);
    } catch (exception) {
      if (id === requestId) {
        latest = null;
        showResults(false);
        status.textContent = exception.message;
        document.querySelector("#ts-export").disabled = true;
      }
    }
  }
  function schedule() { clearTimeout(timer); timer = setTimeout(refresh, 200); }

  function exportCurves() {
    if (!latest) return;
    const api = window.AttenuationWorkbench, table = curvesTable(latest);
    const csv = [table.headers, ...table.rows].map(row => row.join(",")).join("\r\n") + "\r\n";
    api.download(csv, "total-scattering-curves.csv", "text/csv;charset=utf-8");
    const {curves, ...settings} = latest;
    setTimeout(() => api.download(`${JSON.stringify(settings, null, 2)}\n`, "total-scattering-summary.json", "application/json"), 500);
  }

  function install() {
    if (!document.querySelector("#view-scattering")) return;
    ["#ts-layer", "#ts-two-theta", "#ts-capillary", "#ts-target"].forEach(id => document.querySelector(id).addEventListener("input", schedule));
    document.querySelector("#ts-ion-controls").addEventListener("change", event => {
      if (!event.target.dataset.ion) return;
      const index = Number(document.querySelector("#ts-layer").value);
      const choices = Object.fromEntries([...document.querySelectorAll("[data-ion]")].map(s => [s.dataset.ion, s.value]));
      ionsByLayer.set(index, choices); schedule();
    });
    document.querySelector("#ts-export").addEventListener("click", exportCurves);
    window.addEventListener("attenuation:update", () => { ionsByLayer.clear(); schedule(); });
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install); else install();
  }
  if (typeof module !== "undefined") module.exports = {curvesTable};
})();
