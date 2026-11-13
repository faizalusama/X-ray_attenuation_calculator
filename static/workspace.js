"use strict";

// Presentation preferences only. Calculation inputs are never rewritten by this module.
(() => {
  const MODE_KEY = "attenuation-workspace-mode-v1";
  const validMode = value => value === "simple" || value === "advanced";
  function preferredMode(ui, stored) {
    return validMode(ui?.workspaceMode) ? ui.workspaceMode : validMode(stored) ? stored : "simple";
  }
  function advancedActive(key, config, index = 0) {
    const layer = config.layers?.[index] || {};
    if (key === "density") return layer.density_mode != null && layer.density_mode !== "bulk";
    if (key === "areal") return layer.areal_density_mg_cm2 != null;
    if (key === "angle") return layer.angle_deg !== undefined && layer.angle_deg !== 0;
    if (key === "grid") return config.energy?.points !== 500 || config.energy?.spacing !== "log";
    return false;
  }
  function disclosureVisible(mode, key, config, index, invalid = false, revealed = false) {
    return mode === "advanced" || invalid || revealed || advancedActive(key, config, index);
  }
  function activeSettings(config) {
    const settings = [];
    if (config.spectrum) settings.push({label: "Spectrum enabled", selector: "#spectrum-enabled"});
    if (config.uncertainty?.enabled) settings.push({label: "Uncertainty enabled", selector: "#uncertainty-enabled"});
    (config.layers || []).forEach((layer, index) => {
      const prefix = "Layer " + (index + 1);
      for (const [key, label, field] of [
        ["density", "density model", "density_mode"], ["areal", "areal density", "areal_density_mg_cm2"], ["angle", "angle from normal", "angle_deg"],
      ]) if (advancedActive(key, config, index)) settings.push({label: prefix + ": " + label, selector: '[data-field="' + field + '"][data-layer="' + index + '"]'});
      if (layer.porosity != null && layer.porosity !== 0) settings.push({label: prefix + ": porosity", selector: '[data-field="porosity"][data-layer="' + index + '"]'});
    });
    if (advancedActive("grid", config)) settings.push({label: "Custom energy grid", selector: "#energy-points"});
    return settings;
  }
  function create(document, callbacks, storage = undefined) {
    const $ = selector => document.querySelector(selector);
    const $$ = selector => [...document.querySelectorAll(selector)];
    // Accessing localStorage itself can fail in restricted/private browser contexts.
    if (storage === undefined) try { storage = window.localStorage; } catch { storage = null; }
    let stored;
    try { stored = storage?.getItem(MODE_KEY); } catch { /* Defaults work without storage. */ }
    let mode = preferredMode(null, stored), invalidInput = null;
    const host = $(".workspace");
    function syncCollapse() {
      const expanded = !host.classList.contains("dashboard-wide");
      $("#workspace-inputs").textContent = expanded ? "Hide configuration" : "Show configuration";
      for (const button of [$("#workspace-inputs"), $("#close-inputs"), $("#dashboard-focus")]) {
        button.setAttribute("aria-controls", "configuration-panel");
        button.setAttribute("aria-expanded", String(expanded));
      }
    }
    function reveal(input) {
      if (!input) return;
      const view = input.closest('[role="tabpanel"]');
      if (view) callbacks.activateTab(view.id.replace("view-", ""));
      if (input.closest(".configuration")) callbacks.setInputs(true);
      const card = input.closest(".layer-card");
      if (card) {
        const toggle = card.querySelector(".layer-toggle");
        callbacks.onLayerOpen(Number(toggle.dataset.index));
        card.classList.add("open");
        card.querySelector(".layer-body").hidden = false;
        toggle.setAttribute("aria-expanded", "true");
      }
      for (let parent = input; parent; parent = parent.parentElement) {
        if (parent.dataset?.workspaceAdvanced) { parent.dataset.workspaceRevealed = "true"; parent.hidden = false; }
        if (parent.tagName === "DETAILS") parent.open = true;
      }
      syncCollapse();
      input.focus();
    }
    function update() {
      const config = callbacks.getConfiguration();
      invalidInput = [...$$("#config-form input, #config-form select"),
        ...(config.uncertainty?.enabled ? $$("#view-uncertainty input") : [])]
        .find(input => !input.disabled && input.validity && !input.validity.valid) || null;
      $("#workspace-invalid").hidden = !invalidInput;
      for (const element of $$("[data-workspace-advanced]")) {
        const invalid = [...element.querySelectorAll("input, select, textarea")].some(input => !input.disabled && input.validity && !input.validity.valid);
        const visible = disclosureVisible(mode, element.dataset.workspaceAdvanced, config, Number(element.dataset.layer || 0), invalid, element.dataset.workspaceRevealed === "true");
        // A mode switch must never hide the focused input.
        element.hidden = !visible && !element.contains(document.activeElement);
      }
      document.documentElement.dataset.workspaceMode = mode;
      $$("[data-workspace-mode]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.workspaceMode === mode)));
      const summary = $("#workspace-active-settings"), settings = activeSettings(config);
      summary.replaceChildren();
      summary.hidden = !settings.length;
      if (settings.length) {
        const label = document.createElement("strong");
        label.textContent = "Active advanced settings:";
        summary.append(label);
        for (const setting of settings) {
          const button = document.createElement("button");
          button.type = "button"; button.className = "button outline small"; button.textContent = setting.label;
          button.addEventListener("click", () => reveal($(setting.selector)));
          summary.append(button);
        }
      }
      syncCollapse();
    }
    function setMode(value, persist = true) {
      if (!validMode(value)) return;
      mode = value;
      if (persist) try { storage?.setItem(MODE_KEY, mode); } catch { /* Applies to this visit. */ }
      update();
      if (persist) callbacks.onChange();
    }
    $$("[data-workspace-mode]").forEach(button => button.addEventListener("click", () => setMode(button.dataset.workspaceMode)));
    $("#workspace-invalid").addEventListener("click", () => reveal(invalidInput));
    function hideInputs() {
      // Reuse the dashboard preference, with focus returning to a control on every tab.
      callbacks.setInputs(false);
      syncCollapse();
      $("#workspace-inputs").focus();
    }
    $("#workspace-inputs").addEventListener("click", () => {
      if (host.classList.contains("dashboard-wide")) { callbacks.setInputs(true); syncCollapse(); $("#close-inputs").focus(); }
      else hideInputs();
    });
    $("#close-inputs").addEventListener("click", event => {
      // Supersede the older Overview-only focus return without changing plot code.
      event.stopImmediatePropagation(); hideInputs();
    }, true);
    new MutationObserver(syncCollapse).observe(host, {attributes: true, attributeFilter: ["class"]});
    function placeLibrary() {
      const button = $("#open-materials-library");
      if (button) $("#workspace-library-slot").append(button);
    }
    // Deferred modules may install before or after DOMContentLoaded.
    if (!$("#open-materials-library")) {
      const libraryObserver = new MutationObserver(() => {
        if (!$("#open-materials-library")) return;
        libraryObserver.disconnect();
        placeLibrary();
      });
      libraryObserver.observe(document.body, {childList: true, subtree: true});
    } else placeLibrary();
    update();
    return Object.freeze({
      update, reveal,
      preferences: () => ({workspaceMode: mode}),
      restore: ui => setMode(preferredMode(ui, mode), false),
    });
  }
  const api = {MODE_KEY, preferredMode, advancedActive, disclosureVisible, activeSettings, create};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.AttenuationWorkspace = Object.freeze(api);
})();
