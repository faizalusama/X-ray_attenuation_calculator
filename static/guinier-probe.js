"use strict";
(function () {
  const svgNamespace = "http://www.w3.org/2000/svg";

  function svgElement(tag, attributes) {
    const element = document.createElementNS(svgNamespace, tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
    return element;
  }

  function displayNumber(value) {
    return String(Number(value.toPrecision(10)));
  }

  function create(answer) {
    if (answer?.tool !== "guinier" || !answer.data) return null;
    const {q_inv_angstrom: q, ln_intensity_fit: fitted, ln_intensity_residuals: residuals} = answer.data;
    const fitPlot = answer.plots?.[0];
    const qSquared = fitPlot?.x;
    const measured = fitPlot?.traces?.[0]?.y;
    const count = q?.length;
    if (!count || ![q, qSquared, measured, fitted, residuals].every(
      series => Array.isArray(series) && series.length === count && series.every(Number.isFinite)
    )) return null;

    const qSquaredSpan = qSquared[count - 1] - qSquared[0];
    if (!(qSquaredSpan > 0) || !Number.isFinite(qSquaredSpan)) return null;
    const amplitude = residuals.reduce((max, value) => Math.max(max, Math.abs(value)), 0) || 1;
    const x = index => (8 + 344 * ((qSquared[index] - qSquared[0]) / qSquaredSpan)).toFixed(2);
    const y = index => (40 - 32 * (residuals[index] / amplitude)).toFixed(2);

    const root = document.createElement("section");
    root.className = "technique-probe";
    root.setAttribute("aria-label", "Guinier sample probe");

    const slider = document.createElement("input");
    slider.id = "guinier-observation";
    slider.type = "range";
    slider.min = "1";
    slider.max = String(count);
    slider.step = "1";
    slider.value = "1";
    slider.style.minHeight = "var(--rd-target, 24px)";
    const label = document.createElement("label");
    label.setAttribute("for", slider.id);
    label.textContent = "Guinier observation";
    const control = document.createElement("div");
    control.className = "technique-probe-control";
    control.append(label, slider);

    const readout = document.createElement("output");
    readout.id = "guinier-observation-readout";
    readout.className = "technique-probe-readout";
    readout.setAttribute("for", slider.id);
    readout.setAttribute("role", "status");
    readout.setAttribute("aria-atomic", "true");

    const caption = document.createElement("p");
    caption.className = "guinier-residual-caption";
    caption.textContent = "Residual vs. q² (Å⁻²). The dashed line marks zero.";
    const trace = svgElement("svg", {class: "guinier-residual-trace", viewBox: "0 0 360 80", width: "100%",
      "aria-hidden": "true", focusable: "false"});
    const baseline = svgElement("line", {x1: 8, y1: 40, x2: 352, y2: 40,
      stroke: "var(--wb-muted, currentColor)", "stroke-dasharray": "4 3", "vector-effect": "non-scaling-stroke"});
    const curve = svgElement("polyline", {points: residuals.map((_, index) => x(index) + "," + y(index)).join(" "),
      fill: "none", stroke: "var(--wb-accent, currentColor)", "vector-effect": "non-scaling-stroke"});
    const marker = svgElement("circle", {class: "guinier-residual-selection", r: 4, fill: "var(--wb-panel, Canvas)",
      stroke: "var(--wb-ink, currentColor)", "stroke-width": 2, "vector-effect": "non-scaling-stroke"});
    trace.append(baseline, curve, marker);

    function update() {
      const index = Number(slider.value) - 1;
      const position = slider.value + " of " + count;
      const residual = residuals[index];
      const sign = residual > 0 ? "+" : "";
      slider.setAttribute("aria-valuetext", position);
      readout.textContent = "Observation " + position + " · q = " + displayNumber(q[index]) + " Å⁻¹ · " +
        "Measured ln(I) = " + displayNumber(measured[index]) + " · Fitted ln(I) = " + displayNumber(fitted[index]) +
        " · Residual (measured − fitted) = " + sign + displayNumber(residual);
      marker.setAttribute("cx", x(index));
      marker.setAttribute("cy", y(index));
    }

    slider.addEventListener("input", update);
    root.append(control, readout, caption, trace);
    update();
    return root;
  }

  globalThis.WorkbenchGuinierProbe = {create};
})();
