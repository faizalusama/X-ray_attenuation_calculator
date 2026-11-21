"use strict";
(function () {
  const escape = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const format = value => Number(value).toLocaleString("en-US", {maximumSignificantDigits: 5});
  const line = (x1, y1, x2, y2, kind = "") => '<line class="' + kind + '" x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '"/>';
  function svg(body, title, description) {
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300" role="img" aria-labelledby="sg-title sg-desc">' +
      '<title id="sg-title">' + escape(title) + '</title><desc id="sg-desc">' + escape(description) + '</desc>' + body + '</svg>';
  }
  function detectorScene(data, index) {
    const i = Math.min(data.radius_mm.length - 1, Math.max(0, Math.round(index)));
    const radius = data.radius_mm[i];
    const scale = Math.min(420 / data.distance_mm, 110 / data.outer_radius_mm);
    const sample = 70, centre = 150, detector = sample + scale * data.distance_mm;
    const outer = scale * data.outer_radius_mm, inner = scale * data.inner_radius_mm;
    const probe = scale * radius;
    const summary = 'Probe radius ' + format(radius) + ' mm, scattering angle 2θ ' +
      format(data.two_theta_deg[i]) + ' degrees, q ' + format(data.q_inv_angstrom[i]) + ' inverse ångström.';
    const body = '<path class="sg-envelope" d="M70 150 L' + detector + ' ' + (centre - outer) +
      ' L' + detector + ' ' + (centre + outer) + ' Z"/>' +
      line(20, centre, detector + 28, centre, "sg-axis") +
      line(20, centre, sample, centre, "sg-incident") +
      line(detector, centre - outer, detector, centre + outer, "sg-detector") +
      line(detector, centre - inner, detector, centre + inner, "sg-blocked") +
      line(sample, centre, detector, centre - outer, "sg-boundary") +
      line(sample, centre, detector, centre - inner, "sg-boundary") +
      line(sample, centre, detector, centre - probe, "sg-probe") +
      '<circle cx="70" cy="150" r="6" class="sg-sample"/>' +
      '<circle cx="' + detector + '" cy="' + (centre - probe) + '" r="5" class="sg-point"/>' +
      line(sample, 280, detector, 280, "sg-dimension") +
      line(sample, 272, sample, 288, "sg-dimension") +
      line(detector, 272, detector, 288, "sg-dimension");
    return {markup: svg(body, "Detector radial coverage", summary +
      " The sample is the circle on the left. The detector is the vertical line on the right. Horizontal and vertical distances share one scale."),
      summary, radius, angle: data.two_theta_deg[i], q: data.q_inv_angstrom[i], index: i};
  }
  function projectionScene(data) {
    const maximum = Math.max(data.sample_counts, data.flat_counts);
    const sampleWidth = 440 * (data.sample_counts / maximum);
    const flatWidth = 440 * (data.flat_counts / maximum);
    const darkWidth = 440 * (data.dark_counts / maximum);
    const summary = 'Sample ' + format(data.sample_counts) + ', flat ' + format(data.flat_counts) +
      ', dark ' + format(data.dark_counts) + ' counts. Transmission ' + format(data.transmission) +
      ', projection ' + format(data.line_integral) + '.';
    const body = '<rect x="70" y="50" width="' + flatWidth + '" height="65" rx="6" class="sg-flat"/>' +
      '<rect x="70" y="50" width="' + darkWidth + '" height="65" class="sg-dark-counts"/>' +
      '<rect x="70" y="165" width="' + sampleWidth + '" height="65" rx="6" class="sg-transmitted"/>' +
      '<rect x="70" y="165" width="' + darkWidth + '" height="65" class="sg-dark-counts"/>' +
      line(70 + darkWidth, 36, 70 + darkWidth, 249, "sg-dark-marker") +
      line(70, 270, 510, 270, "sg-dimension");
    return {markup: svg(body, "Raw flat and sample counts with dark contribution", summary +
      " The upper bar is the flat reading. The lower bar is the sample reading. The dark contribution is the first segment of each bar. All counts share one linear scale."),
      summary};
  }
  function create(host, changeInput) {
    let current = null, tool = null, probe = 100, spatial = true, azimuth = 40, elevation = 20, drag = null;
    const setText = (selector, text) => { host.querySelector(selector).textContent = text; };
    function clear(nextTool) {
      current = null;
      if (nextTool !== tool) host.hidden = true;
      host.setAttribute("aria-busy", "true");
      const drawing = host.querySelector(".sg-drawing");
      if (drawing) drawing.innerHTML = "";
      const values = host.querySelector(".sg-values");
      if (values) values.innerHTML = "";
      const exact = host.querySelector(".sg-exact");
      if (exact) exact.textContent = "";
      host.querySelectorAll("#sg-probe,button").forEach(control => { control.disabled = true; });
      const note = host.querySelector(".sg-state");
      if (note) note.textContent = "Waiting for a valid calculation.";
    }
    function install(nextTool) {
      tool = nextTool;
      const detector = tool === "detector";
      const extended = tool === "optics" || tool === "pdf";
      const caption = tool === "optics" ?
        '<span class="sg-key sg-key-beam">Incident ray</span><span class="sg-key sg-key-edge">Reflected ray</span><span class="sg-key sg-key-probe">Normal depth marker</span><span class="sg-view-description">Grazing angle is measured from the surface. The depth marker is schematic; the readout gives the intensity 1/e depth.</span>' :
        tool === "pdf" ? '<span class="sg-key sg-key-beam">Windowed input · Q space</span><span class="sg-key sg-key-probe">G(r) · real space</span><span class="sg-view-description">Each preview uses its own vertical scale. The full Q interval contributes to every r point.</span>' :
        detector ? '<span class="sg-key sg-key-beam">Incident beam</span><span class="sg-key sg-key-edge">Usable radial limit</span><span class="sg-key sg-key-probe">Probe ray</span><span class="sg-view-description"></span>' :
        '<span class="sg-key sg-key-beam">Upper · open-beam flat</span><span class="sg-key sg-key-probe">Lower · sample</span><span class="sg-view-description">The grey segment is the dark contribution. Both bars use the same count scale.</span>';
      const model = tool === "optics" ?
        "The probe selects an existing grazing-angle sample and its calculated normal intensity depth. Optical constants and the approximate critical angle come from the current material and reference energy." :
        tool === "pdf" ? "G(r) is computed from all supplied Q samples over the displayed interval, with the selected window. The probe reads the supplied S(Q) and modified transform integrand at a single Q." :
        detector ? "The rays use the validated radius-to-angle results. 2θ = atan2(R, L), q = 4π sin(θ)/λ. The probe selects a calculated point. The numerical inputs above accept distances beyond the slider range." :
        "T = (sample − dark)/(flat − dark). The projection is −ln(T). The displayed values come from the current calculation. Above-flat counts retain T greater than one and a negative projection.";
      host.hidden = false;
      const heading = {detector:"Detector geometry", projection:"Tomography signal", optics:"Grazing incidence", pdf:"Q space to real space"};
      host.innerHTML = '<div class="sg-heading"><h3>' + heading[tool] + '</h3><button type="button" class="button outline small" data-sg-export>Save diagram SVG</button></div>' +
        (detector ? '<div class="sg-view segmented" role="group" aria-label="Detector view"><button type="button" data-sg-view="spatial">3D view</button><button type="button" data-sg-view="side">Side view</button></div>' : '') +
        '<p class="sg-state"></p><figure class="sg-figure"><div class="sg-drawing"></div>' +
        '<figcaption>' + caption + '</figcaption></figure>' +
        '<div class="sg-values"></div><div class="sg-controls">' +
        (extended ? '<label for="sg-probe">' + (tool === "optics" ? 'Probe grazing angle' : 'Probe input Q') + '<input id="sg-probe" type="range" min="0" max="200" step="1" value="100"></label>' :
          detector ? '<label for="sg-distance">Move detector (mm)<input id="sg-distance" type="range" data-sg-field="distance_mm" min="10" max="5000" step="1"></label>' +
          '<label for="sg-probe">Probe detector radius<input id="sg-probe" type="range" min="0" max="200" step="1" value="100"></label>' :
          '<label for="sg-sample">Explore sample counts<input id="sg-sample" type="range" data-sg-field="sample_counts" min="51" max="2100" step="1"></label>') +
        '</div>' + (detector ? '<div class="sg-orbit sg-controls"><label for="sg-azimuth">Rotate view <span id="sg-azimuth-value" aria-hidden="true"></span><input id="sg-azimuth" type="range" min="15" max="165" step="1"></label><label for="sg-elevation">Tilt view <span id="sg-elevation-value" aria-hidden="true"></span><input id="sg-elevation" type="range" min="-45" max="55" step="1"></label></div>' : '') +
        '<details class="sg-equations"><summary>Model and exact values</summary><p>' +
        model +
        '</p><pre class="sg-exact"></pre></details>';
    }
    function draw() {
      if (!current) return;
      const data = current.data, detector = tool === "detector";
      const extended = tool === "optics" || tool === "pdf";
      const scene = tool === "optics" ? window.OpticsSchematic.opticsScene(current, probe) :
        tool === "pdf" ? window.PdfSchematic.pdfScene(current, probe) :
        detector ? detectorScene(data, probe) : projectionScene(data);
      const drawing = host.querySelector(".sg-drawing");
      drawing.toggleAttribute("data-sg-spatial", detector && spatial);
      drawing.innerHTML = detector && spatial ?
        svg(window.DetectorSpatial.detectorSpatial(data, scene.radius, azimuth, elevation), "Rotatable 3D detector geometry",
          scene.summary + " Circular outlines show radial limits, not the outline of a physical detector. The camera uses orthographic projection.") : scene.markup;
      if (detector) {
        setText(".sg-view-description", spatial ? "Radial limits on an ideal plane. Drag to rotate, or use the view controls below." : "Side view with a common distance scale.");
        host.querySelector(".sg-orbit").hidden = !spatial;
        host.querySelectorAll("[data-sg-view]").forEach(button => button.setAttribute("aria-pressed", String(spatial === (button.dataset.sgView === "spatial"))));
        for (const [id, angle] of [["azimuth", azimuth], ["elevation", elevation]]) {
          const control = host.querySelector("#sg-" + id);
          control.value = String(angle);
          control.setAttribute("aria-valuetext", angle + " degrees");
          setText("#sg-" + id + "-value", angle + "°");
        }
      }
      const values = extended ? scene.values : detector ? [
        ["Distance", data.distance_mm, "mm"], ["Probe radius", scene.radius, "mm"],
        ["Scattering angle 2θ", scene.angle, "°"], ["Probe q", scene.q, "Å⁻¹"],
      ] : [["Flat − dark", data.flat_counts - data.dark_counts, "counts"],
        ["Sample − dark", data.sample_counts - data.dark_counts, "counts"],
        ["Transmission", data.transmission, ""], ["Projection −ln(T)", data.line_integral, ""]];
      host.querySelector(".sg-values").innerHTML = values.map(([label, value, unit]) =>
        '<div><span>' + escape(label) + '</span><strong>' + escape(typeof value === "number" ? format(value) : value ?? "undefined") + ' ' + escape(unit) + '</strong></div>').join("");
      setText(".sg-exact", JSON.stringify(extended ? scene.exact : detector ? {distance_mm: data.distance_mm, radius_mm: scene.radius,
        two_theta_deg: scene.angle, q_inv_angstrom: scene.q} : data, null, 2));
      if (detector || extended) {
        const range = host.querySelector("#sg-probe");
        range.max = String(extended ? scene.maxIndex : data.radius_mm.length - 1);
        range.value = String(scene.index);
        range.setAttribute("aria-valuetext", scene.summary);
      }
    }
    function update(answer) {
      if (!["detector", "projection", "optics", "pdf"].includes(answer.tool)) { clear(); host.hidden = true; tool = null; return; }
      if (answer.tool !== tool) install(answer.tool);
      host.hidden = false;
      current = answer;
      host.setAttribute("aria-busy", "false");
      host.querySelectorAll("input,button").forEach(control => { control.disabled = false; });
      const data = answer.data;
      const field = host.querySelector("[data-sg-field]");
      if (field) {
        const value = data[field.dataset.sgField];
        const minimum = answer.tool === "detector" ? Math.min(10, value) : Math.min(data.dark_counts + 1, value);
        const maximum = answer.tool === "detector" ? Math.max(5000, value) : Math.max(data.flat_counts * 2, value, minimum + 1);
        field.min = String(minimum); field.max = String(maximum);
        field.step = "any"; field.value = String(value);
        field.setAttribute("aria-valuetext", format(value) + (answer.tool === "detector" ? " mm" : " counts"));
      }
      setText(".sg-state", answer.tool === "optics" || answer.tool === "pdf" ? "Probe a calculated sample with the slider or arrow keys." :
        answer.tool === "detector" ? "Move the detector or probe a ray. Arrow keys work on each slider." :
        "Change the sample counts to follow flat-field correction.");
      draw();
    }
    host.addEventListener("input", event => {
      const input = event.target;
      if (input.id === "sg-azimuth") { azimuth = Number(input.value); draw(); }
      if (input.id === "sg-elevation") { elevation = Number(input.value); draw(); }
      if (input.id === "sg-probe") { probe = Number(input.value); draw(); }
      if (input.dataset.sgField) changeInput(input.dataset.sgField, input.value);
    });
    host.addEventListener("click", event => {
      const view = event.target.closest("[data-sg-view]");
      if (view && current) { spatial = view.dataset.sgView === "spatial"; draw(); }
      if (event.target.closest("[data-sg-export]") && current) {
        const graphic = host.querySelector("svg").cloneNode(true);
        const original = host.querySelectorAll("svg, svg *");
        [graphic, ...graphic.querySelectorAll("*")].forEach((element, i) => {
          const style = getComputedStyle(original[i]);
          ["fill", "fill-opacity", "stroke", "stroke-width", "stroke-dasharray", "opacity"].forEach(key => element.style.setProperty(key, style.getPropertyValue(key)));
        });
        const background = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        background.setAttribute("width", "100%"); background.setAttribute("height", "100%");
        background.setAttribute("fill", getComputedStyle(host.querySelector(".sg-drawing")).backgroundColor);
        graphic.insertBefore(background, graphic.querySelector("desc")?.nextSibling ?? null);
        const source = new XMLSerializer().serializeToString(graphic);
        window.AttenuationWorkbench.download(source, "technique-diagram.svg", "image/svg+xml;charset=utf-8");
      }
    });
    host.addEventListener("pointerdown", event => {
      const drawing = event.target.closest("[data-sg-spatial]");
      if (!drawing || !current || event.button !== 0) return;
      drawing.setPointerCapture(event.pointerId);
      drag = {id:event.pointerId, x:event.clientX, y:event.clientY, azimuth, elevation};
    });
    host.addEventListener("pointermove", event => {
      if (drag?.id !== event.pointerId) return;
      azimuth = Math.round(Math.max(15, Math.min(165, drag.azimuth + (event.clientX - drag.x) / 2)));
      elevation = Math.round(Math.max(-45, Math.min(55, drag.elevation - (event.clientY - drag.y) / 2)));
      draw();
    });
    host.addEventListener("lostpointercapture", () => { drag = null; });
    return {clear, update};
  }
  const api = {detectorScene, projectionScene, create};
  if (typeof module !== "undefined") module.exports = api;
  if (typeof window !== "undefined") window.TechniqueSchematics = api;
})();
