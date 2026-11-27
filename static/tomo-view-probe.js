"use strict";

(() => {
  const svgNS = "http://www.w3.org/2000/svg";

  function element(tag, properties, ...children) {
    const node = Object.assign(document.createElement(tag), properties);
    node.append(...children);
    return node;
  }

  function schedule(count, step) {
    return Number.isInteger(count) && count > 0 && Number.isFinite(step) && step > 0
      ? {count, step}
      : null;
  }

  const displayAngle = angle => Number(angle.toPrecision(13));

  function orientSample(sample, degrees) {
    // These coordinates only describe the illustration. The vertical axis stays fixed.
    const turn = degrees * Math.PI / 180;
    const cos = Math.cos(turn);
    const sin = Math.sin(turn);
    function project(x, z, height) {
      const across = x * cos - z * sin;
      const depth = x * sin + z * cos;
      return [160 + across + 0.55 * depth, 71 - height + 0.35 * depth];
    }
    const footprint = [[-23, -12], [14, -12], [23, -4], [23, 12], [-23, 12]];
    const top = footprint.map(([x, z]) => project(x, z, 20));
    const base = footprint.map(([x, z]) => project(x, z, -18));
    const polygon = (points, fill) =>
      `<polygon points="${points.join(" ")}" fill="var(${fill})"/>`;
    const sides = top.map((point, i) => {
      const next = (i + 1) % top.length;
      return point[0] > top[next][0]
        ? polygon([point, top[next], base[next], base[i]], "--wb-soft")
        : "";
    });
    const [x, y] = project(13, 3, 20);
    sample.innerHTML = sides.join("") + polygon(top, "--wb-accent-soft") +
      `<circle cx="${x}" cy="${y}" r="3" fill="var(--wb-accent)"/>`;
  }

  function diagram() {
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("class", "tomo-view-diagram");
    svg.setAttribute("viewBox", "0 0 320 134");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "A fixed X-ray source on the left sends parallel rays through a sample to a fixed detector on the right. The sample rotates about a vertical axis.");
    svg.style.cssText = "display:block;width:100%;max-width:30rem;height:auto";
    svg.innerHTML = `
      <g stroke="var(--wb-muted)" stroke-width="1.5" stroke-linejoin="round">
        <path d="M20 59h24l9 8v13H29l-9-8Z" fill="var(--wb-soft)"/>
        <path d="M29 67h24m-24 0v13m0-13-9-8" fill="none"/>
        <path d="m251 45 30-13v59l-30 13Z" fill="var(--wb-soft)"/>
        <path d="m251 45-5-4 30-13 5 4m-35 9v59l5 4m10-63v59m10-64v60m-20-31 30-13m-30 33 30-13" fill="none"/>
        <path d="M160 23v80" stroke-dasharray="4 4" fill="none"/>
      </g>
      <g stroke="var(--wb-accent)" stroke-width="2" fill="none">
        <path d="M53 67H252m-199 6H252m-199 6H252"/>
        <path d="m86 63 5 4-5 4m0-2 5 4-5 4m0-2 5 4-5 4"/>
      </g>
      <g fill="var(--wb-ink)" font-size="17" text-anchor="middle">
        <text x="38" y="125">Source</text>
        <text x="160" y="125">Sample</text>
        <text x="266" y="125">Detector</text>
      </g>`;
    const sample = document.createElementNS(svgNS, "g");
    sample.setAttribute("class", "tomo-view-sample");
    sample.setAttribute("stroke", "var(--wb-accent)");
    sample.setAttribute("stroke-width", "1.5");
    sample.setAttribute("stroke-linejoin", "round");
    svg.append(sample);
    return {svg, sample};
  }

  function create(answer) {
    if (answer?.tool !== "tomo_sampling" || !answer.data) return null;
    const data = answer.data;
    const minimum = schedule(data.minimum_views, data.angular_step_deg);
    if (!minimum) return null;
    const planned = schedule(data.planned_views, data.planned_angular_step_deg);
    const schedules = {minimum, planned};
    const selector = element("select", {id: "tomo-view-schedule", disabled: !planned},
      element("option", {value: "minimum", textContent: "Minimum views"}));
    if (planned) selector.append(element("option", {value: "planned", textContent: "Planned views"}));
    selector.value = planned && Number.isSafeInteger(planned.count) ? "planned" : "minimum";

    const range = element("input", {id: "tomo-view-index", type: "range", min: "1", step: "1", value: "1"});
    const readout = element("p", {id: "tomo-view-readout", className: "technique-probe-readout"});
    readout.setAttribute("role", "status");
    const {svg, sample} = diagram();
    const controls = element("div", {className: "technique-probe-controls"},
      element("label", {htmlFor: selector.id, textContent: "View schedule"}, selector),
      element("label", {htmlFor: range.id, textContent: "View number"}, range));
    const panel = element("div", {className: "technique-probe"}, controls, readout);

    if (Number.isFinite(data.nyquist_resolution_um) && typeof data.pitch_supports_resolution === "boolean") {
      const floor = `${data.nyquist_resolution_um} µm`;
      const pixel = data.pitch_supports_resolution
        ? `Pixel sampling supports the target period. The floor is ${floor}.`
        : `The target period is below the pixel sampling floor of ${floor}.`;
      panel.append(element("p", {id: "tomo-view-pixel-status", textContent: pixel}));
    }
    if (planned && typeof data.planned_views_meet_minimum === "boolean") {
      const shortfall = data.planned_view_shortfall;
      let angular = "The planned schedule meets the angular minimum (0 additional views required).";
      if (!data.planned_views_meet_minimum) {
        angular = Number.isSafeInteger(shortfall) && shortfall > 0
          ? `The planned schedule falls short by ${shortfall} ${shortfall === 1 ? "view" : "views"}.`
          : "The planned schedule falls short of the angular minimum.";
      }
      panel.append(element("p", {id: "tomo-view-angular-status", textContent: angular}));
    }
    panel.append(element("figure", {className: "technique-probe-figure"}, svg,
      element("figcaption", {textContent: "Illustrative geometry. The sample turns about the dashed axis while the source, parallel rays and detector stay fixed."})));

    function render() {
      const {count, step} = schedules[selector.value];
      if (range.disabled) {
        const message = "This schedule has too many views for a slider with exact integer steps.";
        readout.textContent = message;
        range.setAttribute("aria-valuetext", message);
        orientSample(sample, 0);
        return;
      }
      const view = Number(range.value);
      const offset = view - 1;
      const angle = offset * step;
      const displayed = displayAngle(angle);
      const rotation = data.rotation_range_deg;
      let angleText = `${displayed}°`;
      let spokenAngle = `${displayed} degrees`;
      if (displayed >= rotation) {
        // Use the remaining steps directly to avoid losing the gap to the excluded endpoint.
        const remaining = displayAngle((count - offset) * step);
        angleText = `${rotation}° − ${remaining}°`;
        spokenAngle = `${rotation} degrees minus ${remaining} degrees`;
      }
      const viewText = `View ${view} of ${count}`;
      readout.textContent = `${viewText} · ${angleText}`;
      range.setAttribute("aria-valuetext", `${viewText}, angle ${spokenAngle}`);
      orientSample(sample, angle);
    }

    function changeSchedule() {
      const {count} = schedules[selector.value];
      const previous = Number(range.value);
      const atEnd = previous > 1 && previous === Number(range.max);
      range.disabled = !Number.isSafeInteger(count);
      range.max = range.disabled ? "1" : String(count);
      range.value = String(range.disabled ? 1 : atEnd ? count : Math.min(previous, count));
      render();
    }

    selector.addEventListener("change", changeSchedule);
    range.addEventListener("input", render);
    changeSchedule();
    return panel;
  }

  globalThis.WorkbenchTomoViewProbe = {create};
})();
