"use strict";

(() => {
  // Drawing coordinates only. The parallel beam runs along x and the sample turns about z.
  const cameras = {
    Oblique: [Math.PI / 6, Math.PI / 7],
    Overhead: [0, Math.PI / 2],
    Side: [0, 0]
  };
  const poses = {
    "Along the beam": 0,
    "Diagonal to the beam": Math.PI / 4,
    "Across the beam": Math.PI / 2
  };

  function draw(svg, camera, pose) {
    const [yaw, tilt] = cameras[camera];
    function project([x, y, z]) {
      const across = x * Math.cos(yaw) - y * Math.sin(yaw);
      const up = (x * Math.sin(yaw) + y * Math.cos(yaw)) * Math.sin(tilt) + z * Math.cos(tilt);
      return [210 + across, 110 - up];
    }

    const scene = document.createDocumentFragment();
    function shape(tag, kind, attrs) {
      const node = document.createElementNS(svg.namespaceURI, tag);
      node.setAttribute("class", "gv-" + kind);
      for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
      scene.append(node);
    }
    function polygon(points, kind) {
      shape("polygon", kind, {points: points.map(point => project(point).join(",")).join(" ")});
    }
    function segment(start, end, kind) {
      const [x1, y1] = project(start), [x2, y2] = project(end);
      shape("line", kind, {x1, y1, x2, y2});
    }

    // The detector and ray endpoints stay in instrument coordinates for every sample pose.
    polygon([[155,-53,-53], [155,53,-53], [155,53,53], [155,-53,53]], "detector");
    for (const offset of [-26, 0, 26]) {
      segment([155,offset,-53], [155,offset,53], "grid");
      segment([155,-53,offset], [155,53,offset], "grid");
    }
    for (const offset of [-18, 0, 18]) {
      const tip = project([-112,offset,offset]);
      const tail = project([-170,offset,offset]);
      segment([-170,offset,offset], [155,offset,offset], "ray");
      const angle = Math.atan2(tip[1] - tail[1], tip[0] - tail[0]) * 180 / Math.PI;
      shape("path", "arrow", {d: "M -8 -4 L 0 0 L -8 4", transform: `translate(${tip}) rotate(${angle})`});
    }

    // A clipped corner makes the specimen orientation visible even from overhead.
    const outline = [[-49,-23], [33,-23], [49,-7], [49,23], [-49,23]];
    const turn = poses[pose];
    function ring(height) {
      return outline.map(([x, y]) => [
        x * Math.cos(turn) - y * Math.sin(turn),
        x * Math.sin(turn) + y * Math.cos(turn), height
      ]);
    }
    const base = ring(-33), lid = ring(33);
    segment([0,0,-62], [0,0,62], "axis");
    base.forEach((a, i) => {
      const j = (i + 1) % base.length, b = base[j];
      // Outward face normal dotted with the horizontal component of the camera direction.
      const facing = (b[0] - a[0]) * Math.cos(yaw) - (b[1] - a[1]) * Math.sin(yaw);
      if (facing * Math.cos(tilt) > 1e-8) polygon([a, b, lid[j], lid[i]], "side");
    });
    if (tilt > 0) polygon(lid, "lid");
    segment([0,0,33], [0,0,62], "axis");
    const [cx, cy] = project([0,0,33]);
    shape("circle", "pivot", {cx, cy, r: 3.5});
    svg.replaceChildren(scene);
  }

  globalThis.WorkbenchGuideVolume = function WorkbenchGuideVolume() {
    const details = document.createElement("details");
    details.className = "guide-volume";
    details.innerHTML = `
      <summary>Explore tomography geometry</summary>
      <div class="gv-body">
        <div class="gv-controls">
          <label>Camera view<select class="gv-camera"></select></label>
          <label>Sample orientation<select class="gv-pose"></select></label>
        </div>
        <figure>
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 220" role="img"
            aria-label="Parallel rays travel from the left through a clipped block to a flat detector on the right. A dot marks the sample’s vertical rotation axis."></svg>
          <figcaption>The arrows trace parallel rays through the sample block to the detector plate on the right. Change the camera to look around the apparatus. Rotate the sample to change the view recorded by the fixed detector.</figcaption>
        </figure>
        <p class="gv-state" role="status" aria-atomic="true"></p>
      </div>`;
    function populate(selector, choices) {
      const select = details.querySelector(selector);
      for (const name of Object.keys(choices)) select.add(new Option(name, name));
      return select;
    }
    const camera = populate(".gv-camera", cameras), pose = populate(".gv-pose", poses);
    const svg = details.querySelector("svg"), state = details.querySelector(".gv-state");
    function update() {
      draw(svg, camera.value, pose.value);
      state.textContent = `${camera.value} camera. The sample’s long edge lies ${pose.value.toLowerCase()}.`;
    }
    camera.addEventListener("change", update);
    pose.addEventListener("change", update);
    update();
    return details;
  };
})();
