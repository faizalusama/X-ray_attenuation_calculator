"use strict";
(function () {
  function opticsScene(answer, selected) {
    const curve = answer.plots[0], data = answer.data;
    const index = Math.max(0, Math.min(curve.x.length - 1, Number.isFinite(selected) ? Math.round(selected) : 0));
    const angle = curve.x[index], depth = curve.traces[0].y[index];
    const radians = angle * Math.PI / 180;
    const dx = 140 * Math.cos(radians), dy = 140 * Math.sin(radians);
    const summary = "Grazing angle " + angle + " degrees from surface; normal intensity 1/e depth " + depth +
      " nm. The rays illustrate incidence and reflection; the depth marker is symbolic.";
    const markup = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300" role="img" aria-labelledby="sg-title sg-desc">' +
      '<title id="sg-title">Grazing-incidence optical geometry</title><desc id="sg-desc">' + summary + '</desc>' +
      '<rect x="40" y="180" width="520" height="90" class="sg-envelope"/>' +
      '<line x1="40" y1="180" x2="560" y2="180" class="sg-axis"/>' +
      '<line x1="' + (300 - dx) + '" y1="' + (180 - dy) + '" x2="300" y2="180" class="sg-incident"/>' +
      '<line x1="300" y1="180" x2="' + (300 + dx) + '" y2="' + (180 - dy) + '" class="sg-boundary"/>' +
      '<line x1="300" y1="180" x2="300" y2="250" class="sg-probe"/>' +
      '<line x1="288" y1="250" x2="312" y2="250" class="sg-dimension"/>' +
      '<circle cx="300" cy="180" r="5" class="sg-point"/></svg>';
    return {markup, summary, index, maxIndex: curve.x.length - 1,
      values: [["Grazing angle", angle, "° from surface"], ["Intensity 1/e depth", depth, "nm"],
        ["Approximate critical angle", data.critical_angle_deg, "°"], ["Wavelength", data.wavelength_angstrom, "Å"]],
      exact: {grazing_angle_deg: angle, intensity_depth_nm: depth, critical_angle_deg: data.critical_angle_deg,
        delta: data.delta, beta: data.beta, wavelength_angstrom: data.wavelength_angstrom}};
  }
  const api = {opticsScene};
  if (typeof module !== "undefined") module.exports = api;
  if (typeof window !== "undefined") window.OpticsSchematic = api;
})();
