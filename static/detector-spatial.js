"use strict";
(function () {
  function detectorSpatial(data, radius, azimuth, elevation) {
    // The lengths share one scale before projection into the viewing plane.
    const extent = Math.max(data.distance_mm / 2, data.outer_radius_mm);
    const halfLength = 100 * (data.distance_mm / 2 / extent);
    const outer = 100 * (data.outer_radius_mm / extent);
    const inner = outer * (data.inner_radius_mm / data.outer_radius_mm);
    const probe = outer * (radius / data.outer_radius_mm);
    const turn = azimuth * Math.PI / 180, tilt = elevation * Math.PI / 180;
    const yawCos = Math.cos(turn), yawSin = Math.sin(turn);
    const pitchCos = Math.cos(tilt), pitchSin = Math.sin(tilt);
    const origin = -halfLength, detector = halfLength;

    function rotate(x, y, z) {
      const across = x * yawCos - z * yawSin;
      const away = x * yawSin + z * yawCos;
      return [across, -y * pitchCos + away * pitchSin];
    }
    const angles = Array.from({length:64}, (_, index) => index * Math.PI / 32);
    const outline = radialDistance => angles.map(angle =>
      rotate(detector, radialDistance * Math.cos(angle), radialDistance * Math.sin(angle)));
    const beamStart = rotate(origin - 58, 0, 0);
    const detectorOutline = outline(outer);
    const fit = [beamStart, ...detectorOutline];
    const xmin = Math.min(...fit.map(point => point[0])), xmax = Math.max(...fit.map(point => point[0]));
    const ymin = Math.min(...fit.map(point => point[1])), ymax = Math.max(...fit.map(point => point[1]));
    const zoom = Math.min(470 / (xmax - xmin), 205 / (ymax - ymin), 4);
    function locate([x, y]) {
      return [300 + zoom * (x - (xmin + xmax) / 2), 150 + zoom * (y - (ymin + ymax) / 2)].map(value => value.toFixed(2));
    }
    function project(x, y, z) {
      return locate(rotate(x, y, z));
    }
    function segment(from, to, kind) {
      return '<line class="' + kind + '" x1="' + from[0] + '" y1="' + from[1] +
        '" x2="' + to[0] + '" y2="' + to[1] + '"/>';
    }
    function ring(outlinePoints, kind) {
      const points = outlinePoints.map(point => locate(point).join(","));
      return '<polygon class="' + kind + '" points="' + points.join(" ") + '"/>';
    }
    function dot(point, kind, size) {
      return '<circle class="' + kind + '" cx="' + point[0] + '" cy="' + point[1] + '" r="' + size + '"/>';
    }
    const sample = project(origin, 0, 0);
    const centre = project(detector, 0, 0);
    const hit = project(detector, probe, 0);
    return segment(locate(beamStart), sample, "sg-incident") +
      segment(sample, centre, "sg-axis") + ring(detectorOutline, "sg-detector-face") +
      (inner > 0 ? ring(outline(inner), "sg-detector-mask") : "") +
      segment(sample, project(detector, outer, 0), "sg-boundary") +
      segment(sample, hit, "sg-probe") +
      dot(centre, "sg-centre", 3) + dot(sample, "sg-sample", 6) + dot(hit, "sg-point", 5);
  }
  if (typeof module !== "undefined") module.exports = {detectorSpatial};
  if (typeof window !== "undefined") window.DetectorSpatial = {detectorSpatial};
})();
