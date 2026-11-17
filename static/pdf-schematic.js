"use strict";
(function () {
  function tracePath(xs, ys, left, right) {
    const low = Math.min(0, ...ys), high = Math.max(0, ...ys);
    const width = xs.at(-1) - xs[0];
    const amplitude = Math.max(Math.abs(low), high) || 1;
    const floor = low / amplitude, height = high / amplitude - floor;
    // Preserve the first, last and extrema per bucket in the small preview.
    const stride = Math.max(1, Math.ceil(xs.length / 140));
    const points = [];
    for (let start = 0; start < xs.length; start += stride) {
      const end = Math.min(xs.length - 1, start + stride - 1);
      let minimum = start, maximum = start;
      for (let i = start + 1; i <= end; i++) {
        if (ys[i] < ys[minimum]) minimum = i;
        if (ys[i] > ys[maximum]) maximum = i;
      }
      for (const i of new Set([start, minimum, maximum, end].sort((a, b) => a - b))) {
        const x = left + (right - left) * (width ? (xs[i] - xs[0]) / width : 0);
        const y = height ? 218 - 136 * (ys[i] / amplitude - floor) / height : 150;
        points.push(x.toFixed(1) + " " + y.toFixed(1));
      }
    }
    return "M" + points.join(" L");
  }
  function pdfScene(answer, selected) {
    const data = answer.data, q = data.q, r = data.r_angstrom;
    const index = Math.max(0, Math.min(q.length - 1, Number.isFinite(selected) ? Math.round(selected) : 0));
    const inputCurve = answer.plots[1].traces[0].y;
    const integrand = inputCurve[index];
    const marker = 60 + 160 * (q[index] - q[0]) / (q[q.length - 1] - q[0]);
    const summary = "Input Q sample " + q[index] + " inverse ångström, S(Q) " + data.sq[index] +
      ", modified integrand " + integrand + " inverse ångström. The entire Q interval contributes to every output r sample.";
    const markup = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300" role="img" aria-labelledby="sg-title sg-desc">' +
      '<title id="sg-title">Finite-range pair distribution transform</title><desc id="sg-desc">' + summary + '</desc>' +
      '<rect x="38" y="56" width="204" height="184" rx="10" class="sg-detector-face"/>' +
      '<rect x="358" y="56" width="204" height="184" rx="10" class="sg-detector-face"/>' +
      '<path class="sg-mini-input" d="' + tracePath(q, inputCurve, 60, 220) + '"/>' +
      '<line x1="' + marker + '" y1="74" x2="' + marker + '" y2="222" class="sg-probe"/>' +
      '<line x1="268" y1="150" x2="330" y2="150" class="sg-incident"/>' +
      '<path d="M318 138 L331 150 L318 162" class="sg-arrow"/>' +
      '<path class="sg-mini-output" d="' + tracePath(r, data.g_inv_angstrom2, 380, 540) + '"/></svg>';
    return {markup, summary, index, maxIndex: q.length - 1,
      values: [["Selected Q", q[index], "Å⁻¹"], ["S(Q)", data.sq[index], ""],
        ["F(Q) M(Q)", integrand, "Å⁻¹"], ["Window", data.window, ""],
        ["Input Q interval", q[0] + " – " + q[q.length - 1], "Å⁻¹"],
        ["Output r interval", r[0] + " – " + r[r.length - 1], "Å"]],
      exact: {q_inv_angstrom: q[index], sq: data.sq[index], modified_integrand_inv_angstrom: integrand,
        window: data.window, q_min_inv_angstrom: q[0], q_max_inv_angstrom: q[q.length - 1],
        r_min_angstrom: r[0], r_max_angstrom: r[r.length - 1]}};
  }
  const api = {pdfScene};
  if (typeof module !== "undefined") module.exports = api;
  if (typeof window !== "undefined") window.PdfSchematic = api;
})();
