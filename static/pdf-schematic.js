"use strict";
(function () {
  function pdfScene(answer, selected) {
    const data = answer.data, q = data.q, r = data.r_angstrom;
    const index = Math.max(0, Math.min(q.length - 1, Number.isFinite(selected) ? Math.round(selected) : 0));
    const integrand = answer.plots[1].traces[0].y[index];
    const marker = 60 + 140 * (q[index] - q[0]) / (q[q.length - 1] - q[0]);
    const summary = "Input Q sample " + q[index] + " inverse ångström, S(Q) " + data.sq[index] +
      ", modified integrand " + integrand + " inverse ångström. The entire Q interval contributes to every output r sample.";
    const markup = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 300" role="img" aria-labelledby="sg-title sg-desc">' +
      '<title id="sg-title">Finite-range pair distribution transform</title><desc id="sg-desc">' + summary + '</desc>' +
      '<path d="M40 50 H220 V250 H40 Z M380 50 H560 V250 H380 Z" fill="none" stroke="currentColor" stroke-width="2"/>' +
      '<line x1="60" y1="200" x2="200" y2="200" class="sg-axis"/>' +
      '<line x1="' + marker + '" y1="80" x2="' + marker + '" y2="220" class="sg-probe"/>' +
      '<line x1="240" y1="150" x2="360" y2="150" class="sg-incident"/>' +
      '<path d="M345 140 L360 150 L345 160" fill="none" stroke="currentColor" stroke-width="2"/>' +
      '<line x1="400" y1="200" x2="540" y2="200" class="sg-axis"/>' +
      '<circle cx="' + marker + '" cy="200" r="5" class="sg-point"/></svg>';
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
