"use strict";
(function () {
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  function exportCSV(result) {
    const keys = ["row", "name", "mu_mass_cm2_g", "mu_linear_cm_inv", "transmission", "attenuation_length_mm", "error"];
    const cell = value => {
      let text = String(value ?? "");
      if (typeof value === "string" && /^[=+@-]/.test(text.trimStart())) text = "'" + text;
      return '"' + text.replaceAll('"', '""') + '"';
    };
    return [keys, ...result.rows.map(r => keys.map(k => r[k]))].map(row => row.map(cell).join(",")).join("\r\n") + "\r\n";
  }
  function install() {
    const input = document.querySelector("#batch-csv"), status = document.querySelector("#batch-status"), output = document.querySelector("#batch-result");
    let latest = null, revision = 0;
    const invalidate = () => { revision++; latest = null; output.innerHTML = ""; document.querySelector("#batch-export").disabled = true; status.textContent = "Run comparison to refresh results."; };
    input.addEventListener("input", invalidate);
    window.addEventListener("attenuation:update", () => { if (latest && latest.energy_keV !== window.AttenuationWorkbench.referenceEnergy()) invalidate(); });
    document.querySelector("#batch-file").addEventListener("change", async event => {
      const file = event.target.files[0]; if (!file) return;
      try {
        if (file.size > 1000000) throw new Error("CSV must be smaller than 1 MB.");
        input.value = await file.text(); invalidate();
      } catch (exception) { status.textContent = exception.message; }
      finally { event.target.value = ""; }
    });
    document.querySelector("#batch-run").addEventListener("click", async () => {
      invalidate(); const id = revision;
      status.textContent = "Comparing…";
      try {
        const response = await fetch("/api/compare", {method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({csv: input.value, energy_keV: window.AttenuationWorkbench.referenceEnergy()})});
        const result = await response.json(); if (id !== revision) return;
        if (!response.ok) throw new Error(result.detail || "Comparison failed.");
        latest = result;
        status.textContent = result.successes + " of " + result.rows.length + " rows calculated at " + result.energy_keV + " keV.";
        output.innerHTML = '<div class="ts-table-wrap" tabindex="0" role="region" aria-label="Batch comparison"><table class="ts-table"><caption>Independent samples at normal incidence</caption><thead><tr><th>Row / name</th><th>μ/ρ (cm²/g)</th><th>Transmission (%)</th><th>Status</th></tr></thead><tbody>' +
          result.rows.map(r => '<tr><th scope="row">' + r.row + " / " + esc(r.name) + '</th><td>' + (r.error ? "—" : r.mu_mass_cm2_g.toPrecision(6)) +
            '</td><td>' + (r.error ? "—" : (r.transmission * 100).toPrecision(6)) + '</td><td>' + esc(r.error || "OK") + '</td></tr>').join("") + '</tbody></table></div>';
        document.querySelector("#batch-export").disabled = false;
      } catch (exception) { if (id === revision) status.textContent = exception.message; }
    });
    document.querySelector("#batch-export").addEventListener("click", () => {
      if (latest) window.AttenuationWorkbench.download(exportCSV(latest), "attenuation-comparison.csv", "text/csv;charset=utf-8");
    });
  }
  if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", install);
  if (typeof module !== "undefined") module.exports = {exportCSV};
})();
