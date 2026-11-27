"use strict";
(function () {
  // Match the character limit in xray_workbench/guinier.py and bound file reads.
  const maxSize = 800_000;

  function install(input, textarea, status) {
    let revision = 0;
    textarea.addEventListener("input", () => { revision++; });
    input.addEventListener("change", async () => {
      const id = ++revision;
      const file = input.files[0];
      if (!file) return;
      input.value = "";
      const fail = message => {
        if (id === revision) status.textContent = file.name + " — " + message;
      };
      if (file.size > maxSize) {
        fail("Choose a file of at most 800,000 bytes.");
        return;
      }

      let bytes;
      try {
        bytes = await file.arrayBuffer();
      } catch {
        fail("Could not read this file.");
        return;
      }
      if (id !== revision) return;

      let text;
      try {
        text = new TextDecoder("utf-8", {fatal: true, ignoreBOM: true}).decode(bytes);
      } catch {
        fail("Choose a plain-text UTF-8 CSV file.");
        return;
      }
      if (text.length > maxSize) {
        fail("Intensity data must be at most 800,000 characters.");
        return;
      }
      if (/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]/.test(text)) {
        fail("Choose a plain-text UTF-8 CSV file.");
        return;
      }
      if (!text.trim()) {
        fail("The file contains no intensity text.");
        return;
      }

      textarea.value = text;
      status.textContent = "Loaded " + file.name + ".";
      textarea.dispatchEvent(new Event("input", {bubbles: true}));
    });
  }

  globalThis.WorkbenchGuinierFileImport = {install};
})();
