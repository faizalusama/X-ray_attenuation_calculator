"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {readFileSync} = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

require("../static/guinier-file-import.js");
const {install} = globalThis.WorkbenchGuinierFileImport;
const csv = "q,I\n0.01,0.992\n0.02,0.970\n0.03,0.935\n0.04,0.887\n0.05,0.828\n";
const limit = 800_000;
const settle = () => new Promise(resolve => setImmediate(resolve));

class FileInput extends EventTarget {
  get files() {
    return this.selected ? [this.selected] : [];
  }

  set value(value) {
    assert.equal(value, "");
    this.selected = null;
  }

  pick(file) {
    if (this.selected === file) return;
    this.selected = file;
    this.dispatchEvent(new Event("change"));
  }
}

function setup() {
  const input = new FileInput();
  const textarea = Object.assign(new EventTarget(), {
    value: "previous data", dataset: {arg: "intensity_data"},
    hasAttribute: () => false,
  });
  const status = {textContent: ""};
  const edits = [];
  textarea.addEventListener("input", event => edits.push([textarea.value, event.bubbles]));
  install(input, textarea, status);
  return {input, textarea, status, edits};
}

test("imports UTF-8 text and reports a markup-like filename as text", async () => {
  const {input, textarea, status, edits} = setup();
  const text = "\uFEFF# échantillon\r\n" + csv.replaceAll("\n", "\r\n");
  const name = "<sample&1>.csv";
  input.pick(new File([text], name, {type: "text/csv"}));
  assert.deepEqual(input.files, []);
  await settle();

  assert.equal(textarea.value, text);
  assert.deepEqual(edits, [[text, true]]);
  assert.equal(status.textContent, "Loaded " + name + ".");
});

test("accepts the inclusive byte and text limit and skips reading an oversized file", async () => {
  const {input, textarea, status, edits} = setup();
  const atLimit = csv + "#".repeat(limit - csv.length);
  input.pick(new File([atLimit], "limit.csv"));
  await settle();
  assert.equal(textarea.value, atLimit);
  assert.equal(edits.length, 1);

  let read = false;
  input.pick({name: "large.csv", size: limit + 1, arrayBuffer() { read = true; }});
  await settle();
  assert.equal(read, false);
  assert.match(status.textContent, /large\.csv.*800,000 bytes/);
  assert.equal(textarea.value, atLimit);
  assert.equal(edits.length, 1);
});

test("checks the decoded text length separately from the reported file size", async () => {
  const {input, textarea, status, edits} = setup();
  const file = new File(["#".repeat(limit + 1)], "long.csv");
  input.pick({name: file.name, size: limit, arrayBuffer: () => file.arrayBuffer()});
  await settle();

  assert.match(status.textContent, /long\.csv.*800,000 characters/);
  assert.equal(textarea.value, "previous data");
  assert.deepEqual(edits, []);
});

test("preserves edits on empty, binary, invalid UTF-8 and unreadable files", async () => {
  const {input, textarea, status, edits} = setup();
  for (const [file, message] of [
    [new File([""], "empty.csv"), /no intensity text/],
    [new File([" \t\r\n"], "blank.txt"), /no intensity text/],
    [new File(["q,I\n\u0000"], "binary.csv", {type: "text/csv"}), /plain-text UTF-8/],
    [new File([Uint8Array.of(0xc3, 0x28)], "invalid.csv"), /plain-text UTF-8/],
    [{name: "<unreadable>.csv", size: 12, async arrayBuffer() { throw new Error("Read failed"); }},
      /Could not read this file/],
  ]) {
    input.pick(file);
    await settle();
    assert.ok(status.textContent.startsWith(file.name));
    assert.match(status.textContent, message);
    assert.equal(textarea.value, "previous data");
    assert.deepEqual(edits, []);
    assert.deepEqual(input.files, []);
  }
});

test("ignores stale successes and failures after the newest file loads", async () => {
  const {input, textarea, status, edits} = setup();
  const oldSuccess = Promise.withResolvers();
  const oldFailure = Promise.withResolvers();
  input.pick({name: "old.txt", size: 3, arrayBuffer: () => oldSuccess.promise});
  input.pick({name: "broken.txt", size: 3, arrayBuffer: () => oldFailure.promise});
  input.pick(new File([csv], "latest.csv"));
  await settle();

  oldFailure.reject(new Error("Late read failure"));
  oldSuccess.resolve(new TextEncoder().encode("old").buffer);
  await settle();
  assert.equal(textarea.value, csv);
  assert.equal(status.textContent, "Loaded latest.csv.");
  assert.deepEqual(edits, [[csv, true]]);
});

test("a failed newest selection prevents a pending older file from applying", async () => {
  const {input, textarea, status, edits} = setup();
  const pending = Promise.withResolvers();
  input.pick({name: "older.csv", size: csv.length, arrayBuffer: () => pending.promise});
  input.pick({name: "latest.csv", size: limit + 1});
  pending.resolve(new TextEncoder().encode(csv).buffer);
  await settle();

  assert.match(status.textContent, /latest\.csv.*800,000 bytes/);
  assert.equal(textarea.value, "previous data");
  assert.deepEqual(edits, []);
});

test("the same file can be picked again while a read is pending and after it succeeds", async () => {
  const {input, status, edits} = setup();
  const pending = Promise.withResolvers();
  const file = new File([csv], "repeat.csv");
  let reads = 0;
  const sameFile = {
    name: file.name, size: file.size,
    arrayBuffer: () => ++reads === 1 ? pending.promise : file.arrayBuffer(),
  };
  input.pick(sameFile);
  input.pick(sameFile);
  await settle();
  pending.resolve(await file.arrayBuffer());
  await settle();
  input.pick(sameFile);
  await settle();

  assert.equal(reads, 3);
  assert.deepEqual(edits, [[csv, true], [csv, true]]);
  assert.equal(status.textContent, "Loaded repeat.csv.");
});

test("a manual edit wins over a pending import and the same file can still be reimported", async () => {
  const {input, textarea, status, edits} = setup();
  const pending = Promise.withResolvers();
  const file = new File([csv], "curve.csv");
  let reads = 0;
  const selection = {
    name: file.name, size: file.size,
    arrayBuffer: () => ++reads === 1 ? pending.promise : file.arrayBuffer(),
  };
  input.pick(selection);
  const manualText = "newer manual edit";
  textarea.value = manualText;
  textarea.dispatchEvent(new Event("input", {bubbles: true}));
  pending.resolve(await file.arrayBuffer());
  await settle();

  assert.equal(textarea.value, manualText);
  assert.equal(status.textContent, "");
  assert.deepEqual(edits, [[manualText, true]]);

  input.pick(selection);
  await settle();
  assert.equal(textarea.value, csv);
  assert.equal(status.textContent, "Loaded curve.csv.");
  assert.deepEqual(edits, [[manualText, true], [csv, true]]);
});

test("paste and file import run the existing controller and give identical native Guinier results", async () => {
  const {input, textarea} = setup();
  const ids = new Map();
  for (const id of ["st-sq-file", "st-demo-pdf", "st-tool", "st-controls", "st-run", "st-export",
    "st-export-csv", "st-status", "st-output", "st-layer-label", "st-layer", "view-tools"]) {
    ids.set("#" + id, Object.assign(new EventTarget(), {dataset: {}}));
  }
  ids.get("#st-tool").value = "guinier";
  const bound = (arg, value) => ({dataset: {arg}, type: "number", value});
  const fields = [textarea, bound("qmin_inv_angstrom", "0.01"), bound("qmax_inv_angstrom", "0.05")];
  const requests = [];
  let scheduled;
  const cwd = path.join(__dirname, "..");
  const document = Object.assign(new EventTarget(), {
    querySelector: selector => ids.get(selector),
    querySelectorAll: selector => selector === '[data-st-tool="guinier"] [data-arg]' ? fields : [],
  });
  const window = Object.assign(new EventTarget(), {
    AttenuationWorkbench: {layers: () => [], referenceEnergy: () => null},
    TechniqueSchematics: {create: () => null},
  });
  // Model propagation from the textarea to the real controller's parent listener.
  textarea.addEventListener("input", event => {
    if (event.bubbles) ids.get("#st-controls").dispatchEvent(new Event("input"));
  });
  vm.runInNewContext(readFileSync(path.join(cwd, "static", "scattering-tools.js"), "utf8"), {
    document, window,
    setTimeout(callback) { scheduled = callback; return callback; },
    clearTimeout(callback) { if (scheduled === callback) scheduled = undefined; },
    async fetch(url, {method, body}) {
      const native = spawnSync("python", ["-B", "-c", [
        "import json, sys",
        "from xray_workbench.scattering_tools import scattering_tools",
        "print(json.dumps(scattering_tools(json.load(sys.stdin)), allow_nan=False))",
      ].join("\n")], {cwd, input: body, encoding: "utf8"});
      assert.ifError(native.error);
      assert.equal(native.status, 0, native.stderr);
      const answer = JSON.parse(native.stdout);
      requests.push({url, method, payload: JSON.parse(body), answer});
      return {ok: true, json: async () => answer};
    },
  });
  document.dispatchEvent(new Event("DOMContentLoaded"));

  textarea.value = csv;
  textarea.dispatchEvent(new Event("input", {bubbles: true}));
  await scheduled();
  textarea.value = "";
  input.pick(new File([csv], "curve.csv"));
  await settle();
  await scheduled();

  assert.match(ids.get("#st-status").textContent, /^Calculation complete\./);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].url, "/api/scattering-tools");
  assert.equal(requests[0].method, "POST");
  assert.equal(requests[0].payload.intensity_data, csv);
  assert.equal(requests[0].answer.data.selected_rows, 5);
  assert.deepEqual(requests[1], requests[0]);
});
