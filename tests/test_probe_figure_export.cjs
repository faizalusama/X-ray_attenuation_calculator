"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

class InlineStyle {
  properties = new Map();

  setProperty(name, value) {
    this.properties.set(name, value);
  }

  [Symbol.iterator]() {
    return this.properties.keys();
  }
}

class Element extends EventTarget {
  attributes = new Map();
  children = [];
  style = new InlineStyle();
  textContent = "";

  constructor(namespace, tag) {
    super();
    this.ownerDocument = document;
    this.namespace = namespace;
    this.tag = tag;
  }

  append(...children) {
    for (const child of children) child.parentElement = this;
    this.children.push(...children);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  getAttribute(name) {
    return this.attributes.get(name) ?? null;
  }

  removeAttribute(name) {
    this.attributes.delete(name);
    if (name === "style") this.style = new InlineStyle();
  }

  querySelector(selector) {
    assert.equal(selector, ":scope > desc");
    return this.children.find(child => child.tag === "desc");
  }

  querySelectorAll(selector) {
    assert.equal(selector, "*");
    return this.children.flatMap(child => [child, ...child.querySelectorAll("*")]);
  }

  cloneNode(deep) {
    assert.equal(deep, true);
    const clone = new Element(this.namespace, this.tag);
    clone.attributes = new Map(this.attributes);
    clone.style.properties = new Map(this.style.properties);
    clone.textContent = this.textContent;
    clone.append(...this.children.map(child => child.cloneNode(true)));
    return clone;
  }

  click() {
    this.dispatchEvent(new Event("click"));
  }
}

function tree(element) {
  return {
    namespace: element.namespace, tag: element.tag, text: element.textContent,
    attributes: Object.fromEntries(element.attributes), style: Object.fromEntries(element.style.properties),
    children: element.children.map(tree),
  };
}

const svgNS = "http://www.w3.org/2000/svg";
const document = {
  createElement: tag => tag === "canvas"
    ? {getContext: () => ({measureText: value => ({width: Array.from(value).length * 7})})}
    : new Element(null, tag),
  createElementNS: (namespace, tag) => new Element(namespace, tag),
  defaultView: {
    XMLSerializer: class {
      serializeToString(element) { return JSON.stringify(tree(element)); }
    },
    getComputedStyle: element => ({
      color: element.computed?.color ?? "rgb(0, 0, 0)",
      backgroundColor: element.computed?.["background-color"] ?? "rgba(0, 0, 0, 0)",
      getPropertyValue: property => element.computed?.[property] ?? "",
    }),
  },
};

require("../static/probe-figure-export.js");
const {createButton} = globalThis.WorkbenchProbeFigureExport;

function fixture() {
  const panel = document.createElement("section");
  panel.computed = {"background-color": "rgb(255, 255, 255)", color: "rgb(17, 43, 59)"};
  const readout = document.createElement("output");
  readout.textContent = "Observation 1 of 101 · q = 0.0103382 Å⁻¹";
  readout.computed = {...panel.computed, "background-color": "rgb(243, 247, 249)"};
  const graphic = document.createElementNS(svgNS, "svg");
  graphic.setAttribute("viewBox", "-12 6 360 80");
  graphic.setAttribute("preserveAspectRatio", "xMinYMid meet");
  graphic.setAttribute("width", "100%");
  graphic.setAttribute("aria-hidden", "true");
  graphic.setAttribute("aria-label", "Residual vs. q² (Å⁻²).");
  graphic.style.setProperty("max-width", "30rem");
  graphic.style.setProperty("width", "100%");
  graphic.viewBox = {baseVal: {width: 360, height: 80}};
  graphic.clientWidth = 180;
  graphic.clientHeight = 40;
  const marker = document.createElementNS(svgNS, "circle");
  marker.setAttribute("cx", "8.32");
  marker.setAttribute("cy", "40.08");
  marker.setAttribute("r", "4");
  marker.setAttribute("fill", "var(--wb-panel, Canvas)");
  marker.setAttribute("stroke", "currentColor");
  marker.style.setProperty("--marker-stroke", "2px");
  marker.style.setProperty("stroke-width", "var(--marker-stroke)");
  marker.computed = {
    color: "rgb(17, 43, 59)", fill: "rgb(255, 255, 255)", stroke: "currentcolor",
    "stroke-width": "2px", "vector-effect": "non-scaling-stroke",
  };
  const label = document.createElementNS(svgNS, "text");
  label.setAttribute("x", "20");
  label.setAttribute("y", "60");
  label.textContent = "Residual";
  label.computed = {"font-family": "monospace", "font-size": "17px", "font-weight": "700", "text-anchor": "end"};
  graphic.append(marker, label);
  panel.append(readout, graphic);
  const downloads = [];
  const title = "Guinier <fit> & residual";
  const button = createButton({
    svg: graphic, title, readout, filename: "guinier-probe.svg",
    download: (source, filename, mime) => downloads.push({figure: JSON.parse(source), filename, mime}),
  });
  return {panel, readout, graphic, marker, label, downloads, title, button};
}

const children = (element, tag) => element.children.filter(child => child.tag === tag);
const child = (element, tag) => children(element, tag)[0];

test("a native non-submit button exports the current selection without changing the live DOM", () => {
  const {readout, graphic, marker, downloads, title, button} = fixture();
  assert.equal(button.tag, "button");
  assert.equal(button.type, "button");
  assert.equal(button.textContent, "Save figure as SVG");
  assert.equal(button.getAttribute("data-probe-figure-export"), "");
  assert.deepEqual(downloads, []);

  readout.textContent = "Observation 101 of 101 · q = 0.08249249 Å⁻¹ · Measured ln(I) = -0.43102 · " +
    "Fitted ln(I) = -0.42 · Residual (measured − fitted) = -0.01102 <&>";
  marker.setAttribute("cx", "352.00");
  marker.setAttribute("cy", "71.53");
  const before = tree(graphic);
  button.click();
  assert.equal(downloads.length, 1);
  const [{figure, filename, mime}] = downloads;
  assert.equal(filename, "guinier-probe.svg");
  assert.equal(mime, "image/svg+xml;charset=utf-8");
  assert.equal(figure.namespace, svgNS);
  assert.equal(figure.attributes.role, "img");
  assert.equal(child(figure, "title").text, title);
  assert.equal(child(figure, "title").attributes.id, figure.attributes["aria-labelledby"]);
  assert.equal(child(figure, "desc").text, readout.textContent + "\nResidual vs. q² (Å⁻²).");
  assert.equal(child(figure, "desc").attributes.id, figure.attributes["aria-describedby"]);

  const drawing = child(figure, "svg");
  assert.equal(drawing.attributes.viewBox, "-12 6 360 80");
  assert.equal(drawing.attributes.preserveAspectRatio, "xMinYMid meet");
  assert.equal(drawing.attributes.width, "360");
  assert.equal(drawing.attributes.height, "80");
  assert.equal(drawing.style.width, undefined);
  assert.equal(drawing.style["max-width"], undefined);
  assert.equal(drawing.attributes["aria-hidden"], "true");
  const exportedMarker = child(drawing, "circle");
  assert.equal(exportedMarker.attributes.cx, "352.00");
  assert.equal(exportedMarker.attributes.cy, "71.53");
  assert.equal(exportedMarker.attributes.r, "4");

  const [heading, ...lines] = children(figure, "text");
  assert.equal(heading.text, title);
  assert.ok(Number(heading.attributes.y) < Number(drawing.attributes.y));
  assert.ok(lines.length > 1);
  assert.equal(lines.map(line => line.text).join(" "), readout.textContent);
  assert.ok(lines.every(line => line.text.length * 7 + Number(line.attributes.x) < Number(figure.attributes.width)));
  assert.ok(Number(lines[0].attributes.y) > Number(drawing.attributes.y) + Number(drawing.attributes.height));
  assert.ok(Number(lines.at(-1).attributes.y) < Number(figure.attributes.height));
  assert.deepEqual(tree(graphic), before);

  readout.textContent = "Observation 2 of 101";
  marker.setAttribute("cx", "10.27");
  button.click();
  assert.equal(child(downloads[1].figure, "desc").text.split("\n")[0], readout.textContent);
  assert.equal(child(child(downloads[1].figure, "svg"), "circle").attributes.cx, "10.27");
  assert.equal(exportedMarker.attributes.cx, "352.00");
});

test("captures dark and restyled backgrounds, literal paint, strokes and SVG label typography at click time", () => {
  const {panel, readout, graphic, marker, label, downloads, button} = fixture();
  button.click();
  panel.computed = {"background-color": "rgb(21, 37, 47)", color: "rgb(238, 246, 250)"};
  readout.computed = {...panel.computed, "background-color": "rgb(28, 48, 60)"};
  graphic.computed = {"background-color": "rgba(10, 20, 30, 0.4)"};
  marker.computed = {...marker.computed, color: "rgb(140, 227, 215)", fill: "rgb(21, 37, 47)",
    "stroke-width": "3px", "stroke-dasharray": "4px, 3px", "stroke-linejoin": "round"};
  label.computed.fill = "rgb(238, 246, 250)";
  label.computed["font-size"] = "20px";
  button.click();

  const [light, dark] = downloads.map(download => download.figure);
  assert.equal(child(light, "rect").attributes.fill, "rgb(255, 255, 255)");
  assert.deepEqual(children(dark, "rect").map(rect => rect.attributes.fill), [
    "rgb(21, 37, 47)", "rgba(10, 20, 30, 0.4)", "rgb(28, 48, 60)",
  ]);
  assert.ok(children(dark, "text").every(text => text.attributes.fill === "rgb(238, 246, 250)"));
  const drawing = child(dark, "svg");
  const exportedMarker = child(drawing, "circle");
  assert.equal(exportedMarker.attributes.fill, "rgb(21, 37, 47)");
  assert.equal(exportedMarker.attributes.stroke, "rgb(140, 227, 215)");
  assert.deepEqual(exportedMarker.style, {
    color: "rgb(140, 227, 215)", fill: "rgb(21, 37, 47)", stroke: "rgb(140, 227, 215)",
    "stroke-width": "3px", "stroke-dasharray": "4px, 3px", "stroke-linejoin": "round",
    "vector-effect": "non-scaling-stroke",
  });
  assert.deepEqual(child(drawing, "text").style, {
    fill: "rgb(238, 246, 250)", "font-family": "monospace", "font-size": "20px",
    "font-weight": "700", "text-anchor": "end",
  });
  assert.doesNotMatch(JSON.stringify(dark), /var\(|currentcolor/i);
  assert.equal(marker.getAttribute("fill"), "var(--wb-panel, Canvas)");
  assert.equal(marker.style.properties.get("stroke-width"), "var(--marker-stroke)");
});

test("wraps an unbroken readout label without losing any text", () => {
  const {readout, downloads, button} = fixture();
  readout.textContent = "Å".repeat(110);
  button.click();
  const {figure} = downloads[0];
  const [, ...lines] = children(figure, "text");
  assert.ok(lines.length > 1);
  assert.ok(lines.every(line => line.text.length * 7 <= 344));
  assert.equal(lines.map(line => line.text).join(""), readout.textContent);
});
