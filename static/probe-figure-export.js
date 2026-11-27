"use strict";

(() => {
  const svgNS = "http://www.w3.org/2000/svg";
  const presentation = [
    "color", "fill", "fill-opacity", "fill-rule", "stroke", "stroke-opacity", "stroke-width",
    "stroke-dasharray", "stroke-dashoffset", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit",
    "vector-effect", "opacity", "paint-order", "visibility", "display", "shape-rendering",
    "font-family", "font-size", "font-style", "font-weight", "font-stretch", "font-variant",
    "font-variant-numeric", "letter-spacing", "word-spacing", "text-anchor", "text-decoration",
    "dominant-baseline", "alignment-baseline", "baseline-shift", "white-space", "direction",
    "transform", "transform-origin", "transform-box",
  ];

  function svgElement(doc, tag, attributes = {}, text) {
    const element = doc.createElementNS(svgNS, tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function cloneDrawing(svg, computedStyle) {
    const drawing = svg.cloneNode(true);
    const copies = [drawing, ...drawing.querySelectorAll("*")];
    [svg, ...svg.querySelectorAll("*")].forEach((original, index) => {
      const copy = copies[index];
      const style = computedStyle(original);
      const properties = index === 0 ? presentation : new Set([...presentation, ...Array.from(original.style)]);
      copy.removeAttribute("style");
      for (const property of properties) {
        if (property.startsWith("--")) continue;
        const value = style.getPropertyValue(property).replace(/\bcurrentcolor\b/gi, style.color);
        if (!value) continue;
        copy.style.setProperty(property, value);
        if (/\b(?:var\(|currentcolor\b)/i.test(original.getAttribute(property) ?? "")) {
          copy.setAttribute(property, value);
        }
      }
    });
    return drawing;
  }

  function wrap(text, context, width) {
    const lines = [];
    let line = "";
    for (const word of text.match(/\S+/gu) ?? []) {
      const next = line ? line + " " + word : word;
      if (context.measureText(next).width <= width) {
        line = next;
        continue;
      }
      if (line) lines.push(line);
      line = word;
      if (context.measureText(word).width <= width) continue;
      // Keep even an unbroken label inside the figure.
      line = "";
      for (const character of word) {
        if (line && context.measureText(line + character).width > width) {
          lines.push(line);
          line = "";
        }
        line += character;
      }
    }
    if (line) lines.push(line);
    return lines;
  }

  function hasBackground(color) {
    return color && color !== "transparent" && color !== "rgba(0, 0, 0, 0)";
  }

  function exportFigure(svg, title, readout) {
    const doc = svg.ownerDocument;
    const view = doc.defaultView;
    const computedStyle = element => view.getComputedStyle(element);
    const selection = readout.textContent.trim();
    const drawing = cloneDrawing(svg, computedStyle);
    const diagramWidth = svg.viewBox.baseVal.width || svg.clientWidth || 300;
    const diagramHeight = svg.viewBox.baseVal.height || svg.clientHeight || 150;
    const padding = 16;
    const selectionInset = 8;
    const contentWidth = Math.max(320, diagramWidth);
    const width = contentWidth + 2 * padding;
    const fontFamily = "Arial, sans-serif";
    const context = doc.createElement("canvas").getContext("2d");
    context.font = "bold 18px " + fontFamily;
    const titleLines = wrap(title, context, contentWidth);
    context.font = "14px " + fontFamily;
    const selectionLines = wrap(selection, context, contentWidth - 2 * selectionInset);
    const titleStep = 24;
    const selectionStep = 21;
    const diagramX = (width - diagramWidth) / 2;
    const diagramY = padding + titleLines.length * titleStep + 4;
    const selectionY = diagramY + diagramHeight + 8;
    const selectionHeight = selectionLines.length * selectionStep + 2 * selectionInset;
    const height = selectionY + selectionHeight + padding;
    const figure = svgElement(doc, "svg", {
      width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
      "aria-labelledby": "probe-figure-title", "aria-describedby": "probe-figure-desc",
    });
    const description = [selection, svg.getAttribute("aria-label"),
      svg.querySelector(":scope > desc")?.textContent.trim()].filter(Boolean).join("\n");
    figure.append(
      svgElement(doc, "title", {id: "probe-figure-title"}, title),
      svgElement(doc, "desc", {id: "probe-figure-desc"}, description),
    );

    const drawingStyle = computedStyle(svg);
    const readoutStyle = computedStyle(readout);
    const panelStyle = svg.parentElement ? computedStyle(svg.parentElement) : drawingStyle;
    const backgrounds = [];
    for (let ancestor = svg.parentElement; ancestor; ancestor = ancestor.parentElement) {
      const color = computedStyle(ancestor).backgroundColor;
      if (hasBackground(color)) backgrounds.unshift(color);
      if (color?.startsWith("rgb(")) break;
    }
    if (!backgrounds[0]?.startsWith("rgb(")) backgrounds.unshift("#fff");
    for (const fill of backgrounds) figure.append(svgElement(doc, "rect", {width, height, fill}));

    for (const [index, line] of titleLines.entries()) {
      figure.append(svgElement(doc, "text", {
        x: padding, y: padding + 18 + index * titleStep, fill: panelStyle.color,
        "font-family": fontFamily, "font-size": 18, "font-weight": "bold",
      }, line));
    }
    if (hasBackground(drawingStyle.backgroundColor)) {
      figure.append(svgElement(doc, "rect", {
        x: diagramX, y: diagramY, width: diagramWidth, height: diagramHeight, fill: drawingStyle.backgroundColor,
      }));
    }
    // Keep the original viewBox and its aspect ratio inside a native-size viewport.
    for (const [name, value] of Object.entries({
      x: diagramX, y: diagramY, width: diagramWidth, height: diagramHeight, "aria-hidden": "true",
    })) drawing.setAttribute(name, value);
    figure.append(drawing);
    if (selectionLines.length && hasBackground(readoutStyle.backgroundColor)) {
      figure.append(svgElement(doc, "rect", {
        x: padding, y: selectionY, width: contentWidth, height: selectionHeight, fill: readoutStyle.backgroundColor,
      }));
    }
    for (const [index, line] of selectionLines.entries()) {
      figure.append(svgElement(doc, "text", {
        x: padding + selectionInset, y: selectionY + selectionInset + 14 + index * selectionStep,
        fill: readoutStyle.color, "font-family": fontFamily, "font-size": 14,
      }, line));
    }
    return new view.XMLSerializer().serializeToString(figure);
  }

  function createButton({svg, title, readout, filename, download}) {
    const button = svg.ownerDocument.createElement("button");
    button.type = "button";
    button.className = "button outline small";
    button.setAttribute("data-probe-figure-export", "");
    button.textContent = "Save figure as SVG";
    button.addEventListener("click", () => {
      download(exportFigure(svg, title, readout), filename, "image/svg+xml;charset=utf-8");
    });
    return button;
  }

  globalThis.WorkbenchProbeFigureExport = {createButton};
})();
