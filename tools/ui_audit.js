// In-page interface audit for WCAG 2.2 AA failures that can be measured from
// computed styles. Evaluated by tools/ui_audit.py through Playwright; the same
// expression can be pasted into a browser console.
//
// It is deliberately small and transparent rather than complete: it checks
// text contrast (§1.4.3), a 12 px text floor (project rule), target size
// (§2.5.8), accessible names (§4.1.2), heading order, a skip link and the
// document language, and text descriptions for the scientific plots. It does
// not replace a full audit or testing with assistive technology. Unmeasured
// contrast (gradients, transparency and SVG) is reported explicitly below.
(() => {
  const parse = c => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null; const p = m[1].split(",").map(parseFloat); return {r: p[0], g: p[1], b: p[2], a: p[3] ?? 1}; };
  const lum = ({r, g, b}) => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  // Text over a background image or gradient cannot be judged from computed
  // colours alone, so it is skipped rather than guessed.
  const bgOf = el => { for (let e = el; e; e = e.parentElement) { const s = getComputedStyle(e); if (s.backgroundImage !== "none" || Number(s.opacity) < 1) return null; const c = parse(s.backgroundColor); if (c && c.a > 0 && c.a < 1) return null; if (c && c.a === 1) return c; } return {r: 255, g: 255, b: 255, a: 1}; };
  // showModal() makes the underlying page inert without adding an [inert]
  // attribute. Audit the active dialog, not the dimmed, inaccessible page.
  const modal = document.querySelector("dialog:modal"), scope = modal || document.body;
  const visible = el => {
    const r = el.getBoundingClientRect(), s = getComputedStyle(el);
    if (!r.width || !r.height || s.visibility === "hidden" || s.display === "none" || el.closest("[hidden], [inert]")) return false;
    for (let parent = el.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === "DETAILS" && !parent.open && !parent.querySelector(":scope > summary")?.contains(el)) return false;
    }
    return true;
  };
  const describe = el => `${el.tagName.toLowerCase()}${typeof el.className === "string" && el.className.trim() ? "." + el.className.trim().split(/\s+/).join(".") : ""} "${(el.getAttribute("aria-label") || el.textContent || "").trim().slice(0, 40)}"`;
  const references = (el, attribute) => (el.getAttribute(attribute) || "").trim().split(/\s+/).filter(Boolean).map(id => document.getElementById(id));
  const nameOf = el => (references(el, "aria-labelledby").map(node => node?.textContent || "").join(" ") || el.getAttribute("aria-label") ||
    (el.labels && [...el.labels].map(l => l.textContent).join(" ")) || el.title || el.textContent ||
    (["button", "submit", "reset"].includes(el.type) ? el.value : "") || "").trim();

  const contrast = [], smallText = [], unmeasuredContrast = [];
  for (const el of scope.querySelectorAll("*")) {
    if (!visible(el) || el.closest("svg") || el.classList.contains("skip-link")) continue;
    if (![...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim().length > 1)) continue;
    const s = getComputedStyle(el), bg = bgOf(el);
    const size = parseFloat(s.fontSize);
    if (size < 12) smallText.push(`${size}px ${describe(el)}`);
    const fg = parse(s.color);
    if (!bg || !fg || fg.a < 1) { unmeasuredContrast.push(describe(el)); continue; }
    const ratio = (Math.max(lum(fg), lum(bg)) + 0.05) / (Math.min(lum(fg), lum(bg)) + 0.05);
    const large = size >= 24 || (parseInt(s.fontWeight, 10) >= 700 && size >= 18.66);
    if (ratio < (large ? 3 : 4.5)) contrast.push(`${ratio.toFixed(2)}:1 ${size}px ${describe(el)}`);
  }

  const interactive = [...scope.querySelectorAll("button, input:not([type=hidden]), select, textarea, a[href], summary, [role=button], [role=tab]")].filter(visible);
  const unnamed = interactive.filter(el => !nameOf(el)).map(describe);
  const smallTargets = interactive.filter(el => {
    // WCAG 2.5.8 exempts links embedded in prose. A standalone reference link
    // or navigation link still needs the normal target size.
    if (el.matches("a[href]") && getComputedStyle(el).display === "inline" &&
      [...el.parentElement.childNodes].some(n => n.nodeType === 3 && n.textContent.trim().length > 1)) return false;
    // A checkbox or radio wrapped in its label: the label is the target.
    const target = (el.type === "checkbox" || el.type === "radio") && el.closest("label") ? el.closest("label") : el;
    const r = target.getBoundingClientRect();
    return r.width < 24 || r.height < 24;
  }).map(el => { const r = el.getBoundingClientRect(); return `${Math.round(r.width)}x${Math.round(r.height)} ${describe(el)}`; });

  const headings = [...scope.querySelectorAll("h1,h2,h3,h4,h5,h6")].filter(visible).map(h => +h.tagName[1]);
  const headingSkips = headings.filter((h, i) => i && h > headings[i - 1] + 1).length;
  const plots = [...scope.querySelectorAll(".scientific-plot")].filter(visible);
  const missingSummaries = plots.filter(plot => {
    const nodes = references(plot, "aria-describedby");
    return !nodes.length || nodes.some(node => !node || !node.textContent.trim() || plot.contains(node) || !visible(node) || node.closest('[aria-hidden="true"]'));
  }).map(describe);

  return {
    viewport: `${innerWidth}x${innerHeight}`,
    violations: {
      contrast: [...new Set(contrast)],
      text_under_12px: [...new Set(smallText)],
      targets_under_24px: [...new Set(smallTargets)],
      unnamed_controls: [...new Set(unnamed)],
      missing_plot_descriptions: missingSummaries,
      heading_level_skips: headingSkips,
      missing_skip_link: document.querySelector("a.skip-link[href^='#']") ? 0 : 1,
      missing_lang: document.documentElement.lang ? 0 : 1,
      multiple_h1: headings.filter(h => h === 1).length > 1 ? 1 : 0,
    },
    stats: {
      interactive_controls: interactive.length,
      described_plots: plots.length - missingSummaries.length,
      scope: modal ? "modal dialog" : "page",
      unmeasured_text_contrast: [...new Set(unmeasuredContrast)],
      svg_text_not_audited: [...scope.querySelectorAll("svg text")].filter(visible).length,
      dom_nodes: document.querySelectorAll("*").length,
      horizontal_overflow: document.documentElement.scrollWidth > innerWidth + 1,
    },
  };
})()
