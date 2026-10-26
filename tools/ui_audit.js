// In-page interface audit for WCAG 2.2 AA failures that can be measured from
// computed styles. Evaluated by tools/ui_audit.py through Playwright; the same
// expression can be pasted into a browser console.
//
// It is deliberately small and transparent rather than complete: it checks
// text contrast (§1.4.3), a 12 px text floor (project rule), target size
// (§2.5.8), accessible names (§4.1.2), heading order, a skip link and the
// document language. It does not replace a full audit or testing with
// assistive technology.
(() => {
  const parse = c => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null; const p = m[1].split(",").map(parseFloat); return {r: p[0], g: p[1], b: p[2], a: p[3] ?? 1}; };
  const lum = ({r, g, b}) => { const f = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  // Text over a background image or gradient cannot be judged from computed
  // colours alone, so it is skipped rather than guessed.
  const bgOf = el => { for (let e = el; e; e = e.parentElement) { const s = getComputedStyle(e); if (s.backgroundImage !== "none") return null; const c = parse(s.backgroundColor); if (c && c.a > 0.5) return c; } return {r: 255, g: 255, b: 255, a: 1}; };
  const visible = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el); return r.width > 0 && r.height > 0 && s.visibility !== "hidden" && s.display !== "none" && !el.closest("[hidden]"); };
  const describe = el => `${el.tagName.toLowerCase()}${typeof el.className === "string" && el.className.trim() ? "." + el.className.trim().split(/\s+/).join(".") : ""} "${(el.getAttribute("aria-label") || el.textContent || "").trim().slice(0, 40)}"`;
  const nameOf = el => (el.getAttribute("aria-label") || (el.labels && [...el.labels].map(l => l.textContent).join(" ")) ||
    (el.getAttribute("aria-labelledby") && document.getElementById(el.getAttribute("aria-labelledby"))?.textContent) || el.title || el.textContent || el.value || "").trim();

  const contrast = [], smallText = [];
  for (const el of document.querySelectorAll("body *")) {
    if (!visible(el) || el.closest("svg") || el.classList.contains("skip-link")) continue;
    if (![...el.childNodes].some(n => n.nodeType === 3 && n.textContent.trim().length > 1)) continue;
    const s = getComputedStyle(el), bg = bgOf(el);
    const size = parseFloat(s.fontSize);
    if (size < 12) smallText.push(`${size}px ${describe(el)}`);
    if (!bg) continue;
    const fg = parse(s.color);
    const ratio = (Math.max(lum(fg), lum(bg)) + 0.05) / (Math.min(lum(fg), lum(bg)) + 0.05);
    const large = size >= 24 || (parseInt(s.fontWeight, 10) >= 700 && size >= 18.66);
    if (ratio < (large ? 3 : 4.5)) contrast.push(`${ratio.toFixed(2)}:1 ${size}px ${describe(el)}`);
  }

  const interactive = [...document.querySelectorAll("button, input:not([type=hidden]), select, textarea, a[href], summary, [role=button], [role=tab]")].filter(visible);
  const unnamed = interactive.filter(el => !nameOf(el)).map(describe);
  const smallTargets = interactive.filter(el => {
    // A checkbox or radio wrapped in its label: the label is the target.
    const target = (el.type === "checkbox" || el.type === "radio") && el.closest("label") ? el.closest("label") : el;
    const r = target.getBoundingClientRect();
    return r.width < 24 || r.height < 24;
  }).map(el => { const r = el.getBoundingClientRect(); return `${Math.round(r.width)}x${Math.round(r.height)} ${describe(el)}`; });

  const headings = [...document.querySelectorAll("h1,h2,h3,h4,h5,h6")].filter(visible).map(h => +h.tagName[1]);
  const headingSkips = headings.filter((h, i) => i && h > headings[i - 1] + 1).length;

  return {
    viewport: `${innerWidth}x${innerHeight}`,
    violations: {
      contrast: [...new Set(contrast)],
      text_under_12px: [...new Set(smallText)],
      targets_under_24px: [...new Set(smallTargets)],
      unnamed_controls: [...new Set(unnamed)],
      heading_level_skips: headingSkips,
      missing_skip_link: document.querySelector("a.skip-link[href^='#']") ? 0 : 1,
      missing_lang: document.documentElement.lang ? 0 : 1,
      multiple_h1: headings.filter(h => h === 1).length > 1 ? 1 : 0,
    },
    stats: {
      interactive_controls: interactive.length,
      dom_nodes: document.querySelectorAll("*").length,
      horizontal_overflow: document.documentElement.scrollWidth > innerWidth + 1,
    },
  };
})()
