# Improvement backlog

The shared queue for engineering and interface work, used by the owner, Codex,
Claude Code and the weekly improvement routine. Research items (XCOM adapter,
correlated uncertainty, validation datasets) live in `ROADMAP.md`; this file is
for the application itself.

**How to use it:** take the highest unclaimed item you can finish in one pull
request. Claim it under *In progress* on your branch. When the PR is opened,
move the item to *Done* with the PR link. Add new findings at the end of the
right section, with the evidence that motivated them.

## In progress

_(agent · branch · date · item)_

## Ready

### Accessibility and readability
1. **Dark theme.** The page declares `color-scheme: light` and ~100 KB of CSS
   hard-codes colours. First move colours to CSS custom properties (no visual
   change, verified by screenshots), then add a `prefers-color-scheme: dark`
   palette that passes the interface audit. Plotly figures need matching
   template colours.
2. **Fold `static/readability.css` into the base stylesheets.** It is a
   deliberate override layer from the October audit. Move each rule to its
   source and delete it there; the audit must stay at zero.
3. **Screen-reader pass on the plots.** Each Plotly figure needs a text
   summary of what it shows at the reference energy, as the materials library
   already provides.

### Performance
4. **Render off-screen plots lazily.** All 11 plots draw at load; render a plot
   when it scrolls into view (IntersectionObserver). Measure DOMContentLoaded
   and time-to-first-plot before and after.
5. **Consider a partial Plotly bundle.** The full 4.3 MB bundle (1.3 MB gzipped)
   is loaded; only scatter and heatmap traces are used. Licence and SVG export
   must be checked before swapping.

### Usability
6. **First-run guidance.** A short, dismissible tour of the three areas (system
   editor, plots, materials library) for new users; remembered locally.
7. **Keyboard shortcuts** for frequent actions (refresh, add layer, open
   library), listed in a help dialog opened with `?`.
8. **Undo for destructive edits** (remove layer, remove component).
9. **Unit-aware inputs:** accept `2 mm`, `500 µm` or `0.2 cm` typed into a
   thickness field and convert explicitly.

### Engineering
10. **Frontend tests for `static/materials.js`.** It has no Node unit tests;
    extract pure helpers (filtering, formatting) and test them like
    `tests/test_frontend.cjs`.
11. **Export provenance, remaining slices.** Audit 2026-11-02: CSV, JSON and
    dashboard SVG exports already record the backend identifier, dataset and
    dataset version plus the workbench version and configuration SHA-256.
    Slice 1 (Done) records the materials-library origin of each layer in the
    configuration. Still to do: (a) show the library origin in the visible
    SVG footer and the provenance panel, not only in metadata; (b) add
    per-layer `layer_N_library_material_id` columns to the CSV, which carries
    no configuration; (c) `exportedSVG` in `static/app.js` is no longer called
    by the app (only by `tests/test_frontend.cjs`) and its footer labels the
    dataset version "XrayDB" even for the xraylib backend; remove it or route
    the dashboard export through it.
12. **Exact-equality physics gate is platform-sensitive.** On the routine's
    Linux container (Python 3.13.18, numpy 2.5.3 and 2.5.4, scipy 1.18.1,
    Intel Xeon with AVX-512), unmodified `interface-refresh` gives
    `mean_energy_out_keV` 44.189562006464314, one unit in the last place above
    the 44.18956200646431 recorded in `docs/VALIDATION.md`; transmission and
    target scale match exactly. Consistent with the per-platform differences
    VALIDATION.md already documents. The owner should decide whether the
    routine's gate compares against a same-machine baseline or a ULP bound.
13. **`mypy` fails on `tools/ui_audit.py` when Playwright is installed.**
    With playwright 1.63.0 present, line 62 (`page.on("pageerror", lambda exc,
    errors=errors: ...)`) is rejected: the two-argument lambda matches no
    `Page.on` overload. CI's mypy job does not install Playwright, so it stays
    green; a contributor with Playwright installed sees a red `python -m mypy`.
    Bind `errors` with a one-argument closure or `functools.partial` instead.

## Done

- 2026-11-02 · Claude routine · `claude/routine-2026-11-02-library-origin` ·
  item 11 slice 1: layers added from the materials library carry
  `library: {material_id, dataset, dataset_sha256, density_status}` through
  project save/load, the calculation route and every export that embeds the
  configuration; dropped when composition or density is edited.

- 2026-10-26 · Claude · `interface-refresh` · WCAG 2.2 AA round 1: contrast
  53→0, text under 12 px 51→0, targets under 24 px 71→0; skip link; gzip and
  versioned caching; content-hash asset stamps; CI interface audit.
- 2026-09-25 · Codex · `materials-library` · PNNL-15870 Rev. 2 import (359
  materials), responsive workspace, plot presentation.
- 2026-09-21 · Claude · `materials-library` · materials library browser and
  landscape view.
