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
11. **Single source for asset stamps in exports.** SVG exports embed the
    workbench version; confirm they also record the materials-library dataset
    and backend versions used.
12. **Reference values depend on the OpenBLAS kernel by one ULP.** On an
    AVX-512 Linux container (numpy 2.5.3 and 2.5.4, Python 3.13), the
    `examples/borosilicate-multilayer.json` run gives `mean_energy_out_keV`
    44.189562006464314 instead of the documented 44.18956200646431; with
    `OPENBLAS_CORETYPE=Haswell` or `Sandybridge` it is the documented value,
    with `SkylakeX` it is not. Transmission and thickness scale are unaffected.
    The `np.dot` reductions in `physics.py` (spectrum block) use BLAS `ddot`,
    whose summation order varies by CPU. Options for the owner: document the
    one-ULP platform dependence in `VALIDATION.md`, or switch these
    reductions to an order-fixed sum (`math.fsum`), which may itself move the
    last digit and so needs a deliberate, documented physics PR.
13. **mypy fails when Playwright is installed.** The mypy override for
    `playwright.*` only helps while the package is absent (as in CI). With
    playwright 1.63.0 or 1.64.1 installed, `python -m mypy` reports
    `tools/ui_audit.py:62: No overload variant of "on" of "Page"` for the
    `pageerror` handler's extra default argument. Type the handler as a
    one-argument callable so local runs match CI.

## Done

- 2026-10-28 · Claude routine · `claude/routine-2026-10-28-materials-tests` ·
  Item 10: pure helpers of `static/materials.js` (escaping, number and formula
  formatting, family mapping, search filter, library order, mass percent,
  comparison toggle, shortest half-value layer) exported and covered by 10
  Node tests in `tests/test_materials.cjs`, which CI now runs.
- 2026-10-26 · Claude · `interface-refresh` · WCAG 2.2 AA round 1: contrast
  53→0, text under 12 px 51→0, targets under 24 px 71→0; skip link; gzip and
  versioned caching; content-hash asset stamps; CI interface audit.
- 2026-09-25 · Codex · `materials-library` · PNNL-15870 Rev. 2 import (359
  materials), responsive workspace, plot presentation.
- 2026-09-21 · Claude · `materials-library` · materials library browser and
  landscape view.
