# Handoff: PR #7 (release 1.1.0, branch `claude/web-version`)

**Do not merge, tag, release or enable GitHub Pages without the owner's explicit go-ahead.** Read `AGENTS.md` first.

## Completed on 1 November 2026 (Codex)

The failing test was a missing math.log import; the scattering-table generator
also had two obsolete lint suppressions. These were fixed first in 58cbbab,
with green CI before feature work. All 15 items below were then implemented
in order, with a separate commit after each feature.

826 Python tests and 58 Node tests pass locally; ruff, strict mypy and manifest
verification pass. The expanded UI audit checks populated tools and dialogs at
1440, 768 and 375 px in light and dark themes. Real Pyodide exercises the 16
attenuation configurations plus nine additional API cases.

Reproduce with python tools/native_digest.py. The attenuation digest remains exactly:
4d50d8cd05b54d9c0251e633a5d803b6ced194e1ec3eb4c5134ee02fdb5372e4.

The inherited parity rule for T < 1/e scales relative transmission differences by
max(1,-ln T). Its 1e-12 limit was not changed; see docs/VALIDATION.md. This is an
optical-depth-scaled comparison, not byte equality of browser results.

Both generated data tables are now used. xraylib elastic form factors and f' are
EPDL97, not Hubbell; only its incoherent scattering function is from Hubbell.

## Implemented features (approved development; keep unpublished)
Total scattering tab:
1. Compton vs Q: <S(Q)> with Klein-Nishina recoil, and coherent fraction. Check by integrating dσ/dΩ against Elam coherent and incoherent μ/ρ.
2. Ionic form factors: per-element choice from `xraydb.f0_ions`; f0(0) should be about Z - charge.
3. Capillary absorption vs 2θ: numerical integral over the disc. Check against 1D integrals at 2θ = 0 and 180°. Add polarization factors.
4. Neutron pair weights beside X-ray weights; handle <b> near 0.
5. f' and f'' near an edge (Chantler).

General:
6. Share link: configuration in the URL fragment, validated with `cleanProject`, gzip if available.
7. Transmission vs thickness plot, Materials tab, from `layers[i].reference.mu_linear_cm_inv`.
8. Batch CSV `name,basis,density_g_cm3,thickness_mm,composition` (`SiO2:80;Na2O:20`), `POST /api/compare`, per-row errors, CSV export.
9. "Cite this" dialog with BibTeX, matching `CITATION.cff` by test.

New "Scattering tools" tab, proposed by Claude:
10. δ, β, SLD, critical angle, penetration depth vs angle. Check Si at 8.048 keV: δ 7.57e-6, αc about 0.223°.
11. Multilayer reflectivity: Parratt recursion with Névot-Croce roughness.
12. SLD contrast between layers and two-phase Porod invariant.
13. 2θ/Q/d/s converter, source-to-source conversion, Scherrer size.
14. Powder peaks from lattice and atoms, with |F|², LP and B. Check Si (111) at 28.44° with Cu Kα1, and F(200) = 0.
15. S(Q) to G(r) with optional Lorch window. Check against the analytic damped-sine case.

## Conventions
- Add POST routes to both `server.py` and `browser.py`, and to the `ROUTE` regex in `web/shim.js`; test that the two answer identically.
- Add any new file names to the `/api/download` allowlist in `server.py`.
- Plots use `displayModeBar: false`, because Plotly's toolbar can upload charts.
- After CSS or HTML edits, run `tools/build_dark_css.py`, then `tools/stamp_assets.py`, then `tools/manifest.py write`.
- The native digest must stay `4d50d8cd05b54d9c…`. Do not invent data or citations; verify DOIs. Commit as `faizalusama`.
- Node and Playwright are now installed locally. Check CI as well as local results.
- Site build needs the Pyodide release (340 MB, downloaded automatically).

## Review and limits

New routes: /api/compare, /api/citation and /api/scattering-tools in both runtimes.
See docs/SCATTERING_TOOLS.md and docs/TOTAL_SCATTERING.md for formats and formulas.
No experimental validation or fitting is claimed. Disc integration is withheld
above muR=30. Powder symmetry is not expanded: supply the complete unit cell.
The Fourier tool accepts normalized S(Q), not raw intensities.

Preview from this worktree with python launch.py. For an isolated browser build:
python tools/build_web.py --output _site/review
python tools/web_parity.py --site _site/review
Serve _site/review locally to inspect the browser distribution.

Publishing gates remain closed. Before an approved release, review/remove the
unreleased markers in README and citation.py, supply the actual release date,
and adjust the citation-status test. Never invent a version DOI.

## After the owner approves publishing
Enable Pages before merging, merge, check the live site, release v1.1.0, record the DOI, update `date-released`. Then regenerate Fig. S2 footer, `.ris`/`.enw` and the citation paragraph in `paper-2608.15675-corrections`.
