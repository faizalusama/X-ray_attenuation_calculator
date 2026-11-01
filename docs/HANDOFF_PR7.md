# Handoff: PR #7 (release 1.1.0, branch `claude/web-version`)

**Do not merge, tag, release or enable GitHub Pages without the owner's explicit go-ahead.** Read `AGENTS.md` first.

## Done in the last commit, not yet confirmed by CI
Fixes for three CI failures: `.pyodide-cache` excluded from the manifest; transmissions below 1/e judged by τ in `tools/web_parity.py` (disclose this to the owner; see `docs/VALIDATION.md`); export buttons wrap on phones.

Data tables generated in `xray_workbench/scattering_data/` but not yet used: incoherent S (Hubbell 1975, via xraylib) and neutron lengths (Sears 1992, via NIST).

xraylib's elastic form factors and f' are EPDL97, not Hubbell. Only its S function is Hubbell.

## To build (owner approved; keep unpublished)
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
- Check phone width on every tab. Node and Playwright may be absent locally, so read the CI logs.
- Site build needs the Pyodide release (340 MB, downloaded automatically).

## After the owner approves publishing
Enable Pages before merging, merge, check the live site, release v1.1.0, record the DOI, update `date-released`. Then regenerate Fig. S2 footer, `.ris`/`.enw` and the citation paragraph in `paper-2608.15675-corrections`.
