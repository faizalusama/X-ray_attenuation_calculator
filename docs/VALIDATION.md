# Validation record — versions 1.0.0 to 1.1.0

Numbering note: this project was developed as `0.2.0` and renumbered to `1.0.0`
for its first stable release. Entries below written before the renumber refer to
the same code.

Verified locally on **20 September 2026**, using Windows and Python **3.12.14**. This record concerns numerical implementation and application behavior, not certification of experimental accuracy.

## Automated verification

**110 Python tests and 14 additional unittest subtests passed.** The full 108-test suite passed before the final export compatibility change; the affected API suite then passed all six tests, including its two new native-download cases. The physics and CLI code was unchanged by that export change.

**14 JavaScript tests passed**, covering input normalization, spectrum parsing, project round trips, logarithmic plots, interval gaps, SVG metadata, and the live-update scheduler. The scheduler tests check debounce, coalescing edits during a running calculation, manual refresh, and recovery after errors. JavaScript syntax checking also passed.

Commands:

```powershell
python -m pytest -p no:cacheprovider
node tests/test_frontend.cjs
node --check static/app.js
```

The Python suite emits two upstream test-client deprecation warnings (Starlette/httpx and AnyIO). These do not fail the tests. On this host, pytest's temporary-folder tests required execution outside the restrictive Windows sandbox; no filesystem permissions or access controls were changed. The application itself was exercised with its normal local runtime.

## Independent numerical references

Twelve total mass attenuation values for Al, Fe, Pb, and water at 20, 50, and 100 keV were compared with independently transcribed NIST tables. The maximum absolute relative difference was **0.031245%**, below the **1% benchmark agreement threshold**. Full values and package versions are recorded in [benchmark-results.json](benchmark-results.json); primary source links are in [SCIENTIFIC_METHODS.md](SCIENTIFIC_METHODS.md).

This is agreement with selected rounded reference values, not a claim of that accuracy at arbitrary energies or for real specimens. Databases can share theoretical or tabulated inputs. Absorption-edge chemistry and application-specific geometry still require experimental validation.

The original double mass weighting is covered by a mixture regression. Additional tests cover composition bases and molecular masses, grouping/hydrate formulas, density and porosity, thickness units, oblique incidence, equivalent/reordered layers, zero thickness, direct reference evaluation, channel sums, spectra, lognormal uncertainty, and finite outputs at extreme allowed inputs.

All **543 supported H–Cf absorption edges at or above 1 keV** were checked against the current Elam coefficient knots. A relative bracket of ±10⁻⁴ encompasses the difference between rounded coefficient-knot locations and reported atomic edge energies. Coefficients are not shifted to force agreement. Explicit near-edge comparisons and reference evaluations test this distinction.

## Browser verification

The local application was exercised in the Codex embedded browser. Checks included:

- Initial automatic calculation and successful loading of a two-layer project.
- Typing thickness from 1 to 2 mm changed reference transmission from 84.27% to 71.01% automatically; moving the thickness slider caused a further automatic update.
- Invalid chemical formulas retained the previous plot under an explicit pending/error label and disabled exports; correcting the input restored automatic calculation.
- A project containing spectrum and uncertainty settings recalculated after import without a button press.
- Switching spectrum weighting from photon fluence to energy fluence updated transmission from 78.03% to 86.20% automatically for the supplied example.
- Materials, spectrum, uncertainty, and interaction-channel views rendered computed results. Logarithmic plotting omits true zero values rather than substituting invented positive values.
- Native SVG attachment download completed. The shared attachment endpoint preserves exact Unicode/content and rejects unexpected filenames. CSV and JSON use that same download mechanism; their data/provenance construction is covered by code inspection and the project/API tests.
- The desktop layout was visually inspected. A 720-pixel viewport used a single column without horizontal overflow, with a plot sized to its available width. Temporary viewport overrides were reset.

Exports use native HTTP attachments because the embedded browser did not report completion for the initial blob-URL download. Export content is returned locally to the same browser and is not stored by the server.

## Reproducible example

The saved two-layer example has an illustrative borosilicate glass plus an aluminium support. Its photon-weighted spectral transmission is **0.7802626916899833**, and the mean photon energy changes from **40** to **44.18956200646431 keV**. A common thickness multiplier of **11.417978438587083** reaches reference transmission **0.10000000000000002** within floating-point precision.

All five code cells in the new example notebook executed sequentially and its assertions passed. The batch command generated the complete finite JSON result under `examples/verified-example-result.json`. The dependency lock's installed distributions and active dependency constraints were audited and satisfied in the tested environment.

## Engineering foundation pass — 20 September 2026

A second verification run on the same date, on Windows with Python **3.14.6**
(the archive's bundled `.venv`), covering the packaging and quality work rather
than the physics. The physics results were unchanged by it.

**Re-verified:** 110 Python tests and 14 unittest subtests pass. The example
notebook executes all five code cells with no error output. `ruff check` and
`mypy --strict` both report no issues across the package, launcher and tools.
A wheel builds and carries its own browser assets; both branches of the static
asset lookup were exercised — the source tree by the test suite, the installed
layout by importing the unpacked wheel from an unrelated working directory.

**Corrected during this pass:**

- `xray_workbench/__init__.py` declared `__version__ = "1.0.0"` while
  `pyproject.toml`, the health endpoint, the launcher's reuse check and the
  result provenance all reported `0.2.0`. The version now has one definition
  that the others read.
- The shipped `MANIFEST.sha256` did **not** match its own archive. Three files
  differed: `static/app.js`, `static/index.html` and `xray_workbench/server.py`
  — the files changed by the export-compatibility work described above, which
  landed after the manifest was written. The manifest is now generated by
  `tools/manifest.py` and verified in CI.
- Thirty-two rejection tests asserted only that `ValueError` was raised. Because
  the engine signals every invalid input with that one type, each could have
  passed because an unrelated guard fired. All now assert the message that
  identifies the specific guard; all pass unchanged, which confirms the
  originally intended guard was in fact the one firing in every case.
- Two `zip()` calls over sequences built in a single pass now use `strict=True`,
  so a length mismatch raises instead of silently truncating a mixture.

**Not verified on this host:** the 14 JavaScript tests and `node --check`.
Node.js is not installed on the verifying machine. That suite is unchanged
since the run recorded above and now executes in CI, but it was **not**
re-executed as part of this pass.

## Backend independence — 20 September 2026

The pluggable backend contract was added and the xraylib backend implemented.
The comparison produced a result worth recording prominently.

**XrayDB/Elam and xraylib return the same coefficients to within floating-point
rounding.** Across 99 element/energy samples (H, C, O, Al, Si, Fe, Cu, Ba, W,
Pb, U at 5, 10, 17.3, 30, 60, 100, 200, 500 and 700 keV), xraylib 4.3.0's
`CS_Photo`, `CS_Rayl` and `CS_Compt` were compared with XrayDB 4.5.8's
`mu_elam`, channel by channel.

*Correction, 21 September 2026.* This entry originally said "bit-identical" on
every platform. That was measured on Windows only. CI then measured the maximum
relative difference per platform:

| Platform | Max relative difference |
|---|---|
| Windows, Python 3.12 / 3.13 / 3.14 | 0 (bit-identical) |
| Ubuntu, Python 3.13 / 3.14 | 0 (bit-identical) |
| Ubuntu, Python 3.12 | 1.9 × 10⁻¹⁶ |
| macOS, Python 3.14 | 1.9 × 10⁻¹⁵ |

The largest is about eight units in the last place of a float64: differently
compiled maths libraries evaluating the same splines. Genuinely different
evaluated data would differ by ~10⁻³ or more. The conclusion is unchanged —
the two packages expose one evaluated dataset through two interfaces — and the
test now bounds the difference at 10⁻⁹ relative instead of demanding equality.

Consequently **this project still has no independent cross-database
validation.** Agreement between these two backends measures nothing. The
xraylib backend declares `shares_data_with = ("elam",)`,
`backends/compare.py` excludes the pair from legitimate cross-checks, and
`tests/test_backend_equivalence.py` asserts agreement within 10⁻⁹ relative so that any
future upstream divergence surfaces as a test failure rather than a silent
change in published numbers.

The independent NIST reference comparison recorded above (twelve values, max
0.031245% difference) remains the only genuine external check, and it covers a
small off-edge subset only.

Two further details were pinned by measurement rather than assumption:
xraylib's spline domain ends at **800.026475 keV** (it raises rather than
extrapolating), and the two sources report the Ba K edge at 37.4410 keV
(XrayDB) versus 37.4406 keV (xraylib) — a 0.4 eV label difference despite
identical coefficients, so edge labels are not interchangeable.

## Remaining release work

Version 1.0.0 was released on 21 September 2026. Experimental validation and independent scientific review remain outstanding. Transport of scattered/fluorescent photons, calibrated source generation, diffraction/near-edge fine structure, detector response, and broader correlated uncertainty are explicitly outside this implemented model; see [ROADMAP.md](ROADMAP.md).

The licence decision is now made: the project is released under the MIT License, recorded in [LICENSE](../LICENSE), `CITATION.cff` and the package metadata, with third-party terms listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Experimental validation, independent scientific review and installation testing on the operating systems now covered by CI remain outstanding.

## Interactive dashboard pass — 21 September 2026

On Windows with Python 3.12.14, the complete Python suite passed **123 tests and 14 subtests**, with **one optional xraylib test skipped** and two upstream deprecation warnings. No physics implementation changed in this dashboard pass. The two JavaScript suites passed **23 tests** covering the existing frontend and dashboard units, attenuation lengths, target thickness, heatmap values, secondary axes, uncertainty bands, preference validation, backend preservation and the eleven-plot default.

Browser checks exercised live thickness changes, dimension-preserving unit changes, secondary axes, linked zoom, gesture locking, remembered settings, Ideas, native SVG download and responsive plot sizing. An exported SVG was parsed and checked for dimensions and configuration metadata. This is application verification, not a new scientific accuracy claim.

Additional commands: `node tests/test_dashboard.cjs` and `node --check static/dashboard.js`.

Per-plot controls follow-up: all 25 JavaScript tests passed, including independent scale/typography restoration and linear/logarithmic linked-energy range conversion. Browser verification confirmed different font sizes and y-axis scales persist independently after reload, bold SVG axis titles render, and the page has no horizontal overflow or console errors.

Typography follow-up: enlarged headings and plot labels, compact figure margins and logarithmic power labels were checked in the browser. The transmission plot rendered spaced decade labels without horizontal page overflow or console errors. All 25 JavaScript tests passed.

## Version 1.0.1 — 31 October 2026

Documentation, metadata and input warnings only. `xray_workbench/physics.py` gained two warning messages for mistyped formulas; no calculation line changed.

**Numbers unchanged.** Every calculated value was compared between the `v1.0.0` tag and this version for 16 configurations: the four glasses of arXiv:2608.15675 with uncertainty and a spectrum, all ten presets and a four-layer stack on a 1500-point 1–800 keV sweep, plus the saved example. The serialized results are byte-identical (the same SHA-256, `4d50d8cd05b54d9c…`, from both). Provenance and warning text were excluded from that comparison because they carry the version and timestamp. Regenerating `examples/verified-example-result.json` changed only its version and timestamp.

**Checks on this version.** 156 Python tests and 14 subtests passed on Windows with Python 3.14.6. `ruff check` and `mypy --strict` report no issues. The JavaScript suites need Node.js, which this machine lacks; they run in CI. The renamed interface was checked in a browser at desktop and 375-pixel widths, with no horizontal overflow or console errors.

**Independent comparison.** This was a one-off run on a clean copy of the `v1.0.0` tag; it is not part of the test suite.

- Those four glasses at seven energies from 5 to 100 keV matched a hand calculation using separately typed atomic masses and raw `mu_elam` values, within 6.5 × 10⁻⁶ relative.
- Their mass attenuation coefficients, built from NIST's elemental tables (SRD 126), agreed within 0.04% at 10, 15 and 20 keV.
- NIST's own tabulated values for borosilicate (Pyrex) and lead glass agreed within 0.03% at seven energies from 5 to 100 keV, using NIST's Table 2 compositions.

Agreement this close suggests the Elam and NIST tabulations share underlying theory. It checks units, composition handling and the mixture rule. It is not experimental validation.

**Corrected in this version.**

- The README said "No new DOI, release, repository commit, or publication has been made", which was already false when 1.0.0 was published.
- This record said the backend test asserts exact equality; it allows 10⁻⁹ relative, as the correction above explains.
- The package classifier said Alpha although 1.0.0 was described as stable.
- The 1.0.0 Zenodo record had no ORCID, affiliation or keywords. `.zenodo.json` and `CITATION.cff` now carry them.
- The software had several names, including "Attenuation Lab" in the interface. It now has one title, and `tests/test_citation.py` keeps the README, interface and citation files in agreement. The Python module `xray_workbench`, the `xray-workbench` command and the result key `workbench_version` are unchanged, so existing scripts and saved projects keep working.

`legacy/` is unchanged.

## Version 1.1.0 — 31 October 2026

Adds a browser version. The same `xray_workbench` package runs inside the browser through Pyodide 314.0.7; nothing is reimplemented. `physics.py`, `results.py` and the backends are unchanged from 1.0.1. New files are `xray_workbench/browser.py`, `web/`, `tools/build_web.py` and `tools/web_parity.py`.

**Native numbers unchanged.** For the 16 configurations described under 1.0.1, the serialized native results are byte-identical across `v1.0.0`, `v1.0.1` and this version (the same SHA-256, `4d50d8cd05b54d9c…`).

**Browser against native.** The same 16 configurations were run in the browser and compared with the native engine on Windows (Python 3.14.6, NumPy 2.5.3). The browser used Python 3.14.2 and NumPy 2.4.6.

- Of 247,930 numbers, 242,569 are identical and 5,361 differ. The browser version is therefore **not** byte-identical to the native engine.
- The largest relative difference is 5.7 × 10⁻¹⁴, on a transmission of 7.9 × 10⁻⁹⁴. Differences of this size are consistent with different maths libraries and NumPy builds rounding the last bits differently.
- The documented reference values are identical: spectral transmission 0.7802626916899833 and thickness multiplier 11.417978438587083. So is the 14 keV transmission of each glass of arXiv:2608.15675, to every digit.

`tools/web_parity.py` repeats this comparison in CI, on Linux, and fails above 10⁻¹² relative. That limit was set after the measurement above, at about 20 times its maximum.

**Loading.** A first visit transfers about 30 MB. With the page visible on one desktop computer, five loads took between 7.5 and 17.5 s, including 13.3 s with an empty cache. Two other loads took about 60 s: one with the page in a hidden pane, and one under phone-size emulation. That second stall was in the library-loading step and did not recur when repeated, and its cause was not established. The page reports its progress and says so if loading takes more than 30 s. A calculation then took 0.05–0.3 s.

**Interface.** Checked in a Chromium-based browser at desktop and 375-pixel widths: all eleven plots render, a misspelled formula gives the same message as the local server, the formula warnings appear, and project, SVG, CSV and JSON exports reach the browser's download path with the expected content. No horizontal overflow or console errors appeared.

**Not checked.** Firefox, Safari and real phones. The local JavaScript suites need Node.js, which this machine lacks; they run in CI. As before, this is a check of the numerical implementation, not experimental validation.

**Other new checks.** `tests/test_web.py` holds the browser routes to the same answers as the local server and checks that the site runs the same `physics.py`. Results from the browser record their runtime in `provenance.runtime`.

**Continuous integration.** On the GitHub runner, the same comparison finds 154 differences, the largest 1.43 × 10⁻¹⁴ relative, with the engine loading in 9.4 s.

**Additions to the input and export interface.** Atomic percent, areal density, X-ray sources, CSV and PNG exports, dark mode and undo were added on this version without changing how any existing input is calculated. The native SHA-256 above was recomputed after these changes.

- Atomic percent is treated as mole fractions of single elements. A test checks that it gives the same coefficients as the mole calculation, and mass fractions matching ones worked out from atomic masses.
- Areal density sets thickness to areal density / effective density. Tests check that optical depth does not depend on density, that 27 mg/cm² of Al at 2.7 g/cm³ is 0.1 mm, and that optical depth equals (μ/ρ) × areal density from the Elam data.
- X-ray line energies come from the X-ray Data Booklet (LBNL), Table 1-2. Tests check them against wavelengths typed independently (for example Cu Kα1 1.5406 Å, Mo Kα1 0.7093 Å) and, within 0.2%, against the Elam tables. A source sets one reference energy. It does not model a tube spectrum.
- `tools/ui_audit.py` now also audits every analysis tab in the dark theme at three widths. Plot colours inside Plotly figures are chosen by hand for the dark theme and are not all covered by the computed-style contrast check.

**Change to the browser comparison.** `tools/web_parity.py` originally compared every number by its relative difference, with a limit of 10⁻¹². On 1 November 2026, the same commit passed on one CI runner and failed on another: the largest difference was 1.71 × 10⁻¹², on a transmission of 4.4 × 10⁻²³⁷ at τ = 544, which is a difference of about 3 × 10⁻¹⁵ in τ. Because T = exp(−τ), rounding in a large τ is magnified in T. Transmissions below 1/e are now judged by the relative difference in τ. This loosens the check only for transmissions below 1/e; the limit itself and all other numbers are unchanged.

**Self-hosted runtime.** While testing these changes on 1 November 2026, the jsDelivr CDN served the SciPy wheel at about 30 KB/s and sometimes refused it with HTTP 403, while other files arrived normally, and a page load failed. This may also explain the unexplained 60 s load above, though that was not established. The site now serves its own copies from the official Pyodide 314.0.7 release, pinned by SHA-256, with each wheel checked against Pyodide's lock file, so a visit no longer contacts any third party.

**Materials library.** The library is available in both the local and browser versions, and `tests/test_web.py` holds their answers together. A test also compares the library overview with the full engine. Provenance is described in `docs/MATERIALS.md`. Stoichiometric entries use typical densities that have not been checked against a source. They are marked unverified in the interface and in the data.

**Total scattering.** The new tab adds calculations without changing any existing one. Its checks and limits are listed in `docs/TOTAL_SCATTERING.md`. Up to Q = 25 Å⁻¹, form factors agree with the separate EPDL97 tabulation in xraylib within 0.5% to 4.8%, depending on the element, and up to Q = 50.7 Å⁻¹ within 3.5% to 11.4%. For Cu, Hf and Pb, f′ agrees within 6% and f″ within 4%. These are planning values, and no comparison with measured total-scattering data has been made.

## Differential scattering (PR7)
Hubbell incoherent S(Q) is combined with the unpolarized Klein–Nishina cross section.
Elastic independent-atom f0 gives the Thomson coherent term. Solid-angle Gauss–Legendre
integration at 20 keV for SiO2 differs from Elam by +0.01097% coherent and +0.39338%
incoherent. The UI reports differences, without fitting or mixing data. Integration is
unavailable if full angular coverage exceeds the tabulated f0 domain. Attenuation is unchanged.

## Capillary absorption
The disc area integral averages exp[-mu(l_in+l_out)] with unit-radius Gauss quadrature.
Independent chord integrals at 0 and 180 degrees agree to 2e-6 absolute for muR=0,0.2,1,5.
Polarization is the Thomson factor, separately reported for unpolarized and linear states.
No inversion is supplied above muR=30, where fixed quadrature may miss the surface layer.

## Optical tools
Chantler optical constants use n=1-delta+i beta and exp(i kz z); positive beta attenuates.
Si at 8.048 keV, 2.329 g/cm3 gives delta=7.5730e-6 and critical grazing angle about 0.223°.
At normal incidence, the plotted intensity depth recovers lambda/(4 pi beta).
Optical data are separate from Elam attenuation; no data sources are blended.

## Multilayer reflectivity
Parratt recursion uses vacuum / films / semi-infinite substrate, with grazing angles.
The zero-film limit equals Fresnel; a zero-thickness interposed film gives the same result.
Identical media give zero reflection. Névot–Croce factors reduce high-angle reflection.
Nonphysical reflectivity is refused, never clipped into range.
References: https://xraypy.github.io/XrayDB/python.html (optical constants and mirror conventions);
https://www.reflectometry.org/learn/3_reflectometry_slab_models/roughness_and_microslicing.html.

## Powder peaks
The diamond-Si demonstration at a=5.431 Å and Cu Kα1 (1.540593 Å) gives (111) at
about 28.44 degrees. F(200) cancels to numerical roundoff; multiplicity of (111) is 8.
Only explicit unit-cell positions are used, with isotropic B and neutral f0.
The structure-factor convention follows https://dictionary.iucr.org/Structure_factor.

## S(Q) to G(r)
For F(Q)=exp(-aQ) sin(bQ), the infinite-range transform is
a/pi [1/(a²+(b-r)²) - 1/(a²+(b+r)²)]. At a=1, b=3, Q=0..40 Å⁻¹,
8001 samples reproduce that analytic result within 1e-8 Å⁻².
The Lorch window has its continuous limit at Q=0. Flat S=1 gives G=0.
No raw-intensity reduction, extrapolation or fitted baseline is performed.

## PR7 completion checks — 1 November 2026, unpublished

826 Python tests and 58 Node tests pass locally; ruff, strict mypy and manifest
verification pass. The expanded UI audit includes all populated tools, batch
results and share/citation dialogs at 1440, 768 and 375 px in both themes.
No automated violations or page errors were found. This is not a complete
accessibility certification. Safari, Firefox and real devices remain untested.

The original attenuation digest was recomputed, unchanged:
4d50d8cd05b54d9c0251e633a5d803b6ced194e1ec3eb4c5134ee02fdb5372e4.
Real Pyodide compared 247,930 numbers: 242,569 identical, 5,361 different;
nine additional API smoke cases also succeeded. The maximum comparison metric
was 5.46e-16 in this Windows run. The inherited rule below T=1/e divides relative
transmission difference by max(1,-ln T), approximating relative optical-depth
error for small differences. It does not claim raw relative transmission error
at that scaled value or byte equality. The 1e-12 limit was not changed.

New utilities are checked against analytic limits and reference examples, not
measured specimens. No model tolerances were widened. No merge, tag, release,
Pages enablement or new DOI occurred during this work.
