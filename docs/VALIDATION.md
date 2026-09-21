# Validation record — version 1.0.0

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

**XrayDB/Elam and xraylib return bit-identical coefficients.** Across 99
element/energy samples (H, C, O, Al, Si, Fe, Cu, Ba, W, Pb, U at 5, 10, 17.3,
30, 60, 100, 200, 500 and 700 keV), xraylib 4.3.0's `CS_Photo`, `CS_Rayl` and
`CS_Compt` equalled XrayDB 4.5.8's `mu_elam` **exactly** — to the last bit of a
float64, in every channel separately, with a maximum relative difference of
0.000e+00. The two packages expose one evaluated dataset through two interfaces.

Consequently **this project still has no independent cross-database
validation.** Agreement between these two backends measures nothing. The
xraylib backend declares `shares_data_with = ("elam",)`,
`backends/compare.py` excludes the pair from legitimate cross-checks, and
`tests/test_backend_equivalence.py` asserts the exact equality so that any
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

The new application is an unpublished development build. Experimental validation, independent scientific review, additional operating-system installation testing, a formal license decision, and a version-specific release citation remain for publication. Transport of scattered/fluorescent photons, calibrated source generation, diffraction/near-edge fine structure, detector response, and broader correlated uncertainty are explicitly outside this implemented model; see [ROADMAP.md](ROADMAP.md).

The licence decision is now made: the project is released under the MIT License, recorded in [LICENSE](../LICENSE), `CITATION.cff` and the package metadata, with third-party terms listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Experimental validation, independent scientific review, a version-specific release citation and installation testing on the operating systems now covered by CI remain outstanding.

## Interactive dashboard pass — 21 September 2026

On Windows with Python 3.12.14, the complete Python suite passed **123 tests and 14 subtests**, with **one optional xraylib test skipped** and two upstream deprecation warnings. No physics implementation changed in this dashboard pass. The two JavaScript suites passed **23 tests** covering the existing frontend and dashboard units, attenuation lengths, target thickness, heatmap values, secondary axes, uncertainty bands, preference validation, backend preservation and the eleven-plot default.

Browser checks exercised live thickness changes, dimension-preserving unit changes, secondary axes, linked zoom, gesture locking, remembered settings, Ideas, native SVG download and responsive plot sizing. An exported SVG was parsed and checked for dimensions and configuration metadata. This is application verification, not a new scientific accuracy claim.

Additional commands: `node tests/test_dashboard.cjs` and `node --check static/dashboard.js`.

Per-plot controls follow-up: all 25 JavaScript tests passed, including independent scale/typography restoration and linear/logarithmic linked-energy range conversion. Browser verification confirmed different font sizes and y-axis scales persist independently after reload, bold SVG axis titles render, and the page has no horizontal overflow or console errors.
