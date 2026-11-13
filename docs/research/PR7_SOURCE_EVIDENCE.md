# PR7 source evidence

Verification date: **2026-11-13, Europe/Copenhagen**.  
Integration base: `claude/web-version`, `2de66a63a78ddc6c4f7cec3cb4a27ae7049625ee`.  
Scope: R1 research evidence only; no scientific values, production changes, or new validation claims.

## Verification provenance

The R1 worker read `AGENTS.md`, `docs/WORK_PLAN.md`, existing scientific documentation, relevant tests, and table-generation code in `outputs/xray-task-research`.

All five assigned DOIs are **VERIFIED_BY_COORDINATOR** against Crossref. The worker read and checked the coordinator's supplied records in `outputs/xray-crossref-review.json`; the worker did not successfully retrieve those API records personally. Crossref API locators and resolved metadata are recorded in `pr7-source-evidence.json`. DataCite was not queried.

Worker registry requests through the web tool were inaccessible. The local DOI verifier failed with `getaddrinfo failed`; the proposed unrestricted retry was interrupted. These access failures do not invalidate the coordinator's subsequently supplied Crossref evidence.

Primary-page inspection, registry resolution, numerical agreement, and human verification of transcribed data are separate evidence categories. No named human data reviewer or new human verification is asserted here.

## Layout and accessibility

### WCAG 2.2

Use the official [12 December 2024 W3C Recommendation](https://www.w3.org/TR/2024/REC-WCAG22-20241212/). No DOI is supplied by this standard; use its versioned URL.

Relevant locators are SC 1.3.1, 1.4.1, 1.4.3, 1.4.10, 1.4.11, 2.1.1, 2.4.3, 2.4.7, 2.4.11, 2.5.8, 3.3.1, 3.3.2, 4.1.2 and 4.1.3. They support meaningful grouping, keyboard operation, visible focus, error discovery and accessible state announcements.

Text contrast thresholds are 4.5:1, or 3:1 for large text; required component/state contrast is 3:1. Target sizing specifies 24 by 24 CSS pixels with exceptions. Reflow addresses 320 CSS pixels, with exceptions for necessary two-dimensional content. A 375-pixel audit alone therefore does not establish reflow conformance. The repository's 12-pixel font floor is a project policy, not a WCAG minimum-font-size rule. These are selected criteria, not a complete conformance assessment. [Normative criteria](https://www.w3.org/TR/2024/REC-WCAG22-20241212/)

Existing support: `tools/ui_audit.py` measures selected interface properties and exercises tabs and reference disclosure. It does not establish complete accessibility conformance.

### Tabs

The official [APG Tabs Pattern](https://www.w3.org/WAI/ARIA/apg/patterns/tabs/) provides implementation guidance, rather than a WCAG success criterion. No DOI is supplied.

Exact locators: “Keyboard Interaction” and “WAI-ARIA Roles, States, and Properties.” Preserve the active tab's tab stop, arrow navigation, selected state, labels and tab/panel relationships. Home/End are optional. Automatic activation is recommended only when panels appear without noticeable latency.

Application to PR7: visual category headings must preserve coherent tab navigation. Inspect the final DOM and actual keyboard behavior; headings alone do not prove appropriate grouping.

Existing support: `tools/ui_audit.py` checks arrow-key navigation, selected state and focus, and exercises Home. Final grouped navigation still needs integrated testing and assistive-technology review.

### Progressive disclosure

The official [APG Disclosure Pattern](https://www.w3.org/WAI/ARIA/apg/patterns/disclosure/) describes a button controlling content visibility, Enter/Space operation, and synchronized `aria-expanded`; `aria-controls` is optional. No DOI is supplied.

Application to PR7: a collapsed configuration panel needs a discoverable reopening control. Simple/Advanced modes should expose relevant errors and active settings. Preserving values and making hidden invalid settings discoverable are project safeguards, not verbatim APG requirements.

The sources do not establish that a left sidebar, three categories, Simple/Advanced naming, or exactly four initial plots is optimal. Those remain design decisions requiring integrated review or user testing.

## Existing scientific features and source locators

### Elam, Ravel and Sieber — attenuation and atomic parameters

Crossref resolved *A new atomic database for X-ray spectroscopic calculations*, W.T. Elam, B.D. Ravel and J.R. Sieber, 2002, volume 63(2), pages 121–128, DOI `10.1016/S0969-806X(01)00227-4`.

The [publisher record](https://www.sciencedirect.com/science/article/abs/pii/S0969806X01002274) was available as a search-indexed publisher extract, including Abstract, Introduction, Discussion and Acknowledgements. A subsequent direct open returned 403. The extract describes the compiled database and acknowledges XCOM cross-section input.

Relevant existing checks:

- `tests/test_backend_equivalence.py::test_elam_and_xraylib_evaluate_the_same_data`
- `tests/test_backend_equivalence.py::test_the_two_backends_are_not_offered_as_a_cross_check`
- `tests/test_sources.py::test_lines_agree_with_the_elam_tables`

These support provenance and implementation consistency. **xraylib/Elam attenuation agreement is shared-source agreement, not independent cross-database or experimental validation.**

Missing: named human review of selected source rows, upstream versions and interpolation conventions. No inaccessible paper table number is asserted.

### Hubbell et al. — incoherent scattering functions

Crossref resolved the 1975 paper, volume 4(3), pages 471–538, DOI `10.1063/1.555523`, with six authors: J.H. Hubbell, Wm. J. Veigele, E.A. Briggs, R.T. Brown, D.T. Cromer and R.J. Howerton.

Verified primary locators: [NIST-hosted reprint](https://srd.nist.gov/jpcrdreprint/1.555523.pdf), §1.2, §3, §5.1 and Table I. Table I begins on PDF page 26, counting the publisher cover as page 1. Its columns distinguish atomic form factors from incoherent scattering functions. The reprint's 2009 online-publication date must not replace the original 1975 citation.

`tools/build_scattering_tables.py` obtains bundled incoherent functions through `xraylib.SF_Compt`. The [upstream API documentation, “Scattering factors”](https://github.com/tschoonj/xraylib/wiki/The-xraylib-API-list-of-all-functions#scattering-factors), distinguishes its momentum-transfer convention from crystallographic Q.

Existing check: `tests/test_scattering.py::test_differential_scattering_limits_and_integral`.

This connects the implemented approximation to its source; that test's comparison with Elam does not certify experimental accuracy. Do not assign Hubbell provenance to every xraylib function.

Missing: human comparison of selected original Table I entries against bundled samples, interpolation, rounding and applicable errata. The reprint identifies a 1977 erratum; applicability was not assessed.

### Sears — neutron lengths and cross sections

Crossref resolved Varley F. Sears, *Neutron scattering lengths and cross sections*, 1992, volume 3(3), pages 26–37, DOI `10.1080/10448639208218770`.

The [official NIST reproduction](https://www.ncnr.nist.gov/resources/n-lengths/list.html) identifies its source data as the Special Feature, pages **29–37**. This data locator differs from the complete article range **26–37**. Exact web locators are the column-definition list and natural-element rows, including H and O. The page distinguishes coherent lengths, incoherent cross sections and absorption at 2200 m/s. The [publisher landing page](https://www.tandfonline.com/doi/abs/10.1080/10448639208218770) returned 403.

Existing check: `tests/test_scattering.py::test_neutron_negative_weights_and_zero_mean_guard`.

It tests normalization, negative weights and a zero-mean guard. It is not a complete comparison of every bundled value with Sears. NIST reproduces the Sears data and is not an independent neutron dataset.

Missing: named human review of natural-abundance row selection, complex-length convention, uncertainties and missing entries against the original article and bundled data.

### Chantler — anomalous factors and edge scans

Crossref resolved C.T. Chantler's 2000 detailed soft-X-ray tabulation, volume 29(4), DOI `10.1063/1.1321055`, with pages **597–1056**.

**Unresolved discrepancy:** the [XrayDB maintainer bibliography](https://xraypy.github.io/XrayDB/biblio.html) gives **597–1048**. Preserve the Crossref value as registry metadata; do not silently resolve the disagreement.

The [official NIST FFAST page](https://www.nist.gov/pml/x-ray-form-factor-attenuation-and-scattering-tables), under “NIST Standard Reference Database 66,” describes the broader database. It must not be treated as identical in scope to this particular article. The [XrayDB API](https://xraypy.github.io/XrayDB/python.html) identifies `f1_chantler`, `f2_chantler` and `chantler_energies`; its bibliography additionally identifies a 2016 finer-grid personal communication. Citation identity alone does not pin installed data to the printed paper.

Existing checks:

- `tests/test_scattering.py::test_chantler_edge_scan_has_absorption_jump_and_explicit_selection`
- `tests/test_scattering.py::test_anomalous_terms_agree_with_xraylib`
- `tests/test_scattering_tools.py::test_silicon_optics_matches_reference_and_normal_absorption_limit`

These cover behavior, selected library agreement and a reference regression. This review did not establish their independent experimental provenance.

Missing: publisher confirmation of the page range; exact original table/row locators for selected factors; named human verification of source versions, units, signs and edge interpolation. The attempted NIST reprint URL was inaccessible.

### Waasmaier and Kirfel — neutral and ionic form factors

Crossref resolved D. Waasmaier and A. Kirfel, *New analytical scattering-factor functions for free atoms and ions*, 1995, volume 51(3), pages 416–431, DOI `10.1107/S0108767394013292`.

Verified primary locator: the [IUCr publisher abstract](https://journals.iucr.org/paper?S0108767394013292), describing five-Gaussian representations for neutral atoms and ions over sin(theta)/lambda from 0 to 6 inverse angstroms. The linked article PDF returned 401. The [XrayDB `f0` API](https://xraypy.github.io/XrayDB/python.html) identifies the source and explicitly distinguishes its input from Q by the factor 4*pi.

Existing checks:

- `tests/test_scattering.py::test_form_factors_start_at_the_atomic_number`
- `tests/test_scattering.py::test_ionic_electron_counts_and_neutral_attenuation`
- `tests/test_scattering.py::test_curves_stop_at_the_limit_of_the_form_factor_fits`
- `tests/test_scattering.py::test_form_factors_agree_with_epdl97_in_xraylib`

The last is a comparison of form-factor tabulations, distinct from the shared Elam attenuation comparison. Its EPDL97 provenance is asserted by the existing test; this round did not independently audit that upstream data chain.

Missing: original coefficient-table locators and named human checks of selected neutral/ionic coefficients and fit errors.

## Existing NIST attenuation benchmark

The primary locators used by `tests/test_nist.py` are:

- [Aluminium elemental table](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html)
- [Iron elemental table](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html)
- [Lead elemental table](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z82.html)
- [Liquid-water table](https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html)

Exact locators are the **mu/rho** column at energy rows `2.00000E-02`, `5.00000E-02` and `1.00000E-01` MeV. Iron additionally has two edge rows at `7.11200E-03` MeV. These pages were opened and their headers and relevant rows inspected; no new reference values are introduced here.

Mapped checks are `NistBenchmarks.test_independent_pure_material_coefficients`, `test_fe_k_edge_one_sided_nist_limits` and `test_nist_equal_mass_mixture_prevents_squared_fraction_regression`.

This is an external reference-table comparison, not independent experimental validation. The Elam publisher extract's XCOM acknowledgement further cautions against assuming independent underlying inputs. The selected agreement threshold is not a general accuracy bound.

Missing: a named human sign-off for source transcription and benchmark provenance. This review does not convert an earlier transcription comment into human verification.

## Checks and handoff

Completed read-only checks:

- Parsed the coordinator's JSON and confirmed exactly the five requested DOIs.
- Checked author counts, volumes, issues and page ranges against those supplied records.
- Parsed relevant test modules and confirmed all 15 mapped test functions exist.
- Confirmed `tools/ui_audit.py` exists.
- Inspected the primary pages identified above, recording partial access and failures.

No numerical suite, UI audit, full integration gate, generated-file regeneration or commit was performed. Git status inspection failed in the restricted environment, so no successful Git cleanliness check is claimed.

Per the final coordination instruction, this content was returned in chat for coordinator writing. **The worker changed zero files.** The coordinator should save only the two assigned evidence files, validate the saved JSON, review the unresolved source limitations, and perform the required integration checks and sealing.