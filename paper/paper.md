---
title: 'X-ray Attenuation Workbench: reproducible attenuation calculations for multicomponent materials and multilayers'
tags:
  - Python
  - X-ray attenuation
  - materials science
  - glasses
  - ceramics
  - multilayers
authors:
  - name: Mohamed Faizal Ussama Jalaludeen
    given-names: Mohamed Faizal Ussama
    surname: Jalaludeen
    orcid: 0000-0003-0077-691X
    affiliation: 1
affiliations:
  - name: Aalborg University, Denmark
    index: 1
date: 27 October 2026
bibliography: paper.bib
---

<!-- AUTHOR CHECK: confirm authorship, affiliation, funding, actual research use,
     historical AI model identifiers and final human review before submission.
     See SUBMISSION_NOTES.md. Do not submit this draft with these checks open. -->

# Summary

Choosing an X-ray energy and specimen thickness requires knowing how much of
the incident beam will reach the detector. For glasses, ceramics and layered
specimens, this depends on several chemical constituents, their proportions,
the measured density and the beam path. X-ray Attenuation Workbench brings
these inputs into a local application for calculating the fraction of the
primary beam transmitted through a specimen. Researchers can explore a
material interactively, repeat a calculation from a saved project, or run a
series of calculations through a command-line interface or Python library.

The Workbench combines established atomic interaction data with explicit
composition and density models. Its contribution is a reproducible materials
workflow: composition conversion, layer geometry, energy-dependent analysis,
measured-spectrum weighting, uncertainty propagation and traceable exports
share one calculation engine. A materials library provides reference
compositions while distinguishing sourced densities from unverified estimates.
The software is intended for experiment planning and interpretation within
the assumptions of primary-beam attenuation.

# Statement of need

Materials researchers often describe a glass in oxide mole percentages, a
composite in mass fractions, or a porous ceramic by its solid density and
porosity. Evaluating attenuation from such descriptions requires consistent
conversion to elemental mass fractions, a justified bulk density and a path
length that accounts for incidence angle. A multilayer or measured energy
spectrum adds further bookkeeping. These operations can be performed in
scripts, but differences in conventions and undocumented intermediate steps
make calculations difficult to compare or reproduce.

The Workbench targets researchers working with multicomponent glasses,
ceramics, composites, filters and planar multilayers. It supports mass, mole
and volume composition inputs, normal thickness and incidence angle, and
measured bulk, porosity-adjusted solid or ideal additive-volume density.
Measured specimen density is preferred for quantitative work; the ideal model
does not account for nonadditive volume in real mixtures. Interactive plots
help explore energy and thickness choices, while saved configurations and
batch operation connect exploration to repeatable analysis.

# State of the field

NIST XCOM supplies photon cross sections for elements, compounds and mixtures
[@xcom], and NIST SRD 126 provides reference attenuation tables [@nist126].
xraylib exposes a broad collection of X-ray interaction calculations to
scientific programs [@schoonjans2011]. XrayDB supplies atomic data, compound
attenuation functions and related materials calculations, including the
Elam, Ravel and Sieber coefficients used here [@xraydb; @elam2002]. Henke and
colleagues provide complementary optical data for the soft-X-ray region
[@henke1993]. These resources already address the underlying data and many
individual calculations.

The Workbench builds on this ecosystem rather than introducing new atomic
coefficients. Its application layer addresses a combined workflow: oxide
composition conversion, explicit density assumptions, multilayer geometry,
spectral transmission, uncertainty and recoverable input provenance. A
separate application keeps those domain-specific policies and interface
choices together while leaving atomic-data maintenance to established
libraries. Extending the calculation through a backend contract also avoids
coupling the workflow to a single library interface. Backends declare their
supported domains, interaction channels and edge conventions; their outputs
are not averaged. The measured agreement between XrayDB/Elam and xraylib is
identified as shared-data agreement, not independent cross-database validation.

# Software design

A Python calculation layer is separated from the local FastAPI service and
browser interface. The same engine supports graphical exploration and batch
analysis, limiting the risk of different formulas or defaults in different
interfaces. JSON projects retain the configuration, including spectrum and
uncertainty settings. The service and command-line result envelope attach
the input, its SHA-256 fingerprint, schema and software versions, and backend
provenance. These records make assumptions inspectable and calculations
repeatable; a fingerprint identifies an input and does not validate it.

The engine converts formula-unit mole or volume inputs to elemental mass
fractions before applying the independent-atom mixture rule. For layers
indexed by $i$ and elements by $j$, primary transmission is

$$
T(E)=\exp\!\left[-\sum_i\frac{\rho_i t_i}{\cos\theta_i}
\sum_j w_{ij}(\mu/\rho)_j(E)\right].
$$

Here $w_{ij}$ is elemental mass fraction, $\rho_i$ is effective density in
g/cm$^3$, and $\theta_i$ is measured from the surface normal. Normal thickness
entered in mm is converted to cm at the path-length boundary. Layer optical
depths add, so layer order does not change primary transmission within this
model. Explicit density modes prevent porosity from being applied twice to
a measured bulk density.

Coefficients are evaluated at each requested energy. The sweep adds samples
around reported absorption edges; reference and supplied spectrum energies
are evaluated directly. Straight plotted segments preserve those samples
without implying resolved chemical edge structure. The accepted energy
domain is 1--800 keV, while the default backend reports a recommended
subdomain of 1--250 keV and warns beyond it. This distinguishes numerical
availability from a claim of uniform physical reliability.

Discrete spectra use integrated photon fluence per energy bin, with optional
energy-fluence weighting. Seeded Monte Carlo sampling propagates specified
density and normal-thickness uncertainties using independent positive
lognormal draws. The reported 95% intervals are pointwise; they exclude
composition, angle, coefficient uncertainty and correlations. This bounded
model makes the included uncertainty sources explicit.

The current review snapshot's 501-entry materials library combines 47 NIST
reference entries, 359 multielement definitions from PNNL-15870 Revision 2
[@pnnl2021], and 95 stoichiometric definitions generated from formulas or stated
designations. Imported records retain source metadata, retrieval information
and source-file hashes. Formula-derived compositions and nominal reference
densities are not certified specimen measurements; densities lacking checked
sources remain labelled unverified.

# Validation and limitations

Automated tests cover composition bases, unit conversion, mixture weighting,
density models, oblique paths, equivalent and reordered layers, spectra,
uncertainty and invalid inputs. Continuous integration exercises Python
3.12--3.14 on Windows and Linux, and Python 3.14 on macOS, alongside frontend
and package checks. A documented benchmark compares twelve total mass
attenuation values for aluminium, iron, lead and water at 20, 50 and 100 keV
with rounded NIST tables [@nist126]. The largest absolute relative difference
is 0.031245%. A separate composition check for eleven formula-annotated NIST
compounds finds absolute elemental mass-fraction differences below
$3.4\times10^{-5}$, consistent with the precision of the source tables.

These checks address implementation and selected reference cases. They do
not establish experimental accuracy across all materials and energies. The
model removes photons absorbed or scattered out of the primary beam; the
removed fraction is not absorbed energy or dose. Buildup, multiple scattering,
secondary-fluorescence transport, detector response, diffraction, coherent
interfaces and microstructure-dependent transport are outside its scope.
Independent-atom coefficients and edge brackets do not predict XANES or EXAFS.

# Research impact statement

The earlier version 1.0.0 is archived with a version-specific Zenodo DOI
[@workbench2026]. This paper describes the subsequent review snapshot,
including its expanded materials library; the existing archive is not
presented as containing all of those later additions.

The repository supplies an executable notebook and a saved two-layer example
combining an illustrative borosilicate glass, an aluminium support, a discrete
spectrum and seeded uncertainty propagation. That example connects graphical,
command-line and library workflows and provides a reproducible starting point
for evaluating a research calculation. Its documented photon-weighted
transmission is 0.7802626916899833; this is a software regression value for
illustrative inputs, not an experimental result.

At present, the evidence of significance is the inspectable benchmark,
reproducible example and integration of several materials-analysis steps in
one workflow. The current revision has not yet been applied in the author's
research manuscript. Such an application is planned, and adoption by
independent research groups is not claimed. The tests and contribution
guidelines support evaluation and reuse, but planned use should not be
interpreted as demonstrated research impact.

<!-- SUBMISSION HOLD: after completing a real research application, supply the
     material system, scientific task, software version/commit and supporting
     publication/preprint or an accurate description of private ongoing work.
     The software's own Zenodo record is not evidence of a study using it.
     Align the reviewed release and its archive before final publication. -->

# AI usage disclosure

ChatGPT assisted development of the earlier calculator. Anthropic Claude Code
and OpenAI Codex assisted work on version 1.0.0 and subsequent maintenance,
including code generation and refactoring, tests, documentation and manuscript
drafting or editing. Historical model identifiers have not yet been confirmed.
Verification of AI-assisted changes includes automated regression and input
validation tests, selected comparisons with published reference data,
bibliographic metadata checks, and interface and package checks. These
automated checks support review and do not replace scientific judgement or
human responsibility for the submitted work.

<!-- SUBMISSION HOLD: the author must supply available tool/model versions,
     confirm the assistance described above, and explicitly affirm personal
     review/editing/validation of all AI-assisted outputs and responsibility
     for core design decisions before submission. Do not assert completed
     human review on the author's behalf. -->

# Acknowledgements

The Workbench builds on the efforts of the XrayDB and xraylib developers and
the authors of the atomic-data and material-composition sources cited here.

<!-- AUTHOR ACTION: confirm funding and other acknowledgements. Do not infer
     that no funding was received from the absence of supplied grant details. -->

# References
