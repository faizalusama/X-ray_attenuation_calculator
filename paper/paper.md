---
title: 'X-ray Attenuation Workbench: provenance-aware narrow-beam attenuation for multicomponent glasses, ceramics and multilayers'
tags:
  - Python
  - X-ray attenuation
  - materials science
  - glass
  - ceramics
  - Beer-Lambert law
authors:
  - name: Mohamed Faizal Ussama Jalaludeen
    orcid: 0000-0003-0077-691X
    affiliation: 1
affiliations:
  - name: Aalborg University, Denmark
    index: 1
date: 27 October 2026
bibliography: paper.bib
---

<!-- DRAFT prepared with AI assistance (Claude Code). Before submission:
     check JOSS's current policy on AI-assisted writing and disclose as required,
     rewrite in your own voice, confirm the affiliation wording, and add the
     research you have used the software for (JOSS asks for evidence of use). -->

# Summary

X-ray experiments on glasses and ceramics routinely need the attenuation of a
specimen of known composition and thickness: to choose an energy, to size a
sample so that enough of the beam is transmitted, or to correct scattering and
imaging data for absorption. The X-ray Attenuation Workbench computes
narrow-beam (primary-beam) attenuation for multicomponent materials and planar
multilayers from chemical composition and density. It provides a local
graphical application, a command-line batch mode and a Python API that share
one validated engine.

Compositions are entered as chemical formulas with mass, mole or volume
fractions; density can be a measured bulk value, a solid density with stated
porosity, or an ideal additive estimate, and each choice is reported as an
explicit assumption. For every layer the Workbench evaluates mass and linear
attenuation coefficients, their photoelectric, coherent and incoherent
contributions, transmission, optical depth, half- and tenth-value layers and
attenuation lengths over 1–800 keV, with additional sampling around absorption
edges. It also weights transmission by a measured discrete photon spectrum and
propagates density and thickness uncertainty by seeded Monte Carlo sampling.
Every result records its exact input, a SHA-256 fingerprint of that input, and
the data source and version used.

# Statement of need

Elemental cross-sections are widely available through XCOM [@xcom], the NIST
attenuation tables [@nist126], Henke et al. [@henke1993], xraylib
[@schoonjans2011] and XrayDB [@xraydb], which implements the Elam, Ravel and
Sieber compilation [@elam2002]. Turning these into a trustworthy number for a
real oxide glass or ceramic still requires several error-prone steps:
converting mole or volume composition to mass fractions, applying the mixture
rule correctly, choosing a density, handling multilayers and oblique
incidence, and sampling densely enough near edges. In our own earlier script
the mixture rule weighted each component's mass fraction twice, underestimating
the attenuation coefficient of common silicate glasses by 20–60%. The
Workbench exists to make these steps explicit, tested and reproducible for
materials scientists who are not specialists in radiation physics.

Three design choices distinguish it. First, assumptions are part of the
output: density mode, energy range relative to the data's recommended range,
and the narrow-beam scope (removed photons are not absorbed dose) are reported
with every result. Second, attenuation data sources are pluggable but are never
averaged; each declares its energy domain, channels and edge convention, and
sources that share an underlying evaluation are marked as such. xraylib and
XrayDB, for example, return the same coefficients to within floating-point
rounding and are therefore not treated as independent validation. Third, a
materials library supplies compositions with enforced provenance: entries
imported from NIST SRD 126 [@nist126] and PNNL-15870 Rev. 2 [@pnnl2021] carry
their source, retrieval date and hash, stoichiometric phases are exact by
formula, and any density without a checked source is labelled as unverified.

# Validation and limitations

The test suite (over 680 tests) covers the mixture rule, composition bases,
units, multilayer identities, spectra, uncertainty, input validation and
selected NIST reference values, and runs on Windows, Linux and macOS with
Python 3.12–3.14. Elemental mass fractions of eleven stoichiometric NIST
materials agree with those derived from their formulas to within
3.4 × 10⁻⁵. Agreement with tabulated values is a regression check, not proof
of experimental accuracy. The model computes primary-beam transmission only:
it does not treat buildup, scattered or fluorescent photon transport, detector
response, or chemical near-edge structure.

# Acknowledgements

<!-- Add funding, beamline or supervisor acknowledgements here. -->

# References
