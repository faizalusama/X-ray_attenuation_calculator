# Scientific methods and validation boundaries

Research review date: **19 September 2026**. This document describes the workbench's narrow-beam attenuation model. It does not establish experimental accuracy for a particular glass, ceramic, composite, or instrument. The separate [roadmap](ROADMAP.md) identifies capabilities requiring further implementation and validation.

## Physical model and data

The engine uses **XrayDB's Elam mass attenuation coefficients**. Photon energies enter the interface in keV and are converted to eV for XrayDB. Mass attenuation coefficients are in cm²/g; linear attenuation coefficients are in cm⁻¹. Total attenuation includes photoelectric absorption, coherent (Rayleigh) scattering, and incoherent (Compton) scattering. The component curves are reported separately. XrayDB documents these functions and their units in its [Python API reference](https://xraypy.github.io/XrayDB/python.html). The underlying compilation is Elam, Ravel and Sieber, *A new atomic database for X-ray spectroscopic calculations* (2002), [doi:10.1016/S0969-806X(01)00227-4](https://doi.org/10.1016/S0969-806X(01)00227-4).

The application accepts **1–800 keV**. This is an input boundary, not an accuracy certificate. XrayDB's [upstream implementation](https://raw.githubusercontent.com/xraypy/XrayDB/master/python/xraydb/xraydb.py) clamps Elam calculations below 100 eV and above 800 keV; this workbench rejects out-of-range requests instead. XrayDB describes approximately **0.25–250 keV** as the range where its energy-dependent cross sections are generally most reliable. Results above 250 keV therefore carry an additional caution. See the [XrayDB overview](https://xraypy.github.io/XrayDB/). The installed package version is recorded with results; reproducing a study requires retaining that version, the input configuration, and its output.

## Composition and density

For component mass fractions \(w_i\), the compound/mixture rule is

\[
(\mu/\rho)_\mathrm{mix}(E)=\sum_i w_i(\mu/\rho)_i(E),\qquad
\mu(E)=\rho_\mathrm{bulk}(\mu/\rho)_\mathrm{mix}(E),\qquad
\sum_iw_i=1.
\]

Chemical formulas are decomposed into elements using stoichiometric counts and XrayDB atomic masses. Oxide molar fractions describe formula units: 20 mol% Na₂O is not 20 atomic% Na. Fractions supplied as positive amounts are normalized; inspect the resulting mass fractions before using a calculation. Component identity is chemical composition, not a phase-dependent absorption measurement.

- Mass basis: \(w_i=a_i/\sum_j a_j\).
- Mole basis: \(w_i=x_iM_i/\sum_j x_jM_j\), with formula molar mass \(M_i\).
- Volume basis: \(w_i=v_i\rho_i/\sum_jv_j\rho_j\); each component density is required.

Three density modes keep measured and inferred quantities distinct:

- **Bulk:** use the supplied measured bulk density directly. Porosity must be zero because its effect is already included in that density.
- **Solid:** use \(\rho_\mathrm{bulk}=(1-p)\rho_\mathrm{solid}\), with void fraction \(p\).
- **Ideal:** estimate \(\rho_\mathrm{solid}=1/\sum_i(w_i/\rho_i)\), then apply \((1-p)\).

The ideal-density relation assumes additive constituent volumes. It does not predict glass densification, reaction products, crystal structure, sintering shrinkage, or nonideal excess volume. Measured specimen density is preferred for quantitative work. The porosity correction treats pores as empty and uniformly distributed on the scale relevant to the beam. Resolved pores, phase segregation, agglomerates, and cracks can produce a distribution of path integrals; replacing that distribution by one mean density can give a different transmission.

The mixture rule is supported by [NIST XCOM's introduction](https://physics.nist.gov/PhysRefData/Xcom/Text/intro.html), which also explains that isolated-atom tabulations omit chemical and solid-state modifications close to absorption edges. That limitation matters for glasses and ceramics even when their nominal stoichiometry is known precisely.

## Layers, geometry, and output quantities

For layer \(\ell\), thickness \(t_\ell\) is its normal thickness. The angle \(\theta_\ell\) is measured **from the surface normal**, giving path length

\[
s_\ell=t_\ell/\cos\theta_\ell,\qquad
\tau(E)=\sum_\ell\mu_\ell(E)s_\ell,\qquad
T(E)=\exp[-\tau(E)].
\]

Millimetres are converted to centimetres before multiplication by \(\mu\). Each layer is homogeneous, and the beam is assumed to traverse its full calculated path. Finite lateral dimensions, surface roughness, refraction, and grazing-incidence reflection are outside this geometry. Layer order does not affect this primary-beam result; an order-dependent result requires secondary-radiation or wave-optics modelling.

**Removed fraction** is \(1-T\): the fraction of primary photons removed from the uncollided beam. It is not absorbed energy, dose, or the fraction collected by an arbitrary detector. Scattered and fluorescent photons may leave the sample or reach a detector. Even the photoelectric coefficient is not a mass energy-absorption coefficient. NIST separately tabulates attenuation and energy-absorption coefficients in its [mass attenuation database](https://physics.nist.gov/PhysRefData/XrayMassCoef/tab3.html).

For a homogeneous material at the reference energy:

\[
\mathrm{HVL}=\ln(2)/\mu,\quad
\mathrm{TVL}=\ln(10)/\mu,\quad
\lambda_{1/e}=1/\mu.
\]

These are material **path lengths**, converted to mm in the interface. Normal thickness at nonzero angle is the path length multiplied by \(\cos\theta\). They are monochromatic values; successive half-value layers of a broadband beam generally differ because its spectrum changes during filtering.

The stack target calculation returns a common scale factor
\(k=-\ln(T_\mathrm{target})/\tau(E_\mathrm{ref})\).
Multiplying every layer's thickness by \(k\), while keeping composition, density, and angle fixed, reaches the specified monochromatic transmission within the model. This is not a thickness optimization under mechanical or manufacturing constraints.

## Absorption edges and numerical evaluation

The sweep adds the nominal edge energy and points displaced by a relative **±10⁻⁴** on each side, retaining only points within the requested range. This offset covers the limited precision of Elam's stored logarithmic energy knots: in XrayDB 4.5.8, the largest difference between an atomic edge's metadata and its associated coefficient step is approximately **4.92 × 10⁻⁵ relative** across all 543 supported edges above 1 keV. A regression test checks this coverage against the installed database. For example, the Fe K-edge metadata specify 7112 eV, while its repeated log-energy knot 8.86954 places the coefficient step at **7112.0086966 eV**. The engine does not shift or reinterpret backend coefficients; nominal edge metadata are labels, not a promise of the exact numerical step location.

The requested reference energy and all spectrum-bin energies are evaluated directly, rather than interpolated from the plotted sweep. This avoids averaging plotted values across a discontinuity. A value exactly at a tabulated knot follows the backend's interpolation convention and need not equal either one-sided limit; use explicit energies that bracket the coefficient step when comparing tabulations. The relative bracket is recorded in result provenance. NIST likewise includes both sides of edges in its [XCOM output](https://physics.nist.gov/PhysRefData/Xcom/Text/chap4.html).

Additional points expose tabulated jumps; they do not calculate chemical edge shifts, XANES, EXAFS, finite monochromator bandwidth, or lifetime broadening. A smooth-looking curve is not evidence that these effects are present. Numerical precision, interpolation error, database uncertainty, material uncertainty, and model adequacy are separate issues.

## User-supplied spectra

Each input weight \(n_j\) is **integrated photon fluence in a discrete bin** represented by \(E_j\), not a density per keV. For a spectral density input, integrate over each bin first; using density values directly on a nonuniform energy grid gives incorrect weighting. A wide bin spanning an edge needs subdivision.

The outputs follow directly from the monoenergetic model:

\[
T_\mathrm{photon}=\frac{\sum_j n_jT(E_j)}{\sum_jn_j},\qquad
T_\mathrm{energy}=\frac{\sum_j E_jn_jT(E_j)}{\sum_jE_jn_j}.
\]

Transmitted weights are \(n_jT(E_j)\). Mean input and output energies are photon-fluence-weighted means. Selecting energy weighting changes the integrated transmission ratio, not the interpretation of the supplied photon weights. The output mean uses logarithmic normalization and can remain finite even when the absolute transmission underflows to zero; it then describes a theoretical conditional spectrum, not evidence of measurable transmitted fluence. Spectral hardening follows from energy-dependent attenuation; no X-ray tube emission spectrum is synthesized. Detector efficiency, response matrices, escape peaks, pulse pile-up, and scattered counts are not modelled.

## Uncertainty propagation

The optional seeded Monte Carlo calculation propagates independent relative one-standard-deviation uncertainties in **effective bulk density and normal thickness** for each layer. A positive lognormal multiplier \(L\) has mean one and coefficient of variation \(c\):

\[
\sigma_\log^2=\ln(1+c^2),\qquad
\mu_\log=-\sigma_\log^2/2,\qquad
L\sim\mathrm{Lognormal}(\mu_\log,\sigma_\log).
\]

The same sampled material/thickness realization is propagated across the entire energy sweep. Reported 2.5th and 97.5th percentiles are a central 95% interval under these input assumptions. They are not a confidence interval for all physical error. Composition, energy calibration, angle, porosity, cross-section errors, shared calibration errors, and correlations are excluded. Avoid entering independent density/thickness uncertainties when both were inferred from the same mass/area measurement without considering their covariance. The seed and sample count are returned for reproducibility; finite-sample percentile error remains.

## Original correction and independent checks

The supplied original script multiplied component mass fraction twice: an outer \(w_i\) multiplied `material_mu` evaluated at density \(\rho w_i\). Its effective mixture was \(\rho\sum_iw_i^2(\mu/\rho)_i\). The corrected implementation applies mass weighting once. For two equal-mass components, the original linear coefficient was one half of the correct coefficient, so it could substantially overestimate transmission. A single-component example cannot detect this bug.

The archived notebook also contains experimental calculation cells that must not be treated as a reference method: a mass-weighted arithmetic density in place of an additive-specific-volume estimate; weighted averages of individual reciprocal attenuation depths in place of the reciprocal of the mixed coefficient; a mole-to-mass conversion divided by total mole fraction instead of total mass; and an incorrectly entered Na₂O molar mass. Formula-derived masses and the explicit mixture/density equations above replace those calculations. The unchanged `legacy/` copy preserves the original work for comparison; use the new engine for subsequent results.

`tests/test_nist.py` uses independently transcribed NIST values, rather than calling the backend again as its expected result. At **20, 50, and 100 keV**, respectively, the total mass attenuation benchmarks in cm²/g are:

- Aluminium: **3.441, 0.3681, 0.1704**. [NIST aluminium](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html)
- Iron: **25.68, 1.958, 0.3717**. [NIST iron](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html)
- Lead: **86.36, 8.041, 5.549**. [NIST lead](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z82.html)
- Water: **0.8096, 0.2269, 0.1707**. [NIST liquid water](https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html)

An additional Fe K-edge check compares values below and above **7.112 keV** with NIST's limiting values **53.19** and **407.6 cm²/g**, respectively. It uses energies displaced by a relative \(10^{-4}\) from the edge, which exceeds the backend's energy-knot rounding discrepancy; smooth-region variation over this offset is much smaller than the 1% comparison threshold. A relative \(10^{-8}\) offset would leave both evaluations below the actual Elam coefficient step. This is a check of the tabulated jump, not a test of real iron's near-edge fine structure. [NIST iron](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html)

The selected reference comparisons use a **1% relative agreement threshold**. Passing them checks units, composition conversion, and backend consistency for those examples. It does not establish 1% accuracy throughout the domain, nor does it constitute experimental validation. NIST's tabulated precision also limits a comparison to those rounded values. Cross-library agreement can share underlying theory or tabulations. [NIST's database comparison](https://www.nist.gov/pml/note-nist-x-ray-attenuation-databases) explicitly distinguishes the XCOM and FFAST methods.

Before publishing quantitative claims, retain specimen-specific composition/density evidence, characterize the source and detector, compare with measurements at multiple energies and thicknesses, and publish residuals and an uncertainty budget alongside the software version. See the [roadmap](ROADMAP.md) for the remaining scientific work.
