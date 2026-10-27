# Bounded calculation review before research reuse

Review date: **27 October 2026**. This is an AI-assisted technical check of
specified calculations, not an experimental validation, author sign-off or
independent scientific certification. No engine, backend, material data or
acceptance tolerance was changed during this review.

## Assessed code and environment

The assessed source is commit
`40e8dd857d9ac58da5ef8586455dc1c2318f6736`, with manuscript/documentation edits
in the `codex/joss-submission` checkout. Its calculation code was unchanged.
Git comparison also confirms identical tracked contents in `physics.py`,
`backends/` and `results.py` between this source and the `v1.0.0` tag
(`ec1fcca`). Thus the numerical engine reviewed here is the engine tagged
v1.0.0, evaluated with the dependency versions stated below; the newer
materials library is a separate post-tag feature. No Zenodo ZIP archive was
inspected, so this is a source-control comparison rather than archive-byte
verification.
The runtime was Windows, Python **3.14.6**, NumPy **2.5.4** and XrayDB
**4.5.8**, using the default Elam backend. The package currently reports
`1.0.0`, but this assessment identifies the **commit**, not an assumed match
to an archived release.

In particular, the `v1.0.0` tag predates the current materials library. This
review does not establish that the Zenodo archive and current branch contain
identical code or features. Research reruns must preserve the actual commit,
dependencies, complete inputs and result files. The root package version alone
is insufficient to distinguish these snapshots.

## Reproducing this review

From the repository root, in an environment with the project's dependencies:

```powershell
python -m paper.validation_check
```

The separate scalar implementation is preserved in
[`validation_check.py`](validation_check.py), and its recorded numerical
results are in
[`calculation-review-results.json`](calculation-review-results.json).
This script is a bounded audit supplement with deliberately enumerated
formulas and cases, not a second general-purpose physics engine. Re-running it
under another dependency set should produce a new result record for review;
differences must be investigated rather than hidden by changing tolerances.

## What was independently checked

The calculation was reimplemented as scalar arithmetic using `math.fsum`,
`math.exp`, explicit component/element weights and manually specified atom
counts. This separate calculation did **not** call the engine's private
formula parser, layer-preparation, density, energy-grid or uncertainty helpers.
It called only the public `calculate()` entry point to obtain observed outputs.

For the scalar expectations, elemental coefficients and atomic masses still
came directly from XrayDB. Agreement therefore checks the workbench's
composition algebra, units and transport calculation around those inputs; it
does **not** independently validate the underlying atomic data. The separate
NIST comparisons below use externally tabulated values instead.

The model equations checked were:

\[
w_i=\frac{a_i}{\sum_j a_j},\quad
w_i=\frac{x_iM_i}{\sum_j x_jM_j},\quad
w_i=\frac{v_i\rho_i}{\sum_jv_j\rho_j}
\]

for mass, formula-unit mole and volume parts respectively. Within a compound,
elemental mass fractions were calculated from the stated atom counts and
atomic masses, then combined once with component mass fractions.

\[
\left(\frac{\mu}{\rho}\right)_\ell
=\sum_e w_{\ell e}\left(\frac{\mu}{\rho}\right)_e,\qquad
\tau=\sum_\ell \rho_\ell
\left(\frac{\mu}{\rho}\right)_\ell
\frac{t_\ell/10}{\cos\theta_\ell},\qquad T=e^{-\tau}.
\]

Here normal thickness `t` is in **mm**, `t/10` is in **cm**, density is in
**g/cm³**, elemental mass attenuation is in **cm²/g**, and the angle is measured
from the surface normal. Optical depth is dimensionless. Both scattering
channels contribute to removal from the primary beam. NIST documents the
[mass-weighted mixture rule and isolated-atom limitations](https://physics.nist.gov/PhysRefData/Xcom/Text/intro.html).

The three effective-density cases were checked separately: measured bulk
density used directly; solid density multiplied by `1 - porosity`; and ideal
additive-volume density `1 / sum(w_i / rho_i)`, subsequently multiplied by
`1 - porosity`. The latter is a mixture assumption, not a prediction of the
density of a reacted glass, crystal or sintered specimen.

## Constructed arithmetic cases

Ten cases were checked across their returned 5-800 keV sweeps, including added
edge samples: **179 energy evaluations** in total. These are deliberately
constructed calculation checks, not measured specimens or research-use
evidence. Unless stated otherwise, angles and porosity were zero and density
was supplied as bulk density.

| Case | Explicit inputs | Result at 30 keV |
|---|---|---:|
| Pure Al | 2.699 g/cm³; 0.3 mm | T = 0.9126856950306909 |
| Equal-mass Al/Fe | 50:50 mass parts; 4 g/cm³; 0.4 mm | T = 0.4749867935584340 |
| Illustrative oxide mixture | SiO₂:B₂O₃:Na₂O = 80:15:5 mole parts; 2.23 g/cm³; 1 mm | T = 0.8426553349785963 |
| Mass-basis ideal density | SiO₂:Al₂O₃ = 0.7:0.3 mass parts; constituent densities 2.2 and 3.95 g/cm³; porosity 0.1; 0.5 mm | effective density = 2.283503649635037 g/cm³; T = 0.9081914999105459 |
| Volume-basis ideal density | SiO₂:Al₂O₃ = 0.6:0.4 volume parts; constituent densities 2.2 and 3.95 g/cm³; porosity 0.12; 0.8 mm | effective density = 2.552 g/cm³; T = 0.8459312080678261 |
| Solid density and porosity | Al₂O₃; solid density 3.95 g/cm³; porosity 0.25; 0.75 mm; 15° | effective density = 2.9625 g/cm³; T = 0.8366967450548580 |
| Parenthesized formula | Ca(OH)₂, independently expanded to Ca₁O₂H₂; 2.2 g/cm³; 0.2 mm | T = 0.9005902953326923 |
| Hydrate formula | CuSO₄·5H₂O, independently expanded to Cu₁S₁O₉H₁₀; 2.2 g/cm³; 0.2 mm | T = 0.8655495627665267 |
| Zero thickness | Al; 2.699 g/cm³; 0 mm | T = 1; no finite target-thickness multiplier |
| Two oblique layers | Al: 2.7 g/cm³, 0.2 mm; Fe: 7.8 g/cm³, 0.05 mm; both 45° | T = 0.5843855417249815 |

Across these cases, the largest difference from the independent scalar
implementation was **4.20 × 10⁻¹⁶ relative** in mass attenuation,
**5.75 × 10⁻¹⁶ relative** in nonzero optical depth, and
**1.12 × 10⁻¹⁶ absolute** in transmission. These are differences between two
implementations of the same model and data; they are not estimates of physical
accuracy. The arithmetic checks used a `2e-12` relative / `1e-14` absolute
floating-point comparison criterion selected for this review; no existing
project test or tolerance was altered.

Additional identity checks passed:

- Reversing the order of two layers preserves primary transmission.
- Splitting one Al slab into two equal-thickness slabs preserves transmission.
- A 0.4 mm Fe layer at 60° has the same attenuation as a 0.8 mm layer at normal
  incidence.
- Reference HVL, TVL and attenuation length equal `10 ln(2) / mu`,
  `10 ln(10) / mu` and `10 / mu`, respectively. They are **path lengths** in
  mm; an oblique layer's normal thickness is a different quantity.
- Applying the reported common target multiplier to all normal thicknesses
  produces the requested reference transmission of 0.1 for the nonzero cases.
- Component and elemental mass fractions, effective density and ray path agree
  with the independently constructed expectations.

## Comparison with directly checked NIST tables

The total attenuation column, **μ/ρ**, was read directly from NIST's
[Al](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html),
[Fe](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z26.html),
[Pb](https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z82.html) and
[liquid-water](https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html)
pages on the review date. NIST tabulates energies in MeV: 0.02, 0.05 and
0.1 MeV correspond to 20, 50 and 100 keV. The adjacent energy-absorption
column, **μ_en/ρ**, was not substituted for attenuation.

| Material | Energy · keV | NIST μ/ρ · cm²/g | Workbench μ/ρ · cm²/g | Relative difference · % |
|---|---:|---:|---:|---:|
| Al | 20 | 3.441 | 3.44187048850 | +0.02529754 |
| Al | 50 | 0.3681 | 0.368150072131 | +0.01360286 |
| Al | 100 | 0.1704 | 0.170417208478 | +0.01009887 |
| Fe | 20 | 25.68 | 25.6830231325 | +0.01177232 |
| Fe | 50 | 1.958 | 1.95738822607 | -0.03124484 |
| Fe | 100 | 0.3717 | 0.371723530972 | +0.00633064 |
| Pb | 20 | 86.36 | 86.3741133764 | +0.01634249 |
| Pb | 50 | 8.041 | 8.04199332153 | +0.01235321 |
| Pb | 100 | 5.549 | 5.54875387755 | -0.00443544 |
| H₂O | 20 | 0.8096 | 0.809831164602 | +0.02855294 |
| H₂O | 50 | 0.2269 | 0.226935737025 | +0.01575012 |
| H₂O | 100 | 0.1707 | 0.170723585217 | +0.01381676 |

The largest absolute relative difference was **0.03124484%**. All twelve
comparisons meet the project's existing **1% reference-agreement threshold**.
This is limited agreement with selected rounded off-edge tables; it does not
establish that the whole energy/material domain is accurate to either 1% or
0.03124484%. Different compilations may share theoretical inputs.

An additional Fe K-edge check used NIST's limiting values at 7.112 keV:
53.19 cm²/g below and 407.6 cm²/g above. At energies displaced by ±10⁻⁴
relative, the workbench returned 53.21149642697 cm²/g and
407.59806951457 cm²/g, differences of +0.0404144% and -0.000473623%
respectively. The displaced samples bracket the tabulated discontinuity;
they do not model XANES/EXAFS or validate real-material near-edge structure.

## Spectrum and reproducible example

For each supplied spectrum bin, an independent scalar calculation formed the
layer optical depths, then evaluated `exp(-sum(tau_layer))`. The photon-weighted
mean transmission was evaluated as `sum(W * T) / sum(W)`, and the
energy-weighted value as `sum(W * E * T) / sum(W * E)`. The weights are
**integrated photon fluences per bin**, not spectral density samples.

For `examples/borosilicate-multilayer.json`:

- Engine photon-weighted transmission: **0.7802626916899833**.
- Independent scalar photon-weighted transmission: **0.7802626916899835**.
- Energy-weighted transmission: **0.8619866648941318**; the engine agrees with
  the independently weighted sum.
- Incident and transmitted photon-fluence-weighted mean energies:
  **40.0** and **44.189562006464314 keV**.
- Direct reference transmission at 30 keV: **0.817370241763721**.
- Common thickness multiplier for target transmission 0.1:
  **11.417978438587083**.

The illustrative spectrum is a mathematical example, not a calibrated source
or measured research dataset. These calculations are reproducibility evidence,
not evidence that the application has already been used in a research paper.

## Bounded uncertainty check

Code inspection confirmed the intended mean-one lognormal multipliers:
`sigma_log = sqrt(log(1 + CV²))`, `mu_log = -sigma_log² / 2`. Density and
normal-thickness draws are independent, and one realization is used across the
energy sweep.

A separate analytic quantile check used a constructed Al layer at 30 keV,
2.699 g/cm³, 1 mm, 10% relative density standard deviation and zero thickness
uncertainty. With 20,000 samples and seed 14, the engine's central interval was
**[0.69118484, 0.77919263]**, compared with analytic population quantiles
**[0.69179304, 0.77940830]**. The largest absolute quantile difference was
**0.00060821** in transmission fraction, consistent with finite Monte Carlo
sampling in this example. This check used a review criterion of 0.005 absolute
for finite-sample quantiles; it did not change the automated suite's criteria.

This check supports the stated propagation under its assumptions. It does not
validate the user's chosen uncertainties or cover composition, angle,
cross-section errors, shared calibration error or correlations. The reported
intervals are pointwise, not simultaneous bands or experimental error bounds.

## Legacy error and implications for earlier calculations

Read-only inspection of
`legacy/src/X-ray_attenuation_calculator_multi_component.py` found the outer
mass-fraction weighting and a second weighting inside the density passed to
`material_mu`. Its resulting coefficient was

\[
\mu_\mathrm{old}=\rho\sum_i w_i^2(\mu/\rho)_i
\quad\text{instead of}\quad
\mu_\mathrm{correct}=\rho\sum_i w_i(\mu/\rho)_i.
\]

The archived program was **not executed or edited**. A separately constructed
50:50 mass Al/Fe example at 50 keV, density 4 g/cm³ and normal thickness
0.4 mm gave:

- Correct mass attenuation: **1.1627691490990975 cm²/g**.
- Double-weighted mass attenuation: **0.5813845745495487 cm²/g**.
- Correct primary transmission: **0.8302378409693760**.
- Transmission with that legacy double weighting: **0.9111738807545879**.

Thus the coefficient is halved in this equal-mass example, while transmission
changes exponentially. The effect is composition- and energy-dependent for
other mixtures. Earlier research must be rerun from its actual compositions,
densities, thicknesses, angles and energies; a generic percentage correction
cannot repair it. This review does not assess or certify any prior paper.

## Remaining validation before quantitative claims

No discrepancy was found in the specific algebraic checks above. That supports
correct implementation of the stated independent-atom, homogeneous,
narrow-beam model for the assessed cases. It does not prove correctness for
all allowed inputs or accuracy for an actual specimen or instrument.

Before relying on a new research rerun, the author should inspect its inputs
and exported configuration, compare representative energies/materials with
external references, and where appropriate compare against measured
transmission using the actual sample density, geometry, source spectrum and
detector acceptance. Extra attention is needed near absorption edges,
for porous/heterogeneous specimens and when scattered radiation reaches the
detector. Removed primary photons are not an absorbed-energy or dose result.

The [XrayDB documentation](https://xraypy.github.io/XrayDB/) describes
approximately 0.25-250 keV as the typically most reliable range for these
cross-sections. The workbench accepts 1-800 keV, so successful evaluation above
250 keV is not an accuracy guarantee. It excludes buildup, secondary transport,
fluorescence transport, coherent-interface/refraction effects, diffraction,
microstructure-dependent transport and chemical near-edge fine structure.

The companion automated suite passed **688 Python tests** and **26 JavaScript
tests** on 27 October 2026. Ruff and mypy also passed (22 configured source
files for typing), and the supplementary scalar check passed after being
preserved in the repository. These checks provide additional regression
coverage. Neither automated checks nor this AI-assisted audit substitute for
the human author review required by JOSS.
