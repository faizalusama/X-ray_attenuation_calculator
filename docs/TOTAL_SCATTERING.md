# Total scattering section

The **Total scattering** tab gives the numbers needed to plan and interpret an X-ray total-scattering (pair distribution function) measurement. It uses the composition and density of a layer in the stack and the reference energy chosen by the X-ray source, energy or wavelength input. **It does not reduce measured data.**

## What it reports

| Quantity | How it is obtained |
|---|---|
| Qmax | 4π sin(θmax)/λ |
| r-grid step | π/Qmax, the Nyquist step (Farrow et al. 2011) |
| Peak width from Qmax | FWHM of the peak a finite Qmax imposes: 3.791/Qmax without a window, 5.437/Qmax with a Lorch window. Both are computed from the integral in `tests/test_scattering.py`. |
| Number density ρ₀ | ρ N_A Σ(wᵢ/Aᵢ), in atoms/Å³ |
| G(r) slope at low r | −4πρ₀ |
| f₀(Q) | Waasmaier & Kirfel (1995), neutral atoms, through XrayDB. The fits are valid for sinθ/λ from 0 to 6 Å⁻¹, so curves stop at Q = 75.4 Å⁻¹. |
| f′, f″ | Chantler (2000), through XrayDB, at the current energy. Not available above uranium. |
| ⟨f⟩², ⟨f²⟩, Laue term | Atomic-fraction averages of f₀(Q) |
| Pair weights | wᵢⱼ(Q) = (2 − δᵢⱼ) cᵢcⱼfᵢfⱼ/⟨f⟩², using f₀ alone. They sum to 1 at every Q. |
| μ, μD, μR, transmission | The calculator's own Elam-based attenuation, with the path equal to the capillary diameter |
| Best flat-plate thickness | 1/μ, which maximizes t·exp(−μt) at normal incidence |
| Compton energy at 2θmax | E/(1 + (E/mₑc²)(1 − cos 2θ)) |
| Incoherent share | The incoherent fraction of μ/ρ at the current energy, integrated over angle |
| Fluorescence | K and L3 edges below the beam, with each shell's strongest line, if that line is at or above 1 keV |

## What it does not do

- It does not model the sample's structure or calculate S(Q) or G(r) from measured data.
- Pair weights ignore f′ and f″, so they do not apply near an absorption edge. The tab warns when the beam is within 2% of a K or L3 edge. The 2% is a display threshold, not a physical limit.
- Form factors are for independent, neutral atoms, without oxidation state or bonding.
- Compton scattering is not given as a function of Q. XrayDB has no such table, and adding one would mean importing and checking another dataset.
- μD and transmission describe the beam path only. They are not an absorption correction, and multiple scattering, polarization and container scattering are not included.
- A source row is one energy, not a tube spectrum. The 60, 80 and 100 keV rows are examples, not particular beamlines.
- It covers X-rays only. Neutron scattering lengths are not included.

## How it was checked

`tests/test_scattering.py` checks these values against hand results and independent data:

- Number densities of silica, water, silicon and copper, and silica's pair weights at Q = 0.
- Cu Kα1 and Mo Kα1 Qmax at backscattering, the Compton formula, and the flat-plate optimum by direct numerical maximization.
- Form factors against the separate Hubbell tabulation in xraylib. Up to Q = 25 Å⁻¹ the largest differences are 0.75% (O), 0.48% (Si), 1.8% (Fe), 3.6% (Hf) and 4.8% (Pb). Up to Q = 50.7 Å⁻¹ they are 5.0%, 3.5%, 6.1%, 9.9% and 11.4%. The test is skipped when xraylib is not installed.
- f′ and f″ against xraylib for Cu, Hf and Pb: f′ within 0.7%, 4.5% and 5.9%, and f″ within 3.6%, 2.1% and 1.3%. xraylib's f″ has the opposite sign convention. A small f′ can differ more in relative terms: Zr at 60 keV differs by 15%, which is 0.02 electrons.
- Attenuation against the calculator's own result, and the local server against the browser version.

Every reference in the tab was resolved against Crossref or OpenAlex on 1 November 2026. Authors' reduction programs may use different form-factor tables, so small differences from their output are expected.
