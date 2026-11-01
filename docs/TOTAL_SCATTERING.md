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
| f₀(Q) | Waasmaier & Kirfel (1995), neutral or selected ionic fits, through XrayDB. The fits are valid for sinθ/λ from 0 to 6 Å⁻¹, so curves stop at Q = 75.4 Å⁻¹. |
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
- Ionic choices affect f0 and pair weights only. They neither enforce charge neutrality nor model bonding. The Compton/coherent cross-check retains neutral-atom data.
- Compton curves use the checked Hubbell incoherent scattering function and Klein–Nishina recoil. This atomic incoherent function differs from structural S(Q). Binding profiles and detector response are excluded.
- Direct transmission is the central beam path. The additional disc integral assumes full uniform illumination, an axis perpendicular to the scattering plane and no wall. muR>30 is withheld. Thomson polarization factors are separate: (1+cos²2θ)/2 unpolarized, cos²2θ in-plane linear, and 1 perpendicular linear. Multiple scattering remains excluded.
- A source row is one energy, not a tube spectrum. The 60, 80 and 100 keV rows are examples, not particular beamlines.
- Natural-isotope neutron weights use Sears/NIST bound lengths. Negative weights are physical. Normalized weights are undefined near zero mean length. Thermal absorption values are not an energy-dependent neutron transport model.

## How it was checked

`tests/test_scattering.py` checks these values against hand results and independent data:

- Number densities of silica, water, silicon and copper, and silica's pair weights at Q = 0.
- Cu Kα1 and Mo Kα1 Qmax at backscattering, the Compton formula, and the flat-plate optimum by direct numerical maximization.
- Form factors against the separate EPDL97 tabulation in xraylib. Up to Q = 25 Å⁻¹ the largest differences are 0.75% (O), 0.48% (Si), 1.8% (Fe), 3.6% (Hf) and 4.8% (Pb). Up to Q = 50.7 Å⁻¹ they are 5.0%, 3.5%, 6.1%, 9.9% and 11.4%. The test is skipped when xraylib is not installed.
- f′ and f″ against xraylib's EPDL97 values for Cu, Hf and Pb: f′ within 0.7%, 4.5% and 5.9%, and f″ within 3.6%, 2.1% and 1.3%. xraylib's f″ has the opposite sign convention. A small f′ can differ more in relative terms: Zr at 60 keV differs by 15%, which is 0.02 electrons.
- Attenuation against the calculator's own result, and the local server against the browser version.

Every reference in the tab was resolved against Crossref or OpenAlex on 1 November 2026. Authors' reduction programs may use different form-factor tables, so small differences from their output are expected.

## Added checks

The SiO2 differential integrals at 20 keV differ from Elam by +0.01097% coherent
and +0.39338% incoherent, reported without forced normalization. Full-angle
integration is unavailable beyond the f0 fit domain. Neutron weights sum to one
when defined; H–O has a negative weight. Fe2+ gives f0(0) about 24 electrons,
without changing attenuation. Chantler edge scans do not resolve chemical XANES.

Bundled sources: Hubbell (1975), doi:10.1063/1.555523, incoherent functions via
xraylib; Sears (1992), doi:10.1080/10448639208218770, via
https://www.ncnr.nist.gov/resources/n-lengths/list.html.
The separate tools are described in [SCATTERING_TOOLS.md](SCATTERING_TOOLS.md).
