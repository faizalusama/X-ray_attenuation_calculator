# Scattering tools (unreleased PR7 candidate)

These tools are separate from primary-beam attenuation. Reference energy is keV.
Optical grazing angles are from the **surface**; attenuation incidence is from
the normal. Every tool exports its inputs/results as JSON; plotted curves export CSV.

## Detector coverage and tomography in the Technique tools tab

The live experiment view draws detector geometry using a common distance
scale and tomography bars using a common count scale. Its sliders edit the
same numerical inputs as the tool. Arrow keys and the Home/End keys work.
The detector's 3D view rotates and tilts using the sliders or pointer drag.
Circular outlines mark radial limits on an ideal plane. A camera movement
changes the view while the same probe remains selected. Side view gives
an orthogonal cross-section using the same inputs and calculated sample.
The detector radius probe selects one of the calculated curve points and
shows its radius, scattering angle and Q. Model and exact values expands
the current numeric readout. Save diagram SVG exports the drawing with its
current theme colours and an accessible description.

For SAXS and WAXS, enter the sample-to-detector distance L and two usable
beam-centred radii R in mm. For an untilted flat detector, the scattering
angle is 2θ = atan2(R,L). At wavelength λ in Å the elastic momentum transfer
is q = 4π sin(θ)/λ in Å⁻¹. The result includes q and angle limits, plus a
radius-to-q curve for export. Use measured masks and beamstop shadows to
select usable radii. An outer corner can receive part of a ring even where
full azimuthal integration is impossible. See [pyFAI geometry, qFunction](https://www.silx.org/doc/pyFAI/2025.12/api/geometry/index.html).

For a transmission tomography ray, enter sample S, flat F and dark D counts
from matched exposures at the same pixel. The tool calculates T = (S−D)/(F−D)
and p = −ln(T). In monochromatic absorption tomography p estimates the
dimensionless integral of μ along the ray. Both corrected counts must be
positive. Counts above the flat give a negative projection which is retained
in the results and JSON export. Starting values are illustrative. See
[TomoPy normalize and minus_log](https://tomopy.readthedocs.io/en/stable/_modules/tomopy/prep/normalize.html).

## Optical constants and penetration depth

The live surface schematic uses grazing angle from the surface. Its probe
reads the angle and normal intensity 1/e depth from the calculated curve.
The illustrative ray diagram and numeric depth readout have separate scales.
The PDF schematic shows the supplied Q interval and resulting r interval.
Its probe reads S(Q) and the windowed transform integrand at an input sample;
the transform still uses the whole supplied interval. Both diagrams support
keyboard probing, exact readouts and SVG export.

Element number densities N_j follow from effective density and mass fractions.
SLD_x = r_e sum N_j (Z_j + f'_j - i f''_j), using Chantler through XrayDB.
delta and beta are lambda²/(2 pi) times the real and absorptive SLD magnitudes.
The convention is n=1-delta+i beta and exp(i kz z), where
kz=(2 pi/lambda) sqrt(n²-cos² alpha). Normal **intensity** 1/e penetration depth
is 1/[2 Im(kz)], converted from Å to nm. Approximate critical angle is sqrt(2 delta);
it is undefined for nonpositive delta. Optical data are not blended with Elam.

## Multilayer reflectivity

Vacuum is the incident medium. The stack runs top to bottom; the **last** layer
is a semi-infinite substrate. Earlier layers are films. Film overrides are in Å;
blank uses the stack's normal thickness in mm converted to Å.
A thick attenuation sample may have fringes too dense for the displayed angular grid.

Parratt recursion combines Fresnel amplitudes with exp(2 i kz d).
Névot–Croce multiplies each interface amplitude by exp(-2 kz_j kz_(j+1) sigma²).
RMS roughness is uniform across interfaces. s or p polarization is explicit.
Resolution, footprint, diffuse scattering, fitting and overlapping graded interfaces
are excluded. Nonphysical reflectance is refused, not clipped.

## SLD contrast and Porod invariant

Choose the primary layer, second phase and an independent volume fraction phi.
Nuclear neutron SLD uses natural-isotope Sears lengths; missing data stay undefined.

For an isotropic, homogeneous, sharp two-phase material:
integral q² I(q) dq = 2 pi² phi(1-phi)|SLD_1-SLD_2|², in Å⁻⁴.
This is an absolute-unit model, not an integral inferred from a finite measured
curve. Magnetic scattering and neutron resonances are excluded.

## Diffraction conversion and Scherrer size

Q=4 pi sin(theta)/lambda=2 pi/d; s=sin(theta)/lambda=Q/(4 pi).
2theta is a scattering angle. Target energy converts the same d spacing to a
second source; inaccessible angles remain undefined. At Q=0, d is undefined.

Scherrer L=K lambda/[beta cos(theta)] is returned in nm. beta is FWHM in
2theta radians after **Gaussian** quadrature subtraction of instrument width.
Zero corrected width gives undefined size. It estimates coherent-domain size,
not particle size; strain and detailed profile effects are excluded.

## Powder peaks

Supply a,b,c in Å, alpha,beta,gamma in degrees, and the **complete unit cell**.
Rows are element,x,y,z,occupancy,B_angstrom2, with fractional coordinates.
There is no space-group expansion or structure inference.

F_hkl=sum occupancy_j f0_j(s) exp(-B_j s²) exp[2 pi i (h x_j+k y_j+l z_j)].
The reciprocal metric gives d_hkl and Bragg geometry gives peak positions.
Indices within the specified bound, including Friedel pairs, are enumerated.
Equal-d groups sum intensities. LP=(1+cos²2theta)/(2 sin²theta cos(theta))
for unpolarized Bragg–Brentano powder geometry. The plotted sticks are relative
integrated intensities. Texture, absorption and line broadening are excluded.
Default diamond-Si is an editable demonstration, not an inferred sample structure.

## S(Q) to G(r)

Paste/import q,S CSV: Q in Å⁻¹, dimensionless, already normalized S(Q).
Unsorted, duplicate and nonfinite data are rejected. No extrapolation is performed.

F(Q)=Q[S(Q)-1]; G(r)=(2/pi) integral F(Q) M(Q) sin(Qr) dQ.
Lorch M(Q)=sin(pi Q/Qmax)/(pi Q/Qmax), including its continuous limit at zero.
Trapezoidal integration supports nonuniform Q with explicit r sampling limits.
Finite-Q ripples, missing low Q and noise remain. This does not reduce raw intensities.

The labelled analytic demo is F(Q)=exp(-Q) sin(3Q). Its known infinite-range
transform is [1/(1+(3-r)²)-1/(1+(3+r)²)]/pi.
See [VALIDATION.md](VALIDATION.md) for numerical reference checks.

## PDF Qmax comparison

Supply the same normalized q,S data for both transforms. Qmax A must be
less than Qmax B. A requested cutoff between samples uses the last measured
Q at or below it. Each curve uses its own effective Qmax in the Lorch
window and both curves share the same r grid. CSV exports include both
curves. The starter data are synthetic. Finite-Q changes in peak shape
and ripples can be compared directly using the method of
[Wang et al. (2025), equations (3) and (21)](https://doi.org/10.1107/S1600576725004340).

## References

- [XrayDB optical and atomic data](https://xraypy.github.io/XrayDB/python.html).
- [Reflectometry interface roughness](https://www.reflectometry.org/learn/3_reflectometry_slab_models/roughness_and_microslicing.html).
- [IUCr structure-factor convention](https://dictionary.iucr.org/Structure_factor).
- [NIST neutron lengths](https://www.ncnr.nist.gov/resources/n-lengths/list.html).
