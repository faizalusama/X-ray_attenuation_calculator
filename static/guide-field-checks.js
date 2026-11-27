"use strict";

globalThis.WorkbenchGuideFieldChecks = {
  total: [
    {
      title: "Record the PDF instrument resolution",
      stage: "Measure",
      guidance: "Collect a crystalline reference such as nickel powder at the sample wavelength and detector geometry. Fit its PDF to obtain the instrument damping and broadening parameters, Qdamp and Qbroad.",
      check: "Save the reference run and fitted parameters with the sample PDFs. Repeat the calibration after a geometry change before comparing peak amplitudes and widths at larger r.",
      source: {label: "Liu et al. (2021), Pair Distribution Function Measurements", url: "https://doi.org/10.1002/advs.202003046"},
      target: null,
    },
    {
      title: "Check the short-distance limit of the PDF correction",
      stage: "Interpret",
      guidance: "In PDFgetX3, set the polynomial correction limit rpoly below the shortest expected bond distance. Reprocess with a nearby lower rpoly while keeping the transform settings fixed.",
      check: "Treat features below rpoly as unreliable. If the first assigned peak changes substantially, keep both reductions with the reported settings and flag the assignment for review.",
      source: {label: "Juhás et al. (2013), polynomial correction and Eq. (10)", url: "https://doi.org/10.1107/S0021889813005190"},
      target: null,
    },
  ],
  saxs: [
    {
      title: "Verify the overlap before merging SAXS curves",
      stage: "Interpret",
      guidance: "For SEC-SAXS and batch data from the same sample, put the curves on a common intensity scale over a shared q interval. Inspect the scaled profiles and their uncertainties throughout the overlap.",
      check: "Investigate systematic disagreement before merging. Save both input curves, the scale factor and the retained q ranges with the merged curve.",
      source: {label: "Trewhella, Vachette and Larsen (2024), §2.6, merging profiles", url: "https://doi.org/10.1107/S205225252400486X"},
      target: null,
    },
    {
      title: "Flag overlapping species in an SEC-SAXS run",
      stage: "Interpret",
      guidance: "Compare the SAXS elution trace with the UV trace after accounting for the delay between detectors. Mark any shoulders or overlapping peaks before selecting frames to average.",
      check: "If the species are poorly separated, flag the affected frames for peak decomposition or recollection. Save the UV trace and frame range used to produce each curve.",
      source: {label: "Matsui et al. (2024), SEC-SAXS peak separation and frame selection", url: "https://doi.org/10.1002/pro.4946"},
      target: null,
    },
  ],
  waxs: [
    {
      title: "Rule out a powder specimen height shift",
      stage: "Measure",
      guidance: "In Bragg–Brentano reflection geometry, check that the powder surface is flush with the holder’s reference plane. If several peaks shift together, reseat the specimen and repeat the scan.",
      check: "A height error shifts peak positions by an amount that depends on angle. Compare several reflections after reseating before attributing the original offsets to a lattice change.",
      source: {label: "Gateshki et al. (2024), Bragg–Brentano specimen displacement", url: "https://doi.org/10.1107/S1600576724006460"},
      target: null,
    },
    {
      title: "Save the area-detector integration settings",
      stage: "Interpret",
      guidance: "Keep the raw frames, calibrant image, fitted detector geometry and pixel mask with the integrated pattern. Record the radial units, binning and azimuthal range used for integration.",
      check: "Reintegrate one frame with the saved settings and compare its peak positions and intensity profile with the exported pattern.",
      source: {label: "Ashiotis et al. (2015), pyFAI detector geometry, calibration and integration", url: "https://doi.org/10.1107/S1600576715004306"},
      target: null,
    },
  ],
  tomo: [
    {
      title: "Check for clipped projections during a trial rotation",
      stage: "Plan",
      guidance: "For a full-object scan, take trial views through the rotation and check that the specimen silhouette stays inside the illuminated detector area, including at its widest orientation.",
      check: "If an edge leaves the field, change the centering or field of view, or plan overlapping acquisitions. Extra angular views alone leave the missing ray paths unmeasured.",
      source: {label: "Du et al. (2018), limited field of view and overlapping acquisitions", url: "https://doi.org/10.1364/JOSAA.35.001871"},
      target: null,
    },
    {
      title: "Hand off measured angles with the projection stack",
      stage: "Measure",
      guidance: "Save the measured rotation angle and its unit for each raw projection. Keep dark and flat images in separately identified groups and record where they were collected in the scan.",
      check: "Confirm that each retained projection has its own angle entry. When dropping a failed frame, drop the matching angle entry and preserve the original frame identifier.",
      source: {label: "De Carlo et al. (2014), §4, Data Exchange for full-field tomography", url: "https://doi.org/10.1107/S160057751401604X"},
      target: null,
    },
  ],
  optics: [
    {
      title: "Measure background beside the specular reflection",
      stage: "Measure",
      guidance: "On area-detector frames, integrate a same-size region beside the reflected spot to estimate the background. Use the same normalization for both regions before subtracting.",
      check: "Check that the background region clears the specular spot and detector gaps. Flag angles where the corrected signal is comparable to its uncertainty before fitting the curve.",
      source: {label: "Mocuta et al. (2018), §4, specular and offset integration regions", url: "https://doi.org/10.1107/S1600577517015703"},
      target: null,
    },
    {
      title: "Check reflectivity joins between attenuator settings",
      stage: "Measure",
      guidance: "When changing attenuators during a reflectivity scan, leave an angular overlap where both settings give usable signals in the detector’s linear range. Record each attenuator’s calibrated transmission at the scan energy.",
      check: "Apply the attenuation correction to both segments and compare them in the overlap. Investigate a step at the join before fitting the combined scan.",
      source: {label: "Mocuta et al. (2018), §3, calibrated attenuators and scan segments", url: "https://doi.org/10.1107/S1600577517015703"},
      target: null,
    },
  ],
  attenuation: [
    {
      title: "Test a nominally monochromatic beam with a foil series",
      stage: "Measure",
      guidance: "Measure narrow-beam transmission T through several thicknesses of the same foil material at fixed nominal energy. Plot −ln(T) against total foil thickness.",
      check: "A straight line is expected for a monoenergetic beam and linear detector. If the curve bends, check detector linearity separately and repeat with stronger harmonic rejection if available.",
      source: {label: "Chantler et al. (2001), detector linearity and harmonic contribution", url: "https://doi.org/10.1103/PhysRevA.64.062506"},
      target: null,
    },
    {
      title: "Check stack transmission against individual layers",
      stage: "Interpret",
      guidance: "At one photon energy, multiply the individual narrow-beam transmissions as fractions and compare with the measured stack transmission. Use the same beam path through each layer when measuring it alone and in the stack.",
      check: "Use Materials to review the predicted layer transmissions at the reference energy. If the stack disagrees with the product of measured transmissions, check whether assembly changed the illuminated region or clipped the beam.",
      source: {label: "Hubbell and Seltzer, NIST narrow-beam attenuation law, Eq. (1)", url: "https://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html"},
      target: "tab:materials",
    },
  ],
};
