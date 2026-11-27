"""Dark- and flat-field correction for a single transmission tomography ray."""
from math import isfinite, log

from ._types import JsonObject
from .physics import _number


def projection(payload: JsonObject) -> JsonObject:
    from .scattering_tools import result

    sample = _number(payload.get("sample_counts", 550), "Sample counts", minimum=0)
    flat = _number(payload.get("flat_counts", 1050), "Flat counts", minimum=0)
    dark = _number(payload.get("dark_counts", 50), "Dark counts", minimum=0)
    if flat <= dark:
        raise ValueError("Flat counts must exceed dark counts for normalization.")
    if sample <= dark:
        raise ValueError("Sample counts must exceed dark counts for the logarithm.")

    transmission = (sample - dark) / (flat - dark)
    if transmission == 0 or not isfinite(transmission):
        raise ValueError("Transmission must be representable as a positive finite number.")
    line_integral = -log(transmission)

    notes = ["Use sample, open-beam flat, and dark counts from the same detector pixel with matched exposures.",
             "For a monochromatic ray with negligible scatter, −ln(T) is the attenuation line integral."]
    if transmission > 1:
        notes.append("Caution: T > 1 gives a negative projection. Check the counts and exposure matching.")
    data = {"sample_counts": sample, "flat_counts": flat, "dark_counts": dark,
            "transmission": transmission, "line_integral": line_integral}
    answer = result("projection", data,
                    [("Transmission (T)", transmission, "1"), ("Projection (−ln T)", line_integral, "1")],
                    [], notes)
    answer["provenance"].update({
        "optical_data": "Not used",
        "method": "Gürsoy et al. (2014), §4.1, Eq. (1), doi:10.1107/S1600577514013939",
    })
    return answer
