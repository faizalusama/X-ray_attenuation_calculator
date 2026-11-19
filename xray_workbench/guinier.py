"""Fixed-interval Guinier analysis of background-corrected SAXS intensity."""

import csv
import io
from math import exp, isfinite, sqrt

import numpy as np

from ._types import Array, JsonObject
from .physics import _number

_MAX_CSV_CHARS = 800_000
_MAX_ROWS = 20_000
_GLOBULAR_QRG_LIMIT = 1.3


def _selected_rows(text: str, qmin: float, qmax: float) -> tuple[Array, int]:
    reader = csv.reader(io.StringIO(text.removeprefix("\ufeff")), strict=True)
    selected: list[tuple[float, ...]] = []
    width = count = 0
    try:
        for row in reader:
            cells = [cell.strip() for cell in row]
            if not cells or cells == [""] or cells[0].startswith("#"):
                continue
            if not width and [cell.lower() for cell in cells] in (["q", "i"], ["q", "i", "sigma"]):
                width = len(cells)
                continue
            width = width or len(cells)
            line = reader.line_num
            if width not in (2, 3) or len(cells) != width:
                raise ValueError(f"CSV line {line} must match the q,I or q,I,sigma column layout.")
            count += 1
            if count > _MAX_ROWS:
                raise ValueError(f"Intensity CSV accepts at most {_MAX_ROWS:,} data rows.")
            try:
                sample = tuple(float(cell) for cell in cells)
            except ValueError as exc:
                raise ValueError(f"CSV line {line} must contain numeric values.") from exc
            q = _number(sample[0], f"q on CSV line {line}", minimum=0)
            if qmin <= q <= qmax:
                if not all(value > 0 and isfinite(value) for value in sample):
                    raise ValueError(f"Selected q, intensity, and any sigma must be finite and positive on CSV line {line}.")
                selected.append(sample)
    except csv.Error as exc:
        raise ValueError(f"Malformed intensity CSV near line {reader.line_num}.") from exc
    if len(selected) < 3:
        raise ValueError("The Guinier interval must contain at least three data rows.")
    selected.sort(key=lambda sample: sample[0])
    return np.array(selected, dtype=np.float64), count


def guinier(payload: JsonObject) -> JsonObject:
    """Fit ln(I) = ln(I(0)) - q² Rg² / 3 within an explicit inclusive q range.

    Supply q in Å⁻¹ and background-corrected I in consistent units. An optional
    sigma column gives one standard deviation in the same units as I. Use
    equal weights without sigma and inverse first-order log variance
    1/(sigma/I)² with sigma. The fit treats points as independent.
    """
    from .scattering_tools import plot, result

    qmin = _number(payload.get("qmin_inv_angstrom"), "Minimum q (Å⁻¹)", minimum=0)
    qmax = _number(payload.get("qmax_inv_angstrom"), "Maximum q (Å⁻¹)", minimum=0)
    if qmin >= qmax:
        raise ValueError("Maximum q must exceed minimum q.")
    text = payload.get("intensity_data")
    if not isinstance(text, str) or len(text) > _MAX_CSV_CHARS:
        raise ValueError(f"Intensity data must be q,I or q,I,sigma CSV text of at most {_MAX_CSV_CHARS:,} characters.")

    rows, count = _selected_rows(text, qmin, qmax)
    q, intensity = rows[:, :2].T
    sigma = rows[:, 2] if rows.shape[1] == 3 else None
    weighting = "1/(sigma/I)^2" if sigma is not None else "equal"
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            x = np.square(q)
            span = float(x[-1] - x[0])
            if span == 0:
                raise ValueError("Selected q values must span a nonzero range in q².")
            log_i = np.log(intensity)
            sigma_log = sigma / intensity if sigma is not None else None
            weights = np.ones_like(q) if sigma_log is None else np.square(sigma_log.min() / sigma_log)
            if np.any(weights == 0):
                raise ValueError("Selected sigma/I ratios span too large a range for the weighted fit.")

            # Scale q² for the regression. Referencing ln(I) to the first point
            # also keeps a constant input exactly flat after centering.
            scaled_x = (x - x[0]) / span
            relative_y = log_i - log_i[0]
            mean_x = float(np.average(scaled_x, weights=weights))
            mean_y = float(np.average(relative_y, weights=weights))
            centered_x = scaled_x - mean_x
            weighted_x = weights * centered_x
            variance = float(np.dot(weighted_x, centered_x))
            if variance <= 0:
                raise ValueError("Selected q values and weights do not resolve a slope.")
            scaled_slope = float(np.dot(weighted_x, relative_y - mean_y)) / variance
            slope = scaled_slope / span
            if not isfinite(slope):
                raise ValueError("The Guinier slope must be representable as a finite number.")
            if slope >= 0:
                raise ValueError("The Guinier slope must be negative to estimate Rg.")

            intercept = float(log_i[0]) + (mean_y - scaled_slope * mean_x) - slope * float(x[0])
            i0 = exp(intercept)
            rg = sqrt(-slope) * sqrt(3)
            fit_relative = mean_y + scaled_slope * centered_x
            fitted = fit_relative + log_i[0]
            residuals = relative_y - fit_relative
            rms = float(np.sqrt(np.mean(np.square(residuals))))
            reduced_chi_squared = (
                float(np.sum(np.square(residuals / sigma_log)) / (len(q) - 2))
                if sigma_log is not None else None
            )
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("Selected data exceed the numerical range of the Guinier fit.") from exc
    qmin_rg, qmax_rg = float(q[0]) * rg, float(q[-1]) * rg
    if i0 <= 0 or not isfinite(i0) or not isfinite(qmax_rg):
        raise ValueError("I(0) and qRg must be representable as finite positive numbers.")

    data: JsonObject = {
        "rg_angstrom": rg, "i0": i0, "slope_angstrom2": slope, "ln_i0": intercept,
        "qmin_inv_angstrom": qmin, "qmax_inv_angstrom": qmax,
        "qmin_used_inv_angstrom": float(q[0]), "qmax_used_inv_angstrom": float(q[-1]),
        "qmin_rg": qmin_rg, "qmax_rg": qmax_rg,
        "input_rows": count, "selected_rows": len(q), "weighting": weighting,
        "q_inv_angstrom": q.tolist(), "intensity": intensity.tolist(),
        "sigma": sigma.tolist() if sigma is not None else None,
        "ln_intensity_fit": fitted.tolist(), "ln_intensity_residuals": residuals.tolist(),
        "rms_ln_intensity_residual": rms, "reduced_chi_squared_log": reduced_chi_squared,
    }
    notes = [
        "Supply background-corrected intensity I(q) with q in Å⁻¹. I(0) uses the input intensity units.",
        "The fit uses every row inside the inclusive q interval. Inspect the log-intensity residuals for curvature.",
        "A common upper guide for compact globular particles is qmax Rg ≈ 1.3. Other shapes can require a smaller limit.",
    ]
    if sigma is not None:
        notes.append("Weights are 1/(sigma/I)² using first-order, independent log errors. Reduced χ² is in log space.")
    else:
        notes.append("All points have equal weight in log space.")
    if qmax_rg > _GLOBULAR_QRG_LIMIT:
        notes.append(f"The last fitted point has qRg = {qmax_rg:.6g}, above the usual globular guide of 1.3.")
    answer = result("guinier", data,
                    [("Radius of gyration (Rg)", rg, "Å"), ("Forward intensity I(0)", i0, "input units"),
                     ("Fitted rows", len(q), ""), ("Maximum fitted qRg", qmax_rg, "")],
                    [plot("Guinier fit", "q² (Å⁻²)", "ln I(q)", x, {"Measured": log_i, "Fit": fitted}),
                     plot("Guinier residuals", "q² (Å⁻²)", "Measured ln I(q) − fitted ln I(q)", x,
                          {"Residual": residuals})],
                    notes)
    answer["provenance"].update({
        "optical_data": "Not used",
        "input_data": "User-supplied background-corrected intensity CSV",
        "method": "Fixed-interval least squares of ln(I) versus q² using the Guinier approximation",
        "method_reference": "Trewhella, Vachette and Larsen (2024), §2.2, doi:10.1107/S205225252400486X",
        "weighting": weighting,
    })
    return answer
