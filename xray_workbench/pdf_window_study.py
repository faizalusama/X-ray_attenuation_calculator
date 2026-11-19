"""Compare finite-Q pair distribution functions on the same r grid."""
from __future__ import annotations

import csv
import io

import numpy as np

from ._types import Array, JsonObject
from .physics import _integer, _number

_MAX_Q_SAMPLES = 20_000
_MAX_QR_SAMPLES = 20_000_000


def _parse_sq(payload: JsonObject) -> tuple[Array, Array]:
    """Validate the complete S(Q) dataset before selecting either prefix."""
    if "q" in payload or "sq" in payload:
        try:
            q = np.asarray(payload.get("q"), dtype=float)
            sq = np.asarray(payload.get("sq"), dtype=float)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("q and S must be numeric arrays.") from exc
    else:
        text = payload.get("sq_data", "")
        if not isinstance(text, str) or len(text) > 800_000:
            raise ValueError("S(Q) data must be CSV text smaller than 800,000 characters.")
        pairs: list[tuple[float, float]] = []
        try:
            for row in csv.reader(io.StringIO(text), strict=True):
                if not row or row[0].lstrip().startswith("#"):
                    continue
                if [value.strip().lower() for value in row] == ["q", "s"]:
                    continue
                if len(row) != 2:
                    raise ValueError("S(Q) CSV rows must contain exactly q,S.")
                try:
                    pairs.append((float(row[0]), float(row[1])))
                except ValueError as exc:
                    raise ValueError("S(Q) CSV rows must contain numeric q,S values.") from exc
                if len(pairs) > _MAX_Q_SAMPLES:
                    raise ValueError("At most 20,000 S(Q) samples.")
        except csv.Error as exc:
            raise ValueError(f"Malformed S(Q) CSV: {exc}") from exc
        if len(pairs) < 3:
            raise ValueError("S(Q) needs at least three q,S samples.")
        samples = np.asarray(pairs, dtype=float)
        q, sq = samples[:, 0], samples[:, 1]

    if q.ndim != 1 or sq.shape != q.shape or not 3 <= len(q) <= _MAX_Q_SAMPLES:
        raise ValueError("q and S must be equal-length arrays with 3-20,000 samples.")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(sq)) or np.any((q < 0) | (q > 200)):
        raise ValueError("S(Q) data must be finite. Q must lie in 0-200 Å⁻¹.")
    if np.any(np.diff(q) <= 0):
        raise ValueError("Q must increase strictly. Sort and resolve duplicates before transforming.")
    return q, sq


def pdf_window_study(payload: JsonObject) -> JsonObject:
    """Transform two measured Q prefixes with one window choice and r grid."""
    from .scattering_tools import pdf_transform, plot, result

    q, sq = _parse_sq(payload)
    q_min, q_max = float(q[0]), float(q[-1])
    requested_a = _number(payload.get("qmax_a_inv_angstrom"), "Qmax A (Å⁻¹)", q_min, q_max)
    requested_b = _number(payload.get("qmax_b_inv_angstrom"), "Qmax B (Å⁻¹)", q_min, q_max)
    if requested_a >= requested_b:
        raise ValueError("Qmax A must be strictly less than Qmax B.")

    count_a = int(np.searchsorted(q, requested_a, side="right"))
    count_b = int(np.searchsorted(q, requested_b, side="right"))
    if count_a < 3:
        raise ValueError("Each Qmax must include at least three Q samples.")
    if count_a == count_b:
        raise ValueError("Both Qmax values select the same last Q sample. Choose distinct effective cutoffs.")
    effective_a, effective_b = float(q[count_a - 1]), float(q[count_b - 1])

    r_max = _number(payload.get("r_max_angstrom", 20), "Maximum r (Å)", 0.01, 1000)
    points = _integer(payload.get("r_points", 501), "r grid points", 3, 4000)
    if points * (count_a + count_b) > _MAX_QR_SAMPLES:
        raise ValueError("Use at most 20 million Qxr samples across both PDF curves.")
    window = payload.get("window", "none")
    if not isinstance(window, str) or window not in ("none", "lorch"):
        raise ValueError("Window must be none or lorch.")
    options: JsonObject = {"r_max_angstrom": r_max, "r_points": points, "window": window}

    # The longer prefix imposes both sampling constraints for the shared r grid.
    try:
        with np.errstate(over="raise", invalid="raise"):
            curve_b: JsonObject = pdf_transform({**options, "q": q[:count_b], "sq": sq[:count_b]})["data"]
            curve_a: JsonObject = pdf_transform({**options, "q": q[:count_a], "sq": sq[:count_a]})["data"]
    except FloatingPointError as exc:
        raise ValueError("The PDF transform exceeds the finite numerical range for the supplied S(Q).") from exc

    r: list[float] = curve_b["r_angstrom"]
    g_a: list[float] = curve_a["g_inv_angstrom2"]
    g_b: list[float] = curve_b["g_inv_angstrom2"]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("r_angstrom", "g_a_inv_angstrom2", "g_b_inv_angstrom2"))
    writer.writerows(zip(r, g_a, g_b, strict=True))

    data: JsonObject = {
        "q_min_inv_angstrom": q_min,
        "qmax_a_inv_angstrom": requested_a, "qmax_b_inv_angstrom": requested_b,
        "qmax_a_effective_inv_angstrom": effective_a, "qmax_b_effective_inv_angstrom": effective_b,
        "samples_a": count_a, "samples_b": count_b, "window": window,
        "r_angstrom": r, "g_a_inv_angstrom2": g_a, "g_b_inv_angstrom2": g_b,
        "csv": output.getvalue(),
    }
    answer = result(
        "pdf_window", data,
        [("Qmin", q_min, "Å⁻¹"),
         ("Requested Qmax A", requested_a, "Å⁻¹"), ("Effective Qmax A", effective_a, "Å⁻¹"),
         ("Requested Qmax B", requested_b, "Å⁻¹"), ("Effective Qmax B", effective_b, "Å⁻¹"),
         ("Samples A", count_a, ""), ("Samples B", count_b, "")],
        [plot("PDF at two Q cutoffs", "r (Å)", "G(r) (Å⁻²)", np.asarray(r),
              {f"A (Qmax = {effective_a:g} Å⁻¹)": np.asarray(g_a),
               f"B (Qmax = {effective_b:g} Å⁻¹)": np.asarray(g_b)})],
        ["Each requested cutoff snaps down to the last measured Q at or below it. Integration uses only "
         "samples from the measured Qmin through that effective cutoff.",
         "The input must be corrected, normalized S(Q). Both curves use the same r grid and window choice. "
         "The Lorch window, when selected, uses each curve's effective Qmax."],
    )
    answer["provenance"].update(
        optical_data="None (user-supplied normalized S(Q))",
        method="Wang et al. (2025), Eqs. (3) and (21), doi:10.1107/S1600576725004340",
    )
    return answer
