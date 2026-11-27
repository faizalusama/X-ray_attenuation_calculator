"""Angular and object-space pixel sampling for parallel-beam tomography."""
from math import ceil, isfinite, pi

from ._types import JsonObject
from .physics import _number


def tomo_sampling(payload: JsonObject) -> JsonObject:
    """Estimate the Crowther view count for a target full spatial period."""
    from .scattering_tools import result

    diameter = _number(payload.get("diameter_um"), "Diameter (µm)", positive=True)
    resolution = _number(payload.get("resolution_um"), "Resolution (µm)", positive=True)
    pixel_pitch = _number(payload.get("pixel_pitch_um"), "Effective pixel pitch (µm)", positive=True)

    # Du et al. Eq. (6) gives 2D/d samples across the support at period d.
    # Eq. (7) multiplies that number by pi/2 before rounding up to whole views.
    view_requirement = pi * (diameter / resolution)
    if not isfinite(view_requirement):
        raise ValueError("The Crowther view requirement must be finite.")
    minimum_views = max(1, ceil(view_requirement))
    rotation_range = 180
    angular_step = rotation_range / minimum_views

    nyquist_resolution = 2 * pixel_pitch
    if not isfinite(nyquist_resolution):
        raise ValueError("The pixel sampling resolution floor must be finite.")
    pitch_supports_resolution = resolution >= nyquist_resolution

    data: JsonObject = {
        "diameter_um": diameter, "resolution_um": resolution, "pixel_pitch_um": pixel_pitch,
        "rotation_range_deg": rotation_range, "minimum_views": minimum_views, "angular_step_deg": angular_step,
        "nyquist_resolution_um": nyquist_resolution, "pitch_supports_resolution": pitch_supports_resolution,
    }
    values: list[tuple[str, int | float, str]] = [
        ("Minimum angular views", minimum_views, "views"),
        ("Angular step at minimum", angular_step, "°"),
        ("Pixel sampling resolution floor", nyquist_resolution, "µm"),
    ]
    notes = [
        "Assumes ideal parallel-beam projections of a circular support centered on the rotation axis.",
        "Views are uniformly spaced over [0°, 180°), with a step of 180° divided by the number of views.",
        "Resolution is a full spatial period and requires two samples at the effective object-space pixel pitch.",
        "These are ideal sampling bounds. Achieved resolution also depends on blur, noise, motion and reconstruction.",
    ]
    if not pitch_supports_resolution:
        notes.append("The target period is below the pixel sampling floor and remains undersampled at any view count.")

    raw_planned_views = payload.get("planned_views")
    if isinstance(raw_planned_views, str) and not raw_planned_views.strip():
        raw_planned_views = None
    if raw_planned_views is not None:
        planned_count = _number(raw_planned_views, "Planned views", minimum=1)
        if not planned_count.is_integer():
            raise ValueError("Planned views must be an integer.")
        planned_views = raw_planned_views if isinstance(raw_planned_views, int) else int(planned_count)
        planned_step = rotation_range / planned_views
        shortfall = max(0, minimum_views - planned_views)
        data.update(planned_views=planned_views, planned_angular_step_deg=planned_step,
                    planned_views_meet_minimum=shortfall == 0, planned_view_shortfall=shortfall)
        values.extend([("Planned views", planned_views, "views"),
                       ("Planned angular step", planned_step, "°"),
                       ("Additional views to meet angular minimum", shortfall, "views")])

    answer = result("tomo_sampling", data, values, [], notes)
    answer["provenance"].update(
        optical_data="None (sampling geometry only)",
        method="Du et al. (2021), §3.2, Eqs. (6)–(7), doi:10.1107/S1600576721000194",
        original_reference="Crowther, DeRosier & Klug (1970), doi:10.1098/rspa.1970.0119",
    )
    return answer
