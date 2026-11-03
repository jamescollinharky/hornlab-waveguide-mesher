"""Config normalization and config-driven mesh build orchestration.

This module owns the conversion from external TOML/JSON/imported ATH config
names into profile parameters, `PointGridHornGeometry`, `MeshDensity`, and the
final `BuildResult`. The CLI imports these helpers but does not own this
translation layer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import cost
from .config_parser import ConfigError
from .freeform import build_freeform_geometry, _validate_freeform_config
from .geometry import (
    HornEnclosure,
    HornInterface,
    MeshDensity,
    PointGridHornGeometry,
    _StretchedPointGridHornGeometry,
    validate_mesh_density,
)
from .mesher import MesherError, TriangleBudgetExceeded, build_mesh_with_info
from .profile_common import (
    _normalise_formula as _normalise_formula_common,
    _normalise_quadrants as _normalise_quadrants_common,
    _parse_number_list,
    _symmetry_planes_for_quadrants as _symmetry_planes_for_quadrants_common,
)
from .profile_sampling import (
    ACOUSTIC_AXIAL_STATIONS_KEY,
    ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY,
    ACOUSTIC_MORPH_START_KEY,
    MORPH_KEEPS_SLOT_KEY,
    build_point_grid_arrays,
    FREEFORM_CONTINUOUS_COLLAPSE_KEY,
    GCURVE_CROSS_SECTION_CONFLICT,
    _classify_zmap_kind,
    _cross_section_is_circular,
    freeform_azimuth_twist_note,
)
from .profiles import azimuthal_mean, build_point_grid, eval_param
from .builders.point_grid_freestanding import (
    _freestanding_mouth_closure_points,
    _outer_wall_axial_ring_indices,
    _restored_outer_throat_points,
)
from .builders.point_grid_surfaces import _rear_rim_points
from .tags import PhysicalGroup
from .throat_stretch import COMPOSITION_PROFILE_KEYS, COMPOSITION_GUIDE_KEYS, canonical_stretch_params, validate_stretch_composition, stretch_config_errors, stretch_is_inactive, validate_supplied_stretch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildResult:
    mesh_path: Path
    formula: str
    mode: str
    n_vertices: int
    n_triangles: int
    units: str
    physical_groups: dict[int, str]
    # Quadrant coverage of the built grid and the solver symmetry flag a
    # reduced mesh requires (hornlab-metal-bem SolveConfig.native_symmetry_plane).
    quadrants: str = "1234"
    native_symmetry_plane: str | None = None
    # Whether the metal solver's cut-plane open-edge guard applies. Bare horns
    # are open shells whose free mouth rims are legitimate solve boundaries, so
    # the guard must be relaxed (hornlab-metal-bem
    # SolveConfig.native_check_open_edges=False). Closed/coupled modes cap the
    # mouth and keep the strict check.
    native_check_open_edges: bool = True
    # Realized per-group geometric edge statistics in millimetres.
    mesh_report: dict[str, dict[str, float]] = field(default_factory=dict)
    # Dense-BEM cost for the realized triangle count. See hornlab_mesher.cost.
    solve_cost: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mesh_path": str(self.mesh_path),
            "formula": self.formula,
            "mode": self.mode,
            "n_vertices": self.n_vertices,
            "n_triangles": self.n_triangles,
            "units": self.units,
            "physical_groups": {str(k): v for k, v in self.physical_groups.items()},
            "quadrants": self.quadrants,
            "native_symmetry_plane": self.native_symmetry_plane,
            "native_check_open_edges": self.native_check_open_edges,
            "mesh_report": self.mesh_report,
            "solve_cost": self.solve_cost,
            "metadata": self.metadata,
        }


def _number_list(value: Any) -> list[float]:
    return _parse_number_list(
        value,
        allow_scalar=True,
        finite_only=True,
        invalid="skip",
        evaluate=False,
    )


def _first_number(
    *sources: Mapping[str, Any], names: tuple[str, ...], default: float
) -> float:
    value = _pick(*sources, names=names, default=default)
    numbers = _number_list(value)
    return float(numbers[0]) if numbers else float(default)


def _section(config: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = config.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


def _pick(
    *sources: Mapping[str, Any], names: tuple[str, ...], default: Any = None
) -> Any:
    for source in sources:
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return default


def _float(
    *sources: Mapping[str, Any], names: tuple[str, ...], default: float
) -> float:
    value = _pick(*sources, names=names, default=default)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{names[0]} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise ConfigError(f"{names[0]} must be finite, got {value!r}")
    return out


def _optional_float(
    *sources: Mapping[str, Any], names: tuple[str, ...]
) -> float | None:
    value = _pick(*sources, names=names, default=None)
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{names[0]} must be numeric, got {value!r}") from exc
    if not np.isfinite(out):
        raise ConfigError(f"{names[0]} must be finite, got {value!r}")
    return out


def _numeric_param(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{name} must be numeric when deriving a driver adapter, got {value!r}"
        ) from exc
    if not np.isfinite(out):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    return out


def _scalar_or_expr(
    *sources: Mapping[str, Any], names: tuple[str, ...], default: Any
) -> Any:
    value = _pick(*sources, names=names, default=default)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            out = float(stripped)
        except ValueError:
            return stripped
        return int(out) if out.is_integer() else out
    return value


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    try:
        out = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigError(f"{name} must be an integer, got {value!r}") from exc
    if not np.isfinite(numeric) or numeric != out:
        raise ConfigError(f"{name} must be an integer, got {value!r}")
    return out


def _int(*sources: Mapping[str, Any], names: tuple[str, ...], default: int) -> int:
    value = _pick(*sources, names=names, default=default)
    return _integer(value, name=names[0])


def _bool(*sources: Mapping[str, Any], names: tuple[str, ...], default: bool) -> bool:
    value = _pick(*sources, names=names, default=default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _normalise_formula(value: Any) -> str:
    """:func:`profile_common._normalise_formula`, raising :class:`ConfigError`."""

    try:
        return _normalise_formula_common(value)
    except ValueError as exc:
        raise ConfigError(str(exc)) from None


def _has_any(*sources: Mapping[str, Any], names: tuple[str, ...]) -> bool:
    for source in sources:
        for name in names:
            if name in source and source[name] is not None:
                return True
    return False


def _validate_formula_specific_keys(
    formula: str,
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if formula == "FREEFORM":
        names = (
            "R_mm",
            "R",
            "r",
            "b",
            "m",
            "tmax",
            "L_mm",
            "L",
            "s",
            "n",
            "k",
            "q",
            "s1",
            "s2",
        )
        if _has_any(profile, config, names=names):
            raise ConfigError(
                "formula FREEFORM does not accept OSSE/R-OSSE profile coefficient keys"
            )
        if "h" in profile and profile["h"] is not None:
            raise ConfigError(
                "formula FREEFORM does not accept OSSE profile coefficient key h"
            )
        foreign_icw = sorted(
            {
                str(key)
                for source in (profile, config)
                for key, value in source.items()
                if str(key).startswith("icw_") and value is not None
            }
        )
        if foreign_icw:
            raise ConfigError(
                "formula FREEFORM does not accept ICW profile coefficient keys: "
                + ", ".join(foreign_icw)
            )
        return

    if formula == "OSSE":
        names = ("R_mm", "R", "tmax", "m", "r", "b")
        if _has_any(profile, config, names=names):
            raise ConfigError(
                "R-OSSE-only profile keys are not valid with formula OSSE"
            )
        return

    if formula == "LOOKUP":
        # LOOKUP carries only a precomputed profile; analytic coefficients of
        # either formula family are out of place.
        names = (
            "L_mm",
            "L",
            "n",
            "s",
            "rot_deg",
            "s1",
            "s2",
            "rot",
            "R_mm",
            "R",
            "tmax",
            "m",
            "r",
            "b",
        )
        if _has_any(profile, config, names=names):
            raise ConfigError("formula LOOKUP does not accept OSSE/R-OSSE profile keys")
        # The lookup profile is the whole meridian: nothing prepends a throat
        # extension, slot or driver adapter to it, and no coverage/flare
        # coefficient shapes it. These were accepted and then silently ignored
        # (a 25.4 -> 40 mm adapter still built the lookup's own 10 mm throat).
        # ``a0`` stays accepted: it sets the automatic source-cap angle.
        ignored = [
            name
            for name in ("a_deg", "a", "k", "q", "h")
            if _has_any(profile, config, names=(name,))
        ]
        ignored.extend(
            key
            for key, aliases in (
                ("throatExtLength", ("throat_ext_length_mm", "throatExtLength")),
                ("throatExtAngle", ("throat_ext_angle_deg", "throatExtAngle")),
                ("slotLength", ("slot_length_mm", "slotLength")),
            )
            if _param_is_nonzero(
                _pick(profile, config, names=aliases, default=None), name=key
            )
        )
        ignored.extend(
            name
            for name in _DRIVER_ADAPTER_KEYS
            if _has_any(profile, config, names=(name,))
        )
        if ignored:
            raise ConfigError(
                "formula LOOKUP does not support "
                + ", ".join(ignored)
                + ": the lookupProfile defines the whole meridian, so these keys "
                "would be ignored; build the extension or adapter into the "
                "lookupProfile instead"
            )
        return

    if formula == "ICW":
        # ICW accepts its own intrinsic-curvature keys (throat r0/a0, targets
        # L/R/theta1/x_aperture/depth/x_setback, kappa0, n_coeff, termination,
        # and the seed/direct inputs). Both L and R are legitimate ICW size
        # targets, so unlike OSSE/R-OSSE neither is rejected here. OSSE-only
        # shape coefficients (n, s, rot) and R-OSSE-only shape coefficients
        # (m, r, b, tmax) have no meaning on an ICW curve and are rejected at
        # the TOP LEVEL -- they may still appear nested inside icw_seed (a
        # separate OSSE/R-OSSE profile dict), which _has_any does not scan.
        # Coverage/manufacturability keys (coverage_angle, hold_*, kappa_abs_max,
        # dkappa_ds_abs_max, theta_max_deg, pin_mouth_radius) are valid ICW
        # top-level keys and are intentionally not part of this reject set.
        names = ("n", "s", "s1", "s2", "rot_deg", "rot", "m", "r", "b", "tmax")
        if _has_any(profile, config, names=names):
            raise ConfigError("OSSE/R-OSSE shape keys are not valid with formula ICW")
        return

    names = ("L_mm", "L", "n", "s", "rot_deg", "rot")
    if not stretch_is_inactive({key: _pick(profile, config, names=(key,), default=0)
                                for key in ("s1", "s2")}):
        # The shared composition boundary already refused active R-OSSE Rot.
        # A numeric zero rotation is harmless, just as on the text path.
        names = ("L_mm", "L", "n", "s")
    if _has_any(profile, config, names=names):
        raise ConfigError("OSSE-only profile keys are not valid with formula R-OSSE")


def _static_float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        raise ConfigError(f"numeric config value must be finite, got {value!r}")
    return out


def _gcurve_type_width(
    gcurve: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[int | None, float | None]:
    raw_type = _pick(gcurve, config, names=("gcurve_type", "gcurveType"), default=0)
    raw_width = _pick(
        gcurve, config, names=("gcurve_width_mm", "gcurveWidth"), default=0
    )
    type_value = _static_float_or_none(raw_type)
    width = _static_float_or_none(raw_width)
    curve_type = int(round(type_value)) if type_value is not None else None
    return curve_type, width


def _has_gcurve_keys(gcurve: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    names = (
        "gcurve_type",
        "gcurveType",
        "gcurve_width_mm",
        "gcurveWidth",
        "gcurve_aspect_ratio",
        "gcurveAspectRatio",
        "gcurve_dist",
        "gcurveDist",
        "gcurve_rot_deg",
        "gcurveRot",
        "gcurve_sf",
        "gcurveSf",
        "gcurveSF",
        "gcurve_se_n",
        "gcurveSeN",
        "gcurve_sf_a",
        "gcurveSfA",
        "gcurve_sf_b",
        "gcurveSfB",
        "gcurve_sf_m1",
        "gcurveSfM1",
        "gcurve_sf_m2",
        "gcurveSfM2",
        "gcurve_sf_n1",
        "gcurveSfN1",
        "gcurve_sf_n2",
        "gcurveSfN2",
        "gcurve_sf_n3",
        "gcurveSfN3",
    )
    return _has_any(gcurve, config, names=names)


def _gcurve_could_be_active(
    gcurve: Mapping[str, Any], config: Mapping[str, Any]
) -> bool:
    curve_type, width = _gcurve_type_width(gcurve, config)
    if curve_type is None or width is None:
        return _has_gcurve_keys(gcurve, config)
    return curve_type in {1, 2} and width > 0.0


def _validate_static_gcurve_type(
    gcurve: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    curve_type, _width = _gcurve_type_width(gcurve, config)
    if curve_type is not None and curve_type not in {0, 1, 2}:
        raise ConfigError(f"unsupported GCurve type {curve_type}")


def _validate_formula_features(
    formula: str,
    profile: Mapping[str, Any],
    cross: Mapping[str, Any],
    morph: Mapping[str, Any],
    gcurve: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    _validate_static_gcurve_type(gcurve, config)
    raw_morph_target = _pick(
        morph, config, names=("morph_target", "morphTarget"), default=0
    )
    static_morph_target = _static_float_or_none(raw_morph_target)
    if (
        static_morph_target is not None
        and int(round(static_morph_target)) not in {0, 1, 2, 3}
    ):
        raise ConfigError(
            "morphTarget must resolve to one of the valid values "
            f"0, 1, 2, or 3; got {raw_morph_target!r}"
        )
    if formula == "FREEFORM":
        if static_morph_target is None:
            raise ConfigError(
                "FREEFORM morphTarget expression cannot be proven inactive; "
                "crossSections owns the outline"
            )
        if _gcurve_could_be_active(gcurve, config):
            raise ConfigError(
                "FREEFORM does not support active guiding curves; "
                "use crossSections stations instead"
            )
        exponent = _float(
            cross,
            profile,
            config,
            names=("exponent", "cross_section_exponent"),
            default=2.0,
        )
        aspect_ratio = _float(
            cross,
            profile,
            config,
            names=("aspect_ratio", "aspectRatio"),
            default=1.0,
        )
        if not math.isclose(exponent, 2.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ConfigError("FREEFORM requires cross-section exponent=2")
        if not math.isclose(aspect_ratio, 1.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise ConfigError("FREEFORM requires cross-section aspectRatio=1")
        for key, aliases in (
            ("rot", ("rot_deg", "rot")),
            ("h", ("h",)),
            ("throatExtLength", ("throat_ext_length_mm", "throatExtLength")),
            ("throatExtAngle", ("throat_ext_angle_deg", "throatExtAngle")),
            ("slotLength", ("slot_length_mm", "slotLength")),
        ):
            value = _pick(profile, config, names=aliases, default=0.0)
            if _param_is_nonzero(value, name=key):
                raise ConfigError(f"FREEFORM does not support active {key}")
        return
    if formula != "OSSE" and _gcurve_could_be_active(gcurve, config):
        raise ConfigError("guiding curves are only supported with formula OSSE")
    if formula == "OSSE" and _gcurve_could_be_active(gcurve, config):
        exponent = _float(
            cross, profile, config, names=("exponent", "cross_section_exponent"), default=2.0
        )
        aspect_ratio = _float(
            cross, profile, config, names=("aspect_ratio", "aspectRatio"), default=1.0
        )
        if not _cross_section_is_circular(exponent, aspect_ratio):
            raise ConfigError(GCURVE_CROSS_SECTION_CONFLICT)


def _enc_depth_mm(
    config: Mapping[str, Any],
    mesh: Mapping[str, Any],
    enclosure: Mapping[str, Any],
    formula: str,
) -> float:
    """Resolve the enclosure depth (mm); 0.0 means "no enclosure".

    A bare ``depth`` is read straight from the ``enclosure`` / ``mesh`` sections, where it is
    unambiguous. At the TOP LEVEL of the config a bare ``depth`` is NOT treated as enclosure depth
    for an ICW profile -- there it is the rollback axial target (a profile param), and letting it
    trip enclosure mode silently wrapped a free-standing rollback ICW in a box. ICW enclosures must
    therefore name the depth explicitly (``encDepth`` / ``depth_mm``) or nest it under the
    ``enclosure`` section; other formulas keep the historical top-level bare-``depth`` fallback.
    """
    sectioned = _optional_float(
        enclosure, mesh, names=("depth_mm", "depth", "encDepth")
    )
    if sectioned is not None:
        return sectioned
    top_names = (
        ("depth_mm", "encDepth")
        if formula == "ICW"
        else ("depth_mm", "depth", "encDepth")
    )
    return _float(config, names=top_names, default=0.0)


def _normalise_mode(
    config: Mapping[str, Any],
    mesh: Mapping[str, Any],
    enclosure: Mapping[str, Any],
    formula: str = "OSSE",
) -> str:
    raw = (
        str(_pick(config, mesh, names=("mode",), default=""))
        .strip()
        .lower()
        .replace("_", "-")
    )
    enc_depth = _enc_depth_mm(config, mesh, enclosure, formula)
    if raw in {"enclosure", "enclosed"}:
        return "enclosure"
    if enc_depth > 0:
        if raw == "":
            # An enclosure depth implies enclosure mode when no mode is given.
            return "enclosure"
        raise ConfigError(
            f"mode {raw!r} contradicts the configured enclosure depth {enc_depth:g} mm; "
            "drop the enclosure or use mode='enclosure'"
        )
    if raw in {"bare", "inner", "open"}:
        return "bare"
    if raw in {"infinite-baffle", "infinitebaffle", "ib", "baffle"}:
        return "infinite-baffle"
    if raw == "":
        # Imported ATH text configs carry ABEC.SimType (1 = infinite baffle,
        # 2 = free standing); native configs without a mode stay freestanding.
        sim_type = _pick(config, mesh, names=("simType", "sim_type"), default=None)
        if sim_type is not None:
            try:
                sim_int = _integer(sim_type, name="simType")
            except ConfigError as exc:
                raise ConfigError(f"simType must be 1 or 2, got {sim_type!r}") from exc
            if sim_int == 1:
                return "infinite-baffle"
            if sim_int == 2:
                return "freestanding"
            raise ConfigError(f"simType must be 1 or 2, got {sim_type!r}")
        return "freestanding"
    if raw in {"free-standing", "freestanding", "free"}:
        return "freestanding"
    raise ConfigError(
        f"mode must be freestanding, enclosure, bare, or infinite-baffle, got {raw!r}"
    )


def _enclosure_from_config(
    config: Mapping[str, Any],
    mesh: Mapping[str, Any],
    enclosure: Mapping[str, Any],
    formula: str = "OSSE",
) -> HornEnclosure | None:
    depth = _enc_depth_mm(config, mesh, enclosure, formula)
    if depth <= 0.0:
        return None
    return HornEnclosure(
        depth_mm=depth,
        space_l_mm=_float(
            enclosure, names=("space_l_mm", "space_l", "left_margin_mm"), default=25.0
        ),
        space_t_mm=_float(
            enclosure, names=("space_t_mm", "space_t", "top_margin_mm"), default=25.0
        ),
        space_r_mm=_float(
            enclosure, names=("space_r_mm", "space_r", "right_margin_mm"), default=25.0
        ),
        space_b_mm=_float(
            enclosure, names=("space_b_mm", "space_b", "bottom_margin_mm"), default=25.0
        ),
        edge_mm=_float(enclosure, names=("edge_mm", "edge", "encEdge"), default=18.0),
        edge_type=_int(
            enclosure, names=("edge_type", "edgeType", "encEdgeType"), default=1
        ),
        plan_type=_int(
            enclosure, names=("plan_type", "planType", "encPlanType"), default=1
        ),
        plan_n=_float(enclosure, names=("plan_n", "planN", "encPlanN"), default=2.0),
        depth_margin_mm=_float(
            enclosure,
            names=("depth_margin_mm", "depth_margin", "encDepthMargin"),
            default=1.0,
        ),
        front_mesh_size_mm=_first_number(
            enclosure,
            names=(
                "front_mesh_size_mm",
                "frontMeshSize",
                "enc_front_resolution",
                "encFrontResolution",
            ),
            default=0.0,
        ),
        back_mesh_size_mm=_first_number(
            enclosure,
            names=(
                "back_mesh_size_mm",
                "backMeshSize",
                "enc_back_resolution",
                "encBackResolution",
            ),
            default=0.0,
        ),
    )


_DRIVER_ADAPTER_KEYS = (
    "driver_throat_diameter_mm",
    "driver_throat_diameter",
    "driverThroatDiameterMm",
    "driverThroatDiameter",
    "driver_throat_diameter_in",
    "driverThroatDiameterIn",
    "waveguide_throat_diameter_mm",
    "waveguide_throat_diameter",
    "waveguideThroatDiameterMm",
    "waveguideThroatDiameter",
    "waveguide_throat_diameter_in",
    "waveguideThroatDiameterIn",
)


def _diameter_radius_mm(
    *sources: Mapping[str, Any],
    mm_names: tuple[str, ...],
    inch_names: tuple[str, ...],
) -> float | None:
    diameter_mm = _optional_float(*sources, names=mm_names)
    if diameter_mm is not None:
        return 0.5 * diameter_mm
    diameter_in = _optional_float(*sources, names=inch_names)
    if diameter_in is not None:
        return 0.5 * diameter_in * 25.4
    return None


def _apply_driver_adapter(
    common: dict[str, Any],
    profile: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    driver_radius = _diameter_radius_mm(
        profile,
        config,
        mm_names=(
            "driver_throat_diameter_mm",
            "driver_throat_diameter",
            "driverThroatDiameterMm",
            "driverThroatDiameter",
        ),
        inch_names=("driver_throat_diameter_in", "driverThroatDiameterIn"),
    )
    waveguide_radius = _diameter_radius_mm(
        profile,
        config,
        mm_names=(
            "waveguide_throat_diameter_mm",
            "waveguide_throat_diameter",
            "waveguideThroatDiameterMm",
            "waveguideThroatDiameter",
        ),
        inch_names=("waveguide_throat_diameter_in", "waveguideThroatDiameterIn"),
    )
    if driver_radius is None and waveguide_radius is None:
        return
    if driver_radius is None or waveguide_radius is None:
        raise ConfigError(
            "driver adapter requires both driver and waveguide throat diameters"
        )
    if driver_radius <= 0.0 or waveguide_radius <= 0.0:
        raise ConfigError("driver and waveguide throat diameters must be > 0")
    if waveguide_radius < driver_radius:
        raise ConfigError(
            "driver adapter cannot shrink from waveguide throat to driver throat"
        )

    delta_radius = waveguide_radius - driver_radius
    # Both formulas anchor r0 (Throat.Diameter) at the MAIN waveguide throat
    # and taper the extension BACK to the driver end (r0 - ext*tan == driver
    # radius) — the ATH convention. Setting r0 to the driver radius here (the
    # old forward-expansion assumption) built a horn whose requested waveguide
    # throat diameter appeared nowhere in the geometry.
    common["r0"] = waveguide_radius
    if delta_radius <= 1.0e-12:
        common["throatExtLength"] = 0.0
        common["throatExtAngle"] = 0.0
        return

    ext_len = _numeric_param(common.get("throatExtLength", 0.0), name="throatExtLength")
    ext_angle_deg = _numeric_param(
        common.get("throatExtAngle", 0.0), name="throatExtAngle"
    )
    if ext_len <= 0.0 and abs(ext_angle_deg) <= 1.0e-12:
        raise ConfigError("driver adapter requires throatExtLength or throatExtAngle")
    if ext_len <= 0.0:
        tan_angle = np.tan(np.deg2rad(ext_angle_deg))
        if tan_angle <= 1.0e-12:
            raise ConfigError(
                "throatExtAngle must be > 0 when deriving driver adapter length"
            )
        common["throatExtLength"] = delta_radius / tan_angle
        return
    if abs(ext_angle_deg) <= 1.0e-12:
        common["throatExtAngle"] = float(np.rad2deg(np.arctan(delta_radius / ext_len)))
        return

    derived_radius = driver_radius + ext_len * np.tan(np.deg2rad(ext_angle_deg))
    if abs(derived_radius - waveguide_radius) > 1.0e-6:
        raise ConfigError(
            "driver adapter throatExtLength/throatExtAngle do not reach waveguide throat diameter"
        )


def _param_is_nonzero(value: Any, *, name: str) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    try:
        out = float(value)
    except (TypeError, ValueError):
        return True
    if not np.isfinite(out):
        raise ConfigError(f"{name} must be finite, got {value!r}")
    return abs(out) > 1.0e-12


def _reject_icw_throat_extension(common: Mapping[str, Any]) -> None:
    names = [
        key
        for key in ("throatExtLength", "throatExtAngle")
        if _param_is_nonzero(common.get(key), name=key)
    ]
    if names:
        joined = "/".join(names)
        raise ConfigError(f"formula ICW does not support throat extension ({joined})")


@stretch_config_errors
def build_geometry_params(config: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    profile = _section(config, "profile", "parameters")
    mesh = _section(config, "mesh")
    enclosure = _section(config, "enclosure")
    cross = _section(config, "cross_section", "crossSection")
    morph = _section(config, "morph", "MORPH")
    gcurve = _section(config, "gcurve", "GCurve", "GCURVE")
    source = _section(config, "source", "Source")

    formula = _normalise_formula(
        _pick(config, profile, names=("formula", "type"), default="OSSE")
    )
    validate_supplied_stretch(config, formula)
    if formula in {"OSSE", "R-OSSE"}:
        composition = {key: _pick(profile, config, names=(key,), default=0.0)
                       for key in ("s1", "s2")}
        for section, keys in ((profile, COMPOSITION_PROFILE_KEYS), (gcurve, COMPOSITION_GUIDE_KEYS)):
            for key, (_, aliases) in keys.items():
                for supplied in (section, config):
                    matches = [name for name in aliases if name in supplied]
                    if matches:
                        composition[key] = supplied[matches[0]]
                        break
        validate_stretch_composition(
            composition, formula,
            length_supplied=_has_any(profile, config, names=("L_mm", "L", "Length")),
        )
    _validate_formula_specific_keys(formula, profile, config)
    _validate_formula_features(formula, profile, cross, morph, gcurve, config)
    mode = _normalise_mode(config, mesh, enclosure, formula)
    enc_depth = 0.0
    enclosure_obj = _enclosure_from_config(config, mesh, enclosure, formula)
    if enclosure_obj is not None:
        enc_depth = enclosure_obj.depth_mm
    elif mode == "enclosure":
        raise ConfigError("enclosure mode requires enclosure.depth_mm > 0")

    default_wall = 0.0 if mode in {"bare", "enclosure", "infinite-baffle"} else 6.0
    wall_thickness = _float(
        mesh,
        config,
        names=("wall_thickness_mm", "wall_thickness", "wallThickness"),
        default=default_wall,
    )
    if mode in {"bare", "enclosure", "infinite-baffle"}:
        wall_thickness = 0.0
    z_map_points = _pick(
        mesh,
        config,
        names=("z_map_points", "zMapPoints", "zmapPoints", "ZMapPoints"),
        default=None,
    )
    z_map_kind = _pick(
        mesh,
        config,
        names=("z_map_kind", "zMapKind"),
        default=None,
    )
    length_segments = _int(
        mesh,
        config,
        names=("length_segments", "lengthSegments"),
        default=32,
    )
    if z_map_kind is None and z_map_points is not None:
        try:
            z_map_kind = _classify_zmap_kind(length_segments, z_map_points)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    default_sampling_mode = "zmap" if z_map_points is not None else "uniform"

    common: dict[str, Any] = {
        "type": formula,
        "lookupProfile": _pick(
            profile, config, names=("lookupProfile", "lookup_profile"), default=None
        ),
        "r0": _scalar_or_expr(profile, config, names=("r0_mm", "r0"), default=12.7),
        "a": _scalar_or_expr(profile, config, names=("a_deg", "a"), default=60.0),
        "a0": _scalar_or_expr(profile, config, names=("a0_deg", "a0"), default=15.5),
        "k": _scalar_or_expr(profile, config, names=("k",), default=1.0),
        "q": _scalar_or_expr(
            profile, config, names=("q",), default=1.0 if formula == "R-OSSE" else 0.995
        ),
        "throatExtLength": _scalar_or_expr(
            profile,
            config,
            names=("throat_ext_length_mm", "throatExtLength"),
            default=0.0,
        ),
        "throatExtAngle": _scalar_or_expr(
            profile,
            config,
            names=("throat_ext_angle_deg", "throatExtAngle"),
            default=0.0,
        ),
        "slotLength": _scalar_or_expr(
            profile, config, names=("slot_length_mm", "slotLength"), default=0.0
        ),
        "angularSegments": _int(
            mesh, config, names=("angular_segments", "angularSegments"), default=64
        ),
        "cornerSegments": _int(
            mesh, config, names=("corner_segments", "cornerSegments"), default=0
        ),
        "lengthSegments": length_segments,
        "samplingMode": _pick(
            mesh,
            config,
            names=("sampling_mode", "samplingMode"),
            default=default_sampling_mode,
        ),
        "athParitySampling": _bool(
            mesh,
            config,
            names=("ath_parity_sampling", "athParitySampling"),
            default=False,
        ),
        "zMapPoints": z_map_points,
        "zMapKind": z_map_kind,
        "wallThickness": wall_thickness,
        "encDepth": enc_depth,
        "morphTarget": _scalar_or_expr(
            morph, config, names=("morph_target", "morphTarget"), default=0
        ),
        "morphWidth": _scalar_or_expr(
            morph, config, names=("morph_width_mm", "morphWidth"), default=0
        ),
        "morphHeight": _scalar_or_expr(
            morph, config, names=("morph_height_mm", "morphHeight"), default=0
        ),
        "morphCorner": _scalar_or_expr(
            morph, config, names=("morph_corner_mm", "morphCorner"), default=0
        ),
        "morphExponent": _scalar_or_expr(
            morph, config, names=("morph_exponent", "morphExponent"), default=2.0
        ),
        "morphRate": _scalar_or_expr(
            morph, config, names=("morph_rate", "morphRate"), default=3.0
        ),
        "morphFixed": _scalar_or_expr(
            morph, config, names=("morph_fixed", "morphFixed"), default=0
        ),
        "morphAllowShrinkage": _scalar_or_expr(
            morph,
            config,
            names=("morph_allow_shrinkage", "morphAllowShrinkage"),
            default=0,
        ),
        "gcurveType": _scalar_or_expr(
            gcurve, config, names=("gcurve_type", "gcurveType"), default=0
        ),
        "gcurveWidth": _scalar_or_expr(
            gcurve, config, names=("gcurve_width_mm", "gcurveWidth"), default=0
        ),
        "gcurveAspectRatio": _scalar_or_expr(
            gcurve,
            config,
            names=("gcurve_aspect_ratio", "gcurveAspectRatio"),
            default=1,
        ),
        "gcurveDist": _scalar_or_expr(
            gcurve, config, names=("gcurve_dist", "gcurveDist"), default=0
        ),
        "gcurveRot": _scalar_or_expr(
            gcurve, config, names=("gcurve_rot_deg", "gcurveRot"), default=0
        ),
        "gcurveSF": _pick(
            gcurve, config, names=("gcurve_sf", "gcurveSf", "gcurveSF"), default=""
        ),
        "gcurveSf": _pick(
            gcurve, config, names=("gcurve_sf", "gcurveSf", "gcurveSF"), default=""
        ),
        "gcurveSeN": _scalar_or_expr(
            gcurve, config, names=("gcurve_se_n", "gcurveSeN"), default=3
        ),
        "gcurveSfA": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_a", "gcurveSfA"), default=1
        ),
        "gcurveSfB": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_b", "gcurveSfB"), default=1
        ),
        "gcurveSfM1": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_m1", "gcurveSfM1"), default=4
        ),
        "gcurveSfM2": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_m2", "gcurveSfM2"), default=None
        ),
        "gcurveSfN1": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_n1", "gcurveSfN1"), default=2
        ),
        "gcurveSfN2": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_n2", "gcurveSfN2"), default=2
        ),
        "gcurveSfN3": _scalar_or_expr(
            gcurve, config, names=("gcurve_sf_n3", "gcurveSfN3"), default=2
        ),
        "quadrants": _normalised_quadrants(
            _pick(mesh, config, names=("quadrants",), default="1234")
        ),
        "scale": _float(config, profile, names=("scale", "Scale"), default=1.0),
        "verticalOffset": _float(
            mesh, config, names=("vertical_offset_mm", "verticalOffset"), default=0.0
        ),
        "throatResolution": _float(
            mesh, config, names=("throat_res_mm", "throatResolution"), default=4.0
        ),
        "mouthResolution": _float(
            mesh, config, names=("mouth_res_mm", "mouthResolution"), default=26.0
        ),
        "rearResolution": _float(
            mesh, config, names=("rear_res_mm", "rearResolution"), default=15.0
        ),
        "subdomainSlices": _scalar_or_expr(
            mesh, config, names=("subdomain_slices", "subdomainSlices"), default=""
        ),
        # None (not 0.0) when omitted: an omitted offset with SubdomainSlices
        # set takes ATH's 5 mm default, while an explicit 0 disables interfaces.
        "interfaceOffset": _scalar_or_expr(
            mesh, config, names=("interface_offset_mm", "interfaceOffset"), default=None
        ),
        "interfaceResolution": _optional_float(
            mesh,
            config,
            names=("interface_res_mm", "interfaceResolution"),
        ),
        "sourceShape": _scalar_or_expr(
            source, config, names=("source_shape", "sourceShape"), default=1
        ),
        "sourceRadius": _scalar_or_expr(
            source, config, names=("source_radius_mm", "sourceRadius"), default=-1
        ),
        "sourceCurv": _scalar_or_expr(
            source, config, names=("source_curv", "sourceCurv"), default=0
        ),
        "profileSystem": {
            "crossSection": {
                "exponent": _float(
                    cross,
                    profile,
                    config,
                    names=("exponent", "cross_section_exponent"),
                    default=2.0,
                ),
                "aspectRatio": _float(
                    cross,
                    profile,
                    config,
                    names=("aspect_ratio", "aspectRatio"),
                    default=1.0,
                ),
            },
        },
    }
    keeps_slot = _pick(morph, config, names=(MORPH_KEEPS_SLOT_KEY,), default=None)
    if keeps_slot is not None:
        common[MORPH_KEEPS_SLOT_KEY] = bool(keeps_slot)
    length_mode = _pick(
        profile,
        config,
        names=("_athLengthMode", "athLengthMode", "length_mode", "lengthMode"),
        default=None,
    )
    if length_mode is not None:
        common["_athLengthMode"] = length_mode
    _apply_driver_adapter(common, profile, config)
    if formula in {"OSSE", "R-OSSE"}:
        for name in ("s1", "s2"):
            common[name] = _pick(profile, config, names=(name,), default=0.0)
        common = canonical_stretch_params(common)
    if formula == "ICW":
        _reject_icw_throat_extension(common)

    if formula == "OSSE":
        common.update(
            {
                "L": _scalar_or_expr(
                    profile, config, names=("L_mm", "L"), default=120.0
                ),
                "n": _scalar_or_expr(profile, config, names=("n",), default=4.0),
                "s": _scalar_or_expr(profile, config, names=("s",), default=0.0),
                # The half-sine bulge. It was parsed (profile ``h`` / ATH-text
                # ``OS.h``) and applied by the sampler, but never copied here, so
                # every config-driven build -- preview, solve and export alike --
                # silently built it as 0.
                "h": _scalar_or_expr(profile, config, names=("h",), default=0.0),
                "rot": _scalar_or_expr(
                    profile, config, names=("rot_deg", "rot"), default=0.0
                ),
            }
        )
    elif formula == "FREEFORM":
        if _has_any(profile, config, names=("overshootPolicy",)):
            # Dropped silently here while build_freeform_geometry refuses it,
            # so ``overshootPolicy = "allow"`` was accepted and ignored.
            raise ConfigError(
                "FREEFORM overshootPolicy was removed; tangent speed is now solved "
                "automatically"
            )
        for key in (
            "profileH",
            "profileV",
            "crossSections",
            "inflectionPolicy",
        ):
            value = _pick(profile, names=(key,), default=None)
            if value is not None:
                common[key] = value
    elif formula == "LOOKUP":
        # No analytic coefficients: the precomputed lookupProfile (threaded
        # into common above) fully defines the radial profile.
        pass
    elif formula == "ICW":
        # ICW reads its targets/seed straight off the params dict in
        # build_icw_curve. r0/a0 are already in ``common``; thread the rest
        # through. Only keys actually present are forwarded so the kernel keeps
        # applying its own defaults for the optional targets.
        common["termination"] = _pick(
            profile, config, names=("termination",), default="flat_baffle"
        )
        termination = str(common["termination"] or "flat_baffle").strip().lower()
        if termination == "flat_baffle":
            common["L"] = _scalar_or_expr(
                profile, config, names=("L_mm", "L"), default=120.0
            )
            common["R"] = _scalar_or_expr(
                profile, config, names=("R_mm", "R"), default=150.0
            )
        else:
            # Rollback reads R with r_aperture as a fallback; materialising the
            # flat-baffle defaults here would silently override an explicit
            # r_aperture with R=150. Thread L/R only when actually configured.
            for key, src_names in (("L", ("L_mm", "L")), ("R", ("R_mm", "R"))):
                if _pick(profile, config, names=src_names, default=None) is not None:
                    common[key] = _scalar_or_expr(
                        profile, config, names=src_names, default=None
                    )
        for key, src_names in (
            ("kappa0", ("kappa0",)),
            ("r_aperture", ("r_aperture",)),
            ("n_coeff", ("n_coeff",)),
            ("theta1", ("theta1_deg", "theta1")),
            ("x_aperture", ("x_aperture",)),
            ("depth", ("depth",)),
            ("x_setback", ("x_setback",)),
            ("coverage_angle", ("coverage_angle", "coverage_angle_deg")),
            ("hold_start", ("hold_start",)),
            ("hold_end", ("hold_end",)),
            ("kappa_abs_max", ("kappa_abs_max",)),
            ("dkappa_ds_abs_max", ("dkappa_ds_abs_max",)),
            ("theta_max_deg", ("theta_max_deg",)),
            ("icw_S", ("icw_S",)),
        ):
            value = _pick(profile, config, names=src_names, default=None)
            if value is not None:
                common[key] = _scalar_or_expr(
                    profile, config, names=src_names, default=None
                )
        pin_mouth_radius = _pick(
            profile, config, names=("pin_mouth_radius", "pinMouthRadius"), default=None
        )
        if pin_mouth_radius is not None:
            common["pin_mouth_radius"] = _bool(
                profile,
                config,
                names=("pin_mouth_radius", "pinMouthRadius"),
                default=False,
            )
        # Seed / direct-coefficient inputs are passed through verbatim (nested
        # dict / list), not coerced to scalars.
        seed = _pick(profile, config, names=("icw_seed",), default=None)
        if seed is not None:
            common["icw_seed"] = seed
        coeffs = _pick(profile, config, names=("icw_coeffs",), default=None)
        if coeffs is not None:
            common["icw_coeffs"] = coeffs
    else:
        common.update(
            {
                "R": _scalar_or_expr(
                    profile, config, names=("R_mm", "R"), default=150.0
                ),
                "tmax": _scalar_or_expr(profile, config, names=("tmax",), default=1.0),
            }
        )
        for key in ("m", "r", "b"):
            value = _pick(profile, config, names=(key,), default=None)
            if value is not None:
                common[key] = _scalar_or_expr(
                    profile, config, names=(key,), default=None
                )

    if formula == "FREEFORM":
        try:
            _validate_freeform_config(common)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    if formula in {"OSSE", "R-OSSE"}:
        if not stretch_is_inactive(common):
            # Omit absent optional guide inputs on the numeric-only path.
            # Explicit nonnumeric values were refused before normalization.
            for key, absent in (("gcurveSF", ""), ("gcurveSf", ""), ("gcurveSfM2", None)):
                if common[key] == absent:
                    common.pop(key)
        validate_stretch_composition(
            {**common, "rot": _pick(profile, config, names=("rot_deg", "rot"), default=0.0)},
            formula,
            length_supplied=_has_any(profile, config, names=("L_mm", "L", "Length")),
        )
    return common, formula, mode


def _normalised_quadrants(value: Any) -> str:
    """Normalise Mesh.Quadrants to Ath's canonical coverage ({1, 12, 14, 1234}).

    Delegates to the shared parser, which follows Ath's atoi rule: it reads a leading
    integer and maps 1234/12/14 to full/half-y/half-x and every other value —
    including junk, empties and permutations like "21" — to a quarter model. It never
    raises, matching Ath, which silently treats unrecognised values as the quarter
    default (a well-defined Q1 grid) rather than erroring or building the degenerate
    open full-circle grid this mesher's earlier set-based logic produced.
    """
    return _normalise_quadrants_common(value)


def _native_symmetry_plane_for_quadrants(quadrants: str) -> str | None:
    """Map grid quadrant coverage to the metal solver's symmetry-plane flag.

    Quadrant 1 spans x >= 0, y >= 0 and mirrors about both the yz and xz
    planes; "12" spans y >= 0 (xz mirror); "14" spans x >= 0 (yz mirror).
    """

    return {
        "1": "yz+xz",
        "12": "xz",
        "14": "yz",
    }.get(_normalised_quadrants(quadrants))


def _symmetry_planes_for_quadrants(quadrants: str) -> tuple[str, ...]:
    """Map grid quadrant coverage to open-grid snap axes.

    Mirrors :func:`_native_symmetry_plane_for_quadrants` but in the mesher's
    ``"x"``/``"y"`` snap-axis convention (``"x"`` is the x=0 / yz plane, ``"y"``
    is the y=0 / xz plane): quadrant 1 is bounded by both planes, "12" by the
    xz plane (snap ``"y"``), "14" by the yz plane (snap ``"x"``). Full coverage
    ("1234") returns no planes.
    """

    return _symmetry_planes_for_quadrants_common(quadrants)


def _native_check_open_edges_for_mode(mode: str) -> bool:
    """Whether the metal solver's cut-plane open-edge guard applies to ``mode``.

    ``bare`` horns radiate from an open mouth, so their free mouth rims are
    legitimate solve boundaries and the guard must be disabled. Coupled
    infinite-baffle, freestanding, and enclosure modes cap the mouth, so any
    reduced-domain open edges lie on cut planes and the strict check stays on.
    """

    return mode != "bare"


def _mesh_report(
    info_physical_groups: Mapping[int, str],
    edge_stats_mm: Mapping[int, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for tag, stats in edge_stats_mm.items():
        name = info_physical_groups.get(int(tag), str(tag))
        entry = {key: float(value) for key, value in stats.items()}
        report[name] = entry
    return report


_REMOVED_MESH_KEYS: dict[str, str] = {
    "max_frequency_hz": "use throat_res_mm, mouth_res_mm, and rear_res_mm",
    "maxFrequencyHz": "use throat_res_mm, mouth_res_mm, and rear_res_mm",
    "maxFrequency": "use throat_res_mm, mouth_res_mm, and rear_res_mm",
    "f_max_hz": "use throat_res_mm, mouth_res_mm, and rear_res_mm",
    "fMaxHz": "use throat_res_mm, mouth_res_mm, and rear_res_mm",
    "elements_per_wavelength": "use the semantic *_res_mm values",
    "elementsPerWavelength": "use the semantic *_res_mm values",
    "throat_epw": "use throat_res_mm",
    "throatEpw": "use throat_res_mm",
    "mouth_epw": "use mouth_res_mm",
    "mouthEpw": "use mouth_res_mm",
    "rear_epw": "use rear_res_mm",
    "rearEpw": "use rear_res_mm",
    "interface_epw": "use interface_res_mm",
    "interfaceEpw": "use interface_res_mm",
    "aperture_epw": "use aperture_res_scale and mouth_res_mm",
    "apertureEpw": "use aperture_res_scale and mouth_res_mm",
    "speed_of_sound_m_s": "mesh sizing is millimetre-only",
    "speedOfSound": "mesh sizing is millimetre-only",
    "curvature_segments": "curvature sizing is disabled; use the local *_res_mm values",
    "curvatureSegments": "curvature sizing is disabled; use the local *_res_mm values",
}


def _reject_removed_mesh_keys(
    config: Mapping[str, Any], mesh: Mapping[str, Any]
) -> None:
    for mapping in (mesh, config):
        for key, migration in _REMOVED_MESH_KEYS.items():
            if key in mapping:
                raise ConfigError(
                    f"mesh key {key!r} was removed by the millimetre-only mesh contract; "
                    f"{migration}"
                )


def _mesh_topology_mode(mesh: Mapping[str, Any]) -> str:
    raw = _pick(
        mesh, names=("topology", "topology_mode", "topologyMode"), default="acoustic"
    )
    mode = str(raw or "acoustic").strip().lower()
    if mode not in {"acoustic", "legacy"}:
        raise ConfigError("mesh topology must be 'acoustic' or 'legacy'")
    preserve = _bool(mesh, names=("preserve_grid", "preserveGrid"), default=False)
    if preserve and mode != "legacy":
        raise ConfigError(
            "preserve_grid pins sampled CAD faces and is only available with mesh.topology='legacy'"
        )
    return mode


def _mesh_surface_fit(mesh: Mapping[str, Any]) -> str:
    """Read mesh.surface_fit, the acoustic B-spline patch fitting mode.

    Defaults to ``auto``, which ``PointGridHornGeometry`` resolves to
    ``interpolate`` on everything it can mesh and ``approximate`` on FREEFORM.
    ``interpolate`` removes the inward pole-fit bias at the same triangle count.
    """

    raw = _pick(mesh, names=("surface_fit", "surfaceFit"), default="auto")
    mode = str(raw or "auto").strip().lower()
    if mode not in {"auto", "approximate", "interpolate"}:
        raise ConfigError(
            "mesh surface_fit must be 'auto', 'approximate' or 'interpolate'"
        )
    return mode


def _mesh_density_from_config(
    config: Mapping[str, Any],
    *,
    allow_large_mesh: bool | None = None,
) -> MeshDensity:
    mesh = _section(config, "mesh")
    enclosure = _section(config, "enclosure")
    _reject_removed_mesh_keys(config, mesh)
    configured_allow_large = _bool(
        mesh,
        names=("allow_large_mesh", "allowLargeMesh"),
        default=False,
    )
    density = MeshDensity(
        throat_res_mm=_float(
            mesh,
            config,
            names=("throat_res_mm", "throat_res", "throatResolution"),
            default=4.0,
        ),
        mouth_res_mm=_float(
            mesh,
            config,
            names=("mouth_res_mm", "mouth_res", "mouthResolution"),
            default=26.0,
        ),
        rear_res_mm=_float(
            mesh,
            config,
            names=("rear_res_mm", "rear_res", "rearResolution"),
            default=15.0,
        ),
        aperture_res_scale=_float(
            mesh,
            config,
            names=(
                "aperture_res_scale",
                "apertureResolutionScale",
                "aperture_cap_coarsening",
                "apertureCapCoarsening",
            ),
            default=1.0,
        ),
        enc_front_res_mm=_pick(
            mesh,
            enclosure,
            names=("enc_front_res_mm", "enc_front_resolution", "encFrontResolution"),
            default=None,
        ),
        enc_back_res_mm=_pick(
            mesh,
            enclosure,
            names=("enc_back_res_mm", "enc_back_resolution", "encBackResolution"),
            default=None,
        ),
        interface_res_mm=_optional_float(
            mesh,
            names=("interface_res_mm", "interface_res", "interfaceResolution"),
        ),
        max_triangles=_int(
            mesh,
            names=("max_triangles", "maxTriangles"),
            default=18_000,
        ),
        allow_large_mesh=(
            configured_allow_large
            if allow_large_mesh is None
            else bool(allow_large_mesh)
        ),
    )
    try:
        validate_mesh_density(density)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return density


def _num_or_default(value: Any, default: float) -> float:
    """Numeric param with an explicit unset check.

    ``0`` is a meaningful value for several params (``sourceShape = 0`` is the
    flat disc), so only ``None``/blank counts as unset — never falsiness.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return float(default)
    return float(value)


def _interfaces_from_params(
    params: Mapping[str, Any],
    n_length: int,
    *,
    requested_slice_map: Any = None,
    fitted_slice_map: Any = None,
) -> tuple[HornInterface, ...]:
    slices = [
        int(round(value)) for value in _number_list(params.get("subdomainSlices"))
    ]
    if not slices:
        return ()

    offsets = _number_list(params.get("interfaceOffset"))
    if not offsets:
        # ATH defaults Mesh.InterfaceOffset to 5 mm when SubdomainSlices are
        # set; an omitted offset used to silently drop the interfaces entirely.
        offsets = [5.0]
    if len(offsets) == 1 and len(slices) > 1:
        offsets = offsets * len(slices)
    if len(offsets) != len(slices):
        raise ConfigError(
            f"Mesh.SubdomainSlices lists {len(slices)} slices but Mesh.InterfaceOffset "
            f"lists {len(offsets)} offsets; give one offset or one per slice"
        )

    interfaces: list[HornInterface] = []
    fitted_last_ring = int(n_length)
    # SubdomainSlices belong to the caller's requested control grid. The
    # acoustic fit is allowed to refine or trim that grid, so applying the raw
    # indices to the fitted grid silently moves every interface (for example,
    # requested slice 10 of 20 became slice 10 of 144). With both slice maps
    # at hand the requested ring's axial station is looked up in the fitted
    # grid, which the fit pins it into. Scaling the index instead preserves the
    # position only when both grids share an axial map, which the acoustic fit
    # does not: requested z = 25/50/75 mm landed at 14.0/51.9/88.5 mm.
    requested_map = (
        None
        if requested_slice_map is None
        else np.asarray(requested_slice_map, dtype=np.float64)
    )
    fitted_map = (
        None if fitted_slice_map is None else np.asarray(fitted_slice_map, dtype=np.float64)
    )
    if requested_map is not None:
        requested_last_ring = len(requested_map) - 1
    else:
        requested_last_ring = max(
            1, int(round(_num_or_default(params.get("lengthSegments"), fitted_last_ring)))
        )
    for slice_index, offset in zip(slices, offsets):
        if offset <= 0.0:
            continue
        # Imported text configs address grid slices; keep valid indices and ignore
        # out-of-range declarations rather than guessing a different topology.
        if 0 <= int(slice_index) <= requested_last_ring:
            if requested_map is not None and fitted_map is not None:
                station = float(requested_map[int(slice_index)])
                fitted_slice_index = int(np.argmin(np.abs(fitted_map - station)))
                if abs(float(fitted_map[fitted_slice_index]) - station) > 1.0e-7:
                    raise ConfigError(
                        f"Mesh.SubdomainSlices ring {int(slice_index)} (normalised "
                        f"station {station:.6g}) is missing from the fitted acoustic grid"
                    )
            else:
                fitted_slice_index = int(
                    round(int(slice_index) * fitted_last_ring / requested_last_ring)
                )
            interfaces.append(
                HornInterface(
                    slice_index=fitted_slice_index,
                    offset_mm=float(offset),
                )
            )
        else:
            logger.warning(
                "[hornlab-mesher] ignoring out-of-range Mesh.SubdomainSlices index %d "
                "(requested grid has rings 0..%d)",
                int(slice_index),
                requested_last_ring,
            )
    return tuple(interfaces)


def _reshape_grid(raw: Any, n_phi: int, n_length: int, name: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    expected = n_phi * (n_length + 1) * 3
    if arr.size != expected:
        raise ConfigError(f"{name} has {arr.size} values; expected {expected}")
    return arr.reshape(n_phi, n_length + 1, 3)


def _validate_mode_contract(params: Mapping[str, Any], mode: str) -> None:
    if mode == "freestanding":
        thickness = float(params.get("wallThickness") or 0.0)
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ConfigError(
                "freestanding mode requires Mesh.WallThickness/wall_thickness_mm > 0; "
                "use mode='bare' for an inner-only open horn"
            )


def _freeform_source_auto_angle_deg(report: Mapping[str, Any]) -> float:
    """FREEFORM's stand-in for ATH's average throat opening angle.

    FREEFORM has no ATH counterpart to match, but its throat opens at a
    different angle in H and V, so taking H alone would aim the automatic cap at
    one axis. The mean of the two is the direct analogue of the azimuthal mean
    the formula families use.
    """

    throat = report["tangentAnglesDeg"]
    return 0.5 * (float(throat["H"]["throat"]) + float(throat["V"]["throat"]))


def _source_auto_angle_deg(params: Mapping[str, Any], formula: str) -> float:
    """Throat opening angle used for automatic source cap construction."""
    if formula == "FREEFORM":
        return _freeform_source_auto_angle_deg(build_freeform_geometry(params).report())
    return azimuthal_mean(params.get("a0"), 15.5)


# Bounds on the control grid the acoustic fit is actually allowed to produce.
MAX_EFFECTIVE_PHI_PROFILES = 4096
MAX_EFFECTIVE_AXIAL_RINGS = 4097
MAX_EFFECTIVE_CONTROL_POINTS = 1_000_000
# A converged direction with more slack than this is over-resolved and gets
# trimmed back toward its requirement.
TRIM_SLACK = 0.95


def _corner_arc_edge_mask(grid: Mapping[str, Any]) -> np.ndarray | None:
    """Flag the azimuth intervals that lie inside a fixed morph corner arc.

    The full-circle angle list is built by mirroring one quadrant, so folding an
    azimuth onto ``[0, pi/2]`` with ``atan2(|sin|, |cos|)`` recovers exactly the
    quadrant sample it came from -- no tolerance games, and it survives the
    symmetry-reduced angle subsets unchanged. An interval counts as a corner
    interval only when *both* endpoints sit on the arc, so the wall interval
    that ends at the tangency point still refines with the angular budget.
    """

    freeform_spans = grid.get("freeform_corner_arc_spans")
    if freeform_spans is not None:
        phi_grid = np.asarray(grid.get("phi_grid") or (), dtype=np.float64)
        if phi_grid.ndim != 2 or phi_grid.shape[0] < 2:
            return None
        spans = list(freeform_spans)
        if len(spans) != phi_grid.shape[1]:
            return None
        folded = np.arctan2(np.abs(np.sin(phi_grid)), np.abs(np.cos(phi_grid)))
        edges = np.zeros(
            (
                phi_grid.shape[0] if bool(grid.get("full_circle", True)) else phi_grid.shape[0] - 1,
                phi_grid.shape[1],
            ),
            dtype=bool,
        )
        for ring, span in enumerate(spans):
            if not span:
                continue
            theta1, theta2 = float(span[0]), float(span[1])
            on_arc = (folded[:, ring] >= theta1 - 1.0e-12) & (
                folded[:, ring] <= theta2 + 1.0e-12
            )
            ring_edges = on_arc[:-1] & on_arc[1:]
            if bool(grid.get("full_circle", True)):
                ring_edges = np.concatenate(
                    (ring_edges, [bool(on_arc[-1] and on_arc[0])])
                )
            edges[:, ring] = ring_edges
        return edges if np.any(edges) else None

    span = grid.get("morph_corner_arc_span")
    if not span:
        return None
    theta1, theta2 = float(span[0]), float(span[1])
    angles = np.asarray(grid.get("angle_list") or (), dtype=np.float64)
    if angles.size < 2:
        return None
    folded = np.arctan2(np.abs(np.sin(angles)), np.abs(np.cos(angles)))
    on_arc = (folded >= theta1 - 1.0e-12) & (folded <= theta2 + 1.0e-12)
    edges = on_arc[:-1] & on_arc[1:]
    if bool(grid.get("full_circle", True)):
        edges = np.concatenate((edges, [bool(on_arc[-1] and on_arc[0])]))
    return edges


def _sampling_metadata(
    working: Mapping[str, Any], grid: Mapping[str, Any]
) -> dict[str, Any]:
    """Report both the nominal segment counts and the grid they actually produced."""

    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    fold = grid.get("outer_offset_fold")
    return {
        **({"outerOffsetFold": str(fold)} if fold else {}),
        "geometrySampleAngularSegments": int(working["angularSegments"]),
        "geometrySampleLengthSegments": int(working["lengthSegments"]),
        "geometrySamplePhiProfiles": n_phi,
        "geometrySampleAxialRings": n_length + 1,
        "geometrySampleControlPoints": n_phi * (n_length + 1),
        "geometrySampleCornerArcSubdivision": int(
            working.get(ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY) or 1
        ),
        # The fit samples on its own axial map (ATH's throat-clustered default
        # unless a custom z-map was given); say so rather than leave it implicit.
        "geometrySampleAxialMap": str(grid.get("sampling_mode") or ""),
    }


def _saturated_reason(
    working: Mapping[str, Any],
    max_segments: int,
    max_arc_subdivision: int,
    ordinary_factor: float,
    corner_factor: float,
    axial_factor: float,
) -> str:
    """Which cap actually stopped refinement (they are not interchangeable)."""

    hit: list[str] = []
    if ordinary_factor > 1.0 and int(working["angularSegments"]) >= max_segments:
        hit.append(f"the {max_segments}-segment azimuth limit")
    if axial_factor > 1.0 and int(working["lengthSegments"]) >= max_segments:
        hit.append(f"the {max_segments}-segment axial limit")
    if corner_factor > 1.0 and (
        int(working.get(ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY) or 1) >= max_arc_subdivision
    ):
        hit.append(f"the {max_arc_subdivision}x corner-arc subdivision limit")
    return " and ".join(hit) if hit else "a refinement step that made no progress"


def _check_effective_grid_caps(grid: Mapping[str, Any], density: MeshDensity) -> None:
    """Bound the grid that was actually produced, not the inputs that seeded it.

    ``points_per_quadrant = ceil((angularSegments + cornerSegments)/4)`` is
    mirrored over four quadrants and then grows again with the corner arc, so
    capping the two inputs never bounded the real control net.
    """

    n_phi = int(grid["grid_n_phi"])
    rings = int(grid["grid_n_length"]) + 1
    for value, limit, what in (
        (n_phi, MAX_EFFECTIVE_PHI_PROFILES, f"{n_phi} azimuth profiles"),
        (rings, MAX_EFFECTIVE_AXIAL_RINGS, f"{rings} axial rings"),
        (
            n_phi * rings,
            MAX_EFFECTIVE_CONTROL_POINTS,
            f"{n_phi * rings:,} control points",
        ),
    ):
        if value > limit:
            raise ConfigError(
                "the acoustic control grid needed for the requested mm resolution "
                f"(throat {density.throat_res_mm:g} mm, mouth "
                f"{density.mouth_res_mm:g} mm) is too large: {what} exceeds the "
                f"limit of {limit:,}. Use a coarser throat/mouth resolution."
            )


def _sampling_failure_message(
    *,
    angular_ratio: float,
    corner_angular_ratio: float,
    sagitta_ratio: float,
    axial_ratio: float,
    density: MeshDensity,
    n_phi: int,
    n_length: int,
    arc_subdivision: int,
    stop_reason: str,
    twist_note: str = "",
) -> str:
    """Name the feature that actually blocked the fit, not just the cap."""

    candidates = (
        (corner_angular_ratio, "corner-arc chord",
         "reduce Morph Corner Radius or raise the mouth resolution"),
        (angular_ratio, "azimuth chord",
         "raise the throat/mouth resolution, or simplify a discontinuous phi expression"),
        (sagitta_ratio, "azimuth curvature",
         "raise the throat/mouth resolution, or simplify a high-curvature phi expression"),
        (axial_ratio, "axial chord",
         "raise the throat/mouth resolution"),
    )
    worst_ratio, worst_name, remedy = max(candidates, key=lambda item: item[0])
    if twist_note:
        # The chord across a twisted ring pair does not shrink with the axial
        # spacing, so "raise the resolution" would be a false remedy.
        remedy = twist_note
    return (
        "cannot fit the acoustic geometry to the requested mm resolution "
        f"(throat {density.throat_res_mm:g} mm, mouth {density.mouth_res_mm:g} mm): "
        f"the {worst_name} is still {worst_ratio:.2f}x its limit; {remedy}. "
        f"Refinement stopped at {stop_reason} with a control grid of "
        f"{n_phi} x {n_length + 1} samples "
        f"(corner-arc subdivision {arc_subdivision}x)."
    )


def _requested_axial_layout(
    params: Mapping[str, Any], formula: str
) -> dict[str, Any] | None:
    """What the requested grid resolves by axial station, or ``None``.

    Only configs whose geometry depends on a station need the requested grid
    built: a morph whose start snaps to a station or sits behind a throat
    prefix, and subdomain interfaces, which name rings. Every other config
    skips the extra build and is sampled exactly as before.
    """

    slices = _number_list(params.get("subdomainSlices"))
    morph_param = params.get("morphTarget")
    if morph_param is None:
        morph_possible = False
    elif isinstance(morph_param, (int, float)):
        morph_possible = int(round(float(morph_param))) in {1, 2, 3}
    else:
        morph_possible = True
    morph_possible = morph_possible and formula != "FREEFORM"
    if not slices and not morph_possible:
        return None
    # The outer shell plays no part in stations; skip its offset work.
    grid = build_point_grid_arrays({**params, "wallThickness": 0.0})
    slice_map = [float(value) for value in grid["slice_map"]]
    t_max = float(slice_map[-1]) if slice_map and slice_map[-1] > 0.0 else 1.0
    stations: list[float] = []
    morph_start = None
    start_station = grid.get("morph_start_station")
    if morph_possible and start_station is not None and float(start_station) > 0.0:
        morph_start = float(grid["morph_start"])
        stations.append(float(start_station))
    for value in slices:
        index = int(round(value))
        if 0 < index < len(slice_map) - 1:
            stations.append(slice_map[index] / t_max)
    return {
        "slice_map": slice_map,
        "morph_start": morph_start,
        "stations": stations,
    }


def _build_acoustic_sampling_grid(
    params: Mapping[str, Any],
    density: MeshDensity,
    *,
    topology_mode: str,
    requested_layout: Any = ...,
    interpolate_mouth: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit geometry from a control grid finer than the requested final mesh.

    ``requested_layout`` is ``_requested_axial_layout`` for these ``params``
    when the caller already has it (building it costs a full grid); left at its
    default it is computed here.

    OCC B-spline surfaces approximate their control points; they do not
    interpolate every sample. Keep angular chords within twice the local mesh
    target, angular sagitta within 5% of it, and axial chords within half the
    target. Coarse viewport/export sampling therefore cannot visibly shrink the
    acoustic geometry.
    """

    working = dict(params)
    if topology_mode != "acoustic":
        grid = build_point_grid(working)
        fold = grid.get("outer_offset_fold")
        return grid, {
            # A legacy-topology build detects an offset fold exactly as an
            # acoustic one does; dropping it here left the consumer's warning
            # unreachable for those builds.
            **({"outerOffsetFold": str(fold)} if fold else {}),
            "geometrySampleAngularSegments": int(working["angularSegments"]),
            "geometrySampleLengthSegments": int(working["lengthSegments"]),
        }

    formula = _normalise_formula(working.get("type"))
    requested = (
        _requested_axial_layout(params, formula)
        if requested_layout is ...
        else requested_layout
    )
    if requested is not None:
        # The fit resamples the surface on its own axial map, so everything the
        # requested grid resolves *by station* is carried over explicitly: the
        # morph start it snapped to (a different map would snap elsewhere) and
        # the rings that must exist -- that start and every subdomain
        # interface. Preview and solve then describe one surface.
        stations = list(requested["stations"])
        if requested["morph_start"] is not None:
            working[ACOUSTIC_MORPH_START_KEY] = requested["morph_start"]
        if stations:
            working[ACOUSTIC_AXIAL_STATIONS_KEY] = stations
    if formula == "FREEFORM":
        working[FREEFORM_CONTINUOUS_COLLAPSE_KEY] = True
    if not working.get("zMapPoints") and formula != "FREEFORM":
        working["samplingMode"] = "ath-default-zmap"

    def finish_grid(grid, fitted_params):
        # Only the accepted acoustic control net becomes a surface. Oversized
        # seeds and refinement probes may be discarded, so do not calculate an
        # expensive exterior envelope for them. All inner-fit checks below and
        # the final offset validation stay unchanged.
        if formula == "OSSE" and grid.get("outer_offset_fold"):
            grid = build_point_grid(fitted_params)
        fit_metadata = {}
        if interpolate_mouth:
            from .mouth_fit import refine_mouth_grid

            grid, fit_metadata = refine_mouth_grid(fitted_params, grid)
            _check_effective_grid_caps(grid, density)
        return grid, {**_sampling_metadata(fitted_params, grid), **fit_metadata}

    # The sagitta limit is a smooth-curvature heuristic, so it is only applied
    # where the target is smooth. A morph target can carry a genuine vertex (a
    # sharp rectangle is the default WG corner setting) whose sagitta decays as
    # 1/n rather than 1/n^2; enforcing it there stalls against the segment cap
    # instead of converging. See docs/builder-invariants.md.
    enforce_angular_sagitta = (
        formula == "FREEFORM"
        or (
            _static_float_or_none(working.get("morphTarget", 0)) == 0.0
            and _static_float_or_none(working.get("gcurveType", 0)) == 0.0
        )
    )

    stretch_active = not stretch_is_inactive(working)
    # Sector fits preserve solve/CAD identity; curved morph corners also need
    # sufficient samples to represent the analytic preview between fit nodes.
    curved_morph = stretch_active and _static_float_or_none(working.get("morphTarget", 0)) == 1.0
    angular_floor = 64 if curved_morph else 4
    corner_floor = 4 if curved_morph else 1
    if curved_morph:
        working["angularSegments"] = max(angular_floor, int(working["angularSegments"]))
        working[ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY] = max(
            corner_floor, int(working.get(ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY) or 1))
    max_segments = 2048
    # 3*k intervals per corner quadrant; the cap only exists to keep a genuinely
    # non-convergent geometry from looping, not as a practical ceiling.
    max_arc_subdivision = 256
    attempts = 12
    stop_reason = f"the {attempts}-attempt refinement limit"
    best: tuple[dict[str, Any], dict[str, Any]] | None = None
    trims_left = 6
    for attempt in range(attempts):
        grid = build_point_grid(working, defer_osse_offset_repair=formula == "OSSE")
        _check_effective_grid_caps(grid, density)
        n_phi = int(grid["grid_n_phi"])
        n_length = int(grid["grid_n_length"])
        # The outer freestanding grid is derived from the inner fit and can
        # contain intentional closure transitions. The inner acoustic boundary
        # is the fidelity authority; both surfaces receive the same sampling.
        surfaces = [
            _reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
        ]

        if formula == "FREEFORM":
            ring_t = np.asarray(grid.get("slice_map"), dtype=np.float64)
            if ring_t.shape != (n_length + 1,):
                raise ConfigError("FREEFORM acoustic grid has an invalid axial slice map")
        else:
            ring_t = np.linspace(0.0, 1.0, n_length + 1, dtype=np.float64)
        ring_h = (
            float(density.throat_res_mm)
            + (float(density.mouth_res_mm) - float(density.throat_res_mm)) * ring_t
        )
        # A rounded-rectangle morph samples its corner arc with a fixed number
        # of intervals (ATH parity), so those chords do not shrink when the
        # angular budget grows -- only subdividing the arc itself moves them.
        # Classify each azimuth interval so a corner violation refines the arc
        # and an ordinary violation refines the angular budget.
        corner_edge = _corner_arc_edge_mask(grid)
        full_circle = bool(grid.get("full_circle", True))
        angular_ratio = 0.0
        corner_angular_ratio = 0.0
        ordinary_sagitta_ratio = 0.0
        corner_sagitta_ratio = 0.0
        axial_ratio = 0.0
        for points in surfaces:
            angular_delta = np.diff(points, axis=0)
            if bool(grid.get("full_circle", True)):
                angular_delta = np.concatenate(
                    (angular_delta, points[:1, :, :] - points[-1:, :, :]), axis=0
                )
            if angular_delta.size:
                angular_lengths = np.linalg.norm(angular_delta, axis=2)
                angular_ratios = angular_lengths / (2.0 * ring_h[None, :])
                if corner_edge is not None and corner_edge.shape == angular_ratios.shape:
                    ordinary_ratios = angular_ratios[~corner_edge]
                    corner_ratios = angular_ratios[corner_edge]
                elif corner_edge is not None and corner_edge.shape[0] == angular_ratios.shape[0]:
                    ordinary_ratios = angular_ratios[~corner_edge, :]
                    corner_ratios = angular_ratios[corner_edge, :]
                else:
                    ordinary_ratios = angular_ratios
                    corner_ratios = angular_ratios[:0, :]
                if ordinary_ratios.size:
                    angular_ratio = max(angular_ratio, float(np.max(ordinary_ratios)))
                if corner_ratios.size:
                    corner_angular_ratio = max(
                        corner_angular_ratio, float(np.max(corner_ratios))
                    )
                angular_points = points
                if full_circle:
                    angular_points = np.concatenate(
                        (points[-1:, :, :], points, points[:1, :, :]), axis=0
                    )
                if enforce_angular_sagitta and angular_points.shape[0] >= 3:
                    previous = angular_points[:-2, :, :]
                    current = angular_points[1:-1, :, :]
                    following = angular_points[2:, :, :]
                    chord = following - previous
                    chord_sq = np.sum(chord * chord, axis=2)
                    alpha = np.divide(
                        np.sum((current - previous) * chord, axis=2),
                        chord_sq,
                        out=np.zeros_like(chord_sq),
                        where=chord_sq > 1.0e-18,
                    )
                    projection = previous + alpha[:, :, None] * chord
                    sagitta = np.linalg.norm(current - projection, axis=2)
                    sagitta_ratios = sagitta / (0.05 * ring_h[None, :])
                    corner_triplet: np.ndarray | None = None
                    if corner_edge is not None and corner_edge.ndim == 2:
                        if full_circle:
                            corner_triplet = np.roll(corner_edge, 1, axis=0) & corner_edge
                        else:
                            corner_triplet = corner_edge[:-1, :] & corner_edge[1:, :]
                    if corner_triplet is not None and corner_triplet.shape == sagitta_ratios.shape:
                        ordinary_sagitta = sagitta_ratios[~corner_triplet]
                        corner_sagitta = sagitta_ratios[corner_triplet]
                    else:
                        ordinary_sagitta = sagitta_ratios
                        corner_sagitta = sagitta_ratios[:0]
                    if ordinary_sagitta.size:
                        ordinary_sagitta_ratio = max(
                            ordinary_sagitta_ratio, float(np.max(ordinary_sagitta))
                        )
                    if corner_sagitta.size:
                        corner_sagitta_ratio = max(
                            corner_sagitta_ratio, float(np.max(corner_sagitta))
                        )

            axial_delta = np.diff(points, axis=1)
            if axial_delta.size:
                axial_lengths = np.linalg.norm(axial_delta, axis=2)
                axial_h = 0.5 * (ring_h[:-1] + ring_h[1:])
                axial_ratio = max(
                    axial_ratio,
                    float(np.max(axial_lengths / (0.5 * axial_h[None, :]))),
                )

        # Chord error falls as 1/n; sagitta error falls as 1/n^2. Extrapolating a
        # sagitta violation linearly (as this loop used to) overshoots by sqrt of
        # the ratio, which is why the fitted grid used to depend so strongly on
        # the caller's starting segment counts.
        sagitta_ratio = max(ordinary_sagitta_ratio, corner_sagitta_ratio)
        ordinary_factor = max(angular_ratio, math.sqrt(ordinary_sagitta_ratio))
        corner_factor = max(
            corner_angular_ratio, math.sqrt(corner_sagitta_ratio)
        )
        axial_factor = axial_ratio

        converged = (
            ordinary_factor <= 1.0 and corner_factor <= 1.0 and axial_factor <= 1.0
        )
        # Each direction is trimmed on its own slack. Sharing one gate let an
        # axial term sitting at its limit pin a wildly over-resolved azimuth.
        has_corner = corner_edge is not None
        slack = {
            "angular": ordinary_factor < TRIM_SLACK,
            "axial": axial_factor < TRIM_SLACK,
            "corner": has_corner and corner_factor < TRIM_SLACK,
        }
        if converged:
            best = (grid, dict(working))
            # A grid that clears every limit with room to spare is oversized --
            # usually because the caller asked for far more segments than the mm
            # targets need. Trim toward the requirement, keeping this fit as the
            # fallback if the smaller grid turns out to violate something.
            if trims_left > 0 and any(slack.values()):
                trims_left -= 1
            else:
                return finish_grid(grid, working)
        elif best is not None:
            # The trim overshot; the last fit that met every limit stands.
            good_grid, good_working = best
            return finish_grid(good_grid, good_working)

        current_angular = int(working["angularSegments"])
        current_length = int(working["lengthSegments"])
        current_arc = int(working.get(ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY) or 1)
        # The caller's own segment counts are the probe density for the first
        # measurement, so a feature they resolved is visible to the guards. From
        # that measurement the requirement is extrapolated in either direction.
        # Outside the first probe and the trim passes refinement only increases,
        # so it cannot oscillate.
        def _retarget(
            current: int, factor: float, cap: int, *, floor: int, may_shrink: bool
        ) -> int:
            if factor > 1.0:
                return min(cap, max(current + 1, int(math.ceil(current * factor * 1.05))))
            if not may_shrink:
                return current
            return max(floor, min(cap, int(math.ceil(current * factor * 1.05))))

        first_probe = attempt == 0
        working["angularSegments"] = _retarget(
            current_angular, ordinary_factor, max_segments,
            floor=angular_floor, may_shrink=first_probe or slack["angular"],
        )
        working["lengthSegments"] = _retarget(
            current_length, axial_factor, max_segments,
            floor=4, may_shrink=first_probe or slack["axial"],
        )
        if has_corner:
            working[ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY] = _retarget(
                current_arc, corner_factor, max_arc_subdivision,
                floor=corner_floor, may_shrink=first_probe or slack["corner"],
            )

        if (
            int(working["angularSegments"]) == current_angular
            and int(working["lengthSegments"]) == current_length
            and int(working.get(ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY) or 1) == current_arc
        ):
            if best is not None:
                good_grid, good_working = best
                return finish_grid(good_grid, good_working)
            stop_reason = _saturated_reason(
                working, max_segments, max_arc_subdivision, ordinary_factor,
                corner_factor, axial_factor,
            )
            break

    raise ConfigError(
        _sampling_failure_message(
            angular_ratio=angular_ratio,
            corner_angular_ratio=corner_angular_ratio,
            sagitta_ratio=sagitta_ratio,
            axial_ratio=axial_ratio,
            density=density,
            n_phi=n_phi,
            n_length=n_length,
            arc_subdivision=int(working.get(ACOUSTIC_CORNER_ARC_SUBDIVISION_KEY) or 1),
            stop_reason=stop_reason,
            twist_note=(
                freeform_azimuth_twist_note(grid.get("phi_grid"), grid.get("slice_map"))
                if formula == "FREEFORM" and grid.get("phi_grid") is not None
                else ""
            ),
        )
    )


@dataclass(frozen=True)
class ResolvedGeometry:
    """What a config resolves to before anything is meshed or exported.

    Meshing and CAD export share every step up to this point, so they share the
    code that produces it -- a STEP file and a solve therefore describe the same
    waveguide by construction rather than by two implementations agreeing.
    """

    geometry: PointGridHornGeometry
    density: MeshDensity
    formula: str
    mode: str
    quadrants: str
    native_symmetry_plane: str | None
    scale_to_metres: bool
    sampling_metadata: dict[str, Any] = field(default_factory=dict)
    freeform_report: dict[str, Any] | None = None


@stretch_config_errors
def resolve_geometry(
    config: Mapping[str, Any],
    *,
    allow_large_mesh: bool | None = None,
) -> ResolvedGeometry:
    """Build the point-grid geometry a config describes, without meshing it."""

    params, formula, mode = build_geometry_params(config)
    _validate_mode_contract(params, mode)
    mesh = _section(config, "mesh")
    enclosure = _section(config, "enclosure")
    topology_mode = _mesh_topology_mode(mesh)
    density = _mesh_density_from_config(config, allow_large_mesh=allow_large_mesh)
    enclosure_obj = _enclosure_from_config(config, mesh, enclosure, formula)

    wall_thickness_mm = float(params.get("wallThickness") or 0.0)
    acoustic_wall_floor_mm = 0.025 * min(
        float(density.throat_res_mm),
        float(density.mouth_res_mm),
        float(density.rear_res_mm),
    )
    if (
        topology_mode == "acoustic"
        and mode == "freestanding"
        and 0.0 < wall_thickness_mm < acoustic_wall_floor_mm
    ):
        raise ConfigError(
            f"wall_thickness_mm={wall_thickness_mm:g} is below the stable acoustic "
            f"feature floor {acoustic_wall_floor_mm:g} mm for the requested mesh; "
            "set wall thickness to 0 for bare mode or use a resolvable thickness"
        )

    requested_layout = (
        _requested_axial_layout(params, formula)
        if topology_mode == "acoustic"
        else None
    )
    grid, geometry_sampling_metadata = _build_acoustic_sampling_grid(
        params,
        density,
        topology_mode=topology_mode,
        requested_layout=requested_layout,
        interpolate_mouth=_mesh_surface_fit(mesh) != "approximate",
    )

    n_phi = int(grid["grid_n_phi"])
    n_length = int(grid["grid_n_length"])
    inner_points = _reshape_grid(grid["inner_points"], n_phi, n_length, "inner_points")
    outer_points = None
    if grid.get("outer_points") is not None and enclosure_obj is None:
        outer_points = _reshape_grid(
            grid["outer_points"], n_phi, n_length, "outer_points"
        )
    for name, points in (("inner", inner_points), ("outer", outer_points)):
        if points is not None and not np.all(np.isfinite(points)):
            bad = int(np.count_nonzero(~np.isfinite(points).all(axis=2)))
            raise ConfigError(
                f"the resolved {formula} geometry has {bad} non-finite {name} "
                "control points; the profile parameters leave the formula's domain"
            )

    interface_offsets = _number_list(params.get("interfaceOffset"))
    if topology_mode == "acoustic" and _number_list(params.get("subdomainSlices")):
        interfaces = _interfaces_from_params(
            params,
            n_length,
            requested_slice_map=requested_layout["slice_map"],
            fitted_slice_map=grid.get("slice_map"),
        )
    else:
        interfaces = _interfaces_from_params(params, n_length)
    # ATH builds free-standing subdomain models (mouth interface I1-2 plus an
    # SD2 exterior); this mesher only builds interfaces for enclosure models, so
    # an explicit request on other modes must fail loudly instead of silently
    # dropping the subdomain topology.
    if mode != "enclosure" and (
        _number_list(params.get("subdomainSlices")) or interface_offsets
    ):
        raise ConfigError(
            "Mesh.SubdomainSlices/Mesh.InterfaceOffset request subdomain interfaces, "
            "which are only supported for enclosure builds "
            "(ATH's free-standing two-subdomain construction is not implemented)"
        )
    quadrants = _normalised_quadrants(params.get("quadrants"))
    native_plane = _native_symmetry_plane_for_quadrants(quadrants)
    source_auto_angle_deg = azimuthal_mean(params.get("a0"), 15.5)
    freeform_axis_samples_mm: np.ndarray | None = None
    freeform_report: dict[str, Any] | None = None
    if formula == "FREEFORM":
        freeform_geometry = build_freeform_geometry(params)
        freeform_report = freeform_geometry.report()
        source_auto_angle_deg = _freeform_source_auto_angle_deg(freeform_report)
        slice_map = np.asarray(grid.get("slice_map"), dtype=np.float64)
        z0 = float(params["profileH"]["points"][0][0])
        analytic_z = z0 + slice_map * freeform_geometry.length_mm
        analytic_h, analytic_v = freeform_geometry.evaluate_radii(analytic_z)
        if mode == "infinite-baffle":
            analytic_z = analytic_z - analytic_z[-1]
        scale = float(eval_param(params.get("scale"), 0.0, 1.0))
        h_samples = np.column_stack(
            (analytic_h, np.zeros_like(analytic_h), analytic_z)
        )
        v_samples = np.column_stack(
            (np.zeros_like(analytic_v), analytic_v, analytic_z)
        )
        freeform_axis_samples_mm = scale * np.vstack((h_samples, v_samples))
    geometry_cls = PointGridHornGeometry if stretch_is_inactive(params) else _StretchedPointGridHornGeometry
    probe_kwargs = {}
    if outer_points is not None and "outer_clearance_points" in grid:
        from .geometry import (
            _MouthFittedPointGridHornGeometry, _MouthFittedStretchedPointGridHornGeometry,
        )

        geometry_cls = (_MouthFittedPointGridHornGeometry if stretch_is_inactive(params)
                        else _MouthFittedStretchedPointGridHornGeometry)
        probe_kwargs["outer_clearance_points_mm"] = np.asarray(
            grid["outer_clearance_points"], dtype=np.float64)
    geometry = geometry_cls(
        inner_points=inner_points,
        outer_points=outer_points,
        topology_mode=topology_mode,
        surface_fit=_mesh_surface_fit(mesh),
        # ATH does not scale Mesh.WallThickness by global Scale; the rear-cap
        # depth follows the unscaled wall offset.
        wall_thickness_mm=float(params["wallThickness"] or 0.0),
        preserve_grid=(
            topology_mode == "legacy"
            and _bool(mesh, names=("preserve_grid", "preserveGrid"), default=False)
        ),
        closed=bool(grid.get("full_circle", True)),
        symmetry_planes=_symmetry_planes_for_quadrants(quadrants),
        vertical_offset_mm=float(grid.get("vertical_offset_mm", 0.0) or 0.0),
        freeform_axis_samples_mm=freeform_axis_samples_mm,
        freeform_report=freeform_report,
        source_shape=int(_num_or_default(params.get("sourceShape"), 1)),
        source_radius_mm=_num_or_default(params.get("sourceRadius"), -1),
        source_curv=int(_num_or_default(params.get("sourceCurv"), 0)),
        source_auto_angle_deg=source_auto_angle_deg,
        interface_offset_mm=float(interface_offsets[0] if interface_offsets else 0.0),
        interfaces=interfaces,
        enclosure=enclosure_obj,
        infinite_baffle=(mode == "infinite-baffle"),
        **probe_kwargs,
    )
    scale_to_metres = _bool(
        mesh, names=("scale_to_metres", "scaleToMetres"), default=True
    )
    return ResolvedGeometry(
        geometry=geometry,
        density=density,
        formula=formula,
        mode=mode,
        quadrants=quadrants,
        native_symmetry_plane=native_plane,
        scale_to_metres=scale_to_metres,
        sampling_metadata=geometry_sampling_metadata,
        freeform_report=freeform_report,
    )


@stretch_config_errors
def build_from_config(
    config: Mapping[str, Any],
    output_path: str | Path,
    *,
    allow_large_mesh: bool | None = None,
) -> BuildResult:
    from .mid_ports import parse_mid_ports

    resolved = resolve_geometry(config, allow_large_mesh=allow_large_mesh)
    geometry = resolved.geometry
    # Opt-in mid_ports block: absent/empty = None so the mesh path is
    # byte-identical to the pre-mid_ports mesher (no call into apply_mid_ports).
    mid_ports_recipe = parse_mid_ports(_section(config, "mid_ports") or None)
    chose_automatically = (
        _mesh_surface_fit(_section(config, "mesh")) == "auto"
        and getattr(geometry, "surface_fit", None) == "interpolate"
    )
    try:
        mesh_path, info = build_mesh_with_info(
            geometry,
            resolved.density,
            output_path,
            scale_to_metres=resolved.scale_to_metres,
            mid_ports=mid_ports_recipe,
        )
    except TriangleBudgetExceeded:
        # A budget refusal is not a fit failure: the other fit meshes about the
        # same triangle count, so retrying only repeats a full mesh generation
        # and can ship the inward-biased surface just because it landed under
        # the limit.
        raise
    except MesherError:
        # Automatic fitting can retry a geometry that the interpolating
        # surface could not mesh. Keep this general recovery path even though
        # the coarse small-horn sampling defect is covered in the mesher.
        #
        # Only the automatic choice falls back. An explicit
        # ``surface_fit = "interpolate"`` still fails loudly, because someone
        # who named it wants to know.
        if not chose_automatically:
            raise
        logger.warning(
            "[hornlab-mesher] the interpolating surface fit could not mesh this "
            "geometry; falling back to the approximating fit, whose acoustic "
            "wall sits slightly inside the design. Set mesh.surface_fit "
            "explicitly to choose either one and see the failure."
        )
        geometry = replace(geometry, surface_fit="approximate")
        mesh_path, info = build_mesh_with_info(
            geometry,
            resolved.density,
            output_path,
            scale_to_metres=resolved.scale_to_metres,
            mid_ports=mid_ports_recipe,
        )
    mesh_report = _mesh_report(info.physical_groups, info.edge_stats_mm)
    freeform_report = resolved.freeform_report
    fit_used = getattr(geometry, "surface_fit", None)
    return BuildResult(
        mesh_path=mesh_path,
        formula=resolved.formula,
        mode=resolved.mode,
        n_vertices=info.n_vertices,
        n_triangles=info.n_triangles,
        units=info.units,
        physical_groups=info.physical_groups,
        quadrants=resolved.quadrants,
        native_symmetry_plane=resolved.native_symmetry_plane,
        native_check_open_edges=_native_check_open_edges_for_mode(resolved.mode),
        mesh_report=mesh_report,
        solve_cost=cost.estimate_solve_cost(info.n_triangles).to_dict(),
        metadata={
            **info.metadata,
            **resolved.sampling_metadata,
            **({"freeformReport": freeform_report} if freeform_report is not None else {}),
            **({"surfaceFit": fit_used} if fit_used is not None else {}),
        },
    )


__all__ = [
    "BuildResult",
    "ResolvedGeometry",
    "build_from_config",
    "resolve_geometry",
    "build_geometry_params",
    "_bool",
    "_enclosure_from_config",
    "_first_number",
    "_float",
    "_int",
    "_interfaces_from_params",
    "_normalise_formula",
    "_normalise_mode",
    "_number_list",
    "_pick",
    "_reshape_grid",
    "_scalar_or_expr",
    "_section",
]
