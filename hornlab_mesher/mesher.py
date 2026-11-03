from __future__ import annotations

import math
import tempfile
import threading
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np

from .builders import (
    build_osse_waveguide,
    build_point_grid,
)
from .builders._occ import add_physical_groups
from .density import (
    CLEARANCE_CAP_BUDGET_ADVICE,
    TriangleBudgetExceeded as _PremeshTriangleBudgetExceeded,
    _parse_quadrant_resolutions,
    clearance_caps_active,
    configure_density,
    effective_triangle_limit,
)
from .edges import build_edge_table
from .geometry import (
    MESH_ALGORITHM_MESHADAPT,
    BuiltGeometry,
    HornGeometry,
    MeshDensity,
    MeshInfo,
    OsseHornGeometry,
    PointGridHornGeometry,
    validate_mesh_density,
)
from .native_env import preserve_native_windows_path
from .normals import (
    MeshOrientationError,
    remove_degenerate_triangles,
    repair_orientation,
    validate_orientation,
)
from .quality import evaluate_quality_gate, mesh_quality_report
from .tags import PHYSICAL_NAMES, PhysicalGroup


_GMSH_LOCK = threading.RLock()


class MesherError(Exception):
    pass


# Options the pipeline sets itself, or that would silently override the
# millimetre density contract if inherited from a caller-owned session. The
# build restores every one of them afterwards.
_PINNED_MESH_OPTIONS: dict[str, float] = {
    # Results depend on these: a caller's size factor rescales every mm target,
    # and a second-order or recombined mesh is not a triangle mesh at all.
    "Mesh.MeshSizeFactor": 1.0,
    "Mesh.ElementOrder": 1.0,
    "Mesh.Smoothing": 1.0,
    "Mesh.SaveAll": 0.0,
    "Mesh.RecombineAll": 0.0,
    "Mesh.SubdivisionAlgorithm": 0.0,
    "Mesh.MinimumCirclePoints": 7.0,
    # The mesh is written as ASCII msh 2.2; a caller's binary setting would
    # change the bytes of the file without changing its meaning.
    "Mesh.Binary": 0.0,
}
_RESTORED_MESH_OPTIONS = (
    "General.Terminal",
    "Geometry.Tolerance",
    "Geometry.ToleranceBoolean",
    "Mesh.Algorithm",
    "Mesh.MshFileVersion",
    "Mesh.MeshSizeMin",
    "Mesh.MeshSizeMax",
    "Mesh.MeshSizeExtendFromBoundary",
    "Mesh.MeshSizeFromPoints",
    "Mesh.MeshSizeFromCurvature",
    *_PINNED_MESH_OPTIONS,
)


class TriangleBudgetExceeded(MesherError):
    """A build was refused for exceeding its triangle budget, not for a fit or
    topology failure.

    Kept as a distinct type so a caller -- notably the ``auto`` surface-fit
    fallback in ``config_builder.build_from_config`` -- can tell "this
    geometry needs a smaller mesh or a larger budget" apart from "the
    interpolating fit could not mesh this geometry, retry with the
    approximating one". Retrying a budget refusal with a different fit wastes
    a full mesh generation and usually still fails, since both fits mesh a
    similar triangle count. The message text is unchanged from a plain
    ``MesherError`` for this refusal (WG matches substrings of it), so this
    subclass only changes what ``isinstance`` sees, never what str() returns.
    """


def _acoustic_geometry(
    geometry: HornGeometry,
    density: MeshDensity,
) -> tuple[HornGeometry, dict[str, object]]:
    """Apply semantic acoustic LOD without changing manufacturing geometry."""

    if not isinstance(geometry, PointGridHornGeometry) or geometry.enclosure is None:
        return geometry, {}
    if geometry.topology_mode == "legacy" or geometry.preserve_grid:
        return geometry, {"meshTopologyMode": "legacy"}

    from .builders.enclosure import enclosure_box_bounds

    enclosure = geometry.enclosure
    bounds = enclosure_box_bounds(
        geometry.inner_points,
        enclosure,
        closed=geometry.closed,
        symmetry_planes=tuple(geometry.symmetry_planes),
    )
    clamped_edge = float(bounds.get("clamped_edge", 0.0) or 0.0)
    edge_depth = float(bounds.get("edge_depth", 0.0) or 0.0)
    front = _parse_quadrant_resolutions(density.enc_front_res_mm, density.mouth_res_mm)
    back = _parse_quadrant_resolutions(density.enc_back_res_mm, density.mouth_res_mm)
    adjacent_h = min(*front, *back)
    feature_length = (
        0.5 * math.pi * clamped_edge
        if int(enclosure.edge_type) == 1
        else math.hypot(clamped_edge, edge_depth)
    )
    suppressed = clamped_edge > 0.0 and feature_length < adjacent_h
    metadata: dict[str, object] = {
        "meshTopologyMode": "acoustic",
        "acousticEnclosureRequestedEdgeMm": float(enclosure.edge_mm),
        "acousticEnclosureClampedEdgeMm": clamped_edge,
        "acousticEnclosureFeatureLengthMm": feature_length,
        "acousticEnclosureAdjacentResolutionMm": adjacent_h,
        "acousticEnclosureEdgeSuppressed": suppressed,
    }
    if not suppressed:
        return geometry, metadata
    # The rounded-rectangle sharp path is the robust semantic representation
    # for both suppressed fillets and suppressed chamfers.
    return replace(
        geometry,
        enclosure=replace(enclosure, edge_mm=0.0, edge_type=1),
    ), metadata


def _dispatch_builder(geometry: HornGeometry) -> BuiltGeometry:
    if isinstance(geometry, OsseHornGeometry):
        return build_osse_waveguide(geometry)
    if isinstance(geometry, PointGridHornGeometry):
        return build_point_grid(geometry)
    raise TypeError(f"unsupported geometry type: {type(geometry)!r}")


def _freeform_profile_deviation_mm(
    geometry: PointGridHornGeometry, built: BuiltGeometry
) -> float | None:
    """Measure fitted OCC wall deviation from FREEFORM's analytic H/V cuts."""

    samples = geometry.freeform_axis_samples_mm
    if samples is None or geometry.topology_mode != "acoustic" or geometry.preserve_grid:
        return None
    expected = np.asarray(samples, dtype=np.float64)
    if expected.ndim != 2 or expected.shape[1] != 3 or expected.shape[0] == 0:
        return None
    surface_tags = [int(tag) for tag in built.mesh_surface_groups.get("inner", ())]
    if not surface_tags:
        return None

    gmsh = __import__("gmsh")
    best_distance = np.full(expected.shape[0], np.inf, dtype=np.float64)
    best_radial = np.full(expected.shape[0], np.nan, dtype=np.float64)
    coordinates = expected.reshape(-1).tolist()
    for tag in surface_tags:
        closest, _parameters = gmsh.model.getClosestPoint(2, tag, coordinates)
        projected = np.asarray(closest, dtype=np.float64).reshape(-1, 3)
        distance = np.linalg.norm(projected - expected, axis=1)
        improve = distance < best_distance
        best_distance[improve] = distance[improve]
        best_radial[improve] = np.linalg.norm(projected[improve, :2], axis=1)
    if not np.all(np.isfinite(best_radial)):
        raise MesherError("could not sample the fitted FREEFORM OCC wall surface")
    expected_radial = np.linalg.norm(expected[:, :2], axis=1)
    return float(np.max(np.abs(best_radial - expected_radial)))


def build_mesh(
    geometry: HornGeometry,
    density: MeshDensity | str | Path | None = None,
    output_path: str | Path | None = None,
    scale_to_metres: bool = True,
) -> Path:
    """Build a tagged, validated Gmsh ``.msh`` file."""

    return build_mesh_with_info(
        geometry,
        density,
        output_path,
        scale_to_metres=scale_to_metres,
    )[0]


def build_mesh_with_info(
    geometry: HornGeometry,
    density: MeshDensity | str | Path | None = None,
    output_path: str | Path | None = None,
    scale_to_metres: bool = True,
    quality_gate: str = "report",
    mid_ports: object | None = None,
) -> tuple[Path, MeshInfo]:
    """Build a ``.msh`` file and return it with the inspection info.

    Equivalent to ``build_mesh`` followed by ``load_mesh``, but the info is
    collected from the post-processed arrays at write time so the file is not
    read back.

    ``quality_gate`` selects what happens to the element-shape and chord
    deviation measures in ``info.metadata["quality"]``. ``"report"``, the
    default, only records them; ``"strict"`` additionally refuses a build whose
    measures cross the thresholds in ``quality``. The default is ``"report"``
    for the same reason the self-intersection guard shipped as a warning: seven
    of the thirty meshes in the application's own library cross the element
    shape threshold and twelve cross the chord one, so a gate enabled by default
    would reject a large part of the existing library on the day it landed.
    """

    if isinstance(density, (str, Path)) and output_path is None:
        output_path = density
        density = None
    mesh_density = density if isinstance(density, MeshDensity) else MeshDensity()
    validate_mesh_density(mesh_density)

    import gmsh

    owns_out_path = output_path is None
    if output_path is None:
        handle = tempfile.NamedTemporaryFile(
            prefix="hornlab-mesher-", suffix=".msh", delete=False
        )
        out_path = Path(handle.name)
        handle.close()
    else:
        out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with _GMSH_LOCK:
        initialized_here = False
        saved_options: dict[str, float] = {}
        build_succeeded = False
        raw_path: Path | None = None
        staged_path: Path | None = None
        try:
            if not gmsh.isInitialized():
                # interruptible=False skips gmsh's SIGINT handler, which only
                # the main thread may install — required for worker threads.
                #
                # On Windows the open can truncate the native PATH, so it runs
                # inside the guard. Counterparts: the session open in
                # cad.write_step, and the application's own guarded session
                # (Waveguide Generator server/mesh/gmsh_worker.py). Recording
                # the open inside the guard means a failed PATH restore still
                # closes the session below.
                with preserve_native_windows_path():
                    gmsh.initialize(interruptible=False)
                    initialized_here = True
            if not initialized_here:
                # A caller-owned session: hand it back as it was found.
                for name in _RESTORED_MESH_OPTIONS:
                    try:
                        saved_options[name] = gmsh.option.getNumber(name)
                    except Exception:
                        pass
            for name, value in _PINNED_MESH_OPTIONS.items():
                gmsh.option.setNumber(name, value)
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
            gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-8)
            gmsh.clear()
            gmsh.model.add("HornLabMesher")

            acoustic_geometry, acoustic_metadata = _acoustic_geometry(
                geometry, mesh_density
            )
            built = _dispatch_builder(acoustic_geometry)
            built.metadata.update(acoustic_metadata)
            gmsh.model.occ.synchronize()
            if isinstance(acoustic_geometry, PointGridHornGeometry):
                deviation = _freeform_profile_deviation_mm(acoustic_geometry, built)
                if deviation is not None:
                    built.metadata["freeformProfileDeviationMm"] = deviation
                if acoustic_geometry.freeform_report is not None:
                    built.metadata["freeformReport"] = acoustic_geometry.freeform_report

            configure_density(built, mesh_density)
            add_physical_groups(built.surface_groups)

            # A builder that leaves ``mesh_algorithm`` unset asks for gmsh's own
            # default, which is MeshAdapt. Naming it keeps the fallback readable
            # and matches what BARE deliberately selects by returning ``None``.
            gmsh.option.setNumber(
                "Mesh.Algorithm",
                int(built.mesh_algorithm or MESH_ALGORITHM_MESHADAPT),
            )
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            # A very coarse closed interpolating spline can otherwise be
            # sampled with only three curve points. Gmsh then produces an
            # nonmanifold edge on small bare horns. Keep at least five
            # samples on these curves, independent of the model's units.
            if (
                isinstance(acoustic_geometry, PointGridHornGeometry)
                and acoustic_geometry.surface_fit == "interpolate"
            ):
                previous_curve_points = gmsh.option.getNumber("Mesh.MinimumCurvePoints")
                try:
                    gmsh.option.setNumber(
                        "Mesh.MinimumCurvePoints", max(previous_curve_points, 5)
                    )
                    gmsh.model.mesh.generate(2)
                finally:
                    gmsh.option.setNumber("Mesh.MinimumCurvePoints", previous_curve_points)
            else:
                gmsh.model.mesh.generate(2)
            gmsh.model.mesh.removeDuplicateNodes()
            if (getattr(acoustic_geometry, 'outer_clearance_points_mm', None) is not None
                    and np.all(np.diff(acoustic_geometry.inner_points[..., 2], axis=1) >= -1e-6)):
                from .facet_fit import repair_fitted_wall_mesh

                repair_fitted_wall_mesh(built.mesh_surface_groups)

            with tempfile.NamedTemporaryFile(
                prefix="hornlab-mesher-raw-", suffix=".msh", delete=False
            ) as tmp:
                raw_path = Path(tmp.name)
            gmsh.write(str(raw_path))

            with tempfile.NamedTemporaryFile(
                dir=out_path.parent,
                prefix=f".{out_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                staged_path = Path(tmp.name)

            is_infinite_baffle = bool(getattr(geometry, "infinite_baffle", False))
            info = _postprocess_mesh(
                raw_path,
                staged_path,
                built.source_axis,
                scale_to_metres,
                symmetry_snap_axes=built.symmetry_snap_axes,
                symmetry_snap_tol_mm=built.symmetry_snap_tol_mm,
                vertical_offset_mm=float(
                    getattr(geometry, "vertical_offset_mm", 0.0) or 0.0
                ),
                open_shell_wall_points_mm=built.open_shell_wall_points_mm,
                open_shell_wall_normals=built.open_shell_wall_normals,
                # Coupled infinite-baffle meshes are interior-domain BIE
                # surfaces. Keep one consistent negative-volume winding:
                # source/wall normals point into the cavity and aperture
                # normals point -Z. Rayleigh exterior evaluation is selected by
                # the aperture tag, not by changing triangle winding.
                require_positive_volume=not is_infinite_baffle,
                infinite_baffle=is_infinite_baffle,
                shell_surface_groups=(built.mesh_surface_groups
                    if isinstance(geometry, PointGridHornGeometry)
                    and geometry.build_mode.value == 'freestanding' else None),
                wall_thickness_mm=float(getattr(geometry, 'wall_thickness_mm', 0)),
                shell_mesh_density=mesh_density,
                mid_ports=mid_ports,
            )
            limit = effective_triangle_limit(built, mesh_density)
            if limit is not None:
                built.metadata["meshTriangleCount"] = int(info.n_triangles)
                built.metadata["meshTriangleLimitExceeded"] = bool(
                    info.n_triangles > limit
                )
            if (
                limit is not None
                and not mesh_density.allow_large_mesh
                and info.n_triangles > limit
            ):
                if clearance_caps_active(built.metadata):
                    advice = CLEARANCE_CAP_BUDGET_ADVICE
                else:
                    advice = (
                        "increase the relevant mm resolution, raise "
                        "max_triangles, or set allow_large_mesh=true explicitly"
                    )
                raise TriangleBudgetExceeded(
                    f"generated mesh contains {info.n_triangles:,} triangles, exceeding "
                    f"the effective limit {limit:,}; {advice}"
                )
            if built.metadata:
                info.metadata.update(built.metadata)
            if quality_gate == "strict":
                verdict = evaluate_quality_gate(
                    info.metadata.get("quality") or {}, strict=True
                )
                if not verdict.passed:
                    raise MesherError("; ".join(verdict.failures))
            elif quality_gate != "report":
                raise MesherError(
                    f"quality_gate must be 'report' or 'strict', "
                    f"got {quality_gate!r}"
                )
            _validate_physical_tags(set(info.physical_groups))
            staged_path.replace(out_path)
            staged_path = None
            info = replace(info, path=out_path)
            build_succeeded = True
            return out_path, info
        except Exception as exc:
            if isinstance(exc, (TriangleBudgetExceeded, _PremeshTriangleBudgetExceeded)):
                raise TriangleBudgetExceeded(f"mesh build failed: {exc}") from exc
            raise MesherError(f"mesh build failed: {exc}") from exc
        finally:
            if raw_path is not None:
                raw_path.unlink(missing_ok=True)
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
            if owns_out_path and not build_succeeded:
                out_path.unlink(missing_ok=True)
            if initialized_here and gmsh.isInitialized():
                with preserve_native_windows_path():
                    gmsh.finalize()
            elif saved_options and gmsh.isInitialized():
                for name, value in saved_options.items():
                    gmsh.option.setNumber(name, value)


def load_mesh(path: str | Path) -> MeshInfo:
    """Load and inspect a HornLab ``.msh`` file without creating a BEM grid."""

    mesh_path = Path(path)
    mesh = meshio.read(mesh_path)
    triangles, phys = _triangles_and_physical_tags(mesh)
    if len(triangles) == 0:
        raise MesherError("mesh contains no triangles")
    names = _physical_names(mesh)
    return MeshInfo(
        path=mesh_path,
        n_vertices=int(len(mesh.points)),
        n_triangles=int(len(triangles)),
        physical_groups=names,
        bounding_box=(np.min(mesh.points, axis=0), np.max(mesh.points, axis=0)),
        units="m" if _looks_like_metres(mesh.points) else "mm",
    )


def _postprocess_mesh(
    raw_path: Path,
    out_path: Path,
    source_axis: str,
    scale_to_metres: bool,
    *,
    symmetry_snap_axes: tuple[str, ...] = (),
    symmetry_snap_tol_mm: float = 1.0e-6,
    vertical_offset_mm: float = 0.0,
    open_shell_wall_points_mm: np.ndarray | None = None,
    open_shell_wall_normals: np.ndarray | None = None,
    require_positive_volume: bool = True,
    infinite_baffle: bool = False,
    shell_surface_groups: dict[str, list[int]] | None = None,
    wall_thickness_mm: float = 0.0,
    shell_mesh_density: MeshDensity | None = None,
    mid_ports: object | None = None,
) -> MeshInfo:
    mesh = meshio.read(raw_path)
    triangles, phys = _triangles_and_physical_tags(mesh)
    if len(triangles) == 0:
        raise MesherError("gmsh produced no triangle elements")

    points = np.asarray(mesh.points, dtype=np.float64)
    # Keep OCC patch identity through both removal passes. Physical tags alone
    # deliberately combine bore, shell and caps and cannot identify a repair.
    surfaces = None
    if shell_surface_groups is not None:
        surfaces = np.concatenate([
            mesh.cell_data['gmsh:geometrical'][i]
            for i, block in enumerate(mesh.cells)
            if block.type in ('triangle', 'triangle3')])
        phys = np.column_stack((phys, surfaces))
    # Snap BEFORE welding: snapping near-plane vertices onto the symmetry
    # planes can create coincident nodes, and only a subsequent weld folds
    # them back into single vertices (welding first would leave the
    # near-identical rows that make dense BEM solves singular).
    _snap_symmetry_planes(points, symmetry_snap_axes, symmetry_snap_tol_mm)
    # Gmsh stitches adjacent OCC patch boundaries with near-duplicate nodes
    # (micrometres apart on fine grids); welding them prevents overlapping
    # elements whose near-identical rows make dense BEM solves singular.
    triangles = _weld_near_duplicate_vertices(points, triangles, tol_mm=5.0e-3)
    triangles, phys = _remove_symmetry_plane_slivers(
        points, triangles, phys, symmetry_snap_axes
    )
    triangles, phys, _ = remove_degenerate_triangles(
        points, triangles, phys, min_quality=1.0e-4
    )
    if len(triangles) == 0:
        raise MesherError("gmsh produced only degenerate triangle elements")
    shell_stats = None
    if shell_surface_groups is not None:
        phys, surfaces = phys[:, 0], phys[:, 1]
        from .shell_facets import validate_shell_facets

        shell_stats = validate_shell_facets(
            points, triangles, surfaces, phys, shell_surface_groups,
            wall_thickness_mm, symmetry_snap_axes, mesh_density=shell_mesh_density)
    # Gmsh/OCC can emit an otherwise valid canonical surface with the whole
    # triangle set inward-wound, most visibly on non-monotonic freestanding
    # R-OSSE point grids. The mesher owns output orientation, so normalize its
    # generated triangles before the validation/write boundary.
    triangles, _ = repair_orientation(
        points,
        triangles,
        phys,
        source_axis=source_axis,
        open_shell_wall_points_mm=open_shell_wall_points_mm,
        open_shell_wall_normals=open_shell_wall_normals,
    )
    report = validate_orientation(
        points,
        triangles,
        phys,
        source_axis=source_axis,
        require_watertight=False,
        require_edge_consistency=True,
        require_positive_volume=require_positive_volume,
        require_source_normal=True,
        # Bare open shells are the one mode whose winding no closed-surface
        # indicator can check, and the mode whose repair depends on
        # builder-supplied references. Re-derive the contract from the emitted
        # triangles alone so a lost or wrong reference fails the build instead
        # of shipping a horn that solves as an acoustically invisible sheet.
        require_open_shell_bore_normal=open_shell_wall_points_mm is not None,
    )
    if infinite_baffle:
        _validate_infinite_baffle_contract(
            points,
            triangles,
            phys,
            report=report,
            symmetry_snap_axes=symmetry_snap_axes,
            symmetry_snap_tol_mm=symmetry_snap_tol_mm,
        )
    # Runs before the vertical offset, while the cut planes are still on the
    # coordinate axes.
    _validate_closed_shell_contract(
        points,
        triangles,
        phys,
        symmetry_snap_axes=symmetry_snap_axes,
        symmetry_snap_tol_mm=symmetry_snap_tol_mm,
        is_open_shell=open_shell_wall_points_mm is not None,
    )
    mid_ports_report = None
    if mid_ports is not None:
        # Opt-in L1 flush mid-port patches: re-tag wall triangles after
        # orientation is locked (same winding as walls; SOURCE_TAGS includes 10).
        from .mid_ports import apply_mid_ports

        points, triangles, phys, mid_ports_report = apply_mid_ports(
            points, triangles, phys, mid_ports
        )
    points, triangles = _compact_unused_vertices(points, triangles)
    if vertical_offset_mm:
        # Mesh.VerticalOffset: rigid +y placement of the finished reduced/full
        # model, applied here (still in millimetres) after every cut-plane snap
        # and the enclosure build have run at the origin. The declared symmetry
        # plane stays at y=0, so a y-cut mesh reconstructs about y=0 as ATH does.
        points[:, 1] += float(vertical_offset_mm)
    edge_stats = _edge_stats_by_tag(points, triangles, phys)
    if scale_to_metres:
        points = points * 0.001

    used_tags = sorted({int(tag) for tag in phys.tolist()})
    physical_names = {
        int(tag): PHYSICAL_NAMES.get(int(tag), f"SD1D{1000 + int(tag) - 1}")
        for tag in used_tags
    }
    out_mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", triangles.astype(np.int64))],
        cell_data={
            "gmsh:physical": [phys.astype(np.int32)],
            "gmsh:geometrical": [phys.astype(np.int32)],
        },
        field_data={
            name: np.array([tag, 2], dtype=np.int32)
            for tag, name in physical_names.items()
        },
    )
    meshio.write(out_path, out_mesh, file_format="gmsh22", binary=False)
    return MeshInfo(
        path=out_path,
        n_vertices=int(len(points)),
        n_triangles=int(len(triangles)),
        physical_groups=physical_names,
        bounding_box=(np.min(points, axis=0), np.max(points, axis=0)),
        # Generated arrays have known units. Keep load_mesh's conservative
        # magnitude heuristic for arbitrary third-party files, but never use it
        # to mislabel a small unscaled build as metres.
        units="m" if scale_to_metres else "mm",
        edge_stats_mm=edge_stats,
        # Measured here rather than by a caller because this is the one place
        # that holds the finished arrays and knows their units. Recording it
        # costs a pass over the triangles; acting on it is the caller's choice.
        metadata={
            **({'shellFacetValidation': shell_stats}
               if shell_stats is not None and shell_stats['pairs_before'] else {}),
            **({"midPorts": mid_ports_report} if mid_ports_report is not None else {}),
            "quality": mesh_quality_report(
                points,
                triangles,
                vertex_units="m" if scale_to_metres else "mm",
            )
        },
    )


def _tag_axis_projections(
    points: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
    *,
    tag: int,
    axis_idx: int,
) -> np.ndarray:
    mask = phys == int(tag)
    if not np.any(mask):
        return np.empty((0,), dtype=np.float64)
    p0 = points[triangles[mask, 0]]
    p1 = points[triangles[mask, 1]]
    p2 = points[triangles[mask, 2]]
    return np.cross(p1 - p0, p2 - p0)[:, axis_idx]


def _validate_infinite_baffle_contract(
    points: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
    *,
    report: object,
    symmetry_snap_axes: tuple[str, ...],
    symmetry_snap_tol_mm: float,
) -> None:
    """Enforce the coupled interior-BEM/Rayleigh mesh contract at write time."""

    wall_tag = int(PhysicalGroup.RIGID_WALL)
    source_tag = int(PhysicalGroup.PRIMARY_SOURCE)
    aperture_tag = int(PhysicalGroup.MOUTH_APERTURE)
    used_tags = {int(value) for value in phys.tolist()}
    missing = sorted({wall_tag, source_tag, aperture_tag} - used_tags)
    if missing:
        raise MesherError(
            "infinite-baffle mesh is missing required physical tag(s): "
            + ", ".join(str(tag) for tag in missing)
        )

    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    area_eps = max(1.0e-12, scale * scale * 1.0e-12)
    volume_eps = max(1.0e-12, scale * scale * scale * 1.0e-12)
    source_face_projections = _tag_axis_projections(
        points, triangles, phys, tag=source_tag, axis_idx=2
    )
    aperture_face_projections = _tag_axis_projections(
        points, triangles, phys, tag=aperture_tag, axis_idx=2
    )
    source_projection = float(np.sum(source_face_projections))
    aperture_projection = float(np.sum(aperture_face_projections))
    if source_projection <= area_eps:
        raise MesherError(
            "infinite-baffle primary source must have positive driven area with +Z normals "
            f"(projection {source_projection:.6g})"
        )
    if np.any(source_face_projections < -area_eps):
        raise MesherError(
            "infinite-baffle primary source contains triangle normals opposite +Z"
        )
    if aperture_projection >= -area_eps:
        raise MesherError(
            "infinite-baffle mouth aperture must have nonzero area with -Z normals "
            f"(projection {aperture_projection:.6g})"
        )
    if np.any(aperture_face_projections > area_eps):
        raise MesherError(
            "infinite-baffle mouth aperture contains triangle normals opposite -Z"
        )
    if float(getattr(report, "signed_volume")) >= -volume_eps:
        raise MesherError(
            "infinite-baffle interior domain must have negative signed volume "
            f"(got {float(getattr(report, 'signed_volume')):.6g})"
        )
    if int(getattr(report, "nonmanifold_edges")):
        raise MesherError(
            "infinite-baffle mesh is nonmanifold: "
            f"{int(getattr(report, 'nonmanifold_edges'))} edge(s)"
        )
    if int(getattr(report, "inconsistent_edges")):
        raise MesherError(
            "infinite-baffle mesh has inconsistent winding on "
            f"{int(getattr(report, 'inconsistent_edges'))} shared edge(s)"
        )

    aperture_nodes = np.unique(triangles[phys == aperture_tag])
    plane_tol = max(float(symmetry_snap_tol_mm), 1.0e-7)
    if np.any(np.abs(points[aperture_nodes, 2]) > plane_tol):
        span = float(np.ptp(points[aperture_nodes, 2]))
        raise MesherError(
            "infinite-baffle mouth aperture must be coplanar at z=0 "
            f"(z span {span:.6g} mm)"
        )

    table = build_edge_table(triangles, drop_degenerate=True)
    pair_edges = np.flatnonzero(table.count == 2)
    by_edge = np.argsort(table.edge, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(table.count)))[:-1]
    tag_first = phys[table.tri[by_edge[offsets[pair_edges]]]]
    tag_second = phys[table.tri[by_edge[offsets[pair_edges] + 1]]]
    is_rim = ((tag_first == wall_tag) & (tag_second == aperture_tag)) | (
        (tag_first == aperture_tag) & (tag_second == wall_tag)
    )
    rim_ids = pair_edges[is_rim]
    shared_rim = list(zip(table.lo[rim_ids].tolist(), table.hi[rim_ids].tolist()))
    if not shared_rim:
        raise MesherError(
            "infinite-baffle wall and mouth aperture do not share a welded rim"
        )
    for edge in shared_rim:
        if np.any(np.abs(points[list(edge), 2]) > plane_tol):
            raise MesherError("infinite-baffle wall/aperture rim is not on z=0")

    boundary_ids = np.flatnonzero(table.count == 1)
    boundary_edges = list(
        zip(table.lo[boundary_ids].tolist(), table.hi[boundary_ids].tolist())
    )
    axes = tuple(str(axis) for axis in symmetry_snap_axes)
    if not axes:
        if boundary_edges:
            raise MesherError(
                "full infinite-baffle domain must be watertight; "
                f"found {len(boundary_edges)} open edge(s)"
            )
        return

    axis_idx = {"x": 0, "y": 1, "z": 2}
    unknown = [axis for axis in axes if axis not in axis_idx]
    if unknown:
        raise MesherError(f"unknown infinite-baffle cut plane axis/axes: {unknown!r}")
    for edge in boundary_edges:
        if not any(
            abs(float(points[edge[0], axis_idx[axis]])) <= plane_tol
            and abs(float(points[edge[1], axis_idx[axis]])) <= plane_tol
            for axis in axes
        ):
            raise MesherError(
                "reduced infinite-baffle domain has an open edge away from its declared cut planes"
            )


_WELD_FORWARD_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
)


def _weld_near_duplicate_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
    *,
    tol_mm: float = 5.0e-3,
) -> np.ndarray:
    """Remap triangle indices so vertices closer than ``tol_mm`` coincide.

    Uses a spatial hash with cells of the weld tolerance; clusters merge to
    the lowest vertex index. Orphaned duplicate vertices become unused and
    are dropped by the final compaction. Returns the remapped triangles.
    """

    if len(points) == 0 or len(triangles) == 0:
        return triangles
    cells = np.floor(points / tol_mm).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, cells)):
        buckets.setdefault(key, []).append(index)

    parent = np.arange(len(points))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    tol_sq = tol_mm * tol_mm

    def merge_if_close(i: int, j: int) -> None:
        delta = points[j] - points[i]
        if float(delta @ delta) > tol_sq:
            return
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for key, indices in buckets.items():
        for position, i in enumerate(indices):
            for j in indices[position + 1 :]:
                merge_if_close(i, j)
        # Visit each pair of neighboring cells once. The old all-neighbor
        # traversal visited every pair from both cells and discarded half of
        # the candidates by vertex index.
        for dx, dy, dz in _WELD_FORWARD_NEIGHBOR_OFFSETS:
            others = buckets.get((key[0] + dx, key[1] + dy, key[2] + dz))
            if others is None:
                continue
            for i in indices:
                for j in others:
                    merge_if_close(i, j)

    roots = np.fromiter(
        (find(i) for i in range(len(points))), dtype=np.int64, count=len(points)
    )
    if np.array_equal(roots, np.arange(len(points))):
        return triangles
    return roots[triangles]


def _compact_unused_vertices(
    points: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(triangles)
    if len(used) == len(points):
        return points, triangles
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return points[used], remap[triangles]


_SYMMETRY_SLIVER_MAX_AREA_MM2 = 0.25

# Interface surfaces (ATH ``Mesh.SubdomainSlices``) are internal partitions
# authored on their own points, so their base rim is free by construction. They
# are the one tag the closure contract cannot judge; see the note in
# ``_validate_closed_shell_contract``.
#
# Do not "fix" this by making the interface skirt share the wall's slice lines.
# That was tried and measured: it produces 56 NON-MANIFOLD edges, because an
# interior slice line then carries three faces (wall above, wall below, skirt).
# That is not a bug in the interface -- a real ABEC subdomain interface *is*
# non-manifold by definition -- it is a mismatch with this validator, which
# assumes a manifold single-domain surface and has no way to express a
# three-face edge. So a conforming interface cannot be represented here today,
# and the rim-sharing change was reverted. Exempting the tag is the honest
# option until the contract itself learns about subdomains.
_CONTRACT_EXEMPT_TAGS = frozenset({int(PhysicalGroup.INTERFACE)})


def _validate_closed_shell_contract(
    points: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
    *,
    symmetry_snap_axes: tuple[str, ...],
    symmetry_snap_tol_mm: float,
    is_open_shell: bool,
) -> None:
    """Refuse to emit a closed-shell mesh with a hole in it.

    A reduced domain is legitimately open along its cut planes -- the solver's
    mirror closes those. A free edge anywhere else is a hole, and a hole in an
    enclosure is a box that radiates out of its own back: the solve returns a
    confident, wrong, near-omnidirectional answer with nothing to show for it.

    Nothing used to catch that. ``validate_orientation`` runs with
    ``require_watertight=False`` for every mode, because bare open shells and
    reduced domains both legitimately have free edges, so the one check that
    could have caught it was disabled for everyone. Splitting "open along a
    declared cut plane" from "open anywhere else" is what makes the check
    affordable again.

    Bare open shells are not exempt, only held to a weaker rule: their mouth rim
    is the model, so they may keep exactly one free-edge loop. A second loop is
    a torn throat seam, which is how an independently authored source cap fails.
    """

    if len(triangles) == 0:
        return
    axis_index = {"x": 0, "y": 1, "z": 2}
    axes = tuple(
        axis_index[axis] for axis in symmetry_snap_axes if axis in axis_index
    )
    # Snapped cut-plane vertices are set exactly to zero; allow the snap
    # tolerance anyway so a caller that widens it cannot turn its own seam into
    # a reported hole.
    tolerance = max(float(symmetry_snap_tol_mm), 1.0e-9)

    table = build_edge_table(triangles)
    free = table.count == 1
    if axes:
        on_plane = np.zeros(table.n_edges, dtype=bool)
        for axis in axes:
            on_plane |= (np.abs(points[table.lo, axis]) <= tolerance) & (
                np.abs(points[table.hi, axis]) <= tolerance
            )
        free &= ~on_plane
    owner_tri = table.tri[table.first_use]
    free &= ~np.isin(phys[owner_tri], sorted(_CONTRACT_EXEMPT_TAGS))
    hole_ids = np.flatnonzero(free)
    holes = list(zip(table.lo[hole_ids].tolist(), table.hi[hole_ids].tolist()))
    owner = dict(zip(holes, owner_tri[hole_ids].tolist()))
    if not holes:
        return

    # A bare shell keeps its mouth rim. Group the free edges into connected
    # loops rather than measuring where they sit: a mouth rim need not be
    # planar, so "one boundary" is the honest test and an axial-extreme test
    # would reject legitimate non-planar mouths.
    allowed_boundaries = 1 if is_open_shell else 0
    boundaries = _count_edge_components(holes)
    if boundaries <= allowed_boundaries:
        return

    tags = sorted({int(phys[owner[edge]]) for edge in holes})
    sample = points[holes[0][0]]
    planes = ", ".join(f"{axis}=0" for axis in symmetry_snap_axes) or "none"
    shell = "open-shell" if is_open_shell else "closed-shell"
    detail = (
        f"{boundaries} separate free-edge loops where an open shell may have "
        "only its mouth rim"
        if is_open_shell
        else f"{len(holes)} free edges"
    )
    raise MesherError(
        f"{shell} mesh has {detail} away from its symmetry cut planes "
        f"({planes}) on surface tags {tags}; the model is not closed. "
        f"First gap near ({sample[0]:.3f}, {sample[1]:.3f}, {sample[2]:.3f}) mm."
    )


def _count_edge_components(edges: list[tuple[int, int]]) -> int:
    """Count connected components of an undirected edge set (union-find)."""

    parent: dict[int, int] = {}

    def find(node: int) -> int:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for start, end in edges:
        a, b = find(start), find(end)
        if a != b:
            parent[a] = b
    return len({find(node) for node in parent})


def _remove_symmetry_plane_slivers(
    points: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
    symmetry_snap_axes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Drop snap-flattened sliver triangles lying entirely in a symmetry plane.

    Snapping near-plane vertices exactly onto the symmetry planes can flatten
    seam slivers into the plane. Solvers reject in-plane triangles (the plane
    is an image plane, not a physical boundary), so remove the negligible-area
    artifacts and fail loudly if a substantial triangle ends up in-plane,
    which would indicate a real geometry defect.
    """

    if not symmetry_snap_axes or len(triangles) == 0:
        return triangles, phys
    axis_index = {"x": 0, "y": 1, "z": 2}
    keep = np.ones(len(triangles), dtype=bool)
    corners = points[triangles]
    for axis in symmetry_snap_axes:
        idx = axis_index.get(axis)
        if idx is None:
            continue
        in_plane = np.all(np.abs(corners[:, :, idx]) <= 1.0e-9, axis=1)
        if not np.any(in_plane):
            continue
        areas = 0.5 * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )
        large = in_plane & (areas > _SYMMETRY_SLIVER_MAX_AREA_MM2)
        if np.any(large):
            raise MesherError(
                f"{int(np.sum(large))} non-sliver triangle(s) lie entirely in the "
                f"{axis}=0 symmetry plane; the geometry is defective"
            )
        keep &= ~in_plane
    if np.all(keep):
        return triangles, phys
    return triangles[keep], phys[keep]


def _edge_stats_by_tag(
    points_mm: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Per physical tag edge-length statistics in millimetres."""

    stats: dict[int, dict[str, float]] = {}
    for tag in sorted({int(value) for value in phys.tolist()}):
        tris = triangles[phys == tag]
        if len(tris) == 0:
            continue
        edges = np.concatenate(
            [
                np.linalg.norm(points_mm[tris[:, 0]] - points_mm[tris[:, 1]], axis=1),
                np.linalg.norm(points_mm[tris[:, 1]] - points_mm[tris[:, 2]], axis=1),
                np.linalg.norm(points_mm[tris[:, 2]] - points_mm[tris[:, 0]], axis=1),
            ]
        )
        stats[int(tag)] = {
            "min_edge_mm": float(np.min(edges)),
            "p05_edge_mm": float(np.percentile(edges, 5)),
            "median_edge_mm": float(np.median(edges)),
            "p95_edge_mm": float(np.percentile(edges, 95)),
            "max_edge_mm": float(np.max(edges)),
        }
    return stats


def _snap_symmetry_planes(
    points: np.ndarray, axes: tuple[str, ...], tolerance: float
) -> None:
    if not axes or tolerance <= 0.0:
        return
    axis_indices = {"x": 0, "y": 1, "z": 2}
    for axis in axes:
        idx = axis_indices.get(str(axis))
        if idx is None:
            continue
        mask = np.abs(points[:, idx]) <= float(tolerance)
        points[mask, idx] = 0.0


def _triangles_and_physical_tags(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray]:
    triangles: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    physical_data = mesh.cell_data.get("gmsh:physical") or mesh.cell_data.get(
        "physical"
    )
    for idx, cell_block in enumerate(mesh.cells):
        if cell_block.type not in ("triangle", "triangle3"):
            continue
        triangles.append(np.asarray(cell_block.data, dtype=np.int64))
        if physical_data is not None and idx < len(physical_data):
            tags.append(np.asarray(physical_data[idx], dtype=np.int32))
        else:
            tags.append(
                np.full(
                    len(cell_block.data), int(PhysicalGroup.RIGID_WALL), dtype=np.int32
                )
            )
    if not triangles:
        return np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)
    return np.vstack(triangles), np.concatenate(tags)


def _physical_names(mesh: meshio.Mesh) -> dict[int, str]:
    _, used = _triangles_and_physical_tags(mesh)
    used_tags = {int(tag) for tag in used.tolist()}
    out: dict[int, str] = {}
    for name, raw in mesh.field_data.items():
        if len(raw) >= 2 and int(raw[1]) == 2 and int(raw[0]) in used_tags:
            out[int(raw[0])] = str(name)
    return out


def _looks_like_metres(points: np.ndarray) -> bool:
    span = np.max(points, axis=0) - np.min(points, axis=0)
    return bool(np.max(span) < 10.0)


def _validate_physical_tags(tags: set[int]) -> None:
    if int(PhysicalGroup.PRIMARY_SOURCE) not in tags:
        raise MesherError("mesh has no primary source physical group (tag 2)")
    if int(PhysicalGroup.RIGID_WALL) not in tags:
        raise MesherError("mesh has no rigid wall physical group (tag 1)")
