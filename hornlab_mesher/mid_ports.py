"""Opt-in L1 mid-port patches on native OSSE / freestanding horn walls.

Absent ``mid_ports`` in the build config = no call into this module (byte-identical).

When present, wall triangles whose centroids fall in rectangular patches on the
inner wall are re-tagged to ``PhysicalGroup.MID_PORT_EXIT_LEFT`` (10). This is
the flush-exit L1 contract used by hornlab-search MEH (tag 10 driven sources),
not mid chambers / port interiors (tags 8–9).

Port placement mirrors the MEH CAD recipe semantics (z_frac, across_frac,
port_along_mm, slide_mm, walls tb|lr|tblr) on the curved OSSE wall by using
axial z and wall-sector (|y|>=|x| for top, |x|>=|y| for right) in the quarter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .tags import PhysicalGroup


WALL_TAG = int(PhysicalGroup.RIGID_WALL)
PORT_TAG = int(PhysicalGroup.MID_PORT_EXIT_LEFT)

_VALID_WALLS = ("tb", "lr", "tblr")


@dataclass(frozen=True)
class MidPortsRecipe:
    walls: str
    z_frac: float
    across_frac: float
    port_along_mm: float
    slide_mm: float

    @property
    def walls_list(self) -> tuple[str, ...]:
        if self.walls == "tb":
            return ("tb",)
        if self.walls == "lr":
            return ("lr",)
        return ("tb", "lr")


def parse_mid_ports(raw: Mapping[str, Any] | None) -> MidPortsRecipe | None:
    """Return a recipe or None when the block is absent / empty."""
    if not raw:
        return None
    walls = str(raw.get("walls", "tblr")).strip().lower()
    if walls not in _VALID_WALLS:
        raise ValueError(
            f"mid_ports.walls must be one of {_VALID_WALLS}, got {walls!r}"
        )
    z_frac = float(raw["z_frac"])
    across_frac = float(raw["across_frac"])
    port_along_mm = float(raw["port_along_mm"])
    slide_mm = float(raw.get("slide_mm", 0.0))
    if not 0.0 < z_frac < 1.0:
        raise ValueError(f"mid_ports.z_frac must be in (0, 1), got {z_frac}")
    if not 0.0 < across_frac <= 1.0:
        raise ValueError(f"mid_ports.across_frac must be in (0, 1], got {across_frac}")
    if not port_along_mm > 0.0:
        raise ValueError(f"mid_ports.port_along_mm must be > 0, got {port_along_mm}")
    return MidPortsRecipe(
        walls=walls,
        z_frac=z_frac,
        across_frac=across_frac,
        port_along_mm=port_along_mm,
        slide_mm=slide_mm,
    )


def apply_mid_ports(
    points_mm: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
    recipe: MidPortsRecipe,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Re-tag wall triangles inside port patches. Points must be millimetres.

    Selected wall triangles are 4-way midpoint-subdivided once (refine near
    patches) before retagging to tag 10.

    Returns (points_mm, triangles, phys, report).
    """
    points_mm = np.asarray(points_mm, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    phys = np.asarray(phys, dtype=np.int32).copy()
    wall = phys == WALL_TAG
    if not np.any(wall):
        raise ValueError("mid_ports: mesh has no rigid-wall triangles to patch")

    cents = _centroids(points_mm, triangles)
    # Freestanding shells tag both inner (acoustic) and outer walls as 1.
    # Ports sit on the inner wall: normals into the air (toward the axis).
    wall = wall & _inner_wall_mask(points_mm, triangles, cents, wall)
    if not np.any(wall):
        raise ValueError("mid_ports: no inner-wall triangles (toward-axis normals)")

    z = points_mm[:, 2]
    z_min = float(np.min(z[triangles[wall].ravel()]))
    z_max = float(np.max(z[triangles[wall].ravel()]))
    L = max(z_max - z_min, 1.0)
    z_port = z_min + recipe.z_frac * L + recipe.slide_mm
    z_port = float(np.clip(z_port, z_min + 5.0, z_max - 10.0))
    half_along = 0.5 * recipe.port_along_mm

    # Horn radius at z_port from wall centroids near that station (for across).
    near = wall & (np.abs(cents[:, 2] - z_port) < max(half_along, 8.0))
    if np.any(near):
        r_ref = float(np.median(np.hypot(cents[near, 0], cents[near, 1])))
    else:
        r_ref = float(np.median(np.hypot(cents[wall, 0], cents[wall, 1])))
    across = max(recipe.across_frac * max(2.0 * r_ref - 6.0, 1.0), 1.0)
    half_across = 0.5 * across

    # Pre-refine a padded z-band on the inner wall so the final centroid test
    # does not under-fill the recipe rectangle on a coarse throat mesh.
    pad = max(half_along, half_across) + 6.0
    band = wall & (np.abs(cents[:, 2] - z_port) <= pad)
    points_mm, triangles, phys = _subdivide_once(
        points_mm, triangles, phys, band
    )
    # Second pass on the already-refined band neighbourhood.
    cents = _centroids(points_mm, triangles)
    wall = (phys == WALL_TAG) & _inner_wall_mask(
        points_mm, triangles, cents, phys == WALL_TAG
    )
    band = wall & (np.abs(cents[:, 2] - z_port) <= pad)
    points_mm, triangles, phys = _subdivide_once(
        points_mm, triangles, phys, band
    )

    cents = _centroids(points_mm, triangles)
    wall = (phys == WALL_TAG) & _inner_wall_mask(
        points_mm, triangles, cents, phys == WALL_TAG
    )
    expected = float(
        len(recipe.walls_list) * recipe.port_along_mm * half_across
    )
    # Discrete triangle sets under/over-fill a continuous rectangle. Search a
    # small grow of the (z, across) window so total tag-10 area lands within
    # ~1% of the recipe (plan gate).
    edge = _median_edge_mm(points_mm, triangles[wall]) if np.any(wall) else 4.0
    # Negative grow shrinks the window when the discrete set already overshoots.
    grow_lo, grow_hi = -0.75 * edge, max(3.0 * edge, 12.0)
    selected = np.zeros(len(triangles), dtype=bool)
    patches: list[dict[str, Any]] = []
    best_rel = 1.0e9
    best_pack: tuple[np.ndarray, list[dict[str, Any]], float] | None = None
    for _ in range(12):
        grow = 0.5 * (grow_lo + grow_hi)
        selected, patches, area = _select_ports(
            points_mm,
            triangles,
            wall,
            cents,
            z_port,
            half_along,
            half_across,
            recipe.walls_list,
            grow,
        )
        if area <= 0.0:
            grow_lo = grow
            continue
        rel = abs(area - expected) / expected
        if rel < best_rel:
            best_rel = rel
            best_pack = (selected.copy(), list(patches), area)
        if rel <= 0.01:
            break
        if area < expected:
            grow_lo = grow
        else:
            grow_hi = grow
    if best_pack is None or not np.any(best_pack[0]):
        raise ValueError(
            f"mid_ports: no wall triangles at z_port={z_port:.3f} mm "
            f"(across={across:.3f}); widen across_frac or port_along_mm / check z_frac"
        )
    selected, patches, area = best_pack
    report: dict[str, Any] = {
        "z_port_mm": z_port,
        "across_mm": across,
        "port_along_mm": recipe.port_along_mm,
        "walls": recipe.walls,
        "r_ref_mm": r_ref,
        "expected_area_mm2": expected,
        "patches": patches,
        "n_port_triangles": int(np.count_nonzero(selected)),
        "port_area_mm2": area,
        "area_rel_err": float(abs(area - expected) / expected),
    }

    phys[selected] = PORT_TAG
    if not np.any(phys == WALL_TAG):
        raise ValueError("mid_ports: retag consumed every wall triangle; refuse")
    return points_mm, triangles, phys, report


def _select_ports(
    points_mm: np.ndarray,
    triangles: np.ndarray,
    wall: np.ndarray,
    cents: np.ndarray,
    z_port: float,
    half_along: float,
    half_across: float,
    walls_list: tuple[str, ...],
    grow: float,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    selected = np.zeros(len(triangles), dtype=bool)
    patches: list[dict[str, Any]] = []
    for wall_kind in walls_list:
        mask = _port_mask(
            points_mm,
            triangles,
            wall,
            cents,
            z_port,
            half_along + grow,
            half_across + grow,
            wall_kind,
        )
        n_sel = int(np.count_nonzero(mask))
        if n_sel == 0:
            return selected, patches, 0.0
        selected |= mask
        patches.append(
            {
                "wall": wall_kind,
                "n_triangles": n_sel,
                "area_mm2": _tri_area_mm2(points_mm, triangles[mask]),
            }
        )
    return selected, patches, _tri_area_mm2(points_mm, triangles[selected])


def _centroids(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    return (
        points[triangles[:, 0]] + points[triangles[:, 1]] + points[triangles[:, 2]]
    ) / 3.0


def _inner_wall_mask(
    points: np.ndarray,
    triangles: np.ndarray,
    cents: np.ndarray,
    wall: np.ndarray,
) -> np.ndarray:
    """True for wall tris whose normals point toward the axis (acoustic bore)."""
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    toward_axis = (n[:, 0] * cents[:, 0] + n[:, 1] * cents[:, 1]) < 0.0
    return wall & toward_axis


def _port_mask(
    points: np.ndarray,
    triangles: np.ndarray,
    wall: np.ndarray,
    cents: np.ndarray,
    z_port: float,
    half_along: float,
    half_across: float,
    wall_kind: str,
) -> np.ndarray:
    """Rectangle on the wall in (z, across), pinned to the local wall radius.

    A triangle is kept when its centroid OR any vertex lies in the rectangle
    (reduces centroid under-fill on coarse meshes) and |r - r_local| is small
    so a superformula knuckle cannot stretch the patch along a fold.
    """
    in_rect = _points_in_port_rect(
        cents, z_port, half_along, half_across, wall_kind
    )
    # Vertex inclusion (any of the three).
    v_in = np.zeros(len(triangles), dtype=bool)
    for k in range(3):
        v_in |= _points_in_port_rect(
            points[triangles[:, k]],
            z_port,
            half_along,
            half_across,
            wall_kind,
        )
    candidates = wall & (in_rect | v_in)
    if not np.any(candidates):
        return candidates
    r = np.hypot(cents[:, 0], cents[:, 1])
    r_tgt = float(np.median(r[candidates]))
    r_tol = max(8.0, 0.08 * r_tgt)
    return candidates & (np.abs(r - r_tgt) <= r_tol)


def _points_in_port_rect(
    pts: np.ndarray,
    z_port: float,
    half_along: float,
    half_across: float,
    wall_kind: str,
) -> np.ndarray:
    z_ok = np.abs(pts[:, 2] - z_port) <= half_along
    if wall_kind == "tb":
        sector = (np.abs(pts[:, 1]) >= np.abs(pts[:, 0])) & (pts[:, 1] > 0.0)
        across_ok = pts[:, 0] <= half_across + 1.0e-6
    else:
        sector = (np.abs(pts[:, 0]) >= np.abs(pts[:, 1])) & (pts[:, 0] > 0.0)
        across_ok = pts[:, 1] <= half_across + 1.0e-6
    return z_ok & sector & across_ok


def _subdivide_once(
    points: np.ndarray,
    triangles: np.ndarray,
    phys: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4-way midpoint split of selected triangles (refine near port patches)."""
    if not np.any(selected):
        return points, triangles, phys
    keep = triangles[~selected]
    keep_phys = phys[~selected]
    sel = triangles[selected]
    sel_phys = phys[selected]
    # Midpoint index cache keyed by undirected edge.
    edge_mid: dict[tuple[int, int], int] = {}
    new_pts: list[np.ndarray] = []

    def mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        hit = edge_mid.get(key)
        if hit is not None:
            return hit
        idx = len(points) + len(new_pts)
        new_pts.append(0.5 * (points[a] + points[b]))
        edge_mid[key] = idx
        return idx

    children: list[np.ndarray] = []
    child_phys: list[int] = []
    for tri, tag in zip(sel, sel_phys):
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        m01, m12, m20 = mid(i0, i1), mid(i1, i2), mid(i2, i0)
        for child in (
            (i0, m01, m20),
            (i1, m12, m01),
            (i2, m20, m12),
            (m01, m12, m20),
        ):
            children.append(np.array(child, dtype=np.int64))
            child_phys.append(int(tag))

    if new_pts:
        points = np.vstack([points, np.asarray(new_pts, dtype=np.float64)])
    triangles = np.vstack([keep, np.asarray(children, dtype=np.int64)])
    phys = np.concatenate(
        [keep_phys, np.asarray(child_phys, dtype=np.int32)]
    )
    return points, triangles, phys


def _tri_area_mm2(points: np.ndarray, tris: np.ndarray) -> float:
    if len(tris) == 0:
        return 0.0
    p0 = points[tris[:, 0]]
    p1 = points[tris[:, 1]]
    p2 = points[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum())


def _median_edge_mm(points: np.ndarray, tris: np.ndarray) -> float:
    if len(tris) == 0:
        return 4.0
    sample = tris[: min(len(tris), 400)]
    e01 = np.linalg.norm(points[sample[:, 1]] - points[sample[:, 0]], axis=1)
    e12 = np.linalg.norm(points[sample[:, 2]] - points[sample[:, 1]], axis=1)
    e20 = np.linalg.norm(points[sample[:, 0]] - points[sample[:, 2]], axis=1)
    return float(np.median(np.concatenate([e01, e12, e20])))
