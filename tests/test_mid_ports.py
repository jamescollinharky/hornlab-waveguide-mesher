"""Opt-in L1 mid_ports patches on native OSSE(+SF) walls.

Absent mid_ports = byte-identical golden OSSE mesh.
Ports-on Solana-like OSSE+SF has tags {1, 2, 10} and tag-10 area within ~1% of recipe.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import meshio
import pytest

from hornlab_mesher import build_from_config
from hornlab_mesher.mid_ports import parse_mid_ports
from hornlab_mesher.tags import PhysicalGroup


def _solana_like_osse_sf(*, mid_ports: dict | None = None) -> dict:
    """Ath-ish OSSE + SF gcurve + morph, quarter, coarse but solvable."""
    cfg: dict = {
        "formula": "OSSE",
        "mode": "freestanding",
        "profile": {
            "r0_mm": 12.7,
            "a0_deg": 0.0,
            "L_mm": 140.0,
            "a_deg": 45.0,
            "n": 4.95,
            "s": 1.17,
            "q": 0.80,
            "k": 2.33,
        },
        "cross_section": {
            "exponent": 2.0,
            "aspect_ratio": 1.0,
        },
        "gcurve": {
            "gcurve_type": 2,
            "gcurve_width_mm": 65.0,
            "gcurve_aspect_ratio": 1.0,
            "gcurve_dist": 0.28,
            "gcurve_rot_deg": 0.0,
            "gcurve_sf_a": 1.38,
            "gcurve_sf_b": 1.44,
            "gcurve_sf_m1": 8.0,
            "gcurve_sf_m2": 8.0,
            "gcurve_sf_n1": 2.15,
            "gcurve_sf_n2": 5.25,
            "gcurve_sf_n3": 3.02,
        },
        "morph": {
            "morph_target": 1,
            "morph_width_mm": 353.0,
            "morph_height_mm": 353.0,
            "morph_corner_mm": 14.0,
            "morph_exponent": 4.0,
            "morph_rate": 3.7,
            "morph_fixed": 0.35,
            "morph_allow_shrinkage": 1,
        },
        "mesh": {
            "quadrants": "1",
            "throat_res_mm": 3.0,
            "mouth_res_mm": 10.0,
            "rear_res_mm": 10.0,
            "wall_thickness_mm": 5.0,
            "scale_to_metres": True,
            "allow_large_mesh": True,
        },
    }
    if mid_ports is not None:
        cfg["mid_ports"] = mid_ports
    return cfg


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tag_areas_mm2(path: Path) -> dict[int, float]:
    import numpy as np

    mesh = meshio.read(path)
    pts = mesh.points * 1000.0  # metres -> mm
    tris = None
    phys = None
    for i, block in enumerate(mesh.cells):
        if block.type in ("triangle", "triangle3"):
            tris = block.data
            phys = mesh.cell_data["gmsh:physical"][i]
            break
    assert tris is not None and phys is not None
    out: dict[int, float] = {}
    for tag in sorted({int(t) for t in phys}):
        mask = phys == tag
        p0 = pts[tris[mask, 0]]
        p1 = pts[tris[mask, 1]]
        p2 = pts[tris[mask, 2]]
        out[int(tag)] = float(
            0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum()
        )
    return out


def test_parse_mid_ports_absent():
    assert parse_mid_ports(None) is None
    assert parse_mid_ports({}) is None


def test_parse_mid_ports_rejects_bad_walls():
    with pytest.raises(ValueError, match="walls"):
        parse_mid_ports(
            {
                "walls": "xyz",
                "z_frac": 0.35,
                "across_frac": 0.4,
                "port_along_mm": 25.0,
            }
        )


def test_mid_ports_absent_byte_identical(tmp_path):
    a = build_from_config(_solana_like_osse_sf(), tmp_path / "a.msh")
    b = build_from_config(
        _solana_like_osse_sf(mid_ports=None), tmp_path / "b.msh"
    )
    # Explicit empty block must also leave the mesh alone (parse -> None).
    c = build_from_config(_solana_like_osse_sf(mid_ports={}), tmp_path / "c.msh")
    assert _file_sha(Path(a.mesh_path)) == _file_sha(Path(b.mesh_path))
    assert _file_sha(Path(a.mesh_path)) == _file_sha(Path(c.mesh_path))
    assert set(a.physical_groups) == {1, 2}


def test_mid_ports_on_solana_like_tags_and_area(tmp_path):
    recipe = {
        "walls": "tblr",
        "z_frac": 0.35,
        "across_frac": 0.4,
        "port_along_mm": 25.0,
        "slide_mm": 0.0,
    }
    result = build_from_config(
        _solana_like_osse_sf(mid_ports=recipe), tmp_path / "ports.msh"
    )
    tags = set(result.physical_groups)
    assert tags == {1, 2, int(PhysicalGroup.MID_PORT_EXIT_LEFT)}
    report = (result.metadata or {}).get("midPorts")
    assert report is not None
    assert report["n_port_triangles"] > 0
    assert len(report["patches"]) == 2  # quarter tblr -> half-top + half-right
    expected = float(report["expected_area_mm2"])
    got = float(report["port_area_mm2"])
    # Centroid selection on a curved wall is approximate; demand ~1% of recipe.
    assert expected > 0.0
    assert abs(got - expected) / expected <= 0.01, (
        f"tag-10 area {got:.3f} mm^2 vs recipe {expected:.3f} mm^2 "
        f"(rel err {abs(got - expected) / expected:.3%})"
    )
    areas = _tag_areas_mm2(Path(result.mesh_path))
    assert areas[10] > 0.0
    assert abs(areas[10] - got) / max(got, 1.0) < 0.02
