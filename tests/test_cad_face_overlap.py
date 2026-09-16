from pathlib import Path

from auto_mating.geometry import (
    describe_face,
    make_planar_frame,
    project_face_boundary,
)
from auto_mating.geometry.polygon import intersect_convex_polygons
from auto_mating.io import read_step
from auto_mating.topology import TopologyIndex


EXAMPLES = Path("examples/mating_test")


def test_actual_cad_face_overlap():
    part_a = read_step(EXAMPLES / "part_A.step")
    part_b = read_step(EXAMPLES / "part_B.step")

    topology_a = TopologyIndex.build(part_a)
    topology_b = TopologyIndex.build(part_b)

    # Retrieve the actual OCCT faces.
    occt_face_a = topology_a.face(5)
    occt_face_b = topology_b.face(6)

    # Build geometric descriptors.
    face_a = describe_face(occt_face_a, 5)
    face_b = describe_face(occt_face_b, 6)

    assert face_a.plane is not None
    assert face_b.plane is not None

    # Use Face A's plane and oriented normal as the
    # common coordinate system for both faces.
    frame = make_planar_frame(
        origin=face_a.plane.origin,
        normal=face_a.normal,
    )

    polygon_a = project_face_boundary(
        occt_face_a,
        frame,
    )

    polygon_b = project_face_boundary(
        occt_face_b,
        frame,
    )

    intersection = intersect_convex_polygons(
        polygon_a,
        polygon_b,
    )

    assert abs(intersection.area() - 2400.0) < 1.0e-6