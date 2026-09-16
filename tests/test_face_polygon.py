from pathlib import Path
from math import isclose

from auto_mating.geometry import (
    describe_face,
    make_planar_frame,
)
from auto_mating.geometry.boundaries import (
    project_face_boundary,
)
from auto_mating.io import read_step
from auto_mating.topology import iter_faces


EXAMPLES_DIR = Path("examples/mating_test")


def get_face(path, face_index):
    shape = read_step(path)
    faces = list(iter_faces(shape))
    face = faces[face_index - 1]

    return face


def test_part_a_top_face_projects_to_100_by_50_rectangle():
    face = get_face(
        EXAMPLES_DIR / "part_A.step",
        5,
    )

    descriptor = describe_face(
        face,
        index=5,
    )

    frame = make_planar_frame(
        origin=descriptor.centroid,
        normal=descriptor.normal,
    )

    polygon = project_face_boundary(
        face,
        frame,
    )

    assert len(polygon.points) == 4

    xs = [point.x for point in polygon.points]
    ys = [point.y for point in polygon.points]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    assert isclose(
        width,
        50.0,
        abs_tol=1.0e-9,
    )

    assert isclose(
        height,
        100.0,
        abs_tol=1.0e-9,
    )


def test_part_b_bottom_face_projects_to_60_by_40_rectangle():
    face = get_face(
        EXAMPLES_DIR / "part_B.step",
        6,
    )

    descriptor = describe_face(
        face,
        index=6,
    )

    frame = make_planar_frame(
        origin=descriptor.centroid,
        normal=descriptor.normal,
    )

    polygon = project_face_boundary(
        face,
        frame,
    )

    assert len(polygon.points) == 4

    xs = [point.x for point in polygon.points]
    ys = [point.y for point in polygon.points]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    assert isclose(
        width,
        40.0,
        abs_tol=1.0e-9,
    )

    assert isclose(
        height,
        60.0,
        abs_tol=1.0e-9,
    )