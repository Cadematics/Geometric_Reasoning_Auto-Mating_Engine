from pathlib import Path

import pytest

from auto_mating.geometry import describe_face
from auto_mating.io import read_step
from auto_mating.topology import iter_faces


EXAMPLE_STEP = (
    Path(__file__).parent.parent
    / "examples"
    / "simple_plate"
    / "plate.step"
)


def test_simple_plate_face_descriptors():
    shape = read_step(EXAMPLE_STEP)
    faces = list(iter_faces(shape))

    descriptors = [
        describe_face(face, index)
        for index, face in enumerate(faces)
    ]

    assert len(descriptors) == 6

    assert all(
        descriptor.surface_type.value == "plane"
        for descriptor in descriptors
    )

    areas = sorted(
        descriptor.area
        for descriptor in descriptors
    )

    assert areas == pytest.approx([
        500.0,
        500.0,
        1000.0,
        1000.0,
        5000.0,
        5000.0,
    ])


def test_simple_plate_face_centroids():
    shape = read_step(EXAMPLE_STEP)
    faces = list(iter_faces(shape))

    descriptors = [
        describe_face(face, index)
        for index, face in enumerate(faces)
    ]

    centroids = [
        (
            descriptor.centroid.x,
            descriptor.centroid.y,
            descriptor.centroid.z,
        )
        for descriptor in descriptors
    ]

    expected = [
        (0.0, 25.0, 5.0),
        (100.0, 25.0, 5.0),
        (50.0, 0.0, 5.0),
        (50.0, 50.0, 5.0),
        (50.0, 25.0, 0.0),
        (50.0, 25.0, 10.0),
    ]

    for actual, expected_point in zip(centroids, expected):
        assert actual == pytest.approx(expected_point)






def test_simple_plate_face_normals():
    shape = read_step(EXAMPLE_STEP)
    faces = list(iter_faces(shape))

    descriptors = [
        describe_face(face, index)
        for index, face in enumerate(faces)
    ]

    normals = [
        (
            descriptor.normal.x,
            descriptor.normal.y,
            descriptor.normal.z,
        )
        for descriptor in descriptors
    ]

    expected = [
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 0.0, 1.0),
    ]

    for actual, expected_normal in zip(normals, expected):
        assert actual == pytest.approx(expected_normal)