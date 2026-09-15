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