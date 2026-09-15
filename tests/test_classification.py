from pathlib import Path

from auto_mating.geometry import SurfaceType, classify_face
from auto_mating.io import read_step
from auto_mating.topology import iter_faces


EXAMPLE_STEP = (
    Path(__file__).parent.parent
    / "examples"
    / "simple_plate"
    / "plate.step"
)


def test_simple_plate_faces_are_planes():
    shape = read_step(EXAMPLE_STEP)

    faces = list(iter_faces(shape))

    assert len(faces) == 6

    surface_types = [
        classify_face(face)
        for face in faces
    ]

    assert all(
        surface_type == SurfaceType.PLANE
        for surface_type in surface_types
    )