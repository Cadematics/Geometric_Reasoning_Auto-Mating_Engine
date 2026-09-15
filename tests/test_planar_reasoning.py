from auto_mating.geometry import Vector3D
from auto_mating.reasoning import (
    are_opposite_direction,
    are_parallel,
    are_same_direction,
    dot,
)
from auto_mating.geometry import PlaneGeometry, Point3D, Vector3D
from auto_mating.reasoning import compare_planes








def test_dot_product():
    a = Vector3D(1.0, 0.0, 0.0)
    b = Vector3D(0.0, 1.0, 0.0)

    assert dot(a, b) == 0.0


def test_parallel_same_direction():
    a = Vector3D(0.0, 0.0, 1.0)
    b = Vector3D(0.0, 0.0, 1.0)

    assert are_parallel(a, b)
    assert are_same_direction(a, b)
    assert not are_opposite_direction(a, b)


def test_parallel_opposite_direction():
    a = Vector3D(0.0, 0.0, 1.0)
    b = Vector3D(0.0, 0.0, -1.0)

    assert are_parallel(a, b)
    assert not are_same_direction(a, b)
    assert are_opposite_direction(a, b)


def test_non_parallel():
    a = Vector3D(1.0, 0.0, 0.0)
    b = Vector3D(0.0, 1.0, 0.0)

    assert not are_parallel(a, b)




from pathlib import Path

from auto_mating.geometry import (
    SurfaceType,
    describe_face,
)
from auto_mating.io import read_step
from auto_mating.topology import iter_faces


EXAMPLE_STEP = (
    Path(__file__).parent.parent
    / "examples"
    / "simple_plate"
    / "plate.step"
)


def test_planar_face_descriptor_contains_plane():
    shape = read_step(EXAMPLE_STEP)

    face = list(iter_faces(shape))[0]

    descriptor = describe_face(face, 1)

    assert descriptor.surface_type == SurfaceType.PLANE
    assert descriptor.plane is not None
    assert descriptor.cylinder is None



def test_parallel_separated_planes():
    plane_a = PlaneGeometry(
        origin=Point3D(0.0, 0.0, 0.0),
        axis_direction=Vector3D(0.0, 0.0, 1.0),
    )

    plane_b = PlaneGeometry(
        origin=Point3D(0.0, 0.0, 10.0),
        axis_direction=Vector3D(0.0, 0.0, 1.0),
    )

    relationship = compare_planes(plane_a, plane_b)

    assert relationship.parallel
    assert relationship.same_direction
    assert not relationship.opposite_direction
    assert relationship.distance == 10.0
    assert not relationship.coplanar

