from auto_mating.geometry import Vector3D
from auto_mating.reasoning import (
    are_opposite_direction,
    are_parallel,
    are_same_direction,
    dot,
)


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


    