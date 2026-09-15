from math import isclose

from auto_mating.geometry import Vector3D


def dot(a: Vector3D, b: Vector3D) -> float:
    """
    Compute the dot product of two 3D vectors.
    """
    return (
        a.x * b.x
        + a.y * b.y
        + a.z * b.z
    )




def are_parallel(
    a: Vector3D,
    b: Vector3D,
    tolerance: float = 1.0e-6,
) -> bool:
    """
    Return True when two vectors are parallel or anti-parallel.
    """
    value = abs(dot(a, b))

    return isclose(
        value,
        1.0,
        abs_tol=tolerance,
    )




def are_same_direction(
    a: Vector3D,
    b: Vector3D,
    tolerance: float = 1.0e-6,
) -> bool:
    """
    Return True when two unit vectors point in the same direction.
    """
    return isclose(
        dot(a, b),
        1.0,
        abs_tol=tolerance,
    )


def are_opposite_direction(
    a: Vector3D,
    b: Vector3D,
    tolerance: float = 1.0e-6,
) -> bool:
    """
    Return True when two unit vectors point in opposite directions.
    """
    return isclose(
        dot(a, b),
        -1.0,
        abs_tol=tolerance,
    )




