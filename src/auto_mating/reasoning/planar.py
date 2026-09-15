from math import isclose

from auto_mating.geometry import Vector3D
from auto_mating.geometry import PlaneGeometry, Point3D
from .model import PlanarRelationship


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



def plane_distance( a: PlaneGeometry, b: PlaneGeometry,) -> float:
    """
    Compute the perpendicular distance between two parallel planes.
    """
    dx = b.origin.x - a.origin.x
    dy = b.origin.y - a.origin.y
    dz = b.origin.z - a.origin.z

    return abs(
        dx * a.axis_direction.x
        + dy * a.axis_direction.y
        + dz * a.axis_direction.z
    )


def compare_planes(
    a: PlaneGeometry,
    b: PlaneGeometry,
    tolerance: float = 1.0e-6,
) -> PlanarRelationship:
    """
    Determine the geometric relationship between two planes.
    """
    parallel = are_parallel(
        a.axis_direction,
        b.axis_direction,
        tolerance,
    )

    same_direction = are_same_direction(
        a.axis_direction,
        b.axis_direction,
        tolerance,
    )

    opposite_direction = are_opposite_direction(
        a.axis_direction,
        b.axis_direction,
        tolerance,
    )

    if parallel:
        distance = plane_distance(a, b)

        coplanar = distance <= tolerance

    else:
        distance = 0.0
        coplanar = False

    return PlanarRelationship(
        parallel=parallel,
        same_direction=same_direction,
        opposite_direction=opposite_direction,
        distance=distance,
        coplanar=coplanar,
    )














