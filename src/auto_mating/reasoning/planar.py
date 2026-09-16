from math import isclose

from auto_mating.geometry import Vector3D
from auto_mating.geometry import PlaneGeometry, Point3D, FaceDescriptor
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



def are_mating_planes(
    a: PlaneGeometry,
    b: PlaneGeometry,
    distance_tolerance: float = 1.0e-6,
    angular_tolerance: float = 1.0e-6,
) -> bool:
    """
    Determine whether two planar surfaces are candidate mating planes.

    Candidate mating planes must:
    - be parallel,
    - have opposite directions,
    - be separated by no more than distance_tolerance.

    This compares the underlying infinite planes. It does not yet
    verify whether the bounded face regions actually overlap.
    """
    relationship = compare_planes(
        a,
        b,
        tolerance=angular_tolerance,
    )

    if not relationship.parallel:
        return False

    if not relationship.opposite_direction:
        return False

    return relationship.distance <= distance_tolerance



def planar_mating_candidate(
    a: FaceDescriptor,
    b: FaceDescriptor,
    distance_tolerance: float = 1.0e-6,
    angular_tolerance: float = 1.0e-6,
) -> bool:
    """
    Determine whether two planar faces are candidate mating faces.

    Both faces must:
    - be planar,
    - have extracted plane geometry,
    - lie on parallel planes,
    - have opposite B-rep face normals,
    - be within the specified distance tolerance.

    The plane relationship uses the underlying infinite planes.
    Face orientation is determined from the B-rep face normals.

    This is still a candidate test. It does not yet verify
    overlap between the bounded face regions.
    """
    if a.plane is None or b.plane is None:
        return False

    if a.surface_type.value != "plane":
        return False

    if b.surface_type.value != "plane":
        return False

    relationship = compare_planes(
        a.plane,
        b.plane,
        tolerance=angular_tolerance,
    )

    if not relationship.parallel:
        return False

    if relationship.distance > distance_tolerance:
        return False

    return are_opposite_direction(
        a.normal,
        b.normal,
        tolerance=angular_tolerance,
    )



















