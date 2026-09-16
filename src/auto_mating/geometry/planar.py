from dataclasses import dataclass

from .model import Point3D, Vector3D


@dataclass(frozen=True)
class PlanarFrame:
    """
    2D coordinate frame embedded in 3D space.

    origin:
        A point on the plane.

    u:
        First in-plane unit direction.

    v:
        Second in-plane unit direction.

    normal:
        Unit plane normal.
    """

    origin: Point3D
    u: Vector3D
    v: Vector3D
    normal: Vector3D


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


def dot(a: Vector3D, b: Vector3D) -> float:
    return (
        a.x * b.x
        + a.y * b.y
        + a.z * b.z
    )


def cross(a: Vector3D, b: Vector3D) -> Vector3D:
    return Vector3D(
        x=a.y * b.z - a.z * b.y,
        y=a.z * b.x - a.x * b.z,
        z=a.x * b.y - a.y * b.x,
    )


def norm(vector: Vector3D) -> float:
    return (
        vector.x ** 2
        + vector.y ** 2
        + vector.z ** 2
    ) ** 0.5


def normalize(vector: Vector3D) -> Vector3D:
    length = norm(vector)

    if length == 0.0:
        raise ValueError(
            "Cannot normalize a zero-length vector."
        )

    return Vector3D(
        x=vector.x / length,
        y=vector.y / length,
        z=vector.z / length,
    )


def project_point(
    point: Point3D,
    frame: PlanarFrame,
) -> Point2D:
    """
    Project a 3D point into a planar coordinate frame.
    """

    relative = Vector3D(
        x=point.x - frame.origin.x,
        y=point.y - frame.origin.y,
        z=point.z - frame.origin.z,
    )

    return Point2D(
        x=dot(relative, frame.u),
        y=dot(relative, frame.v),
    )

def make_planar_frame(
    origin: Point3D,
    normal: Vector3D,
) -> PlanarFrame:
    """
    Construct a stable orthonormal 2D coordinate frame
    embedded in a 3D plane.
    """

    normal = normalize(normal)

    # Choose a reference direction that is not parallel
    # to the plane normal.
    if abs(normal.x) < 0.9:
        reference = Vector3D(
            x=1.0,
            y=0.0,
            z=0.0,
        )
    else:
        reference = Vector3D(
            x=0.0,
            y=1.0,
            z=0.0,
        )

    # First in-plane direction.
    u = normalize(
        cross(normal, reference)
    )

    # Second in-plane direction.
    v = normalize(
        cross(normal, u)
    )

    return PlanarFrame(
        origin=origin,
        u=u,
        v=v,
        normal=normal,
    )