from auto_mating.geometry import Vector3D, CylinderGeometry, PlaneGeometry, Point3D, FaceDescriptor, SurfaceType
from auto_mating.reasoning import (
    are_opposite_direction,
    are_parallel,
    are_same_direction,
    dot,
    are_mating_planes,
    compare_planes,
    planar_mating_candidate,
)
from auto_mating.geometry import PlaneGeometry, Point3D, Vector3D, FaceDescriptor
from auto_mating.reasoning import compare_planes, planar_mating_candidate   








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




def test_mating_planes_within_tolerance():
    plane_a = PlaneGeometry(
        origin=Point3D(0.0, 0.0, 0.0),
        axis_direction=Vector3D(0.0, 0.0, 1.0),
    )

    plane_b = PlaneGeometry(
        origin=Point3D(0.0, 0.0, 0.001),
        axis_direction=Vector3D(0.0, 0.0, -1.0),
    )

    assert are_mating_planes(
        plane_a,
        plane_b,
        distance_tolerance=0.01,
    )


def test_mating_planes_too_far_apart():
    plane_a = PlaneGeometry(
        origin=Point3D(0.0, 0.0, 0.0),
        axis_direction=Vector3D(0.0, 0.0, 1.0),
    )

    plane_b = PlaneGeometry(
        origin=Point3D(0.0, 0.0, 10.0),
        axis_direction=Vector3D(0.0, 0.0, -1.0),
    )

    assert not are_mating_planes(
        plane_a,
        plane_b,
        distance_tolerance=0.01,
    )


def test_same_direction_planes_are_not_mating():
    plane_a = PlaneGeometry(
        origin=Point3D(0.0, 0.0, 0.0),
        axis_direction=Vector3D(0.0, 0.0, 1.0),
    )

    plane_b = PlaneGeometry(
        origin=Point3D(0.0, 0.0, 0.001),
        axis_direction=Vector3D(0.0, 0.0, 1.0),
    )

    assert not are_mating_planes(
        plane_a,
        plane_b,
        distance_tolerance=0.01,
    )


def test_non_parallel_planes_are_not_mating():
    plane_a = PlaneGeometry(
        origin=Point3D(0.0, 0.0, 0.0),
        axis_direction=Vector3D(0.0, 0.0, 1.0),
    )

    plane_b = PlaneGeometry(
        origin=Point3D(0.0, 0.0, 0.0),
        axis_direction=Vector3D(1.0, 0.0, 0.0),
    )

    assert not are_mating_planes(
        plane_a,
        plane_b,
        distance_tolerance=0.01,
    )





def test_planar_faces_can_be_mating_candidates():
    face_a = FaceDescriptor(
        index=1,
        surface_type=SurfaceType.PLANE,
        area=100.0,
        centroid=Point3D(5.0, 5.0, 0.0),
        normal=Vector3D(0.0, 0.0, 1.0),
        plane=PlaneGeometry(
            origin=Point3D(0.0, 0.0, 0.0),
            axis_direction=Vector3D(0.0, 0.0, 1.0),
        ),
    )

    face_b = FaceDescriptor(
        index=2,
        surface_type=SurfaceType.PLANE,
        area=100.0,
        centroid=Point3D(5.0, 5.0, 0.001),
        normal=Vector3D(0.0, 0.0, -1.0),
        plane=PlaneGeometry(
            origin=Point3D(0.0, 0.0, 0.001),
            axis_direction=Vector3D(0.0, 0.0, -1.0),
        ),
    )

    assert planar_mating_candidate(
        face_a,
        face_b,
        distance_tolerance=0.01,
    )

def test_non_planar_face_is_not_a_planar_mating_candidate():
    face_a = FaceDescriptor(
        index=1,
        surface_type=SurfaceType.CYLINDER,
        area=100.0,
        centroid=Point3D(0.0, 0.0, 0.0),
        normal=Vector3D(1.0, 0.0, 0.0),
        cylinder=CylinderGeometry(
            axis_origin=Point3D(0.0, 0.0, 0.0),
            axis_direction=Vector3D(0.0, 0.0, 1.0),
            radius=5.0,
        ),
    )

    face_b = FaceDescriptor(
        index=2,
        surface_type=SurfaceType.PLANE,
        area=100.0,
        centroid=Point3D(0.0, 0.0, 0.0),
        normal=Vector3D(0.0, 0.0, -1.0),
        plane=PlaneGeometry(
            origin=Point3D(0.0, 0.0, 0.0),
            axis_direction=Vector3D(0.0, 0.0, -1.0),
        ),
    )

    assert not planar_mating_candidate(face_a, face_b)

def test_planar_faces_with_same_direction_are_not_mating():
    face_a = FaceDescriptor(
        index=1,
        surface_type=SurfaceType.PLANE,
        area=100.0,
        centroid=Point3D(5.0, 5.0, 0.0),
        normal=Vector3D(0.0, 0.0, 1.0),
        plane=PlaneGeometry(
            origin=Point3D(0.0, 0.0, 0.0),
            axis_direction=Vector3D(0.0, 0.0, 1.0),
        ),
    )

    face_b = FaceDescriptor(
        index=2,
        surface_type=SurfaceType.PLANE,
        area=100.0,
        centroid=Point3D(5.0, 5.0, 0.001),
        normal=Vector3D(0.0, 0.0, 1.0),
        plane=PlaneGeometry(
            origin=Point3D(0.0, 0.0, 0.001),
            axis_direction=Vector3D(0.0, 0.0, 1.0),
        ),
    )

    assert not planar_mating_candidate(
        face_a,
        face_b,
        distance_tolerance=0.01,
    )
















