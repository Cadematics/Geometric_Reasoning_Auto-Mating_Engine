from math import isclose

from auto_mating.geometry import (
    Point3D,
    Vector3D,
    make_planar_frame,
    project_point,
)


def assert_vector_close(
    actual,
    expected,
    tolerance=1.0e-9,
):
    assert isclose(
        actual.x,
        expected.x,
        abs_tol=tolerance,
    )
    assert isclose(
        actual.y,
        expected.y,
        abs_tol=tolerance,
    )
    assert isclose(
        actual.z,
        expected.z,
        abs_tol=tolerance,
    )


def test_xy_plane_frame():
    frame = make_planar_frame(
        origin=Point3D(0.0, 0.0, 10.0),
        normal=Vector3D(0.0, 0.0, 1.0),
    )

    assert isclose(
        frame.u.x,
        0.0,
        abs_tol=1.0e-9,
    )
    assert isclose(
        frame.u.y,
        1.0,
        abs_tol=1.0e-9,
    )

    assert isclose(
        frame.v.x,
        -1.0,
        abs_tol=1.0e-9,
    )
    assert isclose(
        frame.v.y,
        0.0,
        abs_tol=1.0e-9,
    )

    assert_vector_close(
        frame.normal,
        Vector3D(0.0, 0.0, 1.0),
    )


def test_frame_vectors_are_orthonormal():
    frame = make_planar_frame(
        origin=Point3D(5.0, 6.0, 7.0),
        normal=Vector3D(1.0, 2.0, 3.0),
    )

    from auto_mating.geometry.planar import dot, norm

    assert isclose(
        norm(frame.u),
        1.0,
        abs_tol=1.0e-9,
    )

    assert isclose(
        norm(frame.v),
        1.0,
        abs_tol=1.0e-9,
    )

    assert isclose(
        norm(frame.normal),
        1.0,
        abs_tol=1.0e-9,
    )

    assert isclose(
        dot(frame.u, frame.v),
        0.0,
        abs_tol=1.0e-9,
    )

    assert isclose(
        dot(frame.u, frame.normal),
        0.0,
        abs_tol=1.0e-9,
    )

    assert isclose(
        dot(frame.v, frame.normal),
        0.0,
        abs_tol=1.0e-9,
    )


def test_project_point():
    frame = make_planar_frame(
        origin=Point3D(0.0, 0.0, 10.0),
        normal=Vector3D(0.0, 0.0, 1.0),
    )

    point = Point3D(
        10.0,
        20.0,
        10.0,
    )

    projected = project_point(
        point,
        frame,
    )

    assert isclose(
        projected.x,
        20.0,
        abs_tol=1.0e-9,
    )

    assert isclose(
        projected.y,
        -10.0,
        abs_tol=1.0e-9,
    )