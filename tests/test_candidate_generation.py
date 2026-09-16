


from auto_mating.geometry import (
    FaceDescriptor,
    PlaneGeometry,
    Point3D,
    SurfaceType,
    Vector3D,
)
from auto_mating.mating import (
    MatingCandidate,
    generate_planar_candidates,
)





def make_planar_face(
    index,
    origin,
    direction,
):
    return FaceDescriptor(
        index=index,
        surface_type=SurfaceType.PLANE,
        area=100.0,
        centroid=Point3D(
            origin[0],
            origin[1],
            origin[2],
        ),
        normal=Vector3D(
            direction[0],
            direction[1],
            direction[2],
        ),
        plane=PlaneGeometry(
            origin=Point3D(
                origin[0],
                origin[1],
                origin[2],
            ),
            axis_direction=Vector3D(
                direction[0],
                direction[1],
                direction[2],
            ),
        ),
    )





def test_generate_one_planar_candidate():
    part_a = [
        make_planar_face(
            1,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    ]

    part_b = [
        make_planar_face(
            2,
            (0.0, 0.0, 0.001),
            (0.0, 0.0, -1.0),
        )
    ]

    candidates = generate_planar_candidates(
        part_a,
        part_b,
        distance_tolerance=0.01,
    )

    assert candidates == [
        MatingCandidate(
            part_a_face=1,
            part_b_face=2,
            mating_type="planar",
        )
    ]




def test_generate_no_candidate_for_non_mating_faces():
    part_a = [
        make_planar_face(
            1,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    ]

    part_b = [
        make_planar_face(
            2,
            (0.0, 0.0, 10.0),
            (0.0, 0.0, -1.0),
        )
    ]

    candidates = generate_planar_candidates(
        part_a,
        part_b,
        distance_tolerance=0.01,
    )

    assert candidates == []





def test_generate_candidates_across_multiple_faces():
    part_a = [
        make_planar_face(
            1,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        make_planar_face(
            2,
            (0.0, 0.0, 5.0),
            (0.0, 1.0, 0.0),
        ),
    ]

    part_b = [
        make_planar_face(
            10,
            (0.0, 0.0, 0.001),
            (0.0, 0.0, -1.0),
        ),
        make_planar_face(
            20,
            (0.0, 0.0, 5.001),
            (0.0, -1.0, 0.0),
        ),
    ]

    candidates = generate_planar_candidates(
        part_a,
        part_b,
        distance_tolerance=0.01,
    )

    assert candidates == [
        MatingCandidate(
            part_a_face=1,
            part_b_face=10,
            mating_type="planar",
        ),
        MatingCandidate(
            part_a_face=2,
            part_b_face=20,
            mating_type="planar",
        ),
    ]






