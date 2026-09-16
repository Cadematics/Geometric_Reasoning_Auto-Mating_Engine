from pathlib import Path

from auto_mating.geometry import describe_face
from auto_mating.io import read_step
from auto_mating.mating import generate_planar_candidates
from auto_mating.topology import iter_faces


EXAMPLES_DIR = Path("examples/mating_test")


def describe_step_faces(path):
    shape = read_step(path)

    return [
        describe_face(face, index)
        for index, face in enumerate(
            iter_faces(shape),
            start=1,
        )
    ]


def test_real_step_files_can_generate_planar_candidates():
    part_a_faces = describe_step_faces(
        EXAMPLES_DIR / "part_A.step"
    )

    part_b_faces = describe_step_faces(
        EXAMPLES_DIR / "part_B.step"
    )

    candidates = generate_planar_candidates(
        part_a_faces,
        part_b_faces,
        distance_tolerance=0.01,
    )

    assert len(candidates) > 0

    assert all(
        candidate.mating_type == "planar"
        for candidate in candidates
    )