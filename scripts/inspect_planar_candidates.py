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
        for index, face in enumerate(iter_faces(shape), start=1)
    ]


part_a_faces = describe_step_faces(EXAMPLES_DIR / "part_A.step")
part_b_faces = describe_step_faces(EXAMPLES_DIR / "part_B.step")


candidates = generate_planar_candidates(
    part_a_faces,
    part_b_faces,
    distance_tolerance=10.01,
)


print(f"Found {len(candidates)} planar candidates")
print()

for candidate in candidates:
    face_a = part_a_faces[candidate.part_a_face - 1]
    face_b = part_b_faces[candidate.part_b_face - 1]

    print(
        f"Part A Face {candidate.part_a_face} "
        f"<-> Part B Face {candidate.part_b_face}"
    )

    print(f"  mating type: {candidate.mating_type}")

    print(
        f"  A normal: "
        f"({face_a.normal.x:.3f}, "
        f"{face_a.normal.y:.3f}, "
        f"{face_a.normal.z:.3f})"
    )

    print(
        f"  B normal: "
        f"({face_b.normal.x:.3f}, "
        f"{face_b.normal.y:.3f}, "
        f"{face_b.normal.z:.3f})"
    )

    print(
        f"  A centroid: "
        f"({face_a.centroid.x:.3f}, "
        f"{face_a.centroid.y:.3f}, "
        f"{face_a.centroid.z:.3f})"
    )

    print(
        f"  B centroid: "
        f"({face_b.centroid.x:.3f}, "
        f"{face_b.centroid.y:.3f}, "
        f"{face_b.centroid.z:.3f})"
    )

    print()