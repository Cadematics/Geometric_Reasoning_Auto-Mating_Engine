from typing import Iterable, List

from auto_mating.geometry import FaceDescriptor
from auto_mating.reasoning import planar_mating_candidate

from .model import MatingCandidate


def generate_planar_candidates(
    part_a_faces: Iterable[FaceDescriptor],
    part_b_faces: Iterable[FaceDescriptor],
    distance_tolerance: float = 1.0e-6,
    angular_tolerance: float = 1.0e-6,
) -> List[MatingCandidate]:
    """
    Generate planar mating candidates between two parts.

    Every face in part A is compared against every face in part B.
    Geometric reasoning determines whether a pair is a candidate.

    The function does not modify either part.
    """
    candidates = []

    for face_a in part_a_faces:
        for face_b in part_b_faces:
            if planar_mating_candidate(
                face_a,
                face_b,
                distance_tolerance=distance_tolerance,
                angular_tolerance=angular_tolerance,
            ):
                candidates.append(
                    MatingCandidate(
                        part_a_face=face_a.index,
                        part_b_face=face_b.index,
                        mating_type="planar",
                    )
                )

    return candidates