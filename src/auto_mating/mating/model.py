from dataclasses import dataclass


@dataclass(frozen=True)
class MatingCandidate:
    part_a_face: int
    part_b_face: int
    mating_type: str





    