from dataclasses import dataclass


@dataclass(frozen=True)
class PlanarRelationship:
    parallel: bool
    same_direction: bool
    opposite_direction: bool
    distance: float
    coplanar: bool