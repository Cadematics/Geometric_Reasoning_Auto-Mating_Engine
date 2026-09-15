from dataclasses import dataclass


@dataclass(frozen=True)
class TopologySummary:
    solids: int
    shells: int
    faces: int
    edges: int
    vertices: int