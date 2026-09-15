from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TopologySummary:
    solids: int
    shells: int
    faces: int
    edges: int
    vertices: int


@dataclass(frozen=True)
class EdgeDescriptor:
    index: int


@dataclass(frozen=True)
class VertexDescriptor:
    index: int


@dataclass(frozen=True)
class FaceTopology:
    face_index: int
    edge_indices: Tuple[int, ...]
    vertex_indices: Tuple[int, ...]