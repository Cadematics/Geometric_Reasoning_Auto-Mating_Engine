from .extractor import count_unique_subshapes, extract_topology
from .faces import iter_faces
from .edges import iter_edges
from .vertices import iter_vertices
from .index import TopologyIndex


from .model import (
    TopologySummary,
    EdgeDescriptor,
    VertexDescriptor,
    FaceTopology,
)


__all__ = [
    "TopologySummary",
    "EdgeDescriptor",
    "VertexDescriptor",
    "FaceTopology",
    "count_unique_subshapes",
    "extract_topology",
    "iter_faces",
    "iter_edges",
    "iter_vertices",
    "TopologyIndex",

]