from OCCT.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
)
from OCCT.TopExp import TopExp_Explorer
from OCCT.TopoDS import TopoDS_Shape
from OCCT.TopTools import TopTools_IndexedMapOfShape

from .model import TopologySummary


def count_unique_subshapes(
    shape: TopoDS_Shape,
    shape_type: int,
) -> int:
    """
    Count unique topological sub-shapes.

    Unlike a simple TopExp_Explorer traversal, this avoids
    counting the same edge or vertex multiple times when it
    is shared by several faces.
    """

    shape_map = TopTools_IndexedMapOfShape()

    explorer = TopExp_Explorer(shape, shape_type)

    while explorer.More():
        subshape = explorer.Current()
        shape_map.Add(subshape)
        explorer.Next()

    return shape_map.Extent()


def extract_topology(shape: TopoDS_Shape) -> TopologySummary:
    """
    Extract unique B-rep topology counts from an OpenCASCADE shape.
    """

    return TopologySummary(
        solids=count_unique_subshapes(shape, TopAbs_SOLID),
        shells=count_unique_subshapes(shape, TopAbs_SHELL),
        faces=count_unique_subshapes(shape, TopAbs_FACE),
        edges=count_unique_subshapes(shape, TopAbs_EDGE),
        vertices=count_unique_subshapes(shape, TopAbs_VERTEX),
    )