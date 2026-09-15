from OCCT.TopAbs import TopAbs_EDGE
from OCCT.TopExp import TopExp_Explorer
from OCCT.TopoDS import TopoDS, TopoDS_Edge


def iter_edges(shape):
    """
    Iterate over the edges of an OpenCASCADE shape.
    """
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)

    while explorer.More():
        yield TopoDS.Edge_(explorer.Current())
        explorer.Next()