from OCCT.TopAbs import TopAbs_VERTEX
from OCCT.TopExp import TopExp_Explorer
from OCCT.TopoDS import TopoDS, TopoDS_Vertex


def iter_vertices(shape):
    """
    Iterate over the vertices of an OpenCASCADE shape.
    """
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)

    while explorer.More():
        yield TopoDS.Vertex_(explorer.Current())
        explorer.Next()