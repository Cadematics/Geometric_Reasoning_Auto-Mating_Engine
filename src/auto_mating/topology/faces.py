from OCCT.TopAbs import TopAbs_FACE
from OCCT.TopExp import TopExp_Explorer
from OCCT.TopoDS import TopoDS, TopoDS_Face


def iter_faces(shape):
    """
    Iterate over the faces of an OpenCASCADE shape.

    TopExp_Explorer returns TopoDS_Shape objects, so the
    binding-specific TopoDS.Face_ conversion is performed here.
    """

    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    while explorer.More():
        yield TopoDS.Face_(explorer.Current())
        explorer.Next()





        