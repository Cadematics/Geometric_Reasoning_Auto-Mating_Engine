from typing import Iterator

from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Solid, TopoDS_Face
from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods


def iter_solids(shape: TopoDS_Shape) -> Iterator[TopoDS_Solid]:
    """Yield all solids contained in a shape."""
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        yield topods.Solid(exp.Current())
        exp.Next()


def iter_faces(shape: TopoDS_Shape) -> Iterator[TopoDS_Shape]:
    """Yield all faces contained in a shape."""
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        yield topods.Face(exp.Current())
        exp.Next()
