from typing import List

from OCCT.BRep import BRep_Tool
from OCCT.TopAbs import (
    TopAbs_EDGE,
    TopAbs_REVERSED,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCCT.TopExp import TopExp_Explorer
from OCCT.TopoDS import TopoDS, TopoDS_Face

from .model import Point3D
from .planar import PlanarFrame, project_point
from .polygon import Polygon2D


def face_boundary_points(face: TopoDS_Face) -> List[Point3D]:
    wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)

    if not wire_explorer.More():
        raise ValueError("Face does not contain a wire.")

    wire = TopoDS.Wire_(wire_explorer.Current())

    points = []

    edge_explorer = TopExp_Explorer(wire, TopAbs_EDGE)

    while edge_explorer.More():
        edge = TopoDS.Edge_(edge_explorer.Current())

        vertex_explorer = TopExp_Explorer(edge, TopAbs_VERTEX)

        if not vertex_explorer.More():
            raise ValueError("Edge does not contain a vertex.")

        vertex = TopoDS.Vertex_(vertex_explorer.Current())

        point = BRep_Tool.Pnt_(vertex)

        points.append(
            Point3D(
                point.X(),
                point.Y(),
                point.Z(),
            )
        )

        edge_explorer.Next()

    if len(points) < 3:
        raise ValueError("Face boundary must contain at least 3 points.")

    return points



def project_face_boundary(
    face: TopoDS_Face,
    frame: PlanarFrame,
) -> Polygon2D:

    points_3d = face_boundary_points(face)

    points_2d = tuple(
        project_point(point, frame)
        for point in points_3d
    )

    return Polygon2D(
        points=points_2d,
    )