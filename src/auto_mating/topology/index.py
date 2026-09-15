from dataclasses import dataclass
from typing import Tuple

from OCCT.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_VERTEX,
)
from OCCT.TopExp import TopExp_Explorer
from OCCT.TopoDS import TopoDS, TopoDS_Shape
from OCCT.TopTools import TopTools_IndexedMapOfShape


@dataclass
class TopologyIndex:
    faces: TopTools_IndexedMapOfShape
    edges: TopTools_IndexedMapOfShape
    vertices: TopTools_IndexedMapOfShape

    @classmethod
    def build(cls, shape: TopoDS_Shape):
        faces = TopTools_IndexedMapOfShape()
        edges = TopTools_IndexedMapOfShape()
        vertices = TopTools_IndexedMapOfShape()

        explorer = TopExp_Explorer(shape, TopAbs_FACE)

        while explorer.More():
            faces.Add(explorer.Current())
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_EDGE)

        while explorer.More():
            edges.Add(explorer.Current())
            explorer.Next()

        explorer = TopExp_Explorer(shape, TopAbs_VERTEX)

        while explorer.More():
            vertices.Add(explorer.Current())
            explorer.Next()

        return cls(
            faces=faces,
            edges=edges,
            vertices=vertices,
        )

    @property
    def face_count(self) -> int:
        return self.faces.Extent()

    @property
    def edge_count(self) -> int:
        return self.edges.Extent()

    @property
    def vertex_count(self) -> int:
        return self.vertices.Extent()

    def face(self, index: int):
        return TopoDS.Face_(self.faces.FindKey(index))

    def edge(self, index: int):
        return TopoDS.Edge_(self.edges.FindKey(index))

    def vertex(self, index: int):
        return TopoDS.Vertex_(self.vertices.FindKey(index))

    def face_edges(self, face_index: int) -> Tuple[int, ...]:
        """
        Return the canonical edge indices belonging to a face.

        Indices use the same 1-based indexing as
        TopTools_IndexedMapOfShape.
        """
        face = self.face(face_index)

        explorer = TopExp_Explorer(face, TopAbs_EDGE)

        edge_indices = []

        while explorer.More():
            edge = explorer.Current()

            edge_index = self.edges.FindIndex(edge)

            if edge_index == 0:
                raise RuntimeError(
                    "Face contains an edge that is not present "
                    "in the topology index."
                )

            edge_indices.append(edge_index)

            explorer.Next()

        return tuple(edge_indices)

    def edge_faces(self, edge_index: int) -> Tuple[int, ...]:
        """
        Return the canonical face indices incident to an edge.
        """
        faces = []

        for face_index in range(1, self.face_count + 1):
            edge_indices = self.face_edges(face_index)

            if edge_index in edge_indices:
                faces.append(face_index)

        return tuple(faces)



    