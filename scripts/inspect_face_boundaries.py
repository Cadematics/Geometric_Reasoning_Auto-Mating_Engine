from pathlib import Path

from OCCT.BRep import BRep_Tool
from OCCT.TopLoc import TopLoc_Location
from OCCT.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
from OCCT.TopExp import TopExp_Explorer
from OCCT.TopoDS import TopoDS

from auto_mating.io import read_step
from auto_mating.topology import iter_faces


EXAMPLES_DIR = Path("examples/mating_test")


def inspect_face(path, face_index):
    shape = read_step(path)
    faces = list(iter_faces(shape))
    face = faces[face_index - 1]

    print("=" * 70)
    print(f"{path.name} — Face {face_index}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------

    print("\nEdges:")

    edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)

    edge_number = 1

    while edge_explorer.More():
        edge = TopoDS.Edge_(edge_explorer.Current())

        print(f"\n  Edge {edge_number}")

        # Get the 3D curve associated with the edge
        location = TopLoc_Location()

        curve, first, last = BRep_Tool.Curve_(
            edge,
            0.0,
            0.0,
        )

        print(f"    parameter range: {first:.6f} -> {last:.6f}")

        # --------------------------------------------------------
        # Vertices of this edge
        # --------------------------------------------------------

        vertex_explorer = TopExp_Explorer(
            edge,
            TopAbs_VERTEX,
        )

        vertex_number = 1

        while vertex_explorer.More():
            vertex = TopoDS.Vertex_(
                vertex_explorer.Current()
            )

            point = BRep_Tool.Pnt_(vertex)

            print(
                f"    Vertex {vertex_number}: "
                f"({point.X():.6f}, "
                f"{point.Y():.6f}, "
                f"{point.Z():.6f})"
            )

            vertex_number += 1
            vertex_explorer.Next()

        edge_number += 1
        edge_explorer.Next()

    print()


inspect_face(
    EXAMPLES_DIR / "part_A.step",
    5,
)

inspect_face(
    EXAMPLES_DIR / "part_B.step",
    6,
)