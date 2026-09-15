from OCCT.BRepGProp import BRepGProp
from OCCT.GProp import GProp_GProps
from OCCT.TopoDS import TopoDS_Face

from .classification import classify_face
from .model import FaceDescriptor


def face_area(face: TopoDS_Face) -> float:
    """
    Compute the surface area of a B-rep face.

    OpenCASCADE integrates the actual geometric face,
    rather than approximating it with a mesh.
    """
    properties = GProp_GProps()

    BRepGProp.SurfaceProperties_(face, properties)

    return properties.Mass()


def describe_face(
    face: TopoDS_Face,
    index: int,
) -> FaceDescriptor:
    """
    Extract the initial geometric descriptor for a face.
    """
    return FaceDescriptor(
        index=index,
        surface_type=classify_face(face),
        area=face_area(face),
    )