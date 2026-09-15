

from OCCT.BRepAdaptor import BRepAdaptor_Surface
from OCCT.BRepGProp import BRepGProp
from OCCT.BRepLProp import BRepLProp_SLProps
from OCCT.GProp import GProp_GProps
from OCCT.TopoDS import TopoDS_Face

from .classification import classify_face

from .surfaces import SurfaceType

from .model import (
    CylinderGeometry,
    FaceDescriptor,
    PlaneGeometry,
    Point3D,
    Vector3D,
)

def plane_geometry(face: TopoDS_Face) -> PlaneGeometry:
    """
    Extract the underlying infinite plane geometry of a planar face.
    """
    surface = BRepAdaptor_Surface(face)
    plane = surface.Plane()

    location = plane.Location()

    direction = plane.Axis().Direction()

    # return PlaneGeometry(
    #     origin=Point3D(
    #         x=location.X(),
    #         y=location.Y(),
    #         z=location.Z(),
    #     ),
    #     normal=Vector3D(
    #         x=direction.X(),
    #         y=direction.Y(),
    #         z=direction.Z(),
    #     ),
    # )
    return PlaneGeometry(
    origin=Point3D(
        x=location.X(),
        y=location.Y(),
        z=location.Z(),
    ),
    axis_direction=Vector3D(
        x=direction.X(),
        y=direction.Y(),
        z=direction.Z(),
    ),
)


def face_area(face: TopoDS_Face) -> float:
    """
    Compute the surface area of a B-rep face.

    OpenCASCADE integrates the actual geometric face,
    rather than approximating it with a mesh.
    """
    properties = GProp_GProps()

    BRepGProp.SurfaceProperties_(face, properties)

    return properties.Mass()


def face_centroid(face: TopoDS_Face) -> Point3D:
    """
    Compute the geometric centroid of a B-rep face.
    """
    properties = GProp_GProps()

    BRepGProp.SurfaceProperties_(face, properties)

    point = properties.CentreOfMass()

    return Point3D(
        x=point.X(),
        y=point.Y(),
        z=point.Z(),
    )


def face_normal(face: TopoDS_Face) -> Vector3D:
    """
    Compute the oriented surface normal at the midpoint of the
    face's UV parameter domain.

    The underlying geometric surface has its own intrinsic normal,
    but a B-rep face may be oriented FORWARD or REVERSED relative
    to that surface. We account for the face orientation here so
    the returned normal represents the actual face orientation.
    """
    surface = BRepAdaptor_Surface(face)

    u1 = surface.FirstUParameter()
    u2 = surface.LastUParameter()
    v1 = surface.FirstVParameter()
    v2 = surface.LastVParameter()

    u = (u1 + u2) / 2.0
    v = (v1 + v2) / 2.0

    properties = BRepLProp_SLProps(
        surface,
        u,
        v,
        1,
        1.0e-6,
    )

    if not properties.IsNormalDefined():
        raise ValueError(
            "Face normal is not defined at the evaluation point."
        )

    normal = properties.Normal()

    if face.Orientation().name == "TopAbs_REVERSED":
        normal = normal.Reversed()

    return Vector3D(
        x=normal.X(),
        y=normal.Y(),
        z=normal.Z(),
    )



def describe_face(
    face: TopoDS_Face,
    index: int,
) -> FaceDescriptor:
    """
    Extract the geometric descriptor for a face.
    """
    surface_type = classify_face(face)

    plane = None
    cylinder = None

    if surface_type == SurfaceType.PLANE:
        plane = plane_geometry(face)

    elif surface_type == SurfaceType.CYLINDER:
        cylinder = cylinder_geometry(face)

    return FaceDescriptor(
        index=index,
        surface_type=surface_type,
        area=face_area(face),
        centroid=face_centroid(face),
        normal=face_normal(face),
        plane=plane,
        cylinder=cylinder,
    )



def cylinder_geometry(face: TopoDS_Face) -> CylinderGeometry:
    """
    Extract the underlying infinite cylinder geometry of a
    cylindrical face.
    """
    surface = BRepAdaptor_Surface(face)
    cylinder = surface.Cylinder()

    axis = cylinder.Axis()
    location = axis.Location()
    direction = axis.Direction()

    return CylinderGeometry(
        axis_origin=Point3D(
            x=location.X(),
            y=location.Y(),
            z=location.Z(),
        ),
        axis_direction=Vector3D(
            x=direction.X(),
            y=direction.Y(),
            z=direction.Z(),
        ),
        radius=cylinder.Radius(),
    )