from OCCT.BRep import BRep_Tool
from OCCT.Geom import (
    Geom_BSplineSurface,
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Plane,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
)
from OCCT.TopoDS import TopoDS_Face

from .surfaces import SurfaceType


def classify_face(face: TopoDS_Face) -> SurfaceType:
    """
    Classify a face according to its underlying OpenCASCADE
    geometric surface.
    """

    surface = BRep_Tool.Surface_(face)

    if isinstance(surface, Geom_Plane):
        return SurfaceType.PLANE

    if isinstance(surface, Geom_CylindricalSurface):
        return SurfaceType.CYLINDER

    if isinstance(surface, Geom_ConicalSurface):
        return SurfaceType.CONE

    if isinstance(surface, Geom_SphericalSurface):
        return SurfaceType.SPHERE

    if isinstance(surface, Geom_ToroidalSurface):
        return SurfaceType.TORUS

    if isinstance(surface, Geom_BSplineSurface):
        return SurfaceType.BSPLINE

    return SurfaceType.OTHER